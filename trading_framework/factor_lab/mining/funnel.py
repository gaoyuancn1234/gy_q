"""因子筛选漏斗 — 多精选池并行评估

全局候选池 → 6 种选择策略 → 快速评分 → Top-N 全量回测 → 最优池晋升

策略:
  top_icir_10/20   — |ICIR| 最高的 N 个因子
  importance_10/20 — LGB importance 最高的 N 个
  diverse_05       — 相关性<0.5 贪心选择 (max 15)
  consensus        — 出现在 >=min_votes 个池中的因子
"""
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

from factor_lab.quanta.config import FUNNEL_STRATEGIES, FUNNEL_N_BACKTEST


@dataclass
class CuratedPool:
    """一个精选池"""
    name: str                             # 策略名 "top_icir_10"
    strategy: str                         # 选择方法
    factors: list[tuple[str, str]]        # [(name, expr), ...]
    score: float = 0.0                    # 快速评分
    backtest: dict | None = None          # rolling 回测结果
    rank: int = 0                         # 排名


class FactorFunnel:
    """因子筛选漏斗"""

    def __init__(self, strategies: dict | None = None,
                 n_backtest: int = FUNNEL_N_BACKTEST):
        self.strategies = strategies or FUNNEL_STRATEGIES
        self.n_backtest = n_backtest

    def generate_pools(
        self,
        pool_factors: list[tuple[str, str]],
        icir_dict: dict[str, float],
        importance_dict: dict[str, float],
    ) -> list[CuratedPool]:
        """从全局池生成多个精选池

        Args:
            pool_factors: [(name, expr), ...] — 全局池所有因子
            icir_dict: {name: icir_value} — 来自 PoolFactor
            importance_dict: {name: importance_value} — 来自 screen_by_importance
        """
        pools: list[CuratedPool] = []
        factor_map = {name: expr for name, expr in pool_factors}

        for pool_name, cfg in self.strategies.items():
            method = cfg["method"]
            if method == "top_icir":
                factors = self._select_top_icir(pool_factors, icir_dict, cfg["n"])
            elif method == "importance":
                factors = self._select_top_importance(pool_factors, importance_dict, cfg["n"])
            elif method == "diverse":
                factors = self._select_diverse(
                    pool_factors, icir_dict, cfg.get("corr", 0.5), cfg.get("n", 15))
            elif method == "consensus":
                # consensus 需要其他池先生成
                continue  # 延后处理
            else:
                print(f"  [funnel] 未知策略: {method}")
                continue

            if factors:
                pools.append(CuratedPool(
                    name=pool_name, strategy=method, factors=factors))

        # consensus 池: 统计每个因子在已生成池中的出现次数
        consensus_cfg = self.strategies.get("consensus")
        if consensus_cfg and pools:
            min_votes = consensus_cfg.get("min_votes", 3)
            consensus_factors = self._select_consensus(pools, factor_map, min_votes)
            if consensus_factors:
                pools.append(CuratedPool(
                    name="consensus", strategy="consensus", factors=consensus_factors))

        return pools

    def score_pools(self, pools: list[CuratedPool],
                    importance_dict: dict[str, float]) -> list[CuratedPool]:
        """快速评分: mean(importance) * log(1 + n_factors)

        利用 screen_by_importance 已计算的 importance_dict, 不需要额外训练。
        """
        for pool in pools:
            factor_names = [name for name, _ in pool.factors]
            importances = [importance_dict.get(n, 0) for n in factor_names]
            if importances:
                mean_imp = sum(importances) / len(importances)
                pool.score = mean_imp * math.log(1 + len(pool.factors))
            else:
                pool.score = 0.0

        # 排序 + 标记 rank
        pools.sort(key=lambda p: p.score, reverse=True)
        for i, pool in enumerate(pools):
            pool.rank = i + 1

        return pools

    def select_for_backtest(self, pools: list[CuratedPool],
                            n: int | None = None) -> list[CuratedPool]:
        """选评分最高的 N 个池做全量回测"""
        n = n or self.n_backtest
        return [p for p in pools if p.rank <= n]

    def run_funnel(
        self,
        pool_factors: list[tuple[str, str]],
        icir_dict: dict[str, float],
        importance_dict: dict[str, float],
        n_backtest: int | None = None,
    ) -> list[CuratedPool]:
        """完整漏斗流程

        Args:
            pool_factors: 全局池因子 [(name, expr), ...]
            icir_dict: {name: icir}
            importance_dict: {name: importance}
            n_backtest: 做全量回测的池数量

        Returns:
            所有池 (含评分 + 回测结果)
        """
        t0 = time.time()
        n_bt = n_backtest or self.n_backtest

        # Step 1: 生成精选池
        pools = self.generate_pools(pool_factors, icir_dict, importance_dict)
        if not pools:
            print("  [funnel] 无法生成任何精选池")
            return []

        print(f"  [funnel] 生成 {len(pools)} 个精选池:")
        for p in pools:
            print(f"    {p.name}: {len(p.factors)} 因子")

        # Step 2: 快速评分
        pools = self.score_pools(pools, importance_dict)

        print(f"  [funnel] 快速评分:")
        for p in pools:
            print(f"    #{p.rank} {p.name}: score={p.score:.3f} ({len(p.factors)} 因子)")

        # Step 3: Top-N 全量回测
        top_pools = self.select_for_backtest(pools, n_bt)
        print(f"  [funnel] 对 Top-{n_bt} 池做全量 Rolling 回测:")

        for pool in top_pools:
            print(f"    回测 {pool.name} ({len(pool.factors)} 因子)...")
            try:
                from factor_lab.mining.backtest_runner import run_comparison
                pool.backtest = run_comparison(pool.factors)
            except Exception as e:
                print(f"    [funnel] {pool.name} 回测失败: {e}")
                pool.backtest = {"error": str(e)}

        elapsed = time.time() - t0
        print(f"  [funnel] 漏斗完成 (耗时 {elapsed:.1f}s)")

        return pools

    def best_pool(self, pools: list[CuratedPool]) -> CuratedPool | None:
        """从回测结果中选最优池"""
        backtested = [
            p for p in pools
            if p.backtest and not p.backtest.get("error")
        ]
        if not backtested:
            return None
        return max(
            backtested,
            key=lambda p: p.backtest["improvement"]["sharpe_delta"],
        )

    def summary_table(self, pools: list[CuratedPool]) -> str:
        """漏斗汇总表 (用于飞书推送)"""
        lines = ["漏斗评估结果:"]
        lines.append(f"{'池名':<18} {'因子数':>5} {'评分':>7} {'回测':>12}")
        lines.append("-" * 48)

        for p in pools:
            bt_str = ""
            if p.backtest:
                if p.backtest.get("error"):
                    bt_str = "ERR"
                else:
                    delta = p.backtest["improvement"]["sharpe_delta"]
                    better = p.backtest["improvement"]["is_better"]
                    bt_str = f"{delta:+.3f}" + (" BEAT" if better else "")
            else:
                bt_str = "-"

            rank_mark = "*" if p.rank <= self.n_backtest else " "
            lines.append(
                f"{rank_mark}{p.name:<17} {len(p.factors):>5} "
                f"{p.score:>7.2f} {bt_str:>12}"
            )

        best = self.best_pool(pools)
        if best:
            lines.append("")
            lines.append(f"最优池: {best.name} "
                         f"(Sharpe delta={best.backtest['improvement']['sharpe_delta']:+.3f})")

        return "\n".join(lines)

    def to_audit_record(self, pools: list[CuratedPool],
                        elapsed: float = 0) -> dict:
        """生成审计记录中的 funnel 字段"""
        best = self.best_pool(pools)
        return {
            "pools": [
                {
                    "name": p.name,
                    "strategy": p.strategy,
                    "n_factors": len(p.factors),
                    "score": round(p.score, 4),
                    "rank": p.rank,
                    "backtest": (
                        {
                            "sharpe_delta": p.backtest["improvement"]["sharpe_delta"],
                            "is_better": p.backtest["improvement"]["is_better"],
                        }
                        if p.backtest and not p.backtest.get("error")
                        else {"error": p.backtest.get("error")} if p.backtest else None
                    ),
                }
                for p in pools
            ],
            "best_pool": best.name if best else None,
            "total_time": round(elapsed, 1),
        }

    # --- 选择策略实现 ---

    def _select_top_icir(
        self,
        pool_factors: list[tuple[str, str]],
        icir_dict: dict[str, float],
        n: int,
    ) -> list[tuple[str, str]]:
        """按 |ICIR| 降序选 top-N"""
        scored = [(name, expr, abs(icir_dict.get(name, 0)))
                  for name, expr in pool_factors]
        scored.sort(key=lambda x: x[2], reverse=True)
        return [(name, expr) for name, expr, _ in scored[:n]]

    def _select_top_importance(
        self,
        pool_factors: list[tuple[str, str]],
        importance_dict: dict[str, float],
        n: int,
    ) -> list[tuple[str, str]]:
        """按 importance 降序选 top-N"""
        scored = [(name, expr, importance_dict.get(name, 0))
                  for name, expr in pool_factors]
        scored.sort(key=lambda x: x[2], reverse=True)
        return [(name, expr) for name, expr, _ in scored[:n]]

    def _select_diverse(
        self,
        pool_factors: list[tuple[str, str]],
        icir_dict: dict[str, float],
        corr_threshold: float,
        max_n: int,
    ) -> list[tuple[str, str]]:
        """相关性<threshold 的贪心选择 (复用 FactorPool 逻辑)"""
        try:
            from factor_lab.quanta.factor_pool import _compute_rank_corr
        except ImportError:
            # fallback: 直接返回 top by ICIR
            return self._select_top_icir(pool_factors, icir_dict, max_n)

        # 按 |ICIR| 排序
        sorted_factors = sorted(
            pool_factors,
            key=lambda x: abs(icir_dict.get(x[0], 0)),
            reverse=True,
        )

        if len(sorted_factors) <= max_n:
            return sorted_factors

        try:
            corr_matrix = _compute_rank_corr(sorted_factors)
        except Exception as e:
            print(f"  [funnel:diverse] 相关性计算失败: {e}, 回退到 top-{max_n}")
            return sorted_factors[:max_n]

        selected: list[tuple[str, str]] = []
        selected_names: list[str] = []
        for name, expr in sorted_factors:
            if len(selected) >= max_n:
                break
            redundant = False
            for s_name in selected_names:
                key = tuple(sorted([name, s_name]))
                corr_val = abs(corr_matrix.get(key, 0))
                if corr_val >= corr_threshold:
                    redundant = True
                    break
            if not redundant:
                selected.append((name, expr))
                selected_names.append(name)

        return selected

    def _select_consensus(
        self,
        existing_pools: list[CuratedPool],
        factor_map: dict[str, str],
        min_votes: int,
    ) -> list[tuple[str, str]]:
        """统计每个因子出现在几个池中, 保留 >= min_votes"""
        vote_count: dict[str, int] = {}
        for pool in existing_pools:
            for name, _ in pool.factors:
                vote_count[name] = vote_count.get(name, 0) + 1

        consensus = [
            (name, factor_map[name])
            for name, count in vote_count.items()
            if count >= min_votes and name in factor_map
        ]
        # 按投票数降序
        consensus.sort(key=lambda x: vote_count.get(x[0], 0), reverse=True)
        return consensus
