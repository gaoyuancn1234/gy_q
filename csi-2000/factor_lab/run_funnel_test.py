"""独立运行 Phase D 漏斗评估 — 使用主路径的全局因子池

用法:
    cd trading_framework
    python -m factor_lab.run_funnel_test

从主路径 mining_results/global_factor_pool.json 加载 70 个历史因子，
执行 importance 筛选 → 漏斗多池生成 → 快速评分 → Top-2 全量回测。
"""
import json
import sys
import time
from pathlib import Path

# 因子池路径。本文件可能从 worktree 里运行(paper_researcher 会建 worktree)，
# 此时因子池仍在主仓库里，故按候选顺序探测。
#
# 2026-09-05: 原先第二候选写死为 "/Users/jeric/Desktop/lab/..."，那是原作者
# macOS 机器上的路径，在 Windows 上永远不存在 —— 回退等于回退到一个必然
# 不存在的文件，后续只会拿到"文件不存在"而非明确的配置错误。
_REL = "factor_lab/mining_results/global_factor_pool.json"
_HERE = Path(__file__).resolve()
_CANDIDATES = [
    _HERE.parent.parent / _REL,                              # 正常: trading_framework/
    _HERE.parent.parent.parent.parent / "trading_framework" / _REL,  # 从 worktree 回主仓库
]
MAIN_POOL_FILE = next((p for p in _CANDIDATES if p.exists()), _CANDIDATES[0])


def main():
    t_start = time.time()

    # --- 0. qlib init ---
    print("=== Phase D 漏斗评估 (独立测试) ===\n")
    import qlib
    from qlib.config import C
    if not C.__dict__.get('_config', {}).get('_registered', False):
        qlib.init(provider_uri="~/.qlib/qlib_data/cn_data_bs")

    # --- 1. 加载主路径因子池 ---
    from factor_lab.quanta.factor_pool import FactorPool, PoolFactor

    if not MAIN_POOL_FILE.exists():
        print(f"  错误: 因子池文件不存在: {MAIN_POOL_FILE}")
        sys.exit(1)

    with open(MAIN_POOL_FILE, encoding='utf-8') as f:
        pool_data = json.load(f)

    pool_factors_raw = pool_data.get("factors", [])
    print(f"  因子池文件: {MAIN_POOL_FILE}")
    print(f"  因子数量: {len(pool_factors_raw)}")
    print(f"  total_attempted: {pool_data.get('total_attempted', 0)}")

    # 构建 PoolFactor 列表 (过滤掉主路径多余字段如 admitted_at/last_validated)
    import dataclasses
    valid_fields = {f.name for f in dataclasses.fields(PoolFactor)}
    pool_factor_objects = [
        PoolFactor(**{k: v for k, v in item.items() if k in valid_fields})
        for item in pool_factors_raw
    ]
    pool_factors = [(f.name, f.expr) for f in pool_factor_objects]
    icir_dict = {f.name: f.icir for f in pool_factor_objects}

    print(f"\n  Top-5 by |ICIR|:")
    for f in sorted(pool_factor_objects, key=lambda x: abs(x.icir), reverse=True)[:5]:
        print(f"    {f.name:40s} ICIR={f.icir:+.3f}  RankIC={f.rank_ic:+.4f}")

    # --- 2. Importance 筛选 ---
    print(f"\n--- Step 1: Importance 筛选 ---")
    t1 = time.time()
    try:
        from factor_lab.mining.backtest_runner import screen_by_importance
        from factor_lab.quanta.config import IMPORTANCE_MIN_THRESHOLD
        screened_factors, imp_dict = screen_by_importance(
            pool_factors,
            min_importance=IMPORTANCE_MIN_THRESHOLD,
        )
        print(f"  筛选后: {len(screened_factors)}/{len(pool_factors)} 因子通过")
        print(f"  耗时: {time.time() - t1:.1f}s")

        if screened_factors:
            print(f"\n  通过筛选的因子:")
            for name, expr in screened_factors:
                imp = imp_dict.get(name, 0)
                print(f"    {name:40s} imp={imp:.1f}  ICIR={icir_dict.get(name, 0):+.3f}")
    except Exception as e:
        print(f"  Importance 筛选失败: {e}")
        import traceback; traceback.print_exc()
        # fallback: 用 ICIR 作为 importance 代理
        imp_dict = {name: abs(icir) for name, icir in icir_dict.items()}
        screened_factors = pool_factors
        print(f"  回退: 使用 |ICIR| 作为 importance 代理, 全部 {len(pool_factors)} 因子参与漏斗")

    # --- 3. 漏斗评估 ---
    print(f"\n--- Step 2: 漏斗评估 ---")
    t2 = time.time()

    # 如果 imp_dict 为空, 用 ICIR 作为代理
    if not imp_dict:
        imp_dict = {name: abs(icir) for name, icir in icir_dict.items()}
        print(f"  imp_dict 为空, 使用 |ICIR| 作为代理")

    from factor_lab.mining.funnel import FactorFunnel
    funnel = FactorFunnel()
    funnel_pools = funnel.run_funnel(
        pool_factors, icir_dict, imp_dict,
    )

    if funnel_pools:
        print(f"\n{funnel.summary_table(funnel_pools)}")

        best = funnel.best_pool(funnel_pools)
        if best and best.backtest and not best.backtest.get("error"):
            print(f"\n  最优池: {best.name}")
            print(f"    因子数: {len(best.factors)}")
            delta = best.backtest["improvement"]["sharpe_delta"]
            is_better = best.backtest["improvement"]["is_better"]
            print(f"    Sharpe delta: {delta:+.3f}")
            print(f"    Beat baseline: {is_better}")

            if is_better:
                print(f"\n  最优池因子列表:")
                for name, expr in best.factors:
                    print(f"    {name}: {expr[:80]}")
        else:
            print(f"\n  无池通过回测")

        # 保存审计记录
        audit = funnel.to_audit_record(funnel_pools, time.time() - t2)
        audit_file = Path(__file__).parent / "mining_results" / "funnel_test_result.json"
        audit_file.parent.mkdir(parents=True, exist_ok=True)
        audit_file.write_text(json.dumps(audit, ensure_ascii=False, indent=2))
        print(f"\n  审计记录: {audit_file}")
    else:
        print("  漏斗未生成任何精选池")

    elapsed = time.time() - t_start
    print(f"\n=== 完成 (总耗时 {elapsed:.1f}s) ===")


if __name__ == "__main__":
    main()
