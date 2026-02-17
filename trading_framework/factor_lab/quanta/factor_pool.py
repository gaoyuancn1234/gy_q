"""因子池管理 — 准入规则 + 容量控制

论文 Section 4.3:
- 按 |RankIC| 降序排列
- 准入: 相关系数 < REDUNDANCY_CORR AND AST 相似度 < AST_SIMILARITY
- 池容量: min(总挖掘数 × POOL_CAP_RATIO, POOL_MAX)
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

from factor_lab.utils import atomic_json_dump

from .config import (
    REDUNDANCY_CORR, AST_SIMILARITY, POOL_CAP_RATIO, POOL_MAX,
    QUALITY_MIN_ABS_ICIR, QUALITY_MIN_ABS_RANK_IC,
)
from .ast_dedup import ast_similarity


@dataclass
class PoolFactor:
    """池中的一个因子"""
    name: str
    expr: str
    rank_ic: float = 0.0
    icir: float = 0.0
    hypothesis: str = ""
    direction: str = ""
    source_traj_id: str = ""
    iteration: int = 0


class FactorPool:
    """因子池: 管理已发现的因子 + 准入控制"""

    def __init__(self, save_dir: Path):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._factors: list[PoolFactor] = []
        self._total_attempted = 0  # 总尝试入池数

    @property
    def size(self) -> int:
        return len(self._factors)

    @property
    def capacity(self) -> int:
        return min(int(max(self._total_attempted, 10) * POOL_CAP_RATIO), POOL_MAX)

    def try_admit(self, name: str, expr: str,
                  rank_ic: float = 0.0, icir: float = 0.0,
                  hypothesis: str = "", direction: str = "",
                  source_traj_id: str = "", iteration: int = 0,
                  ) -> tuple[bool, str]:
        """尝试将因子加入池

        Returns:
            (admitted, reason)
        """
        self._total_attempted += 1

        # 基本信号强度门槛
        if abs(rank_ic) < 0.005 and abs(icir) < 0.1:
            return False, f"信号太弱 (rank_ic={rank_ic:.4f}, icir={icir:.3f})"

        # 名字唯一性
        existing_names = {f.name for f in self._factors}
        if name in existing_names:
            return False, f"名字重复: {name}"

        # 检查相关性冗余 (需要 Qlib, 这里用 AST 相似度近似)
        for existing in self._factors:
            sim = ast_similarity(expr, existing.expr)
            if sim >= AST_SIMILARITY:
                return False, f"AST 相似度过高 ({sim:.2f} >= {AST_SIMILARITY}) vs {existing.name}"

        # 容量控制: 如果已满, 需要比最差的好
        if self.size >= self.capacity:
            worst = min(self._factors, key=lambda f: abs(f.rank_ic))
            if abs(rank_ic) <= abs(worst.rank_ic):
                return False, f"池已满 ({self.size}/{self.capacity}), 且 rank_ic={rank_ic:.4f} <= worst={worst.rank_ic:.4f}"
            # 踢出最差的
            self._factors.remove(worst)

        # 加入
        self._factors.append(PoolFactor(
            name=name, expr=expr,
            rank_ic=rank_ic, icir=icir,
            hypothesis=hypothesis, direction=direction,
            source_traj_id=source_traj_id, iteration=iteration,
        ))
        self._sort()
        return True, f"admitted (pool={self.size}/{self.capacity})"

    def check_corr_redundancy(self, expr: str) -> tuple[bool, float, str]:
        """检查与池中因子的 Qlib 相关性 (需要 qlib.init)

        Returns:
            (is_redundant, max_corr, most_similar_name)
        """
        try:
            from factor_lab.mining.evaluator import check_redundancy
            new_factor = [("_new_check", expr)]
            result = check_redundancy(
                new_factor,
                baseline_preset="alpha158_val",
                threshold=REDUNDANCY_CORR,
            )
            info = result.get("_new_check", {})
            is_red = info.get("is_redundant", False)
            max_corr = info.get("max_corr", 0.0)
            most_sim = info.get("most_correlated", "")
            return is_red, max_corr, most_sim
        except Exception as e:
            print(f"  [factor_pool] 冗余检查失败: {e}")
            return False, 0.0, ""

    def rebuild(self):
        """按 |RankIC| 降序重建池"""
        self._sort()
        # 去除 AST 冗余
        cleaned = []
        for f in self._factors:
            redundant = False
            for existing in cleaned:
                if ast_similarity(f.expr, existing.expr) >= AST_SIMILARITY:
                    redundant = True
                    break
            if not redundant:
                cleaned.append(f)
        self._factors = cleaned[:self.capacity]

    def get_all(self) -> list[PoolFactor]:
        return list(self._factors)

    def get_exprs(self) -> list[tuple[str, str]]:
        """返回 [(name, expr), ...] 格式"""
        return [(f.name, f.expr) for f in self._factors]

    def get_quality_exprs(
        self,
        min_abs_icir: float = QUALITY_MIN_ABS_ICIR,
        min_abs_rank_ic: float = QUALITY_MIN_ABS_RANK_IC,
    ) -> list[tuple[str, str]]:
        """返回通过质量筛选的因子 [(name, expr), ...]

        比 get_exprs() 更严格: 要求 abs(ICIR) >= 阈值 AND abs(RankIC) >= 阈值。
        用于写入 mined.py，避免弱因子引入噪声。
        """
        quality = []
        for f in self._factors:
            if abs(f.icir) >= min_abs_icir and abs(f.rank_ic) >= min_abs_rank_ic:
                quality.append((f.name, f.expr))
        return quality

    def stats(self) -> dict:
        rank_ics = [abs(f.rank_ic) for f in self._factors] if self._factors else [0]
        return {
            "size": self.size,
            "capacity": self.capacity,
            "total_attempted": self._total_attempted,
            "avg_rank_ic": sum(rank_ics) / len(rank_ics),
            "max_rank_ic": max(rank_ics),
            "min_rank_ic": min(rank_ics),
            "by_direction": _count_by_key(self._factors, 'direction'),
        }

    def save(self, filename: str = "factor_pool.json"):
        path = self.save_dir / filename
        data = {
            "total_attempted": self._total_attempted,
            "factors": [
                {
                    "name": f.name, "expr": f.expr,
                    "rank_ic": f.rank_ic, "icir": f.icir,
                    "hypothesis": f.hypothesis, "direction": f.direction,
                    "source_traj_id": f.source_traj_id, "iteration": f.iteration,
                }
                for f in self._factors
            ],
        }
        atomic_json_dump(path, data, indent=2, ensure_ascii=False)

    def load(self, filename: str = "factor_pool.json"):
        path = self.save_dir / filename
        if not path.exists():
            return
        with open(path) as f:
            data = json.load(f)
        self._total_attempted = data.get("total_attempted", 0)
        self._factors = [
            PoolFactor(**item) for item in data.get("factors", [])
        ]

    def _sort(self):
        self._factors.sort(key=lambda f: abs(f.rank_ic), reverse=True)


def _count_by_key(factors: list[PoolFactor], key: str) -> dict:
    counts: dict[str, int] = {}
    for f in factors:
        val = getattr(f, key, 'unknown')
        counts[val] = counts.get(val, 0) + 1
    return counts


