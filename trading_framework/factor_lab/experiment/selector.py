"""因子选择 — 基于 ICIR 筛选 + 去冗余

从大量候选因子中选出最优子集:
1. 先按 |ICIR| > 阈值过滤
2. 计算相关性矩阵
3. 去除高相关冗余因子
"""
import pandas as pd

from ..evaluation.single_factor import evaluate_with_qlib
from ..evaluation.correlation import compute_factor_correlation, remove_redundant


def select_factors(factor_exprs: list[tuple[str, str]],
                   instruments: str = "csi300",
                   start_time: str = "2024-01-01",
                   end_time: str = "2026-02-05",
                   icir_threshold: float = 0.3,
                   corr_threshold: float = 0.7,
                   max_factors: int = 100) -> list[tuple[str, str]]:
    """因子选择主流程

    Args:
        factor_exprs: 候选因子列表 [(name, expr), ...]
        instruments: 股票池
        start_time, end_time: 评估区间
        icir_threshold: ICIR 绝对值阈值
        corr_threshold: 相关性去冗余阈值
        max_factors: 最终保留的最大因子数

    Returns:
        筛选后的因子列表 [(name, expr), ...]
    """
    print(f"[selector] 候选因子: {len(factor_exprs)}")

    # Step 1: IC 评估
    print("[selector] Step 1: 计算 IC...")
    ic_result = evaluate_with_qlib(factor_exprs, instruments, start_time, end_time)

    if len(ic_result) == 0:
        print("[selector] 无有效因子")
        return []

    # Step 2: ICIR 筛选
    passed = ic_result[ic_result["ICIR"].abs() >= icir_threshold]
    print(f"[selector] Step 2: |ICIR| >= {icir_threshold} → {len(passed)} 个因子")

    if len(passed) == 0:
        # 放宽标准: 取 top-N
        passed = ic_result.head(max_factors)
        print(f"[selector] 放宽: 取 top {len(passed)}")

    passed_names = set(passed["factor"].tolist())
    passed_exprs = [(n, e) for n, e in factor_exprs if n in passed_names]

    # Step 3: 相关性去冗余
    if len(passed_exprs) > 1:
        print(f"[selector] Step 3: 去冗余 (corr > {corr_threshold})...")
        from qlib.data import D

        names = [n for n, _ in passed_exprs]
        exprs = [e for _, e in passed_exprs]

        inst = D.instruments(instruments)
        factor_df = D.features(
            instruments=inst,
            fields=exprs,
            start_time=start_time,
            end_time=end_time,
        )
        factor_df.columns = names

        corr_matrix = compute_factor_correlation(factor_df)
        ic_scores = dict(zip(passed["factor"], passed["ICIR"].abs()))
        kept_names = remove_redundant(corr_matrix, ic_scores, corr_threshold)

        print(f"[selector] 去冗余后: {len(kept_names)} 个因子")
        passed_exprs = [(n, e) for n, e in passed_exprs if n in set(kept_names)]

    # Step 4: 限制数量
    if len(passed_exprs) > max_factors:
        # 按 ICIR 排序取 top
        icir_map = dict(zip(ic_result["factor"], ic_result["ICIR"].abs()))
        passed_exprs.sort(key=lambda x: icir_map.get(x[0], 0), reverse=True)
        passed_exprs = passed_exprs[:max_factors]
        print(f"[selector] 限制数量: {len(passed_exprs)}")

    print(f"[selector] 最终选出: {len(passed_exprs)} 个因子")
    return passed_exprs
