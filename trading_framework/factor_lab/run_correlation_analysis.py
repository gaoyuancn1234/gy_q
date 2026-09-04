#!/usr/bin/env python3
"""因子相关性分析与去冗余

1. 加载50个扩展因子值
2. 计算截面相关性矩阵
3. 识别高相关因子对 (>0.7)
4. 基于ICIR去冗余
5. 输出筛选结果
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import multiprocessing
try:
    multiprocessing.set_start_method('fork', force=True)
except (ValueError, RuntimeError):
    pass  # Windows 无 fork，使用默认 spawn

import pandas as pd
import numpy as np

# 初始化 Qlib
import qlib
from qlib.constant import REG_CN
qlib.init(provider_uri='~/.qlib/qlib_data/cn_data_bs', region=REG_CN)

from qlib.data import D
from factor_lab.factors import alpha158_ext
from factor_lab.evaluation.correlation import (
    compute_factor_correlation, remove_redundant, find_high_corr_pairs
)

RESULTS_DIR = Path(__file__).parent / "results" / "factor_eval"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    # 1. 获取因子表达式
    factor_exprs = alpha158_ext.get_all_exprs()
    names = [n for n, _ in factor_exprs]
    exprs = [e for _, e in factor_exprs]
    print(f"加载 {len(factor_exprs)} 个扩展因子")

    # 2. 用 Qlib 表达式引擎加载因子值
    print("加载因子值 (2024-01-01 ~ 2026-02-05)...")
    inst = D.instruments("csi300")
    factor_df = D.features(
        instruments=inst,
        fields=exprs,
        start_time="2024-01-01",
        end_time="2026-02-05",
    )
    factor_df.columns = names
    print(f"  数据形状: {factor_df.shape}")

    # 3. 计算截面相关性矩阵
    print("计算截面 Spearman 相关性...")
    corr_matrix = compute_factor_correlation(factor_df, method="spearman")

    # 保存相关性矩阵
    corr_path = RESULTS_DIR / "alpha158_ext_correlation.csv"
    corr_matrix.to_csv(corr_path)
    print(f"  相关性矩阵保存至: {corr_path}")

    # 4. 找出高相关因子对
    print("\n高相关因子对 (|corr| > 0.7):")
    high_corr_pairs = find_high_corr_pairs(corr_matrix, threshold=0.7)
    if high_corr_pairs:
        for a, b, corr in high_corr_pairs:
            print(f"  {a:25s} <-> {b:25s}  corr = {corr:.4f}")
    else:
        print("  无高相关因子对")

    print(f"\n高相关因子对 (|corr| > 0.5):")
    medium_corr_pairs = find_high_corr_pairs(corr_matrix, threshold=0.5)
    if medium_corr_pairs:
        for a, b, corr in medium_corr_pairs:
            print(f"  {a:25s} <-> {b:25s}  corr = {corr:.4f}")

    # 5. 加载 IC 评价结果
    ic_path = RESULTS_DIR / "alpha158_ext_ic_report.csv"
    ic_report = pd.read_csv(ic_path)
    ic_scores = dict(zip(ic_report["factor"], ic_report["ICIR"].abs()))
    print(f"\n加载 IC 评价: {len(ic_scores)} 个因子")

    # 6. 去冗余 (阈值 0.7)
    kept_07 = remove_redundant(corr_matrix, ic_scores, threshold=0.7)
    removed_07 = set(names) - set(kept_07)
    print(f"\n去冗余 (corr > 0.7): {len(names)} → {len(kept_07)} 因子")
    if removed_07:
        print(f"  移除: {', '.join(sorted(removed_07))}")

    # 7. 去冗余 (阈值 0.5)
    kept_05 = remove_redundant(corr_matrix, ic_scores, threshold=0.5)
    removed_05 = set(names) - set(kept_05)
    print(f"\n去冗余 (corr > 0.5): {len(names)} → {len(kept_05)} 因子")
    if removed_05:
        print(f"  移除: {', '.join(sorted(removed_05))}")

    # 8. 保存去冗余后的因子列表
    selected_factors_07 = [(n, e) for n, e in factor_exprs if n in set(kept_07)]
    selected_factors_05 = [(n, e) for n, e in factor_exprs if n in set(kept_05)]

    # 按 ICIR 排序显示保留因子
    print(f"\n保留因子 (corr > 0.7, 按 |ICIR| 排序):")
    for n, e in sorted(selected_factors_07, key=lambda x: ic_scores.get(x[0], 0), reverse=True):
        icir = ic_scores.get(n, 0)
        print(f"  {n:25s}  |ICIR| = {icir:.4f}")

    # 保存筛选结果
    result = {
        "threshold_0.7": {
            "kept": len(kept_07),
            "removed": len(removed_07),
            "factors": kept_07,
        },
        "threshold_0.5": {
            "kept": len(kept_05),
            "removed": len(removed_05),
            "factors": kept_05,
        },
    }
    import json
    result_path = RESULTS_DIR / "factor_selection_result.json"
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n筛选结果保存至: {result_path}")

    # 9. 汇总统计
    print(f"\n{'='*60}")
    print(f"因子相关性分析汇总")
    print(f"{'='*60}")
    print(f"候选因子:           {len(names)}")
    print(f"高相关对(>0.7):     {len(high_corr_pairs)}")
    print(f"中相关对(>0.5):     {len(medium_corr_pairs)}")
    print(f"去冗余后(0.7阈值):  {len(kept_07)} 因子")
    print(f"去冗余后(0.5阈值):  {len(kept_05)} 因子")

    # 10. 相关性分布统计
    upper_tri = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
    abs_upper = np.abs(upper_tri)
    print(f"\n因子间相关性分布:")
    print(f"  均值:   {abs_upper.mean():.4f}")
    print(f"  中位数: {np.median(abs_upper):.4f}")
    print(f"  最大值: {abs_upper.max():.4f}")
    print(f"  >0.7:   {(abs_upper > 0.7).sum()} 对")
    print(f"  >0.5:   {(abs_upper > 0.5).sum()} 对")
    print(f"  >0.3:   {(abs_upper > 0.3).sum()} 对")


if __name__ == "__main__":
    main()
