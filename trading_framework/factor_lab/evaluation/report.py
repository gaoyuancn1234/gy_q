"""因子评价报告生成"""
from pathlib import Path

import pandas as pd


def generate_report(ic_result: pd.DataFrame,
                    corr_pairs: list[tuple[str, str, float]] = None,
                    kept_factors: list[str] = None,
                    preset_name: str = "unknown",
                    output_dir: str = None) -> str:
    """生成因子评价文本报告

    Args:
        ic_result: batch_evaluate 的输出
        corr_pairs: 高相关性因子对
        kept_factors: 去冗余后保留的因子
        preset_name: 预设名
        output_dir: 输出目录

    Returns:
        报告文本
    """
    lines = []
    lines.append("=" * 80)
    lines.append(f"因子评价报告 — {preset_name}")
    lines.append("=" * 80)

    # 1. IC 排名
    lines.append(f"\n## 1. IC 排名 (共 {len(ic_result)} 个因子)")
    lines.append("-" * 80)

    if len(ic_result) > 0:
        top_n = min(20, len(ic_result))
        lines.append(f"\nTop {top_n} 因子 (按 |IC| 降序):")
        header = f"{'排名':<4} {'因子':<25} {'mean_IC':>10} {'ICIR':>8} {'IC>0%':>8} {'|IC|':>8}"
        lines.append(header)
        for i, row in ic_result.head(top_n).iterrows():
            lines.append(
                f"{i+1:<4} {row['factor']:<25} {row['mean_IC']:>10.4f} "
                f"{row['ICIR']:>8.3f} {row['IC>0_ratio']:>7.1%} {row['abs_IC']:>8.4f}"
            )

        # 统计
        lines.append(f"\n汇总:")
        lines.append(f"  有效因子: {len(ic_result)}")
        lines.append(f"  |ICIR| > 0.5: {(ic_result['ICIR'].abs() > 0.5).sum()}")
        lines.append(f"  |ICIR| > 1.0: {(ic_result['ICIR'].abs() > 1.0).sum()}")
        lines.append(f"  平均 |IC|: {ic_result['abs_IC'].mean():.4f}")

    # 2. 高相关性因子对
    if corr_pairs:
        lines.append(f"\n## 2. 高相关性因子对 (共 {len(corr_pairs)} 对)")
        lines.append("-" * 80)
        for a, b, c in corr_pairs[:20]:
            lines.append(f"  {a} ↔ {b}: {c:.3f}")

    # 3. 去冗余结果
    if kept_factors:
        lines.append(f"\n## 3. 去冗余后保留 (共 {len(kept_factors)} 个因子)")
        lines.append("-" * 80)
        for i, f in enumerate(kept_factors, 1):
            lines.append(f"  {i:3d}. {f}")

    report = "\n".join(lines)

    # 保存
    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        report_path = out_path / f"{preset_name}_report.txt"
        with open(report_path, "w") as f:
            f.write(report)
        print(f"报告保存至: {report_path}")

    return report
