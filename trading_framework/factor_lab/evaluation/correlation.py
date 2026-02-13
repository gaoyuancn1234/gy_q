"""因子相关性分析与去冗余

计算因子间截面相关性矩阵，自动去除冗余因子 (相关性 > 阈值时保留 ICIR 更高的)。
"""
import numpy as np
import pandas as pd


def compute_factor_correlation(factor_df: pd.DataFrame, method: str = "spearman") -> pd.DataFrame:
    """计算因子间截面平均相关性

    Args:
        factor_df: MultiIndex (datetime, instrument) DataFrame, columns = factor names
        method: "spearman" 或 "pearson"

    Returns:
        因子相关性矩阵 (n_factors × n_factors)
    """
    dates = factor_df.index.get_level_values(0).unique()
    n_factors = factor_df.shape[1]
    corr_sum = np.zeros((n_factors, n_factors))
    count = 0

    for dt in dates:
        try:
            daily = factor_df.loc[dt].dropna(how="all")
            if len(daily) < 20:
                continue
            if method == "spearman":
                c = daily.rank().corr()
            else:
                c = daily.corr()
            corr_sum += c.values
            count += 1
        except Exception:
            continue

    if count == 0:
        return pd.DataFrame(np.eye(n_factors),
                            index=factor_df.columns, columns=factor_df.columns)

    avg_corr = pd.DataFrame(
        corr_sum / count,
        index=factor_df.columns,
        columns=factor_df.columns,
    )
    return avg_corr


def remove_redundant(corr_matrix: pd.DataFrame,
                     ic_scores: dict[str, float],
                     threshold: float = 0.7) -> list[str]:
    """去冗余: 相关性 > 阈值的因子对，保留 ICIR 更高的

    Args:
        corr_matrix: 因子相关性矩阵
        ic_scores: {因子名: ICIR} 或 {因子名: abs_IC}
        threshold: 相关性阈值

    Returns:
        保留的因子名列表
    """
    factors = list(corr_matrix.columns)
    removed = set()

    # 按 IC 分数从低到高排序（先考虑移除低分的）
    sorted_factors = sorted(factors, key=lambda x: abs(ic_scores.get(x, 0)))

    for f in sorted_factors:
        if f in removed:
            continue
        for g in factors:
            if g == f or g in removed:
                continue
            corr = abs(corr_matrix.loc[f, g])
            if corr > threshold:
                # 移除 IC 分数更低的
                if abs(ic_scores.get(f, 0)) < abs(ic_scores.get(g, 0)):
                    removed.add(f)
                    break
                else:
                    removed.add(g)

    kept = [f for f in factors if f not in removed]
    return kept


def find_high_corr_pairs(corr_matrix: pd.DataFrame,
                         threshold: float = 0.7) -> list[tuple[str, str, float]]:
    """找出所有高相关性的因子对

    Returns:
        [(factor_a, factor_b, correlation), ...]
    """
    pairs = []
    factors = list(corr_matrix.columns)
    for i in range(len(factors)):
        for j in range(i + 1, len(factors)):
            corr = abs(corr_matrix.iloc[i, j])
            if corr > threshold:
                pairs.append((factors[i], factors[j], float(corr)))

    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs
