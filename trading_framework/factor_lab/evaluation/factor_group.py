"""分层回测 — 按因子值分5组，计算各组收益

纯 pandas 实现，评估因子的单调性和区分度。
"""
import numpy as np
import pandas as pd


def compute_group_returns(factor_values: pd.Series, returns: pd.Series,
                          n_groups: int = 5) -> pd.DataFrame:
    """按因子值分组计算各组平均收益

    Args:
        factor_values: MultiIndex (datetime, instrument) → factor value
        returns: MultiIndex (datetime, instrument) → forward return
        n_groups: 分组数量

    Returns:
        DataFrame: (date, group) → mean_return
    """
    common = factor_values.index.intersection(returns.index)
    fv = factor_values.loc[common]
    ret = returns.loc[common]

    dates = fv.index.get_level_values(0).unique()
    group_returns = {g: [] for g in range(1, n_groups + 1)}
    date_list = []

    for dt in dates:
        try:
            f = fv.loc[dt].dropna()
            r = ret.loc[dt]

            common_inst = f.index.intersection(r.index)
            if len(common_inst) < n_groups * 2:
                continue

            f = f[common_inst]
            r = r[common_inst]

            # 按因子值排序，分 n 组
            ranked = f.rank(method="first")
            group_size = len(ranked) / n_groups

            for g in range(1, n_groups + 1):
                lower = (g - 1) * group_size
                upper = g * group_size
                mask = (ranked > lower) & (ranked <= upper)
                if mask.sum() > 0:
                    group_returns[g].append(r[mask].mean())
                else:
                    group_returns[g].append(np.nan)

            date_list.append(dt)
        except Exception:
            continue

    result = pd.DataFrame(group_returns, index=date_list)
    result.index.name = "date"
    result.columns = [f"G{i}" for i in range(1, n_groups + 1)]
    return result


def compute_group_cumulative(group_returns: pd.DataFrame) -> pd.DataFrame:
    """计算各组累计收益"""
    return (1 + group_returns).cumprod()


def compute_long_short(group_returns: pd.DataFrame) -> pd.Series:
    """计算多空收益 (Top组 - Bottom组)"""
    return group_returns.iloc[:, -1] - group_returns.iloc[:, 0]


def group_analysis(factor_values: pd.Series, returns: pd.Series,
                   n_groups: int = 5) -> dict:
    """完整的分层分析

    Returns:
        {
            "group_returns": DataFrame (日均收益),
            "group_annual": Series (年化收益),
            "group_sharpe": Series (Sharpe),
            "long_short_annual": float,
            "long_short_sharpe": float,
            "monotonicity": float (单调性分数, -1到1),
        }
    """
    gr = compute_group_returns(factor_values, returns, n_groups)
    if len(gr) < 10:
        return {"error": "数据不足"}

    # 各组年化收益
    annual = gr.mean() * 252
    # 各组 Sharpe
    sharpe = gr.mean() / (gr.std() + 1e-8) * np.sqrt(252)

    # 多空
    ls = compute_long_short(gr)
    ls_annual = ls.mean() * 252
    ls_sharpe = ls.mean() / (ls.std() + 1e-8) * np.sqrt(252)

    # 单调性: Spearman 相关 of group index vs annual return
    from scipy import stats
    groups = np.arange(1, n_groups + 1)
    mono, _ = stats.spearmanr(groups, annual.values)

    return {
        "group_returns": gr,
        "group_annual": annual,
        "group_sharpe": sharpe,
        "long_short_annual": float(ls_annual),
        "long_short_sharpe": float(ls_sharpe),
        "monotonicity": float(mono) if not np.isnan(mono) else 0.0,
    }
