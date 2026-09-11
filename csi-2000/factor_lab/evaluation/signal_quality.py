"""信号质量评估模块 — 滚动 IC、信号离散度、综合评分

用于检测模型信号衰减，构建信号质量指标以驱动动态仓位管理。

前视偏差防范:
- quality_score 使用 expanding percentile rank
- date t 的决策只使用 t-1 及之前的 IC
"""
import numpy as np
import pandas as pd

from factor_lab.evaluation.single_factor import compute_ic


def compute_rolling_ic(signal: pd.Series, returns: pd.Series,
                       window: int = 20,
                       method: str = "spearman") -> pd.Series:
    """计算滚动 RankIC

    Args:
        signal: MultiIndex (datetime, instrument) 预测信号
        returns: MultiIndex (datetime, instrument) 未来收益
        window: 滚动窗口天数
        method: "spearman" 或 "pearson"

    Returns:
        Series indexed by date, value = rolling mean IC (过去 window 天的均值)
    """
    daily_ic = compute_ic(signal, returns, method=method)
    rolling_ic = daily_ic.rolling(window, min_periods=max(5, window // 2)).mean()
    return rolling_ic.dropna()


def compute_signal_dispersion(signal: pd.Series, topk: int = 12) -> pd.Series:
    """计算 topk 预测值的标准差 (信号离散度)

    离散度高 → 模型对不同股票有明确区分 → 信号强
    离散度低 → 预测值趋同 → 信号弱（噪声）

    Args:
        signal: MultiIndex (datetime, instrument) 预测信号
        topk: 取前 topk 个股票计算

    Returns:
        Series indexed by date
    """
    dates = signal.index.get_level_values(0).unique().sort_values()
    disp = {}
    for dt in dates:
        try:
            day_sig = signal.loc[dt]
            if isinstance(day_sig, pd.DataFrame):
                day_sig = day_sig.iloc[:, 0]
            top = day_sig.nlargest(topk)
            if len(top) >= topk // 2:
                disp[dt] = float(top.std())
        except Exception:
            continue
    return pd.Series(disp, name="dispersion")


def compute_signal_spread(signal: pd.Series, topk: int = 12) -> pd.Series:
    """计算 top 与 bottom 预测均值之差

    spread 大 → 模型区分多空方向明确 → 信号强
    spread 小 → 多空不分 → 信号弱

    Args:
        signal: MultiIndex (datetime, instrument) 预测信号
        topk: top/bottom 各取 topk 个

    Returns:
        Series indexed by date
    """
    dates = signal.index.get_level_values(0).unique().sort_values()
    spread = {}
    for dt in dates:
        try:
            day_sig = signal.loc[dt]
            if isinstance(day_sig, pd.DataFrame):
                day_sig = day_sig.iloc[:, 0]
            day_sig = day_sig.dropna()
            if len(day_sig) < 2 * topk:
                continue
            top_mean = day_sig.nlargest(topk).mean()
            bottom_mean = day_sig.nsmallest(topk).mean()
            spread[dt] = float(top_mean - bottom_mean)
        except Exception:
            continue
    return pd.Series(spread, name="spread")


def build_quality_score(ic_20: pd.Series, ic_60: pd.Series,
                        dispersion: pd.Series, spread: pd.Series) -> pd.Series:
    """构建综合信号质量评分 (0~1)

    使用 expanding percentile rank 避免前视偏差。
    每个分项映射到 [0,1]，然后加权融合。

    权重: IC_20 (0.3) + IC_60 (0.3) + dispersion (0.2) + spread (0.2)

    Args:
        ic_20: 20日滚动 IC
        ic_60: 60日滚动 IC
        dispersion: 信号离散度
        spread: 信号 top-bottom 差

    Returns:
        Series indexed by date, value in [0, 1]
    """
    # 对齐到共同日期
    df = pd.DataFrame({
        'ic_20': ic_20,
        'ic_60': ic_60,
        'disp': dispersion,
        'spread': spread,
    }).dropna()

    if len(df) < 10:
        return pd.Series(dtype=float, name="quality_score")

    # Expanding percentile rank: 用历史分位数映射到 [0, 1]
    scored = pd.DataFrame(index=df.index)
    for col in df.columns:
        scored[col] = df[col].expanding(min_periods=10).apply(
            lambda x: (x.rank().iloc[-1] - 1) / max(len(x) - 1, 1),
            raw=False,
        )

    # 加权融合
    weights = {'ic_20': 0.3, 'ic_60': 0.3, 'disp': 0.2, 'spread': 0.2}
    quality = sum(scored[col] * w for col, w in weights.items())
    quality.name = "quality_score"

    return quality.dropna()


def classify_signal_regime(score: pd.Series,
                           thresholds: tuple[float, float] = (0.3, 0.6)) -> pd.Series:
    """根据 quality_score 分类信号状态

    Args:
        score: quality_score Series
        thresholds: (weak_threshold, strong_threshold)

    Returns:
        Series with values "weak" / "normal" / "strong"
    """
    lo, hi = thresholds
    regime = pd.Series("normal", index=score.index, name="regime")
    regime[score < lo] = "weak"
    regime[score >= hi] = "strong"
    return regime


def compute_forward_returns(close_prices: pd.Series,
                            periods: int = 1) -> pd.Series:
    """从收盘价计算 forward return

    Args:
        close_prices: MultiIndex (datetime, instrument) → $close
        periods: 持有期天数

    Returns:
        MultiIndex (datetime, instrument) → forward return
    """
    dates = close_prices.index.get_level_values(0).unique().sort_values()
    instruments = close_prices.index.get_level_values(1).unique()

    # 转为 pivot 计算 shift，再转回 MultiIndex
    pivot = close_prices.unstack()
    if isinstance(pivot.columns, pd.MultiIndex):
        pivot.columns = pivot.columns.droplevel(0)
    fwd_ret = pivot.shift(-periods) / pivot - 1
    result = fwd_ret.stack()
    result.name = "forward_return"
    return result
