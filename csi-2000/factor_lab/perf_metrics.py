"""多维绩效 — 日收益序列上的机构常用指标

只报 Sharpe 会偏爱高波动样本里的一次右尾。选参至少同时看:

- PSR / DSR: Bailey & Lopez de Prado, 样本长度、偏度、峰度、试次
- Sortino / Calmar / Martin: 惩罚亏损波动和回撤深度
- IR / beta: 相对基准是不是只在吃指数
- CVaR / Omega / 回撤持续: 左尾有多坏

用法:
    from factor_lab.perf_metrics import from_returns
    m = from_returns(rets, bench_rets, n_trials=6)
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd

# 年化天数。与 paper_trader Sharpe 口径一致, 不要改成 242.
ANN_DAYS = 252
# Euler-Mascheroni, DSR 期望最大 Sharpe 的近似系数.
EULER_GAMMA = 0.5772156649015329
# CVaR 左尾比例.
CVAR_Q = 0.05
# Omega 阈值 (日收益, 0 = 现金).
OMEGA_THRESHOLD = 0.0


def _norm_cdf(x: float) -> float:
    """标准正态 CDF。"""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """标准正态分位数。p 落在开区间 (0, 1)。"""
    from scipy.stats import norm
    return float(norm.ppf(p))


def _to_series(x: Iterable[float] | pd.Series) -> pd.Series:
    """转成数值 Series 并丢掉非有限值。"""
    if isinstance(x, pd.Series):
        s = x.astype(float)
    else:
        s = pd.Series(x, dtype=float)
    return s.replace([np.inf, -np.inf], np.nan).dropna()


def sharpe_daily(rets: pd.Series) -> float:
    """单期 Sharpe (日频, 未年化)。DSR/PSR 必须用这个, 不能塞年化值。"""
    sd = float(rets.std(ddof=1))
    if sd <= 0 or len(rets) < 2:
        return 0.0
    return float(rets.mean() / sd)


def sr_estimator_std(sr: float, n: int, skew: float,
                     kurt: float) -> float:
    """Sharpe 估计量的标准差 (Bailey / Lopez de Prado)。

    kurt 是 Pearson 峰度 (正态 = 3), 不是 excess。
    """
    if n <= 2:
        return float("nan")
    inside = 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr * sr
    if inside <= 0:
        return float("nan")
    return math.sqrt(inside / (n - 1))


def probabilistic_sharpe(sr: float, n: int, skew: float,
                         kurt: float, sr_star: float = 0.0) -> float:
    """PSR: P(真实 Sharpe > sr_star | 样本矩)。"""
    se = sr_estimator_std(sr, n, skew, kurt)
    if se is None or not math.isfinite(se) or se <= 0:
        return float("nan")
    return _norm_cdf((sr - sr_star) / se)


def expected_max_sr(se: float, n_trials: int) -> float:
    """N 次独立试次下期望最大 Sharpe (日频)。

    SR* = se * [(1-γ) Z^{-1}(1-1/N) + γ Z^{-1}(1-1/(N e))]
    """
    n = max(int(n_trials), 1)
    if n <= 1 or not math.isfinite(se) or se <= 0:
        return 0.0
    z1 = _norm_ppf(1.0 - 1.0 / n)
    z2 = _norm_ppf(1.0 - 1.0 / (n * math.e))
    return se * ((1.0 - EULER_GAMMA) * z1 + EULER_GAMMA * z2)


def deflated_sharpe(sr: float, n: int, skew: float, kurt: float,
                    n_trials: int) -> tuple[float, float]:
    """紧缩夏普 (DSR) 与对应的 SR* (日频)。

    Returns:
        (dsr, sr_star) — dsr 是概率, 不是又一个年化 Sharpe。
    """
    se = sr_estimator_std(sr, n, skew, kurt)
    if se is None or not math.isfinite(se) or se <= 0:
        return float("nan"), float("nan")
    sr_star = expected_max_sr(se, n_trials)
    return probabilistic_sharpe(sr, n, skew, kurt, sr_star), sr_star


def max_drawdown(rets: pd.Series) -> tuple[float, int]:
    """最大回撤 (负数) 与最长水下交易日。"""
    if rets.empty:
        return 0.0, 0
    nav = (1.0 + rets).cumprod()
    dd = nav / nav.cummax() - 1.0
    mdd = float(dd.min()) if len(dd) else 0.0
    under = dd < 0
    longest = 0
    cur = 0
    for flag in under.tolist():
        cur = cur + 1 if flag else 0
        if cur > longest:
            longest = cur
    return mdd, int(longest)


def from_returns(rets: Iterable[float] | pd.Series,
                 bench: Iterable[float] | pd.Series | None = None,
                 n_trials: int = 1) -> dict:
    """从日收益算一套多维指标。

    Args:
        rets: 策略日收益
        bench: 对齐后的基准日收益; 缺了就不报 IR/beta
        n_trials: 已尝试的独立配置数, 进 DSR。至少 1。
    """
    r = _to_series(rets)
    n = int(len(r))
    out = {
        "n_obs": n,
        "n_trials": int(max(n_trials, 1)),
        "ann_return": 0.0,
        "ann_vol": 0.0,
        "sharpe": 0.0,
        "sortino": 0.0,
        "calmar": 0.0,
        "martin": 0.0,
        "omega": 0.0,
        "cvar": 0.0,
        "ulcer": 0.0,
        "max_drawdown": 0.0,
        "dd_days": 0,
        "skew": 0.0,
        "kurtosis": 3.0,
        "psr": float("nan"),
        "dsr": float("nan"),
        "sr_star": float("nan"),
        "sr_star_ann": float("nan"),
        "ir": None,
        "beta": None,
        "te": None,
        "corr": None,
    }
    if n < 3:
        return out

    ann = math.sqrt(ANN_DAYS)
    mu = float(r.mean())
    sd = float(r.std(ddof=1))
    total = float((1.0 + r).prod() - 1.0)
    ann_ret = (1.0 + total) ** (ANN_DAYS / n) - 1.0
    ann_vol = sd * ann if sd > 0 else 0.0
    sr_d = sharpe_daily(r)
    sharpe = sr_d * ann

    downside = r[r < 0]
    dsd = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0
    sortino = (mu / dsd * ann) if dsd > 0 else 0.0

    mdd, dd_days = max_drawdown(r)
    calmar = (ann_ret / abs(mdd)) if mdd < 0 else 0.0

    nav = (1.0 + r).cumprod()
    dd = nav / nav.cummax() - 1.0
    ulcer = float(np.sqrt(np.mean(np.square(dd.to_numpy()))))
    martin = (ann_ret / ulcer) if ulcer > 0 else 0.0

    gains = float(r[r > OMEGA_THRESHOLD].clip(lower=0).sum())
    losses = float((-r[r < OMEGA_THRESHOLD]).clip(lower=0).sum())
    omega = (gains / losses) if losses > 0 else float("inf")

    k = max(int(math.ceil(CVAR_Q * n)), 1)
    cvar = float(r.nsmallest(k).mean())

    skew = float(r.skew())
    # pandas kurtosis 默认 excess; Pearson = excess + 3.
    kurt = float(r.kurtosis()) + 3.0
    psr = probabilistic_sharpe(sr_d, n, skew, kurt, 0.0)
    dsr, sr_star = deflated_sharpe(sr_d, n, skew, kurt, out["n_trials"])

    out.update({
        "ann_return": float(ann_ret),
        "ann_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "calmar": float(calmar),
        "martin": float(martin),
        "omega": float(omega) if math.isfinite(omega) else None,
        "cvar": float(cvar),
        "ulcer": float(ulcer),
        "max_drawdown": float(mdd),
        "dd_days": dd_days,
        "skew": skew,
        "kurtosis": kurt,
        "psr": float(psr) if math.isfinite(psr) else None,
        "dsr": float(dsr) if math.isfinite(dsr) else None,
        "sr_star": float(sr_star) if math.isfinite(sr_star) else None,
        "sr_star_ann": (
            float(sr_star * ann) if math.isfinite(sr_star) else None
        ),
    })

    if bench is None:
        return out
    b = _to_series(bench)
    aligned = pd.concat([r, b], axis=1, join="inner").dropna()
    if aligned.shape[0] < 3:
        return out
    aligned.columns = ["r", "b"]
    te = float(aligned["r"].sub(aligned["b"]).std(ddof=1))
    ir = (
        float(aligned["r"].sub(aligned["b"]).mean() / te * ann)
        if te > 0 else 0.0
    )
    vb = float(aligned["b"].var(ddof=1))
    beta = (
        float(aligned["r"].cov(aligned["b"]) / vb) if vb > 0 else None
    )
    out["ir"] = ir
    out["beta"] = beta
    out["te"] = float(te * ann) if te > 0 else 0.0
    out["corr"] = float(aligned["r"].corr(aligned["b"]))
    return out
