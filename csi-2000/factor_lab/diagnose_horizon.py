#!/usr/bin/env python3
"""诊断持仓周期 vs 隔夜标签、流动性门槛、分钟缓存。

用户指定区间对齐挖掘评估期, 不碰 holdout。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

# 用户指定
START = "2024-01-02"
END = "2026-02-05"
HOLD_DAYS = 8
MIN_AMOUNT = 2_000_000.0
TOPK = 100


def _init():
    """初始化 Qlib。"""
    import qlib
    from qlib_paths import qlib_init_kwargs
    qlib.init(**qlib_init_kwargs())


def _wide(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """MultiIndex -> date x instrument。"""
    s = df[col]
    if s.index.names[0] == "instrument":
        s = s.swaplevel().sort_index()
    return s.unstack(level=1).sort_index()


def _minute_coverage() -> dict:
    """分钟缓存和截断日线覆盖。"""
    from data_hub.paths import minute_raw_dir, trunc_daily_dir
    minute = minute_raw_dir()
    trunc = trunc_daily_dir()
    m_files = list(minute.glob("*.parquet")) if minute.exists() else []
    t_files = list(trunc.glob("*.parquet")) if trunc.exists() else []
    return {
        "minute_dir": str(minute),
        "minute_exists": minute.exists(),
        "minute_files": len(m_files),
        "trunc_dir": str(trunc),
        "trunc_exists": trunc.exists(),
        "trunc_files": len(t_files),
    }


def _load_bars():
    """拉开高低收和成交额。"""
    from qlib.data import D
    from qlib_paths import current_universe
    inst = D.instruments(current_universe())
    df = D.features(
        inst,
        ["$open", "$close", "$amount"],
        start_time=START,
        end_time=END,
    )
    return (
        _wide(df, "$open"),
        _wide(df, "$close"),
        _wide(df, "$amount"),
    )


def _cross_mean(mat: pd.DataFrame) -> float:
    """逐日截面均值再对时间平均。"""
    daily = mat.mean(axis=1, skipna=True)
    return float(daily.mean()) if len(daily) else float("nan")


def _quintile_table(ovn: pd.DataFrame, hold: pd.DataFrame,
                    amount: pd.DataFrame) -> list[dict]:
    """按当日成交额五分位看隔夜和 8 日收益。"""
    rows = []
    dates = ovn.index.intersection(amount.index)
    q_ovn = {i: [] for i in range(5)}
    q_hold = {i: [] for i in range(5)}
    q_n = {i: [] for i in range(5)}
    for dt in dates:
        a = amount.loc[dt].dropna()
        if len(a) < 50:
            continue
        try:
            bins = pd.qcut(a, 5, labels=False, duplicates="drop")
        except ValueError:
            continue
        o = ovn.loc[dt] if dt in ovn.index else None
        h = hold.loc[dt] if dt in hold.index else None
        for q in range(5):
            codes = bins[bins == q].index
            q_n[q].append(len(codes))
            if o is not None:
                q_ovn[q].append(float(o.reindex(codes).mean()))
            if h is not None:
                q_hold[q].append(float(h.reindex(codes).mean()))
    for q in range(5):
        rows.append({
            "quintile": q + 1,
            "label": "最薄" if q == 0 else ("最厚" if q == 4 else f"Q{q+1}"),
            "n_mean": float(np.nanmean(q_n[q])) if q_n[q] else None,
            "ovn_mean_bp": (
                float(np.nanmean(q_ovn[q]) * 1e4) if q_ovn[q] else None
            ),
            "hold8_mean_bp": (
                float(np.nanmean(q_hold[q]) * 1e4) if q_hold[q] else None
            ),
        })
    return rows


def main() -> None:
    """打印诊断 JSON。"""
    _init()
    open_px, close_px, amount = _load_bars()
    ovn = open_px.shift(-1) / close_px - 1.0
    hold8 = close_px.shift(-HOLD_DAYS) / close_px - 1.0
    rest7 = (1.0 + hold8) / (1.0 + ovn) - 1.0

    liquid = amount >= MIN_AMOUNT
    n_names = amount.notna().sum(axis=1)
    n_liquid = liquid.sum(axis=1)
    # 最薄 20% 里有多少过 200 万门槛
    thin_pass = []
    for dt in amount.index:
        a = amount.loc[dt].dropna()
        if len(a) < 50:
            continue
        try:
            bins = pd.qcut(a, 5, labels=False, duplicates="drop")
        except ValueError:
            continue
        thin = a[bins == 0]
        if len(thin) == 0:
            continue
        thin_pass.append(float((thin >= MIN_AMOUNT).mean()))

    pred_dir = (
        PROJECT_DIR / "factor_lab" / "results" / "rolling" / "predictions"
    )
    pkls = sorted(str(p.name) for p in pred_dir.glob("*.pkl")) if pred_dir.exists() else []

    out = {
        "window": {"start": START, "end": END, "hold_days": HOLD_DAYS},
        "minute": _minute_coverage(),
        "universe": {
            "names_per_day": float(n_names.mean()),
            "liquid_per_day": float(n_liquid.mean()),
            "liquid_share": float((n_liquid / n_names.replace(0, np.nan)).mean()),
            "min_amount": MIN_AMOUNT,
        },
        "horizon": {
            "ovn_cross_mean_bp": _cross_mean(ovn) * 1e4,
            "hold8_cross_mean_bp": _cross_mean(hold8) * 1e4,
            "rest7_after_ovn_bp": _cross_mean(rest7) * 1e4,
            "note": (
                "ovn=T收盘→T+1开盘; hold8=T收盘→T+8收盘; "
                "rest7=去掉第一晚之后的 7 日"
            ),
        },
        "amount_quintile": _quintile_table(ovn, hold8, amount),
        "thin_quintile_pass_min_amount": (
            float(np.nanmean(thin_pass)) if thin_pass else None
        ),
        "predictions_pkl": pkls,
        "model_horizon": _model_horizon(ovn, hold8, rest7, amount),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))


def _model_horizon(ovn, hold8, rest7, amount) -> dict:
    """生产预测 TopK 的隔夜 / 8 日 / 后 7 日。"""
    pkl = (
        PROJECT_DIR / "factor_lab" / "results" / "rolling"
        / "predictions"
        / "D_expand_3v_3r_alpha158_ovn_LightGBM.pkl"
    )
    if not pkl.exists():
        return {"error": "no production pkl"}
    pred = pd.read_pickle(pkl)
    if isinstance(pred, pd.DataFrame):
        pred = pred.iloc[:, 0]
    pred = pred.dropna()
    dates = sorted(set(pred.index.get_level_values(0)))
    dates = [d for d in dates if str(pd.Timestamp(d).date()) <= END]
    dates = [d for d in dates if str(pd.Timestamp(d).date()) >= START]
    # 对齐调仓: 每 HOLD_DAYS 取一次
    reb = dates[::HOLD_DAYS]
    buckets = {
        "topk_ovn": [],
        "topk_hold8": [],
        "topk_rest7": [],
        "thin100_ovn": [],
        "thin100_hold8": [],
        "q1_share": [],
        "q5_share": [],
    }
    for dt in reb:
        try:
            s = pred.xs(dt, level=0)
        except KeyError:
            continue
        dts = pd.Timestamp(dt)
        if dts not in amount.index:
            continue
        amt = amount.loc[dts]
        s = s.reindex(amt.dropna().index).dropna()
        if len(s) < TOPK:
            continue
        ranked = s.sort_values(ascending=False)
        top = list(ranked.index[:TOPK])
        # 分数前 200 里再取成交额最薄 100
        head = ranked.head(min(200, len(ranked)))
        thin = list(amt.reindex(head.index).nsmallest(TOPK).index)
        buckets["topk_ovn"].append(
            float(ovn.loc[dts].reindex(top).mean())
        )
        buckets["topk_hold8"].append(
            float(hold8.loc[dts].reindex(top).mean())
        )
        buckets["topk_rest7"].append(
            float(rest7.loc[dts].reindex(top).mean())
        )
        buckets["thin100_ovn"].append(
            float(ovn.loc[dts].reindex(thin).mean())
        )
        buckets["thin100_hold8"].append(
            float(hold8.loc[dts].reindex(thin).mean())
        )
        a = amt.dropna()
        try:
            bins = pd.qcut(a, 5, labels=False, duplicates="drop")
        except ValueError:
            continue
        held = bins.reindex(top).dropna()
        if len(held) == 0:
            continue
        buckets["q1_share"].append(float((held == 0).mean()))
        buckets["q5_share"].append(float((held == 4).mean()))
    def _bp(xs):
        xs = [x for x in xs if pd.notna(x)]
        return float(np.mean(xs) * 1e4) if xs else None
    return {
        "n_rebalance": len(buckets["topk_ovn"]),
        "topk100_ovn_bp": _bp(buckets["topk_ovn"]),
        "topk100_hold8_bp": _bp(buckets["topk_hold8"]),
        "topk100_rest7_bp": _bp(buckets["topk_rest7"]),
        "thin_from_top200_ovn_bp": _bp(buckets["thin100_ovn"]),
        "thin_from_top200_hold8_bp": _bp(buckets["thin100_hold8"]),
        "topk_in_thinnest_q": (
            float(np.mean(buckets["q1_share"]))
            if buckets["q1_share"] else None
        ),
        "topk_in_thickest_q": (
            float(np.mean(buckets["q5_share"]))
            if buckets["q5_share"] else None
        ),
    }


if __name__ == "__main__":
    main()
