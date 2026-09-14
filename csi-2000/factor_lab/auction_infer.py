#!/usr/bin/env python3
"""14:45 截断特征 + 已落盘 LightGBM 出分。

用户指定: 历史日线用 Qlib, 只覆盖/追加信号日当天的 14:45 截断 OHLCV。
没有落盘模型就失败, 不准用隔日预测冒充。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

LOOKBACK = 80
EPS = 1e-12


def _inst_stem(inst: str) -> str:
    """SH600012 -> sh600012"""
    return inst[:2].lower() + inst[2:]


def _inst_ts(inst: str) -> str:
    """SH600012 -> 600012.SH"""
    num, ex = inst[2:], inst[:2].upper()
    if num[:2] in ("43", "82", "83", "87", "92"):
        return f"{num}.BJ"
    return f"{num}.{ex}"


def _wide(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """MultiIndex -> date x instrument。"""
    s = df[col]
    if s.index.names[0] == "instrument":
        s = s.swaplevel().sort_index()
    return s.unstack(level=1).sort_index()


def _today_members() -> list[str]:
    """最新时点成分, Qlib 代码。"""
    import json
    mem = json.loads(
        (PROJECT_DIR / "data" / "csi2000_membership.json")
        .read_text(encoding="utf-8")
    )
    latest = max(mem)
    out = []
    for code in mem[latest]:
        if "." in code:
            ex, num = code.split(".")
            out.append(ex.upper() + num)
        else:
            out.append(code)
    return sorted(set(out))


def _trunc_table(day: str) -> pd.DataFrame:
    """聚合指定日的 14:45 截断日线。"""
    from data_hub.trunc_daily import build_trunc_for_dates
    from data_hub.paths import trunc_daily_dir

    out = trunc_daily_dir() / f"_auction_{day.replace('-', '')}.parquet"
    path = build_trunc_for_dates(
        [day],
        out=out,
        require_cutoff=False,
    )
    df = pd.read_parquet(path)
    df["date"] = df["date"].astype(str)
    g = df.loc[df["date"] == day].copy()
    if g.empty:
        raise SystemExit(f"{day} 没有截断日线 (分钟缓存缺 14:00 以后的K)")
    if "last_tod" in g.columns:
        print(
            # 2026-09-13: 原写 .median() —— last_tod 是字符串列, 新版 pandas
            # 直接抛 TypeError。这行在 score_day 的必经路径上, 意味着
            # **实盘 14:45 出分会崩**, 而此前只做过空跑(空跑不到这一步)。
            # 字符串取众数用 mode(), 语义也更对(截断点应当一致)。
            f"[auction] 截断K last_tod 众数 "
            f"{g['last_tod'].astype(str).mode().iat[0] if len(g) else '?'} "
            f"只数 {g['stem'].nunique() if 'stem' in g.columns else len(g)}",
            flush=True,
        )
    return g.drop_duplicates("stem").set_index("stem")


def _ratio_row(
    insts: list[str],
    last_close: np.ndarray,
    last_day: str,
) -> np.ndarray:
    """未复权截断价 -> Qlib 前复权, 用昨日官方收盘对齐。"""
    from data_hub.paths import daily_raw_dir

    ratio = np.ones(len(insts), dtype=np.float64)
    f = daily_raw_dir() / f"{last_day.replace('-', '')}.parquet"
    if not f.exists():
        return ratio
    px0 = pd.read_parquet(f).set_index("ts_code")
    for j, inst in enumerate(insts):
        ts = _inst_ts(inst)
        if ts not in px0.index:
            continue
        raw_c = float(px0.loc[ts, "close"])
        q_c = float(last_close[j])
        if raw_c > 0 and np.isfinite(q_c) and q_c > 0:
            ratio[j] = q_c / raw_c
    return ratio


def _patch_or_append(
    opens, highs, lows, closes, vols, vwaps, insts, day, trunc,
):
    """信号日已在日历里则覆盖, 否则追加一行。"""
    loc_of = {str(d)[:10]: i for i, d in enumerate(opens.index)}
    loc = loc_of.get(day)
    o = opens.to_numpy(dtype=np.float64, copy=True)
    h = highs.to_numpy(dtype=np.float64, copy=True)
    l = lows.to_numpy(dtype=np.float64, copy=True)
    c = closes.to_numpy(dtype=np.float64, copy=True)
    v = vols.to_numpy(dtype=np.float64, copy=True)
    w = vwaps.to_numpy(dtype=np.float64, copy=True)
    last_i = o.shape[0] - 1
    last_day = str(opens.index[-1])[:10]
    ratio = _ratio_row(insts, c[last_i], last_day)
    o_t = np.full(len(insts), np.nan)
    h_t = np.full(len(insts), np.nan)
    l_t = np.full(len(insts), np.nan)
    c_t = np.full(len(insts), np.nan)
    v_t = np.full(len(insts), np.nan)
    w_t = np.full(len(insts), np.nan)
    n_hit = 0
    for j, inst in enumerate(insts):
        stem = _inst_stem(inst)
        if stem not in trunc.index:
            continue
        row = trunc.loc[stem]
        r = ratio[j]
        o_t[j] = float(row["open"]) * r
        h_t[j] = float(row["high"]) * r
        l_t[j] = float(row["low"]) * r
        c_t[j] = float(row["close"]) * r
        v_t[j] = float(row["volume"])
        w_t[j] = float(row["vwap"]) * r
        n_hit += 1
    if n_hit < 50:
        raise SystemExit(f"截断覆盖只有 {n_hit} 只, 拒绝出分")
    if loc is not None:
        o[loc], h[loc], l[loc] = o_t, h_t, l_t
        c[loc], v[loc], w[loc] = c_t, v_t, w_t
        return o, h, l, c, v, w, loc, n_hit
    o = np.vstack([o, o_t])
    h = np.vstack([h, h_t])
    l = np.vstack([l, l_t])
    c = np.vstack([c, c_t])
    v = np.vstack([v, v_t])
    w = np.vstack([w, w_t])
    return o, h, l, c, v, w, o.shape[0] - 1, n_hit


def score_day(day: str, preset: str) -> tuple[pd.Series, dict]:
    """对某一日截断特征打分。返回 (分数, 未复权收盘价)。"""
    import qlib
    from qlib.data import D
    from qlib.contrib.data.loader import Alpha158DL
    from qlib_paths import qlib_init_kwargs
    from factor_lab.lgb_store import (
        apply_robust_zscore,
        load_boosters,
        models_ready,
    )
    from factor_lab._trunc_feat import _feat_at, _stack

    if not models_ready(preset):
        raise SystemExit(
            f"没有可用的 {preset} booster, "
            "先跑 python -m factor_lab.persist_boosters"
        )
    qlib.init(**qlib_init_kwargs("csi2000"))
    trunc = _trunc_table(day)
    insts = _today_members()
    cal = [str(x)[:10] for x in D.calendar()]
    if day in cal:
        loc_day = cal.index(day)
        start = cal[max(0, loc_day - LOOKBACK)]
        end = day
    else:
        start = cal[max(0, len(cal) - LOOKBACK)]
        end = cal[-1]
    raw = D.features(
        insts,
        ["$open", "$high", "$low", "$close", "$volume", "$vwap"],
        start_time=start,
        end_time=end,
    )
    if raw is None or raw.empty:
        raise SystemExit("Qlib 取不到历史日线")
    opens = _wide(raw, "$open")
    highs = _wide(raw, "$high")
    lows = _wide(raw, "$low")
    closes = _wide(raw, "$close")
    vols = _wide(raw, "$volume")
    vwaps = _wide(raw, "$vwap")
    cols = [c for c in insts if c in closes.columns]
    opens, highs = opens[cols], highs[cols]
    lows, closes = lows[cols], closes[cols]
    vols, vwaps = vols[cols], vwaps[cols]
    insts = cols
    o, h, l, c, v, w, loc, n_hit = _patch_or_append(
        opens, highs, lows, closes, vols, vwaps, insts, day, trunc,
    )
    if loc < 60:
        raise SystemExit("历史日线不足 60 天, 算不了 Alpha158")
    _, names = Alpha158DL.get_feature_config()
    feat = _stack(_feat_at(o, h, l, c, v, w, loc), names, insts)
    boosters, info = load_boosters(preset)
    x = apply_robust_zscore(feat, info)
    pred = np.mean([b.predict(x) for b in boosters], axis=0)
    scores = pd.Series(pred, index=insts, name="score")
    scores = scores.replace([np.inf, -np.inf], np.nan).dropna()
    px = {
        inst: float(trunc.loc[_inst_stem(inst), "close"])
        for inst in scores.index
        if _inst_stem(inst) in trunc.index
    }
    print(
        f"[auction] {day} 截断 {n_hit} 只 有效分数 {len(scores)} "
        f"种子 {len(boosters)}",
        flush=True,
    )
    return scores, px
