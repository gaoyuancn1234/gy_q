"""把按日 parquet 装配成 Qlib 入仓用的价量表。"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from data_hub.paths import daily_raw_dir, membership_file


def bao_to_qlib(code: str) -> str:
    """sh.600000 -> SH600000"""
    ex, num = code.split(".")
    return f"{ex.upper()}{num}"


def ts_to_bao(ts_code: str) -> str:
    """000001.SZ -> sz.000001"""
    num, ex = ts_code.split(".")
    prefix = "sh" if ex == "SH" else "sz"
    return f"{prefix}.{num}"


def load_membership(universe: str) -> dict:
    """读取时点成分股 {YYYY-MM-DD: [sh.600000, ...]}。"""
    path = membership_file(universe)
    if not path.exists():
        raise SystemExit(f"缺少成分股缓存: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_daily_frames(dates: list[str] | None = None) -> pd.DataFrame:
    """读取原始日线。dates 为 YYYY-MM-DD, 缺省读全部。"""
    folder = daily_raw_dir()
    files = sorted(folder.glob("*.parquet"))
    if dates is not None:
        want = {d.replace("-", "") for d in dates}
        files = [p for p in files if p.stem in want]
    if not files:
        raise SystemExit(f"日线原始层为空: {folder}")
    frames = [pd.read_parquet(p) for p in files]
    return pd.concat(frames, ignore_index=True)


def to_qlib_bars(df: pd.DataFrame, wanted: set[str]) -> dict:
    """未复权日线 -> 前复权 Qlib 字段。wanted 是 baostock 代码。"""
    if df.empty:
        raise SystemExit("没有日线可装配")
    mask = df["ts_code"].map(lambda c: ts_to_bao(c) in wanted)
    df = df.loc[mask].copy()
    df["adj_factor"] = df["adj_factor"].fillna(1.0)
    latest = (
        df.sort_values("date").groupby("ts_code")["adj_factor"].last()
    )
    df["_ratio"] = df["adj_factor"] / df["ts_code"].map(latest)
    out = {}
    for code, g in df.groupby("ts_code"):
        inst = bao_to_qlib(ts_to_bao(code))
        r = g["_ratio"].values
        vol = g["vol"].values * 100.0
        amt = g["amount"].values * 1000.0
        vwap = np.where(
            g["vol"].values > 0,
            amt / np.maximum(vol, 1e-9) * r,
            np.nan,
        )
        o = pd.DataFrame({
            "date": pd.to_datetime(g["date"].values),
            "open": g["open"].values * r,
            "close": g["close"].values * r,
            "high": g["high"].values * r,
            "low": g["low"].values * r,
            "volume": vol,
            "amount": amt,
            "turn": np.nan,
            "pctChg": g["pct_chg"].values,
            "isST": np.nan,
            "factor": r,
            "vwap": vwap,
        }).sort_values("date").reset_index(drop=True)
        o = o[(o["volume"] > 0) & (o["close"] > 0)]
        if len(o):
            out[inst] = o
    return out
