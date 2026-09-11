"""交易日历: 本地缓存 + Tushare trade_cal, 不再问新浪。"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from data_hub.client import call, pro_api
from data_hub.paths import TUSHARE_START, calendar_file, raw_dir

# 用户指定: 缓存路径和向后看多少天。
CAL_CACHE = raw_dir() / "trade_cal.parquet"
LOOKAHEAD_DAYS = 30


def cache_path() -> Path:
    """交易日历 parquet。"""
    return CAL_CACHE


def _ymd(day: date | str) -> str:
    """统一成 YYYY-MM-DD。"""
    if isinstance(day, date):
        return day.strftime("%Y-%m-%d")
    return str(day)[:10]


def refresh_trade_cal(force: bool = False) -> pd.DataFrame:
    """拉上交所日历并落盘。当天已有缓存则跳过。"""
    CAL_CACHE.parent.mkdir(parents=True, exist_ok=True)
    if CAL_CACHE.exists() and not force:
        mtime = date.fromtimestamp(CAL_CACHE.stat().st_mtime)
        if mtime == date.today():
            return pd.read_parquet(CAL_CACHE)
    end = (date.today() + timedelta(days=LOOKAHEAD_DAYS)).strftime(
        "%Y%m%d"
    )
    client = pro_api()
    df = call(
        client.trade_cal,
        exchange="SSE",
        start_date=TUSHARE_START.replace("-", ""),
        end_date=end,
    )
    if df is None or df.empty:
        raise RuntimeError("Tushare trade_cal 为空")
    df = df.copy()
    df["cal_date"] = pd.to_datetime(df["cal_date"]).dt.strftime(
        "%Y-%m-%d"
    )
    df.to_parquet(CAL_CACHE, index=False)
    return df


def open_days() -> list[str]:
    """已开市日期, 优先 Tushare 缓存, 再退到 Qlib 日历。"""
    try:
        df = refresh_trade_cal()
        opened = df[df["is_open"].astype(int) == 1]
        return sorted(opened["cal_date"].astype(str).tolist())
    except Exception:
        path = calendar_file("csi2000")
        if not path.exists():
            return []
        return [
            ln.strip()
            for ln in path.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]


def is_open(day: date | str) -> bool | None:
    """该日是否交易日。答不了返回 None。"""
    key = _ymd(day)
    try:
        df = refresh_trade_cal()
        hit = df[df["cal_date"].astype(str) == key]
        if hit.empty:
            return None
        return int(hit.iloc[0]["is_open"]) == 1
    except Exception:
        lines = open_days()
        if not lines:
            return None
        if key < lines[0] or key > lines[-1]:
            return None
        return key in set(lines)
