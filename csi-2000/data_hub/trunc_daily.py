"""把 15 分钟聚合成 14:45 截断日线。断点续传, 不改 Qlib 供给层。"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from data_hub.paths import minute_raw_dir, trunc_daily_dir

# 包含 14:45 这一根, 丢掉 15:00。
CUTOFF = "14:45:00"
COLS = ["trade_time", "open", "high", "low", "close", "vol", "amount"]


def _agg_one(
    raw: pd.DataFrame,
    want: set[str] | None = None,
    require_cutoff: bool = True,
) -> pd.DataFrame:
    """单只分钟 -> 截断日线。want 为 YYYY-MM-DD, None 表示全部。

    require_cutoff=False 时, 没有 14:45 但已有 14:30 也聚合,
    给盘中预取用。研究路径保持默认必须命中 14:45。
    """
    t = raw["trade_time"].astype(str)
    day = t.str.slice(0, 10)
    tod = t.str.slice(11, 19)
    keep = tod <= CUTOFF
    if want is not None:
        keep = keep & day.isin(want)
    g = raw.loc[keep]
    if g.empty:
        return pd.DataFrame()
    g = g.copy()
    g["_day"] = day.loc[g.index]
    g["_tod"] = tod.loc[g.index]
    rows = []
    for d, x in g.groupby("_day", sort=True):
        x = x.sort_values("_tod")
        if not (x["_tod"] == CUTOFF).any():
            if require_cutoff:
                continue
            last = str(x["_tod"].iloc[-1])
            if last < "14:00:00":
                continue
        vol = float(x["vol"].sum())
        amt = float(x["amount"].sum())
        vwap = amt / vol if vol > 0 else float("nan")
        rows.append({
            "date": d,
            "open": float(x["open"].iloc[0]),
            "high": float(x["high"].max()),
            "low": float(x["low"].min()),
            "close": float(x["close"].iloc[-1]),
            "volume": vol,
            "amount": amt,
            "vwap": vwap,
            "n_bars": int(len(x)),
            "last_tod": str(x["_tod"].iloc[-1]),
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def build_trunc_daily(limit: int = 0) -> int:
    """遍历分钟缓存, 写出截断日线。返回新写入文件数。"""
    src = minute_raw_dir()
    dst = trunc_daily_dir()
    dst.mkdir(parents=True, exist_ok=True)
    files = sorted(src.glob("*.parquet"))
    if limit:
        files = files[:limit]
    n_new = 0
    for i, p in enumerate(files, 1):
        out = dst / p.name
        if out.exists():
            continue
        raw = pd.read_parquet(p, columns=COLS)
        bar = _agg_one(raw, want=None)
        if bar.empty:
            continue
        bar.to_parquet(out, index=False)
        n_new += 1
        if i % 100 == 0 or i == len(files):
            print(f"[trunc] [{i}/{len(files)}] 新写 {n_new}", flush=True)
    print(f"[trunc] 完成 新写 {n_new} -> {dst}")
    return n_new


def build_trunc_for_dates(
    dates: list[str],
    out: Path | None = None,
    require_cutoff: bool = True,
) -> Path:
    """只聚合指定日期, 写成一张表。有缓存且日期齐全则跳过。"""
    want = set(dates)
    dst = trunc_daily_dir()
    dst.mkdir(parents=True, exist_ok=True)
    out = out or (dst / "_rebal_dates.parquet")
    if out.exists():
        old = pd.read_parquet(out)
        have = set(old["date"].astype(str).unique())
        if want <= have:
            print(f"[trunc] 复用 {out} 天数 {len(have)}")
            return out
    src = minute_raw_dir()
    files = sorted(src.glob("*.parquet"))
    parts = []
    for i, p in enumerate(files, 1):
        raw = pd.read_parquet(p, columns=COLS)
        bar = _agg_one(
            raw, want=want, require_cutoff=require_cutoff,
        )
        if bar.empty:
            continue
        bar.insert(0, "stem", p.stem)
        parts.append(bar)
        if i % 100 == 0 or i == len(files):
            print(f"[trunc] 日期聚合 [{i}/{len(files)}] "
                  f"有数据 {len(parts)}", flush=True)
    if not parts:
        raise SystemExit("指定日期没有截断日线")
    df = pd.concat(parts, ignore_index=True)
    df.to_parquet(out, index=False)
    print(f"[trunc] 写出 {len(df)} 行 {df['stem'].nunique()} 只 -> {out}")
    return out


def main(argv: list[str] | None = None) -> int:
    """入口。"""
    import argparse
    ap = argparse.ArgumentParser(description="14:45 截断日线")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)
    build_trunc_daily(limit=args.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
