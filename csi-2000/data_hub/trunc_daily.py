"""把 15 分钟聚合成 14:45 截断日线。断点续传, 不改 Qlib 供给层。"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from data_hub.bar_check import check_frame
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
        # vol 的单位**按日判定**, 不能写死。
        #
        # 2026-09-12: Tushare 分钟 vol 在这份缓存里不是统一单位 ——
        # 实测 sh600012/sh600020/sh600033 各约 1685 个交易日中,
        # 只有最后两天 (2026-09-10 / 09-11) 是「手」, 其余全是「股」:
        #     2019-10-08  vol=140,400   amount=765,618      比值 5.45 = close
        #     2026-01-05  vol=984,900   amount=14,411,614   比值 14.63 = close
        #     2026-09-11  vol=5,600     amount=9,029,262    比值 1612  = close x 102
        # 而 Qlib 日线 $volume 的单位是股, auction_infer 会把这里的 volume
        # 覆盖进信号日喂给 Alpha158 —— 单位错 100 倍会让整组量因子失真、
        # VWAP0 从 ~1.0 变成 ~100, 且全程不报错。
        #
        # 先按全历史 x100 修过一版, 那是错的: 拿 2 天的样本推广到 1673 天。
        # 改为用 amount/vol/close 的关系自己判定, Tushare 再改约定也不会中招。
        vol_raw = float(x["vol"].sum())
        amt = float(x["amount"].sum())
        close_px = float(x["close"].iloc[-1])
        vol = vol_raw
        if vol_raw > 0 and close_px > 0:
            ratio = (amt / vol_raw) / close_px
            if 20.0 < ratio < 500.0:
                vol = vol_raw * 100.0        # vol 是「手」, 换成「股」
            elif not (0.5 <= ratio <= 2.0):
                # 既不像股也不像手 —— 不猜, 留给 bar_check 拦下
                pass
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
    # 共享目录不按股票池分开, 认领一下, 防止两棵树静默互相覆盖
    from data_hub.paths import claim_shared_dir
    claim_shared_dir(dst)
    files = sorted(src.glob("*.parquet"))
    if limit:
        files = files[:limit]
    # 量纲版本。2026-09-12 修了 vol 手->股 / vwap 100 倍, 之前建的文件
    # 数值是错的, 而下面的 `if out.exists(): continue` 不会重建它们。
    # 用一个标记文件记版本, 版本不符就全部重建。
    VER = "3"
    vf = dst / ".unit_version"
    cur = vf.read_text(encoding="utf-8").strip() if vf.exists() else "1"
    if cur != VER:
        stale = [f for f in dst.glob("*.parquet")
                 if not f.name.startswith("_")]
        if stale:
            print(f"[trunc] 量纲版本 {cur} -> {VER}, 重建 {len(stale)} 个旧文件",
                  flush=True)
            for f in stale:
                f.unlink()
        vf.write_text(VER, encoding="utf-8")

    n_new = n_bad = 0
    for i, p in enumerate(files, 1):
        out = dst / p.name
        if out.exists():
            continue
        raw = pd.read_parquet(p, columns=COLS)
        bar = _agg_one(raw, want=None)
        if bar.empty:
            continue
        # 落盘前校验量纲 —— 单位错 100 倍在这里就该炸, 而不是等到
        # auction_infer 把它喂进 Alpha158 之后静默失真。
        bar2 = bar.assign(stem=p.stem)
        probs = check_frame(bar2, sample=50)
        if probs:
            print(f"[trunc] {p.stem} 量纲校验不过, 跳过不落盘:", flush=True)
            for m in probs[:3]:
                print(f"    {m}", flush=True)
            n_bad += 1
            continue
        bar.to_parquet(out, index=False)
        n_new += 1
        if i % 100 == 0 or i == len(files):
            print(f"[trunc] [{i}/{len(files)}] 新写 {n_new}", flush=True)
    print(f"[trunc] 完成 新写 {n_new} 量纲不过 {n_bad} -> {dst}")
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
