"""增量摄入: 只补原始层缺失的交易日, 不直接改 Qlib 目录。"""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

import pandas as pd

from data_hub.assemble import load_membership, ts_to_bao
from data_hub.client import SLEEP, call, pro_api
from data_hub.paths import (
    LEGACY_DAILY,
    LIVE_MINUTE,
    TUSHARE_START,
    daily_raw_dir,
    raw_dir,
)


def missing_dates(start: str, end: str) -> list[str]:
    """交易历里本地还没有 parquet 的日期。"""
    folder = daily_raw_dir()
    have = {p.stem for p in folder.glob("*.parquet")}
    client = pro_api()
    cal = call(
        client.trade_cal,
        exchange="SSE",
        start_date=start.replace("-", ""),
        end_date=end.replace("-", ""),
        is_open="1",
    )
    time.sleep(SLEEP)
    dates = sorted(
        pd.Timestamp(d).strftime("%Y-%m-%d") for d in cal["cal_date"]
    )
    return [d for d in dates if d.replace("-", "") not in have]


def ingest_daily(universe: str = "csi2000",
                 start: str = TUSHARE_START,
                 end: str | None = None) -> int:
    """补齐缺失交易日。返回新写入文件数。"""
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    todo = missing_dates(start, end)
    folder = daily_raw_dir()
    folder.mkdir(parents=True, exist_ok=True)
    if not todo:
        print("[hub] 日线原始层已齐")
        return 0
    snaps = load_membership(universe)
    wanted = {c for v in snaps.values() for c in v}
    client = pro_api()
    print(f"[hub] 待补 {len(todo)} 个交易日 -> {folder}")
    n = 0
    for i, day in enumerate(todo, 1):
        ymd = day.replace("-", "")
        px = call(client.daily, trade_date=ymd)
        adj = call(client.adj_factor, trade_date=ymd)
        if not len(px):
            continue
        merged = px.merge(
            adj[["ts_code", "adj_factor"]], on="ts_code", how="left"
        )
        merged = merged[
            merged["ts_code"].map(lambda c: ts_to_bao(c) in wanted)
        ]
        if len(merged):
            merged = merged.assign(date=day)
            merged.to_parquet(
                folder / f"{ymd}.parquet", index=False
            )
            n += 1
        time.sleep(SLEEP)
        if i % 20 == 0 or i == len(todo):
            print(f"[hub] 摄入 [{i}/{len(todo)}] {day}", flush=True)
    print(f"[hub] 新写入 {n} 个交易日")
    return n


def init_layout() -> None:
    """创建中台目录, 把既有缓存挂成 junction, 不复制、不挪分钟下载。"""
    raw = raw_dir()
    (raw / "index_weight").mkdir(parents=True, exist_ok=True)
    _junction(raw / "daily", LEGACY_DAILY)
    _junction(raw / "stk_mins", LIVE_MINUTE)
    print(f"[hub] 原始层 {raw}")


def _junction(link: Path, target: Path) -> None:
    """Windows 目录联接。目标必须已存在; link 已有数据则跳过。"""
    if not target.exists():
        print(f"[hub] 跳过联接, 目标不存在: {target}")
        return
    if link.exists() and any(link.glob("*.parquet")):
        try:
            same = link.resolve() == target.resolve()
        except OSError:
            same = False
        if same:
            print(f"[hub] 已指向 {target}")
            return
        print(f"[hub] {link} 已有数据, 不覆盖")
        return
    if link.exists():
        try:
            next(link.iterdir())
        except StopIteration:
            link.rmdir()
        else:
            print(f"[hub] {link} 非空且不是目标缓存, 跳过")
            return
    r = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(target)],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        err = (r.stderr or r.stdout).strip()
        print(f"[hub] 联接失败 {link} -> {target}: {err}")
        return
    print(f"[hub] 联接 {link} -> {target}")
