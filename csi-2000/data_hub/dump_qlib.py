"""从原始 parquet 重建 Qlib bin。默认不覆盖生产目录。"""
from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

from data_hub.assemble import load_daily_frames, load_membership, to_qlib_bars
from data_hub.paths import membership_file, qlib_serve_dir
from qlib_engine.data_setup import (
    _build_calendar,
    _build_features,
    _build_instruments,
    snapshots_to_intervals,
)


def dump_qlib(universe: str = "csi2000", commit: bool = False) -> Path:
    """装配日线并写入旁路目录; commit=True 才原子替换生产供给层。"""
    t0 = time.time()
    snaps = load_membership(universe)
    stocks, intervals = snapshots_to_intervals(
        {k: set(v) for k, v in snaps.items()}
    )
    wanted = set(stocks)
    print(f"[hub] 装配 {universe} 并集 {len(wanted)} 只")
    df = load_daily_frames()
    stock_map = to_qlib_bars(df, wanted)
    print(f"[hub] 成功 {len(stock_map)}/{len(wanted)} 只")
    if len(stock_map) < len(wanted) * 0.9:
        raise SystemExit(
            f"成功率过低 ({len(stock_map)}/{len(wanted)}), 未改供给层"
        )
    target = qlib_serve_dir(universe)
    build = target.with_name(target.name + ".hub_building")
    if build.exists():
        shutil.rmtree(build)
    build.mkdir(parents=True)
    all_dates = [d for g in stock_map.values() for d in g["date"]]
    calendar = _build_calendar(all_dates, build)
    _build_instruments(stock_map, build, intervals, universe)
    _build_features(stock_map, calendar, build)
    mem = membership_file(universe)
    mem.parent.mkdir(parents=True, exist_ok=True)
    mem.write_text(
        json.dumps(snaps, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    elapsed = (time.time() - t0) / 60
    print(f"[hub] 旁路目录 {build} 耗时 {elapsed:.1f}min")
    if not commit:
        print("[hub] 未覆盖生产目录。确认后加 --commit")
        return build
    _atomic_swap(build, target)
    return target


def _atomic_swap(build: Path, target: Path) -> None:
    """旁路目录原子替换生产供给层。"""
    old = target.with_name(target.name + ".old")
    if old.exists():
        shutil.rmtree(old, ignore_errors=True)
    if target.exists():
        os.replace(target, old)
    os.replace(build, target)
    shutil.rmtree(old, ignore_errors=True)
    print(f"[hub] 已替换 {target}")
