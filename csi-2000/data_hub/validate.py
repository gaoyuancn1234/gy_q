"""离线核对: 不调 Tushare, 只检查本地原始层和 Qlib 供给层。"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from data_hub.assemble import load_membership
from data_hub.paths import (
    TUSHARE_START,
    calendar_file,
    daily_raw_dir,
    minute_raw_dir,
    qlib_serve_dir,
)


def _read_calendar(universe: str) -> list[str]:
    """Qlib day.txt -> 日期列表。"""
    path = calendar_file(universe)
    if not path.exists():
        return []
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()
            if ln.strip()]


def report(universe: str = "csi2000") -> dict:
    """汇总覆盖情况, 供 CLI 打印。"""
    daily = daily_raw_dir()
    minutes = minute_raw_dir()
    day_files = sorted(daily.glob("*.parquet"))
    min_files = list(minutes.glob("*.parquet"))
    cal = _read_calendar(universe)
    snaps = {}
    mem_err = ""
    try:
        snaps = load_membership(universe)
    except SystemExit as exc:
        mem_err = str(exc)
    union = {c for v in snaps.values() for c in v}
    serve = qlib_serve_dir(universe)
    feat = serve / "features"
    n_feat = 0
    if feat.exists():
        n_feat = sum(1 for p in feat.iterdir() if p.is_dir())
    sample_ok = False
    sample_rows = 0
    if day_files:
        sample = pd.read_parquet(day_files[-1])
        sample_rows = len(sample)
        sample_ok = (
            "adj_factor" in sample.columns
            and "close" in sample.columns
            and sample_rows > 0
        )
    out = {
        "universe": universe,
        "tushare_start": TUSHARE_START,
        "daily_dir": str(daily),
        "daily_files": len(day_files),
        "daily_first": day_files[0].stem if day_files else "",
        "daily_last": day_files[-1].stem if day_files else "",
        "qlib_dir": str(serve),
        "qlib_days": len(cal),
        "qlib_first": cal[0] if cal else "",
        "qlib_last": cal[-1] if cal else "",
        "membership_snaps": len(snaps),
        "membership_union": len(union),
        "membership_error": mem_err,
        "qlib_feature_dirs": n_feat,
        "minute_dir": str(minutes),
        "minute_files": len(min_files),
        "sample_rows": sample_rows,
        "sample_ok": sample_ok,
    }
    return out


def validate(universe: str = "csi2000") -> list[str]:
    """返回错误列表, 空列表表示通过。"""
    info = report(universe)
    errors = []
    if info["daily_files"] == 0:
        errors.append("日线原始层为空")
    if not info["sample_ok"]:
        errors.append("最新一日 parquet 缺 adj_factor/close")
    if info["qlib_days"] == 0:
        errors.append("Qlib 日历缺失")
    if info["daily_files"] and info["qlib_days"]:
        if info["daily_files"] != info["qlib_days"]:
            errors.append(
                f"日线文件 {info['daily_files']} != "
                f"Qlib 日历 {info['qlib_days']}"
            )
        last_qlib = info["qlib_last"].replace("-", "")
        if info["daily_last"] != last_qlib:
            errors.append(
                f"日线末日 {info['daily_last']} != "
                f"Qlib 末日 {last_qlib}"
            )
    if info["membership_error"]:
        errors.append(info["membership_error"])
    elif info["membership_union"] < 2000:
        errors.append(
            f"成分股并集过小: {info['membership_union']}"
        )
    feat_dir = Path(info["qlib_dir"]) / "features"
    if not feat_dir.exists():
        errors.append("Qlib features 目录不存在")
    return errors
