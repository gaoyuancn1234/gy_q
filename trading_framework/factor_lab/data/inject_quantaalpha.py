#!/usr/bin/env python3
"""将 QuantaAlpha 挖掘因子从 H5 缓存注入 Qlib bin 格式

读取 all_factors_library.json 中每个因子的 result.h5，
转换为 Qlib bin 格式写入 ~/.qlib/qlib_data/cn_data_bs/features/

字段名规则: qa_{factor_name_lowercase}  (如 qa_momentum_autocorr_21d)
注入后即可在 Qlib 表达式中用 $qa_xxx 引用
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.qlib_injector import _load_calendar, _merge_and_write_bin

QLIB_DIR = "~/.qlib/qlib_data/cn_data_bs"
FACTOR_LIBRARY = Path(__file__).resolve().parents[3] / "papers" / "QuantaAlpha" / "data" / "factorlib" / "all_factors_library.json"


def load_factor_library(library_path: Path = FACTOR_LIBRARY):
    """加载因子库 JSON"""
    with open(library_path, "r", encoding="utf-8") as f:
        return json.load(f)


def inject_all_factors(library_path: Path = FACTOR_LIBRARY,
                       qlib_dir: str = QLIB_DIR,
                       min_ic: float = 0.0,
                       dry_run: bool = False):
    """注入所有 QuantaAlpha 因子到 Qlib bin

    Args:
        library_path: 因子库 JSON 路径
        qlib_dir: Qlib 数据目录
        min_ic: IC 过滤阈值 (0 表示不过滤)
        dry_run: 仅打印，不实际写入
    """
    data = load_factor_library(library_path)
    cal_index = _load_calendar(qlib_dir)
    feat_base = Path(qlib_dir).expanduser() / "features"

    factors = data["factors"]
    print(f"因子库: {len(factors)} 个因子")

    injected = []
    skipped = []

    for fid, info in factors.items():
        name = info["factor_name"]
        field_name = f"qa_{name.lower()}"
        ic = info.get("backtest_results", {}).get("IC", 0)
        h5_path = info.get("cache_location", {}).get("result_h5_path")

        if abs(ic) < min_ic:
            skipped.append((name, ic, "IC too low"))
            continue

        if not h5_path or not Path(h5_path).exists():
            skipped.append((name, ic, "H5 not found"))
            continue

        if dry_run:
            print(f"  [dry-run] {field_name} (IC={ic:+.4f})")
            injected.append((name, field_name, ic))
            continue

        # Read H5
        series = pd.read_hdf(h5_path, key="data")
        if not isinstance(series.index, pd.MultiIndex):
            skipped.append((name, ic, "bad index"))
            continue

        # Group by instrument and inject
        dt_level = series.index.get_level_values(0)
        inst_level = series.index.get_level_values(1)

        # Build per-instrument data
        df = pd.DataFrame({"date": dt_level, "instrument": inst_level, "value": series.values})
        df["date_str"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

        inst_count = 0
        for instrument, group in df.groupby("instrument"):
            inst_dir = feat_base / str(instrument).lower()
            inst_dir.mkdir(parents=True, exist_ok=True)

            # Filter to calendar dates
            valid = group[group["date_str"].isin(cal_index)]
            if valid.empty:
                continue

            indices = [cal_index[d] for d in valid["date_str"]]
            new_data = {}
            for idx, val in zip(indices, valid["value"].values):
                new_data[idx] = float(val) if pd.notna(val) else float("nan")

            start_idx = min(indices)
            end_idx = max(indices)

            _merge_and_write_bin(
                inst_dir / f"{field_name}.day.bin",
                start_idx, end_idx, new_data
            )
            inst_count += 1

        injected.append((name, field_name, ic))
        print(f"  [{len(injected):>2}/{len(factors)}] {field_name} (IC={ic:+.4f}) → {inst_count} stocks")

    print(f"\n注入完成: {len(injected)} 个因子, 跳过 {len(skipped)} 个")
    if skipped:
        for name, ic, reason in skipped[:5]:
            print(f"  跳过: {name} (IC={ic:+.4f}, {reason})")
        if len(skipped) > 5:
            print(f"  ... 及 {len(skipped) - 5} 个更多")

    return [(name, field_name, ic) for name, field_name, ic in injected]


def get_injected_factor_exprs(library_path: Path = FACTOR_LIBRARY,
                              min_ic: float = 0.0):
    """返回已注入因子的 Qlib 表达式列表 [(name, "$qa_xxx")]

    用于在 presets.py 中引用
    """
    data = load_factor_library(library_path)
    result = []
    for fid, info in data["factors"].items():
        name = info["factor_name"]
        ic = info.get("backtest_results", {}).get("IC", 0)
        if abs(ic) < min_ic:
            continue
        field_name = f"qa_{name.lower()}"
        result.append((name, f"${field_name}"))
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="注入 QuantaAlpha 因子到 Qlib bin")
    parser.add_argument("--min-ic", type=float, default=0.0, help="IC 过滤阈值")
    parser.add_argument("--dry-run", action="store_true", help="仅打印，不写入")
    parser.add_argument("--library", type=str, default=str(FACTOR_LIBRARY))
    args = parser.parse_args()

    inject_all_factors(
        library_path=Path(args.library),
        min_ic=args.min_ic,
        dry_run=args.dry_run,
    )
