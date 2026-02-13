"""Qlib bin 注入器 — 将外部数据注入 Qlib features 目录

关键逻辑:
- 读取 calendars/day.txt 建立日期→索引映射
- 按 instrument 分组，写入 {field_name}.day.bin (float32 数组)
- bin 格式: [start_idx(f32), end_idx(f32), data_0, data_1, ..., data_N]
- 注入后即可在 Qlib 表达式中用 $field_name 引用
"""
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_QLIB_DIR = "~/.qlib/qlib_data/cn_data_bs"


def _load_calendar(qlib_dir: str = DEFAULT_QLIB_DIR) -> dict[str, int]:
    """加载日历，返回 {日期字符串: 索引} 映射"""
    cal_path = Path(qlib_dir).expanduser() / "calendars" / "day.txt"
    if not cal_path.exists():
        raise FileNotFoundError(f"日历文件不存在: {cal_path}")

    date_to_idx = {}
    with open(cal_path) as f:
        for i, line in enumerate(f):
            date_str = line.strip()
            if date_str:
                # 统一格式为 YYYY-MM-DD
                date_str = pd.Timestamp(date_str).strftime("%Y-%m-%d")
                date_to_idx[date_str] = i
    return date_to_idx


def _write_bin(data: np.ndarray, path: Path):
    """写入 Qlib bin 文件"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(data.tobytes())


def _read_existing_bin(bin_path: Path):
    """读取已有 bin 文件，返回 (start_idx, end_idx, data_array) 或 None"""
    if not bin_path.exists():
        return None
    arr = np.fromfile(bin_path, dtype=np.float32)
    if len(arr) < 2:
        return None
    return int(arr[0]), int(arr[1]), arr[2:]


def _merge_and_write_bin(bin_path: Path, new_start: int, new_end: int,
                         new_data: dict):
    """合并新旧数据并写入 bin 文件

    Args:
        bin_path: bin 文件路径
        new_start: 新数据最小日历索引
        new_end: 新数据最大日历索引
        new_data: {日历索引: float32值} 字典
    """
    existing = _read_existing_bin(bin_path)

    if existing is not None:
        old_start, old_end, old_arr = existing
        merged_start = min(old_start, new_start)
        merged_end = max(old_end, new_end)
    else:
        merged_start = new_start
        merged_end = new_end

    length = merged_end - merged_start + 1
    arr = np.full(length + 2, np.nan, dtype=np.float32)
    arr[0] = np.float32(merged_start)
    arr[1] = np.float32(merged_end)

    # 先填入旧数据
    if existing is not None:
        old_start, _, old_arr = existing
        for i in range(len(old_arr)):
            pos = old_start + i - merged_start + 2
            if 2 <= pos < len(arr):
                arr[pos] = old_arr[i]

    # 新数据覆盖
    for idx, val in new_data.items():
        pos = idx - merged_start + 2
        if 2 <= pos < len(arr):
            arr[pos] = np.float32(val)

    _write_bin(arr, bin_path)


def inject_field(instrument: str, field_name: str, dates: list[str],
                 values: list[float], qlib_dir: str = DEFAULT_QLIB_DIR,
                 _cal_index: dict[str, int] | None = None):
    """注入单个字段到单只股票的 features 目录

    Args:
        instrument: Qlib instrument 名，如 "SH600519" (大写)
        field_name: 字段名，如 "pe_ttm"
        dates: 日期列表 ["2024-01-02", "2024-01-03", ...]
        values: 对应值列表 [12.3, 12.5, ...]
        qlib_dir: Qlib 数据根目录
        _cal_index: 可选，预加载的日历索引（避免重复加载）
    """
    cal_index = _cal_index or _load_calendar(qlib_dir)
    feat_dir = Path(qlib_dir).expanduser() / "features" / instrument.lower()
    feat_dir.mkdir(parents=True, exist_ok=True)

    # 过滤有效日期
    valid_pairs = []
    for d, v in zip(dates, values):
        d_str = pd.Timestamp(d).strftime("%Y-%m-%d")
        if d_str in cal_index:
            valid_pairs.append((cal_index[d_str], float(v)))

    if not valid_pairs:
        return

    valid_pairs.sort(key=lambda x: x[0])
    new_start = valid_pairs[0][0]
    new_end = valid_pairs[-1][0]
    new_data = {idx: val for idx, val in valid_pairs}

    _merge_and_write_bin(feat_dir / f"{field_name}.day.bin",
                         new_start, new_end, new_data)


def inject_dataframe(df: pd.DataFrame, field_columns: list[str],
                     instrument_col: str = "instrument",
                     date_col: str = "date",
                     qlib_dir: str = DEFAULT_QLIB_DIR):
    """批量注入 DataFrame 中的多个字段

    Args:
        df: 包含 instrument, date, 和多个字段列的 DataFrame
        field_columns: 要注入的字段名列表
        instrument_col: instrument 列名
        date_col: 日期列名
        qlib_dir: Qlib 数据根目录
    """
    cal_index = _load_calendar(qlib_dir)
    feat_base = Path(qlib_dir).expanduser() / "features"

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col]).dt.strftime("%Y-%m-%d")

    grouped = df.groupby(instrument_col)
    total = len(grouped)
    count = 0

    for instrument, group in grouped:
        inst_dir = feat_base / str(instrument).lower()
        inst_dir.mkdir(parents=True, exist_ok=True)

        # 过滤日历中存在的日期
        group = group[group[date_col].isin(cal_index)].sort_values(date_col)
        if group.empty:
            continue

        indices = [cal_index[d] for d in group[date_col]]
        start_idx = min(indices)
        end_idx = max(indices)
        length = end_idx - start_idx + 1

        for field in field_columns:
            if field not in group.columns:
                continue

            new_data = {}
            for idx, val in zip(indices, group[field].values):
                new_data[idx] = float(val) if pd.notna(val) else float('nan')

            _merge_and_write_bin(inst_dir / f"{field}.day.bin",
                                 start_idx, end_idx, new_data)

        count += 1
        if count % 50 == 0 or count == total:
            print(f"  [injector] 注入进度 [{count}/{total}]")

    print(f"  [injector] 完成: {count} 只股票, {len(field_columns)} 个字段")


def inject_market_field(field_name: str, dates: list[str], values: list[float],
                        qlib_dir: str = DEFAULT_QLIB_DIR):
    """注入市场级指标（如北向资金），为每只股票复制同一值

    Args:
        field_name: 字段名，如 "north_money"
        dates: 日期列表
        values: 对应值列表
        qlib_dir: Qlib 数据根目录
    """
    feat_base = Path(qlib_dir).expanduser() / "features"
    if not feat_base.exists():
        raise FileNotFoundError(f"features 目录不存在: {feat_base}")

    # 获取所有股票目录
    instruments = [d.name for d in feat_base.iterdir() if d.is_dir()]
    print(f"  [injector] 市场级指标 {field_name} → {len(instruments)} 只股票")

    # 预加载日历，避免每只股票重复读取
    cal_index = _load_calendar(qlib_dir)
    for instrument in instruments:
        inject_field(instrument.upper(), field_name, dates, values, qlib_dir,
                     _cal_index=cal_index)


def verify_field(field_name: str, instrument: str = "sh600519",
                 qlib_dir: str = DEFAULT_QLIB_DIR, n_samples: int = 5) -> bool:
    """验证字段是否正确注入

    Args:
        field_name: 字段名
        instrument: 股票代码（小写）
        qlib_dir: Qlib 数据根目录
        n_samples: 打印样本数

    Returns:
        True 如果字段存在且有有效数据
    """
    bin_path = Path(qlib_dir).expanduser() / "features" / instrument / f"{field_name}.day.bin"
    if not bin_path.exists():
        print(f"  [verify] {bin_path} 不存在")
        return False

    arr = np.fromfile(bin_path, dtype=np.float32)
    start_idx = int(arr[0])
    end_idx = int(arr[1])
    data = arr[2:]

    valid_count = np.count_nonzero(~np.isnan(data))
    total_count = len(data)

    print(f"  [verify] {field_name} @ {instrument}")
    print(f"    索引范围: [{start_idx}, {end_idx}]")
    print(f"    数据点: {total_count}, 有效: {valid_count} ({valid_count/max(total_count,1)*100:.1f}%)")

    # 打印几个样本
    cal_index = _load_calendar(qlib_dir)
    idx_to_date = {v: k for k, v in cal_index.items()}

    valid_indices = np.where(~np.isnan(data))[0]
    if len(valid_indices) > 0:
        samples = valid_indices[:n_samples]
        print(f"    样本:")
        for si in samples:
            cal_idx = start_idx + si
            date_str = idx_to_date.get(cal_idx, "?")
            print(f"      {date_str}: {data[si]:.4f}")

    return valid_count > 0
