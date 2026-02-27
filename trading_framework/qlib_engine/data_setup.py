#!/usr/bin/env python3
"""
Qlib 数据下载与初始化

使用 BaoStock 获取沪深300成分股日线数据，转换为 Qlib bin 格式。
BaoStock 使用 TCP 长连接，不易被限流。

用法:
    python -m qlib_engine.data_setup
    python -m qlib_engine.data_setup --target_dir ~/.qlib/qlib_data/cn_data_bs
    python -m qlib_engine.data_setup --start_date 2018-01-01
"""

import shutil
import sys
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import baostock as bs

DEFAULT_TARGET_DIR = "~/.qlib/qlib_data/cn_data_bs"
FIELDS = ["open", "close", "high", "low", "volume", "amount", "turn", "pctChg"]


def _get_csi300_stocks() -> list:
    """获取沪深300成分股列表（baostock 格式 sh.600519）"""
    rs = bs.query_hs300_stocks()
    stocks = []
    while rs.error_code == '0' and rs.next():
        row = rs.get_row_data()
        stocks.append(row[1])  # code 字段
    print(f"[data_setup] 沪深300成分股: {len(stocks)} 只")
    return stocks


def _bao_to_qlib_instrument(bao_code: str) -> str:
    """baostock 代码转 Qlib instrument: sh.600519 -> SH600519"""
    return bao_code.replace(".", "").upper()


def _download_stock_data(bao_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """下载单只股票日线数据"""
    rs = bs.query_history_k_data_plus(
        bao_code,
        "date,open,high,low,close,volume,amount,turn,pctChg",
        start_date=start_date,
        end_date=end_date,
        frequency="d",
        adjustflag="2",  # 前复权
    )
    if rs.error_code != '0':
        return None

    rows = []
    while rs.next():
        rows.append(rs.get_row_data())

    if not rows:
        return None

    df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume", "amount", "turn", "pctChg"])
    df["date"] = pd.to_datetime(df["date"])
    for col in FIELDS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 过滤掉停牌（volume=0 或 close=0）
    df = df[(df["volume"] > 0) & (df["close"] > 0)]
    return df[["date"] + FIELDS] if not df.empty else None


def _download_index_data(bao_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """下载指数日线数据"""
    rs = bs.query_history_k_data_plus(
        bao_code,
        "date,open,high,low,close,volume,amount,turn,pctChg",
        start_date=start_date,
        end_date=end_date,
        frequency="d",
    )
    if rs.error_code != '0':
        return None

    rows = []
    while rs.next():
        rows.append(rs.get_row_data())

    if not rows:
        return None

    df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume", "amount", "turn", "pctChg"])
    df["date"] = pd.to_datetime(df["date"])
    for col in FIELDS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[["date"] + FIELDS]


def _write_bin_file(data: np.ndarray, path: Path):
    """写 Qlib bin 文件"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(data.tobytes())


def _build_calendar(all_dates: list, target_dir: Path):
    """生成日历文件"""
    cal_dir = target_dir / "calendars"
    cal_dir.mkdir(parents=True, exist_ok=True)

    sorted_dates = sorted(set(all_dates))
    with open(cal_dir / "day.txt", "w") as f:
        for d in sorted_dates:
            f.write(pd.Timestamp(d).strftime("%Y-%m-%d") + "\n")

    print(f"[data_setup] 日历: {len(sorted_dates)} 个交易日"
          f" ({sorted_dates[0]} ~ {sorted_dates[-1]})")
    return sorted_dates


def _build_instruments(stock_map: dict, target_dir: Path):
    """生成 instruments 文件"""
    inst_dir = target_dir / "instruments"
    inst_dir.mkdir(parents=True, exist_ok=True)

    lines_all = []
    lines_csi300 = []
    for instrument, df in stock_map.items():
        start = df["date"].min().strftime("%Y-%m-%d")
        end = df["date"].max().strftime("%Y-%m-%d")
        line = f"{instrument}\t{start}\t{end}\n"
        lines_all.append(line)
        if instrument != "SH000300":
            lines_csi300.append(line)

    with open(inst_dir / "all.txt", "w") as f:
        f.writelines(lines_all)
    with open(inst_dir / "csi300.txt", "w") as f:
        f.writelines(lines_csi300)

    print(f"[data_setup] instruments: {len(lines_csi300)} 只股票 + 指数")


def _build_features(stock_map: dict, calendar: list, target_dir: Path):
    """生成 features bin 文件"""
    feat_dir = target_dir / "features"
    cal_index = {pd.Timestamp(d).strftime("%Y-%m-%d"): i
                 for i, d in enumerate(calendar)}

    count = 0
    for instrument, df in stock_map.items():
        inst_dir = feat_dir / instrument.lower()
        inst_dir.mkdir(parents=True, exist_ok=True)

        df = df.sort_values("date").reset_index(drop=True)
        dates = df["date"].apply(lambda x: x.strftime("%Y-%m-%d"))
        indices = [cal_index[d] for d in dates if d in cal_index]

        if not indices:
            continue

        start_idx = min(indices)
        end_idx = max(indices)
        length = end_idx - start_idx + 1

        for field in FIELDS:
            arr = np.full(length + 2, np.nan, dtype=np.float32)
            arr[0] = np.float32(start_idx)
            arr[1] = np.float32(start_idx + length - 1)

            for _, row in df.iterrows():
                d_str = row["date"].strftime("%Y-%m-%d")
                if d_str in cal_index:
                    pos = cal_index[d_str] - start_idx + 2
                    arr[pos] = np.float32(row[field])

            _write_bin_file(arr, inst_dir / f"{field.lower()}.day.bin")

        count += 1

    print(f"[data_setup] features: {count} 只写入完成")


SLOW_THRESHOLD_SEC = 1800  # 30 分钟


def setup_qlib_data(target_dir: str = DEFAULT_TARGET_DIR,
                    start_date: str = "2018-01-01",
                    end_date: str = None):
    """下载 A 股数据并转换为 Qlib bin 格式"""
    import logging
    log = logging.getLogger(__name__)

    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    target_path = Path(target_dir).expanduser()
    print(f"[data_setup] 目标: {target_path}")
    print(f"[data_setup] 区间: {start_date} ~ {end_date}")

    t_total = time.time()

    # 登录 BaoStock (重试3次)
    for attempt in range(1, 4):
        lg = bs.login()
        if lg.error_code == '0':
            break
        print(f"[data_setup] BaoStock 登录失败 (第{attempt}次): {lg.error_msg}")
        if attempt < 3:
            time.sleep(5 * attempt)
    else:
        raise ConnectionError(f"BaoStock 登录失败 (3次重试): {lg.error_msg}")
    print("[data_setup] BaoStock 登录成功")

    # 1. 获取沪深300成分股
    stocks = _get_csi300_stocks()

    # 2. 下载所有股票数据
    stock_map = {}
    all_dates = []
    total = len(stocks)
    t_download = time.time()

    for i, bao_code in enumerate(stocks, 1):
        if i % 50 == 0 or i == total:
            print(f"\r[data_setup] 下载中 [{i}/{total}] 成功 {len(stock_map)}...",
                  end="", flush=True)
        df = _download_stock_data(bao_code, start_date, end_date)
        if df is not None:
            instrument = _bao_to_qlib_instrument(bao_code)
            stock_map[instrument] = df
            all_dates.extend(df["date"].tolist())

    download_sec = time.time() - t_download
    print(f"\n[data_setup] 成功下载 {len(stock_map)}/{total} 只 "
          f"(耗时 {download_sec:.0f}s)")

    # 3. 下载沪深300指数
    print("[data_setup] 下载沪深300指数...")
    idx_df = _download_index_data("sh.000300", start_date, end_date)
    if idx_df is not None:
        stock_map["SH000300"] = idx_df
        all_dates.extend(idx_df["date"].tolist())
        print(f"[data_setup] 指数数据: {len(idx_df)} 条")

    bs.logout()

    if len(stock_map) < 10:
        raise RuntimeError(f"下载的股票数太少 ({len(stock_map)}只)，请检查网络")

    # 4. 构建 Qlib 格式
    if target_path.exists():
        shutil.rmtree(target_path)
    target_path.mkdir(parents=True)

    calendar = _build_calendar(all_dates, target_path)
    _build_instruments(stock_map, target_path)
    _build_features(stock_map, calendar, target_path)

    total_sec = time.time() - t_total
    print(f"\n[data_setup] 完成! provider_uri='{target_dir}' "
          f"(总耗时 {total_sec:.0f}s)")

    if total_sec > SLOW_THRESHOLD_SEC:
        log.warning(
            f"[data_setup] 数据刷新异常慢: {total_sec:.0f}s "
            f"(>{SLOW_THRESHOLD_SEC // 60}分钟)，"
            f"请检查 BaoStock 连接或网络状况")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Qlib A股数据下载（BaoStock）')
    parser.add_argument('--target_dir', type=str, default=DEFAULT_TARGET_DIR,
                        help=f'数据存储目录 (默认: {DEFAULT_TARGET_DIR})')
    parser.add_argument('--start_date', type=str, default='2018-01-01',
                        help='起始日期 (默认: 2018-01-01)')
    parser.add_argument('--end_date', type=str, default=None,
                        help='结束日期 (默认: 今天)')
    args = parser.parse_args()
    setup_qlib_data(args.target_dir, args.start_date, args.end_date)


if __name__ == '__main__':
    main()
