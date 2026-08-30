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

import json
import shutil
import sys
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import baostock as bs

DEFAULT_TARGET_DIR = "~/.qlib/qlib_data/cn_data_bs"
# isST: BaoStock 的日线字段，是"当日是否为ST"的时点状态，无前视偏差。
# 中小盘(中证500)出现ST的概率远高于沪深300，必须能在选股时排除。
FIELDS = ["open", "close", "high", "low", "volume", "amount", "turn", "pctChg", "isST"]


def _get_csi300_stocks() -> list:
    """获取沪深300成分股列表（baostock 格式 sh.600519）

    注意: 返回的是【当前】成分股。仅在 point_in_time=False 时使用，
    会带来幸存者偏差 —— 详见 _get_csi300_membership。
    """
    rs = bs.query_hs300_stocks()
    stocks = []
    while rs.error_code == '0' and rs.next():
        row = rs.get_row_data()
        stocks.append(row[1])  # code 字段
    print(f"[data_setup] 沪深300成分股: {len(stocks)} 只")
    return stocks


def _month_ends(start_date: str, end_date: str) -> list:
    """生成 [start_date, end_date] 区间内各月最后一天"""
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    days = pd.date_range(start, end, freq='ME')
    dates = [d.strftime('%Y-%m-%d') for d in days]
    # 保证最后一个采样点覆盖到 end_date（月中时 date_range 不含当月）
    tail = end.strftime('%Y-%m-%d')
    if not dates or dates[-1] != tail:
        dates.append(tail)
    return dates


INDEX_QUERIES = {
    "csi300": ("query_hs300_stocks",),      # 沪深300 — 大盘
    "csi500": ("query_zz500_stocks",),      # 中证500 — 中盘
    "csi800": ("query_hs300_stocks", "query_zz500_stocks"),  # 合并, 两者零重叠
}


def _get_index_membership(start_date: str, end_date: str,
                          universe: str = "csi300") -> tuple:
    """按时点还原指数成分股历史（消除幸存者偏差 / 前视偏差）

    原实现只取"今天"的 300 只并把存续区间压平为全历史，导致:
      - 幸存者偏差: 期间被调出指数的股票（通常是走弱的）完全不在样本里
      - 前视偏差:   后来才调入的股票被当作期初就可交易，而"能入选"本身
                    就是未来信息

    这里按月采样 query_*_stocks(date=...)，还原每只股票的真实进出区间。
    进入时点取"首次出现的采样点"、退出时点取"最后一次出现的采样点"，
    两端都向内收缩，确保不引入前视。

    Returns:
        (union_codes, intervals)
        union_codes: 期间曾入选过的全部股票 (baostock 格式)
        intervals:   {qlib_instrument: [(start, end), ...]} 成分股存续区间
    """
    dates = _month_ends(start_date, end_date)
    print(f"[data_setup] 按时点还原成分股 ({len(dates)} 个采样点)...")

    # 成分股快照缓存: 指数半年才调整一次，每日重复查询 100+ 个历史时点是浪费。
    # 已有的历史月份直接复用，只查缓存里没有的（通常只有当月）。
    cache_file = (Path(__file__).resolve().parent.parent / "data"
                  / f"{universe}_membership.json")
    cached = {}
    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cached = {k: set(v) for k, v in json.load(f).items()}
        except (json.JSONDecodeError, OSError, TypeError) as e:
            print(f"[data_setup] 成分股缓存损坏，忽略并重新采样: {e}")
            cached = {}

    # 最后一个采样点始终重查（当月成分可能刚变动）
    refresh = {dates[-1]} if dates else set()
    todo = [d for d in dates if d not in cached or d in refresh]
    if cached:
        print(f"[data_setup]   缓存命中 {len(dates) - len(todo)}/{len(dates)} 个时点")

    snapshots = {d: cached[d] for d in dates if d in cached and d not in refresh}
    for i, d in enumerate(todo, 1):
        codes = []
        for qname in INDEX_QUERIES[universe]:
            rs = getattr(bs, qname)(date=d)
            while rs.error_code == '0' and rs.next():
                codes.append(rs.get_row_data()[1])
        if codes:
            snapshots[d] = set(codes)
        if i % 20 == 0 or i == len(todo):
            print(f"\r[data_setup]   采样 [{i}/{len(todo)}]", end="", flush=True)
    if todo:
        print()

    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_file.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({k: sorted(v) for k, v in snapshots.items()}, f)
        tmp.replace(cache_file)          # 原子写入，避免中断留下半截文件
    except OSError as e:
        print(f"[data_setup] 成分股缓存写入失败（不影响本次运行）: {e}")

    if not snapshots:
        raise RuntimeError("成分股历史采样失败，未取到任何快照")

    union = set()
    for codes in snapshots.values():
        union |= codes

    # 为每只股票构建连续存续区间
    sorted_dates = sorted(snapshots)
    intervals = {}
    for code in union:
        inst = _bao_to_qlib_instrument(code)
        runs, run_start, prev = [], None, None
        for d in sorted_dates:
            present = code in snapshots[d]
            if present and run_start is None:
                run_start = d
            elif not present and run_start is not None:
                runs.append((run_start, prev))
                run_start = None
            if present:
                prev = d
        if run_start is not None:
            runs.append((run_start, sorted_dates[-1]))
        intervals[inst] = runs

    current = snapshots[sorted_dates[-1]]
    print(f"[data_setup] 历史并集 {len(union)} 只 | 当前成分 {len(current)} 只 | "
          f"期间调出 {len(union - current)} 只")
    print(f"[data_setup] （原实现只下载当前 {len(current)} 只，"
          f"丢失 {len(union - current)} 只样本 = 幸存者偏差）")
    return sorted(union), intervals


def _bao_to_qlib_instrument(bao_code: str) -> str:
    """baostock 代码转 Qlib instrument: sh.600519 -> SH600519"""
    return bao_code.replace(".", "").upper()


def _download_stock_data(bao_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """下载单只股票日线数据"""
    rs = bs.query_history_k_data_plus(
        bao_code,
        "date,open,high,low,close,volume,amount,turn,pctChg,isST",
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

    df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close",
                                      "volume", "amount", "turn", "pctChg", "isST"])
    df["date"] = pd.to_datetime(df["date"])
    for col in FIELDS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["isST"] = df["isST"].fillna(0)

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

    df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close",
                                      "volume", "amount", "turn", "pctChg"])
    df["date"] = pd.to_datetime(df["date"])
    df["isST"] = 0            # 指数无 ST 概念，补 0 保持字段对齐
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


def _build_instruments(stock_map: dict, target_dir: Path, intervals: dict = None,
                       universe: str = 'csi300'):
    """生成 instruments 文件

    Args:
        intervals: {instrument: [(start, end), ...]} 成分股时点存续区间。
                   提供时 csi300.txt 按真实进出区间写多行（Qlib 支持同一
                   标的多行区间），从而在每个时点只使用当时的成分股；
                   为 None 时退回旧行为（全区间，含幸存者偏差）。
    """
    inst_dir = target_dir / "instruments"
    inst_dir.mkdir(parents=True, exist_ok=True)

    lines_all = []
    lines_csi300 = []
    n_stocks = 0
    for instrument, df in stock_map.items():
        data_start = df["date"].min().strftime("%Y-%m-%d")
        data_end = df["date"].max().strftime("%Y-%m-%d")
        lines_all.append(f"{instrument}\t{data_start}\t{data_end}\n")
        if instrument == "SH000300":
            continue
        n_stocks += 1
        if intervals is None:
            lines_csi300.append(f"{instrument}\t{data_start}\t{data_end}\n")
            continue
        # 成分股区间与实际数据区间取交集
        for seg_start, seg_end in intervals.get(instrument, []):
            s = max(seg_start, data_start)
            e = min(seg_end, data_end)
            if s <= e:
                lines_csi300.append(f"{instrument}\t{s}\t{e}\n")

    with open(inst_dir / "all.txt", "w") as f:
        f.writelines(lines_all)
    with open(inst_dir / f"{universe}.txt", "w") as f:
        f.writelines(lines_csi300)

    mode = "时点成分股" if intervals is not None else "全区间(含幸存者偏差)"
    print(f"[data_setup] instruments: {n_stocks} 只股票 + 指数 | "
          f"{universe}.txt {len(lines_csi300)} 行 [{mode}]")


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
                    end_date: str = None,
                    point_in_time: bool = True,
                    universe: str = 'csi300'):
    """下载 A 股数据并转换为 Qlib bin 格式

    Args:
        point_in_time: True（默认）按时点还原沪深300成分股，下载"期间曾入选过"
            的全部股票并写入真实进出区间，消除幸存者偏差/前视偏差。
            False 退回旧行为（只下当前成分股 + 全区间），仅用于复现历史结果。
    """
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

    # 1. 获取沪深300成分股（时点还原 or 当前快照）
    if point_in_time:
        stocks, membership = _get_index_membership(start_date, end_date, universe)
    else:
        stocks, membership = _get_csi300_stocks(), None

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
    else:
        # 指数是基准/择时的依据，缺失必须显式失败，不能静默继续
        raise RuntimeError(
            "沪深300指数(sh.000300)下载失败。常见原因: BaoStock 会话被其他进程 "
            "logout 踢掉 —— 下载期间不要并发运行其他 baostock 脚本。")

    bs.logout()

    if len(stock_map) < 10:
        raise RuntimeError(f"下载的股票数太少 ({len(stock_map)}只)，请检查网络")

    # 4. 构建 Qlib 格式
    if target_path.exists():
        shutil.rmtree(target_path)
    target_path.mkdir(parents=True)

    calendar = _build_calendar(all_dates, target_path)
    _build_instruments(stock_map, target_path, membership, universe)
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
    parser.add_argument('--no-point-in-time', dest='point_in_time',
                        action='store_false',
                        help='退回旧行为: 只下当前成分股+全区间 (含幸存者偏差)')
    parser.add_argument('--universe', default='csi300',
                        choices=list(INDEX_QUERIES.keys()),
                        help='股票池: csi300(大盘) / csi500(中盘) / csi800(合并)')
    parser.set_defaults(point_in_time=True)
    args = parser.parse_args()
    setup_qlib_data(args.target_dir, args.start_date, args.end_date,
                    point_in_time=args.point_in_time, universe=args.universe)


if __name__ == '__main__':
    main()
