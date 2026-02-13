#!/usr/bin/env python3
"""Step 3: 下载 AKShare 融资融券 → 注入 Qlib bin

新版 AKShare API 按日期下载全市场:
- stock_margin_detail_sse(date) → 上交所
- stock_margin_detail_szse(date) → 深交所
每个交易日一次调用返回全市场，无需逐只下载。

注入字段: margin_balance, short_balance

用法:
    cd trading_framework
    python -m factor_lab.data.download_akshare_margin          # 全量
    python -m factor_lab.data.download_akshare_margin --test   # 测试API
    python -m factor_lab.data.download_akshare_margin --inject # 只注入(已有缓存)
    python -m factor_lab.data.download_akshare_margin --days 60  # 只下最近60天
"""
import sys
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import multiprocessing
multiprocessing.set_start_method('fork', force=True)

import pandas as pd

QLIB_DIR = "~/.qlib/qlib_data/cn_data_bs"
INJECT_FIELDS = ["margin_balance", "short_balance"]


def get_trade_dates(start_date: str = "2018-01-01", end_date: str = "2026-02-08") -> list[str]:
    """从 Qlib 日历获取交易日列表 (YYYYMMDD 格式)"""
    from factor_lab.data.qlib_injector import _load_calendar
    cal_index = _load_calendar(QLIB_DIR)
    all_dates = sorted(cal_index.keys())
    filtered = [d.replace("-", "") for d in all_dates if start_date <= d <= end_date]
    return filtered


def get_csi300_instruments() -> set[str]:
    """从 Qlib instruments 文件获取 CSI300 成分股"""
    inst_file = Path(QLIB_DIR).expanduser() / "instruments" / "csi300.txt"
    instruments = set()
    with open(inst_file) as f:
        for line in f:
            parts = line.strip().split("\t")
            if parts:
                instruments.add(parts[0].upper())
    return instruments


def test_api():
    """测试 AKShare 融资融券 API"""
    import akshare as ak

    print("[测试] AKShare 融资融券 API")

    # 找一个最近的交易日
    trade_dates = get_trade_dates("2026-01-01", "2026-02-08")
    test_date = trade_dates[-1] if trade_dates else "20260205"

    print(f"\n  测试日期: {test_date}")

    print(f"\n  === 上交所 (SSE) ===")
    try:
        df = ak.stock_margin_detail_sse(date=test_date)
        print(f"  列名: {list(df.columns)}")
        print(f"  行数: {len(df)}")
        print(f"  样本:\n{df.head(3)}")
    except Exception as e:
        print(f"  ERROR: {e}")

    print(f"\n  === 深交所 (SZSE) ===")
    try:
        df = ak.stock_margin_detail_szse(date=test_date)
        print(f"  列名: {list(df.columns)}")
        print(f"  行数: {len(df)}")
        print(f"  样本:\n{df.head(3)}")
    except Exception as e:
        print(f"  ERROR: {e}")


def download_margin(trade_dates: list[str], instruments: set[str] = None) -> pd.DataFrame:
    """按交易日下载融资融券数据"""
    from factor_lab.data.akshare_dl import AKShareMarginDownloader

    dl = AKShareMarginDownloader()
    df = dl.download_by_dates(trade_dates, instruments=instruments)
    return df


def inject_dataframe_to_qlib(df: pd.DataFrame):
    """注入 Qlib bin"""
    from factor_lab.data.qlib_injector import inject_dataframe

    if df.empty:
        print("ERROR: 无数据可注入")
        return

    print(f"\n[注入] {len(df)} 行, {df['instrument'].nunique()} 只股票")
    available = [c for c in INJECT_FIELDS if c in df.columns]
    print(f"  注入字段: {available}")

    inject_dataframe(df, available, instrument_col="instrument",
                     date_col="date", qlib_dir=QLIB_DIR)


def inject_from_cache():
    """从缓存加载并注入"""
    cache_dir = Path(__file__).parent.parent / "results" / ".cache" / "akshare_margin"
    cache_path = cache_dir / "margin_all.parquet"

    if cache_path.exists():
        print(f"[注入] 从缓存加载: {cache_path}")
        df = pd.read_parquet(cache_path)
        inject_dataframe_to_qlib(df)
        return True

    print(f"ERROR: 缓存不存在: {cache_path}")
    return False


def verify():
    """验证注入结果"""
    from factor_lab.data.qlib_injector import verify_field

    print(f"\n[验证]")
    for field in INJECT_FIELDS:
        verify_field(field, instrument="sh600519", qlib_dir=QLIB_DIR)
        verify_field(field, instrument="sz000001", qlib_dir=QLIB_DIR)


def main():
    parser = argparse.ArgumentParser(description="AKShare 融资融券下载+注入")
    parser.add_argument("--test", action="store_true", help="测试 API")
    parser.add_argument("--inject", action="store_true", help="只从缓存注入")
    parser.add_argument("--verify", action="store_true", help="只验证")
    parser.add_argument("--days", type=int, default=0,
                        help="只下载最近N个交易日 (0=全量)")
    parser.add_argument("--start", type=str, default="2018-01-01")
    parser.add_argument("--end", type=str, default="2026-02-08")
    args = parser.parse_args()

    print("=" * 60)
    print("Phase 3 Step 3: AKShare 融资融券 → Qlib")
    print("=" * 60)

    if args.test:
        test_api()
        return

    if args.verify:
        verify()
        return

    if args.inject:
        if inject_from_cache():
            verify()
        return

    # 获取交易日和股票列表
    trade_dates = get_trade_dates(args.start, args.end)
    instruments = get_csi300_instruments()

    if args.days > 0:
        trade_dates = trade_dates[-args.days:]

    print(f"CSI300 成分股: {len(instruments)} 只")
    print(f"交易日: {len(trade_dates)} 天 ({trade_dates[0]} ~ {trade_dates[-1]})")
    print(f"  每个交易日 2 次 API 调用 (SSE + SZSE)")
    print(f"  预计: ~{len(trade_dates) * 2 / 60:.0f} 分钟")

    t0 = time.time()
    df = download_margin(trade_dates, instruments)
    elapsed = time.time() - t0

    if df.empty:
        print("ERROR: 未下载到数据")
        return

    n_stocks = df['instrument'].nunique()
    coverage = n_stocks / len(instruments) * 100
    print(f"\n下载完成: {len(df)} 行, {n_stocks} 只股票 ({coverage:.0f}% 覆盖率)")
    print(f"  耗时: {elapsed/60:.1f} 分钟")

    # 缓存
    cache_dir = Path(__file__).parent.parent / "results" / ".cache" / "akshare_margin"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "margin_all.parquet"
    df.to_parquet(cache_path)
    print(f"  缓存至: {cache_path}")

    # 注入
    inject_dataframe_to_qlib(df)

    # 验证
    verify()

    print(f"\n{'='*60}")
    print(f"融资融券注入完成!")
    print(f"  数据范围: {trade_dates[0]} ~ {trade_dates[-1]}")
    print(f"  股票覆盖: {n_stocks} / {len(instruments)} ({coverage:.0f}%)")
    print(f"  注入字段: {INJECT_FIELDS}")
    print(f"  总耗时: {elapsed/60:.1f} 分钟")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
