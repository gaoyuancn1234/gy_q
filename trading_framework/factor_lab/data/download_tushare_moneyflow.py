#!/usr/bin/env python3
"""Tushare 个股资金流下载 → 注入 Qlib bin

Tushare moneyflow 接口: 按股票查询, 单次最大6000行, 覆盖2010~今。
301只 CSI300 成分股, ~2分钟完成。

注入字段: main_net_inflow, super_large_net, large_net, medium_net, small_net

用法:
    cd trading_framework
    python -m factor_lab.data.download_tushare_moneyflow          # 全量下载+注入
    python -m factor_lab.data.download_tushare_moneyflow --test   # 测试单只
    python -m factor_lab.data.download_tushare_moneyflow --inject # 只从缓存注入
    python -m factor_lab.data.download_tushare_moneyflow --verify # 只验证
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
START_DATE = "2018-01-01"
END_DATE = "2026-02-22"
INJECT_FIELDS = ["main_net_inflow", "super_large_net", "large_net", "medium_net", "small_net"]
CACHE_PATH = Path(__file__).parent.parent / "results" / ".cache" / "tushare_moneyflow.parquet"


def get_csi300_instruments() -> list[str]:
    """从 Qlib instruments 文件获取 CSI300 成分股"""
    inst_file = Path(QLIB_DIR).expanduser() / "instruments" / "csi300.txt"
    instruments = []
    with open(inst_file) as f:
        for line in f:
            parts = line.strip().split("\t")
            if parts:
                instruments.append(parts[0].upper())
    return instruments


def test_single_stock():
    """测试单只股票，验证 Tushare moneyflow API 返回"""
    from factor_lab.data.tushare_dl import TuShareMoneyFlowDownloader

    print("[测试] 下载单只股票资金流 (SH600519)...")
    dl = TuShareMoneyFlowDownloader()
    df = dl.fetch_one("SH600519", START_DATE, END_DATE)

    if df is not None:
        print(f"  返回列名: {list(df.columns)}")
        print(f"  数据行数: {len(df)}")
        print(f"  日期范围: {df['date'].min()} ~ {df['date'].max()}")
        print(f"  前5行:\n{df.head()}")
        print(f"\n  后5行:\n{df.tail()}")
        print(f"\n  统计:\n{df[INJECT_FIELDS].describe()}")
    else:
        print("  ERROR: 未获取到数据")


def download_all(instruments: list[str]) -> pd.DataFrame:
    """全量下载所有股票的资金流数据"""
    from factor_lab.data.tushare_dl import TuShareMoneyFlowDownloader

    print(f"\n[下载] {len(instruments)} 只股票, 按股票逐只下载")

    dl = TuShareMoneyFlowDownloader()
    df = dl.download_all(instruments, START_DATE, END_DATE)
    return df


def save_cache(df: pd.DataFrame):
    """保存 Parquet 缓存"""
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE_PATH)
    print(f"  缓存至: {CACHE_PATH} ({len(df)} 行)")


def load_cache() -> pd.DataFrame | None:
    """从缓存加载"""
    if not CACHE_PATH.exists():
        print(f"ERROR: 缓存不存在: {CACHE_PATH}")
        return None
    df = pd.read_parquet(CACHE_PATH)
    print(f"  从缓存加载: {len(df)} 行, {df['instrument'].nunique()} 只股票")
    return df


def inject(df: pd.DataFrame):
    """注入 Qlib bin"""
    from factor_lab.data.qlib_injector import inject_dataframe

    available = [c for c in INJECT_FIELDS if c in df.columns]
    print(f"\n[注入] {len(df)} 行, {df['instrument'].nunique()} 只股票")
    print(f"  字段: {available}")

    inject_dataframe(df, available, instrument_col="instrument",
                     date_col="date", qlib_dir=QLIB_DIR)


def verify():
    """验证注入结果"""
    from factor_lab.data.qlib_injector import verify_field

    print(f"\n[验证]")
    for field in INJECT_FIELDS:
        verify_field(field, instrument="sh600519", qlib_dir=QLIB_DIR)


def verify_qlib_expression():
    """用 Qlib 表达式引擎测试"""
    print(f"\n[Qlib 表达式验证]")
    try:
        import qlib
        from qlib.constant import REG_CN
        qlib.init(provider_uri=QLIB_DIR, region=REG_CN)
        from qlib.data import D

        inst = D.instruments("csi300")
        test_df = D.features(
            instruments=inst,
            fields=["$main_net_inflow", "Mean($main_net_inflow, 5)"],
            start_time="2024-01-01",
            end_time="2024-01-31",
        )
        valid_pct = test_df.iloc[:, 0].notna().mean()
        print(f"  $main_net_inflow 有效率: {valid_pct:.1%}")
        print(f"  样本 (前10行):")
        print(test_df.head(10))
    except Exception as e:
        print(f"  Qlib 验证跳过: {e}")


def main():
    parser = argparse.ArgumentParser(description="Tushare 个股资金流下载+注入")
    parser.add_argument("--test", action="store_true", help="测试单只股票")
    parser.add_argument("--inject", action="store_true", help="只从缓存注入(跳过下载)")
    parser.add_argument("--verify", action="store_true", help="只验证")
    args = parser.parse_args()

    print("=" * 60)
    print("Tushare 个股资金流 (moneyflow) → Qlib")
    print("=" * 60)

    if args.test:
        test_single_stock()
        return

    if args.verify:
        verify()
        verify_qlib_expression()
        return

    if args.inject:
        df = load_cache()
        if df is not None:
            inject(df)
            verify()
        return

    # 全量下载+注入
    instruments = get_csi300_instruments()
    print(f"CSI300 成分股: {len(instruments)} 只")

    t0 = time.time()
    df = download_all(instruments)
    elapsed = time.time() - t0

    if df.empty:
        print("ERROR: 未下载到数据")
        return

    print(f"\n下载完成: {df['instrument'].nunique()} 只, "
          f"{len(df)} 行, 耗时 {elapsed/60:.1f} 分钟")

    # 缓存
    save_cache(df)

    # 注入
    inject(df)

    # 验证
    verify()
    verify_qlib_expression()

    print(f"\n{'='*60}")
    print(f"个股资金流注入完成!")
    print(f"  数据范围: {df['date'].min()} ~ {df['date'].max()}")
    print(f"  股票数: {df['instrument'].nunique()}")
    print(f"  注入字段: {INJECT_FIELDS}")
    print(f"  总耗时: {elapsed/60:.1f} 分钟")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
