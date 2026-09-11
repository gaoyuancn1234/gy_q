#!/usr/bin/env python3
"""Step 1: 下载 Tushare 北向资金 → 注入 Qlib bin

北向资金是市场级指标 (每天一个值，所有股票共享)。
用 inject_market_field() 注入到所有股票的 features 目录。

用法:
    cd trading_framework
    python -m factor_lab.data.download_north_money
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import multiprocessing
try:
    multiprocessing.set_start_method('fork', force=True)
except (ValueError, RuntimeError):
    pass  # Windows 无 fork，使用默认 spawn

import pandas as pd

QLIB_DIR = "~/.qlib/qlib_data/cn_data_bs"
START_DATE = "2018-01-01"
END_DATE = "2026-02-08"


def main():
    print("=" * 60)
    print("Phase 3 Step 1: Tushare 北向资金 → Qlib")
    print("=" * 60)

    # 1. 初始化 Tushare
    from factor_lab.data.tushare_dl import TuShareNorthMoneyDownloader
    dl = TuShareNorthMoneyDownloader()

    # 2. 下载
    print(f"\n[Step 1] 下载北向资金 {START_DATE} ~ {END_DATE}...")
    df = dl.download_north_money(START_DATE, END_DATE)

    if df.empty:
        print("ERROR: 未下载到北向资金数据")
        print("请确认 TUSHARE_TOKEN 已正确设置 (.env 或环境变量)")
        return

    print(f"  下载完成: {len(df)} 条记录")
    print(f"  日期范围: {df['date'].min()} ~ {df['date'].max()}")
    print(f"  北向资金样本:")
    print(df.head(10).to_string(index=False))

    # 3. 缓存
    cache_dir = Path(__file__).parent.parent / "results" / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "tushare_north_money.parquet"
    df.to_parquet(cache_path)
    print(f"\n  缓存至: {cache_path}")

    # 4. 注入 Qlib (市场级指标 → 每只股票)
    print(f"\n[Step 2] 注入 Qlib bin (market-level → all stocks)...")
    from factor_lab.data.qlib_injector import inject_market_field

    dates = df["date"].dt.strftime("%Y-%m-%d").tolist()
    values = df["north_money"].tolist()
    inject_market_field("north_money", dates, values, qlib_dir=QLIB_DIR)

    # 5. 验证
    print(f"\n[Step 3] 验证...")
    from factor_lab.data.qlib_injector import verify_field
    verify_field("north_money", instrument="sh600519", qlib_dir=QLIB_DIR)
    verify_field("north_money", instrument="sz000001", qlib_dir=QLIB_DIR)

    # 6. 用 Qlib 表达式引擎测试
    print(f"\n[Step 4] Qlib 表达式验证...")
    try:
        import qlib
        from qlib.constant import REG_CN
        qlib.init(provider_uri=QLIB_DIR, region=REG_CN)
        from qlib.data import D

        inst = D.instruments("csi300")
        test_df = D.features(
            instruments=inst,
            fields=["$north_money", "Mean($north_money, 5)"],
            start_time="2024-01-01",
            end_time="2024-01-31",
        )
        valid_pct = test_df.iloc[:, 0].notna().mean()
        print(f"  $north_money 有效率: {valid_pct:.1%}")
        print(f"  样本 (前10行):")
        print(test_df.head(10))
    except Exception as e:
        print(f"  Qlib 验证跳过: {e}")

    print(f"\n{'='*60}")
    print(f"北向资金注入完成!")
    print(f"  数据范围: {df['date'].min().date()} ~ {df['date'].max().date()}")
    print(f"  记录数: {len(df)}")
    print(f"  可用因子: NORTH_MA5, NORTH_MA20, NORTH_ACC5, NORTH_CHANGE, NORTH_MOM")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
