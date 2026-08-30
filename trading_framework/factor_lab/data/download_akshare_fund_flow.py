#!/usr/bin/env python3
"""Step 2: 下载 AKShare 个股资金流 → 注入 Qlib bin

300 只 CSI300 成分股，80只/批，批间120s。
支持 CSV 增量缓存，中断可续。

注入字段: main_net_inflow, super_large_net, large_net, medium_net, small_net

用法:
    cd trading_framework
    python -m factor_lab.data.download_akshare_fund_flow          # 全量
    python -m factor_lab.data.download_akshare_fund_flow --test   # 测试单只
    python -m factor_lab.data.download_akshare_fund_flow --inject # 只注入(已有缓存)
"""
import sys
import time
import argparse
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
INJECT_FIELDS = ["main_net_inflow", "super_large_net", "large_net", "medium_net", "small_net"]


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
    """测试单只股票，验证 AKShare API 返回格式"""
    import akshare as ak

    print("[测试] 下载单只股票资金流 (SH600519)...")
    test_code = "600519"
    market = "sh"
    try:
        df = ak.stock_individual_fund_flow(stock=test_code, market=market)
        print(f"  返回列名: {list(df.columns)}")
        print(f"  数据行数: {len(df)}")
        print(f"  前5行:\n{df.head()}")
        print(f"\n  数据类型:\n{df.dtypes}")
        return df
    except Exception as e:
        print(f"  ERROR: {e}")
        return None


def download_all(instruments: list[str]) -> dict[str, pd.DataFrame]:
    """批量下载所有股票的资金流数据"""
    from factor_lab.data.akshare_dl import AKShareFundFlowDownloader

    print(f"\n[下载] {len(instruments)} 只股票, 80只/批, 批间120s")
    print(f"  预计耗时: ~{len(instruments) / 80 * 2 + len(instruments) / 80 * 120 / 60:.0f} 分钟 (不含重试)")

    dl = AKShareFundFlowDownloader(batch_size=80, batch_sleep=120)
    results = dl.download(instruments, START_DATE, END_DATE)
    return results


def inject_from_cache():
    """从缓存加载数据并注入 Qlib"""
    from factor_lab.data.qlib_injector import inject_dataframe

    cache_dir = Path(__file__).parent.parent / "results" / ".cache" / "akshare_fund_flow"
    if not cache_dir.exists():
        print(f"ERROR: 缓存目录不存在: {cache_dir}")
        return False

    csv_files = list(cache_dir.glob("*.csv"))
    if not csv_files:
        print(f"ERROR: 缓存目录为空: {cache_dir}")
        return False

    print(f"[注入] 从缓存加载 {len(csv_files)} 个文件...")
    frames = []
    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path, parse_dates=["date"])
            # 从文件名提取 instrument
            fname = csv_path.stem
            # 格式: SH600519_xxxx 或 SZ000001_xxxx
            instrument = fname.split("_")[0].upper()
            df["instrument"] = instrument
            frames.append(df)
        except Exception as e:
            print(f"  跳过 {csv_path.name}: {e}")

    if not frames:
        print("ERROR: 无有效缓存数据")
        return False

    merged = pd.concat(frames, ignore_index=True)
    print(f"  合并: {len(merged)} 行, {merged['instrument'].nunique()} 只股票")

    # 注入
    available = [c for c in INJECT_FIELDS if c in merged.columns]
    print(f"  注入字段: {available}")

    inject_dataframe(merged, available, instrument_col="instrument",
                     date_col="date", qlib_dir=QLIB_DIR)
    return True


def inject_from_results(results: dict[str, pd.DataFrame]):
    """从下载结果直接注入 Qlib"""
    from factor_lab.data.qlib_injector import inject_dataframe

    frames = []
    for instrument, df in results.items():
        df = df.copy()
        df["instrument"] = instrument
        frames.append(df)

    if not frames:
        print("ERROR: 无数据可注入")
        return

    merged = pd.concat(frames, ignore_index=True)
    print(f"\n[注入] {len(merged)} 行, {merged['instrument'].nunique()} 只股票")

    available = [c for c in INJECT_FIELDS if c in merged.columns]
    print(f"  注入字段: {available}")

    inject_dataframe(merged, available, instrument_col="instrument",
                     date_col="date", qlib_dir=QLIB_DIR)


def verify():
    """验证注入结果"""
    from factor_lab.data.qlib_injector import verify_field

    print(f"\n[验证]")
    for field in INJECT_FIELDS:
        verify_field(field, instrument="sh600519", qlib_dir=QLIB_DIR)


def main():
    parser = argparse.ArgumentParser(description="AKShare 个股资金流下载+注入")
    parser.add_argument("--test", action="store_true", help="测试单只股票")
    parser.add_argument("--inject", action="store_true", help="只从缓存注入(跳过下载)")
    parser.add_argument("--verify", action="store_true", help="只验证")
    args = parser.parse_args()

    print("=" * 60)
    print("Phase 3 Step 2: AKShare 个股资金流 → Qlib")
    print("=" * 60)

    if args.test:
        test_single_stock()
        return

    if args.verify:
        verify()
        return

    if args.inject:
        if inject_from_cache():
            verify()
        return

    # 全量下载+注入
    instruments = get_csi300_instruments()
    print(f"CSI300 成分股: {len(instruments)} 只")

    t0 = time.time()
    results = download_all(instruments)
    elapsed = time.time() - t0
    print(f"\n下载完成: {len(results)} 只, 耗时 {elapsed/60:.1f} 分钟")

    # 注入
    inject_from_results(results)

    # 验证
    verify()

    print(f"\n{'='*60}")
    print(f"个股资金流注入完成!")
    print(f"  成功股票: {len(results)} / {len(instruments)}")
    print(f"  注入字段: {INJECT_FIELDS}")
    print(f"  总耗时: {elapsed/60:.1f} 分钟")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
