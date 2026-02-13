"""全量下载 Tushare daily_basic 并注入 Qlib bin 文件

使用方法:
    cd trading_framework && python -m factor_lab.data.download_tushare_full

特点:
- 每 200 个交易日保存 CSV + 注入 bin (支持断点续传)
- 合并注入 (inject 支持 merge，不会覆盖旧数据)
- 限速: 170次/分钟 (2000积分 = 200次/分钟上限)
"""
import os
import sys
import time
from pathlib import Path

import pandas as pd

# 确保 trading_framework 在 sys.path
PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from factor_lab.data.tushare_dl import TuShareDailyBasicDownloader
from factor_lab.data.qlib_injector import inject_dataframe, verify_field

QLIB_DIR = os.path.expanduser("~/.qlib/qlib_data/cn_data_bs")
CACHE_DIR = PROJECT_DIR / "factor_lab" / "results" / ".cache" / "tushare_full"
FIELDS = ["pe_ttm", "pb", "ps_ttm", "total_mv", "circ_mv", "turnover_rate"]
CHUNK_SIZE = 200  # 每 200 个交易日为一个 chunk


def get_csi300_instruments() -> set:
    """获取 CSI300 成分股代码集合"""
    feat_dir = Path(QLIB_DIR) / "features"
    instruments = set()
    for d in feat_dir.iterdir():
        if d.is_dir() and (d / "close.day.bin").exists():
            instruments.add(d.name.upper())
    return instruments


def get_completed_chunks() -> set:
    """获取已完成的 chunk 编号"""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    completed = set()
    for f in CACHE_DIR.glob("chunk_*.csv"):
        try:
            idx = int(f.stem.split("_")[1])
            completed.add(idx)
        except (IndexError, ValueError):
            pass
    return completed


def main():
    print("=" * 60)
    print("Tushare daily_basic 全量下载 + Qlib 注入")
    print("=" * 60)

    dl = TuShareDailyBasicDownloader()
    instruments = get_csi300_instruments()
    print(f"CSI300 成分股: {len(instruments)} 只")

    # 获取交易日列表
    trade_dates = dl._get_trade_dates("2018-01-01", "2026-02-12")
    total_days = len(trade_dates)
    print(f"交易日总数: {total_days}")

    # 分 chunk
    chunks = []
    for i in range(0, total_days, CHUNK_SIZE):
        chunks.append(trade_dates[i:i + CHUNK_SIZE])
    print(f"总 chunk 数: {len(chunks)}")

    completed = get_completed_chunks()
    if completed:
        print(f"已完成 chunk: {sorted(completed)}")

    dl._ensure_api()

    call_count = 0
    batch_start = time.time()

    for ci, chunk_dates in enumerate(chunks):
        if ci in completed:
            # 已完成的 chunk: 直接从 CSV 加载并注入 (确保 bin 存在)
            csv_path = CACHE_DIR / f"chunk_{ci:03d}.csv"
            df = pd.read_csv(csv_path, parse_dates=["date"])
            if not df.empty:
                print(f"[chunk {ci:03d}] 从缓存注入 ({len(df)} 行)")
                inject_dataframe(df, FIELDS, qlib_dir=QLIB_DIR)
            continue

        start_date = chunk_dates[0]
        end_date = chunk_dates[-1]
        print(f"\n[chunk {ci:03d}] 下载 {start_date}~{end_date} "
              f"({len(chunk_dates)} 天)")

        all_frames = []
        for di, date_str in enumerate(chunk_dates):
            try:
                df = dl._pro.daily_basic(
                    trade_date=date_str,
                    fields="ts_code,trade_date,pe_ttm,pb,ps_ttm,"
                           "total_mv,circ_mv,turnover_rate"
                )
                if df is not None and len(df) > 0:
                    df["date"] = pd.to_datetime(df["trade_date"])
                    df["instrument"] = df["ts_code"].apply(
                        lambda x: f"{x.split('.')[1]}{x.split('.')[0]}")
                    df = df[df["instrument"].isin(instruments)]
                    all_frames.append(df)
                call_count += 1
            except Exception as e:
                err_msg = str(e)
                if "最多访问" in err_msg or "每分钟" in err_msg:
                    elapsed = time.time() - batch_start
                    wait = max(62 - elapsed, 5)
                    print(f"  限速, 等待 {wait:.0f}s ({di}/{len(chunk_dates)})")
                    time.sleep(wait)
                    batch_start = time.time()
                    call_count = 0
                    # 重试
                    try:
                        df = dl._pro.daily_basic(
                            trade_date=date_str,
                            fields="ts_code,trade_date,pe_ttm,pb,ps_ttm,"
                                   "total_mv,circ_mv,turnover_rate"
                        )
                        if df is not None and len(df) > 0:
                            df["date"] = pd.to_datetime(df["trade_date"])
                            df["instrument"] = df["ts_code"].apply(
                                lambda x: f"{x.split('.')[1]}{x.split('.')[0]}")
                            df = df[df["instrument"].isin(instruments)]
                            all_frames.append(df)
                        call_count += 1
                    except Exception as e2:
                        print(f"  重试失败 {date_str}: {e2}")
                else:
                    print(f"  {date_str}: {e}")

            # 限速节流
            if call_count >= 170:
                elapsed = time.time() - batch_start
                if elapsed < 62:
                    wait = 62 - elapsed
                    print(f"  节流 {wait:.0f}s")
                    time.sleep(wait)
                batch_start = time.time()
                call_count = 0

        if not all_frames:
            print(f"  [chunk {ci:03d}] 无数据，跳过")
            continue

        chunk_df = pd.concat(all_frames, ignore_index=True)
        days_got = chunk_df["date"].nunique()
        stocks_got = chunk_df["instrument"].nunique()
        print(f"  [chunk {ci:03d}] {len(chunk_df)} 行, "
              f"{days_got} 天, {stocks_got} 只股票")

        # 保存 CSV
        csv_path = CACHE_DIR / f"chunk_{ci:03d}.csv"
        chunk_df.to_csv(csv_path, index=False)

        # 注入 Qlib (支持合并)
        inject_dataframe(chunk_df, FIELDS, qlib_dir=QLIB_DIR)
        print(f"  [chunk {ci:03d}] 已注入")

    # 验证
    print("\n" + "=" * 60)
    print("验证结果:")
    for field in FIELDS:
        verify_field(field, "sh600036", QLIB_DIR)
    print("=" * 60)
    print("完成!")


if __name__ == "__main__":
    main()
