#!/usr/bin/env python3
"""Tushare daily_basic → Qlib bin

按交易日下载全市场 PE/PB/PS/市值数据，注入 Qlib bin 格式。
Tushare daily_basic 一次返回全市场，按日查询效率极高。

生成字段: pe_ttm, pb, ps_ttm, total_mv, circ_mv
"""
import sys
import time
from pathlib import Path
from datetime import timedelta

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import multiprocessing
multiprocessing.set_start_method('fork', force=True)

import numpy as np
import pandas as pd
import tushare as ts

# 读取 Token
def _get_token():
    env_file = Path(__file__).parent.parent.parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("TUSHARE_TOKEN="):
                return line.split("=", 1)[1].strip().strip("'\"")
    import os
    return os.environ.get("TUSHARE_TOKEN", "")


QLIB_DIR = "~/.qlib/qlib_data/cn_data_bs"
FIELDS = "ts_code,trade_date,pe_ttm,pb,ps_ttm,total_mv,circ_mv"
INJECT_FIELDS = ["pe_ttm", "pb", "ps_ttm", "total_mv", "circ_mv"]


def ts_to_instrument(ts_code: str) -> str:
    """600519.SH → SH600519"""
    parts = ts_code.split(".")
    return f"{parts[1]}{parts[0]}"


def download_daily_basic(pro, trade_dates: list[str],
                         instruments: set[str] | None = None) -> pd.DataFrame:
    """按交易日批量下载 daily_basic

    Args:
        pro: Tushare pro API
        trade_dates: 交易日列表 (YYYYMMDD 格式)
        instruments: 过滤的 Qlib 股票代码集合 (可选)
    """
    all_frames = []
    total = len(trade_dates)

    for i, date_str in enumerate(trade_dates):
        try:
            df = pro.daily_basic(trade_date=date_str, fields=FIELDS)
            if df is not None and len(df) > 0:
                df["date"] = pd.to_datetime(df["trade_date"])
                df["instrument"] = df["ts_code"].apply(ts_to_instrument)
                if instruments:
                    df = df[df["instrument"].isin(instruments)]
                all_frames.append(df)
        except Exception as e:
            print(f"  [tushare] {date_str}: {e}")
            time.sleep(1)  # 出错时等一下

        # Tushare 限速: ~200次/分钟, 每 80 次暂停 1 秒
        if (i + 1) % 80 == 0:
            time.sleep(1)

        if (i + 1) % 100 == 0 or (i + 1) == total:
            print(f"  [tushare] 下载进度 [{i+1}/{total}]")

    if not all_frames:
        return pd.DataFrame()

    result = pd.concat(all_frames, ignore_index=True)
    print(f"  [tushare] 总计: {len(result)} 行, {result['instrument'].nunique()} 只股票")
    return result


def main():
    print("=" * 60)
    print("Phase 2: Tushare daily_basic → Qlib")
    print("=" * 60)

    # 1. 初始化
    token = _get_token()
    if not token:
        print("ERROR: TUSHARE_TOKEN 未设置")
        return
    ts.set_token(token)
    pro = ts.pro_api()

    import qlib
    from qlib.constant import REG_CN
    qlib.init(provider_uri=QLIB_DIR, region=REG_CN)
    from qlib.data import D

    # 2. 获取 CSI300 成分股
    inst = D.instruments("csi300")
    inst_list = D.list_instruments(instruments=inst, as_list=True)
    instruments = set(inst_list)
    print(f"CSI300 成分股: {len(instruments)} 只")

    # 3. 获取交易日历
    from factor_lab.data.qlib_injector import _load_calendar
    cal_index = _load_calendar(QLIB_DIR)
    # 只下载 2018-01-01 ~ 今天
    all_dates = sorted(cal_index.keys())
    trade_dates = [d.replace("-", "") for d in all_dates if d >= "2018-01-01"]
    print(f"交易日: {len(trade_dates)} 天 ({trade_dates[0]} ~ {trade_dates[-1]})")

    # 4. 下载
    print("\n[Step 1] 下载 daily_basic...")
    daily_df = download_daily_basic(pro, trade_dates, instruments)

    if daily_df.empty:
        print("ERROR: 未下载到数据")
        return

    # 5. 缓存
    cache_dir = Path(__file__).parent.parent / "results" / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "tushare_daily_basic.parquet"
    daily_df.to_parquet(cache_path)
    print(f"  缓存至: {cache_path}")

    # 6. 注入 Qlib
    print(f"\n[Step 2] 注入 Qlib bin: {INJECT_FIELDS}")
    from factor_lab.data.qlib_injector import inject_dataframe

    # 转换数值列
    for col in INJECT_FIELDS:
        daily_df[col] = pd.to_numeric(daily_df[col], errors="coerce")

    # total_mv 和 circ_mv 单位是万元，转为元以匹配价格单位
    daily_df["total_mv"] = daily_df["total_mv"] * 10000
    daily_df["circ_mv"] = daily_df["circ_mv"] * 10000

    daily_df["instrument"] = daily_df["instrument"].str.upper()

    inject_dataframe(daily_df, field_columns=INJECT_FIELDS,
                     instrument_col="instrument", date_col="date",
                     qlib_dir=QLIB_DIR)

    # 7. 验证
    print("\n[Step 3] 验证...")
    from factor_lab.data.qlib_injector import verify_field
    for field in INJECT_FIELDS:
        verify_field(field, instrument="sh600519", qlib_dir=QLIB_DIR)

    # 8. 统计
    print(f"\n{'='*60}")
    print(f"Tushare 基本面数据注入完成!")
    print(f"  数据范围: {trade_dates[0]} ~ {trade_dates[-1]}")
    print(f"  股票数: {daily_df['instrument'].nunique()}")
    print(f"  注入字段: {INJECT_FIELDS}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
