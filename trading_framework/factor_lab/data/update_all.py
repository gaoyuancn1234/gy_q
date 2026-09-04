"""一键增量更新所有数据源

用法:
    python -m factor_lab.data.update_all
    python -m factor_lab.data.update_all --source baostock
    python -m factor_lab.data.update_all --source tushare --start 2024-01-01
"""
import sys
from pathlib import Path
from datetime import datetime

# 默认参数
DEFAULT_QLIB_DIR = "~/.qlib/qlib_data/cn_data_bs"
DEFAULT_START = "2018-01-01"


def get_instruments(qlib_dir: str = DEFAULT_QLIB_DIR) -> list[str]:
    """从 Qlib instruments 文件获取股票列表"""
    inst_file = Path(qlib_dir).expanduser() / "instruments" / "csi300.txt"
    if not inst_file.exists():
        print(f"[update] instruments 文件不存在: {inst_file}")
        return []

    instruments = []
    with open(inst_file, encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split("\t")
            if parts:
                instruments.append(parts[0].upper())
    return instruments


def update_baostock(instruments: list[str], start_date: str, end_date: str,
                    qlib_dir: str = DEFAULT_QLIB_DIR):
    """更新 BaoStock 扩展字段 (amount/turn/pctChg)"""
    from .baostock_ext import BaoStockExtDownloader

    print(f"\n{'='*60}")
    print(f"[BaoStock 扩展字段] {start_date} ~ {end_date}")
    print(f"{'='*60}")

    dl = BaoStockExtDownloader()
    dl.download_and_inject(instruments, start_date, end_date, qlib_dir)


def update_tushare_basic(instruments: list[str], start_date: str, end_date: str,
                         qlib_dir: str = DEFAULT_QLIB_DIR):
    """更新 Tushare daily_basic (PE/PB/市值)"""
    from .tushare_dl import TuShareDailyBasicDownloader
    from .qlib_injector import inject_dataframe

    print(f"\n{'='*60}")
    print(f"[Tushare daily_basic] {start_date} ~ {end_date}")
    print(f"{'='*60}")

    dl = TuShareDailyBasicDownloader()
    inst_set = set(instruments)
    df = dl.download_date_range(start_date, end_date, instruments=inst_set)

    if len(df) > 0:
        field_cols = ["pe_ttm", "pb", "ps_ttm", "total_mv", "circ_mv", "turnover_rate"]
        available = [c for c in field_cols if c in df.columns]
        inject_dataframe(df, available, instrument_col="instrument",
                         date_col="date", qlib_dir=qlib_dir)


def update_tushare_north(start_date: str, end_date: str,
                         qlib_dir: str = DEFAULT_QLIB_DIR):
    """更新 Tushare 北向资金 (市场级)"""
    from .tushare_dl import TuShareNorthMoneyDownloader
    from .qlib_injector import inject_market_field

    print(f"\n{'='*60}")
    print(f"[Tushare 北向资金] {start_date} ~ {end_date}")
    print(f"{'='*60}")

    dl = TuShareNorthMoneyDownloader()
    df = dl.download_north_money(start_date, end_date)

    if len(df) > 0:
        inject_market_field(
            "north_money",
            df["date"].dt.strftime("%Y-%m-%d").tolist(),
            df["north_money"].tolist(),
            qlib_dir=qlib_dir,
        )


def update_akshare_fund_flow(instruments: list[str], start_date: str, end_date: str,
                              qlib_dir: str = DEFAULT_QLIB_DIR):
    """更新 AKShare 资金流向"""
    from .akshare_dl import AKShareFundFlowDownloader
    from .qlib_injector import inject_dataframe

    print(f"\n{'='*60}")
    print(f"[AKShare 资金流向] {start_date} ~ {end_date}")
    print(f"  注意: 80只/批，批间120s")
    print(f"{'='*60}")

    dl = AKShareFundFlowDownloader()
    results = dl.download(instruments, start_date, end_date)

    # 合并所有股票的数据
    frames = []
    for instrument, df in results.items():
        df = df.copy()
        df["instrument"] = instrument
        frames.append(df)

    if frames:
        merged = pd.concat(frames, ignore_index=True)
        field_cols = ["main_net_inflow", "super_large_net", "large_net", "medium_net", "small_net"]
        available = [c for c in field_cols if c in merged.columns]
        inject_dataframe(merged, available, instrument_col="instrument",
                         date_col="date", qlib_dir=qlib_dir)


def update_akshare_margin(instruments: list[str], start_date: str, end_date: str,
                           qlib_dir: str = DEFAULT_QLIB_DIR):
    """更新 AKShare 融资融券 (按日期下载全市场)"""
    from .akshare_dl import AKShareMarginDownloader
    from .qlib_injector import inject_dataframe, _load_calendar

    print(f"\n{'='*60}")
    print(f"[AKShare 融资融券] {start_date} ~ {end_date}")
    print(f"{'='*60}")

    # 获取交易日列表
    cal_index = _load_calendar(qlib_dir)
    trade_dates = sorted([d.replace("-", "") for d in cal_index.keys()
                          if start_date <= d <= end_date])

    dl = AKShareMarginDownloader()
    inst_set = set(instruments)
    merged = dl.download_by_dates(trade_dates, instruments=inst_set)

    if len(merged) > 0:
        field_cols = ["margin_balance", "short_balance"]
        available = [c for c in field_cols if c in merged.columns]
        inject_dataframe(merged, available, instrument_col="instrument",
                         date_col="date", qlib_dir=qlib_dir)


# 数据源映射
SOURCE_MAP = {
    "baostock": update_baostock,
    "tushare_basic": update_tushare_basic,
    "tushare_north": update_tushare_north,
    "akshare_fund": update_akshare_fund_flow,
    "akshare_margin": update_akshare_margin,
}


def update_all(start_date: str = DEFAULT_START, end_date: str = None,
               qlib_dir: str = DEFAULT_QLIB_DIR, sources: list[str] = None):
    """更新所有数据源

    Args:
        start_date: 起始日期
        end_date: 结束日期 (默认今天)
        qlib_dir: Qlib 数据目录
        sources: 要更新的数据源列表 (默认全部)
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    instruments = get_instruments(qlib_dir)
    if not instruments:
        print("[update] 无法获取股票列表，请先运行 data_setup.py")
        return

    print(f"[update] 股票数: {len(instruments)}")
    print(f"[update] 日期范围: {start_date} ~ {end_date}")

    if sources is None:
        sources = ["baostock"]  # 默认只更新免费且无限速的
        print("[update] 默认只更新 BaoStock (免费无限速)")
        print("[update] 可选源: " + ", ".join(SOURCE_MAP.keys()))

    for source in sources:
        if source not in SOURCE_MAP:
            print(f"[update] 未知数据源: {source}")
            continue

        try:
            fn = SOURCE_MAP[source]
            # 市场级指标不需要 instruments
            if source == "tushare_north":
                fn(start_date, end_date, qlib_dir)
            else:
                fn(instruments, start_date, end_date, qlib_dir)
        except Exception as e:
            print(f"[update] {source} 更新失败: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n[update] 全部完成!")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="增量更新所有数据源")
    parser.add_argument("--source", type=str, nargs="+", default=None,
                        help=f"数据源 ({', '.join(SOURCE_MAP.keys())})")
    parser.add_argument("--start", type=str, default=DEFAULT_START)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--qlib_dir", type=str, default=DEFAULT_QLIB_DIR)
    args = parser.parse_args()

    update_all(args.start, args.end, args.qlib_dir, args.source)


if __name__ == "__main__":
    main()
