#!/usr/bin/env python3
"""BaoStock 季度财务数据 → Qlib 每日基本面字段

核心逻辑:
1. 下载 profit_data (epsTTM/netProfit/roeAvg/totalShare) + growth_data (YOYNI)
2. 按 pubDate (公告日) 做 point-in-time 映射, 避免未来信息泄露
3. 季报数据 forward-fill 到每个交易日
4. 与每日收盘价结合计算 pe_ttm = close / epsTTM
5. 注入 Qlib bin 格式

生成字段:
- eps_ttm: 滚动每股收益 (来自季报, 按公告日前填充)
- roe_avg: 平均净资产收益率
- net_profit: 净利润 (元)
- total_share: 总股本 (股)
- yoy_ni: 净利润同比增长率
- pe_ttm: 滚动市盈率 (= close / eps_ttm, 需要已有 close 数据)
- total_mv: 总市值 (= close * total_share)
"""
import baostock as bs
import numpy as np
import pandas as pd
from pathlib import Path


DEFAULT_QLIB_DIR = "~/.qlib/qlib_data/cn_data_bs"


def _bao_code(instrument: str) -> str:
    """Qlib instrument → BaoStock code: SH600519 → sh.600519"""
    s = instrument.lower()
    return f"{s[:2]}.{s[2:]}"


def download_quarterly_data(instruments: list[str],
                            start_year: int = 2018,
                            end_year: int = 2025) -> pd.DataFrame:
    """下载季度财务数据

    Returns:
        DataFrame with columns: [instrument, pubDate, statDate,
                                  eps_ttm, roe_avg, net_profit, total_share, yoy_ni]
    """
    bs.login()
    all_rows = []
    total = len(instruments)

    for i, inst in enumerate(instruments):
        bao_code = _bao_code(inst)

        for year in range(start_year, end_year + 1):
            for quarter in range(1, 5):
                # profit_data: epsTTM, roeAvg, netProfit, totalShare
                rs_p = bs.query_profit_data(code=bao_code, year=year, quarter=quarter)
                profit_rows = []
                while rs_p.next():
                    profit_rows.append(rs_p.get_row_data())

                if not profit_rows:
                    continue

                row_p = profit_rows[0]
                pub_date = row_p[1]   # pubDate
                stat_date = row_p[2]  # statDate
                eps_ttm = row_p[7]    # epsTTM
                roe_avg = row_p[3]    # roeAvg
                net_profit = row_p[6] # netProfit
                total_share = row_p[9] # totalShare

                # growth_data: YOYNI
                rs_g = bs.query_growth_data(code=bao_code, year=year, quarter=quarter)
                growth_rows = []
                while rs_g.next():
                    growth_rows.append(rs_g.get_row_data())

                yoy_ni = growth_rows[0][3] if growth_rows else ""  # YOYNI

                all_rows.append({
                    "instrument": inst,
                    "pubDate": pub_date,
                    "statDate": stat_date,
                    "eps_ttm": _safe_float(eps_ttm),
                    "roe_avg": _safe_float(roe_avg),
                    "net_profit": _safe_float(net_profit),
                    "total_share": _safe_float(total_share),
                    "yoy_ni": _safe_float(yoy_ni),
                })

        if (i + 1) % 20 == 0 or (i + 1) == total:
            print(f"  [baostock] 下载季报进度 [{i+1}/{total}]")

    bs.logout()

    df = pd.DataFrame(all_rows)
    df["pubDate"] = pd.to_datetime(df["pubDate"])
    df["statDate"] = pd.to_datetime(df["statDate"])
    print(f"  [baostock] 季报数据: {len(df)} 行, {df['instrument'].nunique()} 只股票")
    return df


def _safe_float(val) -> float:
    """安全转 float，空字符串返回 NaN"""
    if val is None or val == "":
        return float("nan")
    try:
        return float(val)
    except (ValueError, TypeError):
        return float("nan")


def expand_to_daily(quarterly_df: pd.DataFrame,
                    calendar_dates: list[str]) -> pd.DataFrame:
    """将季报数据按公告日(pubDate) forward-fill 到每日

    使用 pubDate 而非 statDate，确保没有未来信息泄露。
    每只股票在其 pubDate 当天更新数据，之后每天沿用直到下一次公告。

    Args:
        quarterly_df: 季报数据 (含 instrument, pubDate, 各字段)
        calendar_dates: Qlib 交易日历日期列表

    Returns:
        DataFrame with (date, instrument) 索引, 每日财务字段
    """
    fields = ["eps_ttm", "roe_avg", "net_profit", "total_share", "yoy_ni"]
    cal_dates = pd.to_datetime(calendar_dates)
    all_frames = []

    instruments = quarterly_df["instrument"].unique()
    for inst in instruments:
        inst_q = quarterly_df[quarterly_df["instrument"] == inst].sort_values("pubDate")
        if inst_q.empty:
            continue

        # 为每个公告日创建一行
        inst_q = inst_q.drop_duplicates(subset="pubDate", keep="last")
        inst_q = inst_q.set_index("pubDate")[fields]

        # Reindex 到日历日期，forward fill
        inst_daily = inst_q.reindex(cal_dates, method="ffill")
        inst_daily = inst_daily.dropna(how="all")

        if inst_daily.empty:
            continue

        inst_daily["instrument"] = inst
        inst_daily.index.name = "date"
        all_frames.append(inst_daily.reset_index())

    if not all_frames:
        return pd.DataFrame()

    result = pd.concat(all_frames, ignore_index=True)
    return result


def compute_pe_and_mv(daily_fundamental: pd.DataFrame,
                      qlib_dir: str = DEFAULT_QLIB_DIR) -> pd.DataFrame:
    """结合每日收盘价计算 pe_ttm 和 total_mv

    pe_ttm = close / eps_ttm
    total_mv = close * total_share
    """
    # 读取收盘价
    from .qlib_injector import _load_calendar

    cal_index = _load_calendar(qlib_dir)
    idx_to_date = {v: k for k, v in cal_index.items()}
    feat_base = Path(qlib_dir).expanduser() / "features"

    instruments = daily_fundamental["instrument"].unique()
    pe_rows = []
    mv_rows = []

    for inst in instruments:
        inst_data = daily_fundamental[daily_fundamental["instrument"] == inst]
        close_path = feat_base / inst.lower() / "close.day.bin"
        if not close_path.exists():
            continue

        # 读取收盘价 bin
        arr = np.fromfile(close_path, dtype=np.float32)
        start_idx = int(arr[0])
        close_data = arr[2:]

        for _, row in inst_data.iterrows():
            date_str = pd.Timestamp(row["date"]).strftime("%Y-%m-%d")
            if date_str not in cal_index:
                continue

            cal_idx = cal_index[date_str]
            data_pos = cal_idx - start_idx
            if data_pos < 0 or data_pos >= len(close_data):
                continue

            close_price = close_data[data_pos]
            if np.isnan(close_price):
                continue

            eps = row["eps_ttm"]
            total_share = row["total_share"]

            if not np.isnan(eps) and abs(eps) > 1e-8:
                pe_rows.append({"date": date_str, "instrument": inst,
                                "pe_ttm": close_price / eps})

            if not np.isnan(total_share) and total_share > 0:
                mv_rows.append({"date": date_str, "instrument": inst,
                                "total_mv": close_price * total_share})

    pe_df = pd.DataFrame(pe_rows)
    mv_df = pd.DataFrame(mv_rows)

    # 合并回 daily_fundamental
    result = daily_fundamental.copy()
    if not pe_df.empty:
        pe_df = pe_df.set_index(["date", "instrument"])["pe_ttm"]
        result = result.set_index(["date", "instrument"])
        result["pe_ttm"] = pe_df
        result = result.reset_index()

    if not mv_df.empty:
        mv_df = mv_df.set_index(["date", "instrument"])["total_mv"]
        result = result.set_index(["date", "instrument"])
        result["total_mv"] = mv_df
        result = result.reset_index()

    print(f"  [计算] pe_ttm: {len(pe_rows)} 行, total_mv: {len(mv_rows)} 行")
    return result


def inject_fundamental_fields(daily_df: pd.DataFrame,
                              qlib_dir: str = DEFAULT_QLIB_DIR):
    """将每日基本面数据注入 Qlib bin

    注入字段: eps_ttm, roe_avg, net_profit, total_share, yoy_ni, pe_ttm, total_mv
    """
    from .qlib_injector import inject_dataframe

    fields_to_inject = [c for c in ["eps_ttm", "roe_avg", "net_profit",
                                     "total_share", "yoy_ni", "pe_ttm", "total_mv"]
                        if c in daily_df.columns]

    # 确保 instrument 列是大写
    daily_df = daily_df.copy()
    daily_df["instrument"] = daily_df["instrument"].str.upper()

    print(f"  [注入] 字段: {fields_to_inject}")
    inject_dataframe(daily_df, field_columns=fields_to_inject,
                     instrument_col="instrument", date_col="date",
                     qlib_dir=qlib_dir)


def main():
    """完整流程: 下载 → 展开 → 计算 → 注入"""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    import multiprocessing
    try:
        multiprocessing.set_start_method('fork', force=True)
    except (ValueError, RuntimeError):
        pass  # Windows 无 fork，使用默认 spawn

    qlib_dir = DEFAULT_QLIB_DIR

    # 1. 获取 CSI300 成分股
    print("=" * 60)
    print("Phase 2: BaoStock 基本面数据 → Qlib")
    print("=" * 60)

    import qlib
    from qlib.constant import REG_CN
    qlib.init(provider_uri=qlib_dir, region=REG_CN)
    from qlib.data import D

    inst = D.instruments("csi300")
    inst_list = D.list_instruments(instruments=inst, as_list=True)
    instruments = sorted(inst_list)
    print(f"CSI300 成分股: {len(instruments)} 只")

    # 2. 下载季报数据
    print("\n[Step 1] 下载季度财务数据...")
    quarterly_df = download_quarterly_data(instruments,
                                           start_year=2018, end_year=2025)

    # 缓存
    cache_dir = Path(__file__).parent.parent / "results" / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    quarterly_df.to_parquet(cache_dir / "quarterly_fundamental.parquet")
    print(f"  缓存至: {cache_dir / 'quarterly_fundamental.parquet'}")

    # 3. 展开到每日
    print("\n[Step 2] 按公告日 forward-fill 到每日...")
    from .qlib_injector import _load_calendar
    cal_index = _load_calendar(qlib_dir)
    calendar_dates = sorted(cal_index.keys())
    daily_df = expand_to_daily(quarterly_df, calendar_dates)
    print(f"  每日数据: {len(daily_df)} 行")

    # 4. 计算 PE/市值
    print("\n[Step 3] 计算 pe_ttm 和 total_mv...")
    daily_df = compute_pe_and_mv(daily_df, qlib_dir)

    # 5. 注入 Qlib
    print("\n[Step 4] 注入 Qlib bin...")
    inject_fundamental_fields(daily_df, qlib_dir)

    # 6. 验证
    print("\n[Step 5] 验证注入结果...")
    from .qlib_injector import verify_field
    for field in ["eps_ttm", "pe_ttm", "roe_avg", "total_mv"]:
        verify_field(field, instrument="sh600519", qlib_dir=qlib_dir)

    print("\n" + "=" * 60)
    print("Phase 2 数据准备完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
