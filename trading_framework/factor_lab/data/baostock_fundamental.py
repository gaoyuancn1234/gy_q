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
import time

import baostock as bs
import numpy as np
import pandas as pd
from pathlib import Path


DEFAULT_QLIB_DIR = "~/.qlib/qlib_data/cn_data_bs"


def _bao_code(instrument: str) -> str:
    """Qlib instrument → BaoStock code: SH600519 → sh.600519"""
    s = instrument.lower()
    return f"{s[:2]}.{s[2:]}"


# BaoStock 查询的失败判据与重连
# 2026-09-05: 原实现完全不看 error_code —— session 一旦失效，每次查询都返回
# 错误结果，而 `while rs.next()` 立刻为 False、profit_rows 为空，循环把它当成
# "这个季度没有财报" 直接 continue。于是进程可以带着一条已死的连接跑满 CPU
# 两个半小时、一行输出都没有、一个字节都没落盘，任务状态还显示 Running。
# 这正是 CLAUDE.md 里"沉默失败"那一节的模式: 看起来在工作，实则什么都没做。
_MAX_RETRY = 3


def _bs_query(fn, what: str, timeout: float = 30):
    """执行一次 baostock 查询; 检查 error_code，失败则重登后重试

    连续失败 _MAX_RETRY+1 次直接抛异常 —— 宁可整个任务失败退出，
    也不要静默地把"查询失败"当成"没有数据"跑完全程。
    """
    import baostock as bs
    from net_guard import run_with_timeout, NetTimeout

    for attempt in range(_MAX_RETRY + 1):
        rs = None
        try:
            rs = run_with_timeout(fn, timeout, what)
        except NetTimeout:
            pass
        if rs is not None and getattr(rs, 'error_code', '0') == '0':
            return rs

        code = getattr(rs, 'error_code', '?') if rs is not None else 'timeout'
        msg = getattr(rs, 'error_msg', '') if rs is not None else f'超过 {timeout}s'
        if attempt == _MAX_RETRY:
            raise RuntimeError(
                f"{what} 连续 {_MAX_RETRY + 1} 次失败 (error_code={code} {msg})")
        print(f"  [baostock] {what} 失败 (error_code={code} {msg})，"
              f"第 {attempt + 1}/{_MAX_RETRY} 次重登...", flush=True)
        try:
            run_with_timeout(bs.logout, 10, 'baostock logout')
        except Exception:
            pass
        try:
            run_with_timeout(bs.login, 30, 'baostock login')
        except Exception as e:
            print(f"  [baostock] 重登失败: {e}", flush=True)
    raise RuntimeError(f"{what}: 不可达")


def download_quarterly_data(instruments: list[str],
                            start_year: int = 2018,
                            end_year: int = 2025) -> pd.DataFrame:
    """下载季度财务数据

    Returns:
        DataFrame with columns: [instrument, pubDate, statDate,
                                  eps_ttm, roe_avg, net_profit, total_share, yoy_ni]

    ⚠ 耗时: 串行 N只 × 年数 × 4季 × 2接口 次调用。549 只 × 2018~2026 约
    4 万次，实测 1.5~2 小时，且 CPU 占用 ~96% (BaoStock SDK 每次调用都要
    建帧+逐行解析，不是网络等待)。

    首次注入必须全量跑一次；之后应改为增量 —— 季报一季度才更新一次，
    每次全量重下 4 万次纯属浪费。增量做法: 读已有 parquet 缓存，只补
    max(pubDate) 之后的季度。(2026-09-05 记录，尚未实现)
    """
    from net_guard import run_with_timeout
    _lg = run_with_timeout(bs.login, 30, 'baostock login')
    if getattr(_lg, 'error_code', '0') != '0':
        raise RuntimeError(f"baostock 登录失败: {_lg.error_code} {_lg.error_msg}")
    all_rows = []
    total = len(instruments)
    n_empty_streak = 0        # 连续多少只股票一行数据都没取到

    _t0 = time.time()
    for i, inst in enumerate(instruments):
        bao_code = _bao_code(inst)
        _rows_before = len(all_rows)

        for year in range(start_year, end_year + 1):
            for quarter in range(1, 5):
                # profit_data: epsTTM, roeAvg, netProfit, totalShare
                rs_p = _bs_query(
                    lambda: bs.query_profit_data(code=bao_code, year=year,
                                                 quarter=quarter),
                    f'query_profit_data {bao_code} {year}Q{quarter}')
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
                rs_g = _bs_query(
                    lambda: bs.query_growth_data(code=bao_code, year=year,
                                                 quarter=quarter),
                    f'query_growth_data {bao_code} {year}Q{quarter}')
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
            # flush 必需: 重定向到日志文件时 stdout 是块缓冲的，不 flush 就看不到
            # 进度 —— 2026-09-05 实测本模块跑 11 分钟日志一个字都没有，
            # 与 data_setup_sina 此前"卡在第 1 只上 8.5 小时无输出"是同一个毛病。
            # 带上当前股票代码，卡住时最后一行就指明卡在哪只。
            el = time.time() - _t0
            eta = el / (i + 1) * (total - i - 1)
            print(f"  [baostock] 下载季报 [{i+1}/{total}] {inst} "
                  f"累计 {len(all_rows)} 行 已用 {el/60:.1f}min "
                  f"ETA {eta/60:.1f}min", flush=True)

        # 连续 20 只股票一行都没有 —— 正常情况下不可能(CSI300 成分股都有财报)，
        # 几乎必然是数据源侧的问题。宁可现在失败，也不要跑完两小时交出空表。
        if len(all_rows) == _rows_before:
            n_empty_streak += 1
            if n_empty_streak >= 20:
                raise RuntimeError(
                    f"连续 {n_empty_streak} 只股票未取到任何财报数据 "
                    f"(已处理 {i+1}/{total})，判定数据源异常，中止")
        else:
            n_empty_streak = 0

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
    # end_year 跟随当前年份: 原先写死 2025，在 2026 年跑会漏掉最近一年半的
    # 财报 —— 而基本面因子恰恰靠最新一期数据，漏掉等于因子失效。
    from datetime import date
    _end_year = date.today().year
    quarterly_df = download_quarterly_data(instruments,
                                           start_year=2018, end_year=_end_year)

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
