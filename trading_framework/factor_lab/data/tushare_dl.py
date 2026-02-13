"""Tushare 数据下载器 — daily_basic / 北向资金 / 财报

需要 Tushare Pro token (免费120积分):
- daily_basic(trade_date=date) → 一次返回全市场300只
- moneyflow_hsgt() → 北向资金 (市场级)
- income() / balancesheet() → 季报

Token 从环境变量 TUSHARE_TOKEN 或 .env 读取
"""
import os
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd

from .base_downloader import BaseDownloader


def _get_token() -> str:
    """获取 Tushare token"""
    token = os.environ.get("TUSHARE_TOKEN", "")
    if not token:
        env_file = Path(__file__).parent.parent.parent / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith("TUSHARE_TOKEN="):
                    token = line.split("=", 1)[1].strip().strip("'\"")
                    break
    return token


def _init_tushare():
    """初始化 Tushare Pro"""
    import tushare as ts
    token = _get_token()
    if not token:
        raise ValueError(
            "TUSHARE_TOKEN 未设置。请注册 Tushare Pro (https://tushare.pro) "
            "并设置环境变量或在 .env 中添加 TUSHARE_TOKEN=xxx"
        )
    ts.set_token(token)
    return ts.pro_api()


def _instrument_to_ts(instrument: str) -> str:
    """Qlib instrument → Tushare code: SH600519 → 600519.SH"""
    prefix = instrument[:2].upper()
    num = instrument[2:]
    return f"{num}.{prefix}"


def _ts_to_instrument(ts_code: str) -> str:
    """Tushare code → Qlib instrument: 600519.SH → SH600519"""
    parts = ts_code.split(".")
    return f"{parts[1]}{parts[0]}"


class TuShareDailyBasicDownloader(BaseDownloader):
    """Tushare daily_basic 下载器 (PE/PB/市值等)

    按日期批量下载，一次返回全市场所有股票，不需要逐只请求。
    """

    source_name = "tushare_daily_basic"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._pro = None

    def _ensure_api(self):
        if self._pro is None:
            self._pro = _init_tushare()

    def _fetch_one(self, code: str, start_date: str, end_date: str) -> pd.DataFrame | None:
        """按日期下载 daily_basic（code 参数在这里是日期）"""
        self._ensure_api()
        trade_date = code.replace("-", "")
        try:
            df = self._pro.daily_basic(
                trade_date=trade_date,
                fields="ts_code,trade_date,pe_ttm,pb,ps_ttm,total_mv,circ_mv,turnover_rate"
            )
            if df is not None and len(df) > 0:
                df["date"] = pd.to_datetime(df["trade_date"])
                df["instrument"] = df["ts_code"].apply(_ts_to_instrument)
                return df
        except Exception as e:
            print(f"  [tushare] daily_basic {trade_date} 失败: {e}")
        return None

    def _get_trade_dates(self, start_date: str, end_date: str) -> list[str]:
        """获取交易日列表 (优先用 Qlib 日历，回退 Tushare)"""
        import os
        from pathlib import Path

        cal_file = Path(os.path.expanduser(
            "~/.qlib/qlib_data/cn_data_bs/calendars/day.txt"))
        if cal_file.exists():
            dates = []
            for line in cal_file.read_text().splitlines():
                d = line.strip()
                if d and start_date <= d <= end_date:
                    dates.append(d.replace("-", ""))
            if dates:
                return sorted(dates)

        # 回退: Tushare 交易日历
        self._ensure_api()
        df = self._pro.trade_cal(
            start_date=start_date.replace("-", ""),
            end_date=end_date.replace("-", ""),
            is_open='1')
        return sorted(df['cal_date'].tolist()) if df is not None else []

    def download_date_range(self, start_date: str, end_date: str,
                            instruments: set[str] | None = None) -> pd.DataFrame:
        """按日期范围下载 daily_basic

        Args:
            start_date: 起始日期
            end_date: 结束日期
            instruments: 可选，过滤的股票集合

        Returns:
            合并后的 DataFrame
        """
        import time as _time

        self._ensure_api()
        all_frames = []

        trade_dates = self._get_trade_dates(start_date, end_date)
        total = len(trade_dates)
        print(f"  [tushare] 交易日: {total} 天")

        call_count = 0
        batch_start = _time.time()

        for i, date_str in enumerate(trade_dates):
            try:
                df = self._pro.daily_basic(
                    trade_date=date_str,
                    fields="ts_code,trade_date,pe_ttm,pb,ps_ttm,total_mv,circ_mv,turnover_rate"
                )
                if df is not None and len(df) > 0:
                    df["date"] = pd.to_datetime(df["trade_date"])
                    df["instrument"] = df["ts_code"].apply(_ts_to_instrument)
                    if instruments:
                        df = df[df["instrument"].isin(instruments)]
                    all_frames.append(df)
                call_count += 1
            except Exception as e:
                err_msg = str(e)
                if "最多访问" in err_msg:
                    # 限速: 等待到下一分钟
                    elapsed = _time.time() - batch_start
                    wait = max(62 - elapsed, 5)
                    print(f"  [tushare] 限速, 等待 {wait:.0f}s "
                          f"({i}/{total})")
                    _time.sleep(wait)
                    batch_start = _time.time()
                    call_count = 0
                    # 重试当前日期
                    try:
                        df = self._pro.daily_basic(
                            trade_date=date_str,
                            fields="ts_code,trade_date,pe_ttm,pb,ps_ttm,total_mv,circ_mv,turnover_rate"
                        )
                        if df is not None and len(df) > 0:
                            df["date"] = pd.to_datetime(df["trade_date"])
                            df["instrument"] = df["ts_code"].apply(
                                _ts_to_instrument)
                            if instruments:
                                df = df[df["instrument"].isin(instruments)]
                            all_frames.append(df)
                        call_count += 1
                    except Exception as e2:
                        print(f"  [tushare] {date_str} 重试失败: {e2}")
                else:
                    print(f"  [tushare] {date_str}: {e}")

            # 限速: 180次/分钟, 每 170 次暂停
            if call_count >= 170:
                elapsed = _time.time() - batch_start
                if elapsed < 62:
                    wait = 62 - elapsed
                    print(f"  [tushare] 节流 {wait:.0f}s ({i}/{total})")
                    _time.sleep(wait)
                batch_start = _time.time()
                call_count = 0

            if (i + 1) % 200 == 0:
                print(f"  [tushare] 进度 [{i+1}/{total}] "
                      f"已获取 {len(all_frames)} 天")

        if not all_frames:
            return pd.DataFrame()

        result = pd.concat(all_frames, ignore_index=True)
        print(f"  [tushare] daily_basic: {len(result)} 行, "
              f"{result['instrument'].nunique()} 只股票")
        return result


class TuShareNorthMoneyDownloader(BaseDownloader):
    """Tushare 北向资金下载器 (市场级指标)"""

    source_name = "tushare_north_money"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._pro = None

    def _ensure_api(self):
        if self._pro is None:
            self._pro = _init_tushare()

    def _fetch_one(self, code: str, start_date: str, end_date: str) -> pd.DataFrame | None:
        self._ensure_api()
        try:
            df = self._pro.moneyflow_hsgt(
                start_date=start_date.replace("-", ""),
                end_date=end_date.replace("-", ""),
            )
            if df is not None and len(df) > 0:
                df["date"] = pd.to_datetime(df["trade_date"])
                # 计算北向资金总额: 沪股通 + 深股通
                df["north_money"] = df["north_money"]  # 已是北向总额
                return df[["date", "north_money"]].sort_values("date")
        except Exception as e:
            print(f"  [tushare] 北向资金下载失败: {e}")
        return None

    def download_north_money(self, start_date: str, end_date: str) -> pd.DataFrame:
        """下载北向资金数据"""
        df = self._fetch_one("market", start_date, end_date)
        if df is not None:
            self._save_cache("north_money", df)
        return df if df is not None else pd.DataFrame()
