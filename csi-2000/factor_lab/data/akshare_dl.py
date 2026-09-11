"""AKShare 数据下载器 — 个股资金流向 / 融资融券

AKShare 免费但有限速:
- 连续请求 ~90 只后会被限速
- 策略: 80只/批，批间120s，增量更新

依赖: pip install akshare
"""
import pandas as pd

from .base_downloader import BaseDownloader


def _instrument_to_akshare(instrument: str) -> str:
    """Qlib instrument → AKShare code: SH600519 → 600519"""
    return instrument[2:]


class AKShareFundFlowDownloader(BaseDownloader):
    """AKShare 个股资金流向下载器"""

    source_name = "akshare_fund_flow"

    def __init__(self, **kwargs):
        kwargs.setdefault("batch_size", 80)
        kwargs.setdefault("batch_sleep", 120)
        super().__init__(**kwargs)

    def _fetch_one(self, code: str, start_date: str, end_date: str) -> pd.DataFrame | None:
        """下载单只股票的资金流向数据"""
        import akshare as ak

        stock_code = _instrument_to_akshare(code) if len(code) > 6 else code
        try:
            df = ak.stock_individual_fund_flow(stock=stock_code, market="sh" if code[:2].upper() == "SH" else "sz")
            if df is not None and len(df) > 0:
                # 标准化列名
                df = df.rename(columns={
                    "日期": "date",
                    "主力净流入-净额": "main_net_inflow",
                    "超大单净流入-净额": "super_large_net",
                    "大单净流入-净额": "large_net",
                    "中单净流入-净额": "medium_net",
                    "小单净流入-净额": "small_net",
                })
                if "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"])
                    # 过滤日期范围
                    mask = (df["date"] >= pd.Timestamp(start_date)) & (df["date"] <= pd.Timestamp(end_date))
                    df = df[mask]

                    # 只保留数值列
                    num_cols = ["main_net_inflow", "super_large_net", "large_net", "medium_net", "small_net"]
                    available_cols = ["date"] + [c for c in num_cols if c in df.columns]
                    return df[available_cols] if len(df) > 0 else None
        except Exception as e:
            # AKShare 可能返回格式不一致
            if "限速" not in str(e) and "频率" not in str(e):
                pass  # 忽略非限速错误
            raise
        return None


class AKShareMarginDownloader:
    """AKShare 融资融券下载器 (按日期下载全市场)

    API 变更: stock_margin_detail 已拆分为:
    - stock_margin_detail_sse(date) → 上交所
    - stock_margin_detail_szse(date) → 深交所
    按日期批量返回，无需逐只下载。
    """

    source_name = "akshare_margin"

    def __init__(self, cache_dir: str = None):
        from pathlib import Path
        if cache_dir is None:
            cache_dir = str(Path(__file__).parent.parent / "results" / ".cache" / self.source_name)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _akshare_to_instrument(self, code: str, exchange: str) -> str:
        """股票代码 → Qlib instrument: 600519 + sse → SH600519"""
        prefix = "SH" if exchange == "sse" else "SZ"
        return f"{prefix}{code}"

    def download_by_dates(self, trade_dates: list[str],
                          instruments: set[str] | None = None) -> pd.DataFrame:
        """按交易日下载融资融券数据

        Args:
            trade_dates: 交易日列表 (YYYYMMDD 格式)
            instruments: 可选，过滤的 Qlib 股票代码集合

        Returns:
            合并的 DataFrame，含 date, instrument, margin_balance, short_balance
        """
        import akshare as ak
        import time

        all_frames = []
        total = len(trade_dates)

        for i, date_str in enumerate(trade_dates):
            try:
                # 上交所
                df_sse = ak.stock_margin_detail_sse(date=date_str)
                if df_sse is not None and len(df_sse) > 0:
                    df_sse = df_sse.rename(columns={
                        "信用交易日期": "trade_date",
                        "标的证券代码": "code",
                        "融资余额": "margin_balance",
                    })
                    # SSE 没有融券余额列，用融券余量（股数）近似
                    if "融券余量" in df_sse.columns:
                        df_sse["short_balance"] = pd.to_numeric(df_sse["融券余量"], errors="coerce")
                    else:
                        df_sse["short_balance"] = 0.0
                    df_sse["margin_balance"] = pd.to_numeric(df_sse["margin_balance"], errors="coerce")
                    df_sse["instrument"] = df_sse["code"].apply(lambda x: f"SH{x}")
                    df_sse["date"] = pd.to_datetime(date_str, format="%Y%m%d")
                    all_frames.append(df_sse[["date", "instrument", "margin_balance", "short_balance"]])
            except Exception as e:
                if "404" not in str(e) and "非交易日" not in str(e):
                    print(f"  [margin] SSE {date_str}: {e}")

            try:
                # 深交所
                df_szse = ak.stock_margin_detail_szse(date=date_str)
                if df_szse is not None and len(df_szse) > 0:
                    df_szse = df_szse.rename(columns={
                        "证券代码": "code",
                        "融资余额": "margin_balance",
                        "融券余额": "short_balance",
                    })
                    df_szse["margin_balance"] = pd.to_numeric(df_szse["margin_balance"], errors="coerce")
                    df_szse["short_balance"] = pd.to_numeric(df_szse["short_balance"], errors="coerce")
                    df_szse["instrument"] = df_szse["code"].apply(lambda x: f"SZ{x}")
                    df_szse["date"] = pd.to_datetime(date_str, format="%Y%m%d")
                    all_frames.append(df_szse[["date", "instrument", "margin_balance", "short_balance"]])
            except Exception as e:
                if "404" not in str(e) and "非交易日" not in str(e):
                    print(f"  [margin] SZSE {date_str}: {e}")

            # 限速: 每 30 次暂停 1 秒
            if (i + 1) % 30 == 0:
                time.sleep(1)

            if (i + 1) % 100 == 0 or (i + 1) == total:
                print(f"  [margin] 下载进度 [{i+1}/{total}]")

        if not all_frames:
            return pd.DataFrame()

        result = pd.concat(all_frames, ignore_index=True)

        if instruments:
            result = result[result["instrument"].isin(instruments)]

        print(f"  [margin] 总计: {len(result)} 行, {result['instrument'].nunique()} 只股票")
        return result
