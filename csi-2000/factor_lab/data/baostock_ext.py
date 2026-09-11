"""BaoStock 扩展下载器 — 下载 amount/turn/pctChg 及季报数据

BaoStock 使用 TCP 长连接，无限速问题。
"""
import baostock as bs
import pandas as pd

from .base_downloader import BaseDownloader


class BaoStockExtDownloader(BaseDownloader):
    """BaoStock 扩展字段下载器"""

    source_name = "baostock_ext"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._logged_in = False

    def login(self):
        if not self._logged_in:
            lg = bs.login()
            if lg.error_code != '0':
                raise RuntimeError(f"BaoStock 登录失败: {lg.error_msg}")
            self._logged_in = True

    def logout(self):
        if self._logged_in:
            bs.logout()
            self._logged_in = False

    def _bao_code(self, instrument: str) -> str:
        """Qlib instrument → BaoStock code: SH600519 → sh.600519"""
        s = instrument.lower()
        return f"{s[:2]}.{s[2:]}"

    def _fetch_one(self, code: str, start_date: str, end_date: str) -> pd.DataFrame | None:
        """下载单只股票的扩展字段 (amount, turn, pctChg)"""
        self.login()
        bao_code = self._bao_code(code) if "." not in code else code

        rs = bs.query_history_k_data_plus(
            bao_code,
            "date,amount,turn,pctChg",
            start_date=start_date,
            end_date=end_date,
            frequency="d",
            adjustflag="2",
        )
        if rs.error_code != '0':
            return None

        rows = []
        while rs.next():
            rows.append(rs.get_row_data())

        if not rows:
            return None

        df = pd.DataFrame(rows, columns=["date", "amount", "turn", "pctChg"])
        df["date"] = pd.to_datetime(df["date"])
        for col in ["amount", "turn", "pctChg"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        return df[df["amount"].notna() | df["turn"].notna()]

    def download_and_inject(self, instruments: list[str], start_date: str, end_date: str,
                            qlib_dir: str = "~/.qlib/qlib_data/cn_data_bs"):
        """下载并注入扩展字段到 Qlib bin"""
        from .qlib_injector import inject_field

        self.login()
        results = self.download(instruments, start_date, end_date)

        count = 0
        for instrument, df in results.items():
            for field in ["amount", "turn", "pctChg"]:
                if field in df.columns:
                    valid = df[df[field].notna()]
                    if len(valid) > 0:
                        inject_field(
                            instrument=instrument.upper(),
                            field_name=field.lower(),
                            dates=valid["date"].dt.strftime("%Y-%m-%d").tolist(),
                            values=valid[field].tolist(),
                            qlib_dir=qlib_dir,
                        )
            count += 1

        self.logout()
        print(f"[baostock_ext] 注入完成: {count} 只股票")
        return results


class BaoStockFinanceDownloader(BaseDownloader):
    """BaoStock 季报数据下载器"""

    source_name = "baostock_finance"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._logged_in = False

    def login(self):
        if not self._logged_in:
            bs.login()
            self._logged_in = True

    def logout(self):
        if self._logged_in:
            bs.logout()
            self._logged_in = False

    def _bao_code(self, instrument: str) -> str:
        s = instrument.lower()
        return f"{s[:2]}.{s[2:]}"

    def _fetch_one(self, code: str, start_date: str, end_date: str) -> pd.DataFrame | None:
        """下载单只股票的季报利润表数据"""
        self.login()
        bao_code = self._bao_code(code) if "." not in code else code

        # 利润表
        rs = bs.query_profit_data(code=bao_code, year=int(start_date[:4]),
                                   quarter=4)
        rows = []
        while rs.next():
            rows.append(rs.get_row_data())

        if not rows:
            return None

        df = pd.DataFrame(rows, columns=rs.fields)
        return df
