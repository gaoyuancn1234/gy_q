"""
A股数据源
支持多种数据源: akshare, tushare, baostock等
"""

import pandas as pd
from datetime import date, datetime
from typing import List, Dict, Any, Optional
import logging

from .data_source import DataSource, StockInfo, StockData, KLineData, KLineType

logger = logging.getLogger(__name__)


class AStockDataSource(DataSource):
    """
    A股数据源
    默认使用akshare（免费）, 也支持tushare（需要积分）和baostock
    """

    def __init__(self, source: str = "akshare", token: str = ""):
        """
        初始化A股数据源

        Args:
            source: 数据源类型 ("akshare", "tushare", "baostock")
            token: API token（tushare需要）
        """
        super().__init__(f"AStock-{source}")
        self.source = source
        self.token = token
        self._api = None

    def connect(self) -> bool:
        """连接数据源"""
        try:
            if self.source == "akshare":
                import akshare as ak
                self._api = ak
                self._connected = True
                logger.info("已连接到 akshare 数据源")

            elif self.source == "tushare":
                import tushare as ts
                ts.set_token(self.token)
                self._api = ts.pro_api()
                self._connected = True
                logger.info("已连接到 tushare 数据源")

            elif self.source == "baostock":
                import baostock as bs
                lg = bs.login()
                if lg.error_code != '0':
                    logger.error(f"baostock登录失败: {lg.error_msg}")
                    return False
                self._api = bs
                self._connected = True
                logger.info("已连接到 baostock 数据源")

            else:
                logger.error(f"不支持的数据源: {self.source}")
                return False

            return True

        except ImportError as e:
            logger.error(f"请安装数据源库: {e}")
            return False
        except Exception as e:
            logger.error(f"连接数据源失败: {e}")
            return False

    def disconnect(self):
        """断开连接"""
        if self.source == "baostock" and self._connected:
            self._api.logout()
        self._connected = False
        self._api = None
        logger.info("已断开数据源连接")

    def get_stock_list(self) -> List[StockInfo]:
        """获取A股股票列表"""
        if not self._connected:
            raise ConnectionError("数据源未连接")

        stocks = []

        try:
            if self.source == "akshare":
                # 获取A股列表
                df = self._api.stock_zh_a_spot_em()
                for _, row in df.iterrows():
                    code = str(row['代码'])
                    market = 'SH' if code.startswith('6') else 'SZ'
                    stocks.append(StockInfo(
                        code=code,
                        name=row['名称'],
                        market=market,
                        market_cap=row.get('总市值', 0),
                        pe_ratio=row.get('市盈率-动态', 0),
                        pb_ratio=row.get('市净率', 0)
                    ))

            elif self.source == "tushare":
                df = self._api.stock_basic(
                    exchange='',
                    list_status='L',
                    fields='ts_code,symbol,name,area,industry,list_date'
                )
                for _, row in df.iterrows():
                    ts_code = row['ts_code']
                    market = 'SH' if ts_code.endswith('.SH') else 'SZ'
                    stocks.append(StockInfo(
                        code=row['symbol'],
                        name=row['name'],
                        market=market,
                        industry=row.get('industry', ''),
                        list_date=datetime.strptime(row['list_date'], '%Y%m%d').date() if row.get('list_date') else None
                    ))

            elif self.source == "baostock":
                rs = self._api.query_stock_basic()
                while rs.next():
                    row = rs.get_row_data()
                    code = row[0].split('.')[1]
                    market = 'SH' if row[0].startswith('sh') else 'SZ'
                    stocks.append(StockInfo(
                        code=code,
                        name=row[1],
                        market=market
                    ))

        except Exception as e:
            logger.error(f"获取股票列表失败: {e}")

        return stocks

    def get_realtime_quote(self, codes: List[str]) -> List[StockData]:
        """获取实时行情"""
        if not self._connected:
            raise ConnectionError("数据源未连接")

        quotes = []

        try:
            if self.source == "akshare":
                df = self._api.stock_zh_a_spot_em()
                df['代码'] = df['代码'].astype(str)
                df = df[df['代码'].isin(codes)]

                for _, row in df.iterrows():
                    quotes.append(StockData(
                        code=str(row['代码']),
                        name=row['名称'],
                        price=float(row['最新价']) if pd.notna(row['最新价']) else 0,
                        open=float(row['今开']) if pd.notna(row['今开']) else 0,
                        high=float(row['最高']) if pd.notna(row['最高']) else 0,
                        low=float(row['最低']) if pd.notna(row['最低']) else 0,
                        close=float(row['最新价']) if pd.notna(row['最新价']) else 0,
                        pre_close=float(row['昨收']) if pd.notna(row['昨收']) else 0,
                        volume=float(row['成交量']) if pd.notna(row['成交量']) else 0,
                        amount=float(row['成交额']) if pd.notna(row['成交额']) else 0,
                        timestamp=datetime.now()
                    ))

            elif self.source == "tushare":
                # tushare实时行情需要高级权限，这里使用日线最新数据
                for code in codes:
                    ts_code = f"{code}.SH" if code.startswith('6') else f"{code}.SZ"
                    df = self._api.daily(ts_code=ts_code, limit=1)
                    if not df.empty:
                        row = df.iloc[0]
                        quotes.append(StockData(
                            code=code,
                            name="",
                            price=row['close'],
                            open=row['open'],
                            high=row['high'],
                            low=row['low'],
                            close=row['close'],
                            pre_close=row['pre_close'],
                            volume=row['vol'],
                            amount=row['amount'] * 1000,
                            timestamp=datetime.now()
                        ))

        except Exception as e:
            logger.error(f"获取实时行情失败: {e}")

        return quotes

    def get_kline(
        self,
        code: str,
        kline_type: KLineType,
        start_date: date,
        end_date: date,
        adjust: int = 1
    ) -> pd.DataFrame:
        """获取K线数据"""
        if not self._connected:
            raise ConnectionError("数据源未连接")

        try:
            if self.source == "akshare":
                # 复权类型映射
                adjust_map = {0: '', 1: 'qfq', 2: 'hfq'}
                adjust_type = adjust_map.get(adjust, '')

                # K线类型映射
                period_map = {
                    KLineType.DAILY: 'daily',
                    KLineType.WEEKLY: 'weekly',
                    KLineType.MONTHLY: 'monthly',
                    KLineType.MINUTE_1: '1',
                    KLineType.MINUTE_5: '5',
                    KLineType.MINUTE_15: '15',
                    KLineType.MINUTE_30: '30',
                    KLineType.MINUTE_60: '60',
                }
                period = period_map.get(kline_type, 'daily')

                if kline_type in [KLineType.DAILY, KLineType.WEEKLY, KLineType.MONTHLY]:
                    df = self._api.stock_zh_a_hist(
                        symbol=code,
                        period=period,
                        start_date=start_date.strftime('%Y%m%d'),
                        end_date=end_date.strftime('%Y%m%d'),
                        adjust=adjust_type
                    )
                    # 重命名列
                    df = df.rename(columns={
                        '日期': 'date',
                        '开盘': 'open',
                        '收盘': 'close',
                        '最高': 'high',
                        '最低': 'low',
                        '成交量': 'volume',
                        '成交额': 'amount',
                        '换手率': 'turnover'
                    })
                else:
                    # 分钟K线
                    df = self._api.stock_zh_a_hist_min_em(
                        symbol=code,
                        period=period,
                        adjust=adjust_type
                    )
                    df = df.rename(columns={
                        '时间': 'date',
                        '开盘': 'open',
                        '收盘': 'close',
                        '最高': 'high',
                        '最低': 'low',
                        '成交量': 'volume',
                        '成交额': 'amount'
                    })

            elif self.source == "tushare":
                ts_code = f"{code}.SH" if code.startswith('6') else f"{code}.SZ"
                df = self._api.daily(
                    ts_code=ts_code,
                    start_date=start_date.strftime('%Y%m%d'),
                    end_date=end_date.strftime('%Y%m%d')
                )
                df = df.rename(columns={
                    'trade_date': 'date',
                    'vol': 'volume'
                })
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date')

            elif self.source == "baostock":
                bs_code = f"sh.{code}" if code.startswith('6') else f"sz.{code}"
                adjust_map = {0: '3', 1: '2', 2: '1'}

                rs = self._api.query_history_k_data_plus(
                    bs_code,
                    "date,open,high,low,close,volume,amount,turn",
                    start_date=start_date.strftime('%Y-%m-%d'),
                    end_date=end_date.strftime('%Y-%m-%d'),
                    frequency="d",
                    adjustflag=adjust_map.get(adjust, '2')
                )

                data_list = []
                while rs.next():
                    data_list.append(rs.get_row_data())

                df = pd.DataFrame(data_list, columns=rs.fields)
                df['date'] = pd.to_datetime(df['date'])
                for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            # 标准化输出
            df['code'] = code
            return df[['code', 'date', 'open', 'high', 'low', 'close', 'volume', 'amount']].reset_index(drop=True)

        except Exception as e:
            logger.error(f"获取K线数据失败: {e}")
            return pd.DataFrame()

    def get_tick_data(self, code: str, trade_date: date) -> pd.DataFrame:
        """获取逐笔成交数据"""
        if not self._connected:
            raise ConnectionError("数据源未连接")

        try:
            if self.source == "akshare":
                df = self._api.stock_zh_a_tick_tx(
                    symbol=code,
                    trade_date=trade_date.strftime('%Y%m%d')
                )
                return df

        except Exception as e:
            logger.error(f"获取逐笔数据失败: {e}")
            return pd.DataFrame()

    def get_financial_data(self, code: str) -> Dict[str, Any]:
        """获取财务数据"""
        if not self._connected:
            raise ConnectionError("数据源未连接")

        try:
            if self.source == "akshare":
                # 获取财务指标
                df = self._api.stock_financial_analysis_indicator(symbol=code)
                if not df.empty:
                    return df.iloc[0].to_dict()

            elif self.source == "tushare":
                ts_code = f"{code}.SH" if code.startswith('6') else f"{code}.SZ"
                df = self._api.fina_indicator(ts_code=ts_code, limit=1)
                if not df.empty:
                    return df.iloc[0].to_dict()

        except Exception as e:
            logger.error(f"获取财务数据失败: {e}")

        return {}

    def get_index_components(self, index_code: str) -> List[str]:
        """获取指数成分股"""
        try:
            if self.source == "akshare":
                # 沪深300成分股
                if index_code == "000300":
                    df = self._api.index_stock_cons_csindex(symbol="000300")
                    return df['成分券代码'].tolist()
                # 上证50
                elif index_code == "000016":
                    df = self._api.index_stock_cons_csindex(symbol="000016")
                    return df['成分券代码'].tolist()
                # 中证500
                elif index_code == "000905":
                    df = self._api.index_stock_cons_csindex(symbol="000905")
                    return df['成分券代码'].tolist()

        except Exception as e:
            logger.error(f"获取指数成分股失败: {e}")

        return []

    def get_trade_calendar(self, start_date: date, end_date: date) -> List[date]:
        """获取交易日历"""
        try:
            if self.source == "akshare":
                df = self._api.tool_trade_date_hist_sina()
                df['trade_date'] = pd.to_datetime(df['trade_date']).dt.date
                mask = (df['trade_date'] >= start_date) & (df['trade_date'] <= end_date)
                return df.loc[mask, 'trade_date'].tolist()

            elif self.source == "tushare":
                df = self._api.trade_cal(
                    start_date=start_date.strftime('%Y%m%d'),
                    end_date=end_date.strftime('%Y%m%d'),
                    is_open='1'
                )
                return [datetime.strptime(d, '%Y%m%d').date() for d in df['cal_date'].tolist()]

        except Exception as e:
            logger.error(f"获取交易日历失败: {e}")

        return []
