"""
港股数据源
支持富途、akshare等数据源
"""

import pandas as pd
from datetime import date, datetime
from typing import List, Dict, Any, Optional
import logging

from .data_source import DataSource, StockInfo, StockData, KLineData, KLineType

logger = logging.getLogger(__name__)


class HKStockDataSource(DataSource):
    """
    港股数据源
    支持 akshare（免费）和 富途API（需要开户）
    """

    def __init__(self, source: str = "akshare", **kwargs):
        """
        初始化港股数据源

        Args:
            source: 数据源类型 ("akshare", "futu")
            **kwargs: 额外配置参数
        """
        super().__init__(f"HKStock-{source}")
        self.source = source
        self.config = kwargs
        self._api = None
        self._quote_ctx = None
        self._trade_ctx = None

    def connect(self) -> bool:
        """连接数据源"""
        try:
            if self.source == "akshare":
                import akshare as ak
                self._api = ak
                self._connected = True
                logger.info("已连接到 akshare 港股数据源")

            elif self.source == "futu":
                from futu import OpenQuoteContext, OpenSecTradeContext
                host = self.config.get('host', '127.0.0.1')
                port = self.config.get('port', 11111)

                self._quote_ctx = OpenQuoteContext(host=host, port=port)
                self._connected = True
                logger.info(f"已连接到富途行情服务 {host}:{port}")

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
        if self.source == "futu":
            if self._quote_ctx:
                self._quote_ctx.close()
            if self._trade_ctx:
                self._trade_ctx.close()

        self._connected = False
        self._api = None
        self._quote_ctx = None
        self._trade_ctx = None
        logger.info("已断开港股数据源连接")

    def get_stock_list(self) -> List[StockInfo]:
        """获取港股股票列表"""
        if not self._connected:
            raise ConnectionError("数据源未连接")

        stocks = []

        try:
            if self.source == "akshare":
                # 获取港股列表
                df = self._api.stock_hk_spot_em()
                for _, row in df.iterrows():
                    stocks.append(StockInfo(
                        code=str(row['代码']),
                        name=row['名称'],
                        market='HK',
                        market_cap=row.get('总市值', 0),
                        pe_ratio=row.get('市盈率', 0),
                        pb_ratio=row.get('市净率', 0)
                    ))

            elif self.source == "futu":
                from futu import Market, SecurityType, SubType

                # 获取港股主板列表
                ret, data = self._quote_ctx.get_stock_basicinfo(
                    market=Market.HK,
                    stock_type=SecurityType.STOCK
                )
                if ret == 0:
                    for _, row in data.iterrows():
                        code = row['code'].replace('HK.', '')
                        stocks.append(StockInfo(
                            code=code,
                            name=row['name'],
                            market='HK',
                            list_date=pd.to_datetime(row.get('listing_date')).date() if row.get('listing_date') else None
                        ))

        except Exception as e:
            logger.error(f"获取港股列表失败: {e}")

        return stocks

    def get_realtime_quote(self, codes: List[str]) -> List[StockData]:
        """获取实时行情"""
        if not self._connected:
            raise ConnectionError("数据源未连接")

        quotes = []

        try:
            if self.source == "akshare":
                df = self._api.stock_hk_spot_em()
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

            elif self.source == "futu":
                from futu import SubType

                # 转换为富途代码格式
                futu_codes = [f"HK.{code}" for code in codes]

                ret, data = self._quote_ctx.get_market_snapshot(futu_codes)
                if ret == 0:
                    for _, row in data.iterrows():
                        code = row['code'].replace('HK.', '')
                        quotes.append(StockData(
                            code=code,
                            name=row['name'],
                            price=row['last_price'],
                            open=row['open_price'],
                            high=row['high_price'],
                            low=row['low_price'],
                            close=row['last_price'],
                            pre_close=row['prev_close_price'],
                            volume=row['volume'],
                            amount=row['turnover'],
                            timestamp=datetime.now()
                        ))

        except Exception as e:
            logger.error(f"获取港股实时行情失败: {e}")

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
                # K线周期映射
                period_map = {
                    KLineType.DAILY: 'daily',
                    KLineType.WEEKLY: 'weekly',
                    KLineType.MONTHLY: 'monthly',
                }
                period = period_map.get(kline_type, 'daily')

                # 复权类型映射
                adjust_map = {0: '', 1: 'qfq', 2: 'hfq'}
                adjust_type = adjust_map.get(adjust, '')

                df = self._api.stock_hk_hist(
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
                    '成交额': 'amount'
                })

            elif self.source == "futu":
                from futu import KLType, AuType

                # K线类型映射
                kl_type_map = {
                    KLineType.MINUTE_1: KLType.K_1M,
                    KLineType.MINUTE_5: KLType.K_5M,
                    KLineType.MINUTE_15: KLType.K_15M,
                    KLineType.MINUTE_30: KLType.K_30M,
                    KLineType.MINUTE_60: KLType.K_60M,
                    KLineType.DAILY: KLType.K_DAY,
                    KLineType.WEEKLY: KLType.K_WEEK,
                    KLineType.MONTHLY: KLType.K_MON,
                }

                # 复权类型映射
                au_type_map = {
                    0: AuType.NONE,
                    1: AuType.QFQ,
                    2: AuType.HFQ
                }

                futu_code = f"HK.{code}"
                ret, data, _ = self._quote_ctx.request_history_kline(
                    futu_code,
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d'),
                    ktype=kl_type_map.get(kline_type, KLType.K_DAY),
                    autype=au_type_map.get(adjust, AuType.QFQ),
                    max_count=10000
                )

                if ret == 0:
                    df = data.rename(columns={
                        'time_key': 'date',
                        'turnover': 'amount',
                        'turnover_rate': 'turnover'
                    })
                else:
                    logger.error(f"获取K线失败: {data}")
                    return pd.DataFrame()

            # 标准化输出
            df['code'] = code
            df['date'] = pd.to_datetime(df['date'])
            return df[['code', 'date', 'open', 'high', 'low', 'close', 'volume', 'amount']].reset_index(drop=True)

        except Exception as e:
            logger.error(f"获取港股K线数据失败: {e}")
            return pd.DataFrame()

    def get_tick_data(self, code: str, trade_date: date) -> pd.DataFrame:
        """获取逐笔成交数据"""
        if not self._connected:
            raise ConnectionError("数据源未连接")

        try:
            if self.source == "futu":
                futu_code = f"HK.{code}"
                ret, data = self._quote_ctx.get_rt_ticker(futu_code, 1000)
                if ret == 0:
                    return data

        except Exception as e:
            logger.error(f"获取港股逐笔数据失败: {e}")

        return pd.DataFrame()

    def subscribe(self, codes: List[str], callback):
        """订阅实时行情（仅支持富途）"""
        if self.source != "futu":
            raise NotImplementedError("只有富途支持实时订阅")

        if not self._connected:
            raise ConnectionError("数据源未连接")

        from futu import SubType

        class QuoteHandler:
            def __init__(self, cb):
                self.callback = cb

            def on_recv_rsp(self, rsp_str):
                ret_code, data = super().on_recv_rsp(rsp_str)
                if ret_code == 0:
                    self.callback(data)
                return ret_code, data

        futu_codes = [f"HK.{code}" for code in codes]
        ret, err = self._quote_ctx.subscribe(futu_codes, [SubType.QUOTE])

        if ret != 0:
            logger.error(f"订阅失败: {err}")
            return False

        logger.info(f"已订阅 {len(codes)} 只港股")
        return True

    def unsubscribe(self, codes: List[str]):
        """取消订阅"""
        if self.source != "futu":
            return

        if not self._connected:
            return

        from futu import SubType

        futu_codes = [f"HK.{code}" for code in codes]
        self._quote_ctx.unsubscribe(futu_codes, [SubType.QUOTE])
        logger.info(f"已取消订阅 {len(codes)} 只港股")

    def get_hk_connect_stocks(self) -> List[str]:
        """获取港股通标的"""
        try:
            if self.source == "akshare":
                # 沪港通成分股
                df = self._api.stock_hk_ggt_components_em()
                return df['代码'].tolist()

        except Exception as e:
            logger.error(f"获取港股通标的失败: {e}")

        return []
