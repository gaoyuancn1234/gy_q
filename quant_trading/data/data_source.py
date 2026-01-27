"""
数据源基类和数据结构定义
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import List, Optional, Dict, Any
from enum import Enum
import pandas as pd


class KLineType(Enum):
    """K线类型"""
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    MINUTE_60 = "60m"
    DAILY = "1d"
    WEEKLY = "1w"
    MONTHLY = "1M"


@dataclass
class StockInfo:
    """股票基本信息"""
    code: str                    # 股票代码
    name: str                    # 股票名称
    market: str                  # 市场（SH/SZ/HK）
    industry: str = ""           # 行业
    sector: str = ""             # 板块
    list_date: Optional[date] = None  # 上市日期
    total_shares: float = 0      # 总股本
    float_shares: float = 0      # 流通股本
    market_cap: float = 0        # 总市值
    pe_ratio: float = 0          # 市盈率
    pb_ratio: float = 0          # 市净率

    @property
    def full_code(self) -> str:
        """完整股票代码（带市场前缀）"""
        if self.market in ['SH', 'SZ']:
            return f"{self.market.lower()}.{self.code}"
        return f"{self.market.lower()}.{self.code}"


@dataclass
class StockData:
    """股票实时行情数据"""
    code: str                    # 股票代码
    name: str                    # 股票名称
    price: float                 # 当前价格
    open: float                  # 开盘价
    high: float                  # 最高价
    low: float                   # 最低价
    close: float                 # 收盘价（或最新价）
    pre_close: float             # 昨收价
    volume: float                # 成交量
    amount: float                # 成交额
    bid_price: List[float] = field(default_factory=list)   # 买一到买五价格
    bid_volume: List[float] = field(default_factory=list)  # 买一到买五数量
    ask_price: List[float] = field(default_factory=list)   # 卖一到卖五价格
    ask_volume: List[float] = field(default_factory=list)  # 卖一到卖五数量
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def change(self) -> float:
        """涨跌额"""
        return self.price - self.pre_close

    @property
    def change_pct(self) -> float:
        """涨跌幅"""
        if self.pre_close == 0:
            return 0
        return (self.price - self.pre_close) / self.pre_close * 100

    @property
    def turnover_rate(self) -> float:
        """换手率（需要额外数据）"""
        return 0


@dataclass
class KLineData:
    """K线数据"""
    code: str
    kline_type: KLineType
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    amount: float
    turnover: float = 0         # 换手率
    adjust_flag: int = 0        # 复权标志 0:不复权 1:前复权 2:后复权

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'code': self.code,
            'timestamp': self.timestamp,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'amount': self.amount,
            'turnover': self.turnover
        }


class DataSource(ABC):
    """数据源抽象基类"""

    def __init__(self, name: str):
        self.name = name
        self._connected = False

    @abstractmethod
    def connect(self) -> bool:
        """连接数据源"""
        pass

    @abstractmethod
    def disconnect(self):
        """断开连接"""
        pass

    @property
    def is_connected(self) -> bool:
        """是否已连接"""
        return self._connected

    @abstractmethod
    def get_stock_list(self) -> List[StockInfo]:
        """获取股票列表"""
        pass

    @abstractmethod
    def get_realtime_quote(self, codes: List[str]) -> List[StockData]:
        """获取实时行情"""
        pass

    @abstractmethod
    def get_kline(
        self,
        code: str,
        kline_type: KLineType,
        start_date: date,
        end_date: date,
        adjust: int = 1
    ) -> pd.DataFrame:
        """
        获取K线数据

        Args:
            code: 股票代码
            kline_type: K线类型
            start_date: 开始日期
            end_date: 结束日期
            adjust: 复权类型 0:不复权 1:前复权 2:后复权

        Returns:
            包含K线数据的DataFrame
        """
        pass

    @abstractmethod
    def get_tick_data(self, code: str, trade_date: date) -> pd.DataFrame:
        """获取逐笔成交数据"""
        pass

    def get_index_data(
        self,
        index_code: str,
        start_date: date,
        end_date: date
    ) -> pd.DataFrame:
        """获取指数数据"""
        return self.get_kline(index_code, KLineType.DAILY, start_date, end_date)

    def get_financial_data(self, code: str) -> Dict[str, Any]:
        """获取财务数据"""
        return {}

    def subscribe(self, codes: List[str], callback):
        """订阅实时行情"""
        raise NotImplementedError("实时订阅功能未实现")

    def unsubscribe(self, codes: List[str]):
        """取消订阅"""
        raise NotImplementedError("取消订阅功能未实现")
