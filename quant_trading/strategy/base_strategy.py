"""
策略基类
定义策略的基本接口和通用功能
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional
from enum import Enum
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """信号类型"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class Signal:
    """交易信号"""
    code: str                    # 股票代码
    signal_type: SignalType      # 信号类型
    price: float                 # 建议价格
    quantity: int = 0            # 建议数量
    strength: float = 1.0        # 信号强度 (0-1)
    reason: str = ""             # 信号原因
    timestamp: datetime = field(default_factory=datetime.now)
    extra: Dict[str, Any] = field(default_factory=dict)

    def __str__(self):
        return f"Signal({self.code}, {self.signal_type.value}, price={self.price:.2f}, strength={self.strength:.2f})"


class BaseStrategy(ABC):
    """策略基类"""

    def __init__(self, name: str, params: Optional[Dict[str, Any]] = None):
        """
        初始化策略

        Args:
            name: 策略名称
            params: 策略参数
        """
        self.name = name
        self.params = params or {}
        self._positions: Dict[str, int] = {}  # 当前持仓
        self._cash: float = 0  # 可用资金
        self._initialized = False

    @property
    def positions(self) -> Dict[str, int]:
        """当前持仓"""
        return self._positions.copy()

    @property
    def cash(self) -> float:
        """可用资金"""
        return self._cash

    def initialize(self, cash: float, positions: Optional[Dict[str, int]] = None):
        """
        初始化策略状态

        Args:
            cash: 初始资金
            positions: 初始持仓
        """
        self._cash = cash
        self._positions = positions or {}
        self._initialized = True
        logger.info(f"策略 {self.name} 初始化完成，资金: {cash:.2f}")

    def update_position(self, code: str, quantity: int, price: float, is_buy: bool):
        """
        更新持仓

        Args:
            code: 股票代码
            quantity: 成交数量
            price: 成交价格
            is_buy: 是否买入
        """
        if is_buy:
            current = self._positions.get(code, 0)
            self._positions[code] = current + quantity
            self._cash -= quantity * price
        else:
            current = self._positions.get(code, 0)
            self._positions[code] = max(0, current - quantity)
            self._cash += quantity * price

            if self._positions[code] == 0:
                del self._positions[code]

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """
        生成交易信号

        Args:
            data: 行情数据DataFrame，至少包含 open, high, low, close, volume 列

        Returns:
            交易信号列表
        """
        pass

    def filter_signals(self, signals: List[Signal]) -> List[Signal]:
        """
        过滤信号

        Args:
            signals: 原始信号列表

        Returns:
            过滤后的信号列表
        """
        # 默认只返回强度大于0.5的信号
        return [s for s in signals if s.strength >= 0.5]

    def on_bar(self, bar: Dict[str, Any]) -> Optional[Signal]:
        """
        每根K线回调

        Args:
            bar: K线数据

        Returns:
            交易信号（如果有）
        """
        return None

    def on_tick(self, tick: Dict[str, Any]) -> Optional[Signal]:
        """
        每个tick回调

        Args:
            tick: tick数据

        Returns:
            交易信号（如果有）
        """
        return None

    def on_trade(self, trade: Dict[str, Any]):
        """
        成交回调

        Args:
            trade: 成交信息
        """
        logger.info(f"策略 {self.name} 成交: {trade}")

    def get_params(self) -> Dict[str, Any]:
        """获取策略参数"""
        return self.params.copy()

    def set_params(self, params: Dict[str, Any]):
        """设置策略参数"""
        self.params.update(params)

    @staticmethod
    def calculate_ma(data: pd.Series, period: int) -> pd.Series:
        """计算移动平均线"""
        return data.rolling(window=period).mean()

    @staticmethod
    def calculate_ema(data: pd.Series, period: int) -> pd.Series:
        """计算指数移动平均线"""
        return data.ewm(span=period, adjust=False).mean()

    @staticmethod
    def calculate_rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """计算RSI指标"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def calculate_macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        """
        计算MACD指标

        Returns:
            (dif, dea, macd)
        """
        ema_fast = data.ewm(span=fast, adjust=False).mean()
        ema_slow = data.ewm(span=slow, adjust=False).mean()
        dif = ema_fast - ema_slow
        dea = dif.ewm(span=signal, adjust=False).mean()
        macd = (dif - dea) * 2
        return dif, dea, macd

    @staticmethod
    def calculate_bollinger_bands(data: pd.Series, period: int = 20, std_dev: float = 2.0):
        """
        计算布林带

        Returns:
            (upper, middle, lower)
        """
        middle = data.rolling(window=period).mean()
        std = data.rolling(window=period).std()
        upper = middle + std_dev * std
        lower = middle - std_dev * std
        return upper, middle, lower

    @staticmethod
    def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """计算ATR（平均真实波幅）"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    @staticmethod
    def calculate_kdj(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 9, m1: int = 3, m2: int = 3):
        """
        计算KDJ指标

        Returns:
            (k, d, j)
        """
        lowest_low = low.rolling(window=n).min()
        highest_high = high.rolling(window=n).max()
        rsv = (close - lowest_low) / (highest_high - lowest_low) * 100

        k = rsv.ewm(span=m1, adjust=False).mean()
        d = k.ewm(span=m2, adjust=False).mean()
        j = 3 * k - 2 * d

        return k, d, j

    def __str__(self):
        return f"Strategy({self.name}, params={self.params})"

    def __repr__(self):
        return self.__str__()
