"""
量化交易系统
支持A股和港股的量化交易、回测和模拟交易

主要功能:
- 数据获取: 支持akshare、tushare、baostock等数据源
- 交易策略: 均线、动量、均值回归、因子等多种策略
- 回测引擎: 完整的回测和绩效分析
- 风险管理: 仓位控制、止损止盈、风险监控
- 真实交易: 支持A股(easytrader/QMT)和港股(富途/长桥)
- 监控告警: 实时监控和多渠道告警
"""

__version__ = "1.0.0"
__author__ = "QuantTrading Team"

from .config import Settings, BrokerConfig
from .data import DataSource, AStockDataSource, HKStockDataSource, Database
from .strategy import (
    BaseStrategy, Signal, SignalType,
    MAStrategy, MACDStrategy,
    MomentumStrategy, RSIStrategy,
    MeanReversionStrategy, BollingerBandStrategy,
    FactorStrategy, MultiFactorStrategy
)
from .backtest import BacktestEngine, BacktestResult, PerformanceAnalyzer
from .risk import PositionManager, PositionSizer, RiskController
from .broker import (
    BaseBroker, Order, OrderStatus, OrderType,
    AStockBroker, EasyTraderBroker,
    HKStockBroker, FutuBroker, LongbridgeBroker,
    PaperTradingBroker
)
from .monitor import Dashboard, AlertManager

__all__ = [
    # 版本
    '__version__',

    # 配置
    'Settings', 'BrokerConfig',

    # 数据
    'DataSource', 'AStockDataSource', 'HKStockDataSource', 'Database',

    # 策略
    'BaseStrategy', 'Signal', 'SignalType',
    'MAStrategy', 'MACDStrategy',
    'MomentumStrategy', 'RSIStrategy',
    'MeanReversionStrategy', 'BollingerBandStrategy',
    'FactorStrategy', 'MultiFactorStrategy',

    # 回测
    'BacktestEngine', 'BacktestResult', 'PerformanceAnalyzer',

    # 风控
    'PositionManager', 'PositionSizer', 'RiskController',

    # 交易
    'BaseBroker', 'Order', 'OrderStatus', 'OrderType',
    'AStockBroker', 'EasyTraderBroker',
    'HKStockBroker', 'FutuBroker', 'LongbridgeBroker',
    'PaperTradingBroker',

    # 监控
    'Dashboard', 'AlertManager'
]
