"""
交易策略模块
包含多种经典交易策略
"""

from .base_strategy import BaseStrategy, Signal, SignalType
from .ma_strategy import MAStrategy, MACDStrategy
from .momentum_strategy import MomentumStrategy, RSIStrategy
from .mean_reversion import MeanReversionStrategy, BollingerBandStrategy
from .factor_strategy import FactorStrategy, MultiFactorStrategy

__all__ = [
    'BaseStrategy', 'Signal', 'SignalType',
    'MAStrategy', 'MACDStrategy',
    'MomentumStrategy', 'RSIStrategy',
    'MeanReversionStrategy', 'BollingerBandStrategy',
    'FactorStrategy', 'MultiFactorStrategy'
]
