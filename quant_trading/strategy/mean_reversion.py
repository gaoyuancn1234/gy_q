"""
均值回归策略
包括均值回归策略和布林带策略
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any
from datetime import datetime
import logging

from .base_strategy import BaseStrategy, Signal, SignalType

logger = logging.getLogger(__name__)


class MeanReversionStrategy(BaseStrategy):
    """
    均值回归策略

    当价格偏离均值一定幅度时，预期价格会回归均值
    """

    def __init__(
        self,
        name: str = "MeanReversion_Strategy",
        lookback_period: int = 20,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.5,
        **kwargs
    ):
        """
        初始化均值回归策略

        Args:
            name: 策略名称
            lookback_period: 回溯周期
            entry_threshold: 入场阈值（标准差倍数）
            exit_threshold: 出场阈值（标准差倍数）
        """
        params = {
            'lookback_period': lookback_period,
            'entry_threshold': entry_threshold,
            'exit_threshold': exit_threshold,
            **kwargs
        }
        super().__init__(name, params)

    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """生成交易信号"""
        signals = []

        if len(data) < self.params['lookback_period']:
            return signals

        data = data.copy()

        # 计算均值和标准差
        data['mean'] = data['close'].rolling(window=self.params['lookback_period']).mean()
        data['std'] = data['close'].rolling(window=self.params['lookback_period']).std()

        # 计算Z-score
        data['zscore'] = (data['close'] - data['mean']) / data['std']

        latest = data.iloc[-1]
        prev = data.iloc[-2]
        code = latest.get('code', 'unknown')

        # 价格低于均值-entry_threshold*std，买入
        if latest['zscore'] < -self.params['entry_threshold']:
            strength = min(1.0, abs(latest['zscore']) / self.params['entry_threshold'] / 2)
            signals.append(Signal(
                code=code,
                signal_type=SignalType.BUY,
                price=latest['close'],
                strength=strength,
                reason=f"价格显著低于均值，Z-score={latest['zscore']:.2f}",
                extra={
                    'zscore': latest['zscore'],
                    'mean': latest['mean'],
                    'std': latest['std']
                }
            ))

        # 价格高于均值+entry_threshold*std，卖出
        elif latest['zscore'] > self.params['entry_threshold']:
            strength = min(1.0, abs(latest['zscore']) / self.params['entry_threshold'] / 2)
            signals.append(Signal(
                code=code,
                signal_type=SignalType.SELL,
                price=latest['close'],
                strength=strength,
                reason=f"价格显著高于均值，Z-score={latest['zscore']:.2f}",
                extra={
                    'zscore': latest['zscore'],
                    'mean': latest['mean'],
                    'std': latest['std']
                }
            ))

        # Z-score回归到出场阈值
        elif (abs(prev['zscore']) > self.params['entry_threshold'] and
              abs(latest['zscore']) < self.params['exit_threshold']):
            signals.append(Signal(
                code=code,
                signal_type=SignalType.SELL if prev['zscore'] < 0 else SignalType.BUY,
                price=latest['close'],
                strength=0.7,
                reason=f"价格回归均值，Z-score={latest['zscore']:.2f}",
                extra={
                    'zscore': latest['zscore'],
                    'mean': latest['mean']
                }
            ))

        return signals


class BollingerBandStrategy(BaseStrategy):
    """
    布林带交易策略

    价格触及下轨时买入
    价格触及上轨时卖出
    """

    def __init__(
        self,
        name: str = "BollingerBand_Strategy",
        period: int = 20,
        std_dev: float = 2.0,
        **kwargs
    ):
        """
        初始化布林带策略

        Args:
            name: 策略名称
            period: 均线周期
            std_dev: 标准差倍数
        """
        params = {
            'period': period,
            'std_dev': std_dev,
            **kwargs
        }
        super().__init__(name, params)

    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """生成交易信号"""
        signals = []

        if len(data) < self.params['period']:
            return signals

        data = data.copy()

        # 计算布林带
        upper, middle, lower = self.calculate_bollinger_bands(
            data['close'],
            self.params['period'],
            self.params['std_dev']
        )
        data['bb_upper'] = upper
        data['bb_middle'] = middle
        data['bb_lower'] = lower

        # 计算带宽
        data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / data['bb_middle']

        # 计算价格在布林带中的位置 (0=下轨, 1=上轨)
        data['bb_position'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])

        latest = data.iloc[-1]
        prev = data.iloc[-2]
        code = latest.get('code', 'unknown')

        # 价格突破下轨后回升，买入
        if prev['close'] < prev['bb_lower'] and latest['close'] >= latest['bb_lower']:
            signals.append(Signal(
                code=code,
                signal_type=SignalType.BUY,
                price=latest['close'],
                strength=0.8,
                reason=f"价格从布林带下轨回升",
                extra={
                    'bb_upper': latest['bb_upper'],
                    'bb_middle': latest['bb_middle'],
                    'bb_lower': latest['bb_lower'],
                    'bb_position': latest['bb_position']
                }
            ))

        # 价格触及下轨，可能的买入机会
        elif latest['close'] <= latest['bb_lower']:
            signals.append(Signal(
                code=code,
                signal_type=SignalType.BUY,
                price=latest['close'],
                strength=0.5,
                reason=f"价格触及布林带下轨",
                extra={
                    'bb_lower': latest['bb_lower'],
                    'bb_position': latest['bb_position']
                }
            ))

        # 价格突破上轨后回落，卖出
        elif prev['close'] > prev['bb_upper'] and latest['close'] <= latest['bb_upper']:
            signals.append(Signal(
                code=code,
                signal_type=SignalType.SELL,
                price=latest['close'],
                strength=0.8,
                reason=f"价格从布林带上轨回落",
                extra={
                    'bb_upper': latest['bb_upper'],
                    'bb_middle': latest['bb_middle'],
                    'bb_lower': latest['bb_lower'],
                    'bb_position': latest['bb_position']
                }
            ))

        # 价格触及上轨，可能的卖出机会
        elif latest['close'] >= latest['bb_upper']:
            signals.append(Signal(
                code=code,
                signal_type=SignalType.SELL,
                price=latest['close'],
                strength=0.5,
                reason=f"价格触及布林带上轨",
                extra={
                    'bb_upper': latest['bb_upper'],
                    'bb_position': latest['bb_position']
                }
            ))

        return signals


class PairsTradingStrategy(BaseStrategy):
    """
    配对交易策略

    利用两只相关股票的价差进行套利
    """

    def __init__(
        self,
        name: str = "PairsTrading_Strategy",
        lookback_period: int = 60,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.5,
        stock_a: str = "",
        stock_b: str = "",
        **kwargs
    ):
        params = {
            'lookback_period': lookback_period,
            'entry_threshold': entry_threshold,
            'exit_threshold': exit_threshold,
            'stock_a': stock_a,
            'stock_b': stock_b,
            **kwargs
        }
        super().__init__(name, params)

    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """
        生成交易信号

        data需要包含两只股票的价格数据：price_a, price_b
        """
        signals = []

        if len(data) < self.params['lookback_period']:
            return signals

        data = data.copy()

        # 计算价差
        data['spread'] = data['price_a'] - data['price_b']
        data['spread_mean'] = data['spread'].rolling(window=self.params['lookback_period']).mean()
        data['spread_std'] = data['spread'].rolling(window=self.params['lookback_period']).std()
        data['zscore'] = (data['spread'] - data['spread_mean']) / data['spread_std']

        latest = data.iloc[-1]

        # 价差过大，做空A买入B
        if latest['zscore'] > self.params['entry_threshold']:
            signals.append(Signal(
                code=self.params['stock_a'],
                signal_type=SignalType.SELL,
                price=latest['price_a'],
                strength=min(1.0, latest['zscore'] / self.params['entry_threshold'] / 2),
                reason=f"配对交易：价差Z-score={latest['zscore']:.2f}，做空{self.params['stock_a']}"
            ))
            signals.append(Signal(
                code=self.params['stock_b'],
                signal_type=SignalType.BUY,
                price=latest['price_b'],
                strength=min(1.0, latest['zscore'] / self.params['entry_threshold'] / 2),
                reason=f"配对交易：价差Z-score={latest['zscore']:.2f}，买入{self.params['stock_b']}"
            ))

        # 价差过小，买入A做空B
        elif latest['zscore'] < -self.params['entry_threshold']:
            signals.append(Signal(
                code=self.params['stock_a'],
                signal_type=SignalType.BUY,
                price=latest['price_a'],
                strength=min(1.0, abs(latest['zscore']) / self.params['entry_threshold'] / 2),
                reason=f"配对交易：价差Z-score={latest['zscore']:.2f}，买入{self.params['stock_a']}"
            ))
            signals.append(Signal(
                code=self.params['stock_b'],
                signal_type=SignalType.SELL,
                price=latest['price_b'],
                strength=min(1.0, abs(latest['zscore']) / self.params['entry_threshold'] / 2),
                reason=f"配对交易：价差Z-score={latest['zscore']:.2f}，做空{self.params['stock_b']}"
            ))

        return signals
