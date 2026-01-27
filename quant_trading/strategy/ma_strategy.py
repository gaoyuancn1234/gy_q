"""
均线策略
包括双均线策略和MACD策略
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from .base_strategy import BaseStrategy, Signal, SignalType

logger = logging.getLogger(__name__)


class MAStrategy(BaseStrategy):
    """
    双均线交易策略

    当短期均线上穿长期均线时买入（金叉）
    当短期均线下穿长期均线时卖出（死叉）
    """

    def __init__(
        self,
        name: str = "MA_Strategy",
        short_period: int = 5,
        long_period: int = 20,
        **kwargs
    ):
        """
        初始化均线策略

        Args:
            name: 策略名称
            short_period: 短期均线周期
            long_period: 长期均线周期
        """
        params = {
            'short_period': short_period,
            'long_period': long_period,
            **kwargs
        }
        super().__init__(name, params)

    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """生成交易信号"""
        signals = []

        if len(data) < self.params['long_period']:
            return signals

        # 计算均线
        data = data.copy()
        data['ma_short'] = self.calculate_ma(data['close'], self.params['short_period'])
        data['ma_long'] = self.calculate_ma(data['close'], self.params['long_period'])

        # 计算金叉死叉
        data['cross'] = 0
        data.loc[data['ma_short'] > data['ma_long'], 'cross'] = 1
        data.loc[data['ma_short'] < data['ma_long'], 'cross'] = -1
        data['cross_signal'] = data['cross'].diff()

        # 获取最新信号
        latest = data.iloc[-1]
        code = latest.get('code', 'unknown')

        if latest['cross_signal'] == 2:  # 金叉
            strength = min(1.0, (latest['ma_short'] - latest['ma_long']) / latest['ma_long'] * 10)
            signals.append(Signal(
                code=code,
                signal_type=SignalType.BUY,
                price=latest['close'],
                strength=abs(strength),
                reason=f"MA{self.params['short_period']}上穿MA{self.params['long_period']}，金叉买入",
                extra={
                    'ma_short': latest['ma_short'],
                    'ma_long': latest['ma_long']
                }
            ))
        elif latest['cross_signal'] == -2:  # 死叉
            strength = min(1.0, (latest['ma_long'] - latest['ma_short']) / latest['ma_long'] * 10)
            signals.append(Signal(
                code=code,
                signal_type=SignalType.SELL,
                price=latest['close'],
                strength=abs(strength),
                reason=f"MA{self.params['short_period']}下穿MA{self.params['long_period']}，死叉卖出",
                extra={
                    'ma_short': latest['ma_short'],
                    'ma_long': latest['ma_long']
                }
            ))

        return signals


class MACDStrategy(BaseStrategy):
    """
    MACD交易策略

    DIF上穿DEA时买入
    DIF下穿DEA时卖出
    """

    def __init__(
        self,
        name: str = "MACD_Strategy",
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        **kwargs
    ):
        """
        初始化MACD策略

        Args:
            name: 策略名称
            fast_period: 快线周期
            slow_period: 慢线周期
            signal_period: 信号线周期
        """
        params = {
            'fast_period': fast_period,
            'slow_period': slow_period,
            'signal_period': signal_period,
            **kwargs
        }
        super().__init__(name, params)

    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """生成交易信号"""
        signals = []

        if len(data) < self.params['slow_period'] + self.params['signal_period']:
            return signals

        # 计算MACD
        data = data.copy()
        dif, dea, macd = self.calculate_macd(
            data['close'],
            self.params['fast_period'],
            self.params['slow_period'],
            self.params['signal_period']
        )
        data['dif'] = dif
        data['dea'] = dea
        data['macd'] = macd

        # 计算金叉死叉
        data['cross'] = 0
        data.loc[data['dif'] > data['dea'], 'cross'] = 1
        data.loc[data['dif'] < data['dea'], 'cross'] = -1
        data['cross_signal'] = data['cross'].diff()

        # 获取最新信号
        latest = data.iloc[-1]
        code = latest.get('code', 'unknown')

        if latest['cross_signal'] == 2:  # DIF上穿DEA
            # 判断是否在零轴上方（更强的信号）
            above_zero = latest['dif'] > 0 and latest['dea'] > 0
            strength = 0.8 if above_zero else 0.6

            signals.append(Signal(
                code=code,
                signal_type=SignalType.BUY,
                price=latest['close'],
                strength=strength,
                reason=f"MACD金叉{'（零轴上方）' if above_zero else ''}",
                extra={
                    'dif': latest['dif'],
                    'dea': latest['dea'],
                    'macd': latest['macd']
                }
            ))
        elif latest['cross_signal'] == -2:  # DIF下穿DEA
            below_zero = latest['dif'] < 0 and latest['dea'] < 0
            strength = 0.8 if below_zero else 0.6

            signals.append(Signal(
                code=code,
                signal_type=SignalType.SELL,
                price=latest['close'],
                strength=strength,
                reason=f"MACD死叉{'（零轴下方）' if below_zero else ''}",
                extra={
                    'dif': latest['dif'],
                    'dea': latest['dea'],
                    'macd': latest['macd']
                }
            ))

        return signals


class TripleMAStrategy(BaseStrategy):
    """
    三均线策略

    使用短中长三条均线判断趋势和交易时机
    """

    def __init__(
        self,
        name: str = "TripleMA_Strategy",
        short_period: int = 5,
        medium_period: int = 10,
        long_period: int = 20,
        **kwargs
    ):
        params = {
            'short_period': short_period,
            'medium_period': medium_period,
            'long_period': long_period,
            **kwargs
        }
        super().__init__(name, params)

    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """生成交易信号"""
        signals = []

        if len(data) < self.params['long_period']:
            return signals

        data = data.copy()
        data['ma_short'] = self.calculate_ma(data['close'], self.params['short_period'])
        data['ma_medium'] = self.calculate_ma(data['close'], self.params['medium_period'])
        data['ma_long'] = self.calculate_ma(data['close'], self.params['long_period'])

        latest = data.iloc[-1]
        prev = data.iloc[-2]
        code = latest.get('code', 'unknown')

        # 多头排列：短>中>长
        bull_trend = (latest['ma_short'] > latest['ma_medium'] > latest['ma_long'])
        # 空头排列：短<中<长
        bear_trend = (latest['ma_short'] < latest['ma_medium'] < latest['ma_long'])

        # 短期均线上穿中期均线，且在多头趋势中
        if (prev['ma_short'] <= prev['ma_medium'] and
            latest['ma_short'] > latest['ma_medium'] and
            latest['ma_medium'] > latest['ma_long']):
            signals.append(Signal(
                code=code,
                signal_type=SignalType.BUY,
                price=latest['close'],
                strength=0.8,
                reason="三均线多头排列确认",
                extra={
                    'ma_short': latest['ma_short'],
                    'ma_medium': latest['ma_medium'],
                    'ma_long': latest['ma_long']
                }
            ))

        # 短期均线下穿中期均线，且在空头趋势中
        elif (prev['ma_short'] >= prev['ma_medium'] and
              latest['ma_short'] < latest['ma_medium'] and
              latest['ma_medium'] < latest['ma_long']):
            signals.append(Signal(
                code=code,
                signal_type=SignalType.SELL,
                price=latest['close'],
                strength=0.8,
                reason="三均线空头排列确认",
                extra={
                    'ma_short': latest['ma_short'],
                    'ma_medium': latest['ma_medium'],
                    'ma_long': latest['ma_long']
                }
            ))

        return signals
