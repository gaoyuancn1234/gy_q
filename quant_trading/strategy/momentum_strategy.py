"""
动量策略
包括动量策略和RSI策略
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any
from datetime import datetime
import logging

from .base_strategy import BaseStrategy, Signal, SignalType

logger = logging.getLogger(__name__)


class MomentumStrategy(BaseStrategy):
    """
    动量交易策略

    买入过去N天涨幅最大的股票
    卖出过去N天跌幅最大的股票
    """

    def __init__(
        self,
        name: str = "Momentum_Strategy",
        lookback_period: int = 20,
        holding_period: int = 5,
        top_n: int = 10,
        momentum_threshold: float = 0.05,
        **kwargs
    ):
        """
        初始化动量策略

        Args:
            name: 策略名称
            lookback_period: 回溯周期
            holding_period: 持有周期
            top_n: 选取前N只股票
            momentum_threshold: 动量阈值
        """
        params = {
            'lookback_period': lookback_period,
            'holding_period': holding_period,
            'top_n': top_n,
            'momentum_threshold': momentum_threshold,
            **kwargs
        }
        super().__init__(name, params)
        self._holding_days: Dict[str, int] = {}

    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """生成交易信号"""
        signals = []

        if len(data) < self.params['lookback_period']:
            return signals

        data = data.copy()

        # 计算动量（过去N天的收益率）
        data['momentum'] = data['close'].pct_change(self.params['lookback_period'])

        # 计算动量排名
        data['momentum_rank'] = data['momentum'].rank(ascending=False)

        latest = data.iloc[-1]
        code = latest.get('code', 'unknown')

        # 动量大于阈值，生成买入信号
        if latest['momentum'] > self.params['momentum_threshold']:
            strength = min(1.0, latest['momentum'] / self.params['momentum_threshold'])
            signals.append(Signal(
                code=code,
                signal_type=SignalType.BUY,
                price=latest['close'],
                strength=strength,
                reason=f"动量因子={latest['momentum']:.2%}，超过阈值{self.params['momentum_threshold']:.2%}",
                extra={'momentum': latest['momentum']}
            ))

        # 动量小于负阈值，生成卖出信号
        elif latest['momentum'] < -self.params['momentum_threshold']:
            strength = min(1.0, abs(latest['momentum']) / self.params['momentum_threshold'])
            signals.append(Signal(
                code=code,
                signal_type=SignalType.SELL,
                price=latest['close'],
                strength=strength,
                reason=f"动量因子={latest['momentum']:.2%}，低于阈值-{self.params['momentum_threshold']:.2%}",
                extra={'momentum': latest['momentum']}
            ))

        return signals


class RSIStrategy(BaseStrategy):
    """
    RSI交易策略

    RSI低于超卖区间时买入
    RSI高于超买区间时卖出
    """

    def __init__(
        self,
        name: str = "RSI_Strategy",
        rsi_period: int = 14,
        oversold: float = 30,
        overbought: float = 70,
        **kwargs
    ):
        """
        初始化RSI策略

        Args:
            name: 策略名称
            rsi_period: RSI计算周期
            oversold: 超卖阈值
            overbought: 超买阈值
        """
        params = {
            'rsi_period': rsi_period,
            'oversold': oversold,
            'overbought': overbought,
            **kwargs
        }
        super().__init__(name, params)

    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """生成交易信号"""
        signals = []

        if len(data) < self.params['rsi_period'] + 1:
            return signals

        data = data.copy()
        data['rsi'] = self.calculate_rsi(data['close'], self.params['rsi_period'])

        latest = data.iloc[-1]
        prev = data.iloc[-2]
        code = latest.get('code', 'unknown')

        # RSI从超卖区域回升，买入信号
        if prev['rsi'] < self.params['oversold'] and latest['rsi'] >= self.params['oversold']:
            strength = (self.params['oversold'] - prev['rsi']) / self.params['oversold']
            signals.append(Signal(
                code=code,
                signal_type=SignalType.BUY,
                price=latest['close'],
                strength=min(1.0, strength),
                reason=f"RSI从超卖区域回升，RSI={latest['rsi']:.2f}",
                extra={'rsi': latest['rsi'], 'prev_rsi': prev['rsi']}
            ))

        # RSI进入超卖区域，可能的抄底机会
        elif latest['rsi'] < self.params['oversold'] and prev['rsi'] >= self.params['oversold']:
            signals.append(Signal(
                code=code,
                signal_type=SignalType.BUY,
                price=latest['close'],
                strength=0.5,  # 较低强度，可能还会下跌
                reason=f"RSI进入超卖区域，RSI={latest['rsi']:.2f}",
                extra={'rsi': latest['rsi']}
            ))

        # RSI从超买区域回落，卖出信号
        elif prev['rsi'] > self.params['overbought'] and latest['rsi'] <= self.params['overbought']:
            strength = (prev['rsi'] - self.params['overbought']) / (100 - self.params['overbought'])
            signals.append(Signal(
                code=code,
                signal_type=SignalType.SELL,
                price=latest['close'],
                strength=min(1.0, strength),
                reason=f"RSI从超买区域回落，RSI={latest['rsi']:.2f}",
                extra={'rsi': latest['rsi'], 'prev_rsi': prev['rsi']}
            ))

        # RSI进入超买区域，可能的卖出机会
        elif latest['rsi'] > self.params['overbought'] and prev['rsi'] <= self.params['overbought']:
            signals.append(Signal(
                code=code,
                signal_type=SignalType.SELL,
                price=latest['close'],
                strength=0.5,
                reason=f"RSI进入超买区域，RSI={latest['rsi']:.2f}",
                extra={'rsi': latest['rsi']}
            ))

        return signals


class KDJStrategy(BaseStrategy):
    """
    KDJ交易策略

    K线上穿D线时买入（金叉）
    K线下穿D线时卖出（死叉）
    """

    def __init__(
        self,
        name: str = "KDJ_Strategy",
        n: int = 9,
        m1: int = 3,
        m2: int = 3,
        oversold: float = 20,
        overbought: float = 80,
        **kwargs
    ):
        params = {
            'n': n,
            'm1': m1,
            'm2': m2,
            'oversold': oversold,
            'overbought': overbought,
            **kwargs
        }
        super().__init__(name, params)

    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """生成交易信号"""
        signals = []

        if len(data) < self.params['n'] + 2:
            return signals

        data = data.copy()
        k, d, j = self.calculate_kdj(
            data['high'], data['low'], data['close'],
            self.params['n'], self.params['m1'], self.params['m2']
        )
        data['k'] = k
        data['d'] = d
        data['j'] = j

        latest = data.iloc[-1]
        prev = data.iloc[-2]
        code = latest.get('code', 'unknown')

        # K上穿D，且在超卖区域，强买入信号
        if prev['k'] <= prev['d'] and latest['k'] > latest['d']:
            in_oversold = latest['k'] < self.params['oversold']
            strength = 0.9 if in_oversold else 0.6

            signals.append(Signal(
                code=code,
                signal_type=SignalType.BUY,
                price=latest['close'],
                strength=strength,
                reason=f"KDJ金叉{'（超卖区）' if in_oversold else ''}，K={latest['k']:.2f}, D={latest['d']:.2f}",
                extra={'k': latest['k'], 'd': latest['d'], 'j': latest['j']}
            ))

        # K下穿D，且在超买区域，强卖出信号
        elif prev['k'] >= prev['d'] and latest['k'] < latest['d']:
            in_overbought = latest['k'] > self.params['overbought']
            strength = 0.9 if in_overbought else 0.6

            signals.append(Signal(
                code=code,
                signal_type=SignalType.SELL,
                price=latest['close'],
                strength=strength,
                reason=f"KDJ死叉{'（超买区）' if in_overbought else ''}，K={latest['k']:.2f}, D={latest['d']:.2f}",
                extra={'k': latest['k'], 'd': latest['d'], 'j': latest['j']}
            ))

        return signals
