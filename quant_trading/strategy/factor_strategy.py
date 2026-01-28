"""
因子策略
包括单因子策略和多因子策略
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Callable
from datetime import datetime
import logging

from .base_strategy import BaseStrategy, Signal, SignalType

logger = logging.getLogger(__name__)


class FactorStrategy(BaseStrategy):
    """
    单因子策略

    根据单一因子对股票进行排序，做多因子值高的股票
    """

    def __init__(
        self,
        name: str = "Factor_Strategy",
        factor_name: str = "momentum",
        top_ratio: float = 0.1,
        bottom_ratio: float = 0.1,
        holding_period: int = 20,
        **kwargs
    ):
        """
        初始化单因子策略

        Args:
            name: 策略名称
            factor_name: 因子名称
            top_ratio: 买入比例（因子值最高的前N%）
            bottom_ratio: 卖出比例（因子值最低的后N%）
            holding_period: 持有周期
        """
        params = {
            'factor_name': factor_name,
            'top_ratio': top_ratio,
            'bottom_ratio': bottom_ratio,
            'holding_period': holding_period,
            **kwargs
        }
        super().__init__(name, params)

    def calculate_factor(self, data: pd.DataFrame) -> pd.Series:
        """
        计算因子值

        Args:
            data: 股票数据

        Returns:
            因子值Series
        """
        factor_name = self.params['factor_name']

        if factor_name == 'momentum':
            # 动量因子：过去20日收益率
            return data['close'].pct_change(20)

        elif factor_name == 'volatility':
            # 波动率因子：过去20日收益率标准差
            return data['close'].pct_change().rolling(20).std()

        elif factor_name == 'volume':
            # 成交量因子：成交量/20日均量
            return data['volume'] / data['volume'].rolling(20).mean()

        elif factor_name == 'reversal':
            # 反转因子：过去5日收益率的负值
            return -data['close'].pct_change(5)

        elif factor_name == 'turnover':
            # 换手率因子
            if 'turnover' in data.columns:
                return data['turnover']
            return data['volume'] / data['volume'].rolling(20).mean()

        else:
            # 默认使用收益率作为因子
            return data['close'].pct_change(20)

    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """生成交易信号"""
        signals = []

        if len(data) < 21:
            return signals

        data = data.copy()

        # 计算因子值
        data['factor'] = self.calculate_factor(data)

        # 获取最新因子值
        latest = data.iloc[-1]
        factor_value = latest['factor']
        code = latest.get('code', 'unknown')

        if pd.isna(factor_value):
            return signals

        # 计算因子排名（需要多只股票的数据才有意义，这里简化处理）
        # 实际使用时应该传入多只股票的因子值

        # 因子值高于阈值，买入
        if factor_value > 0:
            strength = min(1.0, abs(factor_value) * 5)
            signals.append(Signal(
                code=code,
                signal_type=SignalType.BUY,
                price=latest['close'],
                strength=strength,
                reason=f"{self.params['factor_name']}因子={factor_value:.4f}",
                extra={'factor_value': factor_value}
            ))

        # 因子值低于阈值，卖出
        elif factor_value < 0:
            strength = min(1.0, abs(factor_value) * 5)
            signals.append(Signal(
                code=code,
                signal_type=SignalType.SELL,
                price=latest['close'],
                strength=strength,
                reason=f"{self.params['factor_name']}因子={factor_value:.4f}",
                extra={'factor_value': factor_value}
            ))

        return signals


class MultiFactorStrategy(BaseStrategy):
    """
    多因子策略

    综合多个因子进行股票选择
    """

    def __init__(
        self,
        name: str = "MultiFactor_Strategy",
        factors: List[str] = None,
        weights: Dict[str, float] = None,
        top_ratio: float = 0.1,
        holding_period: int = 20,
        **kwargs
    ):
        """
        初始化多因子策略

        Args:
            name: 策略名称
            factors: 因子列表
            weights: 因子权重
            top_ratio: 买入比例
            holding_period: 持有周期
        """
        default_factors = ['momentum', 'volatility', 'reversal']
        default_weights = {'momentum': 0.4, 'volatility': -0.3, 'reversal': 0.3}

        params = {
            'factors': factors or default_factors,
            'weights': weights or default_weights,
            'top_ratio': top_ratio,
            'holding_period': holding_period,
            **kwargs
        }
        super().__init__(name, params)

    def calculate_factor(self, data: pd.DataFrame, factor_name: str) -> pd.Series:
        """计算单个因子"""
        if factor_name == 'momentum':
            return data['close'].pct_change(20)
        elif factor_name == 'volatility':
            return data['close'].pct_change().rolling(20).std()
        elif factor_name == 'reversal':
            return -data['close'].pct_change(5)
        elif factor_name == 'size':
            return np.log(data['close'] * data['volume'])
        elif factor_name == 'liquidity':
            return data['volume'].rolling(20).mean()
        else:
            return pd.Series(0, index=data.index)

    def standardize_factor(self, factor: pd.Series) -> pd.Series:
        """因子标准化（Z-score）"""
        return (factor - factor.mean()) / factor.std()

    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """生成交易信号"""
        signals = []

        if len(data) < 21:
            return signals

        data = data.copy()

        # 计算各因子值
        factor_values = {}
        for factor_name in self.params['factors']:
            factor = self.calculate_factor(data, factor_name)
            factor_values[factor_name] = factor

        # 计算综合得分
        composite_score = pd.Series(0.0, index=data.index)
        for factor_name, weight in self.params['weights'].items():
            if factor_name in factor_values:
                standardized = self.standardize_factor(factor_values[factor_name])
                composite_score += weight * standardized.fillna(0)

        data['composite_score'] = composite_score

        # 获取最新得分
        latest = data.iloc[-1]
        score = latest['composite_score']
        code = latest.get('code', 'unknown')

        if pd.isna(score):
            return signals

        # 综合得分高，买入信号
        if score > 1:
            strength = min(1.0, score / 2)
            signals.append(Signal(
                code=code,
                signal_type=SignalType.BUY,
                price=latest['close'],
                strength=strength,
                reason=f"多因子综合得分={score:.2f}",
                extra={
                    'composite_score': score,
                    'factor_values': {k: v.iloc[-1] for k, v in factor_values.items()}
                }
            ))

        # 综合得分低，卖出信号
        elif score < -1:
            strength = min(1.0, abs(score) / 2)
            signals.append(Signal(
                code=code,
                signal_type=SignalType.SELL,
                price=latest['close'],
                strength=strength,
                reason=f"多因子综合得分={score:.2f}",
                extra={
                    'composite_score': score,
                    'factor_values': {k: v.iloc[-1] for k, v in factor_values.items()}
                }
            ))

        return signals


class AlphaFactorStrategy(BaseStrategy):
    """
    Alpha因子策略

    使用自定义Alpha因子进行交易
    """

    def __init__(
        self,
        name: str = "AlphaFactor_Strategy",
        alpha_func: Callable = None,
        threshold: float = 0,
        **kwargs
    ):
        """
        初始化Alpha因子策略

        Args:
            name: 策略名称
            alpha_func: Alpha因子计算函数
            threshold: 信号阈值
        """
        params = {
            'threshold': threshold,
            **kwargs
        }
        super().__init__(name, params)
        self.alpha_func = alpha_func

    def default_alpha(self, data: pd.DataFrame) -> pd.Series:
        """默认Alpha因子"""
        # Alpha#1: (rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2.), 5)) - 0.5)
        returns = data['close'].pct_change()
        cond = returns < 0
        stddev = returns.rolling(20).std()
        signed_power = np.where(cond, stddev, data['close']) ** 2
        alpha = pd.Series(signed_power).rolling(5).apply(lambda x: x.argmax())
        return (alpha.rank(pct=True) - 0.5)

    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """生成交易信号"""
        signals = []

        if len(data) < 21:
            return signals

        data = data.copy()

        # 计算Alpha因子
        if self.alpha_func:
            data['alpha'] = self.alpha_func(data)
        else:
            data['alpha'] = self.default_alpha(data)

        latest = data.iloc[-1]
        alpha_value = latest['alpha']
        code = latest.get('code', 'unknown')

        if pd.isna(alpha_value):
            return signals

        threshold = self.params['threshold']

        if alpha_value > threshold:
            strength = min(1.0, (alpha_value - threshold) * 2)
            signals.append(Signal(
                code=code,
                signal_type=SignalType.BUY,
                price=latest['close'],
                strength=strength,
                reason=f"Alpha因子={alpha_value:.4f}",
                extra={'alpha': alpha_value}
            ))
        elif alpha_value < -threshold:
            strength = min(1.0, abs(alpha_value + threshold) * 2)
            signals.append(Signal(
                code=code,
                signal_type=SignalType.SELL,
                price=latest['close'],
                strength=strength,
                reason=f"Alpha因子={alpha_value:.4f}",
                extra={'alpha': alpha_value}
            ))

        return signals
