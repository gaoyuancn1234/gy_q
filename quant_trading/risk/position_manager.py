"""
仓位管理模块
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class PositionInfo:
    """持仓信息"""
    code: str
    market: str
    quantity: int
    avg_cost: float
    current_price: float
    market_value: float
    weight: float  # 占总资产比例
    profit_loss: float
    profit_loss_pct: float
    stop_loss_price: float = 0
    take_profit_price: float = 0


class PositionSizer:
    """仓位计算器"""

    def __init__(
        self,
        method: str = "equal_weight",
        max_position_ratio: float = 0.3,
        risk_per_trade: float = 0.02
    ):
        """
        初始化仓位计算器

        Args:
            method: 仓位计算方法
                - equal_weight: 等权重
                - risk_parity: 风险平价
                - kelly: 凯利公式
                - fixed_ratio: 固定比例
                - atr_based: ATR动态调整
            max_position_ratio: 单只股票最大仓位
            risk_per_trade: 每笔交易风险比例
        """
        self.method = method
        self.max_position_ratio = max_position_ratio
        self.risk_per_trade = risk_per_trade

    def calculate(
        self,
        total_capital: float,
        price: float,
        signal_strength: float = 1.0,
        volatility: float = 0.02,
        win_rate: float = 0.5,
        win_loss_ratio: float = 2.0,
        atr: float = None
    ) -> int:
        """
        计算建议仓位

        Args:
            total_capital: 总资金
            price: 当前价格
            signal_strength: 信号强度 (0-1)
            volatility: 波动率
            win_rate: 胜率
            win_loss_ratio: 盈亏比
            atr: ATR值

        Returns:
            建议买入数量（股）
        """
        if self.method == "equal_weight":
            position_value = self._equal_weight(total_capital, signal_strength)

        elif self.method == "risk_parity":
            position_value = self._risk_parity(total_capital, volatility, signal_strength)

        elif self.method == "kelly":
            position_value = self._kelly(total_capital, win_rate, win_loss_ratio, signal_strength)

        elif self.method == "fixed_ratio":
            position_value = self._fixed_ratio(total_capital, signal_strength)

        elif self.method == "atr_based":
            position_value = self._atr_based(total_capital, price, atr, signal_strength)

        else:
            position_value = self._equal_weight(total_capital, signal_strength)

        # 不超过最大仓位限制
        max_value = total_capital * self.max_position_ratio
        position_value = min(position_value, max_value)

        # 计算股数（A股100股整数倍）
        shares = int(position_value / price / 100) * 100
        return max(shares, 0)

    def _equal_weight(self, total_capital: float, signal_strength: float) -> float:
        """等权重方法"""
        base_weight = 0.1  # 基础10%仓位
        return total_capital * base_weight * signal_strength

    def _risk_parity(self, total_capital: float, volatility: float, signal_strength: float) -> float:
        """风险平价方法"""
        target_risk = self.risk_per_trade
        if volatility <= 0:
            volatility = 0.02
        position_ratio = target_risk / volatility
        return total_capital * min(position_ratio, self.max_position_ratio) * signal_strength

    def _kelly(self, total_capital: float, win_rate: float, win_loss_ratio: float, signal_strength: float) -> float:
        """凯利公式"""
        # f = (bp - q) / b
        # f: 仓位比例
        # b: 盈亏比
        # p: 胜率
        # q: 负率 = 1 - p
        q = 1 - win_rate
        kelly_ratio = (win_loss_ratio * win_rate - q) / win_loss_ratio
        # 使用半凯利以降低风险
        kelly_ratio = max(0, kelly_ratio) * 0.5
        return total_capital * min(kelly_ratio, self.max_position_ratio) * signal_strength

    def _fixed_ratio(self, total_capital: float, signal_strength: float) -> float:
        """固定比例方法"""
        return total_capital * self.risk_per_trade * signal_strength

    def _atr_based(self, total_capital: float, price: float, atr: float, signal_strength: float) -> float:
        """ATR动态调整方法"""
        if atr is None or atr <= 0:
            return self._equal_weight(total_capital, signal_strength)

        # 风险金额
        risk_amount = total_capital * self.risk_per_trade
        # ATR倍数作为止损距离
        stop_distance = atr * 2
        # 计算仓位
        shares = risk_amount / stop_distance
        return shares * price * signal_strength


class PositionManager:
    """仓位管理器"""

    def __init__(
        self,
        max_positions: int = 10,
        max_position_ratio: float = 0.3,
        max_sector_ratio: float = 0.4,
        rebalance_threshold: float = 0.1
    ):
        """
        初始化仓位管理器

        Args:
            max_positions: 最大持仓股票数
            max_position_ratio: 单只股票最大仓位比例
            max_sector_ratio: 单个行业最大仓位比例
            rebalance_threshold: 再平衡阈值
        """
        self.max_positions = max_positions
        self.max_position_ratio = max_position_ratio
        self.max_sector_ratio = max_sector_ratio
        self.rebalance_threshold = rebalance_threshold

        self._positions: Dict[str, PositionInfo] = {}
        self._sector_mapping: Dict[str, str] = {}  # code -> sector

    @property
    def positions(self) -> Dict[str, PositionInfo]:
        return self._positions.copy()

    @property
    def position_count(self) -> int:
        return len(self._positions)

    def add_position(
        self,
        code: str,
        market: str,
        quantity: int,
        price: float,
        total_capital: float,
        sector: str = ""
    ):
        """添加持仓"""
        if code in self._positions:
            # 加仓
            pos = self._positions[code]
            new_quantity = pos.quantity + quantity
            pos.avg_cost = (pos.avg_cost * pos.quantity + price * quantity) / new_quantity
            pos.quantity = new_quantity
        else:
            if self.position_count >= self.max_positions:
                logger.warning(f"持仓数已达上限 {self.max_positions}")
                return False

            self._positions[code] = PositionInfo(
                code=code,
                market=market,
                quantity=quantity,
                avg_cost=price,
                current_price=price,
                market_value=quantity * price,
                weight=quantity * price / total_capital,
                profit_loss=0,
                profit_loss_pct=0
            )

        if sector:
            self._sector_mapping[code] = sector

        return True

    def remove_position(self, code: str, quantity: int = None):
        """移除持仓"""
        if code not in self._positions:
            return False

        pos = self._positions[code]
        if quantity is None or quantity >= pos.quantity:
            del self._positions[code]
            if code in self._sector_mapping:
                del self._sector_mapping[code]
        else:
            pos.quantity -= quantity

        return True

    def update_prices(self, prices: Dict[str, float], total_capital: float):
        """更新持仓价格"""
        for code, pos in self._positions.items():
            if code in prices:
                pos.current_price = prices[code]
                pos.market_value = pos.quantity * pos.current_price
                pos.weight = pos.market_value / total_capital if total_capital > 0 else 0
                pos.profit_loss = (pos.current_price - pos.avg_cost) * pos.quantity
                pos.profit_loss_pct = (pos.current_price - pos.avg_cost) / pos.avg_cost if pos.avg_cost > 0 else 0

    def check_position_limit(self, code: str, add_value: float, total_capital: float) -> bool:
        """检查是否超过仓位限制"""
        current_value = self._positions.get(code, PositionInfo(
            code=code, market="", quantity=0, avg_cost=0,
            current_price=0, market_value=0, weight=0,
            profit_loss=0, profit_loss_pct=0
        )).market_value

        new_ratio = (current_value + add_value) / total_capital
        return new_ratio <= self.max_position_ratio

    def check_sector_limit(self, code: str, add_value: float, total_capital: float) -> bool:
        """检查是否超过行业限制"""
        sector = self._sector_mapping.get(code)
        if not sector:
            return True

        sector_value = sum(
            pos.market_value for c, pos in self._positions.items()
            if self._sector_mapping.get(c) == sector
        )

        new_ratio = (sector_value + add_value) / total_capital
        return new_ratio <= self.max_sector_ratio

    def get_rebalance_trades(self, target_weights: Dict[str, float], total_capital: float) -> List[Dict]:
        """
        计算再平衡交易

        Args:
            target_weights: 目标权重 {code: weight}
            total_capital: 总资金

        Returns:
            需要执行的交易列表
        """
        trades = []

        # 计算当前权重
        current_weights = {code: pos.weight for code, pos in self._positions.items()}

        # 需要卖出的（当前持有但目标没有，或权重过高）
        for code, current_weight in current_weights.items():
            target_weight = target_weights.get(code, 0)
            if abs(current_weight - target_weight) > self.rebalance_threshold:
                diff = target_weight - current_weight
                if diff < 0:  # 需要卖出
                    pos = self._positions[code]
                    sell_value = abs(diff) * total_capital
                    sell_quantity = int(sell_value / pos.current_price / 100) * 100
                    if sell_quantity > 0:
                        trades.append({
                            'code': code,
                            'direction': 'SELL',
                            'quantity': min(sell_quantity, pos.quantity),
                            'price': pos.current_price,
                            'reason': f'再平衡：权重{current_weight:.2%} -> {target_weight:.2%}'
                        })

        # 需要买入的（目标有但当前没有，或权重过低）
        for code, target_weight in target_weights.items():
            current_weight = current_weights.get(code, 0)
            if abs(target_weight - current_weight) > self.rebalance_threshold:
                diff = target_weight - current_weight
                if diff > 0:  # 需要买入
                    buy_value = diff * total_capital
                    # 假设价格从外部获取，这里用占位符
                    trades.append({
                        'code': code,
                        'direction': 'BUY',
                        'value': buy_value,
                        'reason': f'再平衡：权重{current_weight:.2%} -> {target_weight:.2%}'
                    })

        return trades

    def get_summary(self) -> Dict[str, Any]:
        """获取持仓汇总"""
        total_market_value = sum(pos.market_value for pos in self._positions.values())
        total_profit_loss = sum(pos.profit_loss for pos in self._positions.values())
        total_cost = sum(pos.avg_cost * pos.quantity for pos in self._positions.values())

        return {
            'position_count': self.position_count,
            'total_market_value': total_market_value,
            'total_cost': total_cost,
            'total_profit_loss': total_profit_loss,
            'total_profit_loss_pct': total_profit_loss / total_cost if total_cost > 0 else 0,
            'positions': [
                {
                    'code': pos.code,
                    'quantity': pos.quantity,
                    'avg_cost': pos.avg_cost,
                    'current_price': pos.current_price,
                    'market_value': pos.market_value,
                    'weight': pos.weight,
                    'profit_loss': pos.profit_loss,
                    'profit_loss_pct': pos.profit_loss_pct
                }
                for pos in self._positions.values()
            ]
        }
