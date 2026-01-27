"""
风险控制模块
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """风险等级"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RiskAlert:
    """风险警报"""
    level: RiskLevel
    type: str
    message: str
    value: float
    threshold: float
    timestamp: datetime
    code: str = ""

    def __str__(self):
        return f"[{self.level.value.upper()}] {self.type}: {self.message}"


class RiskController:
    """风险控制器"""

    def __init__(
        self,
        max_position_ratio: float = 0.3,
        max_daily_loss: float = 0.05,
        max_drawdown: float = 0.2,
        stop_loss_ratio: float = 0.08,
        take_profit_ratio: float = 0.15,
        max_trade_amount: float = 100000,
        max_positions: int = 10,
        cooling_period: int = 3
    ):
        """
        初始化风险控制器

        Args:
            max_position_ratio: 单只股票最大仓位比例
            max_daily_loss: 单日最大亏损比例
            max_drawdown: 最大回撤比例
            stop_loss_ratio: 止损比例
            take_profit_ratio: 止盈比例
            max_trade_amount: 单笔最大交易金额
            max_positions: 最大持仓数
            cooling_period: 止损后冷却期（天）
        """
        self.max_position_ratio = max_position_ratio
        self.max_daily_loss = max_daily_loss
        self.max_drawdown = max_drawdown
        self.stop_loss_ratio = stop_loss_ratio
        self.take_profit_ratio = take_profit_ratio
        self.max_trade_amount = max_trade_amount
        self.max_positions = max_positions
        self.cooling_period = cooling_period

        # 状态追踪
        self._daily_pnl: float = 0
        self._peak_value: float = 0
        self._current_value: float = 0
        self._alerts: List[RiskAlert] = []
        self._cooling_stocks: Dict[str, datetime] = {}  # 冷却中的股票
        self._trading_allowed: bool = True

    @property
    def alerts(self) -> List[RiskAlert]:
        return self._alerts.copy()

    @property
    def is_trading_allowed(self) -> bool:
        return self._trading_allowed

    def reset_daily(self):
        """重置每日状态"""
        self._daily_pnl = 0
        self._alerts = []

    def update_value(self, current_value: float):
        """更新当前资产价值"""
        self._current_value = current_value
        if current_value > self._peak_value:
            self._peak_value = current_value

    def check_all(
        self,
        code: str,
        price: float,
        quantity: int,
        direction: str,
        total_capital: float,
        positions: Dict[str, Any]
    ) -> List[RiskAlert]:
        """
        执行所有风险检查

        Args:
            code: 股票代码
            price: 交易价格
            quantity: 交易数量
            direction: 交易方向 (BUY/SELL)
            total_capital: 总资金
            positions: 当前持仓

        Returns:
            风险警报列表
        """
        alerts = []

        if direction == "BUY":
            # 买入检查
            alerts.extend(self.check_position_limit(code, price, quantity, total_capital, positions))
            alerts.extend(self.check_trade_amount(price, quantity))
            alerts.extend(self.check_position_count(positions))
            alerts.extend(self.check_cooling(code))
            alerts.extend(self.check_drawdown(total_capital))
            alerts.extend(self.check_daily_loss())

        elif direction == "SELL":
            # 卖出检查
            if code in positions:
                alerts.extend(self.check_stop_loss(code, price, positions[code]))
                alerts.extend(self.check_take_profit(code, price, positions[code]))

        self._alerts.extend(alerts)
        return alerts

    def check_position_limit(
        self,
        code: str,
        price: float,
        quantity: int,
        total_capital: float,
        positions: Dict[str, Any]
    ) -> List[RiskAlert]:
        """检查仓位限制"""
        alerts = []
        trade_value = price * quantity

        # 现有仓位价值
        current_value = 0
        if code in positions:
            pos = positions[code]
            current_value = pos.get('current_price', pos.get('avg_cost', 0)) * pos.get('quantity', 0)

        # 检查单只股票仓位
        new_ratio = (current_value + trade_value) / total_capital
        if new_ratio > self.max_position_ratio:
            alerts.append(RiskAlert(
                level=RiskLevel.HIGH,
                type="POSITION_LIMIT",
                message=f"超过单只股票最大仓位限制: {new_ratio:.2%} > {self.max_position_ratio:.2%}",
                value=new_ratio,
                threshold=self.max_position_ratio,
                timestamp=datetime.now(),
                code=code
            ))

        return alerts

    def check_trade_amount(self, price: float, quantity: int) -> List[RiskAlert]:
        """检查单笔交易金额"""
        alerts = []
        trade_amount = price * quantity

        if trade_amount > self.max_trade_amount:
            alerts.append(RiskAlert(
                level=RiskLevel.MEDIUM,
                type="TRADE_AMOUNT",
                message=f"单笔交易金额过大: {trade_amount:.2f} > {self.max_trade_amount:.2f}",
                value=trade_amount,
                threshold=self.max_trade_amount,
                timestamp=datetime.now()
            ))

        return alerts

    def check_position_count(self, positions: Dict[str, Any]) -> List[RiskAlert]:
        """检查持仓数量"""
        alerts = []
        current_count = len(positions)

        if current_count >= self.max_positions:
            alerts.append(RiskAlert(
                level=RiskLevel.MEDIUM,
                type="POSITION_COUNT",
                message=f"持仓数已达上限: {current_count} >= {self.max_positions}",
                value=current_count,
                threshold=self.max_positions,
                timestamp=datetime.now()
            ))

        return alerts

    def check_daily_loss(self) -> List[RiskAlert]:
        """检查单日亏损"""
        alerts = []

        if self._daily_pnl < 0:
            loss_ratio = abs(self._daily_pnl) / self._current_value if self._current_value > 0 else 0
            if loss_ratio > self.max_daily_loss:
                alerts.append(RiskAlert(
                    level=RiskLevel.CRITICAL,
                    type="DAILY_LOSS",
                    message=f"单日亏损超限: {loss_ratio:.2%} > {self.max_daily_loss:.2%}，暂停交易",
                    value=loss_ratio,
                    threshold=self.max_daily_loss,
                    timestamp=datetime.now()
                ))
                self._trading_allowed = False

        return alerts

    def check_drawdown(self, current_value: float) -> List[RiskAlert]:
        """检查最大回撤"""
        alerts = []

        if self._peak_value > 0:
            drawdown = (self._peak_value - current_value) / self._peak_value
            if drawdown > self.max_drawdown:
                alerts.append(RiskAlert(
                    level=RiskLevel.CRITICAL,
                    type="MAX_DRAWDOWN",
                    message=f"最大回撤超限: {drawdown:.2%} > {self.max_drawdown:.2%}，暂停交易",
                    value=drawdown,
                    threshold=self.max_drawdown,
                    timestamp=datetime.now()
                ))
                self._trading_allowed = False

        return alerts

    def check_stop_loss(self, code: str, current_price: float, position: Dict[str, Any]) -> List[RiskAlert]:
        """检查止损"""
        alerts = []
        avg_cost = position.get('avg_cost', 0)

        if avg_cost > 0:
            loss_ratio = (avg_cost - current_price) / avg_cost
            if loss_ratio > self.stop_loss_ratio:
                alerts.append(RiskAlert(
                    level=RiskLevel.HIGH,
                    type="STOP_LOSS",
                    message=f"触发止损: 亏损{loss_ratio:.2%} > {self.stop_loss_ratio:.2%}，建议卖出",
                    value=loss_ratio,
                    threshold=self.stop_loss_ratio,
                    timestamp=datetime.now(),
                    code=code
                ))
                # 加入冷却期
                self._cooling_stocks[code] = datetime.now()

        return alerts

    def check_take_profit(self, code: str, current_price: float, position: Dict[str, Any]) -> List[RiskAlert]:
        """检查止盈"""
        alerts = []
        avg_cost = position.get('avg_cost', 0)

        if avg_cost > 0:
            profit_ratio = (current_price - avg_cost) / avg_cost
            if profit_ratio > self.take_profit_ratio:
                alerts.append(RiskAlert(
                    level=RiskLevel.LOW,
                    type="TAKE_PROFIT",
                    message=f"触发止盈: 盈利{profit_ratio:.2%} > {self.take_profit_ratio:.2%}，建议卖出",
                    value=profit_ratio,
                    threshold=self.take_profit_ratio,
                    timestamp=datetime.now(),
                    code=code
                ))

        return alerts

    def check_cooling(self, code: str) -> List[RiskAlert]:
        """检查冷却期"""
        alerts = []

        if code in self._cooling_stocks:
            cooling_start = self._cooling_stocks[code]
            days_passed = (datetime.now() - cooling_start).days

            if days_passed < self.cooling_period:
                alerts.append(RiskAlert(
                    level=RiskLevel.MEDIUM,
                    type="COOLING_PERIOD",
                    message=f"股票 {code} 在冷却期中，还剩 {self.cooling_period - days_passed} 天",
                    value=days_passed,
                    threshold=self.cooling_period,
                    timestamp=datetime.now(),
                    code=code
                ))
            else:
                # 冷却期结束
                del self._cooling_stocks[code]

        return alerts

    def calculate_stop_loss_price(self, entry_price: float, method: str = "fixed") -> float:
        """
        计算止损价格

        Args:
            entry_price: 入场价格
            method: 计算方法 (fixed, atr, trailing)

        Returns:
            止损价格
        """
        if method == "fixed":
            return entry_price * (1 - self.stop_loss_ratio)
        # 其他方法可以扩展
        return entry_price * (1 - self.stop_loss_ratio)

    def calculate_take_profit_price(self, entry_price: float, method: str = "fixed") -> float:
        """
        计算止盈价格

        Args:
            entry_price: 入场价格
            method: 计算方法 (fixed, atr, trailing)

        Returns:
            止盈价格
        """
        if method == "fixed":
            return entry_price * (1 + self.take_profit_ratio)
        return entry_price * (1 + self.take_profit_ratio)

    def update_daily_pnl(self, pnl: float):
        """更新每日盈亏"""
        self._daily_pnl += pnl

    def resume_trading(self):
        """恢复交易"""
        self._trading_allowed = True
        logger.info("交易已恢复")

    def get_risk_summary(self) -> Dict[str, Any]:
        """获取风险摘要"""
        drawdown = 0
        if self._peak_value > 0:
            drawdown = (self._peak_value - self._current_value) / self._peak_value

        return {
            'trading_allowed': self._trading_allowed,
            'current_value': self._current_value,
            'peak_value': self._peak_value,
            'drawdown': drawdown,
            'daily_pnl': self._daily_pnl,
            'cooling_stocks': list(self._cooling_stocks.keys()),
            'alert_count': len(self._alerts),
            'alerts': [str(a) for a in self._alerts[-10:]]  # 最近10条警报
        }


class TrailingStop:
    """跟踪止损"""

    def __init__(self, trailing_pct: float = 0.05, activation_pct: float = 0.03):
        """
        初始化跟踪止损

        Args:
            trailing_pct: 跟踪止损比例
            activation_pct: 激活比例（盈利多少后激活跟踪止损）
        """
        self.trailing_pct = trailing_pct
        self.activation_pct = activation_pct
        self._highest_prices: Dict[str, float] = {}
        self._entry_prices: Dict[str, float] = {}

    def add_position(self, code: str, entry_price: float):
        """添加持仓"""
        self._entry_prices[code] = entry_price
        self._highest_prices[code] = entry_price

    def update_price(self, code: str, current_price: float) -> Optional[RiskAlert]:
        """
        更新价格并检查是否触发止损

        Args:
            code: 股票代码
            current_price: 当前价格

        Returns:
            风险警报（如果触发止损）
        """
        if code not in self._entry_prices:
            return None

        entry_price = self._entry_prices[code]
        highest_price = self._highest_prices[code]

        # 更新最高价
        if current_price > highest_price:
            self._highest_prices[code] = current_price
            highest_price = current_price

        # 检查是否激活跟踪止损
        profit_pct = (highest_price - entry_price) / entry_price
        if profit_pct < self.activation_pct:
            return None

        # 计算跟踪止损价
        stop_price = highest_price * (1 - self.trailing_pct)

        # 检查是否触发
        if current_price < stop_price:
            return RiskAlert(
                level=RiskLevel.HIGH,
                type="TRAILING_STOP",
                message=f"触发跟踪止损: 当前价 {current_price:.2f} < 止损价 {stop_price:.2f}",
                value=current_price,
                threshold=stop_price,
                timestamp=datetime.now(),
                code=code
            )

        return None

    def remove_position(self, code: str):
        """移除持仓"""
        if code in self._entry_prices:
            del self._entry_prices[code]
        if code in self._highest_prices:
            del self._highest_prices[code]
