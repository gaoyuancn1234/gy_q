"""
模拟交易模块
用于策略测试和验证
"""

from datetime import datetime
from typing import List, Dict, Any, Optional
import logging
import random

from .base_broker import (
    BaseBroker, Order, OrderType, OrderStatus, OrderDirection,
    Position, AccountInfo
)

logger = logging.getLogger(__name__)


class PaperTradingBroker(BaseBroker):
    """
    模拟交易券商
    完全模拟真实交易环境，但不进行实际交易
    """

    def __init__(
        self,
        initial_cash: float = 1000000,
        commission_rate: float = 0.0003,
        stamp_duty_rate: float = 0.001,
        min_commission: float = 5.0,
        slippage: float = 0.001,
        market: str = "A_STOCK"
    ):
        """
        初始化模拟交易

        Args:
            initial_cash: 初始资金
            commission_rate: 佣金率
            stamp_duty_rate: 印花税率
            min_commission: 最低佣金
            slippage: 滑点
            market: 市场类型 (A_STOCK/HK_STOCK)
        """
        super().__init__("PaperTrading")
        self.initial_cash = initial_cash
        self.commission_rate = commission_rate
        self.stamp_duty_rate = stamp_duty_rate
        self.min_commission = min_commission
        self.slippage = slippage
        self.market = market

        # 账户状态
        self._cash = initial_cash
        self._positions: Dict[str, Dict] = {}
        self._trades: List[Dict] = []
        self._current_prices: Dict[str, float] = {}

    def connect(self) -> bool:
        """连接（模拟）"""
        self._connected = True
        logger.info("模拟交易已启动")
        return True

    def disconnect(self):
        """断开连接（模拟）"""
        self._connected = False
        logger.info("模拟交易已停止")

    def reset(self):
        """重置账户"""
        self._cash = self.initial_cash
        self._positions = {}
        self._orders = {}
        self._trades = []
        logger.info("模拟账户已重置")

    def update_prices(self, prices: Dict[str, float]):
        """
        更新股票价格

        Args:
            prices: {股票代码: 当前价格}
        """
        self._current_prices.update(prices)

        # 更新持仓市值
        for code, pos in self._positions.items():
            if code in prices:
                pos['current_price'] = prices[code]
                pos['market_value'] = pos['quantity'] * prices[code]
                pos['profit_loss'] = (prices[code] - pos['avg_cost']) * pos['quantity']
                pos['profit_loss_pct'] = (prices[code] - pos['avg_cost']) / pos['avg_cost'] if pos['avg_cost'] > 0 else 0

    def get_account_info(self) -> AccountInfo:
        """获取账户信息"""
        market_value = sum(pos['market_value'] for pos in self._positions.values())
        total_profit_loss = sum(pos['profit_loss'] for pos in self._positions.values())

        return AccountInfo(
            total_assets=self._cash + market_value,
            available_cash=self._cash,
            frozen_cash=0,
            market_value=market_value,
            total_profit_loss=total_profit_loss
        )

    def get_positions(self) -> List[Position]:
        """获取持仓"""
        positions = []
        for code, pos in self._positions.items():
            if pos['quantity'] > 0:
                positions.append(Position(
                    code=code,
                    quantity=pos['quantity'],
                    available_quantity=pos.get('available_quantity', pos['quantity']),
                    avg_cost=pos['avg_cost'],
                    current_price=pos.get('current_price', pos['avg_cost']),
                    market_value=pos.get('market_value', pos['quantity'] * pos['avg_cost']),
                    profit_loss=pos.get('profit_loss', 0),
                    profit_loss_pct=pos.get('profit_loss_pct', 0)
                ))
        return positions

    def get_orders(self, status: Optional[OrderStatus] = None) -> List[Order]:
        """获取订单"""
        orders = list(self._orders.values())
        if status:
            orders = [o for o in orders if o.status == status]
        return orders

    def submit_order(
        self,
        code: str,
        direction: OrderDirection,
        price: float,
        quantity: int,
        order_type: OrderType = OrderType.LIMIT
    ) -> Order:
        """提交订单"""
        # 创建订单
        order = Order(
            order_id=self._generate_order_id(),
            code=code,
            direction=direction,
            order_type=order_type,
            price=price,
            quantity=quantity,
            status=OrderStatus.PENDING
        )

        # 验证订单
        if direction == OrderDirection.BUY:
            is_valid, msg = self.validate_order(
                code, direction, price, quantity,
                available_cash=self._cash
            )
        else:
            position = self._positions.get(code)
            is_valid, msg = self.validate_order(
                code, direction, price, quantity,
                position=Position(
                    code=code,
                    quantity=position['quantity'] if position else 0,
                    available_quantity=position.get('available_quantity', 0) if position else 0,
                    avg_cost=0, current_price=0, market_value=0,
                    profit_loss=0, profit_loss_pct=0
                ) if position else None
            )

        if not is_valid:
            order.status = OrderStatus.REJECTED
            order.message = msg
            self._update_order(order)
            return order

        # 模拟成交（假设立即成交）
        self._execute_order(order)

        return order

    def _execute_order(self, order: Order):
        """执行订单"""
        # 计算成交价格（考虑滑点）
        if order.direction == OrderDirection.BUY:
            exec_price = order.price * (1 + self.slippage)
        else:
            exec_price = order.price * (1 - self.slippage)

        # 计算费用
        amount = exec_price * order.quantity
        commission = max(amount * self.commission_rate, self.min_commission)

        if order.direction == OrderDirection.SELL:
            stamp_duty = amount * self.stamp_duty_rate
        else:
            stamp_duty = 0

        total_cost = amount + commission + stamp_duty

        # 检查资金（买入）
        if order.direction == OrderDirection.BUY:
            if total_cost > self._cash:
                order.status = OrderStatus.REJECTED
                order.message = "资金不足"
                self._update_order(order)
                return

        # 执行成交
        if order.direction == OrderDirection.BUY:
            self._cash -= total_cost

            if order.code in self._positions:
                pos = self._positions[order.code]
                new_qty = pos['quantity'] + order.quantity
                pos['avg_cost'] = (pos['avg_cost'] * pos['quantity'] + exec_price * order.quantity) / new_qty
                pos['quantity'] = new_qty
            else:
                self._positions[order.code] = {
                    'quantity': order.quantity,
                    'available_quantity': 0 if self.market == "A_STOCK" else order.quantity,  # A股T+1
                    'avg_cost': exec_price,
                    'current_price': exec_price,
                    'market_value': exec_price * order.quantity,
                    'profit_loss': 0,
                    'profit_loss_pct': 0
                }

        else:  # SELL
            net_amount = amount - commission - stamp_duty
            self._cash += net_amount

            if order.code in self._positions:
                pos = self._positions[order.code]
                pos['quantity'] -= order.quantity
                pos['available_quantity'] = max(0, pos['available_quantity'] - order.quantity)

                if pos['quantity'] <= 0:
                    del self._positions[order.code]

        # 更新订单状态
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.filled_price = exec_price
        order.commission = commission
        order.stamp_duty = stamp_duty
        self._update_order(order)

        # 记录成交
        self._trades.append({
            'order_id': order.order_id,
            'code': order.code,
            'direction': order.direction.value,
            'price': exec_price,
            'quantity': order.quantity,
            'amount': amount,
            'commission': commission,
            'stamp_duty': stamp_duty,
            'timestamp': datetime.now()
        })

        logger.info(f"模拟成交: {order.direction.value} {order.code} {order.quantity}股 @ {exec_price:.2f}")

    def cancel_order(self, order_id: str) -> bool:
        """取消订单"""
        order = self._orders.get(order_id)
        if order and order.is_active:
            order.status = OrderStatus.CANCELLED
            self._update_order(order)
            return True
        return False

    def end_of_day(self):
        """日终处理"""
        # A股T+1：当日买入的股票变为可卖
        if self.market == "A_STOCK":
            for pos in self._positions.values():
                pos['available_quantity'] = pos['quantity']

        logger.info("日终处理完成")

    def get_trades(self) -> List[Dict]:
        """获取成交记录"""
        return self._trades.copy()

    def get_summary(self) -> Dict[str, Any]:
        """获取账户摘要"""
        account = self.get_account_info()
        return {
            'total_assets': account.total_assets,
            'available_cash': account.available_cash,
            'market_value': account.market_value,
            'total_profit_loss': account.total_profit_loss,
            'total_return': (account.total_assets - self.initial_cash) / self.initial_cash,
            'position_count': len(self._positions),
            'trade_count': len(self._trades),
            'positions': [
                {
                    'code': pos.code,
                    'quantity': pos.quantity,
                    'avg_cost': pos.avg_cost,
                    'current_price': pos.current_price,
                    'profit_loss': pos.profit_loss,
                    'profit_loss_pct': pos.profit_loss_pct
                }
                for pos in self.get_positions()
            ]
        }
