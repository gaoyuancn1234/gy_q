"""
券商接口基类
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable
from enum import Enum
import uuid
import logging

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """订单类型"""
    LIMIT = "LIMIT"           # 限价单
    MARKET = "MARKET"         # 市价单
    STOP = "STOP"             # 止损单
    STOP_LIMIT = "STOP_LIMIT" # 止损限价单


class OrderStatus(Enum):
    """订单状态"""
    PENDING = "PENDING"       # 待提交
    SUBMITTED = "SUBMITTED"   # 已提交
    PARTIAL = "PARTIAL"       # 部分成交
    FILLED = "FILLED"         # 完全成交
    CANCELLED = "CANCELLED"   # 已取消
    REJECTED = "REJECTED"     # 已拒绝
    EXPIRED = "EXPIRED"       # 已过期


class OrderDirection(Enum):
    """订单方向"""
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class Order:
    """订单"""
    order_id: str
    code: str
    direction: OrderDirection
    order_type: OrderType
    price: float
    quantity: int
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    filled_price: float = 0
    commission: float = 0
    stamp_duty: float = 0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    message: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_active(self) -> bool:
        """订单是否活跃"""
        return self.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIAL]

    @property
    def is_filled(self) -> bool:
        """订单是否完全成交"""
        return self.status == OrderStatus.FILLED

    @property
    def unfilled_quantity(self) -> int:
        """未成交数量"""
        return self.quantity - self.filled_quantity

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'order_id': self.order_id,
            'code': self.code,
            'direction': self.direction.value,
            'order_type': self.order_type.value,
            'price': self.price,
            'quantity': self.quantity,
            'status': self.status.value,
            'filled_quantity': self.filled_quantity,
            'filled_price': self.filled_price,
            'commission': self.commission,
            'stamp_duty': self.stamp_duty,
            'created_at': str(self.created_at),
            'updated_at': str(self.updated_at),
            'message': self.message
        }


@dataclass
class Position:
    """持仓"""
    code: str
    quantity: int
    available_quantity: int  # 可用数量（A股T+1）
    avg_cost: float
    current_price: float
    market_value: float
    profit_loss: float
    profit_loss_pct: float


@dataclass
class AccountInfo:
    """账户信息"""
    total_assets: float      # 总资产
    available_cash: float    # 可用资金
    frozen_cash: float       # 冻结资金
    market_value: float      # 持仓市值
    total_profit_loss: float # 总盈亏


class BaseBroker(ABC):
    """券商接口基类"""

    def __init__(self, name: str):
        self.name = name
        self._connected = False
        self._orders: Dict[str, Order] = {}
        self._callbacks: Dict[str, List[Callable]] = {
            'on_order_update': [],
            'on_trade': [],
            'on_error': []
        }

    @property
    def is_connected(self) -> bool:
        return self._connected

    @abstractmethod
    def connect(self) -> bool:
        """连接券商"""
        pass

    @abstractmethod
    def disconnect(self):
        """断开连接"""
        pass

    @abstractmethod
    def get_account_info(self) -> AccountInfo:
        """获取账户信息"""
        pass

    @abstractmethod
    def get_positions(self) -> List[Position]:
        """获取持仓列表"""
        pass

    @abstractmethod
    def get_orders(self, status: Optional[OrderStatus] = None) -> List[Order]:
        """获取订单列表"""
        pass

    @abstractmethod
    def submit_order(
        self,
        code: str,
        direction: OrderDirection,
        price: float,
        quantity: int,
        order_type: OrderType = OrderType.LIMIT
    ) -> Order:
        """
        提交订单

        Args:
            code: 股票代码
            direction: 买卖方向
            price: 价格
            quantity: 数量
            order_type: 订单类型

        Returns:
            订单对象
        """
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """取消订单"""
        pass

    def buy(
        self,
        code: str,
        price: float,
        quantity: int,
        order_type: OrderType = OrderType.LIMIT
    ) -> Order:
        """买入"""
        return self.submit_order(code, OrderDirection.BUY, price, quantity, order_type)

    def sell(
        self,
        code: str,
        price: float,
        quantity: int,
        order_type: OrderType = OrderType.LIMIT
    ) -> Order:
        """卖出"""
        return self.submit_order(code, OrderDirection.SELL, price, quantity, order_type)

    def get_order(self, order_id: str) -> Optional[Order]:
        """获取订单"""
        return self._orders.get(order_id)

    def _generate_order_id(self) -> str:
        """生成订单ID"""
        return str(uuid.uuid4())[:8].upper()

    def register_callback(self, event: str, callback: Callable):
        """注册回调"""
        if event in self._callbacks:
            self._callbacks[event].append(callback)

    def _trigger_callback(self, event: str, *args, **kwargs):
        """触发回调"""
        for callback in self._callbacks.get(event, []):
            try:
                callback(*args, **kwargs)
            except Exception as e:
                logger.error(f"回调执行失败: {e}")

    def _update_order(self, order: Order):
        """更新订单"""
        order.updated_at = datetime.now()
        self._orders[order.order_id] = order
        self._trigger_callback('on_order_update', order)

    def get_today_trades(self) -> List[Dict]:
        """获取今日成交"""
        return []

    def get_today_entrusts(self) -> List[Dict]:
        """获取今日委托"""
        return []

    @staticmethod
    def validate_order(
        code: str,
        direction: OrderDirection,
        price: float,
        quantity: int,
        available_cash: float = None,
        position: Position = None
    ) -> tuple:
        """
        验证订单

        Returns:
            (is_valid, error_message)
        """
        # 基本验证
        if not code:
            return False, "股票代码不能为空"

        if price <= 0:
            return False, "价格必须大于0"

        if quantity <= 0:
            return False, "数量必须大于0"

        # A股数量必须是100的倍数
        if quantity % 100 != 0:
            return False, "A股数量必须是100的倍数"

        # 买入检查
        if direction == OrderDirection.BUY:
            if available_cash is not None:
                required = price * quantity
                if required > available_cash:
                    return False, f"可用资金不足: 需要{required:.2f}, 可用{available_cash:.2f}"

        # 卖出检查
        if direction == OrderDirection.SELL:
            if position is not None:
                if quantity > position.available_quantity:
                    return False, f"可卖数量不足: 需要{quantity}, 可用{position.available_quantity}"

        return True, ""
