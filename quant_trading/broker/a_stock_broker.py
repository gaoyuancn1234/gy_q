"""
A股券商接口
支持通过easytrader连接多家券商
"""

import time
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging

from .base_broker import (
    BaseBroker, Order, OrderType, OrderStatus, OrderDirection,
    Position, AccountInfo
)

logger = logging.getLogger(__name__)


class AStockBroker(BaseBroker):
    """A股券商基类"""

    def __init__(self, name: str = "AStockBroker"):
        super().__init__(name)
        self.market = "A_STOCK"

    def _is_trading_time(self) -> bool:
        """检查是否在交易时间"""
        now = datetime.now()
        # 周末不交易
        if now.weekday() >= 5:
            return False

        current_time = now.time()
        morning_start = datetime.strptime("09:30", "%H:%M").time()
        morning_end = datetime.strptime("11:30", "%H:%M").time()
        afternoon_start = datetime.strptime("13:00", "%H:%M").time()
        afternoon_end = datetime.strptime("15:00", "%H:%M").time()

        return (morning_start <= current_time <= morning_end or
                afternoon_start <= current_time <= afternoon_end)


class EasyTraderBroker(AStockBroker):
    """
    使用easytrader连接券商
    支持：华泰、银河、广发、海通等
    """

    def __init__(
        self,
        broker: str = "universal_client",
        client_path: str = "",
        **kwargs
    ):
        """
        初始化EasyTrader券商接口

        Args:
            broker: 券商类型 (ht, yh, gf, universal_client等)
            client_path: 客户端路径
            **kwargs: 其他配置
        """
        super().__init__(f"EasyTrader-{broker}")
        self.broker_type = broker
        self.client_path = client_path
        self.config = kwargs
        self._client = None

    def connect(self) -> bool:
        """连接券商客户端"""
        try:
            import easytrader

            # 创建交易对象
            self._client = easytrader.use(self.broker_type)

            # 根据券商类型进行不同的连接方式
            if self.broker_type == "universal_client":
                # 通用客户端（同花顺等）
                self._client.prepare(
                    user=self.config.get('user', ''),
                    password=self.config.get('password', ''),
                    exe_path=self.client_path
                )
            else:
                # 特定券商
                if self.client_path:
                    self._client.connect(self.client_path)

            self._connected = True
            logger.info(f"已连接到 {self.broker_type} 券商客户端")
            return True

        except ImportError:
            logger.error("请安装 easytrader: pip install easytrader")
            return False
        except Exception as e:
            logger.error(f"连接券商失败: {e}")
            return False

    def disconnect(self):
        """断开连接"""
        if self._client:
            try:
                self._client.exit()
            except Exception:
                pass
        self._connected = False
        self._client = None
        logger.info("已断开券商连接")

    def get_account_info(self) -> AccountInfo:
        """获取账户信息"""
        if not self._connected:
            raise ConnectionError("未连接券商")

        try:
            balance = self._client.balance
            return AccountInfo(
                total_assets=float(balance.get('总资产', balance.get('资产总额', 0))),
                available_cash=float(balance.get('可用金额', balance.get('可用资金', 0))),
                frozen_cash=float(balance.get('冻结金额', balance.get('冻结资金', 0))),
                market_value=float(balance.get('证券市值', balance.get('股票市值', 0))),
                total_profit_loss=float(balance.get('参考盈亏', balance.get('总盈亏', 0)))
            )
        except Exception as e:
            logger.error(f"获取账户信息失败: {e}")
            raise

    def get_positions(self) -> List[Position]:
        """获取持仓列表"""
        if not self._connected:
            raise ConnectionError("未连接券商")

        try:
            positions = []
            pos_list = self._client.position

            for pos in pos_list:
                code = str(pos.get('证券代码', pos.get('股票代码', '')))
                quantity = int(pos.get('股票余额', pos.get('持仓数量', 0)))
                available = int(pos.get('可用余额', pos.get('可卖数量', 0)))
                avg_cost = float(pos.get('成本价', pos.get('买入均价', 0)))
                current_price = float(pos.get('当前价', pos.get('市价', 0)))
                market_value = float(pos.get('市值', quantity * current_price))
                profit_loss = float(pos.get('盈亏', pos.get('浮动盈亏', 0)))

                positions.append(Position(
                    code=code,
                    quantity=quantity,
                    available_quantity=available,
                    avg_cost=avg_cost,
                    current_price=current_price,
                    market_value=market_value,
                    profit_loss=profit_loss,
                    profit_loss_pct=profit_loss / (avg_cost * quantity) if avg_cost * quantity > 0 else 0
                ))

            return positions

        except Exception as e:
            logger.error(f"获取持仓失败: {e}")
            raise

    def get_orders(self, status: Optional[OrderStatus] = None) -> List[Order]:
        """获取订单列表"""
        if not self._connected:
            raise ConnectionError("未连接券商")

        orders = []
        try:
            entrusts = self._client.today_entrusts

            for ent in entrusts:
                # 解析委托状态
                status_str = ent.get('委托状态', ent.get('状态', ''))
                order_status = self._parse_order_status(status_str)

                # 过滤状态
                if status and order_status != status:
                    continue

                direction = OrderDirection.BUY if ent.get('操作', '') in ['买入', '证券买入'] else OrderDirection.SELL

                order = Order(
                    order_id=str(ent.get('委托编号', ent.get('合同编号', ''))),
                    code=str(ent.get('证券代码', '')),
                    direction=direction,
                    order_type=OrderType.LIMIT,
                    price=float(ent.get('委托价格', 0)),
                    quantity=int(ent.get('委托数量', 0)),
                    status=order_status,
                    filled_quantity=int(ent.get('成交数量', 0)),
                    filled_price=float(ent.get('成交均价', ent.get('成交价格', 0)))
                )
                orders.append(order)

        except Exception as e:
            logger.error(f"获取订单失败: {e}")

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
        if not self._connected:
            raise ConnectionError("未连接券商")

        # 验证交易时间
        if not self._is_trading_time():
            logger.warning("当前不在交易时间")

        # 创建订单对象
        order = Order(
            order_id=self._generate_order_id(),
            code=code,
            direction=direction,
            order_type=order_type,
            price=price,
            quantity=quantity,
            status=OrderStatus.PENDING
        )

        try:
            # 根据订单类型执行
            if order_type == OrderType.MARKET:
                # 市价单：使用涨停价买入或跌停价卖出
                if direction == OrderDirection.BUY:
                    result = self._client.market_buy(code, quantity)
                else:
                    result = self._client.market_sell(code, quantity)
            else:
                # 限价单
                if direction == OrderDirection.BUY:
                    result = self._client.buy(code, price=price, amount=quantity)
                else:
                    result = self._client.sell(code, price=price, amount=quantity)

            # 解析结果
            if result:
                order.order_id = str(result.get('entrust_no', result.get('委托编号', order.order_id)))
                order.status = OrderStatus.SUBMITTED
                order.message = "委托成功"
                logger.info(f"订单提交成功: {order.order_id}")
            else:
                order.status = OrderStatus.REJECTED
                order.message = "委托失败"
                logger.error(f"订单提交失败: {code}")

        except Exception as e:
            order.status = OrderStatus.REJECTED
            order.message = str(e)
            logger.error(f"订单提交异常: {e}")

        self._update_order(order)
        return order

    def cancel_order(self, order_id: str) -> bool:
        """取消订单"""
        if not self._connected:
            raise ConnectionError("未连接券商")

        try:
            result = self._client.cancel_entrust(order_id)
            if result:
                order = self._orders.get(order_id)
                if order:
                    order.status = OrderStatus.CANCELLED
                    self._update_order(order)
                logger.info(f"订单已取消: {order_id}")
                return True
            return False

        except Exception as e:
            logger.error(f"取消订单失败: {e}")
            return False

    def _parse_order_status(self, status_str: str) -> OrderStatus:
        """解析订单状态"""
        status_map = {
            '已报': OrderStatus.SUBMITTED,
            '已成': OrderStatus.FILLED,
            '部成': OrderStatus.PARTIAL,
            '已撤': OrderStatus.CANCELLED,
            '废单': OrderStatus.REJECTED,
            '未报': OrderStatus.PENDING,
        }
        for key, value in status_map.items():
            if key in status_str:
                return value
        return OrderStatus.PENDING

    def get_today_trades(self) -> List[Dict]:
        """获取今日成交"""
        if not self._connected:
            return []

        try:
            return self._client.today_trades
        except Exception as e:
            logger.error(f"获取今日成交失败: {e}")
            return []

    def get_today_entrusts(self) -> List[Dict]:
        """获取今日委托"""
        if not self._connected:
            return []

        try:
            return self._client.today_entrusts
        except Exception as e:
            logger.error(f"获取今日委托失败: {e}")
            return []

    def auto_ipo(self) -> List[Dict]:
        """自动打新"""
        if not self._connected:
            return []

        try:
            return self._client.auto_ipo()
        except Exception as e:
            logger.error(f"自动打新失败: {e}")
            return []


class XtQuantBroker(AStockBroker):
    """
    迅投QMT量化交易接口
    适用于华泰、中信等支持QMT的券商
    """

    def __init__(self, account: str = "", session_id: int = 0):
        """
        初始化QMT接口

        Args:
            account: 资金账号
            session_id: 会话ID
        """
        super().__init__("XtQuant")
        self.account = account
        self.session_id = session_id
        self._xt = None

    def connect(self) -> bool:
        """连接QMT"""
        try:
            from xtquant import xttrader
            from xtquant.xttype import StockAccount

            # 创建交易对象
            self._xt = xttrader.XtQuantTrader(
                path='',  # MiniQMT路径
                session=self.session_id
            )

            # 启动交易线程
            self._xt.start()

            # 建立连接
            connect_result = self._xt.connect()
            if connect_result != 0:
                logger.error(f"QMT连接失败，错误码: {connect_result}")
                return False

            # 订阅账户
            self._stock_account = StockAccount(self.account)
            subscribe_result = self._xt.subscribe(self._stock_account)
            if subscribe_result != 0:
                logger.error(f"订阅账户失败，错误码: {subscribe_result}")
                return False

            self._connected = True
            logger.info("已连接到QMT")
            return True

        except ImportError:
            logger.error("请安装 xtquant: 从迅投官网下载")
            return False
        except Exception as e:
            logger.error(f"连接QMT失败: {e}")
            return False

    def disconnect(self):
        """断开QMT连接"""
        if self._xt:
            self._xt.stop()
        self._connected = False
        self._xt = None

    def get_account_info(self) -> AccountInfo:
        """获取账户信息"""
        if not self._connected:
            raise ConnectionError("未连接QMT")

        asset = self._xt.query_stock_asset(self._stock_account)
        return AccountInfo(
            total_assets=asset.total_asset,
            available_cash=asset.cash,
            frozen_cash=asset.frozen_cash,
            market_value=asset.market_value,
            total_profit_loss=0  # QMT需要单独计算
        )

    def get_positions(self) -> List[Position]:
        """获取持仓"""
        if not self._connected:
            raise ConnectionError("未连接QMT")

        positions = []
        pos_list = self._xt.query_stock_positions(self._stock_account)

        for pos in pos_list:
            positions.append(Position(
                code=pos.stock_code,
                quantity=pos.volume,
                available_quantity=pos.can_use_volume,
                avg_cost=pos.open_price,
                current_price=pos.market_value / pos.volume if pos.volume > 0 else 0,
                market_value=pos.market_value,
                profit_loss=pos.market_value - pos.open_price * pos.volume,
                profit_loss_pct=(pos.market_value - pos.open_price * pos.volume) / (pos.open_price * pos.volume) if pos.open_price * pos.volume > 0 else 0
            ))

        return positions

    def get_orders(self, status: Optional[OrderStatus] = None) -> List[Order]:
        """获取订单"""
        if not self._connected:
            raise ConnectionError("未连接QMT")

        orders = []
        order_list = self._xt.query_stock_orders(self._stock_account)

        for o in order_list:
            orders.append(Order(
                order_id=str(o.order_id),
                code=o.stock_code,
                direction=OrderDirection.BUY if o.order_type == 23 else OrderDirection.SELL,
                order_type=OrderType.LIMIT,
                price=o.price,
                quantity=o.order_volume,
                status=self._parse_xt_status(o.order_status),
                filled_quantity=o.traded_volume,
                filled_price=o.traded_price
            ))

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
        if not self._connected:
            raise ConnectionError("未连接QMT")

        from xtquant.xttype import StockAccount
        from xtquant import xtconstant

        # 订单类型
        if direction == OrderDirection.BUY:
            xt_type = xtconstant.STOCK_BUY
        else:
            xt_type = xtconstant.STOCK_SELL

        order = Order(
            order_id=self._generate_order_id(),
            code=code,
            direction=direction,
            order_type=order_type,
            price=price,
            quantity=quantity,
            status=OrderStatus.PENDING
        )

        try:
            order_id = self._xt.order_stock(
                self._stock_account,
                code,
                xt_type,
                quantity,
                xtconstant.FIX_PRICE,
                price
            )

            if order_id > 0:
                order.order_id = str(order_id)
                order.status = OrderStatus.SUBMITTED
            else:
                order.status = OrderStatus.REJECTED
                order.message = f"下单失败，错误码: {order_id}"

        except Exception as e:
            order.status = OrderStatus.REJECTED
            order.message = str(e)

        self._update_order(order)
        return order

    def cancel_order(self, order_id: str) -> bool:
        """取消订单"""
        if not self._connected:
            raise ConnectionError("未连接QMT")

        try:
            result = self._xt.cancel_order_stock(self._stock_account, int(order_id))
            return result == 0
        except Exception as e:
            logger.error(f"取消订单失败: {e}")
            return False

    def _parse_xt_status(self, status: int) -> OrderStatus:
        """解析QMT订单状态"""
        status_map = {
            48: OrderStatus.PENDING,
            49: OrderStatus.SUBMITTED,
            50: OrderStatus.PARTIAL,
            51: OrderStatus.FILLED,
            52: OrderStatus.CANCELLED,
            53: OrderStatus.CANCELLED,
            54: OrderStatus.REJECTED,
            56: OrderStatus.REJECTED,
        }
        return status_map.get(status, OrderStatus.PENDING)
