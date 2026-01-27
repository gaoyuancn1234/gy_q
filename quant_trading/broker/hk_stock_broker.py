"""
港股券商接口
支持富途、长桥、盈透等
"""

from datetime import datetime
from typing import List, Dict, Any, Optional
import logging

from .base_broker import (
    BaseBroker, Order, OrderType, OrderStatus, OrderDirection,
    Position, AccountInfo
)

logger = logging.getLogger(__name__)


class HKStockBroker(BaseBroker):
    """港股券商基类"""

    def __init__(self, name: str = "HKStockBroker"):
        super().__init__(name)
        self.market = "HK_STOCK"

    def _is_trading_time(self) -> bool:
        """检查是否在交易时间"""
        now = datetime.now()
        # 周末不交易
        if now.weekday() >= 5:
            return False

        current_time = now.time()
        morning_start = datetime.strptime("09:30", "%H:%M").time()
        morning_end = datetime.strptime("12:00", "%H:%M").time()
        afternoon_start = datetime.strptime("13:00", "%H:%M").time()
        afternoon_end = datetime.strptime("16:00", "%H:%M").time()

        return (morning_start <= current_time <= morning_end or
                afternoon_start <= current_time <= afternoon_end)


class FutuBroker(HKStockBroker):
    """
    富途证券接口
    支持港股、美股交易
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 11111,
        trade_env: str = "REAL",
        **kwargs
    ):
        """
        初始化富途接口

        Args:
            host: OpenD地址
            port: OpenD端口
            trade_env: 交易环境 (REAL/SIMULATE)
        """
        super().__init__("Futu")
        self.host = host
        self.port = port
        self.trade_env = trade_env
        self.config = kwargs
        self._trade_ctx = None
        self._quote_ctx = None

    def connect(self) -> bool:
        """连接富途OpenD"""
        try:
            from futu import OpenQuoteContext, OpenSecTradeContext, TrdEnv, TrdMarket

            # 连接行情
            self._quote_ctx = OpenQuoteContext(host=self.host, port=self.port)

            # 连接交易
            trd_env = TrdEnv.REAL if self.trade_env == "REAL" else TrdEnv.SIMULATE
            self._trade_ctx = OpenSecTradeContext(
                host=self.host,
                port=self.port,
                security_firm=self.config.get('security_firm'),
            )

            # 解锁交易
            if self.trade_env == "REAL":
                pwd = self.config.get('trade_password', '')
                if pwd:
                    ret, data = self._trade_ctx.unlock_trade(pwd)
                    if ret != 0:
                        logger.error(f"解锁交易失败: {data}")
                        return False

            self._connected = True
            logger.info(f"已连接到富途OpenD ({self.trade_env}环境)")
            return True

        except ImportError:
            logger.error("请安装 futu-api: pip install futu-api")
            return False
        except Exception as e:
            logger.error(f"连接富途失败: {e}")
            return False

    def disconnect(self):
        """断开连接"""
        if self._quote_ctx:
            self._quote_ctx.close()
        if self._trade_ctx:
            self._trade_ctx.close()
        self._connected = False
        logger.info("已断开富途连接")

    def get_account_info(self) -> AccountInfo:
        """获取账户信息"""
        if not self._connected:
            raise ConnectionError("未连接富途")

        from futu import TrdEnv, TrdMarket

        trd_env = TrdEnv.REAL if self.trade_env == "REAL" else TrdEnv.SIMULATE

        ret, data = self._trade_ctx.accinfo_query(trd_env=trd_env)
        if ret != 0:
            raise Exception(f"获取账户信息失败: {data}")

        if data.empty:
            raise Exception("账户信息为空")

        row = data.iloc[0]
        return AccountInfo(
            total_assets=row['total_assets'],
            available_cash=row['cash'],
            frozen_cash=row.get('frozen_cash', 0),
            market_value=row['market_val'],
            total_profit_loss=row.get('unrealized_pl', 0)
        )

    def get_positions(self) -> List[Position]:
        """获取持仓"""
        if not self._connected:
            raise ConnectionError("未连接富途")

        from futu import TrdEnv

        trd_env = TrdEnv.REAL if self.trade_env == "REAL" else TrdEnv.SIMULATE

        ret, data = self._trade_ctx.position_list_query(trd_env=trd_env)
        if ret != 0:
            raise Exception(f"获取持仓失败: {data}")

        positions = []
        for _, row in data.iterrows():
            code = row['code'].replace('HK.', '')
            positions.append(Position(
                code=code,
                quantity=int(row['qty']),
                available_quantity=int(row['can_sell_qty']),
                avg_cost=row['cost_price'],
                current_price=row['nominal_price'],
                market_value=row['market_val'],
                profit_loss=row['pl_val'],
                profit_loss_pct=row['pl_ratio']
            ))

        return positions

    def get_orders(self, status: Optional[OrderStatus] = None) -> List[Order]:
        """获取订单"""
        if not self._connected:
            raise ConnectionError("未连接富途")

        from futu import TrdEnv, OrderStatus as FutuOrderStatus

        trd_env = TrdEnv.REAL if self.trade_env == "REAL" else TrdEnv.SIMULATE

        ret, data = self._trade_ctx.order_list_query(trd_env=trd_env)
        if ret != 0:
            return []

        orders = []
        for _, row in data.iterrows():
            order_status = self._parse_futu_status(row['order_status'])

            if status and order_status != status:
                continue

            code = row['code'].replace('HK.', '')
            direction = OrderDirection.BUY if row['trd_side'] == 'BUY' else OrderDirection.SELL

            orders.append(Order(
                order_id=str(row['order_id']),
                code=code,
                direction=direction,
                order_type=OrderType.LIMIT,
                price=row['price'],
                quantity=int(row['qty']),
                status=order_status,
                filled_quantity=int(row['dealt_qty']),
                filled_price=row.get('dealt_avg_price', 0)
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
            raise ConnectionError("未连接富途")

        from futu import TrdEnv, TrdSide, OrderType as FutuOrderType

        trd_env = TrdEnv.REAL if self.trade_env == "REAL" else TrdEnv.SIMULATE
        futu_code = f"HK.{code}"

        # 方向
        trd_side = TrdSide.BUY if direction == OrderDirection.BUY else TrdSide.SELL

        # 订单类型
        if order_type == OrderType.MARKET:
            futu_order_type = FutuOrderType.MARKET
        else:
            futu_order_type = FutuOrderType.NORMAL

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
            ret, data = self._trade_ctx.place_order(
                price=price,
                qty=quantity,
                code=futu_code,
                trd_side=trd_side,
                order_type=futu_order_type,
                trd_env=trd_env
            )

            if ret == 0:
                order.order_id = str(data['order_id'].iloc[0])
                order.status = OrderStatus.SUBMITTED
                logger.info(f"富途订单提交成功: {order.order_id}")
            else:
                order.status = OrderStatus.REJECTED
                order.message = str(data)
                logger.error(f"富途订单提交失败: {data}")

        except Exception as e:
            order.status = OrderStatus.REJECTED
            order.message = str(e)
            logger.error(f"富途订单异常: {e}")

        self._update_order(order)
        return order

    def cancel_order(self, order_id: str) -> bool:
        """取消订单"""
        if not self._connected:
            raise ConnectionError("未连接富途")

        from futu import TrdEnv

        trd_env = TrdEnv.REAL if self.trade_env == "REAL" else TrdEnv.SIMULATE

        try:
            ret, data = self._trade_ctx.modify_order(
                modify_order_op=2,  # 取消
                order_id=int(order_id),
                qty=0,
                price=0,
                trd_env=trd_env
            )
            return ret == 0
        except Exception as e:
            logger.error(f"取消订单失败: {e}")
            return False

    def _parse_futu_status(self, status) -> OrderStatus:
        """解析富途订单状态"""
        status_map = {
            'UNSUBMITTED': OrderStatus.PENDING,
            'WAITING_SUBMIT': OrderStatus.PENDING,
            'SUBMITTING': OrderStatus.PENDING,
            'SUBMIT_FAILED': OrderStatus.REJECTED,
            'SUBMITTED': OrderStatus.SUBMITTED,
            'FILLED_PART': OrderStatus.PARTIAL,
            'FILLED_ALL': OrderStatus.FILLED,
            'CANCELLING_PART': OrderStatus.PARTIAL,
            'CANCELLING_ALL': OrderStatus.SUBMITTED,
            'CANCELLED_PART': OrderStatus.PARTIAL,
            'CANCELLED_ALL': OrderStatus.CANCELLED,
            'FAILED': OrderStatus.REJECTED,
            'DISABLED': OrderStatus.REJECTED,
            'DELETED': OrderStatus.CANCELLED,
        }
        return status_map.get(str(status), OrderStatus.PENDING)


class LongbridgeBroker(HKStockBroker):
    """
    长桥证券接口
    支持港股、美股交易
    """

    def __init__(
        self,
        app_key: str = "",
        app_secret: str = "",
        access_token: str = "",
        **kwargs
    ):
        """
        初始化长桥接口

        Args:
            app_key: App Key
            app_secret: App Secret
            access_token: Access Token
        """
        super().__init__("Longbridge")
        self.app_key = app_key
        self.app_secret = app_secret
        self.access_token = access_token
        self.config = kwargs
        self._trade_ctx = None
        self._quote_ctx = None

    def connect(self) -> bool:
        """连接长桥"""
        try:
            from longbridge.openapi import Config, TradeContext, QuoteContext

            # 创建配置
            config = Config(
                app_key=self.app_key,
                app_secret=self.app_secret,
                access_token=self.access_token
            )

            # 创建行情上下文
            self._quote_ctx = QuoteContext(config)

            # 创建交易上下文
            self._trade_ctx = TradeContext(config)

            self._connected = True
            logger.info("已连接到长桥证券")
            return True

        except ImportError:
            logger.error("请安装 longbridge: pip install longbridge")
            return False
        except Exception as e:
            logger.error(f"连接长桥失败: {e}")
            return False

    def disconnect(self):
        """断开连接"""
        self._connected = False
        self._trade_ctx = None
        self._quote_ctx = None
        logger.info("已断开长桥连接")

    def get_account_info(self) -> AccountInfo:
        """获取账户信息"""
        if not self._connected:
            raise ConnectionError("未连接长桥")

        resp = self._trade_ctx.account_balance()
        if resp:
            cash = resp[0]  # 第一个账户
            return AccountInfo(
                total_assets=float(cash.total_cash),
                available_cash=float(cash.cash_available),
                frozen_cash=float(cash.frozen_cash) if hasattr(cash, 'frozen_cash') else 0,
                market_value=float(cash.market_value) if hasattr(cash, 'market_value') else 0,
                total_profit_loss=0
            )
        raise Exception("获取账户信息失败")

    def get_positions(self) -> List[Position]:
        """获取持仓"""
        if not self._connected:
            raise ConnectionError("未连接长桥")

        positions = []
        resp = self._trade_ctx.stock_positions()

        for channel in resp:
            for pos in channel.positions:
                code = pos.symbol.replace('.HK', '')
                positions.append(Position(
                    code=code,
                    quantity=pos.quantity,
                    available_quantity=pos.available_quantity,
                    avg_cost=float(pos.cost_price),
                    current_price=float(pos.market_value) / pos.quantity if pos.quantity > 0 else 0,
                    market_value=float(pos.market_value),
                    profit_loss=float(pos.unrealized_pl) if hasattr(pos, 'unrealized_pl') else 0,
                    profit_loss_pct=float(pos.unrealized_pl_ratio) if hasattr(pos, 'unrealized_pl_ratio') else 0
                ))

        return positions

    def get_orders(self, status: Optional[OrderStatus] = None) -> List[Order]:
        """获取订单"""
        if not self._connected:
            raise ConnectionError("未连接长桥")

        from longbridge.openapi import OrderStatus as LBOrderStatus

        orders = []
        resp = self._trade_ctx.today_orders()

        for o in resp:
            order_status = self._parse_lb_status(o.status)
            if status and order_status != status:
                continue

            code = o.symbol.replace('.HK', '')
            direction = OrderDirection.BUY if o.side == 'Buy' else OrderDirection.SELL

            orders.append(Order(
                order_id=o.order_id,
                code=code,
                direction=direction,
                order_type=OrderType.LIMIT,
                price=float(o.price),
                quantity=int(o.quantity),
                status=order_status,
                filled_quantity=int(o.executed_quantity),
                filled_price=float(o.executed_price) if o.executed_price else 0
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
            raise ConnectionError("未连接长桥")

        from longbridge.openapi import OrderSide, OrderType as LBOrderType, TimeInForceType

        lb_code = f"{code}.HK"
        side = OrderSide.Buy if direction == OrderDirection.BUY else OrderSide.Sell

        if order_type == OrderType.MARKET:
            lb_order_type = LBOrderType.MO  # Market Order
        else:
            lb_order_type = LBOrderType.LO  # Limit Order

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
            resp = self._trade_ctx.submit_order(
                symbol=lb_code,
                order_type=lb_order_type,
                side=side,
                submitted_quantity=quantity,
                submitted_price=price,
                time_in_force=TimeInForceType.Day
            )

            order.order_id = resp.order_id
            order.status = OrderStatus.SUBMITTED
            logger.info(f"长桥订单提交成功: {order.order_id}")

        except Exception as e:
            order.status = OrderStatus.REJECTED
            order.message = str(e)
            logger.error(f"长桥订单失败: {e}")

        self._update_order(order)
        return order

    def cancel_order(self, order_id: str) -> bool:
        """取消订单"""
        if not self._connected:
            raise ConnectionError("未连接长桥")

        try:
            self._trade_ctx.cancel_order(order_id)
            return True
        except Exception as e:
            logger.error(f"取消订单失败: {e}")
            return False

    def _parse_lb_status(self, status) -> OrderStatus:
        """解析长桥订单状态"""
        status_map = {
            'NotReported': OrderStatus.PENDING,
            'ReplacedNotReported': OrderStatus.PENDING,
            'ProtectedNotReported': OrderStatus.PENDING,
            'VarietiesNotReported': OrderStatus.PENDING,
            'Filled': OrderStatus.FILLED,
            'WaitToNew': OrderStatus.PENDING,
            'New': OrderStatus.SUBMITTED,
            'WaitToReplace': OrderStatus.SUBMITTED,
            'PendingReplace': OrderStatus.SUBMITTED,
            'Replaced': OrderStatus.SUBMITTED,
            'PartialFilled': OrderStatus.PARTIAL,
            'WaitToCancel': OrderStatus.SUBMITTED,
            'PendingCancel': OrderStatus.SUBMITTED,
            'Rejected': OrderStatus.REJECTED,
            'Canceled': OrderStatus.CANCELLED,
            'Expired': OrderStatus.EXPIRED,
            'PartialWithdrawal': OrderStatus.PARTIAL,
        }
        return status_map.get(str(status), OrderStatus.PENDING)
