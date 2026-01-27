"""
监控仪表板
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import threading
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class DashboardData:
    """仪表板数据"""
    # 账户信息
    total_assets: float = 0
    available_cash: float = 0
    market_value: float = 0
    total_profit_loss: float = 0
    total_return: float = 0
    daily_return: float = 0

    # 持仓信息
    position_count: int = 0
    positions: List[Dict] = field(default_factory=list)

    # 交易信息
    today_trades: int = 0
    today_volume: float = 0
    today_commission: float = 0

    # 风险指标
    max_drawdown: float = 0
    current_drawdown: float = 0
    sharpe_ratio: float = 0

    # 策略信号
    active_signals: List[Dict] = field(default_factory=list)

    # 系统状态
    broker_connected: bool = False
    data_source_connected: bool = False
    last_update: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'account': {
                'total_assets': self.total_assets,
                'available_cash': self.available_cash,
                'market_value': self.market_value,
                'total_profit_loss': self.total_profit_loss,
                'total_return': self.total_return,
                'daily_return': self.daily_return
            },
            'positions': {
                'count': self.position_count,
                'details': self.positions
            },
            'trading': {
                'today_trades': self.today_trades,
                'today_volume': self.today_volume,
                'today_commission': self.today_commission
            },
            'risk': {
                'max_drawdown': self.max_drawdown,
                'current_drawdown': self.current_drawdown,
                'sharpe_ratio': self.sharpe_ratio
            },
            'signals': self.active_signals,
            'system': {
                'broker_connected': self.broker_connected,
                'data_source_connected': self.data_source_connected,
                'last_update': str(self.last_update)
            }
        }


class Dashboard:
    """
    交易监控仪表板
    提供实时账户和交易监控
    """

    def __init__(
        self,
        broker=None,
        data_source=None,
        risk_controller=None,
        update_interval: int = 5
    ):
        """
        初始化仪表板

        Args:
            broker: 券商接口
            data_source: 数据源
            risk_controller: 风险控制器
            update_interval: 更新间隔（秒）
        """
        self.broker = broker
        self.data_source = data_source
        self.risk_controller = risk_controller
        self.update_interval = update_interval

        self._data = DashboardData()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._callbacks: List = []

        # 历史数据
        self._equity_history: List[Dict] = []
        self._peak_value: float = 0
        self._start_value: float = 0
        self._yesterday_value: float = 0

    @property
    def data(self) -> DashboardData:
        """获取当前数据"""
        return self._data

    def start(self):
        """启动监控"""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._update_loop, daemon=True)
        self._thread.start()
        logger.info("监控仪表板已启动")

    def stop(self):
        """停止监控"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("监控仪表板已停止")

    def _update_loop(self):
        """更新循环"""
        while self._running:
            try:
                self.update()
            except Exception as e:
                logger.error(f"更新仪表板失败: {e}")

            time.sleep(self.update_interval)

    def update(self):
        """更新数据"""
        # 更新系统状态
        self._data.broker_connected = self.broker.is_connected if self.broker else False
        self._data.data_source_connected = self.data_source.is_connected if self.data_source else False

        if self.broker and self.broker.is_connected:
            self._update_account()
            self._update_positions()
            self._update_trades()

        if self.risk_controller:
            self._update_risk()

        self._data.last_update = datetime.now()

        # 记录历史
        self._equity_history.append({
            'timestamp': datetime.now(),
            'total_assets': self._data.total_assets
        })

        # 保留最近30天数据
        cutoff = datetime.now() - timedelta(days=30)
        self._equity_history = [
            e for e in self._equity_history
            if e['timestamp'] > cutoff
        ]

        # 触发回调
        for callback in self._callbacks:
            try:
                callback(self._data)
            except Exception as e:
                logger.error(f"回调执行失败: {e}")

    def _update_account(self):
        """更新账户信息"""
        try:
            account = self.broker.get_account_info()
            self._data.total_assets = account.total_assets
            self._data.available_cash = account.available_cash
            self._data.market_value = account.market_value
            self._data.total_profit_loss = account.total_profit_loss

            # 计算收益率
            if self._start_value == 0:
                self._start_value = account.total_assets
            if self._yesterday_value == 0:
                self._yesterday_value = account.total_assets

            self._data.total_return = (account.total_assets - self._start_value) / self._start_value if self._start_value > 0 else 0
            self._data.daily_return = (account.total_assets - self._yesterday_value) / self._yesterday_value if self._yesterday_value > 0 else 0

            # 更新峰值
            if account.total_assets > self._peak_value:
                self._peak_value = account.total_assets

        except Exception as e:
            logger.error(f"更新账户信息失败: {e}")

    def _update_positions(self):
        """更新持仓信息"""
        try:
            positions = self.broker.get_positions()
            self._data.position_count = len(positions)
            self._data.positions = [
                {
                    'code': p.code,
                    'quantity': p.quantity,
                    'avg_cost': p.avg_cost,
                    'current_price': p.current_price,
                    'market_value': p.market_value,
                    'profit_loss': p.profit_loss,
                    'profit_loss_pct': p.profit_loss_pct
                }
                for p in positions
            ]
        except Exception as e:
            logger.error(f"更新持仓信息失败: {e}")

    def _update_trades(self):
        """更新交易信息"""
        try:
            trades = self.broker.get_today_trades()
            self._data.today_trades = len(trades)
            self._data.today_volume = sum(float(t.get('成交金额', t.get('amount', 0))) for t in trades)
            self._data.today_commission = sum(float(t.get('佣金', t.get('commission', 0))) for t in trades)
        except Exception as e:
            logger.error(f"更新交易信息失败: {e}")

    def _update_risk(self):
        """更新风险指标"""
        try:
            risk_summary = self.risk_controller.get_risk_summary()
            self._data.current_drawdown = risk_summary.get('drawdown', 0)

            # 计算最大回撤
            if self._peak_value > 0:
                self._data.max_drawdown = max(
                    self._data.max_drawdown,
                    (self._peak_value - self._data.total_assets) / self._peak_value
                )
        except Exception as e:
            logger.error(f"更新风险指标失败: {e}")

    def add_signal(self, signal: Dict):
        """添加策略信号"""
        self._data.active_signals.append({
            **signal,
            'timestamp': str(datetime.now())
        })
        # 保留最近50个信号
        self._data.active_signals = self._data.active_signals[-50:]

    def clear_signals(self):
        """清除信号"""
        self._data.active_signals = []

    def register_callback(self, callback):
        """注册更新回调"""
        self._callbacks.append(callback)

    def reset_daily(self):
        """每日重置"""
        self._yesterday_value = self._data.total_assets
        logger.info("仪表板每日数据已重置")

    def get_equity_history(self) -> List[Dict]:
        """获取权益历史"""
        return self._equity_history.copy()

    def print_summary(self):
        """打印摘要"""
        d = self._data
        print(f"""
========== 交易监控仪表板 ==========
更新时间: {d.last_update.strftime('%Y-%m-%d %H:%M:%S')}

【账户概览】
总资产: {d.total_assets:,.2f}
可用资金: {d.available_cash:,.2f}
持仓市值: {d.market_value:,.2f}
总盈亏: {d.total_profit_loss:,.2f}
总收益率: {d.total_return:.2%}
日收益率: {d.daily_return:.2%}

【持仓情况】
持仓数量: {d.position_count}
{'股票代码':^10} {'数量':^8} {'成本':^10} {'现价':^10} {'盈亏%':^8}
{'-'*50}""")

        for pos in d.positions[:10]:  # 显示前10个
            print(f"{pos['code']:^10} {pos['quantity']:^8} {pos['avg_cost']:^10.2f} {pos['current_price']:^10.2f} {pos['profit_loss_pct']:^8.2%}")

        print(f"""
【今日交易】
成交笔数: {d.today_trades}
成交金额: {d.today_volume:,.2f}
交易费用: {d.today_commission:,.2f}

【风险指标】
当前回撤: {d.current_drawdown:.2%}
最大回撤: {d.max_drawdown:.2%}

【系统状态】
券商连接: {'✓' if d.broker_connected else '✗'}
数据源连接: {'✓' if d.data_source_connected else '✗'}
====================================
""")
