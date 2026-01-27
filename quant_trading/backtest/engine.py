"""
回测引擎
"""

import pandas as pd
import numpy as np
from datetime import date, datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import logging

from ..strategy.base_strategy import BaseStrategy, Signal, SignalType
from ..config.settings import TradingConfig

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """交易记录"""
    code: str
    direction: str  # BUY/SELL
    price: float
    quantity: int
    amount: float
    commission: float
    stamp_duty: float
    timestamp: datetime
    signal_reason: str = ""


@dataclass
class Position:
    """持仓信息"""
    code: str
    quantity: int
    avg_cost: float
    current_price: float = 0

    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price

    @property
    def profit_loss(self) -> float:
        return (self.current_price - self.avg_cost) * self.quantity

    @property
    def profit_loss_pct(self) -> float:
        if self.avg_cost == 0:
            return 0
        return (self.current_price - self.avg_cost) / self.avg_cost * 100


@dataclass
class BacktestResult:
    """回测结果"""
    strategy_name: str
    start_date: date
    end_date: date
    initial_capital: float
    final_capital: float
    trades: List[Trade] = field(default_factory=list)
    equity_curve: pd.DataFrame = field(default_factory=pd.DataFrame)

    # 绩效指标
    total_return: float = 0
    annual_return: float = 0
    max_drawdown: float = 0
    sharpe_ratio: float = 0
    sortino_ratio: float = 0
    calmar_ratio: float = 0
    win_rate: float = 0
    profit_factor: float = 0
    avg_trade_return: float = 0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'strategy_name': self.strategy_name,
            'start_date': str(self.start_date),
            'end_date': str(self.end_date),
            'initial_capital': self.initial_capital,
            'final_capital': self.final_capital,
            'total_return': self.total_return,
            'annual_return': self.annual_return,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'calmar_ratio': self.calmar_ratio,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades
        }

    def summary(self) -> str:
        """生成回测报告摘要"""
        return f"""
========== 回测报告 ==========
策略名称: {self.strategy_name}
回测周期: {self.start_date} ~ {self.end_date}
初始资金: {self.initial_capital:,.2f}
最终资金: {self.final_capital:,.2f}

====== 收益指标 ======
总收益率: {self.total_return:.2%}
年化收益率: {self.annual_return:.2%}
最大回撤: {self.max_drawdown:.2%}

====== 风险指标 ======
夏普比率: {self.sharpe_ratio:.2f}
索提诺比率: {self.sortino_ratio:.2f}
卡尔玛比率: {self.calmar_ratio:.2f}

====== 交易统计 ======
总交易次数: {self.total_trades}
盈利次数: {self.winning_trades}
亏损次数: {self.losing_trades}
胜率: {self.win_rate:.2%}
盈亏比: {self.profit_factor:.2f}
平均收益: {self.avg_trade_return:.2%}
==============================
"""


class BacktestEngine:
    """回测引擎"""

    def __init__(
        self,
        initial_capital: float = 1000000,
        commission_rate: float = 0.0003,
        stamp_duty_rate: float = 0.001,
        min_commission: float = 5.0,
        slippage: float = 0.001
    ):
        """
        初始化回测引擎

        Args:
            initial_capital: 初始资金
            commission_rate: 佣金率
            stamp_duty_rate: 印花税率（仅卖出）
            min_commission: 最低佣金
            slippage: 滑点
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.stamp_duty_rate = stamp_duty_rate
        self.min_commission = min_commission
        self.slippage = slippage

        # 回测状态
        self._cash = initial_capital
        self._positions: Dict[str, Position] = {}
        self._trades: List[Trade] = []
        self._equity_history: List[Dict] = []

    def reset(self):
        """重置回测状态"""
        self._cash = self.initial_capital
        self._positions = {}
        self._trades = []
        self._equity_history = []

    @property
    def cash(self) -> float:
        return self._cash

    @property
    def positions(self) -> Dict[str, Position]:
        return self._positions.copy()

    @property
    def total_value(self) -> float:
        """总资产"""
        market_value = sum(p.market_value for p in self._positions.values())
        return self._cash + market_value

    def _calculate_commission(self, amount: float) -> float:
        """计算佣金"""
        commission = amount * self.commission_rate
        return max(commission, self.min_commission)

    def _calculate_stamp_duty(self, amount: float) -> float:
        """计算印花税"""
        return amount * self.stamp_duty_rate

    def _execute_buy(self, code: str, price: float, quantity: int, timestamp: datetime, reason: str = ""):
        """执行买入"""
        # 考虑滑点
        exec_price = price * (1 + self.slippage)
        amount = exec_price * quantity
        commission = self._calculate_commission(amount)
        total_cost = amount + commission

        if total_cost > self._cash:
            # 资金不足，调整数量
            available_amount = self._cash - self.min_commission
            quantity = int(available_amount / exec_price / 100) * 100  # A股100股整数倍
            if quantity <= 0:
                logger.warning(f"资金不足，无法买入 {code}")
                return
            amount = exec_price * quantity
            commission = self._calculate_commission(amount)
            total_cost = amount + commission

        # 更新持仓
        if code in self._positions:
            pos = self._positions[code]
            new_quantity = pos.quantity + quantity
            pos.avg_cost = (pos.avg_cost * pos.quantity + exec_price * quantity) / new_quantity
            pos.quantity = new_quantity
        else:
            self._positions[code] = Position(
                code=code,
                quantity=quantity,
                avg_cost=exec_price,
                current_price=exec_price
            )

        # 更新资金
        self._cash -= total_cost

        # 记录交易
        self._trades.append(Trade(
            code=code,
            direction="BUY",
            price=exec_price,
            quantity=quantity,
            amount=amount,
            commission=commission,
            stamp_duty=0,
            timestamp=timestamp,
            signal_reason=reason
        ))

        logger.debug(f"买入 {code}: {quantity}股 @ {exec_price:.2f}, 佣金={commission:.2f}")

    def _execute_sell(self, code: str, price: float, quantity: int, timestamp: datetime, reason: str = ""):
        """执行卖出"""
        if code not in self._positions:
            logger.warning(f"没有持仓 {code}，无法卖出")
            return

        pos = self._positions[code]
        sell_quantity = min(quantity, pos.quantity)

        if sell_quantity <= 0:
            return

        # 考虑滑点
        exec_price = price * (1 - self.slippage)
        amount = exec_price * sell_quantity
        commission = self._calculate_commission(amount)
        stamp_duty = self._calculate_stamp_duty(amount)
        net_amount = amount - commission - stamp_duty

        # 更新持仓
        pos.quantity -= sell_quantity
        if pos.quantity == 0:
            del self._positions[code]

        # 更新资金
        self._cash += net_amount

        # 记录交易
        self._trades.append(Trade(
            code=code,
            direction="SELL",
            price=exec_price,
            quantity=sell_quantity,
            amount=amount,
            commission=commission,
            stamp_duty=stamp_duty,
            timestamp=timestamp,
            signal_reason=reason
        ))

        logger.debug(f"卖出 {code}: {sell_quantity}股 @ {exec_price:.2f}, 佣金={commission:.2f}, 印花税={stamp_duty:.2f}")

    def _update_positions_price(self, prices: Dict[str, float]):
        """更新持仓价格"""
        for code, pos in self._positions.items():
            if code in prices:
                pos.current_price = prices[code]

    def _record_equity(self, timestamp: datetime):
        """记录权益"""
        self._equity_history.append({
            'timestamp': timestamp,
            'cash': self._cash,
            'market_value': sum(p.market_value for p in self._positions.values()),
            'total_value': self.total_value
        })

    def run(
        self,
        strategy: BaseStrategy,
        data: pd.DataFrame,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> BacktestResult:
        """
        运行回测

        Args:
            strategy: 交易策略
            data: 行情数据，需要包含 code, date, open, high, low, close, volume 列
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            回测结果
        """
        self.reset()

        # 初始化策略
        strategy.initialize(self.initial_capital)

        # 过滤日期
        if 'date' not in data.columns:
            raise ValueError("数据必须包含 'date' 列")

        data = data.copy()
        data['date'] = pd.to_datetime(data['date'])

        if start_date:
            data = data[data['date'].dt.date >= start_date]
        if end_date:
            data = data[data['date'].dt.date <= end_date]

        if data.empty:
            raise ValueError("过滤后没有数据")

        # 按日期分组处理
        grouped = data.groupby('date')

        for dt, daily_data in grouped:
            timestamp = pd.to_datetime(dt)

            # 更新持仓价格
            prices = dict(zip(daily_data['code'], daily_data['close']))
            self._update_positions_price(prices)

            # 生成信号
            for code in daily_data['code'].unique():
                code_data = daily_data[daily_data['code'] == code]
                history = data[(data['code'] == code) & (data['date'] <= dt)]

                if len(history) < 20:  # 需要足够的历史数据
                    continue

                signals = strategy.generate_signals(history)
                signals = strategy.filter_signals(signals)

                # 执行信号
                for signal in signals:
                    if signal.signal_type == SignalType.BUY:
                        # 计算买入数量（简单等权重分配）
                        available = self._cash * 0.2  # 每只股票最多20%仓位
                        quantity = int(available / signal.price / 100) * 100
                        if quantity > 0:
                            self._execute_buy(
                                signal.code,
                                signal.price,
                                quantity,
                                timestamp,
                                signal.reason
                            )

                    elif signal.signal_type == SignalType.SELL:
                        if signal.code in self._positions:
                            pos = self._positions[signal.code]
                            self._execute_sell(
                                signal.code,
                                signal.price,
                                pos.quantity,  # 全部卖出
                                timestamp,
                                signal.reason
                            )

            # 记录权益
            self._record_equity(timestamp)

        # 计算绩效
        result = self._calculate_performance(strategy.name, data)
        return result

    def _calculate_performance(self, strategy_name: str, data: pd.DataFrame) -> BacktestResult:
        """计算绩效指标"""
        equity_df = pd.DataFrame(self._equity_history)

        if equity_df.empty:
            return BacktestResult(
                strategy_name=strategy_name,
                start_date=data['date'].min().date(),
                end_date=data['date'].max().date(),
                initial_capital=self.initial_capital,
                final_capital=self.initial_capital,
                trades=self._trades
            )

        # 基础指标
        final_capital = self.total_value
        total_return = (final_capital - self.initial_capital) / self.initial_capital

        # 计算日收益率
        equity_df['return'] = equity_df['total_value'].pct_change()

        # 年化收益率
        days = (equity_df['timestamp'].iloc[-1] - equity_df['timestamp'].iloc[0]).days
        annual_return = (1 + total_return) ** (365 / max(days, 1)) - 1 if days > 0 else 0

        # 最大回撤
        equity_df['peak'] = equity_df['total_value'].cummax()
        equity_df['drawdown'] = (equity_df['total_value'] - equity_df['peak']) / equity_df['peak']
        max_drawdown = abs(equity_df['drawdown'].min())

        # 夏普比率 (假设无风险利率为3%)
        risk_free_rate = 0.03
        daily_rf = risk_free_rate / 252
        excess_returns = equity_df['return'] - daily_rf
        sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() > 0 else 0

        # 索提诺比率
        downside_returns = excess_returns[excess_returns < 0]
        downside_std = downside_returns.std()
        sortino_ratio = excess_returns.mean() / downside_std * np.sqrt(252) if downside_std > 0 else 0

        # 卡尔玛比率
        calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0

        # 交易统计
        buy_trades = [t for t in self._trades if t.direction == "BUY"]
        sell_trades = [t for t in self._trades if t.direction == "SELL"]

        # 计算每笔交易的盈亏
        trade_returns = []
        for sell in sell_trades:
            # 找到对应的买入
            matching_buys = [t for t in buy_trades if t.code == sell.code and t.timestamp < sell.timestamp]
            if matching_buys:
                avg_buy_price = sum(t.price * t.quantity for t in matching_buys) / sum(t.quantity for t in matching_buys)
                trade_return = (sell.price - avg_buy_price) / avg_buy_price
                trade_returns.append(trade_return)

        winning_trades = len([r for r in trade_returns if r > 0])
        losing_trades = len([r for r in trade_returns if r <= 0])
        total_trades = len(trade_returns)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        # 盈亏比
        gross_profit = sum(r for r in trade_returns if r > 0)
        gross_loss = abs(sum(r for r in trade_returns if r < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        avg_trade_return = np.mean(trade_returns) if trade_returns else 0

        return BacktestResult(
            strategy_name=strategy_name,
            start_date=data['date'].min().date(),
            end_date=data['date'].max().date(),
            initial_capital=self.initial_capital,
            final_capital=final_capital,
            trades=self._trades,
            equity_curve=equity_df,
            total_return=total_return,
            annual_return=annual_return,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_trade_return=avg_trade_return,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades
        )

    def run_multi_stock(
        self,
        strategy: BaseStrategy,
        data_dict: Dict[str, pd.DataFrame],
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> BacktestResult:
        """
        多股票回测

        Args:
            strategy: 交易策略
            data_dict: 股票代码 -> 数据DataFrame 的字典
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            回测结果
        """
        # 合并所有股票数据
        all_data = []
        for code, df in data_dict.items():
            df = df.copy()
            df['code'] = code
            all_data.append(df)

        combined_data = pd.concat(all_data, ignore_index=True)
        return self.run(strategy, combined_data, start_date, end_date)
