#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模拟交易示例
演示如何使用模拟交易功能
"""

import sys
sys.path.append('..')

import time
from datetime import date, timedelta

from data import AStockDataSource
from strategy import MAStrategy
from broker import PaperTradingBroker
from broker.base_broker import OrderDirection
from risk import RiskController, PositionSizer
from monitor import Dashboard, AlertManager


def run_paper_trading_example():
    """运行模拟交易示例"""
    print("=" * 50)
    print("量化交易系统 - 模拟交易示例")
    print("=" * 50)

    # 1. 初始化组件
    print("\n[1] 初始化系统组件...")

    # 数据源
    data_source = AStockDataSource(source="akshare")
    data_source.connect()

    # 模拟券商
    broker = PaperTradingBroker(
        initial_cash=1000000,
        commission_rate=0.0003,
        stamp_duty_rate=0.001,
        market="A_STOCK"
    )
    broker.connect()

    # 策略
    strategy = MAStrategy(short_period=5, long_period=20)

    # 风险控制
    risk_controller = RiskController(
        max_position_ratio=0.2,
        stop_loss_ratio=0.08,
        take_profit_ratio=0.15
    )

    # 仓位计算器
    position_sizer = PositionSizer(
        method="equal_weight",
        max_position_ratio=0.2
    )

    # 告警管理
    alert_manager = AlertManager()
    alert_manager.start()

    print("  系统组件初始化完成")

    # 2. 获取股票数据
    print("\n[2] 获取股票数据...")
    codes = ['000001', '600000']

    # 获取历史数据用于策略分析
    end_date = date.today()
    start_date = end_date - timedelta(days=60)

    stock_data = {}
    for code in codes:
        from data.data_source import KLineType
        df = data_source.get_kline(code, KLineType.DAILY, start_date, end_date)
        if not df.empty:
            df['code'] = code
            stock_data[code] = df
            print(f"  {code}: {len(df)} 条数据")

    # 3. 生成交易信号
    print("\n[3] 生成交易信号...")
    for code, df in stock_data.items():
        signals = strategy.generate_signals(df)
        signals = strategy.filter_signals(signals)

        for signal in signals:
            print(f"  {signal}")

            # 风险检查
            account = broker.get_account_info()
            positions = {p.code: {'avg_cost': p.avg_cost, 'quantity': p.quantity}
                        for p in broker.get_positions()}

            if signal.signal_type.value == 'BUY':
                # 计算买入数量
                quantity = position_sizer.calculate(
                    account.available_cash,
                    signal.price,
                    signal.strength
                )

                if quantity > 0:
                    # 风险检查
                    alerts = risk_controller.check_all(
                        signal.code, signal.price, quantity, 'BUY',
                        account.total_assets, positions
                    )

                    if not any(a.level.value in ['high', 'critical'] for a in alerts):
                        # 执行买入
                        order = broker.buy(signal.code, signal.price, quantity)
                        print(f"    执行买入: {order.code} {order.quantity}股 @ {order.price}")
                        alert_manager.info("交易执行",
                            f"买入 {signal.code} {quantity}股 @ {signal.price}")

            elif signal.signal_type.value == 'SELL':
                if signal.code in positions:
                    pos = positions[signal.code]
                    order = broker.sell(signal.code, signal.price, pos['quantity'])
                    print(f"    执行卖出: {order.code} {order.quantity}股 @ {order.price}")
                    alert_manager.info("交易执行",
                        f"卖出 {signal.code} {pos['quantity']}股 @ {signal.price}")

    # 4. 显示账户状态
    print("\n[4] 账户状态")
    summary = broker.get_summary()
    print(f"  总资产: {summary['total_assets']:,.2f}")
    print(f"  可用资金: {summary['available_cash']:,.2f}")
    print(f"  持仓市值: {summary['market_value']:,.2f}")
    print(f"  总收益率: {summary['total_return']:.2%}")

    print("\n  持仓明细:")
    for pos in summary['positions']:
        print(f"    {pos['code']}: {pos['quantity']}股, "
              f"成本{pos['avg_cost']:.2f}, 盈亏{pos['profit_loss_pct']:.2%}")

    # 5. 显示交易记录
    print("\n[5] 交易记录")
    trades = broker.get_trades()
    for trade in trades:
        print(f"  {trade['direction']} {trade['code']} {trade['quantity']}股 "
              f"@ {trade['price']:.2f}, 费用: {trade['commission']:.2f}")

    # 清理
    alert_manager.stop()
    broker.disconnect()
    data_source.disconnect()

    print("\n模拟交易示例完成!")


if __name__ == '__main__':
    run_paper_trading_example()
