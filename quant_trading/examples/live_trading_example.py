#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实盘交易示例
演示如何连接真实券商进行交易

警告: 实盘交易涉及真实资金，请谨慎操作！
"""

import sys
sys.path.append('..')

from datetime import date, timedelta

# A股实盘交易示例
def a_stock_live_trading_example():
    """
    A股实盘交易示例

    需要安装: pip install easytrader
    需要: 券商客户端（如同花顺、通达信等）
    """
    print("=" * 50)
    print("A股实盘交易示例")
    print("=" * 50)

    from broker import EasyTraderBroker

    # 创建券商接口（以同花顺为例）
    broker = EasyTraderBroker(
        broker="universal_client",
        client_path=r"C:\同花顺\xiadan.exe"  # 修改为实际路径
    )

    # 连接（需要券商客户端已登录）
    # broker.connect()

    # 查询账户
    # account = broker.get_account_info()
    # print(f"总资产: {account.total_assets}")
    # print(f"可用资金: {account.available_cash}")

    # 查询持仓
    # positions = broker.get_positions()
    # for pos in positions:
    #     print(f"{pos.code}: {pos.quantity}股")

    # 下单示例（注释掉以防误操作）
    # order = broker.buy("000001", price=10.0, quantity=100)
    # print(f"委托编号: {order.order_id}")

    # 撤单
    # broker.cancel_order(order.order_id)

    print("\n注意: 实盘交易代码已注释，请根据实际情况修改后使用")


def hk_stock_live_trading_example():
    """
    港股实盘交易示例

    需要安装: pip install futu-api
    需要: 富途OpenD网关
    """
    print("=" * 50)
    print("港股实盘交易示例 (富途)")
    print("=" * 50)

    from broker import FutuBroker

    # 创建券商接口
    broker = FutuBroker(
        host="127.0.0.1",
        port=11111,
        trade_env="SIMULATE"  # 模拟环境，改为 "REAL" 进行实盘
    )

    # 连接
    # broker.connect()

    # 解锁交易（实盘需要）
    # broker._trade_ctx.unlock_trade("交易密码")

    # 查询账户
    # account = broker.get_account_info()
    # print(f"总资产: {account.total_assets}")

    # 下单示例（港股）
    # order = broker.buy("00700", price=300.0, quantity=100)  # 腾讯
    # print(f"委托编号: {order.order_id}")

    print("\n注意: 实盘交易代码已注释，请根据实际情况修改后使用")


def longbridge_live_trading_example():
    """
    港股实盘交易示例 (长桥)

    需要安装: pip install longbridge
    需要: 长桥开发者账号
    """
    print("=" * 50)
    print("港股实盘交易示例 (长桥)")
    print("=" * 50)

    from broker import LongbridgeBroker

    # 创建券商接口
    broker = LongbridgeBroker(
        app_key="your_app_key",
        app_secret="your_app_secret",
        access_token="your_access_token"
    )

    # 连接
    # broker.connect()

    # 查询账户
    # account = broker.get_account_info()
    # print(f"总资产: {account.total_assets}")

    print("\n注意: 实盘交易代码已注释，请根据实际情况修改后使用")


def automated_trading_system():
    """
    自动化交易系统示例

    展示完整的自动化交易流程
    """
    print("=" * 50)
    print("自动化交易系统示例")
    print("=" * 50)

    # 完整的自动化交易流程代码结构

    code = '''
# 1. 导入模块
from quant_trading import (
    Settings, TradingMode,
    AStockDataSource,
    MAStrategy,
    PaperTradingBroker,  # 使用模拟交易进行测试
    RiskController,
    Dashboard, AlertManager
)

# 2. 配置系统
settings = Settings()
settings.trading_mode = TradingMode.PAPER  # 先用模拟交易测试

# 3. 初始化组件
data_source = AStockDataSource()
data_source.connect()

broker = PaperTradingBroker(initial_cash=1000000)
broker.connect()

strategy = MAStrategy()
risk_controller = RiskController()
alert_manager = AlertManager()
dashboard = Dashboard(broker, data_source, risk_controller)

# 4. 启动监控
alert_manager.start()
dashboard.start()

# 5. 主循环
codes = ['000001', '600000', '000002']  # 监控的股票

while True:
    # 检查是否在交易时间
    if not is_trading_time():
        time.sleep(60)
        continue

    # 获取实时行情
    quotes = data_source.get_realtime_quote(codes)

    for quote in quotes:
        # 获取历史数据
        df = data_source.get_kline(quote.code, ...)

        # 生成信号
        signals = strategy.generate_signals(df)

        for signal in signals:
            # 风险检查
            alerts = risk_controller.check_all(...)

            if signal.type == 'BUY' and not alerts:
                broker.buy(signal.code, signal.price, quantity)
                alert_manager.info("买入", f"{signal.code}")

            elif signal.type == 'SELL':
                broker.sell(signal.code, signal.price, quantity)
                alert_manager.info("卖出", f"{signal.code}")

    # 更新仪表板
    dashboard.update()

    time.sleep(60)  # 每分钟检查一次

# 6. 清理
dashboard.stop()
alert_manager.stop()
broker.disconnect()
data_source.disconnect()
'''

    print(code)
    print("\n以上是自动化交易系统的代码结构，请根据实际需求修改使用")


if __name__ == '__main__':
    print("实盘交易示例\n")
    print("请选择示例:")
    print("1. A股实盘交易 (easytrader)")
    print("2. 港股实盘交易 (富途)")
    print("3. 港股实盘交易 (长桥)")
    print("4. 自动化交易系统")

    choice = input("\n请输入选项 (1-4): ").strip()

    if choice == '1':
        a_stock_live_trading_example()
    elif choice == '2':
        hk_stock_live_trading_example()
    elif choice == '3':
        longbridge_live_trading_example()
    elif choice == '4':
        automated_trading_system()
    else:
        print("无效选项")
