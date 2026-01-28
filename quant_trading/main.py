#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
量化交易系统主程序
支持回测、模拟交易和实盘交易
"""

import argparse
import sys
import time
import signal
from datetime import date, datetime, timedelta
from typing import Optional

from config import Settings, TradingMode
from data import AStockDataSource, HKStockDataSource, Database
from strategy import MAStrategy, MACDStrategy, RSIStrategy, BollingerBandStrategy
from backtest import BacktestEngine, PerformanceAnalyzer
from risk import RiskController, PositionManager
from broker import PaperTradingBroker, EasyTraderBroker, FutuBroker
from monitor import Dashboard, AlertManager
from utils import setup_logger, is_trading_time


class TradingSystem:
    """量化交易系统"""

    def __init__(self, settings: Settings = None):
        """初始化交易系统"""
        self.settings = settings or Settings()
        self.logger = setup_logger("quant_trading")

        # 组件
        self.data_source = None
        self.broker = None
        self.strategy = None
        self.risk_controller = None
        self.position_manager = None
        self.dashboard = None
        self.alert_manager = None
        self.database = None

        # 状态
        self._running = False

    def setup(self, market: str = "A_STOCK", strategy_name: str = "ma"):
        """
        初始化系统组件

        Args:
            market: 市场类型 (A_STOCK/HK_STOCK)
            strategy_name: 策略名称
        """
        self.logger.info(f"初始化交易系统: 市场={market}, 策略={strategy_name}")

        # 初始化数据库
        self.database = Database(self.settings.database.sqlite_path)

        # 初始化数据源
        if market == "A_STOCK":
            self.data_source = AStockDataSource(source="akshare")
        else:
            self.data_source = HKStockDataSource(source="akshare")

        if not self.data_source.connect():
            raise RuntimeError("无法连接数据源")

        # 初始化策略
        self.strategy = self._create_strategy(strategy_name)

        # 初始化风险控制
        self.risk_controller = RiskController(
            max_position_ratio=self.settings.risk.max_position_ratio,
            max_daily_loss=self.settings.risk.max_daily_loss,
            max_drawdown=self.settings.risk.max_drawdown,
            stop_loss_ratio=self.settings.risk.stop_loss_ratio,
            take_profit_ratio=self.settings.risk.take_profit_ratio,
            max_positions=self.settings.risk.max_positions
        )

        # 初始化仓位管理
        self.position_manager = PositionManager(
            max_positions=self.settings.risk.max_positions,
            max_position_ratio=self.settings.risk.max_position_ratio
        )

        # 初始化券商接口
        self._setup_broker(market)

        # 初始化监控
        self.alert_manager = AlertManager()
        self.alert_manager.start()

        self.dashboard = Dashboard(
            broker=self.broker,
            data_source=self.data_source,
            risk_controller=self.risk_controller
        )

        self.logger.info("系统初始化完成")

    def _create_strategy(self, name: str):
        """创建策略"""
        strategies = {
            'ma': MAStrategy(short_period=5, long_period=20),
            'macd': MACDStrategy(),
            'rsi': RSIStrategy(),
            'bollinger': BollingerBandStrategy()
        }
        return strategies.get(name, MAStrategy())

    def _setup_broker(self, market: str):
        """初始化券商接口"""
        if self.settings.trading_mode == TradingMode.BACKTEST:
            self.broker = None
        elif self.settings.trading_mode == TradingMode.PAPER:
            self.broker = PaperTradingBroker(
                initial_cash=1000000,
                market=market
            )
            self.broker.connect()
        elif self.settings.trading_mode == TradingMode.LIVE:
            if market == "A_STOCK":
                # 实盘A股需要配置券商
                self.broker = EasyTraderBroker(broker="universal_client")
            else:
                # 实盘港股需要配置富途
                self.broker = FutuBroker()
            # 实盘不自动连接，需要手动配置和连接

    def backtest(
        self,
        codes: list,
        start_date: date,
        end_date: date,
        initial_capital: float = 1000000
    ):
        """
        运行回测

        Args:
            codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            initial_capital: 初始资金
        """
        self.logger.info(f"开始回测: {codes}, {start_date} ~ {end_date}")

        # 获取数据
        all_data = []
        for code in codes:
            df = self.data_source.get_kline(
                code,
                kline_type=self.data_source.get_kline.__code__.co_varnames[2],
                start_date=start_date,
                end_date=end_date
            )
            if not df.empty:
                df['code'] = code
                all_data.append(df)

        if not all_data:
            self.logger.error("没有获取到数据")
            return None

        import pandas as pd
        combined_data = pd.concat(all_data, ignore_index=True)

        # 运行回测
        engine = BacktestEngine(
            initial_capital=initial_capital,
            commission_rate=self.settings.trading.a_stock_commission,
            stamp_duty_rate=self.settings.trading.a_stock_stamp_duty
        )

        result = engine.run(self.strategy, combined_data, start_date, end_date)

        # 分析结果
        analyzer = PerformanceAnalyzer()
        metrics = analyzer.analyze(result.equity_curve, result.trades)

        # 打印报告
        print(result.summary())

        # 保存结果
        self.database.save_backtest_result({
            'strategy_name': self.strategy.name,
            'start_date': start_date,
            'end_date': end_date,
            'initial_capital': initial_capital,
            'final_capital': result.final_capital,
            'total_return': result.total_return,
            'annual_return': result.annual_return,
            'max_drawdown': result.max_drawdown,
            'sharpe_ratio': result.sharpe_ratio,
            'win_rate': result.win_rate,
            'profit_factor': result.profit_factor,
            'total_trades': result.total_trades,
            'parameters': self.strategy.get_params()
        })

        return result

    def run_live(self, codes: list, interval: int = 60):
        """
        运行实时交易

        Args:
            codes: 股票代码列表
            interval: 检查间隔（秒）
        """
        if not self.broker or not self.broker.is_connected:
            self.logger.error("券商未连接")
            return

        self._running = True
        self.dashboard.start()

        # 注册信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        self.logger.info(f"开始实时交易: {codes}")
        self.alert_manager.info("交易系统", f"开始监控 {len(codes)} 只股票")

        while self._running:
            try:
                if not is_trading_time():
                    time.sleep(60)
                    continue

                # 获取实时行情
                quotes = self.data_source.get_realtime_quote(codes)

                # 更新持仓价格
                prices = {q.code: q.price for q in quotes}
                self.broker.update_prices(prices) if hasattr(self.broker, 'update_prices') else None

                # 检查风险
                account = self.broker.get_account_info()
                positions = {p.code: {'avg_cost': p.avg_cost, 'quantity': p.quantity}
                            for p in self.broker.get_positions()}

                self.risk_controller.update_value(account.total_assets)

                # 生成信号并执行
                for quote in quotes:
                    self._process_quote(quote, account, positions)

                # 更新仪表板
                self.dashboard.update()

                time.sleep(interval)

            except Exception as e:
                self.logger.error(f"交易循环异常: {e}")
                self.alert_manager.error("系统异常", str(e))
                time.sleep(10)

        self.shutdown()

    def _process_quote(self, quote, account, positions):
        """处理行情数据"""
        # 获取历史数据用于策略分析
        end_date = date.today()
        start_date = end_date - timedelta(days=60)

        df = self.data_source.get_kline(
            quote.code,
            kline_type='daily',
            start_date=start_date,
            end_date=end_date
        )

        if df.empty or len(df) < 20:
            return

        df['code'] = quote.code

        # 生成信号
        signals = self.strategy.generate_signals(df)
        signals = self.strategy.filter_signals(signals)

        for signal in signals:
            # 风险检查
            if signal.signal_type.value == 'BUY':
                quantity = int(account.available_cash * 0.1 / signal.price / 100) * 100
                alerts = self.risk_controller.check_all(
                    signal.code, signal.price, quantity, 'BUY',
                    account.total_assets, positions
                )

                if any(a.level.value in ['high', 'critical'] for a in alerts):
                    for alert in alerts:
                        self.alert_manager.warning("风险警告", str(alert))
                    continue

                if not self.risk_controller.is_trading_allowed:
                    continue

                # 执行买入
                order = self.broker.buy(signal.code, signal.price, quantity)
                self.logger.info(f"买入信号执行: {order}")
                self.alert_manager.info("交易执行", f"买入 {signal.code} {quantity}股 @ {signal.price}")
                self.dashboard.add_signal(signal.__dict__)

            elif signal.signal_type.value == 'SELL':
                if signal.code not in positions:
                    continue

                pos = positions[signal.code]
                quantity = pos['quantity']

                # 执行卖出
                order = self.broker.sell(signal.code, signal.price, quantity)
                self.logger.info(f"卖出信号执行: {order}")
                self.alert_manager.info("交易执行", f"卖出 {signal.code} {quantity}股 @ {signal.price}")
                self.dashboard.add_signal(signal.__dict__)

    def _signal_handler(self, signum, frame):
        """信号处理"""
        self.logger.info("收到退出信号")
        self._running = False

    def shutdown(self):
        """关闭系统"""
        self.logger.info("正在关闭系统...")

        if self.dashboard:
            self.dashboard.stop()
        if self.alert_manager:
            self.alert_manager.stop()
        if self.broker and self.broker.is_connected:
            self.broker.disconnect()
        if self.data_source and self.data_source.is_connected:
            self.data_source.disconnect()

        self.logger.info("系统已关闭")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='量化交易系统')
    parser.add_argument('--mode', choices=['backtest', 'paper', 'live'],
                        default='backtest', help='运行模式')
    parser.add_argument('--market', choices=['a_stock', 'hk_stock'],
                        default='a_stock', help='市场')
    parser.add_argument('--strategy', choices=['ma', 'macd', 'rsi', 'bollinger'],
                        default='ma', help='策略')
    parser.add_argument('--codes', nargs='+', default=['000001', '600000'],
                        help='股票代码列表')
    parser.add_argument('--start', type=str, default='2023-01-01',
                        help='回测开始日期')
    parser.add_argument('--end', type=str, default='2023-12-31',
                        help='回测结束日期')
    parser.add_argument('--capital', type=float, default=1000000,
                        help='初始资金')

    args = parser.parse_args()

    # 创建设置
    settings = Settings()
    if args.mode == 'backtest':
        settings.trading_mode = TradingMode.BACKTEST
    elif args.mode == 'paper':
        settings.trading_mode = TradingMode.PAPER
    else:
        settings.trading_mode = TradingMode.LIVE

    # 创建系统
    system = TradingSystem(settings)

    try:
        market = "A_STOCK" if args.market == 'a_stock' else "HK_STOCK"
        system.setup(market=market, strategy_name=args.strategy)

        if args.mode == 'backtest':
            start_date = datetime.strptime(args.start, '%Y-%m-%d').date()
            end_date = datetime.strptime(args.end, '%Y-%m-%d').date()
            system.backtest(args.codes, start_date, end_date, args.capital)
        else:
            system.run_live(args.codes)

    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        print(f"错误: {e}")
        raise
    finally:
        system.shutdown()


if __name__ == '__main__':
    main()
