#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
回测示例
演示如何使用回测引擎测试策略
"""

import sys
sys.path.append('..')

from datetime import date
import pandas as pd

from data import AStockDataSource
from strategy import MAStrategy, MACDStrategy, RSIStrategy
from backtest import BacktestEngine, PerformanceAnalyzer


def run_backtest_example():
    """运行回测示例"""
    print("=" * 50)
    print("量化交易系统 - 回测示例")
    print("=" * 50)

    # 1. 连接数据源
    print("\n[1] 连接数据源...")
    data_source = AStockDataSource(source="akshare")
    if not data_source.connect():
        print("连接数据源失败")
        return

    # 2. 获取数据
    print("\n[2] 获取历史数据...")
    codes = ['000001', '600000', '600036']  # 平安银行、浦发银行、招商银行
    start_date = date(2023, 1, 1)
    end_date = date(2023, 12, 31)

    all_data = []
    for code in codes:
        print(f"  获取 {code} 数据...")
        from data.data_source import KLineType
        df = data_source.get_kline(
            code,
            KLineType.DAILY,
            start_date,
            end_date
        )
        if not df.empty:
            df['code'] = code
            all_data.append(df)
            print(f"    获取到 {len(df)} 条数据")

    if not all_data:
        print("没有获取到数据")
        return

    combined_data = pd.concat(all_data, ignore_index=True)
    print(f"\n  共获取 {len(combined_data)} 条数据")

    # 3. 创建策略
    print("\n[3] 创建策略...")
    strategies = [
        MAStrategy(name="双均线策略", short_period=5, long_period=20),
        MACDStrategy(name="MACD策略"),
        RSIStrategy(name="RSI策略", rsi_period=14)
    ]

    # 4. 运行回测
    print("\n[4] 运行回测...")
    results = []

    for strategy in strategies:
        print(f"\n  回测策略: {strategy.name}")

        engine = BacktestEngine(
            initial_capital=1000000,
            commission_rate=0.0003,
            stamp_duty_rate=0.001
        )

        result = engine.run(strategy, combined_data, start_date, end_date)
        results.append(result)

        # 打印简要结果
        print(f"    总收益率: {result.total_return:.2%}")
        print(f"    年化收益率: {result.annual_return:.2%}")
        print(f"    最大回撤: {result.max_drawdown:.2%}")
        print(f"    夏普比率: {result.sharpe_ratio:.2f}")
        print(f"    交易次数: {result.total_trades}")
        print(f"    胜率: {result.win_rate:.2%}")

    # 5. 比较策略
    print("\n[5] 策略比较")
    print("-" * 70)
    print(f"{'策略名称':<15} {'总收益率':>10} {'年化收益':>10} {'最大回撤':>10} {'夏普比率':>8} {'胜率':>8}")
    print("-" * 70)

    for result in results:
        print(f"{result.strategy_name:<15} {result.total_return:>10.2%} {result.annual_return:>10.2%} "
              f"{result.max_drawdown:>10.2%} {result.sharpe_ratio:>8.2f} {result.win_rate:>8.2%}")

    print("-" * 70)

    # 6. 打印最佳策略详细报告
    best_result = max(results, key=lambda x: x.sharpe_ratio)
    print(f"\n[6] 最佳策略详细报告 ({best_result.strategy_name})")
    print(best_result.summary())

    # 断开连接
    data_source.disconnect()
    print("\n回测完成!")


if __name__ == '__main__':
    run_backtest_example()
