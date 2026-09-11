#!/usr/bin/env python3
"""资金规模对策略表现的影响验证

对比 收盘价成交 下不同资金规模的回测表现:
  - 10万 (实盘)
  - 100万 (Exp009 验证过)
  - 1亿 (标准回测)

重点关注整手约束对小资金的影响: 跳过了多少股票、实际持仓数 vs 目标 TopK。

用法:
    python -m factor_lab.run_capital_impact
"""
import sys
import json
import time
import copy
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))


class CapitalBacktester:
    """收盘价回测器 — 专注资金规模影响分析

    基于 PaperTrader 逻辑，增加:
    - 每日实际持仓数 vs 目标 TopK 的差距追踪
    - 跳过股票原因分析 (高价股 / 资金不足)
    - 不写文件，纯内存计算
    """

    def __init__(self, signal: pd.Series, quality_score: pd.Series,
                 initial_cash: float = 100_000,
                 rebalance_every: int = 5,
                 stop_loss: float = 0.08,
                 open_cost: float = 0.0005,
                 close_cost: float = 0.0015,
                 thresholds: tuple = (0.3, 0.6)):
        self.signal = signal
        self.quality_score = quality_score
        self.initial_cash = initial_cash
        self.rebalance_every = rebalance_every
        self.stop_loss = stop_loss
        self.open_cost = open_cost
        self.close_cost = close_cost
        self.thresholds = thresholds

        from factor_lab.signal_generator import TOPK_BY_REGIME
        self.topk_map = TOPK_BY_REGIME

    def _get_regime(self, date) -> str:
        lo, hi = self.thresholds
        valid = self.quality_score[self.quality_score.index < date]
        if len(valid) == 0:
            return 'normal'
        last_score = valid.iloc[-1]
        if last_score >= hi:
            return 'strong'
        elif last_score < lo:
            return 'weak'
        return 'normal'

    def run(self, prices_df: pd.DataFrame) -> dict:
        """执行回测"""
        trading_days = sorted(prices_df.index.get_level_values(0).unique())
        signal_days = set(self.signal.index.get_level_values(0).unique())

        cash = self.initial_cash
        positions = {}       # {inst: {'shares', 'cost_price'}}
        pending_sells = []
        pending_buys = {}    # {inst: weight}

        daily_values = []
        regime_counts = {'strong': 0, 'normal': 0, 'weak': 0}

        # 资金约束追踪
        skip_events = []     # 每次跳过的详细记录
        holding_gaps = []    # 每个调仓日: 目标TopK vs 实际持仓
        total_trades = 0

        prev_close = {}

        for day_idx, date in enumerate(trading_days):
            if date not in prices_df.index.get_level_values(0):
                continue

            day_data = prices_df.loc[date]
            day_close = day_data['$close'].dropna()
            date_str = date.strftime('%Y-%m-%d')

            # --- 1. 执行挂起卖出 (今日收盘价) ---
            for inst in list(pending_sells):
                if inst not in positions:
                    pending_sells.remove(inst)
                    continue
                if inst not in day_close.index:
                    continue
                # 跌停检查
                if inst in prev_close and prev_close[inst] > 0:
                    if day_close[inst] / prev_close[inst] - 1 < -0.095:
                        continue

                price = day_close[inst]
                shares = positions[inst]['shares']
                cash += shares * price * (1 - self.close_cost)
                del positions[inst]
                pending_sells.remove(inst)
                total_trades += 1

            # --- 2. 执行挂起买入 (今日收盘价, 整手约束) ---
            if pending_buys:
                total_weight = sum(pending_buys.values())
                available_cash = cash * 0.99  # 预留 1%

                if total_weight > 0:
                    n_bought = 0
                    n_skipped_price = 0
                    n_skipped_cash = 0

                    for inst, weight in sorted(pending_buys.items(),
                                               key=lambda x: -x[1]):
                        if inst in positions:
                            continue
                        if inst not in day_close.index:
                            continue
                        # 涨停检查
                        if inst in prev_close and prev_close[inst] > 0:
                            if day_close[inst] / prev_close[inst] - 1 > 0.095:
                                continue

                        price = day_close[inst]
                        alloc = available_cash * (weight / total_weight)
                        shares = int(alloc / (price * (1 + self.open_cost)) / 100) * 100

                        if shares >= 100:
                            cost = shares * price * (1 + self.open_cost)
                            if cost > cash * 0.99:
                                n_skipped_cash += 1
                                skip_events.append({
                                    'date': date_str, 'inst': inst,
                                    'reason': 'insufficient_cash',
                                    'price': float(price),
                                    'min_cost': float(100 * price),
                                    'available': float(cash),
                                })
                                continue
                            cash -= cost
                            positions[inst] = {
                                'shares': shares,
                                'cost_price': float(price),
                            }
                            n_bought += 1
                            total_trades += 1
                        else:
                            n_skipped_price += 1
                            skip_events.append({
                                'date': date_str, 'inst': inst,
                                'reason': 'lot_size_constraint',
                                'price': float(price),
                                'min_cost': float(100 * price),
                                'alloc': float(alloc),
                            })

                pending_buys = {}

            # --- 3. 收盘估值 ---
            port_value = cash
            for inst, pos in positions.items():
                if inst in day_close.index:
                    port_value += pos['shares'] * day_close[inst]
                else:
                    port_value += pos['shares'] * pos['cost_price']
            daily_values.append({'date': date, 'value': port_value,
                                 'n_positions': len(positions)})

            # --- 4. 止损检查 ---
            if self.stop_loss:
                for inst, pos in list(positions.items()):
                    if inst in day_close.index:
                        ret = day_close[inst] / pos['cost_price'] - 1
                        if ret < -self.stop_loss:
                            if inst not in pending_sells:
                                pending_sells.append(inst)

            # --- 5. 调仓日 ---
            if day_idx % self.rebalance_every == 0 and date in signal_days:
                regime = self._get_regime(date)
                regime_counts[regime] += 1
                effective_topk = self.topk_map[regime]

                day_signal = self.signal.loc[date]
                if isinstance(day_signal, pd.DataFrame):
                    day_signal = day_signal.iloc[:, 0]
                day_signal = day_signal.sort_values(ascending=False)
                target_stocks = set(day_signal.head(effective_topk).index)

                # 卖出
                for inst in list(positions.keys()):
                    if inst not in target_stocks:
                        if inst not in pending_sells:
                            pending_sells.append(inst)

                # 买入 — 按信号分数降序排列，保证确定性迭代顺序。
                # 用集合差集会因 Python 字符串哈希随机化导致同配置两次结果不同。
                current_holds = set(positions.keys()) - set(pending_sells)
                to_buy = [i for i in day_signal.head(effective_topk).index
                          if i not in current_holds]
                if to_buy:
                    weight = 1.0 / effective_topk
                    pending_buys = {inst: weight for inst in to_buy}

                # 记录持仓差距 (下一日执行后才能统计实际买入)
                holding_gaps.append({
                    'date': date_str,
                    'regime': regime,
                    'target_topk': effective_topk,
                    'current_holds': len(current_holds),
                    'to_buy': len(to_buy),
                    'to_sell': len(set(positions.keys()) - target_stocks),
                })

            # 更新 prev_close
            for inst in day_close.index:
                prev_close[inst] = day_close[inst]

        # --- 计算绩效 ---
        return self._calc_metrics(daily_values, total_trades, regime_counts,
                                  skip_events, holding_gaps)

    def _calc_metrics(self, daily_values, n_trades, regime_counts,
                      skip_events, holding_gaps) -> dict:
        df = pd.DataFrame(daily_values).set_index('date')
        returns = df['value'].pct_change().dropna()

        total_ret = df['value'].iloc[-1] / self.initial_cash - 1
        n_days = len(returns)
        annual_ret = (1 + total_ret) ** (252 / max(n_days, 1)) - 1

        cumulative = (1 + returns).cumprod()
        drawdown = (cumulative - cumulative.cummax()) / cumulative.cummax()
        max_dd = float(drawdown.min())

        daily_std = float(returns.std())
        sharpe = float(returns.mean() / daily_std * (252 ** 0.5)) if daily_std > 0 else 0

        # 持仓统计
        avg_positions = df['n_positions'].mean()
        min_positions = int(df['n_positions'].min())
        max_positions = int(df['n_positions'].max())

        # 跳过分析
        skip_lot = [s for s in skip_events if s['reason'] == 'lot_size_constraint']
        skip_cash = [s for s in skip_events if s['reason'] == 'insufficient_cash']

        # 跳过的高价股统计
        skipped_prices = [s['price'] for s in skip_lot]
        avg_skip_price = np.mean(skipped_prices) if skipped_prices else 0

        return {
            'initial_cash': self.initial_cash,
            'final_value': float(df['value'].iloc[-1]),
            'total_return': float(total_ret),
            'annual_return': float(annual_ret),
            'sharpe': sharpe,
            'max_drawdown': max_dd,
            'n_trades': n_trades,
            'regime_counts': regime_counts,
            'avg_positions': round(avg_positions, 1),
            'min_positions': min_positions,
            'max_positions': max_positions,
            'skip_lot_size': len(skip_lot),
            'skip_cash': len(skip_cash),
            'total_skips': len(skip_events),
            'avg_skipped_price': round(avg_skip_price, 1),
            'start_date': df.index[0].strftime('%Y-%m-%d'),
            'end_date': df.index[-1].strftime('%Y-%m-%d'),
            'trading_days': n_days + 1,
            'holding_gaps': holding_gaps,
            'skip_events': skip_events,
        }


def main():
    import argparse
    parser = argparse.ArgumentParser(description='资金规模对策略表现的影响验证')
    parser.add_argument('--capitals', nargs='+', default=['10w'],
                        choices=['10w', '100w', '1y'],
                        help='要回测的资金档位 (默认: 10w 实盘规模)')
    args = parser.parse_args()

    import multiprocessing
    try:
        multiprocessing.set_start_method('fork', force=True)
    except (ValueError, RuntimeError):
        pass  # Windows 无 fork，使用默认 spawn

    import qlib
    from qlib.data import D
    from qlib.constant import REG_CN

    try:
        qlib.init(provider_uri='~/.qlib/qlib_data/cn_data_bs', region=REG_CN)
    except Exception:
        pass

    print("=" * 70)
    print("  资金规模对策略表现的影响验证")
    print("  执行模型: 收盘价 | 策略A TopK自适应 | 周度调仓 | 8%止损")
    print("=" * 70)

    # 加载数据
    print("\n[1/3] 加载数据...")
    t0 = time.time()

    from factor_lab.signal_generator import SignalGenerator
    sg = SignalGenerator()
    signal = sg.load_predictions()
    quality = sg.load_quality_score()

    cfg = sg.config
    start = cfg['test_start']
    end = cfg['test_end']

    instruments = D.instruments('csi300')
    prices = D.features(instruments, ['$close'],
                        start_time=start, end_time=end)
    prices = prices.swaplevel().sort_index()
    print(f"  完成 ({time.time()-t0:.1f}s)")

    # 运行回测 — 默认只测实盘规模 10万; --capitals 可指定其他档位
    ALL_CAPITALS = {
        '10w':  (100_000, '10万 (实盘)'),
        '100w': (1_000_000, '100万 (Exp009)'),
        '1y':   (100_000_000, '1亿 (标准回测)'),
    }
    capitals = [ALL_CAPITALS[k] for k in args.capitals]

    results = {}
    print(f"\n[2/3] 运行回测 ({start} ~ {end})...")

    for capital, label in capitals:
        print(f"\n  --- {label} ---")
        t1 = time.time()

        bt = CapitalBacktester(
            signal=signal,
            quality_score=quality,
            initial_cash=capital,
            rebalance_every=int(cfg.get('rebalance_every', 5)),
            stop_loss=float(cfg.get('stop_loss', 0.08)),
            thresholds=tuple(cfg.get('adaptive_thresholds', [0.3, 0.6])),
        )
        result = bt.run(prices)
        results[label] = result
        print(f"  Sharpe: {result['sharpe']:.3f} | "
              f"收益: {result['total_return']:.1%} | "
              f"MDD: {result['max_drawdown']:.1%} | "
              f"交易: {result['n_trades']}")
        print(f"  持仓: 平均{result['avg_positions']}只 "
              f"(min={result['min_positions']}, max={result['max_positions']})")
        print(f"  跳过: 整手约束={result['skip_lot_size']}次, "
              f"资金不足={result['skip_cash']}次")
        print(f"  耗时: {time.time()-t1:.1f}s")

    # 对比报告
    print(f"\n[3/3] 对比报告")
    print("=" * 70)

    # 表头
    header = f"{'指标':<20}"
    for _, label in capitals:
        header += f"  {label:>15}"
    print(header)
    print("-" * 70)

    # Sharpe
    row = f"{'Sharpe':<20}"
    for _, label in capitals:
        row += f"  {results[label]['sharpe']:>15.3f}"
    print(row)

    # 总收益
    row = f"{'总收益':<18}"
    for _, label in capitals:
        row += f"  {results[label]['total_return']:>14.1%}"
    print(row)

    # 年化收益
    row = f"{'年化收益':<17}"
    for _, label in capitals:
        row += f"  {results[label]['annual_return']:>14.1%}"
    print(row)

    # MDD
    row = f"{'最大回撤':<17}"
    for _, label in capitals:
        row += f"  {results[label]['max_drawdown']:>14.1%}"
    print(row)

    # 交易次数
    row = f"{'交易次数':<17}"
    for _, label in capitals:
        row += f"  {results[label]['n_trades']:>15}"
    print(row)

    # 平均持仓
    row = f"{'平均持仓数':<16}"
    for _, label in capitals:
        row += f"  {results[label]['avg_positions']:>15.1f}"
    print(row)

    # 跳过次数
    row = f"{'整手跳过':<17}"
    for _, label in capitals:
        row += f"  {results[label]['skip_lot_size']:>15}"
    print(row)

    row = f"{'资金不足跳过':<15}"
    for _, label in capitals:
        row += f"  {results[label]['skip_cash']:>15}"
    print(row)

    print("-" * 70)

    # 衰减分析 — 以本次跑的最大资金档为基准（可能只跑了部分档位）
    base_label = capitals[-1][1]
    base = results[base_label]
    print(f"\n衰减分析 (相对 {base_label}):")
    for _, label in capitals[:-1]:
        r = results[label]
        sharpe_loss = r['sharpe'] - base['sharpe']
        ret_loss = r['total_return'] - base['total_return']
        mdd_diff = r['max_drawdown'] - base['max_drawdown']
        print(f"  {label}:")
        print(f"    Sharpe:  {r['sharpe']:.3f} vs {base['sharpe']:.3f} "
              f"({sharpe_loss:+.3f}, {sharpe_loss/base['sharpe']*100:+.1f}%)")
        print(f"    收益:    {r['total_return']:.1%} vs {base['total_return']:.1%} "
              f"({ret_loss:+.1%})")
        print(f"    MDD:     {r['max_drawdown']:.1%} vs {base['max_drawdown']:.1%} "
              f"({mdd_diff:+.1%})")

    # 最小资金档跳过的高价股详情（整手约束在小资金下最明显）
    small_label = capitals[0][1]
    skip_lot_10w = [s for s in results[small_label]['skip_events']
                    if s['reason'] == 'lot_size_constraint']
    if skip_lot_10w:
        # 统计被跳过最多的股票
        from collections import Counter
        skip_counter = Counter(s['inst'] for s in skip_lot_10w)
        top_skipped = skip_counter.most_common(10)

        print(f"\n{small_label} 最常被跳过的股票 (整手约束, 共{len(skip_lot_10w)}次):")
        for inst, count in top_skipped:
            # 找到该股票最近一次的价格
            prices_list = [s['price'] for s in skip_lot_10w if s['inst'] == inst]
            avg_price = np.mean(prices_list)
            min_cost = avg_price * 100
            print(f"  {inst}: {count}次 (均价{avg_price:.0f}, "
                  f"100股需{min_cost:,.0f}元)")

    # 保存结果
    output_dir = PROJECT_DIR / "factor_lab" / "results" / "capital_impact"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 精简保存 (去掉大体积的 holding_gaps 和 skip_events)
    save_results = {}
    for label, r in results.items():
        save_r = {k: v for k, v in r.items()
                  if k not in ('holding_gaps', 'skip_events')}
        save_results[label] = save_r

    output_file = output_dir / "capital_impact_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(save_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n结果已保存: {output_file}")


if __name__ == '__main__':
    main()
