#!/usr/bin/env python3
"""模拟盘引擎 — 每日更新 NAV + 周度调仓

状态持久化:
- state.json: 现金 + 持仓 + 元数据
- trades.csv: 交易记录
- daily_nav.csv: 日净值

执行模型 (T+1 收盘价):
- Day T 收盘后: 生成信号 → 产生交易指令 (pending)
- Day T+1 收盘: 以收盘价执行交易 + 更新 NAV

用法:
    # 回放历史 (验证)
    python -m factor_lab.paper_trader replay

    # 查看当前状态
    python -m factor_lab.paper_trader status

    # 重置
    python -m factor_lab.paper_trader reset
"""
import json
import csv
import time
import warnings
import argparse
from pathlib import Path
from datetime import datetime

import yaml
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

PROJECT_DIR = Path(__file__).resolve().parent.parent

from factor_lab.signal_generator import TOPK_BY_REGIME


class PaperTrader:
    """模拟盘引擎 — 收盘价成交，T+1 执行"""

    def __init__(self, config_path: str = 'config/signal_config.yaml'):
        self.config = self._load_config(config_path)
        self.state_dir = PROJECT_DIR / self.config['state_dir']
        self.state_dir.mkdir(parents=True, exist_ok=True)

        # 从 SignalGenerator 加载信号和质量分数
        from factor_lab.signal_generator import SignalGenerator
        self.sg = SignalGenerator(config_path)

        self.state = self._load_state()

    def _load_config(self, config_path: str) -> dict:
        full_path = PROJECT_DIR / config_path
        with open(full_path) as f:
            return yaml.safe_load(f)

    # ── 状态管理 ──

    def _load_state(self) -> dict:
        state_file = self.state_dir / "state.json"
        if state_file.exists():
            with open(state_file) as f:
                return json.load(f)
        return self._initial_state()

    def _initial_state(self) -> dict:
        return {
            'cash': self.config['initial_cash'],
            'positions': {},
            'pending_orders': {'sells': [], 'buys': {}},
            'metadata': {
                'initial_cash': self.config['initial_cash'],
                'start_date': None,
                'last_update': None,
                'day_count': 0,
                'strategy': self.config['adaptive_strategy'],
            },
        }

    def _save_state(self):
        state_file = self.state_dir / "state.json"

        def _convert(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        with open(state_file, 'w') as f:
            json.dump(self.state, f, indent=2, ensure_ascii=False, default=_convert)

    def _append_trade(self, date: str, action: str, instrument: str,
                      shares: int, price: float, cost: float, reason: str):
        """追加交易记录到 CSV"""
        trades_file = self.state_dir / "trades.csv"
        is_new = not trades_file.exists()
        with open(trades_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if is_new:
                writer.writerow(['date', 'action', 'instrument', 'shares',
                                 'price', 'cost', 'reason'])
            writer.writerow([date, action, instrument, shares,
                             f"{price:.4f}", f"{cost:.4f}", reason])

    def _append_nav(self, date: str, nav: float, cash: float,
                    n_positions: int):
        """追加日净值到 CSV"""
        nav_file = self.state_dir / "daily_nav.csv"
        is_new = not nav_file.exists()
        with open(nav_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if is_new:
                writer.writerow(['date', 'nav', 'cash', 'n_positions'])
            writer.writerow([date, f"{nav:.2f}", f"{cash:.2f}", n_positions])

    def reset(self, initial_cash: float = None):
        """重置模拟盘"""
        if initial_cash:
            self.config['initial_cash'] = initial_cash
        self.state = self._initial_state()
        self._save_state()
        # 清除交易和净值记录
        for f in ['trades.csv', 'daily_nav.csv']:
            p = self.state_dir / f
            if p.exists():
                p.unlink()
        print(f"模拟盘已重置 (初始资金: {self.state['cash']:,.0f})")

    # ── 每日更新 ──

    def update_daily(self, date: pd.Timestamp, prices_df: pd.DataFrame,
                     prev_close_map: dict, day_idx: int,
                     signal: pd.Series, quality_score: pd.Series,
                     auto_save: bool = False) -> dict:
        """每日收盘后调用

        Args:
            date: 当日日期
            prices_df: 价格 DataFrame (datetime, instrument)
            prev_close_map: {instrument: 前收盘价}
            day_idx: 交易日序号 (用于判断调仓日)
            signal: 预测信号 Series
            quality_score: 信号质量评分 Series

        Returns:
            {'nav': float, 'trades': int, 'regime': str or None}
        """
        if date not in prices_df.index.get_level_values(0):
            return {'nav': 0, 'trades': 0, 'regime': None}

        day_data = prices_df.loc[date]
        day_close = day_data['$close'].dropna()

        date_str = date.strftime('%Y-%m-%d')
        n_trades = 0
        cfg = self.config

        # --- 1. 执行昨日挂起的交易指令 (以今日收盘价) ---
        pending = self.state['pending_orders']

        # 卖出
        for inst in list(pending['sells']):
            if inst not in self.state['positions']:
                pending['sells'].remove(inst)
                continue
            if inst not in day_close.index:
                continue
            # 跌停不能卖
            if inst in prev_close_map and prev_close_map[inst] > 0:
                limit_ret = day_close[inst] / prev_close_map[inst] - 1
                if limit_ret < -0.095:
                    continue

            price = day_close[inst]
            pos = self.state['positions'][inst]
            shares = pos['shares']
            proceeds = shares * price * (1 - cfg['close_cost'])
            self.state['cash'] += proceeds
            self._append_trade(date_str, 'SELL', inst, shares, price,
                               shares * price * cfg['close_cost'], 'rebalance')
            del self.state['positions'][inst]
            pending['sells'].remove(inst)
            n_trades += 1

        # 买入
        if pending['buys']:
            total_weight = sum(pending['buys'].values())
            available_cash = self.state['cash'] * 0.99  # 预留 1%
            if total_weight > 0:
                for inst, weight in list(pending['buys'].items()):
                    if inst in self.state['positions']:
                        continue
                    if inst not in day_close.index:
                        continue
                    # 涨停不能买
                    if inst in prev_close_map and prev_close_map[inst] > 0:
                        limit_ret = day_close[inst] / prev_close_map[inst] - 1
                        if limit_ret > 0.095:
                            continue

                    price = day_close[inst]
                    alloc = available_cash * (weight / total_weight)
                    shares = int(alloc / (price * (1 + cfg['open_cost'])) / 100) * 100
                    if shares >= 100:
                        cost = shares * price * (1 + cfg['open_cost'])
                        self.state['cash'] -= cost
                        self.state['positions'][inst] = {
                            'shares': shares,
                            'cost_price': float(price),
                            'entry_date': date_str,
                        }
                        self._append_trade(date_str, 'BUY', inst, shares, price,
                                           shares * price * cfg['open_cost'], 'rebalance')
                        n_trades += 1
            pending['buys'] = {}

        # --- 2. 收盘估值 ---
        port_value = self.state['cash']
        for inst, pos in self.state['positions'].items():
            if inst in day_close.index:
                port_value += pos['shares'] * day_close[inst]
            else:
                port_value += pos['shares'] * pos['cost_price']

        self._append_nav(date_str, port_value, self.state['cash'],
                         len(self.state['positions']))

        # --- 3. 收盘后检查止损 ---
        if cfg['stop_loss']:
            for inst, pos in list(self.state['positions'].items()):
                if inst in day_close.index:
                    ret = day_close[inst] / pos['cost_price'] - 1
                    if ret < -cfg['stop_loss']:
                        if inst not in pending['sells']:
                            pending['sells'].append(inst)
                            # 止损卖出不等调仓日，下一日收盘执行

        # --- 4. 调仓日: 生成新的交易指令 ---
        regime = None
        rebal = cfg['rebalance_every']
        signal_dates = set(signal.index.get_level_values(0).unique())

        if day_idx % rebal == 0 and date in signal_dates:
            # 信号质量 → regime
            lo, hi = cfg['adaptive_thresholds']
            valid_q = quality_score[quality_score.index < date]
            if len(valid_q) > 0:
                last_q = valid_q.iloc[-1]
                if last_q >= hi:
                    regime = 'strong'
                elif last_q < lo:
                    regime = 'weak'
                else:
                    regime = 'normal'
            else:
                regime = 'normal'

            effective_topk = TOPK_BY_REGIME[regime]

            # 获取当日信号
            if date in signal_dates:
                day_signal = signal.loc[date]
                if isinstance(day_signal, pd.DataFrame):
                    day_signal = day_signal.iloc[:, 0]
                day_signal = day_signal.sort_values(ascending=False)
                target_stocks = set(day_signal.head(effective_topk).index)

                # 卖出不在目标中的持仓
                for inst in list(self.state['positions'].keys()):
                    if inst not in target_stocks and inst not in pending['sells']:
                        pending['sells'].append(inst)

                # 规划买入
                current_holds = set(self.state['positions'].keys()) - set(pending['sells'])
                to_buy = target_stocks - current_holds
                if to_buy:
                    weight = 1.0 / effective_topk
                    pending['buys'] = {inst: weight for inst in to_buy}

        # 更新元数据
        self.state['pending_orders'] = pending
        self.state['metadata']['last_update'] = date_str
        self.state['metadata']['day_count'] += 1
        if self.state['metadata']['start_date'] is None:
            self.state['metadata']['start_date'] = date_str

        if auto_save:
            self._save_state()

        return {'nav': port_value, 'trades': n_trades, 'regime': regime}

    # ── 批量回放 ──

    def replay(self, start_date: str = None, end_date: str = None,
               verbose: bool = True) -> dict:
        """批量回放历史数据

        逐日调用 update_daily()，用于:
        1. 初始化模拟盘 (从 TEST_START 回放到今天)
        2. 与 AdaptiveBacktester 结果对比验证

        Returns:
            绩效指标 dict
        """
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
            pass  # 已初始化

        cfg = self.config
        start = start_date or cfg['test_start']
        end = end_date or cfg['test_end']

        if verbose:
            print(f"\n=== 模拟盘回放: {start} ~ {end} ===")
            print(f"初始资金: {cfg['initial_cash']:,.0f}")
            print(f"成交价: 收盘价 | 策略: {cfg['adaptive_strategy']}")
            print(f"TopK: strong={TOPK_BY_REGIME['strong']}, "
                  f"normal={TOPK_BY_REGIME['normal']}, "
                  f"weak={TOPK_BY_REGIME['weak']}")

        # 重置状态
        self.reset()

        # 加载数据
        t0 = time.time()
        if verbose:
            print("\n加载数据...", end=" ", flush=True)

        instruments = D.instruments('csi300')
        prices = D.features(instruments, ['$close'],
                            start_time=start, end_time=end)
        prices = prices.swaplevel().sort_index()

        signal = self.sg.load_predictions()
        quality = self.sg.load_quality_score()

        if verbose:
            print(f"完成 ({time.time()-t0:.1f}s)")

        # 交易日列表
        trading_days = sorted(prices.index.get_level_values(0).unique())
        if verbose:
            print(f"交易日: {len(trading_days)} 天")
            print(f"信号日: {len(signal.index.get_level_values(0).unique())} 天")

        # 逐日回放
        prev_close_map = {}
        total_trades = 0
        regime_counts = {'strong': 0, 'normal': 0, 'weak': 0}
        t_start = time.time()

        for day_idx, date in enumerate(trading_days):
            result = self.update_daily(
                date, prices, prev_close_map, day_idx, signal, quality
            )
            total_trades += result['trades']
            if result['regime']:
                regime_counts[result['regime']] += 1

            # 更新 prev_close_map
            if date in prices.index.get_level_values(0):
                day_close = prices.loc[date]['$close'].dropna()
                for inst in day_close.index:
                    prev_close_map[inst] = day_close[inst]

            # 进度显示
            if verbose and (day_idx + 1) % 100 == 0:
                nav = result['nav']
                ret = nav / cfg['initial_cash'] - 1
                print(f"  Day {day_idx+1}/{len(trading_days)}: "
                      f"{date.strftime('%Y-%m-%d')} NAV={nav:,.0f} "
                      f"({ret:+.2%}) 持仓={len(self.state['positions'])}")

        # 保存最终状态
        self._save_state()

        elapsed = time.time() - t_start

        # 计算绩效
        perf = self.get_performance()
        perf['n_trades'] = total_trades
        perf['regime_counts'] = regime_counts
        perf['replay_time'] = round(elapsed, 1)

        if verbose:
            print(f"\n=== 回放完成 ({elapsed:.1f}s) ===")
            self._print_performance(perf)

        return perf

    # ── 绩效计算 ──

    def get_performance(self) -> dict:
        """从 daily_nav.csv 计算绩效指标"""
        nav_file = self.state_dir / "daily_nav.csv"
        if not nav_file.exists():
            return {'error': '无净值数据'}

        df = pd.read_csv(nav_file)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()

        # 去重 (replay 可能追加重复行)
        df = df[~df.index.duplicated(keep='last')]

        nav = df['nav']
        returns = nav.pct_change().dropna()
        initial = self.config['initial_cash']

        total_ret = nav.iloc[-1] / initial - 1
        n_days = len(returns)
        annual_ret = (1 + total_ret) ** (252 / max(n_days, 1)) - 1

        cumulative = (1 + returns).cumprod()
        drawdown = (cumulative - cumulative.cummax()) / cumulative.cummax()
        max_dd = float(drawdown.min())

        daily_std = float(returns.std())
        sharpe = float(returns.mean() / daily_std * (252 ** 0.5)) if daily_std > 0 else 0.0

        # 基准收益
        bench_ret = self._get_benchmark_return(
            df.index[0].strftime('%Y-%m-%d'),
            df.index[-1].strftime('%Y-%m-%d'),
        )

        return {
            'total_return': float(total_ret),
            'annual_return': float(annual_ret),
            'max_drawdown': max_dd,
            'sharpe': sharpe,
            'bench_return': bench_ret,
            'excess_return': float(total_ret) - bench_ret if bench_ret else None,
            'final_nav': float(nav.iloc[-1]),
            'trading_days': n_days + 1,
            'start_date': df.index[0].strftime('%Y-%m-%d'),
            'end_date': df.index[-1].strftime('%Y-%m-%d'),
        }

    def _get_benchmark_return(self, start: str, end: str) -> float | None:
        """获取基准 (沪深300) 收益率"""
        try:
            from qlib.data import D
            bench = D.features(['SH000300'], ['$close'],
                               start_time=start, end_time=end)
            bench_close = bench['$close'].droplevel(0)
            return float(bench_close.iloc[-1] / bench_close.iloc[0] - 1)
        except Exception:
            return None

    def _print_performance(self, perf: dict):
        """打印绩效摘要"""
        print(f"\n{'='*60}")
        print(f"  模拟盘绩效 ({perf.get('start_date','?')} ~ {perf.get('end_date','?')})")
        print(f"{'='*60}")
        print(f"  最终净值:   {perf.get('final_nav', 0):>14,.0f}")
        print(f"  总收益:     {perf.get('total_return', 0):>14.2%}")
        print(f"  年化收益:   {perf.get('annual_return', 0):>14.2%}")
        print(f"  Sharpe:     {perf.get('sharpe', 0):>14.3f}")
        print(f"  最大回撤:   {perf.get('max_drawdown', 0):>14.2%}")
        if perf.get('bench_return') is not None:
            print(f"  基准收益:   {perf['bench_return']:>14.2%}")
            print(f"  超额收益:   {perf.get('excess_return', 0):>14.2%}")
        print(f"  交易次数:   {perf.get('n_trades', 0):>14}")
        print(f"  交易日:     {perf.get('trading_days', 0):>14}")
        if perf.get('regime_counts'):
            rc = perf['regime_counts']
            print(f"  信号状态:   strong={rc.get('strong',0)} "
                  f"normal={rc.get('normal',0)} weak={rc.get('weak',0)}")
        if perf.get('replay_time'):
            print(f"  回放耗时:   {perf['replay_time']:>14.1f}s")
        print(f"{'='*60}")

    # ── 持仓/指令展示 ──

    def get_current_holdings(self) -> str:
        """人类可读的持仓明细"""
        pos = self.state['positions']
        if not pos:
            return "当前无持仓"

        lines = [f"现金: {self.state['cash']:,.0f}",
                 f"持仓 ({len(pos)} 只):"]
        lines.append(f"  {'代码':<12} {'股数':>8} {'成本':>10} {'入场日期':<12}")
        lines.append("  " + "-" * 48)
        for inst, p in sorted(pos.items()):
            lines.append(f"  {inst:<12} {p['shares']:>8} "
                         f"{p['cost_price']:>10.2f} {p.get('entry_date', 'N/A'):<12}")
        return "\n".join(lines)

    def get_trade_instructions(self) -> str:
        """人类可读的调仓指令 (飞书推送格式)"""
        pending = self.state['pending_orders']
        if not pending['sells'] and not pending['buys']:
            return "无待执行指令"

        lines = [f"待执行指令 (下一交易日收盘价执行):"]
        if pending['sells']:
            lines.append(f"  卖出 ({len(pending['sells'])} 只):")
            for inst in pending['sells']:
                pos = self.state['positions'].get(inst, {})
                lines.append(f"    {inst} {pos.get('shares', '?')} 股")
        if pending['buys']:
            lines.append(f"  买入 ({len(pending['buys'])} 只):")
            for inst, weight in pending['buys'].items():
                lines.append(f"    {inst} 目标权重 {weight:.2%}")
        return "\n".join(lines)


# ── CLI 入口 ──

def main():
    parser = argparse.ArgumentParser(description="模拟盘引擎")
    parser.add_argument('action', choices=['replay', 'status', 'reset'],
                        help="replay=回放历史, status=查看状态, reset=重置")
    parser.add_argument('--start', default=None, help="回放起始日期")
    parser.add_argument('--end', default=None, help="回放结束日期")
    args = parser.parse_args()

    pt = PaperTrader()

    if args.action == 'replay':
        result = pt.replay(start_date=args.start, end_date=args.end)
        # 保存绩效
        perf_file = pt.state_dir / "replay_performance.json"
        with open(perf_file, 'w') as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n绩效已保存: {perf_file}")

    elif args.action == 'status':
        print(pt.get_current_holdings())
        print()
        print(pt.get_trade_instructions())
        perf = pt.get_performance()
        if 'error' not in perf:
            pt._print_performance(perf)

    elif args.action == 'reset':
        pt.reset()


if __name__ == '__main__':
    main()
