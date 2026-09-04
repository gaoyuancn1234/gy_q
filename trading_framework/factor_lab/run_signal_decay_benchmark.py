#!/usr/bin/env python3
"""
实验 008: 信号衰减检测与动态仓位管理

检测 rolling 训练中的信号衰减（特别是 Window 8），
构建信号质量评分，驱动自适应策略降低弱信号期的风险暴露。

策略矩阵:
  A: TopK 自适应  — 信号强 topk=12, 正常 topk=16, 弱 topk=20
  B: 仓位缩放    — 信号强 满仓, 正常 80%, 弱 50%
  C: 信号加权    — 按预测值分配权重 (max 15%)
  D: A+B 联动    — 同时调 topk 和仓位

用法:
  # 完整运行
  python -m factor_lab.run_signal_decay_benchmark

  # 仅信号分析 (跳过回测)
  python -m factor_lab.run_signal_decay_benchmark --analysis-only

  # 仅回测 (需已有信号质量缓存)
  python -m factor_lab.run_signal_decay_benchmark --backtest-only
"""
import sys
import json
import time
import pickle
import warnings
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

TEST_START = '2024-01-01'
TEST_END = '2026-02-05'
CONFIG_NAME = 'D_expand_3v_3r'

CACHE_DIR = PROJECT_DIR / "factor_lab" / "results" / "rolling" / "predictions"
RESULTS_DIR = PROJECT_DIR / "factor_lab" / "results" / "signal_decay"


def _get_current_preset() -> str:
    """从 signal_config.yaml 读取当前生产 preset"""
    import yaml
    config_path = PROJECT_DIR / "config" / "signal_config.yaml"
    try:
        with open(config_path, encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        return cfg.get('preset', 'alpha158_val')
    except Exception:
        return 'alpha158_val'


PRESET = _get_current_preset()
ROLLING_JSON = PROJECT_DIR / "factor_lab" / "results" / "rolling" / f"{CONFIG_NAME}_{PRESET}_LightGBM.json"


# ============ Part A: 信号质量分析 ============

def load_predictions() -> pd.Series:
    """加载 LightGBM rolling 预测"""
    pkl_path = CACHE_DIR / f"{CONFIG_NAME}_{PRESET}_LightGBM.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"预测缓存不存在: {pkl_path}\n请先运行 run_execution_benchmark.py")
    return pd.read_pickle(pkl_path)


def load_close_prices() -> pd.Series:
    """从 Qlib 加载收盘价"""
    from qlib.data import D
    instruments = D.instruments('csi300')
    prices = D.features(instruments, ['$close'],
                        start_time=TEST_START, end_time=TEST_END)
    prices = prices.swaplevel().sort_index()
    return prices['$close']


def load_rolling_windows() -> list[dict]:
    """加载 rolling window 信息"""
    with open(ROLLING_JSON, encoding='utf-8') as f:
        data = json.load(f)
    return data['windows']


def run_signal_analysis(signal: pd.Series, close: pd.Series) -> dict:
    """Part A: 计算全部信号质量指标

    Returns:
        dict with quality_score, regime, daily_ic, metrics DataFrame, etc.
    """
    from factor_lab.evaluation.signal_quality import (
        compute_rolling_ic, compute_signal_dispersion, compute_signal_spread,
        build_quality_score, classify_signal_regime, compute_forward_returns,
    )

    print("  [1/5] 计算 forward returns...")
    fwd_ret = compute_forward_returns(close, periods=1)

    print("  [2/5] 计算滚动 IC (20d / 60d)...")
    ic_20 = compute_rolling_ic(signal, fwd_ret, window=20)
    ic_60 = compute_rolling_ic(signal, fwd_ret, window=60)
    # Shift by 1: IC at date T requires T+1 close (forward return), so T's IC
    # is only observable at T+1 close. Shift ensures no lookahead.
    ic_20 = ic_20.shift(1).dropna()
    ic_60 = ic_60.shift(1).dropna()
    print(f"        IC_20: {len(ic_20)} days, mean={ic_20.mean():.4f}")
    print(f"        IC_60: {len(ic_60)} days, mean={ic_60.mean():.4f}")

    print("  [3/5] 计算信号离散度和 spread...")
    dispersion = compute_signal_dispersion(signal, topk=12)
    spread = compute_signal_spread(signal, topk=12)

    print("  [4/5] 构建 quality_score...")
    quality = build_quality_score(ic_20, ic_60, dispersion, spread)
    regime = classify_signal_regime(quality)
    print(f"        quality_score: mean={quality.mean():.3f}, "
          f"strong={int((regime == 'strong').sum())}d, "
          f"normal={int((regime == 'normal').sum())}d, "
          f"weak={int((regime == 'weak').sum())}d")

    print("  [5/5] 按 window 汇总...")
    windows = load_rolling_windows()
    window_stats = []
    for w in windows:
        wnum = w['window_num']
        ws = pd.Timestamp(w['pred_start'])
        we = pd.Timestamp(w['pred_end'])
        mask = (quality.index >= ws) & (quality.index <= we)
        q_window = quality[mask]
        r_window = regime[mask]
        if len(q_window) == 0:
            continue
        window_stats.append({
            'window': wnum,
            'pred_start': w['pred_start'],
            'pred_end': w['pred_end'],
            'best_iter': w.get('best_iteration', None),
            'q_mean': float(q_window.mean()),
            'q_median': float(q_window.median()),
            'q_min': float(q_window.min()),
            'pct_strong': float((r_window == 'strong').mean()),
            'pct_weak': float((r_window == 'weak').mean()),
        })

    window_df = pd.DataFrame(window_stats)
    if len(window_df) > 0 and 'best_iter' in window_df.columns:
        corr = window_df['q_mean'].corr(window_df['best_iter'])
        print(f"        quality_score 与 best_iteration 相关性: {corr:.3f}")

    return {
        'quality_score': quality,
        'regime': regime,
        'ic_20': ic_20,
        'ic_60': ic_60,
        'dispersion': dispersion,
        'spread': spread,
        'fwd_ret': fwd_ret,
        'window_stats': window_df,
    }


# ============ Part B: 自适应回测器 ============

class AdaptiveBacktester:
    """自适应回测器 — 根据信号质量动态调整仓位

    基于 PortfolioBacktester，增加:
    1. 调仓日查 quality_score → 决定 effective_topk 和 cash_fraction
    2. 买入时 available_cash *= (1 - cash_fraction)
    3. 策略 C 用信号值加权替代等权
    """

    STRATEGIES = {
        'baseline_12': {
            'desc': 'Baseline topk=12',
            'topk': {'strong': 12, 'normal': 12, 'weak': 12},
            'cash_frac': {'strong': 0.0, 'normal': 0.0, 'weak': 0.0},
            'signal_weight': False,
        },
        'baseline_20': {
            'desc': 'Baseline topk=20',
            'topk': {'strong': 20, 'normal': 20, 'weak': 20},
            'cash_frac': {'strong': 0.0, 'normal': 0.0, 'weak': 0.0},
            'signal_weight': False,
        },
        'A_topk_adaptive': {
            'desc': 'TopK自适应',
            'topk': {'strong': 12, 'normal': 16, 'weak': 20},
            'cash_frac': {'strong': 0.0, 'normal': 0.0, 'weak': 0.0},
            'signal_weight': False,
        },
        'B_position_scale': {
            'desc': '仓位缩放',
            'topk': {'strong': 12, 'normal': 12, 'weak': 12},
            'cash_frac': {'strong': 0.0, 'normal': 0.2, 'weak': 0.5},
            'signal_weight': False,
        },
        'C_signal_weight': {
            'desc': '信号加权',
            'topk': {'strong': 12, 'normal': 12, 'weak': 12},
            'cash_frac': {'strong': 0.0, 'normal': 0.0, 'weak': 0.0},
            'signal_weight': True,
        },
        'D_combined': {
            'desc': 'A+B联动',
            'topk': {'strong': 12, 'normal': 16, 'weak': 20},
            'cash_frac': {'strong': 0.0, 'normal': 0.2, 'weak': 0.5},
            'signal_weight': False,
        },
    }

    def __init__(self, signal: pd.Series, quality_score: pd.Series,
                 strategy: str = 'A_topk_adaptive',
                 thresholds: tuple[float, float] = (0.3, 0.6),
                 rebalance_every: int = 5,
                 stop_loss: float | None = 0.08,
                 initial_cash: float = 100_000_000,
                 open_cost: float = 0.0005, close_cost: float = 0.0015):
        self.signal = signal
        self.quality_score = quality_score
        self.strategy_name = strategy
        self.strategy_cfg = self.STRATEGIES[strategy]
        self.thresholds = thresholds
        self.rebalance_every = rebalance_every
        self.stop_loss = stop_loss
        self.initial_cash = initial_cash
        self.open_cost = open_cost
        self.close_cost = close_cost

    def _get_regime(self, date) -> str:
        """获取 date 前一天的信号状态 (防前视偏差)"""
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

    def _load_prices(self) -> pd.DataFrame:
        from qlib.data import D
        instruments = D.instruments('csi300')
        prices = D.features(instruments, ['$open', '$close'],
                            start_time=TEST_START, end_time=TEST_END)
        prices = prices.swaplevel().sort_index()
        return prices

    def _load_benchmark(self) -> pd.Series:
        from qlib.data import D
        bench = D.features(['SH000300'], ['$close'],
                           start_time=TEST_START, end_time=TEST_END)
        bench_close = bench['$close'].droplevel(0)
        return bench_close.pct_change().dropna()

    def run(self) -> dict:
        """执行自适应回测"""
        prices = self._load_prices()
        trading_days = sorted(prices.index.get_level_values(0).unique())
        signal_days = set(self.signal.index.get_level_values(0).unique())

        cash = self.initial_cash
        positions = {}
        pending_sells = set()
        pending_buys = {}
        pending_cash_frac = 0.0
        n_trades = 0

        daily_values = []
        daily_regimes = []
        prev_close = {}

        for day_idx, date in enumerate(trading_days):
            if date not in prices.index.get_level_values(0):
                continue

            day_data = prices.loc[date]
            day_open = day_data['$open'].dropna()
            day_close = day_data['$close'].dropna()

            # --- 1. 执行昨天产生的交易指令 (今日开盘价) ---
            for inst in list(pending_sells):
                if inst not in positions:
                    pending_sells.discard(inst)
                    continue
                if inst not in day_open.index:
                    continue
                if inst in prev_close and prev_close[inst] > 0:
                    limit_ret = day_open[inst] / prev_close[inst] - 1
                    if limit_ret < -0.095:
                        continue
                price = day_open[inst]
                shares = positions[inst]['shares']
                proceeds = shares * price * (1 - self.close_cost)
                cash += proceeds
                del positions[inst]
                pending_sells.discard(inst)
                n_trades += 1

            # 买入 — 修改点: 仓位缩放
            if pending_buys:
                total_weight = sum(pending_buys.values())
                available_cash = cash * 0.99 * (1 - pending_cash_frac)
                if total_weight > 0:
                    for inst, weight in list(pending_buys.items()):
                        if inst in positions:
                            continue
                        if inst not in day_open.index:
                            continue
                        if inst in prev_close and prev_close[inst] > 0:
                            limit_ret = day_open[inst] / prev_close[inst] - 1
                            if limit_ret > 0.095:
                                continue
                        price = day_open[inst]
                        alloc = available_cash * (weight / total_weight)
                        shares = int(alloc / (price * (1 + self.open_cost)) / 100) * 100
                        if shares >= 100:
                            cost = shares * price * (1 + self.open_cost)
                            cash -= cost
                            positions[inst] = {'shares': shares, 'entry_price': price}
                            n_trades += 1
                pending_buys = {}
                pending_cash_frac = 0.0

            # --- 2. 收盘估值 ---
            port_value = cash
            for inst, pos in positions.items():
                if inst in day_close.index:
                    port_value += pos['shares'] * day_close[inst]
                else:
                    port_value += pos['shares'] * pos['entry_price']
            daily_values.append({'date': date, 'value': port_value})

            # --- 3. 收盘后检查止损 ---
            if self.stop_loss:
                for inst, pos in list(positions.items()):
                    if inst in day_close.index:
                        ret = day_close[inst] / pos['entry_price'] - 1
                        if ret < -self.stop_loss:
                            pending_sells.add(inst)

            # --- 4. 调仓日: 信号质量驱动 ---
            if day_idx % self.rebalance_every == 0 and date in signal_days:
                regime = self._get_regime(date)
                daily_regimes.append({'date': date, 'regime': regime})

                cfg = self.strategy_cfg
                effective_topk = cfg['topk'][regime]
                cash_fraction = cfg['cash_frac'][regime]

                day_signal = self.signal.loc[date]
                if isinstance(day_signal, pd.DataFrame):
                    day_signal = day_signal.iloc[:, 0]
                day_signal = day_signal.sort_values(ascending=False)
                target_stocks = set(day_signal.head(effective_topk).index)

                # 卖出不在目标中的持仓
                for inst in list(positions.keys()):
                    if inst not in target_stocks:
                        pending_sells.add(inst)

                # 规划买入 — 按信号分数降序排列，保证确定性迭代顺序。
                current_holds = set(positions.keys()) - pending_sells
                to_buy = [i for i in day_signal.head(effective_topk).index
                          if i not in current_holds]
                if to_buy:
                    if cfg['signal_weight']:
                        # 策略 C: 按信号值分配权重, max 15%
                        top_sig = day_signal.loc[to_buy]
                        if len(top_sig) > 0 and top_sig.max() > top_sig.min():
                            top_sig = top_sig - top_sig.min() + 1e-8
                            raw_w = top_sig / top_sig.sum()
                            raw_w = raw_w.clip(upper=0.15)
                            raw_w = raw_w / raw_w.sum()
                            pending_buys = raw_w.to_dict()
                        else:
                            pending_buys = {inst: 1.0 / effective_topk for inst in to_buy}
                    else:
                        weight = 1.0 / effective_topk
                        pending_buys = {inst: weight for inst in to_buy}
                    pending_cash_frac = cash_fraction

            for inst in day_close.index:
                prev_close[inst] = day_close[inst]

        # --- 计算绩效 ---
        return self._calculate_metrics(daily_values, n_trades, daily_regimes)

    def _calculate_metrics(self, daily_values, n_trades, daily_regimes) -> dict:
        df = pd.DataFrame(daily_values).set_index('date')
        df['return'] = df['value'].pct_change()
        returns = df['return'].dropna()

        total_ret = df['value'].iloc[-1] / df['value'].iloc[0] - 1
        n_days = len(returns)
        annual_ret = (1 + total_ret) ** (252 / max(n_days, 1)) - 1

        cumulative = (1 + returns).cumprod()
        drawdown = (cumulative - cumulative.cummax()) / cumulative.cummax()
        max_dd = float(drawdown.min())

        daily_std = float(returns.std())
        sharpe = float(returns.mean() / daily_std * (252 ** 0.5)) if daily_std > 0 else 0.0

        bench_returns = self._load_benchmark()
        common_dates = returns.index.intersection(bench_returns.index)
        bench_total = float((1 + bench_returns.loc[common_dates]).prod() - 1)

        # 统计 regime 分布
        regime_counts = {}
        if daily_regimes:
            rdf = pd.DataFrame(daily_regimes)
            regime_counts = rdf['regime'].value_counts().to_dict()

        return {
            'total_return': float(total_ret),
            'annual_return': float(annual_ret),
            'max_drawdown': max_dd,
            'sharpe': sharpe,
            'bench_return': bench_total,
            'excess_return': float(total_ret) - bench_total,
            'n_trades': n_trades,
            'regime_counts': regime_counts,
            'daily_values': df['value'].to_dict(),
        }


# ============ Part C: 回测矩阵 ============

def run_baseline(signal, topk, rebal=5, sl=0.08, prices_cache=None):
    """运行 baseline (无自适应) — 复用 PortfolioBacktester"""
    from factor_lab.run_execution_benchmark import PortfolioBacktester
    bt = PortfolioBacktester(
        signal=signal, topk=topk,
        rebalance_every=rebal, stop_loss=sl,
    )
    result = bt.run()
    return result


def run_backtest_matrix(signal, quality_score):
    """运行完整回测矩阵: 2 baselines + 6 adaptive"""
    results = []

    # 所有策略统一用 AdaptiveBacktester (baseline 用固定 topk/0 cash_frac)
    all_configs = [
        ('baseline_12', (0.3, 0.6), 'Baseline topk=12 weekly SL=8%'),
        ('baseline_20', (0.3, 0.6), 'Baseline topk=20 weekly SL=8%'),
        ('A_topk_adaptive', (0.3, 0.6), 'Adaptive A: TopK自适应'),
        ('A_topk_adaptive', (0.4, 0.7), 'Adaptive A: TopK自适应 保守阈值'),
        ('B_position_scale', (0.3, 0.6), 'Adaptive B: 仓位缩放'),
        ('B_position_scale', (0.4, 0.7), 'Adaptive B: 仓位缩放 保守阈值'),
        ('C_signal_weight', (0.3, 0.6), 'Adaptive C: 信号加权'),
        ('D_combined', (0.3, 0.6), 'Adaptive D: A+B联动'),
    ]

    for idx, (strategy, thresholds, name) in enumerate(all_configs, 1):
        print(f"  [{idx}/{len(all_configs)}] {name}...", end=" ", flush=True)
        t0 = time.time()
        bt = AdaptiveBacktester(
            signal=signal, quality_score=quality_score,
            strategy=strategy, thresholds=thresholds,
        )
        r = bt.run()
        r['name'] = name
        r['strategy'] = strategy
        # 保留 daily_values 用于 window 拆解
        results.append(r)
        regime_str = ", ".join(f"{k}={v}" for k, v in r.get('regime_counts', {}).items())
        print(f"Sharpe={r['sharpe']:.3f} MDD={r['max_drawdown']:.2%} [{regime_str}] ({time.time()-t0:.1f}s)")

    return results


# ============ Part D: Window 级拆解 ============

def window_level_analysis(results: list[dict], signal: pd.Series):
    """按 rolling window 拆解各策略的收益，重点看 Window 8"""
    windows = load_rolling_windows()
    print("\n=== Window 级收益拆解 ===")

    # 收集有 daily_values 的策略
    strategies_with_daily = [(r['name'], r['daily_values']) for r in results if 'daily_values' in r]
    # baseline 没有 daily_values，需要单独跑
    # 不过 baseline 结果已有，这里只分析 adaptive 策略

    window_results = []
    for w in windows:
        wnum = w['window_num']
        ws = pd.Timestamp(w['pred_start'])
        we = pd.Timestamp(w['pred_end'])

        row = {'window': wnum, 'period': f"{w['pred_start']}~{w['pred_end']}",
               'best_iter': w.get('best_iteration', 'N/A')}

        for name, dv in strategies_with_daily:
            # dv 是 {date_str: value} 或 {Timestamp: value}
            if isinstance(dv, dict):
                dv_series = pd.Series(dv)
                dv_series.index = pd.to_datetime(dv_series.index)
            else:
                dv_series = dv

            mask = (dv_series.index >= ws) & (dv_series.index <= we)
            window_vals = dv_series[mask]
            if len(window_vals) >= 2:
                w_ret = window_vals.iloc[-1] / window_vals.iloc[0] - 1
                w_returns = window_vals.pct_change().dropna()
                w_cum = (1 + w_returns).cumprod()
                w_dd = ((w_cum - w_cum.cummax()) / w_cum.cummax()).min()
                short_name = name.replace('Adaptive ', '').replace('Baseline ', 'BL_')[:20]
                row[f'{short_name}_ret'] = float(w_ret)
                row[f'{short_name}_mdd'] = float(w_dd)

        window_results.append(row)

    wdf = pd.DataFrame(window_results)
    return wdf


# ============ Part E: 报告 ============

def print_results_table(results: list[dict]):
    """打印对比表"""
    print("\n")
    print("=" * 140)
    print("              实验 008: 信号衰减检测与动态仓位管理 — 对比表")
    print("=" * 140)
    print(f"测试区间: {TEST_START} ~ {TEST_END} | Rolling: {CONFIG_NAME}")
    print()

    header = (f"{'#':<3} {'策略':<38} {'Sharpe':>8} {'总收益':>10} {'年化':>10} "
              f"{'超额':>10} {'最大回撤':>10} {'交易次数':>8}")
    print(header)
    print("-" * 140)

    sorted_results = sorted(results, key=lambda x: x.get('sharpe', 0), reverse=True)
    for i, r in enumerate(sorted_results, 1):
        trades_str = f"{r.get('n_trades', 'N/A'):>7}" if 'n_trades' in r else "    N/A"
        print(f"{i:<3} {r['name']:<38} {r.get('sharpe', 0):>7.3f} "
              f"{r.get('total_return', 0):>9.2%} {r.get('annual_return', 0):>9.2%} "
              f"{r.get('excess_return', 0):>9.2%} {r.get('max_drawdown', 0):>9.2%} "
              f"{trades_str}")

    print("=" * 140)


def save_results(analysis: dict, results: list[dict], window_df: pd.DataFrame):
    """保存结果到 results/signal_decay/"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 信号质量指标
    quality_df = pd.DataFrame({
        'quality_score': analysis['quality_score'],
        'regime': analysis['regime'],
        'ic_20': analysis['ic_20'],
        'ic_60': analysis['ic_60'],
        'dispersion': analysis['dispersion'],
        'spread': analysis['spread'],
    })
    quality_df.to_csv(RESULTS_DIR / "signal_quality_daily.csv")
    print(f"  信号质量日度数据: {RESULTS_DIR / 'signal_quality_daily.csv'}")

    # window 统计
    analysis['window_stats'].to_csv(RESULTS_DIR / "window_quality_stats.csv", index=False)
    print(f"  Window 质量统计: {RESULTS_DIR / 'window_quality_stats.csv'}")

    # 回测结果
    results_clean = []
    for r in results:
        rc = {k: v for k, v in r.items() if k != 'daily_values'}
        results_clean.append(rc)
    with open(RESULTS_DIR / "backtest_results.json", 'w', encoding='utf-8') as f:
        json.dump(results_clean, f, indent=2, ensure_ascii=False, default=str)
    print(f"  回测结果: {RESULTS_DIR / 'backtest_results.json'}")

    # window 级拆解
    if len(window_df) > 0:
        window_df.to_csv(RESULTS_DIR / "window_breakdown.csv", index=False)
        print(f"  Window 拆解: {RESULTS_DIR / 'window_breakdown.csv'}")

    # 质量缓存 (供后续复用)
    with open(RESULTS_DIR / "quality_score.pkl", 'wb') as f:
        pickle.dump(analysis['quality_score'], f)


# ============ 主流程 ============

def main():
    parser = argparse.ArgumentParser(description="实验 008: 信号衰减检测与动态仓位管理")
    parser.add_argument('--analysis-only', action='store_true', help="仅信号分析")
    parser.add_argument('--backtest-only', action='store_true', help="仅回测")
    args = parser.parse_args()

    # 初始化 Qlib
    import multiprocessing
    try:
        multiprocessing.set_start_method('fork', force=True)
    except (ValueError, RuntimeError):
        pass  # Windows 无 fork，使用默认 spawn
    import qlib
    from qlib.constant import REG_CN
    qlib.init(provider_uri='~/.qlib/qlib_data/cn_data_bs', region=REG_CN)

    t_total = time.time()

    # === Part A: 信号质量分析 ===
    quality_cache = RESULTS_DIR / "quality_score.pkl"

    if args.backtest_only and quality_cache.exists():
        print("\n=== Part A: 信号质量 (从缓存加载) ===")
        signal = load_predictions()
        with open(quality_cache, 'rb') as f:
            quality_score = pickle.load(f)
        analysis = None
        print(f"  quality_score: {len(quality_score)} days")
    else:
        print("\n=== Part A: 信号质量分析 ===")
        signal = load_predictions()
        print(f"  预测信号: {len(signal)} samples")
        close = load_close_prices()
        print(f"  收盘价: {len(close)} samples")
        analysis = run_signal_analysis(signal, close)
        quality_score = analysis['quality_score']

        # 打印 window 统计
        wdf = analysis['window_stats']
        if len(wdf) > 0:
            print("\n  Window 质量统计:")
            print(f"  {'Window':>6} {'best_iter':>10} {'q_mean':>8} {'q_median':>8} "
                  f"{'%strong':>8} {'%weak':>8}")
            print("  " + "-" * 60)
            for _, row in wdf.iterrows():
                bi = row.get('best_iter', 'N/A')
                bi_str = f"{int(bi):>10}" if pd.notna(bi) else "       N/A"
                print(f"  {int(row['window']):>6} {bi_str} {row['q_mean']:>8.3f} "
                      f"{row['q_median']:>8.3f} {row['pct_strong']:>7.1%} "
                      f"{row['pct_weak']:>7.1%}")

        # 缓存
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(quality_cache, 'wb') as f:
            pickle.dump(quality_score, f)

    if args.analysis_only:
        if analysis:
            save_results(analysis, [], pd.DataFrame())
        print(f"\n信号分析完毕 ({time.time()-t_total:.1f}s)")
        return

    # === Part B+C: 回测矩阵 ===
    print("\n=== Part B+C: 回测矩阵 (2 baselines + 6 adaptive) ===")
    results = run_backtest_matrix(signal, quality_score)

    # 打印总表
    print_results_table(results)

    # === Part D: Window 级拆解 ===
    print("\n=== Part D: Window 级拆解 ===")
    window_df = window_level_analysis(results, signal)

    if len(window_df) > 0:
        # 找 ret 列
        ret_cols = [c for c in window_df.columns if c.endswith('_ret')]
        if ret_cols:
            print(f"\n  {'Window':>6} {'best_iter':>10}", end="")
            for c in ret_cols:
                short = c.replace('_ret', '')[:14]
                print(f" {short:>14}", end="")
            print()
            print("  " + "-" * (18 + 15 * len(ret_cols)))
            for _, row in window_df.iterrows():
                bi = row.get('best_iter', 'N/A')
                bi_str = f"{int(bi):>10}" if isinstance(bi, (int, float)) and not pd.isna(bi) else "       N/A"
                print(f"  {int(row['window']):>6} {bi_str}", end="")
                for c in ret_cols:
                    val = row.get(c, float('nan'))
                    if pd.notna(val):
                        print(f" {val:>13.2%}", end="")
                    else:
                        print(f" {'N/A':>13}", end="")
                print()

    # === Part E: 保存 ===
    print("\n=== Part E: 保存结果 ===")
    if analysis:
        save_results(analysis, results, window_df)
    else:
        # backtest-only mode
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        results_clean = [{k: v for k, v in r.items() if k != 'daily_values'} for r in results]
        with open(RESULTS_DIR / "backtest_results.json", 'w', encoding='utf-8') as f:
            json.dump(results_clean, f, indent=2, ensure_ascii=False, default=str)
        if len(window_df) > 0:
            window_df.to_csv(RESULTS_DIR / "window_breakdown.csv", index=False)
        print(f"  回测结果: {RESULTS_DIR / 'backtest_results.json'}")

    elapsed = time.time() - t_total
    print(f"\n实验 008 完成! 总耗时 {elapsed:.1f}s ({elapsed/60:.1f}min)")


if __name__ == '__main__':
    main()
