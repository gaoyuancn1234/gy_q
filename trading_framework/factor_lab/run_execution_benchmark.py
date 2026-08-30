#!/usr/bin/env python3
"""
执行层优化回测 — 多模型融合、分散持仓、周度调仓、止损

测试矩阵:
A. 标准回测 (Qlib TopkDropout, 日度调仓)
   - 信号源: LGB / Ensemble_EQ / Ensemble_WT
   - TopK: 12 / 20 / 30

B. 自定义回测 (可配置调仓频率 + 止损)
   - 调仓频率: daily(1) / weekly(5)
   - 止损: None / -8%
   - 持仓数: 12 / 20

用法:
  # 完整运行 (含预测生成 ~25分钟 + 回测 ~1分钟)
  python -m factor_lab.run_execution_benchmark

  # 仅跑回测 (需已有预测缓存)
  python -m factor_lab.run_execution_benchmark --backtest-only

  # 仅生成预测缓存
  python -m factor_lab.run_execution_benchmark --predict-only
"""
import sys
import gc
import json
import time
import pickle
import warnings
import argparse
from pathlib import Path
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

# 与 benchmark_models.py / run_rolling_benchmark.py 一致
TEST_START = '2024-01-01'
TEST_END = '2026-02-05'

# 缓存和结果目录
CACHE_DIR = PROJECT_DIR / "factor_lab" / "results" / "rolling" / "predictions"
RESULTS_DIR = PROJECT_DIR / "factor_lab" / "results" / "execution"

# Rolling 配置 (仅用 D_expand)
CONFIG_NAME = 'D_expand_3v_3r'

# 从 run_rolling_benchmark 导入工具函数
from factor_lab.run_rolling_benchmark import (
    ROLLING_CONFIGS, generate_rolling_windows, build_model,
)


# ============ Part 1: 预测生成与缓存 ============

def get_rolling_predictions(model_name: str, preset: str = 'alpha158_val',
                            config_name: str = CONFIG_NAME) -> pd.Series:
    """生成或加载缓存的 rolling 预测

    Returns:
        pd.Series with MultiIndex (datetime, instrument)
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{config_name}_{preset}_{model_name}.pkl"

    if cache_path.exists():
        print(f"  [缓存] {cache_path.name}")
        return pd.read_pickle(cache_path)

    print(f"  [生成] {config_name} × {preset} × {model_name}")
    config = ROLLING_CONFIGS[config_name]
    windows = generate_rolling_windows(config_name, config, TEST_START, TEST_END)

    all_preds = []
    for w in windows:
        wnum = w['window_num']
        print(f"    Window {wnum}/{len(windows)}...", end=" ", flush=True)
        t0 = time.time()

        # 构建 handler
        if preset == 'alpha158':
            from qlib.contrib.data.handler import Alpha158
            handler = Alpha158(
                start_time=w['train_start'], end_time=w['pred_end'],
                fit_start_time=w['train_start'], fit_end_time=w['train_end'],
                instruments='csi300',
            )
        else:
            from factor_lab.factors.presets import build_handler
            handler = build_handler(
                preset,
                start_time=w['train_start'], end_time=w['pred_end'],
                fit_start_time=w['train_start'], fit_end_time=w['train_end'],
            )

        from qlib.data.dataset import DatasetH
        dataset = DatasetH(handler=handler, segments={
            "train": (w['train_start'], w['train_end']),
            "valid": (w['valid_start'], w['valid_end']),
            "test": (w['pred_start'], w['pred_end']),
        })

        model, fit_kwargs = build_model(model_name)
        model.fit(dataset, **fit_kwargs)

        pred = model.predict(dataset)
        if isinstance(pred.index, pd.MultiIndex):
            dates = pred.index.get_level_values(0)
            mask = (dates >= pd.Timestamp(w['pred_start'])) & \
                   (dates <= pd.Timestamp(w['pred_end']))
            pred = pred[mask]

        elapsed = time.time() - t0
        print(f"{len(pred)} samples, {elapsed:.1f}s")

        all_preds.append(pred)
        del handler, dataset, model, pred
        gc.collect()

    combined = pd.concat(all_preds)
    if combined.index.duplicated().any():
        combined = combined[~combined.index.duplicated(keep='last')]

    combined.to_pickle(cache_path)
    print(f"  [保存] {cache_path.name} ({len(combined)} samples)")
    return combined


# ============ Part 2: 信号融合 ============

def ensemble_predictions(preds: dict[str, pd.Series],
                         weights: dict[str, float] | None = None) -> pd.Series:
    """加权融合多个模型的预测信号

    Args:
        preds: {model_name: prediction_series}
        weights: {model_name: weight}, None = 等权

    Returns:
        融合后的 pd.Series
    """
    if weights is None:
        weights = {k: 1.0 / len(preds) for k in preds}

    # 对齐到共同 index
    common_idx = None
    for pred in preds.values():
        if common_idx is None:
            common_idx = pred.index
        else:
            common_idx = common_idx.intersection(pred.index)

    result = pd.Series(0.0, index=common_idx)
    for name, pred in preds.items():
        result += weights[name] * pred.loc[common_idx]

    return result


# ============ Part 3: 标准 Qlib 回测 ============

def run_standard_backtest(pred, topk=12, n_drop=3) -> dict:
    """标准 TopkDropout 回测 (复用 run_rolling_benchmark 的逻辑)"""
    from qlib.contrib.evaluate import backtest_daily
    from qlib.utils import init_instance_by_config

    strategy_config = {
        "class": "TopkDropoutStrategy",
        "module_path": "qlib.contrib.strategy",
        "kwargs": {"signal": pred, "topk": topk, "n_drop": n_drop},
    }
    backtest_config = {
        "start_time": TEST_START, "end_time": TEST_END,
        "account": 100_000_000, "benchmark": "SH000300",
        "exchange_kwargs": {
            "freq": "day", "limit_threshold": 0.095, "deal_price": "close",  # 2026-08-30 由 open 改为 close (对齐生产 signal_config)
            "open_cost": 0.0005, "close_cost": 0.0015, "min_cost": 5, "trade_unit": 100,
        },
    }
    strategy = init_instance_by_config(strategy_config)
    report, positions = backtest_daily(strategy=strategy, executor=None, **backtest_config)

    result = {}
    if 'return' in report.columns:
        returns = report['return']
        total_ret = (1 + returns).prod() - 1
        result['total_return'] = float(total_ret)
        n_days = len(returns)
        result['annual_return'] = float((1 + total_ret) ** (252 / max(n_days, 1)) - 1)
        cumulative = (1 + returns).cumprod()
        drawdown = (cumulative - cumulative.cummax()) / cumulative.cummax()
        result['max_drawdown'] = float(drawdown.min())
        daily_std = float(returns.std())
        result['sharpe'] = float(returns.mean() / daily_std * (252 ** 0.5)) if daily_std > 0 else 0.0
    if 'bench' in report.columns:
        result['bench_return'] = float((1 + report['bench']).prod() - 1)
        result['excess_return'] = result.get('total_return', 0) - result['bench_return']
    return result


# ============ Part 4: 自定义回测引擎 ============

class PortfolioBacktester:
    """自定义回测器 — 支持周度调仓、止损、分散持仓

    执行模型 (T+1 合规):
    - Day T 收盘: 生成信号 / 检查止损 → 产生交易指令
    - Day T+1 开盘: 执行交易
    """

    def __init__(self, signal: pd.Series, topk: int = 20,
                 rebalance_every: int = 5, stop_loss: float | None = None,
                 initial_cash: float = 100_000_000,
                 open_cost: float = 0.0005, close_cost: float = 0.0015):
        self.signal = signal
        self.topk = topk
        self.rebalance_every = rebalance_every
        self.stop_loss = stop_loss
        self.initial_cash = initial_cash
        self.open_cost = open_cost
        self.close_cost = close_cost

    def _load_prices(self) -> pd.DataFrame:
        """从 Qlib 加载价格数据，返回 (datetime, instrument) 格式"""
        from qlib.data import D
        instruments = D.instruments('csi300')
        prices = D.features(instruments, ['$open', '$close'],
                            start_time=TEST_START, end_time=TEST_END)
        # D.features 返回 (instrument, datetime)，转为 (datetime, instrument) 与 signal 一致
        prices = prices.swaplevel().sort_index()
        return prices

    def _load_benchmark(self) -> pd.Series:
        """加载基准收益"""
        from qlib.data import D
        bench = D.features(
            ['SH000300'], ['$close'],
            start_time=TEST_START, end_time=TEST_END,
        )
        # D.features 返回 (instrument, datetime)，取 close 并去掉 instrument 层
        bench_close = bench['$close'].droplevel(0)
        return bench_close.pct_change().dropna()

    def run(self) -> dict:
        """执行回测"""
        prices = self._load_prices()
        trading_days = sorted(prices.index.get_level_values(0).unique())
        signal_days = set(self.signal.index.get_level_values(0).unique())

        cash = self.initial_cash
        positions = {}  # inst -> {shares, entry_price}
        pending_sells = set()
        pending_buys = {}  # inst -> target_weight
        n_trades = 0

        daily_values = []
        prev_close = {}  # inst -> previous close (for limit check)

        for day_idx, date in enumerate(trading_days):
            if date not in prices.index.get_level_values(0):
                continue

            day_data = prices.loc[date]
            day_open = day_data['$open'].dropna()
            day_close = day_data['$close'].dropna()

            # --- 1. 执行昨天产生的交易指令 (今日开盘价) ---

            # 卖出
            for inst in list(pending_sells):
                if inst not in positions:
                    pending_sells.discard(inst)
                    continue
                if inst not in day_open.index:
                    continue
                # 涨跌停检查: 跌停不能卖
                if inst in prev_close and prev_close[inst] > 0:
                    limit_ret = day_open[inst] / prev_close[inst] - 1
                    if limit_ret < -0.095:
                        continue  # 跌停，无法卖出

                price = day_open[inst]
                shares = positions[inst]['shares']
                proceeds = shares * price * (1 - self.close_cost)
                cash += proceeds
                del positions[inst]
                pending_sells.discard(inst)
                n_trades += 1

            # 买入
            if pending_buys:
                # 先计算可用资金
                total_weight = sum(pending_buys.values())
                available_cash = cash * 0.99  # 预留 1% 防止精度问题
                if total_weight > 0:
                    for inst, weight in list(pending_buys.items()):
                        if inst in positions:
                            continue  # 已持有
                        if inst not in day_open.index:
                            continue
                        # 涨停不能买
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
                            positions[inst] = {
                                'shares': shares,
                                'entry_price': price,
                            }
                            n_trades += 1
                pending_buys = {}

            # --- 2. 收盘估值 ---
            port_value = cash
            for inst, pos in positions.items():
                if inst in day_close.index:
                    port_value += pos['shares'] * day_close[inst]
                else:
                    # 无收盘价的用买入价估值
                    port_value += pos['shares'] * pos['entry_price']

            daily_values.append({'date': date, 'value': port_value})

            # --- 3. 收盘后检查止损 ---
            if self.stop_loss:
                for inst, pos in list(positions.items()):
                    if inst in day_close.index:
                        ret = day_close[inst] / pos['entry_price'] - 1
                        if ret < -self.stop_loss:
                            pending_sells.add(inst)

            # --- 4. 调仓日: 生成交易指令 ---
            if day_idx % self.rebalance_every == 0 and date in signal_days:
                day_signal = self.signal.loc[date]
                if isinstance(day_signal, pd.DataFrame):
                    day_signal = day_signal.iloc[:, 0]
                day_signal = day_signal.sort_values(ascending=False)
                target_stocks = set(day_signal.head(self.topk).index)

                # 卖出不在目标中的持仓
                for inst in list(positions.keys()):
                    if inst not in target_stocks:
                        pending_sells.add(inst)

                # 规划买入 (去掉已持有和待卖出)
                current_holds = set(positions.keys()) - pending_sells
                to_buy = target_stocks - current_holds
                if to_buy:
                    weight = 1.0 / self.topk
                    pending_buys = {inst: weight for inst in to_buy}

            # 记录当日收盘价 (供明日涨跌停检查)
            for inst in day_close.index:
                prev_close[inst] = day_close[inst]

        # --- 计算绩效指标 ---
        return self._calculate_metrics(daily_values, n_trades)

    def _calculate_metrics(self, daily_values: list[dict], n_trades: int) -> dict:
        """计算绩效指标"""
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

        # 基准收益
        bench_returns = self._load_benchmark()
        # 对齐到相同日期
        common_dates = returns.index.intersection(bench_returns.index)
        bench_total = float((1 + bench_returns.loc[common_dates]).prod() - 1)

        return {
            'total_return': float(total_ret),
            'annual_return': float(annual_ret),
            'max_drawdown': max_dd,
            'sharpe': sharpe,
            'bench_return': bench_total,
            'excess_return': float(total_ret) - bench_total,
            'n_trades': n_trades,
        }


# ============ Part 5: 主流程 ============

def print_results_table(results: list[dict]):
    """打印对比表"""
    print("\n\n")
    print("=" * 130)
    print("                          执行层优化回测 — 对比表")
    print("=" * 130)
    print(f"测试区间: {TEST_START} ~ {TEST_END} | Rolling: {CONFIG_NAME}")
    print()

    header = (f"{'#':<3} {'实验':<36} {'Sharpe':>8} {'总收益':>10} {'年化':>10} "
              f"{'超额':>10} {'最大回撤':>10} {'交易次数':>8}")
    print(header)
    print("-" * 130)

    sorted_results = sorted(results, key=lambda x: x.get('sharpe', 0), reverse=True)
    for i, r in enumerate(sorted_results, 1):
        trades_str = f"{r['n_trades']:>7}" if 'n_trades' in r else "    N/A"
        print(f"{i:<3} {r['name']:<36} {r.get('sharpe', 0):>7.3f} "
              f"{r.get('total_return', 0):>9.2%} {r.get('annual_return', 0):>9.2%} "
              f"{r.get('excess_return', 0):>9.2%} {r.get('max_drawdown', 0):>9.2%} "
              f"{trades_str}")

    print("\n" + "=" * 130)


def main():
    parser = argparse.ArgumentParser(description="执行层优化回测")
    parser.add_argument('--predict-only', action='store_true', help="仅生成预测缓存")
    parser.add_argument('--backtest-only', action='store_true', help="仅跑回测 (需已有缓存)")
    parser.add_argument('--custom-only', action='store_true', help="仅跑自定义回测 (跳过标准回测)")
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

    # --- Step 1: 生成预测 ---
    print("\n=== Step 1: 生成 Rolling 预测 ===")
    models = ['LightGBM', 'XGBoost', 'CatBoost']
    preds = {}

    for m in models:
        preds[m] = get_rolling_predictions(m)

    if args.predict_only:
        print("预测生成完毕。")
        return

    # --- Step 2: 构建融合信号 ---
    print("\n=== Step 2: 构建融合信号 ===")

    ensemble_eq = ensemble_predictions(preds)
    print(f"  Ensemble_EQ: {len(ensemble_eq)} samples (等权 1/3)")

    ensemble_wt = ensemble_predictions(preds, weights={
        'LightGBM': 0.5, 'CatBoost': 0.3, 'XGBoost': 0.2,
    })
    print(f"  Ensemble_WT: {len(ensemble_wt)} samples (0.5/0.3/0.2)")

    # 所有信号源
    all_signals = {
        'LGB': preds['LightGBM'],
        'Ensemble_EQ': ensemble_eq,
        'Ensemble_WT': ensemble_wt,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    standard_cache = RESULTS_DIR / "standard_results.json"
    custom_cache = RESULTS_DIR / "custom_results.json"

    # --- Step 3: 标准 Qlib 回测 (日度 TopkDropout) ---
    all_results = []

    if args.custom_only and standard_cache.exists():
        print("\n=== Step 3: 标准回测 (从缓存加载) ===")
        with open(standard_cache) as f:
            std_results = json.load(f)
        all_results.extend(std_results)
        print(f"  已加载 {len(std_results)} 条标准回测结果")
    else:
        print("\n=== Step 3: 标准回测 (Qlib TopkDropout) ===")
        standard_configs = [
            ('LGB topk=12 daily', 'LGB', 12, 3),
            ('Ensemble_EQ topk=12 daily', 'Ensemble_EQ', 12, 3),
            ('Ensemble_WT topk=12 daily', 'Ensemble_WT', 12, 3),
            ('LGB topk=20 daily', 'LGB', 20, 5),
            ('Ensemble_EQ topk=20 daily', 'Ensemble_EQ', 20, 5),
            ('Ensemble_WT topk=20 daily', 'Ensemble_WT', 20, 5),
            ('LGB topk=30 daily', 'LGB', 30, 8),
            ('Ensemble_EQ topk=30 daily', 'Ensemble_EQ', 30, 8),
        ]

        std_results = []
        for name, sig_key, topk, n_drop in standard_configs:
            print(f"  {name}...", end=" ", flush=True)
            t0 = time.time()
            r = run_standard_backtest(all_signals[sig_key], topk=topk, n_drop=n_drop)
            elapsed = time.time() - t0
            r['name'] = name
            std_results.append(r)
            print(f"Sharpe={r['sharpe']:.3f} Return={r['total_return']:.2%} "
                  f"MDD={r['max_drawdown']:.2%} ({elapsed:.1f}s)")

        # 中间保存
        with open(standard_cache, 'w') as f:
            json.dump(std_results, f, indent=2, ensure_ascii=False)
        print(f"  标准回测结果已保存: {standard_cache}")
        all_results.extend(std_results)

    # --- Step 4: 自定义回测 (周度调仓 + 止损) ---
    print("\n=== Step 4: 自定义回测 (周度调仓 + 止损) ===")

    custom_configs = [
        # (name, signal_key, topk, rebalance_every, stop_loss)
        ('LGB topk=12 weekly', 'LGB', 12, 5, None),
        ('LGB topk=12 weekly SL=8%', 'LGB', 12, 5, 0.08),
        ('LGB topk=20 weekly', 'LGB', 20, 5, None),
        ('LGB topk=20 weekly SL=8%', 'LGB', 20, 5, 0.08),
        ('LGB topk=20 weekly SL=5%', 'LGB', 20, 5, 0.05),
        ('Ens_EQ topk=20 weekly SL=8%', 'Ensemble_EQ', 20, 5, 0.08),
        ('Ens_WT topk=20 weekly SL=8%', 'Ensemble_WT', 20, 5, 0.08),
        ('Ens_WT topk=20 daily SL=8%', 'Ensemble_WT', 20, 1, 0.08),
    ]

    cust_results = []
    for name, sig_key, topk, rebal, sl in custom_configs:
        print(f"  {name}...", end=" ", flush=True)
        t0 = time.time()
        bt = PortfolioBacktester(
            signal=all_signals[sig_key],
            topk=topk,
            rebalance_every=rebal,
            stop_loss=sl,
        )
        r = bt.run()
        elapsed = time.time() - t0
        r['name'] = name
        cust_results.append(r)
        print(f"Sharpe={r['sharpe']:.3f} Return={r['total_return']:.2%} "
              f"MDD={r['max_drawdown']:.2%} Trades={r['n_trades']} ({elapsed:.1f}s)")

    # 中间保存
    with open(custom_cache, 'w') as f:
        json.dump(cust_results, f, indent=2, ensure_ascii=False)
    print(f"  自定义回测结果已保存: {custom_cache}")
    all_results.extend(cust_results)

    # --- Step 5: 汇总 ---
    print_results_table(all_results)

    # 保存完整结果
    result_file = RESULTS_DIR / "execution_benchmark.json"
    with open(result_file, 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n完整结果已保存: {result_file}")


if __name__ == '__main__':
    main()
