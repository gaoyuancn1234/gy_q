#!/usr/bin/env python3
"""
实验: global_pool diverse 因子 × {LGB baseline, gate_mlp} 2×2 对比

组合:
  A) alpha158_val + LGB baseline         (已有结果)
  B) alpha158_val + gate_mlp             (已有结果)
  C) alpha158_val + diverse_pool + LGB   (本次运行)
  D) alpha158_val + diverse_pool + gate_mlp (本次运行)

关键: C/D 使用 global_pool.get_diverse_exprs() 获取因子，
      与 FactorMiner Phase D 的评估方式一致。

用法:
  python -m factor_lab.run_mined_gate_experiment
  python -m factor_lab.run_mined_gate_experiment --report-only
"""
import sys
import gc
import json
import time
import warnings
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

CONFIG_NAME = 'D_expand_3v_3r'
TEST_START = '2024-01-01'
TEST_END = '2026-02-05'
TOPK = 12
N_DROP = 3

RESULTS_DIR = PROJECT_DIR / "factor_lab" / "results" / "mined_gate"
POOL_FILE = PROJECT_DIR / "factor_lab" / "mining_results" / "global_factor_pool.json"


def load_diverse_factors() -> list[tuple[str, str]]:
    """从 run_011.json 加载原始 20 因子"""
    import json
    run_file = PROJECT_DIR / "factor_lab" / "mining_results" / "runs" / "run_011.json"
    with open(run_file) as f:
        data = json.load(f)
    factors = [(f['name'], f['expr']) for f in data['non_redundant']]
    print(f"  run_011 因子: {len(factors)} 个")
    for i, (name, _) in enumerate(factors):
        print(f"    [{i+1}] {name}")
    return factors


def build_extended_handler(diverse_factors, start_time, end_time,
                           fit_start_time, fit_end_time):
    """构建 alpha158_val + diverse_factors 的 handler (与 backtest_runner 一致)"""
    from factor_lab.factors.custom_handler import build_handler_from_exprs
    from factor_lab.factors.presets import FACTOR_PRESETS

    preset = FACTOR_PRESETS["alpha158_val"]
    extra = preset["extra_factors"]
    if callable(extra):
        extra = extra()
    extended_factors = extra + list(diverse_factors)

    handler, _ = build_handler_from_exprs(
        factor_exprs=extended_factors,
        start_time=start_time,
        end_time=end_time,
        fit_start_time=fit_start_time,
        fit_end_time=fit_end_time,
        instruments="csi300",
        include_alpha158=True,
    )
    return handler


def run_backtest(pred, topk=TOPK, n_drop=N_DROP):
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
            "freq": "day", "limit_threshold": 0.095, "deal_price": "open",
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


def run_diverse_lgb(diverse_factors):
    """C) alpha158_val + diverse_pool + LGB (rolling)"""
    from factor_lab.run_rolling_benchmark import (
        ROLLING_CONFIGS, generate_rolling_windows, build_model,
    )
    from qlib.data.dataset import DatasetH

    config = ROLLING_CONFIGS[CONFIG_NAME]
    windows = generate_rolling_windows(CONFIG_NAME, config, TEST_START, TEST_END)

    print(f"\n{'='*70}")
    print(f"  C) alpha158_val + diverse_pool ({len(diverse_factors)}因子) + LGB")
    print(f"  Windows: {len(windows)}")
    print(f"{'='*70}")

    all_preds = []
    t0 = time.time()

    for w in windows:
        print(f"\n  Window {w['window_num']}/{len(windows)}: "
              f"Pred [{w['pred_start']}, {w['pred_end']}]")

        handler = build_extended_handler(
            diverse_factors,
            start_time=w['train_start'], end_time=w['pred_end'],
            fit_start_time=w['train_start'], fit_end_time=w['train_end'],
        )
        dataset = DatasetH(handler=handler, segments={
            "train": (w['train_start'], w['train_end']),
            "valid": (w['valid_start'], w['valid_end']),
            "test": (w['pred_start'], w['pred_end']),
        })

        model, fit_kwargs = build_model('LightGBM')
        model.fit(dataset, **fit_kwargs)

        pred = model.predict(dataset)
        if isinstance(pred.index, pd.MultiIndex):
            dates = pred.index.get_level_values(0)
            mask = (dates >= pd.Timestamp(w['pred_start'])) & \
                   (dates <= pd.Timestamp(w['pred_end']))
            pred = pred[mask]

        best_iter = getattr(model.model, 'best_iteration', None)
        print(f"    best_iter={best_iter}, samples={len(pred)}")

        all_preds.append(pred)
        del handler, dataset, model
        gc.collect()

    combined = pd.concat(all_preds)
    if combined.index.duplicated().any():
        combined = combined[~combined.index.duplicated(keep='last')]

    bt = run_backtest(combined)
    bt['total_time'] = round(time.time() - t0, 1)
    bt['n_diverse_factors'] = len(diverse_factors)

    print(f"\n  C) diverse+LGB: Sharpe={bt['sharpe']:.3f}  Return={bt['total_return']:.2%}  "
          f"MDD={bt['max_drawdown']:.2%}  Time={bt['total_time']:.1f}s")

    return bt


def run_diverse_gate_mlp(diverse_factors):
    """D) alpha158_val + diverse_pool + gate_mlp (rolling)"""
    from factor_lab.run_rolling_benchmark import ROLLING_CONFIGS, generate_rolling_windows
    from factor_lab.run_gated_benchmark import (
        extract_features_labels, predict_nn,
    )
    from factor_lab.gated_factor_net import (
        GatedFactorNet, MarketStateComputer, build_group_mask,
        train_gated_model,
    )
    import torch

    config = ROLLING_CONFIGS[CONFIG_NAME]
    windows = generate_rolling_windows(CONFIG_NAME, config, TEST_START, TEST_END)

    print(f"\n{'='*70}")
    print(f"  D) alpha158_val + diverse_pool ({len(diverse_factors)}因子) + gate_mlp")
    print(f"  Windows: {len(windows)}")
    print(f"{'='*70}")

    all_preds = []
    t0 = time.time()

    for w in windows:
        print(f"\n  Window {w['window_num']}/{len(windows)}: "
              f"Pred [{w['pred_start']}, {w['pred_end']}]")

        handler = build_extended_handler(
            diverse_factors,
            start_time=w['train_start'], end_time=w['pred_end'],
            fit_start_time=w['train_start'], fit_end_time=w['train_end'],
        )

        train_feat, train_label = extract_features_labels(handler, w['train_start'], w['train_end'])
        valid_feat, valid_label = extract_features_labels(handler, w['valid_start'], w['valid_end'])
        test_feat, test_label = extract_features_labels(handler, w['pred_start'], w['pred_end'])

        feature_names = list(train_feat.columns)
        market_state_computer = MarketStateComputer(feature_names)

        all_feat = pd.concat([train_feat, valid_feat, test_feat])
        all_label = pd.concat([train_label, valid_label, test_label])
        market_states = market_state_computer.compute_all_dates_fast(all_feat, all_label)

        def get_state_array(feat_df):
            dates = feat_df.index.get_level_values(0)
            return np.array([market_states.loc[d].values for d in dates], dtype=np.float32)

        train_state = get_state_array(train_feat)
        valid_state = get_state_array(valid_feat)
        test_state = get_state_array(test_feat)

        train_factors = train_feat.values.astype(np.float32)
        valid_factors = valid_feat.values.astype(np.float32)
        test_factors = test_feat.values.astype(np.float32)

        for arr in [train_factors, valid_factors, test_factors,
                    train_state, valid_state, test_state]:
            np.nan_to_num(arr, copy=False)

        train_labels = train_label.values.astype(np.float32)
        valid_labels = valid_label.values.astype(np.float32)
        np.nan_to_num(train_labels, copy=False)
        np.nan_to_num(valid_labels, copy=False)

        group_mask = build_group_mask(feature_names)

        model, info = train_gated_model(
            train_state, train_factors, train_labels,
            valid_state, valid_factors, valid_labels,
            group_mask, verbose=True,
        )

        device = next(model.parameters()).device
        pred = predict_nn(model, test_state, test_factors, test_feat.index, device)
        print(f"    gate_mlp: best_epoch={info['best_epoch']}, samples={len(pred)}")

        all_preds.append(pred)
        del handler, model, train_feat, valid_feat, test_feat
        gc.collect()

    combined = pd.concat(all_preds)
    if combined.index.duplicated().any():
        combined = combined[~combined.index.duplicated(keep='last')]

    bt = run_backtest(combined)
    bt['total_time'] = round(time.time() - t0, 1)
    bt['n_diverse_factors'] = len(diverse_factors)

    print(f"\n  D) diverse+gate_mlp: Sharpe={bt['sharpe']:.3f}  Return={bt['total_return']:.2%}  "
          f"MDD={bt['max_drawdown']:.2%}  Time={bt['total_time']:.1f}s")

    return bt


def print_2x2_table(results: dict, n_diverse: int = 0):
    """打印 2×2 对比表"""
    print("\n\n")
    print("=" * 90)
    print("     因子集 × 模型 2×2 对比实验")
    print("=" * 90)
    print(f"  测试区间: {TEST_START} ~ {TEST_END} | Rolling: {CONFIG_NAME}")
    print(f"  回测: TopkDropout (topk={TOPK}, n_drop={N_DROP}) | 次日开盘 (T+1)")
    print(f"  C/D 使用 global_pool.get_diverse_exprs() ({n_diverse}因子)")
    print()

    header = f"{'组合':<6} {'因子集':<28} {'模型':<12} {'Sharpe':>8} {'总收益':>10} {'年化':>10} {'最大回撤':>10} {'超额':>10}"
    print(header)
    print("-" * 94)

    order = ['A', 'B', 'C', 'D']
    labels = {
        'A': ('alpha158_val (210)', 'LGB'),
        'B': ('alpha158_val (210)', 'gate_mlp'),
        'C': (f'alpha158_val+diverse ({210+n_diverse})', 'LGB'),
        'D': (f'alpha158_val+diverse ({210+n_diverse})', 'gate_mlp'),
    }

    sorted_keys = sorted(
        [k for k in order if k in results],
        key=lambda k: results[k].get('sharpe', 0), reverse=True
    )

    best_sharpe = max(r.get('sharpe', 0) for r in results.values()) if results else 0
    a_sharpe = results.get('A', {}).get('sharpe', 0)

    for k in sorted_keys:
        r = results[k]
        preset, model = labels[k]
        sharpe = r.get('sharpe', 0)
        delta = f"({sharpe - a_sharpe:+.3f})" if k != 'A' else ""
        marker = " ***" if sharpe == best_sharpe else ""
        print(f"  {k})  {preset:<28} {model:<12} {sharpe:>7.3f} {delta:>8} "
              f"{r.get('total_return', 0):>9.2%} "
              f"{r.get('annual_return', 0):>9.2%} "
              f"{r.get('max_drawdown', 0):>9.2%} "
              f"{r.get('excess_return', 0):>9.2%}{marker}")

    # Interaction analysis
    print()
    print("-" * 94)
    if all(k in results for k in ['A', 'B', 'C', 'D']):
        sa, sb, sc, sd = [results[k].get('sharpe', 0) for k in ['A', 'B', 'C', 'D']]
        mined_effect = sc - sa
        gate_effect = sb - sa
        combined_effect = sd - sa
        interaction = combined_effect - mined_effect - gate_effect
        print(f"  挖掘因子增量 (C-A):  Sharpe {mined_effect:+.3f}")
        print(f"  Gate 增量 (B-A):      Sharpe {gate_effect:+.3f}")
        print(f"  叠加增量 (D-A):       Sharpe {combined_effect:+.3f}")
        print(f"  交互效应:             Sharpe {interaction:+.3f}  "
              f"({'协同增强' if interaction > 0.01 else '互相抵消' if interaction < -0.01 else '独立叠加'})")

    print("\n" + "=" * 90)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report-only', action='store_true')
    parser.add_argument('--skip-lgb', action='store_true', help="跳过 LGB, 只跑 gate_mlp")
    parser.add_argument('--skip-gate', action='store_true', help="跳过 gate_mlp, 只跑 LGB")
    parser.add_argument('--force', action='store_true', help="强制重跑 (忽略缓存)")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result_file = RESULTS_DIR / "mined_gate_2x2.json"

    # 加载已有结果
    results = {}
    if result_file.exists() and not args.force:
        with open(result_file) as f:
            results = json.load(f)

    # 强制模式: 清除 C/D 旧结果
    if args.force:
        results.pop('C', None)
        results.pop('D', None)

    # 填充 A, B 从已有实验结果
    gated_file = PROJECT_DIR / "factor_lab" / "results" / "gated" / "gated_benchmark.json"
    if gated_file.exists():
        with open(gated_file) as f:
            gated = json.load(f)
        if 'baseline' in gated:
            results['A'] = gated['baseline']
        if 'gate_mlp' in gated:
            results['B'] = gated['gate_mlp']

    if args.report_only:
        n_div = results.get('C', results.get('D', {})).get('n_diverse_factors', 0)
        print_2x2_table(results, n_diverse=n_div)
        return

    # 初始化 Qlib
    import multiprocessing
    multiprocessing.set_start_method('fork', force=True)
    import qlib
    from qlib.constant import REG_CN
    qlib.init(provider_uri='~/.qlib/qlib_data/cn_data_bs', region=REG_CN)

    # 加载 diverse 因子
    diverse_factors = load_diverse_factors()

    # C) diverse + LGB
    if not args.skip_lgb:
        if 'C' not in results:
            results['C'] = run_diverse_lgb(diverse_factors)
            with open(result_file, 'w') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        else:
            print(f"\n  [跳过] C) diverse+LGB 已有结果: Sharpe={results['C']['sharpe']:.3f}")

    # D) diverse + gate_mlp
    if args.skip_gate:
        print(f"\n  [跳过] D) diverse+gate_mlp (--skip-gate)")
    elif 'D' not in results:
        results['D'] = run_diverse_gate_mlp(diverse_factors)
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    else:
        print(f"\n  [跳过] D) diverse+gate_mlp 已有结果: Sharpe={results['D']['sharpe']:.3f}")

    # 打印对比表
    print_2x2_table(results, n_diverse=len(diverse_factors))

    print(f"\n结果已保存: {result_file}")


if __name__ == '__main__':
    main()
