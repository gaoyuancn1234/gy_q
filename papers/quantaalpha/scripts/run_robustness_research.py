#!/usr/bin/env python3
"""Window 8 鲁棒性调研 — A/B1/B2 三路实验

Window 8 (2025Q4, valid=Q3 2025 +17.8% 牛市) 所有模型 best_iter=1-3，
根因是验证集大面积因子 IC 反转。本脚本从三条路径调研解决方案。

实验 A: Regime-Robust 因子筛选
  - 牛/熊/震荡三种行情分别计算因子 IC
  - 保留各 regime 下 IC 方向一致的因子子集

实验 B1: 6 个月长验证窗口
  - 延长验证窗口到 6 个月，覆盖更多 regime

实验 B2: Rank-IC 早停
  - 用 rank-IC 替代 MSE 做 LightGBM 早停指标

用法:
  python -m factor_lab.run_robustness_research --experiment A
  python -m factor_lab.run_robustness_research --experiment B1
  python -m factor_lab.run_robustness_research --experiment B2
  python -m factor_lab.run_robustness_research --compare
"""
import sys
import json
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

RESULTS_DIR = PROJECT_DIR / "factor_lab" / "results" / "robustness"


# ============================================================
# 实验 A: Regime-Robust 因子筛选
# ============================================================

def classify_market_regime(start="2018-01-01", end="2026-02-05", window=60,
                           bull_threshold=0.10, bear_threshold=-0.10):
    """CSI300 rolling 60日收益率 → bull/bear/neutral 分类

    Args:
        start, end: 时间范围
        window: 滚动窗口天数
        bull_threshold: 牛市阈值 (60日收益率 > 10%)
        bear_threshold: 熊市阈值 (60日收益率 < -10%)

    Returns:
        pd.Series: index=datetime, values='bull'/'bear'/'neutral'
    """
    from qlib.data import D

    close = D.features(
        ['SH000300'],
        ['$close'],
        start_time=start,
        end_time=end,
    )
    close = close.droplevel(0)['$close']

    rolling_ret = close / close.shift(window) - 1
    rolling_ret = rolling_ret.dropna()

    regime = pd.Series('neutral', index=rolling_ret.index)
    regime[rolling_ret > bull_threshold] = 'bull'
    regime[rolling_ret < bear_threshold] = 'bear'

    counts = regime.value_counts()
    print(f"\nRegime 分布 ({start} ~ {end}, window={window}):")
    for r in ['bull', 'bear', 'neutral']:
        n = counts.get(r, 0)
        pct = n / len(regime) * 100
        print(f"  {r:>8}: {n:>5} 天 ({pct:.1f}%)")

    return regime


def _compute_daily_ic(factor_expr, factor_name, instruments, start, end):
    """计算单个因子的日度截面 IC"""
    from qlib.data import D

    factor_data = D.features(
        instruments,
        [factor_expr],
        start_time=start,
        end_time=end,
    )
    label_data = D.features(
        instruments,
        ['Ref($close, -2)/Ref($close, -1) - 1'],
        start_time=start,
        end_time=end,
    )

    factor_data.columns = ['factor']
    label_data.columns = ['label']
    merged = factor_data.join(label_data, how='inner').dropna()

    if len(merged) == 0:
        return pd.Series(dtype=float)

    daily_ic = merged.groupby(level=0).apply(
        lambda g: g['factor'].corr(g['label'], method='spearman')
        if len(g) > 10 else np.nan
    )
    return daily_ic.dropna()


def evaluate_regime_robustness(factor_exprs, regime_series):
    """计算每个因子在 bull/bear/neutral 下的 IC/ICIR

    Args:
        factor_exprs: [(name, expr), ...] 扩展因子列表
        regime_series: classify_market_regime() 返回的 regime 分类

    Returns:
        DataFrame: factor, ICIR_bull, ICIR_bear, ICIR_neutral,
                   robust_score, is_consistent
    """
    from qlib.data import D

    instruments = D.instruments("csi300")
    start = regime_series.index.min().strftime('%Y-%m-%d')
    end = regime_series.index.max().strftime('%Y-%m-%d')

    results = []
    total = len(factor_exprs)

    for i, (name, expr) in enumerate(factor_exprs):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  评估因子 {i+1}/{total}: {name}")

        daily_ic = _compute_daily_ic(expr, name, instruments, start, end)
        if len(daily_ic) < 30:
            results.append({
                'factor': name, 'expr': expr,
                'ICIR_bull': np.nan, 'ICIR_bear': np.nan, 'ICIR_neutral': np.nan,
                'robust_score': 0.0, 'is_consistent': False,
            })
            continue

        regime_stats = {}
        for r in ['bull', 'bear', 'neutral']:
            regime_dates = regime_series[regime_series == r].index
            ic_r = daily_ic[daily_ic.index.isin(regime_dates)]
            if len(ic_r) > 10:
                ic_mean = ic_r.mean()
                ic_std = ic_r.std()
                icir = ic_mean / ic_std * (252 ** 0.5) if ic_std > 0 else 0.0
                regime_stats[r] = {'ic_mean': ic_mean, 'icir': icir, 'n': len(ic_r)}
            else:
                regime_stats[r] = {'ic_mean': 0.0, 'icir': 0.0, 'n': len(ic_r)}

        icir_bull = regime_stats['bull']['icir']
        icir_bear = regime_stats['bear']['icir']
        icir_neutral = regime_stats['neutral']['icir']

        # IC 方向一致性: 三个 regime 下符号相同 (同正或同负)
        signs = [np.sign(regime_stats[r]['ic_mean']) for r in ['bull', 'bear', 'neutral']]
        nonzero_signs = [s for s in signs if s != 0]
        is_consistent = (len(nonzero_signs) >= 2 and
                         all(s == nonzero_signs[0] for s in nonzero_signs))

        # robust_score = min(|ICIR|) across regimes
        abs_icirs = [abs(icir_bull), abs(icir_bear), abs(icir_neutral)]
        robust_score = min(abs_icirs) if is_consistent else 0.0

        results.append({
            'factor': name, 'expr': expr,
            'IC_bull': regime_stats['bull']['ic_mean'],
            'IC_bear': regime_stats['bear']['ic_mean'],
            'IC_neutral': regime_stats['neutral']['ic_mean'],
            'ICIR_bull': icir_bull,
            'ICIR_bear': icir_bear,
            'ICIR_neutral': icir_neutral,
            'robust_score': robust_score,
            'is_consistent': is_consistent,
        })

    df = pd.DataFrame(results)
    return df


def run_experiment_A():
    """A: Regime-robust 因子筛选"""
    print("\n" + "=" * 70)
    print("  实验 A: Regime-Robust 因子筛选")
    print("=" * 70)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Regime 分类
    regime = classify_market_regime()

    # 2. 获取 alpha158_val 的扩展因子列表
    from factor_lab.factors.presets import _get_selected_exprs
    from factor_lab.factors import fundamental

    ext_exprs = _get_selected_exprs() + fundamental.get_all_exprs()
    print(f"\n扩展因子总数: {len(ext_exprs)}")

    # 3. 评估 regime robustness
    print("\n开始 per-regime IC 评估...")
    df = evaluate_regime_robustness(ext_exprs, regime)

    # 4. 保存详细结果
    result_file = RESULTS_DIR / "regime_robustness.csv"
    df.to_csv(result_file, index=False)
    print(f"\n详细结果保存: {result_file}")

    # 5. 筛选 robust 因子
    consistent = df[df['is_consistent']]
    robust = consistent[consistent['robust_score'] > 0.3]
    robust = robust.sort_values('robust_score', ascending=False)

    print(f"\n筛选结果:")
    print(f"  总因子: {len(df)}")
    print(f"  IC 方向一致: {len(consistent)} ({len(consistent)/len(df)*100:.0f}%)")
    print(f"  Robust (score > 0.3): {len(robust)} ({len(robust)/len(df)*100:.0f}%)")

    if len(robust) > 0:
        print(f"\n  Top Robust 因子:")
        for _, row in robust.head(20).iterrows():
            print(f"    {row['factor']:<20} score={row['robust_score']:.3f}  "
                  f"ICIR: bull={row['ICIR_bull']:.2f} bear={row['ICIR_bear']:.2f} "
                  f"neutral={row['ICIR_neutral']:.2f}")

    # 6. 更新 robust 预设并跑 rolling benchmark
    robust_names = set(robust['factor'].tolist())
    if len(robust_names) < 5:
        print("\n  [警告] Robust 因子不足 5 个，降低阈值到 0.2")
        robust = consistent[consistent['robust_score'] > 0.2]
        robust = robust.sort_values('robust_score', ascending=False)
        robust_names = set(robust['factor'].tolist())

    print(f"\n  Robust 预设因子数: {len(robust_names)} (+ Alpha158 base)")

    from factor_lab.factors.presets import update_robust_factors
    update_robust_factors(robust_names)

    # 保存 robust 因子名单
    robust_meta = {
        'robust_factor_names': sorted(robust_names),
        'n_factors': len(robust_names),
        'threshold': 0.3 if len(robust_names) >= 5 else 0.2,
        'regime_window': 60,
    }
    with open(RESULTS_DIR / "robust_factors.json", 'w') as f:
        json.dump(robust_meta, f, indent=2, ensure_ascii=False)

    # 7. 跑 rolling benchmark
    print("\n开始 rolling benchmark (D_expand_3v_3r + alpha158_val_robust)...")
    from factor_lab.run_rolling_benchmark import (
        run_rolling_single, ROLLING_CONFIGS
    )

    result = run_rolling_single(
        preset="alpha158_val_robust",
        model_name="LightGBM",
        config_name="D_expand_3v_3r",
        config=ROLLING_CONFIGS["D_expand_3v_3r"],
        force=True,
    )

    # 保存实验 A 汇总
    summary = {
        'experiment': 'A',
        'description': 'Regime-Robust 因子筛选',
        'n_robust_factors': len(robust_names),
        'robust_factors': sorted(robust_names),
        'rolling_result': result,
    }
    with open(RESULTS_DIR / "experiment_A.json", 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    _print_single_result("A: Regime-Robust", result)
    return result


# ============================================================
# 实验 B1: 6 个月长验证窗口
# ============================================================

def run_experiment_B1():
    """B1: 6 个月验证窗口"""
    print("\n" + "=" * 70)
    print("  实验 B1: 6 个月长验证窗口")
    print("=" * 70)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    from factor_lab.run_rolling_benchmark import (
        run_rolling_single, ROLLING_CONFIGS
    )

    result = run_rolling_single(
        preset="alpha158_val",
        model_name="LightGBM",
        config_name="E_expand_6v_3r",
        config=ROLLING_CONFIGS["E_expand_6v_3r"],
        force=True,
    )

    summary = {
        'experiment': 'B1',
        'description': '6个月验证窗口',
        'rolling_result': result,
    }
    with open(RESULTS_DIR / "experiment_B1.json", 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    _print_single_result("B1: 6个月验证", result)
    return result


# ============================================================
# 实验 B2: Rank-IC 早停
# ============================================================

def run_experiment_B2():
    """B2: Rank-IC 早停"""
    print("\n" + "=" * 70)
    print("  实验 B2: Rank-IC 早停")
    print("=" * 70)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    from factor_lab.run_rolling_benchmark import (
        run_rolling_single, ROLLING_CONFIGS
    )

    result = run_rolling_single(
        preset="alpha158_val",
        model_name="LightGBM",
        config_name="D_expand_3v_3r",
        config=ROLLING_CONFIGS["D_expand_3v_3r"],
        force=True,
        variant="rank_ic",
    )

    summary = {
        'experiment': 'B2',
        'description': 'Rank-IC 早停',
        'rolling_result': result,
    }
    with open(RESULTS_DIR / "experiment_B2.json", 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    _print_single_result("B2: Rank-IC", result)
    return result


# ============================================================
# 对比汇总
# ============================================================

def _print_single_result(label, result):
    """打印单个实验结果"""
    if not result or 'overall' not in result:
        print(f"\n  {label}: 无结果")
        return

    o = result['overall']
    print(f"\n  {label}:")
    print(f"    Sharpe:  {o.get('sharpe', 0):.3f}")
    print(f"    Return:  {o.get('total_return', 0):.2%}")
    print(f"    MDD:     {o.get('max_drawdown', 0):.2%}")

    # Window 8 best_iter
    for w in result.get('windows', []):
        if w['window_num'] == 8:
            print(f"    W8 best_iter: {w.get('best_iteration', '?')}")
            break


def _load_baseline():
    """加载 baseline (D_expand_3v_3r + alpha158_val + LightGBM)"""
    baseline_file = (PROJECT_DIR / "factor_lab" / "results" / "rolling" /
                     "D_expand_3v_3r_alpha158_val_LightGBM.json")
    if baseline_file.exists():
        with open(baseline_file) as f:
            return json.load(f)
    return None


def compare_results():
    """汇总对比三个实验 + baseline"""
    print("\n" + "=" * 80)
    print("  Window 8 鲁棒性调研 — 三路实验对比")
    print("=" * 80)

    experiments = {}

    # Baseline
    baseline = _load_baseline()
    if baseline:
        experiments['Baseline'] = baseline

    # 三个实验
    for name, filename in [
        ('A: Robust', 'experiment_A.json'),
        ('B1: 6m Val', 'experiment_B1.json'),
        ('B2: RankIC', 'experiment_B2.json'),
    ]:
        fpath = RESULTS_DIR / filename
        if fpath.exists():
            with open(fpath) as f:
                data = json.load(f)
            experiments[name] = data.get('rolling_result', data)

    if not experiments:
        print("  没有找到任何实验结果")
        return

    # 打印对比表
    header = (f"{'实验':<16} {'Sharpe':>8} {'总收益':>10} {'MDD':>10} "
              f"{'W8 iter':>8} {'Windows':>8}")
    print(f"\n{header}")
    print("-" * 65)

    for name, result in experiments.items():
        if not result or 'overall' not in result:
            print(f"{name:<16} {'N/A':>8}")
            continue

        o = result['overall']
        w8_iter = '?'
        for w in result.get('windows', []):
            if w['window_num'] == 8:
                w8_iter = str(w.get('best_iteration', '?'))
                break

        print(f"{name:<16} {o.get('sharpe', 0):>7.3f} "
              f"{o.get('total_return', 0):>9.2%} "
              f"{o.get('max_drawdown', 0):>9.2%} "
              f"{w8_iter:>8} "
              f"{result.get('n_windows', '?'):>8}")

    print()

    # 成功标准检查
    if baseline and 'overall' in baseline:
        bl_sharpe = baseline['overall'].get('sharpe', 0)
        threshold = bl_sharpe * 0.9
        print(f"  成功标准:")
        print(f"    Baseline Sharpe: {bl_sharpe:.3f}")
        print(f"    90% 阈值: {threshold:.3f}")
        print(f"    W8 best_iter 目标: > 10")

        for name, result in experiments.items():
            if name == 'Baseline' or 'overall' not in result:
                continue
            sharpe = result['overall'].get('sharpe', 0)
            w8_iter = None
            for w in result.get('windows', []):
                if w['window_num'] == 8:
                    w8_iter = w.get('best_iteration')
                    break

            sharpe_ok = sharpe >= threshold
            iter_ok = w8_iter is not None and w8_iter > 10
            status = "PASS" if (sharpe_ok and iter_ok) else "FAIL"
            print(f"    {name:<16}: Sharpe {'OK' if sharpe_ok else 'LOW'} ({sharpe:.3f}), "
                  f"W8 iter {'OK' if iter_ok else 'LOW'} ({w8_iter}), "
                  f"→ {status}")

    print("=" * 80)


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Window 8 鲁棒性调研")
    parser.add_argument('--experiment', choices=['A', 'B1', 'B2'],
                        help="运行指定实验")
    parser.add_argument('--compare', action='store_true',
                        help="汇总对比所有实验结果")
    args = parser.parse_args()

    if args.compare:
        compare_results()
        return

    if not args.experiment:
        parser.print_help()
        return

    # 初始化 Qlib
    import multiprocessing
    multiprocessing.set_start_method('fork', force=True)

    import qlib
    from qlib.constant import REG_CN
    qlib.init(provider_uri='~/.qlib/qlib_data/cn_data_bs', region=REG_CN)

    if args.experiment == 'A':
        run_experiment_A()
    elif args.experiment == 'B1':
        run_experiment_B1()
    elif args.experiment == 'B2':
        run_experiment_B2()

    # 跑完后自动显示对比
    compare_results()


if __name__ == '__main__':
    main()
