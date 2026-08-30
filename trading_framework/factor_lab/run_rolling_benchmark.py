#!/usr/bin/env python3
"""
Rolling Training Benchmark — Walk-Forward 滚动训练对比实验

对比不同 rolling 配置 × 因子预设 × 模型的表现，与 single-shot baseline 比较。

用法:
  # 快速测试 (1 config × 1 preset × 1 model)
  python -m factor_lab.run_rolling_benchmark --configs A_4yr_3v_3r --presets alpha158_val --models LightGBM

  # 核心对比 (4 configs × alpha158_val × LightGBM)
  python -m factor_lab.run_rolling_benchmark --presets alpha158_val --models LightGBM

  # 全量 (4 configs × 3 presets × 3 models)
  python -m factor_lab.run_rolling_benchmark

  # 强制重跑已有结果
  python -m factor_lab.run_rolling_benchmark --force
"""
import sys
import gc
import json
import time
import warnings
import argparse
import traceback
from pathlib import Path
from datetime import date
from dateutil.relativedelta import relativedelta

import pandas as pd

warnings.filterwarnings('ignore')

# 项目路径
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

# 测试区间 (与 benchmark_models.py 一致)
TEST_START = '2024-01-01'
TEST_END = '2026-02-05'
RESULT_SUFFIX = ''   # 由 --tag 设置，区分不同测试期的结果文件
# 成交价: 'close'=次日收盘 (默认), 'open'=次日开盘。由 --deal-price 覆盖。
#
# 2026-08-30 由 open 改为 close，理由 (one-switch 实验，其余变量全部固定):
#     open  Sharpe 1.643  超额 27.11%
#     close Sharpe 1.332  超额 11.90%   ← 超额腰斩
# 开盘价是实盘最难兑现的价格 (集合竞价冲击大、滑点高)，按开盘价成交属乐观
# 假设；且生产配置 signal_config.yaml 本就是 deal_price: close，研究口径
# 必须与之一致，否则挖掘会用生产不采用的执行假设去评判因子。
DEAL_PRICE = 'close'
# 已使用过的 tag，用于把带 tag 的结果排除出默认对比表。
# 新增 tag 时必须登记在这里，否则不同测试期/口径的结果会混进同一张表 ——
# 表头写着当前区间、数字却来自别的区间，极易误判。
KNOWN_TAGS = ('holdout', 'closeprice', 'openprice',
              'windowcmp', 'winseg1', 'seg2pred', 'pre2024')

# 回测参数 (与 benchmark_models.py 一致)
TOPK = 12
N_DROP = 3

# Rolling 配置矩阵
ROLLING_CONFIGS = {
    "A_4yr_3v_3r": {
        "description": "4年滑动训练, 3个月验证, 3个月重训",
        "train_years": 4,
        "valid_months": 3,
        "retrain_months": 3,
        "expanding": False,
    },
    "B_4yr_3v_6r": {
        "description": "4年滑动训练, 3个月验证, 6个月重训",
        "train_years": 4,
        "valid_months": 3,
        "retrain_months": 6,
        "expanding": False,
    },
    "F_2yr_3v_3r": {
        "description": "2年滑动训练, 3个月验证, 3个月重训 (2026-08-30 新增)",
        # 加入原因: 窗口越短表现越好的趋势 (2024-01~2026-02 测试期):
        #   D_expand 1.332 -> A_4yr 1.609 -> C_3yr 1.943
        # 需要探明拐点在哪，否则不知道 3 年是最优还是仍未触底。
        "train_years": 2,
        "valid_months": 3,
        "retrain_months": 3,
        "expanding": False,
    },
    "C_3yr_3v_3r": {
        "description": "3年滑动训练, 3个月验证, 3个月重训",
        "train_years": 3,
        "valid_months": 3,
        "retrain_months": 3,
        "expanding": False,
    },
    "D_expand_3v_3r": {
        "description": "扩展窗口训练, 3个月验证, 3个月重训",
        "train_years": 4,  # 初始窗口大小, 之后持续扩展
        "valid_months": 3,
        "retrain_months": 3,
        "expanding": True,
    },
    "E_expand_6v_3r": {
        "description": "扩展窗口训练, 6个月验证, 3个月重训",
        "train_years": 4,
        "valid_months": 6,
        "retrain_months": 3,
        "expanding": True,
    },
}

# 默认参数
DEFAULT_CONFIGS = list(ROLLING_CONFIGS.keys())
DEFAULT_PRESETS = ['alpha158_val', 'alpha158', 'full']
DEFAULT_MODELS = ['LightGBM', 'XGBoost', 'CatBoost']

# 结果目录
RESULTS_DIR = PROJECT_DIR / "factor_lab" / "results" / "rolling"


def generate_rolling_windows(config_name: str, config: dict,
                             test_start: str, test_end: str) -> list[dict]:
    """生成 walk-forward 滚动窗口列表

    窗口从 test_start 的预测期开始，向后推导训练/验证期。
    每个窗口的 pred 区间长度 = retrain_months，最后一个窗口截断到 test_end。

    Args:
        config_name: 配置名
        config: 配置字典 (train_years, valid_months, retrain_months, expanding)
        test_start: 测试起始日
        test_end: 测试结束日

    Returns:
        list[dict]: 每个窗口的时间信息
    """
    ts = pd.Timestamp(test_start)
    te = pd.Timestamp(test_end)

    train_years = config['train_years']
    valid_months = config['valid_months']
    retrain_months = config['retrain_months']
    expanding = config['expanding']

    # 扩展窗口模式的固定起始点
    expand_start = ts - relativedelta(months=valid_months) - relativedelta(years=train_years)

    windows = []
    cursor = ts
    window_num = 0

    while cursor < te:
        window_num += 1

        # 预测区间
        pred_start = cursor
        pred_end_candidate = cursor + relativedelta(months=retrain_months) - relativedelta(days=1)
        pred_end = min(pred_end_candidate, te)

        # 验证区间: pred_start 前的 valid_months
        valid_end = pred_start - relativedelta(days=1)
        valid_start = pred_start - relativedelta(months=valid_months)

        # 训练区间
        train_end = valid_start - relativedelta(days=1)
        if expanding:
            train_start = expand_start
        else:
            train_start = valid_start - relativedelta(years=train_years)

        fmt = lambda d: d.strftime('%Y-%m-%d')
        windows.append({
            "window_num": window_num,
            "train_start": fmt(train_start),
            "train_end": fmt(train_end),
            "valid_start": fmt(valid_start),
            "valid_end": fmt(valid_end),
            "pred_start": fmt(pred_start),
            "pred_end": fmt(pred_end),
        })

        cursor = cursor + relativedelta(months=retrain_months)

    return windows


def _rank_ic_feval(preds, train_data):
    """LightGBM 自定义 feval: 返回 rank IC (越大越好)

    metric="None" → 进入 self.params → 禁用默认 MSE metric
    feval → 通过 LGBModel.fit(**kwargs) → lgb.train(feval=...) 透传
    """
    labels = train_data.get_label()
    ic = pd.Series(preds).corr(pd.Series(labels), method='spearman')
    if pd.isna(ic):
        ic = 0.0
    return 'rank_ic', ic, True  # (name, value, is_higher_better)


def build_model(model_name: str, variant: str = "default"):
    """创建模型实例 (超参与 benchmark_models.py 一致)

    Args:
        model_name: 模型名
        variant: "default" 或 "rank_ic" (rank-IC 早停)

    Returns:
        (model, fit_kwargs) 元组
    """
    if model_name == 'LightGBM':
        from qlib.contrib.model.gbdt import LGBModel
        if variant == "rank_ic":
            model = LGBModel(
                loss="mse", metric="None",
                learning_rate=0.01, num_leaves=64,
                num_boost_round=500, early_stopping_rounds=80,
                feature_fraction=0.75, bagging_fraction=0.75, bagging_freq=5,
                lambda_l1=0.1, lambda_l2=0.1, min_data_in_leaf=80,
            )
            return model, {"feval": _rank_ic_feval}
        model = LGBModel(
            loss="mse", learning_rate=0.01, num_leaves=64,
            num_boost_round=500, early_stopping_rounds=80,
            feature_fraction=0.75, bagging_fraction=0.75, bagging_freq=5,
            lambda_l1=0.1, lambda_l2=0.1, min_data_in_leaf=80,
        )
        return model, {}

    elif model_name == 'XGBoost':
        from qlib.contrib.model.xgboost import XGBModel
        model = XGBModel(
            learning_rate=0.01, max_depth=6,
            n_estimators=500, early_stopping_rounds=80,
            reg_alpha=0.1, reg_lambda=0.1,
            subsample=0.75, colsample_bytree=0.75,
        )
        return model, {}

    elif model_name == 'CatBoost':
        from qlib.contrib.model.catboost_model import CatBoostModel
        model = CatBoostModel(
            loss="RMSE", learning_rate=0.01, depth=6,
            iterations=500, l2_leaf_reg=3.0, subsample=0.75,
        )
        return model, {"early_stopping_rounds": 80}

    else:
        raise ValueError(f"未知模型: {model_name}")


def run_backtest(pred):
    """统一回测逻辑 (复用 benchmark_models.py 的逻辑)"""
    from qlib.contrib.evaluate import backtest_daily
    from qlib.utils import init_instance_by_config

    strategy_config = {
        "class": "TopkDropoutStrategy",
        "module_path": "qlib.contrib.strategy",
        "kwargs": {"signal": pred, "topk": TOPK, "n_drop": N_DROP},
    }
    backtest_config = {
        "start_time": TEST_START, "end_time": TEST_END,
        "account": 100_000_000, "benchmark": "SH000300",
        "exchange_kwargs": {
            "freq": "day", "limit_threshold": 0.095, "deal_price": DEAL_PRICE,
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


def run_rolling_single(preset: str, model_name: str,
                       config_name: str, config: dict,
                       force: bool = False,
                       variant: str = "default") -> dict | None:
    """执行一个 (preset × model × config) 的滚动训练

    Args:
        preset: 因子预设名
        model_name: 模型名
        config_name: rolling 配置名
        config: rolling 配置字典
        force: 是否强制重跑
        variant: 模型变体 ("default" 或 "rank_ic")

    Returns:
        结果字典, 或 None (已有结果且非 force)
    """
    suffix = f"_{variant}" if variant != "default" else ""
    if RESULT_SUFFIX:
        suffix += f"_{RESULT_SUFFIX}"
    result_file = RESULTS_DIR / f"{config_name}_{preset}_{model_name}{suffix}.json"

    if result_file.exists() and not force:
        print(f"  [跳过] 已有结果: {result_file.name}")
        with open(result_file) as f:
            return json.load(f)

    windows = generate_rolling_windows(config_name, config, TEST_START, TEST_END)
    print(f"\n{'='*70}")
    print(f"  Config: {config_name} ({config['description']})")
    print(f"  Preset: {preset} | Model: {model_name}")
    print(f"  Windows: {len(windows)}")
    print(f"{'='*70}")

    # 打印窗口时间线
    for w in windows:
        print(f"  Window {w['window_num']}: "
              f"Train [{w['train_start']}, {w['train_end']}]  "
              f"Valid [{w['valid_start']}, {w['valid_end']}]  "
              f"Pred [{w['pred_start']}, {w['pred_end']}]")

    all_preds = []
    window_details = []
    total_start = time.time()

    for w in windows:
        wnum = w['window_num']
        print(f"\n  --- Window {wnum}/{len(windows)} ---")

        t0 = time.time()

        # 1. 构建 handler
        if preset == 'alpha158':
            from qlib.contrib.data.handler import Alpha158
            handler = Alpha158(
                start_time=w['train_start'],
                end_time=w['pred_end'],
                fit_start_time=w['train_start'],
                fit_end_time=w['train_end'],
                instruments='csi300',
            )
        else:
            from factor_lab.factors.presets import build_handler
            handler = build_handler(
                preset,
                start_time=w['train_start'],
                end_time=w['pred_end'],
                fit_start_time=w['train_start'],
                fit_end_time=w['train_end'],
            )

        # 2. 构建 DatasetH
        from qlib.data.dataset import DatasetH
        dataset = DatasetH(handler=handler, segments={
            "train": (w['train_start'], w['train_end']),
            "valid": (w['valid_start'], w['valid_end']),
            "test": (w['pred_start'], w['pred_end']),
        })

        # 3. 构建模型并训练
        model, fit_kwargs = build_model(model_name, variant=variant)
        model.fit(dataset, **fit_kwargs)

        # 4. 预测并过滤到 pred 区间
        pred = model.predict(dataset)
        if isinstance(pred.index, pd.MultiIndex):
            dates = pred.index.get_level_values(0)
            mask = (dates >= pd.Timestamp(w['pred_start'])) & \
                   (dates <= pd.Timestamp(w['pred_end']))
            pred = pred[mask]

        elapsed = time.time() - t0

        # 记录窗口信息
        # CatBoost uses 'best_iteration_', others use 'best_iteration'
        best_iter = getattr(model.model, 'best_iteration', None)
        if best_iter is None:
            best_iter = getattr(model.model, 'best_iteration_', None)

        detail = {
            "window_num": wnum,
            "train_start": w['train_start'],
            "train_end": w['train_end'],
            "pred_start": w['pred_start'],
            "pred_end": w['pred_end'],
            "n_samples": int(len(pred)),
            "best_iteration": best_iter,
            "train_time": round(elapsed, 1),
        }
        window_details.append(detail)
        all_preds.append(pred)

        print(f"    Best iter: {best_iter}, Samples: {len(pred)}, Time: {elapsed:.1f}s")

        # 5. 释放内存
        del handler, dataset, model, pred
        gc.collect()

    # 拼接所有窗口预测
    combined_pred = pd.concat(all_preds)

    # 检查无重复 index
    if combined_pred.index.duplicated().any():
        n_dup = combined_pred.index.duplicated().sum()
        print(f"  [警告] 发现 {n_dup} 个重复 index, 保留最后一个窗口的预测")
        combined_pred = combined_pred[~combined_pred.index.duplicated(keep='last')]

    print(f"\n  总预测样本: {len(combined_pred)}")

    # 落盘预测: 事后做 IC/归因分析必需 (原先只在内存中用完即弃，
    # 想查"某段时间为何表现差"就得整轮重训)
    pred_dir = RESULTS_DIR / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_path = pred_dir / f"{config_name}_{preset}_{model_name}{suffix}.pkl"
    combined_pred.to_pickle(pred_path)
    print(f"  预测已保存: {pred_path.name}")

    # 回测
    print("  运行回测...")
    bt_result = run_backtest(combined_pred)
    total_time = time.time() - total_start

    result = {
        "config_name": config_name,
        "config_description": config['description'],
        "preset": preset,
        "model": model_name,
        "variant": variant,
        "n_windows": len(windows),
        "windows": window_details,
        "overall": bt_result,
        "total_time": round(total_time, 1),
        "timestamp": pd.Timestamp.now().isoformat(),
    }

    # 保存结果
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n  结果: Sharpe={bt_result.get('sharpe', 0):.3f}  "
          f"Return={bt_result.get('total_return', 0):.2%}  "
          f"MDD={bt_result.get('max_drawdown', 0):.2%}  "
          f"Time={total_time:.1f}s")
    print(f"  保存: {result_file.name}")

    return result


def print_comparison_table(all_results: list[dict]):
    """打印对比表"""
    print("\n\n")
    print("=" * 110)
    print("                     Rolling Training Benchmark 对比表")
    print("=" * 110)
    print(f"测试区间: {TEST_START} ~ {TEST_END} | 基准: 沪深300")
    _dp = "次日开盘" if DEAL_PRICE == "open" else "次日收盘"
    print(f"策略: TopkDropout (topk={TOPK}, n_drop={N_DROP}) | "
          f"成交价: {_dp} ({DEAL_PRICE}, T+1)")
    print()

    # 按 Sharpe 降序排列
    sorted_results = sorted(all_results,
                            key=lambda x: x['overall'].get('sharpe', 0),
                            reverse=True)

    header = (f"{'排名':<4} {'Config':<16} {'Preset':<16} {'Model':<12} {'Variant':<10} "
              f"{'Windows':>7} {'总收益':>10} {'年化':>10} {'超额':>10} "
              f"{'Sharpe':>8} {'最大回撤':>10} {'耗时(s)':>8}")
    print(header)
    print("-" * 120)

    for rank, r in enumerate(sorted_results, 1):
        o = r['overall']
        print(f"{rank:<4} {r['config_name']:<16} {r['preset']:<16} {r['model']:<12} "
              f"{r.get('variant', 'default'):<10} "
              f"{r['n_windows']:>7} {o.get('total_return', 0):>9.2%} "
              f"{o.get('annual_return', 0):>9.2%} "
              f"{o.get('excess_return', 0):>9.2%} "
              f"{o.get('sharpe', 0):>7.3f} "
              f"{o.get('max_drawdown', 0):>9.2%} "
              f"{r.get('total_time', 0):>7.1f}")

    # Single-shot baseline 参考
    print()
    print("--- Single-Shot Baseline (参考) ---")
    baseline_file = PROJECT_DIR / "benchmark_results" / "results.json"
    if baseline_file.exists():
        with open(baseline_file) as f:
            baselines = json.load(f)
        for model_name in ['LightGBM', 'XGBoost', 'CatBoost']:
            if model_name in baselines and 'total_return' in baselines[model_name]:
                b = baselines[model_name]
                print(f"     {'single-shot':<16} {'alpha158_val':<16} {model_name:<12} "
                      f"{'1':>7} {b.get('total_return', 0):>9.2%} "
                      f"{b.get('annual_return', 0):>9.2%} "
                      f"{b.get('excess_return', 0):>9.2%} "
                      f"{b.get('sharpe', 0):>7.3f} "
                      f"{b.get('max_drawdown', 0):>9.2%} "
                      f"{b.get('train_time', 0):>7.1f}")
    else:
        print("  (未找到 benchmark_results/results.json)")

    print("\n" + "=" * 110)


def load_all_results() -> list[dict]:
    """加载当前测试期对应的结果

    只读与 RESULT_SUFFIX 匹配的文件：否则切换测试期(--test-start/--test-end)
    时会把旧测试期的结果混进对比表，表头写着新区间、数字却是旧的，
    极易误判。
    """
    results = []
    if not RESULTS_DIR.exists():
        return results
    for f in sorted(RESULTS_DIR.glob("*.json")):
        stem = f.stem
        if RESULT_SUFFIX:
            if not stem.endswith(f"_{RESULT_SUFFIX}"):
                continue
        elif any(stem.endswith(f"_{t}") for t in KNOWN_TAGS):
            continue  # 默认测试期不应混入带 tag 的结果
        with open(f) as fp:
            results.append(json.load(fp))
    return results


def main():
    global TEST_START, TEST_END, RESULT_SUFFIX, DEAL_PRICE
    parser = argparse.ArgumentParser(description="Rolling Training Benchmark")
    parser.add_argument('--configs', nargs='+', default=DEFAULT_CONFIGS,
                        choices=list(ROLLING_CONFIGS.keys()),
                        help=f"Rolling 配置 (默认: 全部)")
    parser.add_argument('--presets', nargs='+', default=DEFAULT_PRESETS,
                        help=f"因子预设 (默认: {DEFAULT_PRESETS})")
    parser.add_argument('--models', nargs='+', default=DEFAULT_MODELS,
                        choices=DEFAULT_MODELS,
                        help=f"模型 (默认: {DEFAULT_MODELS})")
    parser.add_argument('--force', action='store_true',
                        help="强制重跑已有结果")
    parser.add_argument('--report-only', action='store_true',
                        help="只打印对比表, 不跑实验")
    parser.add_argument('--test-start', default=None,
                        help=f"覆盖测试期起点 (默认 {TEST_START})，用于样本外holdout检验")
    parser.add_argument('--test-end', default=None,
                        help=f"覆盖测试期终点 (默认 {TEST_END})")
    parser.add_argument('--tag', default='',
                        help="结果文件名后缀，避免覆盖默认测试期的结果")
    parser.add_argument('--deal-price', default=None, choices=['open', 'close'],
                        help="成交价 (默认 open=次日开盘)。one-switch 实验用: "
                             "固定其他所有变量，只切换此项以隔离执行假设的影响")
    args = parser.parse_args()

    if args.deal_price:
        DEAL_PRICE = args.deal_price

    # 覆盖测试区间（挖掘的评估期 = 默认测试期，故重测默认期属样本内；
    # 用 --test-start/--test-end 指定挖掘未见过的区间才是真正的样本外检验）
    if args.test_start:
        TEST_START = args.test_start
    if args.test_end:
        TEST_END = args.test_end
    RESULT_SUFFIX = args.tag

    if args.report_only:
        all_results = load_all_results()
        if all_results:
            print_comparison_table(all_results)
        else:
            print("没有找到任何结果文件")
        return

    # 初始化 Qlib
    import multiprocessing
    try:
        multiprocessing.set_start_method('fork', force=True)
    except (ValueError, RuntimeError):
        pass  # Windows 无 fork，使用默认 spawn

    import qlib
    from qlib.constant import REG_CN
    qlib.init(provider_uri='~/.qlib/qlib_data/cn_data_bs', region=REG_CN)

    # 计算总任务数
    total = len(args.configs) * len(args.presets) * len(args.models)
    print(f"\n{'='*70}")
    print(f"Rolling Training Benchmark")
    print(f"Configs: {args.configs}")
    print(f"Presets: {args.presets}")
    print(f"Models:  {args.models}")
    print(f"总任务: {total}")
    print(f"{'='*70}")

    all_results = []
    task_num = 0

    for config_name in args.configs:
        config = ROLLING_CONFIGS[config_name]
        for preset in args.presets:
            for model_name in args.models:
                task_num += 1
                print(f"\n\n[Task {task_num}/{total}]")

                try:
                    result = run_rolling_single(
                        preset, model_name, config_name, config,
                        force=args.force,
                    )
                    if result:
                        all_results.append(result)
                except Exception as e:
                    print(f"  [失败] {e}")
                    traceback.print_exc()
                    error_result = {
                        "config_name": config_name,
                        "preset": preset,
                        "model": model_name,
                        "error": str(e),
                        "overall": {},
                        "n_windows": 0,
                        "total_time": 0,
                    }
                    all_results.append(error_result)

    # 本次运行的失败必须显式暴露，不能被"打印历史结果"掩盖
    failed = [r for r in all_results if r.get("error")]

    # 打印对比表 (只含当前测试期的结果)
    all_results = load_all_results()
    if all_results:
        print_comparison_table(all_results)

    if failed:
        print(f"\n⚠ 本次运行有 {len(failed)} 个任务失败，上表不含它们的结果:")
        for r in failed:
            print(f"    {r['config_name']} / {r['preset']} / {r['model']}: {r['error']}")
        sys.exit(1)


if __name__ == '__main__':
    main()
