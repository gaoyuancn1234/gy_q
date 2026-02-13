#!/usr/bin/env python3
"""数据泄露诊断脚本"""
import multiprocessing
import sys
from pathlib import Path


def main():
    sys.path.insert(0, str(Path(__file__).parent))

    import warnings
    warnings.filterwarnings('ignore')

    import qlib
    import pandas as pd
    import numpy as np
    from qlib.constant import REG_CN
    from qlib.contrib.model.gbdt import LGBModel
    from qlib.contrib.data.handler import Alpha158
    from qlib.data.dataset import DatasetH

    qlib.init(provider_uri='~/.qlib/qlib_data/cn_data/qlib_bin', region=REG_CN)

    config = {
        'train_start': '2020-01-01',
        'train_end': '2023-09-30',
        'valid_start': '2023-10-01',
        'valid_end': '2023-12-31',
        'test_start': '2024-01-01',
        'test_end': '2024-12-05',
    }

    print("=" * 60)
    print("数据泄露诊断")
    print("=" * 60)

    # ========== 检查 1: Handler fit 区间 ==========
    print("\n[检查1] Handler fit 区间 vs 数据区间")
    handler_config = {
        "start_time": config['train_start'],
        "end_time": config['test_end'],
        "fit_start_time": config['train_start'],
        "fit_end_time": config['train_end'],
        "instruments": "csi300",
    }
    print(f"  数据加载区间: {handler_config['start_time']} ~ {handler_config['end_time']}")
    print(f"  归一化拟合区间: {handler_config['fit_start_time']} ~ {handler_config['fit_end_time']}")
    if handler_config['fit_end_time'] < config['test_start']:
        print("  --> OK: fit_end_time 在 test_start 之前，归一化参数不包含测试集")
    else:
        print("  --> 警告: fit_end_time >= test_start，可能存在归一化泄露!")

    handler = Alpha158(**handler_config)

    # ========== 检查 2: Dataset segments 重叠 ==========
    print("\n[检查2] Dataset segments 是否重叠")
    segments = {
        "train": (config['train_start'], config['train_end']),
        "valid": (config['valid_start'], config['valid_end']),
        "test": (config['test_start'], config['test_end']),
    }
    print(f"  train: {segments['train']}")
    print(f"  valid: {segments['valid']}")
    print(f"  test:  {segments['test']}")

    if config['train_end'] < config['valid_start'] and config['valid_end'] < config['test_start']:
        print("  --> OK: 三个区间无重叠")
    else:
        print("  --> 警告: 区间有重叠!")

    dataset = DatasetH(handler=handler, segments=segments)

    # ========== 检查 3: 训练集/测试集样本数 ==========
    print("\n[检查3] 各 segment 样本数")
    for seg in ["train", "valid", "test"]:
        X = dataset.prepare(seg, col_set="feature")
        y = dataset.prepare(seg, col_set="label")
        print(f"  {seg}: X={X.shape}, y={y.shape}")
        date_range = X.index.get_level_values(0)
        print(f"    日期范围: {date_range.min()} ~ {date_range.max()}")

    # ========== 检查 4: model.predict 返回哪个 segment ==========
    print("\n[检查4] model.predict() 返回的数据区间")
    model = LGBModel(
        loss="mse", learning_rate=0.01, num_leaves=64,
        num_boost_round=500, early_stopping_rounds=80,
        feature_fraction=0.75, bagging_fraction=0.75,
        bagging_freq=5, lambda_l1=0.1, lambda_l2=0.1,
        min_data_in_leaf=80,
    )
    model.fit(dataset)
    pred = model.predict(dataset)

    pred_dates = pred.index.get_level_values(0)
    print(f"  预测结果日期范围: {pred_dates.min()} ~ {pred_dates.max()}")
    print(f"  预测样本总数: {len(pred)}")

    # 检查是否只有 test 区间
    train_end_ts = pd.Timestamp(config['train_end'])
    valid_end_ts = pd.Timestamp(config['valid_end'])
    test_start_ts = pd.Timestamp(config['test_start'])

    n_before_test = (pred_dates < test_start_ts).sum()
    print(f"  test_start 之前的预测数: {n_before_test}")
    if n_before_test > 0:
        print("  --> 警告: 预测结果包含 test 之前的数据!")
        print(f"     最早预测日期: {pred_dates.min()}")
        print(f"     这些预测是否参与了回测策略？需要进一步检查")
    else:
        print("  --> OK: 预测结果只包含 test 区间")

    # ========== 检查 5: 预测分布 - 是否过于自信 ==========
    print("\n[检查5] 预测分数分布")
    print(f"  mean={pred.values.mean():.6f}")
    print(f"  std={pred.values.std():.6f}")
    print(f"  min={pred.values.min():.6f}")
    print(f"  max={pred.values.max():.6f}")
    print(f"  median={np.median(pred.values):.6f}")

    # ========== 检查 6: 模型复杂度 - early stopping ==========
    print("\n[检查6] 模型复杂度")
    best_iter = model.model.best_iteration if hasattr(model.model, 'best_iteration') else 'N/A'
    print(f"  最佳迭代轮数: {best_iter} / 500")
    if isinstance(best_iter, int) and best_iter < 30:
        print("  --> 注意: 模型在很早期就 early stop，信号可能很弱")

    # ========== 检查 7: close 价成交的偏乐观性 ==========
    print("\n[检查7] 回测执行假设")
    print("  deal_price = 'close'")
    print("  --> 注意: 使用当日收盘价成交")
    print("     模型特征基于当日 close 计算，然后以 close 价成交")
    print("     这在实盘中不可行（无法在看到收盘价后再以收盘价成交）")
    print("     属于 '乐观执行假设'，非严格的数据泄露但会高估收益")

    # ========== 检查 8: 打乱预测分数，对比回测 ==========
    print("\n[检查8] 随机信号对照实验")
    from qlib.contrib.strategy import TopkDropoutStrategy
    from qlib.contrib.evaluate import backtest_daily
    from qlib.utils import init_instance_by_config

    backtest_config = {
        "start_time": config['test_start'],
        "end_time": config['test_end'],
        "account": 100_000_000,
        "benchmark": "SH000300",
        "exchange_kwargs": {
            "freq": "day",
            "limit_threshold": 0.095,
            "deal_price": "close",
            "open_cost": 0.0005,
            "close_cost": 0.0015,
            "min_cost": 5,
        },
    }

    # 原始信号回测
    strategy_config = {
        "class": "TopkDropoutStrategy",
        "module_path": "qlib.contrib.strategy",
        "kwargs": {"signal": pred, "topk": 12, "n_drop": 3},
    }
    strategy = init_instance_by_config(strategy_config)
    report_real, _ = backtest_daily(strategy=strategy, executor=None, **backtest_config)
    ret_real = (1 + report_real['return']).prod() - 1

    # 打乱信号回测（保持 index 不变，值随机排列）
    np.random.seed(42)
    pred_shuffled = pred.copy()
    for date in pred_shuffled.index.get_level_values(0).unique():
        mask = pred_shuffled.index.get_level_values(0) == date
        vals = pred_shuffled.loc[mask].values.copy()
        np.random.shuffle(vals)
        pred_shuffled.loc[mask] = vals

    strategy_config_shuf = {
        "class": "TopkDropoutStrategy",
        "module_path": "qlib.contrib.strategy",
        "kwargs": {"signal": pred_shuffled, "topk": 12, "n_drop": 3},
    }
    strategy_shuf = init_instance_by_config(strategy_config_shuf)
    report_shuf, _ = backtest_daily(strategy=strategy_shuf, executor=None, **backtest_config)
    ret_shuf = (1 + report_shuf['return']).prod() - 1

    bench_ret = (1 + report_real['bench']).prod() - 1

    print(f"  基准(沪深300): {bench_ret:.2%}")
    print(f"  原始信号收益:  {ret_real:.2%}  (超额: {ret_real - bench_ret:.2%})")
    print(f"  随机信号收益:  {ret_shuf:.2%}  (超额: {ret_shuf - bench_ret:.2%})")
    diff = ret_real - ret_shuf
    print(f"  模型 vs 随机差异: {diff:.2%}")
    if abs(diff) < 0.03:
        print("  --> 警告: 模型与随机信号差异 < 3%，模型选股能力存疑")
    else:
        print("  --> 模型相比随机信号有一定区分度")

    print("\n" + "=" * 60)
    print("诊断完成")
    print("=" * 60)


if __name__ == '__main__':
    multiprocessing.set_start_method('fork', force=True)
    main()
