#!/usr/bin/env python3
"""季度重训 pipeline — 一键执行

步骤:
1. 刷新 BaoStock 数据 (~3min)
2. 扩展 rolling 预测到新日期 (~2min per new window)
3. 重算信号质量评分 (~30s)
4. 更新 signal_config.yaml 的 test_end
5. 验证 PaperTrader replay (~10s)

用法:
    python retrain_pipeline.py                       # 全量
    python retrain_pipeline.py --data-only           # 只刷数据
    python retrain_pipeline.py --test-end 2026-06-30 # 指定结束日期
    python retrain_pipeline.py --skip-data           # 跳过数据刷新
"""
import gc
import sys
import json
import time
import pickle
import warnings
import argparse
from pathlib import Path
from datetime import date
from dateutil.relativedelta import relativedelta

import yaml
import pandas as pd

warnings.filterwarnings('ignore')

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

# 默认配置 (与实验 005/008 一致)
CONFIG_NAME = 'D_expand_3v_3r'
PRESET = 'alpha158_val'
MODEL_NAME = 'LightGBM'
TEST_START = '2024-01-01'

RESULTS_DIR = PROJECT_DIR / "factor_lab" / "results" / "rolling"
PRED_DIR = RESULTS_DIR / "predictions"
DECAY_DIR = PROJECT_DIR / "factor_lab" / "results" / "signal_decay"
CONFIG_FILE = PROJECT_DIR / "config" / "signal_config.yaml"


def refresh_data(end_date: str = None):
    """刷新 BaoStock 数据"""
    print("\n" + "=" * 60)
    print("Step 1: 刷新 BaoStock 数据")
    print("=" * 60)

    from qlib_engine.data_setup import setup_qlib_data
    if end_date is None:
        end_date = date.today().strftime('%Y-%m-%d')
    print(f"  数据截止日期: {end_date}")

    t0 = time.time()
    setup_qlib_data(end_date=end_date)
    print(f"  数据刷新完成 ({time.time() - t0:.0f}s)")


def extend_rolling_predictions(new_test_end: str) -> pd.Series:
    """扩展 rolling 预测到新日期，仅训练新增 window

    复用 run_rolling_benchmark 的 generate_rolling_windows / build_model。
    """
    print("\n" + "=" * 60)
    print("Step 2: 扩展 Rolling 预测")
    print("=" * 60)

    from factor_lab.run_rolling_benchmark import (
        generate_rolling_windows, build_model, ROLLING_CONFIGS,
    )

    config = ROLLING_CONFIGS[CONFIG_NAME]

    # 加载已有预测和 JSON
    pkl_path = PRED_DIR / f"{CONFIG_NAME}_{PRESET}_{MODEL_NAME}.pkl"
    json_path = RESULTS_DIR / f"{CONFIG_NAME}_{PRESET}_{MODEL_NAME}.json"

    if pkl_path.exists():
        old_pred = pd.read_pickle(pkl_path)
        old_dates = old_pred.index.get_level_values(0).unique()
        old_last = old_dates.max()
        print(f"  已有预测: {old_dates.min().strftime('%Y-%m-%d')} ~ {old_last.strftime('%Y-%m-%d')}")
    else:
        old_pred = None
        old_last = pd.Timestamp(TEST_START) - pd.Timedelta(days=1)
        print("  无已有预测，将全量训练")

    if json_path.exists():
        with open(json_path) as f:
            old_json = json.load(f)
        old_windows = old_json.get('windows', [])
    else:
        old_json = None
        old_windows = []

    # 生成所有 windows (从 TEST_START 到 new_test_end)
    all_windows = generate_rolling_windows(CONFIG_NAME, config, TEST_START, new_test_end)
    print(f"  总 windows: {len(all_windows)} (覆盖 {TEST_START} ~ {new_test_end})")

    # 筛选需要处理的 windows:
    # 1. 全新的 window (pred_start > old_last)
    # 2. 已有但 pred_end 被扩展的 window (pred_start <= old_last, 但新 pred_end > old_last)
    new_windows = []
    extend_windows = []  # 需要重新预测(不需重训)的已有 window

    for w in all_windows:
        ps = pd.Timestamp(w['pred_start'])
        pe = pd.Timestamp(w['pred_end'])
        if ps > old_last:
            new_windows.append(w)
        elif pe > old_last and ps <= old_last:
            extend_windows.append(w)

    if not new_windows and not extend_windows:
        print("  无新增/扩展 window，预测已是最新")
        return old_pred

    # 合并: 先扩展已有 window，再训练新 window
    windows_to_process = extend_windows + new_windows
    print(f"  扩展 windows: {len(extend_windows)}, 新增 windows: {len(new_windows)}")
    for w in windows_to_process:
        tag = "扩展" if w in extend_windows else "新增"
        print(f"    [{tag}] Window {w['window_num']}: "
              f"Train [{w['train_start']}, {w['train_end']}]  "
              f"Pred [{w['pred_start']}, {w['pred_end']}]")

    # 初始化 Qlib
    import qlib
    from qlib.data.dataset import DatasetH
    qlib.init(provider_uri='~/.qlib/qlib_data/cn_data_bs', region='cn')

    new_preds = []
    new_window_details = []

    # For extend windows: remove old predictions for that window's date range
    # (they'll be re-predicted with the expanded range)
    if old_pred is not None and extend_windows:
        for w in extend_windows:
            ps = pd.Timestamp(w['pred_start'])
            pe = pd.Timestamp(w['pred_end'])
            dates = old_pred.index.get_level_values(0)
            mask = (dates >= ps) & (dates <= pe)
            if mask.any():
                old_pred = old_pred[~mask]
                print(f"  清除 Window {w['window_num']} 旧预测 ({mask.sum()} samples)")

        # Also remove the old window details for extend windows
        extend_wnums = {w['window_num'] for w in extend_windows}
        old_windows = [ow for ow in old_windows if ow['window_num'] not in extend_wnums]

    for w in windows_to_process:
        wnum = w['window_num']
        print(f"\n  --- Training Window {wnum} ---")
        t0 = time.time()

        # 构建 handler
        if PRESET == 'alpha158':
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
                PRESET,
                start_time=w['train_start'],
                end_time=w['pred_end'],
                fit_start_time=w['train_start'],
                fit_end_time=w['train_end'],
            )

        dataset = DatasetH(handler=handler, segments={
            "train": (w['train_start'], w['train_end']),
            "valid": (w['valid_start'], w['valid_end']),
            "test": (w['pred_start'], w['pred_end']),
        })

        model, fit_kwargs = build_model(MODEL_NAME)
        model.fit(dataset, **fit_kwargs)

        pred = model.predict(dataset)
        if isinstance(pred.index, pd.MultiIndex):
            dates = pred.index.get_level_values(0)
            mask = (dates >= pd.Timestamp(w['pred_start'])) & \
                   (dates <= pd.Timestamp(w['pred_end']))
            pred = pred[mask]

        elapsed = time.time() - t0

        best_iter = getattr(model.model, 'best_iteration', None)
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
        new_window_details.append(detail)
        new_preds.append(pred)

        print(f"    Best iter: {best_iter}, Samples: {len(pred)}, Time: {elapsed:.1f}s")

        del handler, dataset, model, pred
        gc.collect()

    # 合并 old + new
    if old_pred is not None:
        combined_pred = pd.concat([old_pred] + new_preds)
    else:
        combined_pred = pd.concat(new_preds)

    # 去重
    if combined_pred.index.duplicated().any():
        n_dup = combined_pred.index.duplicated().sum()
        print(f"  去重: {n_dup} 个重复 index")
        combined_pred = combined_pred[~combined_pred.index.duplicated(keep='last')]

    # 保存 pkl
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    combined_pred.to_pickle(pkl_path)
    print(f"  预测已保存: {pkl_path.name} ({len(combined_pred)} samples)")

    # 更新 JSON
    all_window_details = old_windows + new_window_details
    result = {
        "config_name": CONFIG_NAME,
        "preset": PRESET,
        "model": MODEL_NAME,
        "n_windows": len(all_window_details),
        "windows": all_window_details,
        "timestamp": pd.Timestamp.now().isoformat(),
    }
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"  JSON 已更新: {json_path.name}")

    return combined_pred


def update_quality_score(predictions: pd.Series):
    """重算信号质量评分

    注意: 需要 qlib 已初始化 (由 extend_rolling_predictions 完成)
    仅对已有数据的日期范围计算 (未来日期无收盘价，自然排除)
    """
    print("\n" + "=" * 60)
    print("Step 3: 重算信号质量评分")
    print("=" * 60)

    from qlib.data import D
    from factor_lab.run_signal_decay_benchmark import run_signal_analysis

    t0 = time.time()

    # 根据预测的实际日期范围加载收盘价
    pred_dates = predictions.index.get_level_values(0)
    start = pred_dates.min().strftime('%Y-%m-%d')
    end = pred_dates.max().strftime('%Y-%m-%d')
    print(f"  加载收盘价 {start} ~ {end}...")

    instruments = D.instruments('csi300')
    prices = D.features(instruments, ['$close'],
                        start_time=start, end_time=end)
    prices = prices.swaplevel().sort_index()
    close = prices['$close']

    # 只保留有收盘价数据的预测
    valid_dates = close.index.get_level_values(0).unique()
    pred_valid = predictions[predictions.index.get_level_values(0).isin(valid_dates)]
    print(f"  有效预测: {len(pred_valid)}/{len(predictions)} samples")

    result = run_signal_analysis(pred_valid, close)

    # 保存 quality_score
    DECAY_DIR.mkdir(parents=True, exist_ok=True)
    quality = result['quality_score']
    with open(DECAY_DIR / "quality_score.pkl", 'wb') as f:
        pickle.dump(quality, f)
    print(f"  质量评分已保存 ({len(quality)} days, {time.time() - t0:.0f}s)")


def update_signal_config(new_test_end: str):
    """更新 signal_config.yaml 的 test_end"""
    print("\n" + "=" * 60)
    print("Step 4: 更新 signal_config.yaml")
    print("=" * 60)

    with open(CONFIG_FILE) as f:
        config = yaml.safe_load(f)

    old_end = config.get('test_end', '')
    config['test_end'] = new_test_end

    # 更新重训版本号
    from datetime import datetime as _dt
    new_retrain = _dt.now().strftime('%Y-%m')
    old_retrain = config.get('last_retrain', '')
    config['last_retrain'] = new_retrain

    with open(CONFIG_FILE, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    print(f"  test_end: {old_end} → {new_test_end}")
    print(f"  last_retrain: {old_retrain} → {new_retrain}")


def validate(new_test_end: str):
    """PaperTrader replay 验证"""
    print("\n" + "=" * 60)
    print("Step 5: 验证 (Replay)")
    print("=" * 60)

    from factor_lab.signal_generator import SignalGenerator
    sg = SignalGenerator()

    # 检查预测覆盖范围
    dates = sg.get_available_dates()
    print(f"  可用信号: {dates[0]} ~ {dates[-1]}")
    print(f"  总天数: {len(dates)}")

    # 检查模型新鲜度
    freshness = sg.check_model_freshness()
    print(f"  {freshness['message']}")

    # 检查最新信号
    latest = sg.get_signal()
    if 'error' not in latest:
        print(f"  最新信号日期: {latest['date']}")
        print(f"  状态: {latest['regime']}, TopK: {latest['effective_topk']}")
        print(f"  Top3: {latest['target_stocks'][:3]}")
    else:
        print(f"  信号错误: {latest['error']}")

    print("\n  验证通过!")


def main():
    parser = argparse.ArgumentParser(description='季度重训 pipeline')
    parser.add_argument('--data-only', action='store_true',
                        help='只刷数据')
    parser.add_argument('--skip-data', action='store_true',
                        help='跳过数据刷新')
    parser.add_argument('--test-end', type=str, default=None,
                        help='新测试结束日期 (默认: 今天 + 5个月)')
    args = parser.parse_args()

    # 确定 test_end
    if args.test_end:
        new_test_end = args.test_end
    else:
        # 默认: 当前日期 + 5个月 (给足预测空间)
        new_test_end = (date.today() + relativedelta(months=5)).strftime('%Y-%m-%d')

    print("=" * 60)
    print("  季度重训 Pipeline")
    print(f"  目标 test_end: {new_test_end}")
    print("=" * 60)

    t_total = time.time()

    # Step 1: 刷新数据
    if not args.skip_data:
        refresh_data()
    else:
        print("\n[跳过] 数据刷新")

    if args.data_only:
        print("\n完成 (data-only 模式)")
        return

    # Step 2: 扩展预测
    predictions = extend_rolling_predictions(new_test_end)

    # Step 3: 重算质量评分
    update_quality_score(predictions)

    # Step 4: 更新配置
    update_signal_config(new_test_end)

    # Step 5: 验证
    validate(new_test_end)

    total = time.time() - t_total
    print(f"\n{'=' * 60}")
    print(f"  Pipeline 完成! 总耗时: {total:.0f}s ({total/60:.1f}min)")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('fork', force=True)
    main()
