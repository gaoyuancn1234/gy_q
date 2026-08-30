#!/usr/bin/env python3
"""生成 GatedMLP rolling 预测并保存为 pkl — 供 shadow 系统使用

用法:
    python -m factor_lab.generate_gate_mlp_predictions
    python -m factor_lab.generate_gate_mlp_predictions --output-dir shadow/state/shadow_009
"""
import gc
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

CONFIG_NAME = 'D_expand_3v_3r'
PRESET = 'alpha158_val'
TEST_START = '2024-01-01'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=str,
                        default='shadow/state/shadow_009',
                        help='输出目录 (相对于 PROJECT_DIR)')
    parser.add_argument('--test-end', type=str, default='2026-06-30')
    args = parser.parse_args()

    output_dir = PROJECT_DIR / args.output_dir
    pred_dir = output_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    import multiprocessing
    try:
        multiprocessing.set_start_method('fork', force=True)
    except (ValueError, RuntimeError):
        pass  # Windows 无 fork，使用默认 spawn
    import qlib
    qlib.init(provider_uri='~/.qlib/qlib_data/cn_data_bs', region='cn')

    from factor_lab.run_rolling_benchmark import (
        ROLLING_CONFIGS, generate_rolling_windows,
    )
    from factor_lab.run_gated_benchmark import (
        run_gate_mlp_window, build_handler_and_dataset,
    )

    config = ROLLING_CONFIGS[CONFIG_NAME]
    windows = generate_rolling_windows(CONFIG_NAME, config, TEST_START, args.test_end)

    print(f"GatedMLP Rolling Predictions")
    print(f"  Preset: {PRESET}")
    print(f"  Windows: {len(windows)}")
    print(f"  Output: {pred_dir}")

    all_preds = []
    window_details = []
    t_total = time.time()

    for w in windows:
        wnum = w['window_num']
        print(f"\n  --- Window {wnum}/{len(windows)} ---")
        print(f"    Train [{w['train_start']}, {w['train_end']}]  "
              f"Pred [{w['pred_start']}, {w['pred_end']}]")
        t0 = time.time()

        try:
            pred, info = run_gate_mlp_window(w)
            elapsed = time.time() - t0

            all_preds.append(pred)
            window_details.append({
                "window_num": wnum,
                "train_start": w['train_start'],
                "train_end": w['train_end'],
                "pred_start": w['pred_start'],
                "pred_end": w['pred_end'],
                "n_samples": int(len(pred)),
                "best_epoch": info.get('best_epoch'),
                "train_time": round(elapsed, 1),
            })
            print(f"    Samples: {len(pred)}, Time: {elapsed:.1f}s")
        except Exception as e:
            print(f"    Window {wnum} 失败: {e}")
            continue

        gc.collect()

    if not all_preds:
        print("无有效预测，退出")
        return

    # 合并
    combined = pd.concat(all_preds)
    if combined.index.duplicated().any():
        combined = combined[~combined.index.duplicated(keep='last')]

    # 保存 pkl
    pkl_name = f"{CONFIG_NAME}_{PRESET}_GatedMLP.pkl"
    pkl_path = pred_dir / pkl_name
    combined.to_pickle(pkl_path)
    print(f"\n  预测已保存: {pkl_name} ({len(combined)} samples)")

    # 保存 JSON
    json_name = f"{CONFIG_NAME}_{PRESET}_GatedMLP.json"
    json_path = output_dir / json_name
    result = {
        "config_name": CONFIG_NAME,
        "preset": PRESET,
        "model": "GatedMLP",
        "test_start": TEST_START,
        "test_end": args.test_end,
        "total_samples": int(len(combined)),
        "total_time": round(time.time() - t_total, 1),
        "windows": window_details,
    }
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # 生成 quality_score (复用 retrain_pipeline 逻辑)
    try:
        from retrain_pipeline import _compute_quality_score
        quality = _compute_quality_score(combined)
        with open(output_dir / "quality_score.pkl", 'wb') as f:
            pickle.dump(quality, f)
        print(f"  质量评分已保存")
    except Exception as e:
        # 如果没有 _compute_quality_score，用现有的 quality_score
        src = PROJECT_DIR / "factor_lab" / "results" / "signal_decay" / "quality_score.pkl"
        if src.exists():
            import shutil
            shutil.copy2(src, output_dir / "quality_score.pkl")
            print(f"  质量评分已复制 (使用主基线)")

    total = time.time() - t_total
    print(f"\n  完成! 总耗时: {total:.0f}s ({total/60:.1f}min)")


if __name__ == '__main__':
    main()
