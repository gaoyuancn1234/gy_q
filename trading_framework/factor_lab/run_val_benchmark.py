#!/usr/bin/env python3
"""Phase 2: alpha158_val (量价去冗余 + 基本面) benchmark"""
import sys
import time
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import multiprocessing
try:
    multiprocessing.set_start_method('fork', force=True)
except (ValueError, RuntimeError):
    pass  # Windows 无 fork，使用默认 spawn

import qlib
from qlib.constant import REG_CN
qlib.init(provider_uri='~/.qlib/qlib_data/cn_data_bs', region=REG_CN)

import benchmark_models as bm

RESULTS_DIR = Path(__file__).parent / "results" / "experiments"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

preset = "alpha158_val"
models = ["LightGBM", "XGBoost", "CatBoost"]

bm.HANDLER_PRESET = preset
print(f"预设: {preset}, d_feat={bm.get_d_feat()}")

results = {}
for model_name in models:
    out_path = RESULTS_DIR / f"{preset}_{model_name.lower()}.json"
    if out_path.exists():
        print(f"\n[跳过] {preset}_{model_name} — 已有结果")
        with open(out_path) as f:
            results[model_name] = json.load(f)
        continue

    model_map = {m[0]: m for m in bm.ALL_MODELS}
    _, category, run_fn = model_map[model_name]

    print(f"\n{'='*60}")
    print(f"[{preset}] × [{model_name}] ({category})")
    print(f"{'='*60}")

    t0 = time.time()
    try:
        pred = run_fn()
        elapsed_train = time.time() - t0
        print(f"  训练+预测完成 ({elapsed_train:.1f}s)")

        bt_result = bm.run_backtest(pred)
        bt_result['train_time'] = elapsed_train
        bt_result['category'] = category
        bt_result['preset'] = preset
        bt_result['d_feat'] = bm.get_d_feat()

        print(f"  总收益: {bt_result.get('total_return', 0):.2%}")
        print(f"  Sharpe: {bt_result.get('sharpe', 0):.3f}")
        print(f"  最大回撤: {bt_result.get('max_drawdown', 0):.2%}")

        with open(out_path, 'w') as f:
            json.dump(bt_result, f, indent=2)
        results[model_name] = bt_result
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  FAILED ({elapsed:.1f}s): {e}")
        import traceback
        traceback.print_exc()

# 汇总
print(f"\n{'='*70}")
print(f"{'预设':22s} {'模型':12s} {'Sharpe':>8s} {'最大回撤':>10s} {'总收益':>10s}")
print(f"{'-'*70}")

# 加载 alpha158_selected 对照
for m in models:
    for p in ["alpha158_selected", "alpha158_val"]:
        path = RESULTS_DIR / f"{p}_{m.lower()}.json"
        if path.exists():
            with open(path) as f:
                r = json.load(f)
            print(f"{p:22s} {m:12s} {r.get('sharpe', 0):>7.3f} {r.get('max_drawdown', 0):>9.2%} {r.get('total_return', 0):>9.2%}")
