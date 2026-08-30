#!/usr/bin/env python3
"""对比 alpha158 vs alpha158_ext vs alpha158_selected 在 LightGBM/XGBoost 上的表现"""
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


def run_one(preset, model_name):
    """跑一个 preset × model 组合"""
    bm.HANDLER_PRESET = preset

    model_map = {m[0]: m for m in bm.ALL_MODELS}
    if model_name not in model_map:
        print(f"Unknown model: {model_name}")
        return None

    _, category, run_fn = model_map[model_name]
    print(f"\n{'='*60}")
    print(f"[{preset}] × [{model_name}] ({category})")
    print(f"因子数: {bm.get_d_feat()}")
    print(f"{'='*60}")

    t0 = time.time()
    try:
        pred = run_fn()
        elapsed_train = time.time() - t0
        print(f"  训练+预测完成 ({elapsed_train:.1f}s), 预测样本: {len(pred)}")

        t1 = time.time()
        bt_result = bm.run_backtest(pred)
        elapsed_bt = time.time() - t1

        bt_result['train_time'] = elapsed_train
        bt_result['category'] = category
        bt_result['preset'] = preset
        bt_result['d_feat'] = bm.get_d_feat()

        print(f"  回测完成 ({elapsed_bt:.1f}s)")
        print(f"  总收益: {bt_result.get('total_return', 0):.2%}")
        print(f"  年化: {bt_result.get('annual_return', 0):.2%}")
        print(f"  Sharpe: {bt_result.get('sharpe', 0):.3f}")
        print(f"  最大回撤: {bt_result.get('max_drawdown', 0):.2%}")

        # 保存
        out_path = RESULTS_DIR / f"{preset}_{model_name.lower()}.json"
        with open(out_path, 'w') as f:
            json.dump(bt_result, f, indent=2)
        print(f"  保存至: {out_path}")

        return bt_result

    except Exception as e:
        elapsed = time.time() - t0
        print(f"  FAILED ({elapsed:.1f}s): {e}")
        import traceback
        traceback.print_exc()
        return {'error': str(e)}


def main():
    experiments = [
        ("alpha158_selected", "LightGBM"),
        ("alpha158_selected", "XGBoost"),
    ]

    results = {}
    for preset, model in experiments:
        key = f"{preset}_{model}"
        out_path = RESULTS_DIR / f"{preset}_{model.lower()}.json"
        if out_path.exists():
            print(f"\n[跳过] {key} — 已有结果")
            with open(out_path) as f:
                results[key] = json.load(f)
            continue
        results[key] = run_one(preset, model)

    # 加载之前的结果
    for preset in ["alpha158", "alpha158_ext"]:
        for model in ["lightgbm", "xgboost"]:
            path = RESULTS_DIR / f"{preset}_{model}.json"
            if path.exists():
                with open(path) as f:
                    results[f"{preset}_{model.title()}"] = json.load(f)

    # 打印对比表
    print(f"\n\n{'='*90}")
    print(f"{'预设':20s} {'模型':12s} {'总收益':>10s} {'年化':>10s} {'Sharpe':>8s} {'最大回撤':>10s} {'因子数':>6s}")
    print(f"{'-'*90}")

    for key in sorted(results.keys()):
        r = results[key]
        if 'error' in r:
            continue
        parts = key.rsplit('_', 1)
        if len(parts) == 2:
            preset, model = parts
        else:
            preset, model = key, ""
        print(f"{preset:20s} {model:12s} "
              f"{r.get('total_return', 0):>9.2%} "
              f"{r.get('annual_return', 0):>9.2%} "
              f"{r.get('sharpe', 0):>7.3f} "
              f"{r.get('max_drawdown', 0):>9.2%} "
              f"{r.get('d_feat', '?'):>6}")


if __name__ == "__main__":
    main()
