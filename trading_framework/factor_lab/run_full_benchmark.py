#!/usr/bin/env python3
"""Step 5: full preset benchmark — 全因子体系 vs alpha158_val

对比:
  full (精选量价+基本面+资金流) × LightGBM/XGBoost/CatBoost
  vs alpha158_val (精选量价+基本面)

用法:
    cd trading_framework
    python -m factor_lab.run_full_benchmark
    python -m factor_lab.run_full_benchmark --models lightgbm catboost
    python -m factor_lab.run_full_benchmark --preset full_selected   # 去冗余后
"""
import sys
import time
import json
import argparse
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


def run_preset_benchmark(preset: str, models: list[str], force: bool = False):
    """对指定 preset 跑多个模型的 benchmark"""
    bm.HANDLER_PRESET = preset
    d_feat = bm.get_d_feat()
    print(f"\n预设: {preset}, d_feat={d_feat}")

    results = {}
    model_map = {m[0]: m for m in bm.ALL_MODELS}

    for model_name in models:
        out_path = RESULTS_DIR / f"{preset}_{model_name.lower()}.json"
        if out_path.exists() and not force:
            print(f"\n[跳过] {preset}_{model_name} — 已有结果")
            with open(out_path) as f:
                results[model_name] = json.load(f)
            continue

        if model_name not in model_map:
            print(f"[跳过] 未知模型: {model_name}")
            continue

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
            bt_result['d_feat'] = d_feat

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

    return results


def print_comparison(presets: list[str], models: list[str]):
    """打印对比表"""
    print(f"\n{'='*80}")
    print(f"{'预设':22s} {'模型':12s} {'因子数':>6s} {'Sharpe':>8s} {'最大回撤':>10s} {'总收益':>10s} {'超额':>10s}")
    print(f"{'-'*80}")

    for preset in presets:
        for model in models:
            path = RESULTS_DIR / f"{preset}_{model.lower()}.json"
            if path.exists():
                with open(path) as f:
                    r = json.load(f)
                d_feat = r.get('d_feat', '?')
                sharpe = r.get('sharpe', 0)
                mdd = r.get('max_drawdown', 0)
                total = r.get('total_return', 0)
                excess = r.get('excess_return', 0)
                print(f"{preset:22s} {model:12s} {d_feat:>6} {sharpe:>7.3f} {mdd:>9.2%} {total:>9.2%} {excess:>9.2%}")


def main():
    parser = argparse.ArgumentParser(description="Full preset benchmark")
    parser.add_argument("--preset", type=str, default="full",
                        help="因子预设 (full, full_selected)")
    parser.add_argument("--models", type=str, nargs="+",
                        default=["LightGBM", "XGBoost", "CatBoost"],
                        help="模型列表")
    parser.add_argument("--force", action="store_true",
                        help="强制重跑(忽略已有结果)")
    args = parser.parse_args()

    print("=" * 60)
    print(f"Phase 3 Step 5: {args.preset} Benchmark")
    print("=" * 60)

    # 跑 benchmark
    run_preset_benchmark(args.preset, args.models, args.force)

    # 对比输出
    compare_presets = ["alpha158_selected", "alpha158_val", "full"]
    if args.preset not in compare_presets:
        compare_presets.append(args.preset)
    print_comparison(compare_presets, args.models)


if __name__ == "__main__":
    main()
