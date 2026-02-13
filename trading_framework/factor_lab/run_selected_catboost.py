#!/usr/bin/env python3
"""跑 alpha158_selected × CatBoost"""
import sys
import time
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import multiprocessing
multiprocessing.set_start_method('fork', force=True)

import qlib
from qlib.constant import REG_CN
qlib.init(provider_uri='~/.qlib/qlib_data/cn_data_bs', region=REG_CN)

import benchmark_models as bm

RESULTS_DIR = Path(__file__).parent / "results" / "experiments"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

bm.HANDLER_PRESET = 'alpha158_selected'
print(f"alpha158_selected × CatBoost, d_feat={bm.get_d_feat()}")

t0 = time.time()
pred = bm.run_catboost()
elapsed_train = time.time() - t0
print(f"训练+预测完成 ({elapsed_train:.1f}s), 预测样本: {len(pred)}")

bt_result = bm.run_backtest(pred)
bt_result['train_time'] = elapsed_train
bt_result['category'] = 'Tree'
bt_result['preset'] = 'alpha158_selected'
bt_result['d_feat'] = bm.get_d_feat()

print(f"总收益: {bt_result.get('total_return', 0):.2%}")
print(f"年化: {bt_result.get('annual_return', 0):.2%}")
print(f"Sharpe: {bt_result.get('sharpe', 0):.3f}")
print(f"最大回撤: {bt_result.get('max_drawdown', 0):.2%}")

with open(RESULTS_DIR / 'alpha158_selected_catboost.json', 'w') as f:
    json.dump(bt_result, f, indent=2)
print("结果已保存")
