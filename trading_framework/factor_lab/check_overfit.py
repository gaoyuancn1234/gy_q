#!/usr/bin/env python3
"""Per-window 诊断: gate_mlp vs baseline 逐窗口 IC + TopK收益对比

不做 backtest_daily（避免 hang），只算 Rank IC 和 TopK 均值收益。
"""
import sys, os, gc, warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
warnings.filterwarnings('ignore')

# 禁用 qlib recorder 避免锁
os.environ['QLIB_LOG_LEVEL'] = 'WARNING'

import multiprocessing
try:
    multiprocessing.set_start_method('fork', force=True)
except (ValueError, RuntimeError):
    pass  # Windows 无 fork，使用默认 spawn
import qlib
from qlib.constant import REG_CN
qlib.init(provider_uri='~/.qlib/qlib_data/cn_data_bs', region=REG_CN)

from factor_lab.run_gated_benchmark import (
    build_handler_and_dataset, extract_features_labels,
    train_lgb, predict_lgb,
    CONFIG_NAME, TEST_START, TEST_END, LGB_PARAMS,
)
from factor_lab.run_rolling_benchmark import ROLLING_CONFIGS, generate_rolling_windows

config = ROLLING_CONFIGS[CONFIG_NAME]
windows = generate_rolling_windows(CONFIG_NAME, config, TEST_START, TEST_END)

print(f"{'Win':>4} {'Period':<24} {'BL_IC':>7} {'GM_IC':>7} {'BL_ret':>8} {'GM_ret':>8} {'Winner':>7}")
print("-" * 75)
sys.stdout.flush()

bl_ics, gm_ics = [], []
bl_rets, gm_rets = [], []

for w in windows:
    wn = w['window_num']
    print(f"  [Win {wn}] building data...", end='', flush=True)

    handler, dataset = build_handler_and_dataset(w)
    test_feat, test_label = extract_features_labels(handler, w['pred_start'], w['pred_end'])

    # ── Baseline LGB ──
    print(" LGB...", end='', flush=True)
    model_lgb, best_iter = train_lgb(dataset)
    pred_bl = predict_lgb(model_lgb, dataset, w['pred_start'], w['pred_end'])

    # ── Gate MLP ──
    print(" Gate+MLP...", end='', flush=True)
    import torch
    from factor_lab.gated_factor_net import (
        MarketStateComputer, build_group_mask, train_gated_model,
    )
    from factor_lab.run_gated_benchmark import predict_nn

    train_feat, train_label = extract_features_labels(handler, w['train_start'], w['train_end'])
    valid_feat, valid_label = extract_features_labels(handler, w['valid_start'], w['valid_end'])
    feature_names = list(train_feat.columns)

    msc = MarketStateComputer(feature_names)
    all_feat = pd.concat([train_feat, valid_feat, test_feat])
    all_label = pd.concat([train_label, valid_label, test_label])
    market_states = msc.compute_all_dates_fast(all_feat, all_label)

    def get_state_array(feat_df):
        dates = feat_df.index.get_level_values(0)
        return np.array([market_states.loc[d].values for d in dates], dtype=np.float32)

    train_state = get_state_array(train_feat)
    valid_state = get_state_array(valid_feat)
    test_state = get_state_array(test_feat)

    train_factors = train_feat.values.astype(np.float32)
    valid_factors = valid_feat.values.astype(np.float32)
    test_factors = test_feat.values.astype(np.float32)
    np.nan_to_num(train_factors, copy=False)
    np.nan_to_num(valid_factors, copy=False)
    np.nan_to_num(test_factors, copy=False)
    np.nan_to_num(train_state, copy=False)
    np.nan_to_num(valid_state, copy=False)
    np.nan_to_num(test_state, copy=False)
    train_labels = train_label.values.astype(np.float32)
    valid_labels = valid_label.values.astype(np.float32)
    np.nan_to_num(train_labels, copy=False)
    np.nan_to_num(valid_labels, copy=False)

    group_mask = build_group_mask(feature_names)
    gate_model, gate_info = train_gated_model(
        train_state, train_factors, train_labels,
        valid_state, valid_factors, valid_labels,
        group_mask, verbose=False,
    )

    device = next(gate_model.parameters()).device
    pred_gm = predict_nn(gate_model, test_state, test_factors, test_feat.index, device)

    # ── 对齐 + 计算 ──
    common = pred_bl.index.intersection(pred_gm.index).intersection(test_label.index)
    lb = test_label.loc[common]
    if isinstance(lb, pd.DataFrame):
        lb = lb.iloc[:, 0]

    ic_bl = spearmanr(pred_bl.loc[common].values, lb.values)[0]
    ic_gm = spearmanr(pred_gm.loc[common].values, lb.values)[0]
    bl_ics.append(ic_bl)
    gm_ics.append(ic_gm)

    # TopK 均值日收益
    dates = common.get_level_values(0).unique()
    bl_daily, gm_daily = [], []
    for dt in dates:
        dt_bl = pred_bl.loc[common].xs(dt, level=0).nlargest(12)
        dt_gm = pred_gm.loc[common].xs(dt, level=0).nlargest(12)
        dt_lb = lb.xs(dt, level=0)
        bl_daily.append(dt_lb.reindex(dt_bl.index).mean())
        gm_daily.append(dt_lb.reindex(dt_gm.index).mean())

    bl_ret = np.nanmean(bl_daily)
    gm_ret = np.nanmean(gm_daily)
    bl_rets.append(bl_ret)
    gm_rets.append(gm_ret)

    winner = "GM" if ic_gm > ic_bl else "BL"
    period = f"{w['pred_start']}~{w['pred_end']}"
    print(f"\r{wn:>4} {period:<24} {ic_bl:>7.4f} {ic_gm:>7.4f} {bl_ret:>7.4f} {gm_ret:>7.4f} {winner:>7}")
    sys.stdout.flush()

    del handler, dataset, gate_model, train_feat, valid_feat
    gc.collect()

print("-" * 75)
bl_ic_arr = np.array(bl_ics)
gm_ic_arr = np.array(gm_ics)
print(f"{'Mean':>4} {'':24} {bl_ic_arr.mean():>7.4f} {gm_ic_arr.mean():>7.4f} {np.mean(bl_rets):>7.4f} {np.mean(gm_rets):>7.4f}")
print(f"{'Std':>4} {'':24} {bl_ic_arr.std():>7.4f} {gm_ic_arr.std():>7.4f}")
print(f"\ngate_mlp IC > baseline: {sum(1 for a,b in zip(gm_ics, bl_ics) if a>b)}/{len(bl_ics)} windows")
print(f"gate_mlp Ret > baseline: {sum(1 for a,b in zip(gm_rets, bl_rets) if a>b)}/{len(bl_rets)} windows")
print(f"IC positive: baseline {sum(1 for x in bl_ics if x>0)}/9  gate_mlp {sum(1 for x in gm_ics if x>0)}/9")
