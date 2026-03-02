#!/usr/bin/env python3
"""
Exp 015b/c: 可学习 Gate + TabNet 因子择时 — Walk-Forward 回测

8 策略对比:
  baseline     — 标准 LGB, 无择时
  timing_v2    — 手工 regime 规则 + LGB (TopK 自适应)
  gate_mlp     — 可学习 gate + MLP prediction head (端到端)
  gate_lgb     — 可学习 gate 提取权重 → apply 到 LGB (混合)
  mlp_no_gate  — MLP prediction head, 无 gate (消融)
  tabnet       — 纯 TabNet + grouped_features (Exp 015c)
  tabnet_ms    — TabNet + market_state 拼接 (Exp 015c)
  gate_tabnet  — Gate 预加权 + TabNet 精细选择 (Exp 015c)

用法:
  # 单窗口快速测试
  python -m factor_lab.run_gated_benchmark --strategies tabnet --windows 1

  # 全量回测
  python -m factor_lab.run_gated_benchmark

  # 仅报告
  python -m factor_lab.run_gated_benchmark --report-only
"""
import sys
import gc
import json
import time
import warnings
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

# 测试区间
TEST_START = '2024-01-01'
TEST_END = '2026-02-05'

# Rolling 配置 (D_expand)
CONFIG_NAME = 'D_expand_3v_3r'

# 回测参数
TOPK = 12
N_DROP = 3

# LGB 参数 (与现有 benchmark 一致)
LGB_PARAMS = dict(
    loss="mse", learning_rate=0.01, num_leaves=64,
    num_boost_round=500, early_stopping_rounds=80,
    feature_fraction=0.75, bagging_fraction=0.75, bagging_freq=5,
    lambda_l1=0.1, lambda_l2=0.1, min_data_in_leaf=80,
)

# TopK by regime (与 signal_generator 一致)
TOPK_BY_REGIME = {'strong': 12, 'normal': 16, 'weak': 20}

RESULTS_DIR = PROJECT_DIR / "factor_lab" / "results" / "gated"
PRESET = 'alpha158_val'
ALL_STRATEGIES = ['baseline', 'timing_v2', 'gate_mlp', 'gate_lgb', 'mlp_no_gate',
                  'tabnet', 'tabnet_ms', 'gate_tabnet']


# ── 数据提取 ──

def extract_features_labels(handler, start: str, end: str) -> tuple[pd.DataFrame, pd.Series]:
    """从 handler 提取指定时间段的 features 和 labels。"""
    df = handler.fetch(col_set="feature", data_key="infer")
    lb = handler.fetch(col_set="label", data_key="learn")

    # 过滤时间
    ts, te = pd.Timestamp(start), pd.Timestamp(end)
    dates = df.index.get_level_values(0)
    mask = (dates >= ts) & (dates <= te)
    df = df[mask]

    dates_lb = lb.index.get_level_values(0)
    mask_lb = (dates_lb >= ts) & (dates_lb <= te)
    lb = lb[mask_lb]

    # 对齐
    common = df.index.intersection(lb.index)
    df = df.loc[common]
    lb = lb.loc[common]

    if isinstance(lb, pd.DataFrame):
        lb = lb.iloc[:, 0]

    return df, lb


def slice_by_date(df: pd.DataFrame | pd.Series, start: str, end: str):
    """按日期过滤 MultiIndex(datetime, instrument) 数据。"""
    ts, te = pd.Timestamp(start), pd.Timestamp(end)
    dates = df.index.get_level_values(0)
    return df[(dates >= ts) & (dates <= te)]


# ── 训练/预测工具 ──

def train_lgb(dataset):
    """训练 LGB 模型，返回 (model, best_iter)。"""
    from qlib.contrib.model.gbdt import LGBModel
    model = LGBModel(**LGB_PARAMS)
    model.fit(dataset)
    best_iter = getattr(model.model, 'best_iteration', None)
    return model, best_iter


def predict_lgb(model, dataset, pred_start: str, pred_end: str) -> pd.Series:
    """LGB 预测并过滤到 pred 区间。"""
    pred = model.predict(dataset)
    if isinstance(pred.index, pd.MultiIndex):
        dates = pred.index.get_level_values(0)
        mask = (dates >= pd.Timestamp(pred_start)) & (dates <= pd.Timestamp(pred_end))
        pred = pred[mask]
    return pred


def predict_nn(model, market_state_arr: np.ndarray, factor_arr: np.ndarray,
               index: pd.MultiIndex, device) -> pd.Series:
    """NN 模型批量预测。"""
    import torch
    model.eval()
    with torch.no_grad():
        state_t = torch.tensor(market_state_arr, dtype=torch.float32).to(device)
        factor_t = torch.tensor(factor_arr, dtype=torch.float32).to(device)
        pred = model(state_t, factor_t).cpu().numpy()
    return pd.Series(pred, index=index, name='score')


def predict_mlp(model, factor_arr: np.ndarray, index: pd.MultiIndex, device) -> pd.Series:
    """MLP (无 gate) 批量预测。"""
    import torch
    model.eval()
    with torch.no_grad():
        factor_t = torch.tensor(factor_arr, dtype=torch.float32).to(device)
        pred = model(factor_t).cpu().numpy()
    return pd.Series(pred, index=index, name='score')


# ── 回测复用 ──

def run_backtest(pred, topk=TOPK, n_drop=N_DROP):
    """标准 TopkDropout 回测 (复用 run_rolling_benchmark 逻辑)。"""
    from qlib.contrib.evaluate import backtest_daily
    from qlib.utils import init_instance_by_config

    strategy_config = {
        "class": "TopkDropoutStrategy",
        "module_path": "qlib.contrib.strategy",
        "kwargs": {"signal": pred, "topk": topk, "n_drop": n_drop},
    }
    backtest_config = {
        "start_time": TEST_START, "end_time": TEST_END,
        "account": 100_000_000, "benchmark": "SH000300",
        "exchange_kwargs": {
            "freq": "day", "limit_threshold": 0.095, "deal_price": "open",
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


def detect_regime_for_date(index_close: pd.Series, date: pd.Timestamp) -> str:
    """简易 regime 检测: MA10 vs MA60。"""
    hist = index_close[index_close.index <= date].tail(70)
    if len(hist) < 60:
        return 'normal'
    ma10 = hist.tail(10).mean()
    ma60 = hist.tail(60).mean()
    ratio = ma10 / ma60 - 1
    if ratio > 0.02:
        return 'strong'
    elif ratio < -0.02:
        return 'weak'
    return 'normal'


# ── Window 运行逻辑 ──

def build_handler_and_dataset(w: dict):
    """构建 handler + dataset (共用工具)。"""
    from factor_lab.factors.presets import build_handler
    from qlib.data.dataset import DatasetH

    handler = build_handler(
        PRESET,
        start_time=w['train_start'], end_time=w['pred_end'],
        fit_start_time=w['train_start'], fit_end_time=w['train_end'],
    )
    dataset = DatasetH(handler=handler, segments={
        "train": (w['train_start'], w['train_end']),
        "valid": (w['valid_start'], w['valid_end']),
        "test": (w['pred_start'], w['pred_end']),
    })
    return handler, dataset


def run_baseline_window(w: dict) -> pd.Series:
    """标准 LGB baseline (一个窗口)。"""
    handler, dataset = build_handler_and_dataset(w)
    model, best_iter = train_lgb(dataset)
    pred = predict_lgb(model, dataset, w['pred_start'], w['pred_end'])
    print(f"    baseline: best_iter={best_iter}, samples={len(pred)}")
    del handler, dataset, model
    gc.collect()
    return pred


def run_gate_mlp_window(w: dict, market_state_computer=None) -> tuple[pd.Series, dict]:
    """端到端 GatedFactorNet 训练+预测 (一个窗口)。

    Returns:
        (pred_series, gate_info_dict)
    """
    import torch
    from factor_lab.gated_factor_net import (
        GatedFactorNet, MarketStateComputer, build_group_mask,
        train_gated_model,
    )

    handler, dataset = build_handler_and_dataset(w)

    # 提取原始数据
    train_feat, train_label = extract_features_labels(handler, w['train_start'], w['train_end'])
    valid_feat, valid_label = extract_features_labels(handler, w['valid_start'], w['valid_end'])
    test_feat, test_label = extract_features_labels(handler, w['pred_start'], w['pred_end'])

    feature_names = list(train_feat.columns)

    # 计算 market state
    if market_state_computer is None:
        market_state_computer = MarketStateComputer(feature_names)

    all_feat = pd.concat([train_feat, valid_feat, test_feat])
    all_label = pd.concat([train_label, valid_label, test_label])
    market_states = market_state_computer.compute_all_dates_fast(all_feat, all_label)

    def get_state_array(feat_df, states_df):
        dates = feat_df.index.get_level_values(0)
        # broadcast: 每个 (date, instrument) pair 用该 date 的 market state
        return np.array([states_df.loc[d].values for d in dates], dtype=np.float32)

    train_state = get_state_array(train_feat, market_states)
    valid_state = get_state_array(valid_feat, market_states)
    test_state = get_state_array(test_feat, market_states)

    # NaN 处理
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

    # 构建 group mask
    group_mask = build_group_mask(feature_names)

    # 训练
    model, info = train_gated_model(
        train_state, train_factors, train_labels,
        valid_state, valid_factors, valid_labels,
        group_mask, verbose=True,
    )

    # 预测
    device = next(model.parameters()).device
    pred = predict_nn(model, test_state, test_factors, test_feat.index, device)
    print(f"    gate_mlp: best_epoch={info['best_epoch']}, samples={len(pred)}")

    # 提取 gate 权重 (per date) 用于分析
    gate_analysis = {}
    test_dates = test_feat.index.get_level_values(0).unique()
    for dt in test_dates:
        dt_state = market_states.loc[dt].values.astype(np.float32)
        state_t = torch.tensor(dt_state, dtype=torch.float32).unsqueeze(0).to(device)
        gw = model.get_gate_weights(state_t).cpu().numpy()[0]
        gate_analysis[dt.strftime('%Y-%m-%d')] = gw.tolist()

    info['gate_weights'] = gate_analysis

    del handler, dataset, model, train_feat, valid_feat, test_feat
    gc.collect()
    return pred, info


def run_gate_lgb_window(w: dict, market_state_computer=None) -> tuple[pd.Series, dict]:
    """混合策略: 训练 gate → 提取权重 → apply 到因子 → 喂给 LGB。

    Returns:
        (pred_series, gate_info_dict)
    """
    import torch
    from factor_lab.gated_factor_net import (
        MarketStateComputer, build_group_mask, train_gated_model,
    )
    from qlib.data.dataset import DatasetH

    handler, dataset = build_handler_and_dataset(w)

    # 提取原始数据
    train_feat, train_label = extract_features_labels(handler, w['train_start'], w['train_end'])
    valid_feat, valid_label = extract_features_labels(handler, w['valid_start'], w['valid_end'])
    test_feat, test_label = extract_features_labels(handler, w['pred_start'], w['pred_end'])

    feature_names = list(train_feat.columns)

    # 计算 market state
    if market_state_computer is None:
        market_state_computer = MarketStateComputer(feature_names)

    all_feat = pd.concat([train_feat, valid_feat, test_feat])
    all_label = pd.concat([train_label, valid_label, test_label])
    market_states = market_state_computer.compute_all_dates_fast(all_feat, all_label)

    def get_state_array(feat_df, states_df):
        dates = feat_df.index.get_level_values(0)
        return np.array([states_df.loc[d].values for d in dates], dtype=np.float32)

    train_state = get_state_array(train_feat, market_states)
    valid_state = get_state_array(valid_feat, market_states)
    test_state = get_state_array(test_feat, market_states)

    train_factors = train_feat.values.astype(np.float32)
    valid_factors = valid_feat.values.astype(np.float32)

    np.nan_to_num(train_factors, copy=False)
    np.nan_to_num(valid_factors, copy=False)
    np.nan_to_num(train_state, copy=False)
    np.nan_to_num(valid_state, copy=False)
    np.nan_to_num(test_state, copy=False)

    train_labels = train_label.values.astype(np.float32)
    valid_labels = valid_label.values.astype(np.float32)
    np.nan_to_num(train_labels, copy=False)
    np.nan_to_num(valid_labels, copy=False)

    group_mask = build_group_mask(feature_names)

    # Step 1: 训练 GatedFactorNet 学习 gate
    gate_model, gate_info = train_gated_model(
        train_state, train_factors, train_labels,
        valid_state, valid_factors, valid_labels,
        group_mask, verbose=True,
    )

    # Step 2: 用 gate 提取 per-date 权重并 apply 到因子
    device = next(gate_model.parameters()).device
    gate_model.eval()

    def apply_gate_to_features(feat_df, state_arr):
        """将 gate 权重 apply 到因子 DataFrame。"""
        dates = feat_df.index.get_level_values(0)
        unique_dates = dates.unique()

        # 预计算每日 gate weights → per_factor_weights
        date_weights = {}
        with torch.no_grad():
            for dt in unique_dates:
                dt_state = market_states.loc[dt].values.astype(np.float32)
                np.nan_to_num(dt_state, copy=False)
                state_t = torch.tensor(dt_state, dtype=torch.float32).unsqueeze(0).to(device)
                gw = gate_model.gate(state_t)  # (1, n_groups)
                pfw = (gw @ gate_model.group_mask.T).cpu().numpy()[0]  # (n_features,)
                date_weights[dt] = pfw

        # Apply weights
        weighted = feat_df.copy()
        for dt in unique_dates:
            mask = dates == dt
            weighted.loc[mask] = weighted.loc[mask].values * date_weights[dt]
        return weighted

    # Apply gate 到 train/valid/test
    train_feat_gated = apply_gate_to_features(train_feat, train_state)
    valid_feat_gated = apply_gate_to_features(valid_feat, valid_state)
    test_feat_gated = apply_gate_to_features(test_feat, test_state)

    # Step 3: 直接用 lightgbm 训练 (绕过 Qlib DatasetH)
    import lightgbm as lgb

    train_y = train_label.values.astype(np.float64)
    valid_y = valid_label.values.astype(np.float64)
    train_X = train_feat_gated.values.astype(np.float64)
    valid_X = valid_feat_gated.values.astype(np.float64)
    test_X = test_feat_gated.values.astype(np.float64)

    np.nan_to_num(train_X, copy=False)
    np.nan_to_num(valid_X, copy=False)
    np.nan_to_num(test_X, copy=False)
    np.nan_to_num(train_y, copy=False)
    np.nan_to_num(valid_y, copy=False)

    dtrain = lgb.Dataset(train_X, label=train_y.ravel())
    dvalid = lgb.Dataset(valid_X, label=valid_y.ravel(), reference=dtrain)

    lgb_params_native = {
        'objective': 'regression',
        'metric': 'mse',
        'learning_rate': LGB_PARAMS.get('learning_rate', 0.01),
        'num_leaves': LGB_PARAMS.get('num_leaves', 64),
        'feature_fraction': LGB_PARAMS.get('feature_fraction', 0.75),
        'bagging_fraction': LGB_PARAMS.get('bagging_fraction', 0.75),
        'bagging_freq': LGB_PARAMS.get('bagging_freq', 5),
        'lambda_l1': LGB_PARAMS.get('lambda_l1', 0.1),
        'lambda_l2': LGB_PARAMS.get('lambda_l2', 0.1),
        'min_data_in_leaf': LGB_PARAMS.get('min_data_in_leaf', 80),
        'verbose': -1,
    }

    es_rounds = LGB_PARAMS.get('early_stopping_rounds', 80)
    callbacks = [lgb.early_stopping(es_rounds, verbose=False), lgb.log_evaluation(0)]
    lgb_model = lgb.train(
        lgb_params_native, dtrain, num_boost_round=LGB_PARAMS.get('num_boost_round', 500),
        valid_sets=[dvalid], callbacks=callbacks,
    )
    best_iter = lgb_model.best_iteration

    # Predict on test
    test_pred = lgb_model.predict(test_X)
    pred = pd.Series(test_pred, index=test_feat.index, name='score')
    # 过滤到 pred 区间
    dates = pred.index.get_level_values(0)
    pred = pred[(dates >= pd.Timestamp(w['pred_start'])) & (dates <= pd.Timestamp(w['pred_end']))]
    print(f"    gate_lgb: gate_best={gate_info['best_epoch']}, lgb_best={best_iter}, samples={len(pred)}")

    # Gate 权重分析
    gate_analysis = {}
    test_dates = test_feat.index.get_level_values(0).unique()
    for dt in test_dates:
        dt_state = market_states.loc[dt].values.astype(np.float32)
        np.nan_to_num(dt_state, copy=False)
        state_t = torch.tensor(dt_state, dtype=torch.float32).unsqueeze(0).to(device)
        gw = gate_model.get_gate_weights(state_t).cpu().numpy()[0]
        gate_analysis[dt.strftime('%Y-%m-%d')] = gw.tolist()

    gate_info['gate_weights'] = gate_analysis
    gate_info['lgb_best_iter'] = best_iter

    del handler, dataset, gate_model, train_feat, valid_feat, test_feat
    gc.collect()
    return pred, gate_info


def run_mlp_no_gate_window(w: dict) -> tuple[pd.Series, dict]:
    """MLP 无 gate 消融 (一个窗口)。"""
    from factor_lab.gated_factor_net import train_mlp_no_gate

    handler, dataset = build_handler_and_dataset(w)

    train_feat, train_label = extract_features_labels(handler, w['train_start'], w['train_end'])
    valid_feat, valid_label = extract_features_labels(handler, w['valid_start'], w['valid_end'])
    test_feat, test_label = extract_features_labels(handler, w['pred_start'], w['pred_end'])

    train_factors = train_feat.values.astype(np.float32)
    valid_factors = valid_feat.values.astype(np.float32)
    test_factors = test_feat.values.astype(np.float32)
    np.nan_to_num(train_factors, copy=False)
    np.nan_to_num(valid_factors, copy=False)
    np.nan_to_num(test_factors, copy=False)

    train_labels = train_label.values.astype(np.float32)
    valid_labels = valid_label.values.astype(np.float32)
    np.nan_to_num(train_labels, copy=False)
    np.nan_to_num(valid_labels, copy=False)

    model, info = train_mlp_no_gate(
        train_factors, train_labels,
        valid_factors, valid_labels,
        verbose=True,
    )

    device = next(model.parameters()).device
    pred = predict_mlp(model, test_factors, test_feat.index, device)
    print(f"    mlp_no_gate: best_epoch={info['best_epoch']}, samples={len(pred)}")

    del handler, dataset, model, train_feat, valid_feat, test_feat
    gc.collect()
    return pred, info


# ── TabNet 工具函数 ──

def build_grouped_features(feature_names: list[str], include_market_state: bool = False) -> list[list[int]]:
    """从 FACTOR_GROUPS 构建 TabNet grouped_features 索引列表。

    Args:
        feature_names: 因子名列表 (210 个)
        include_market_state: 是否在前面包含 market_state 的 3 个组

    Returns:
        list of list[int], 每个子列表是一个组内特征的列索引
    """
    from factor_lab.gated_factor_net import classify_all_factors, GROUP_NAMES

    mapping = classify_all_factors(feature_names)
    n_groups = len(GROUP_NAMES)

    # 因子组索引 (offset by market_state groups if needed)
    offset = 0
    if include_market_state:
        # 3 个 market_state 组: regime(3), index(2), rolling_ic(7+1=8? no, 7 groups-1=7)
        # market_state = [regime_bull, regime_bear, regime_sideways,
        #                 ma_ratio, index_ret_20d, index_vol_20d,
        #                 rolling_ic_momentum, ..., rolling_ic_market_cap]
        # 分 3 组: regime(0-2), index_feats(3-5), rolling_ic(6-12)
        offset = 13  # market_state 维度

    groups = [[] for _ in range(n_groups)]
    for i, fname in enumerate(feature_names):
        gidx = mapping[fname]
        groups[gidx].append(i + offset)

    if include_market_state:
        # 前缀 3 组 market_state
        ms_groups = [
            list(range(0, 3)),    # regime one-hot
            list(range(3, 6)),    # index features
            list(range(6, 13)),   # rolling IC (7 dims)
        ]
        return ms_groups + groups

    return groups


def _prepare_tabnet_data(handler, w: dict, market_state_computer=None, include_market_state: bool = False):
    """TabNet 通用数据准备。

    Returns:
        (train_X, train_y, valid_X, valid_y, test_X, test_index,
         feature_names, grouped_features, market_state_computer)
    """
    from factor_lab.gated_factor_net import MarketStateComputer

    train_feat, train_label = extract_features_labels(handler, w['train_start'], w['train_end'])
    valid_feat, valid_label = extract_features_labels(handler, w['valid_start'], w['valid_end'])
    test_feat, test_label = extract_features_labels(handler, w['pred_start'], w['pred_end'])

    feature_names = list(train_feat.columns)

    train_X = train_feat.values.astype(np.float64)
    valid_X = valid_feat.values.astype(np.float64)
    test_X = test_feat.values.astype(np.float64)
    np.nan_to_num(train_X, copy=False)
    np.nan_to_num(valid_X, copy=False)
    np.nan_to_num(test_X, copy=False)

    train_y = train_label.values.astype(np.float64).ravel()
    valid_y = valid_label.values.astype(np.float64).ravel()
    np.nan_to_num(train_y, copy=False)
    np.nan_to_num(valid_y, copy=False)

    if include_market_state:
        if market_state_computer is None:
            market_state_computer = MarketStateComputer(feature_names)

        all_feat = pd.concat([train_feat, valid_feat, test_feat])
        all_label = pd.concat([train_label, valid_label, test_label])
        market_states = market_state_computer.compute_all_dates_fast(all_feat, all_label)

        def get_state_array(feat_df):
            dates = feat_df.index.get_level_values(0)
            arr = np.array([market_states.loc[d].values for d in dates], dtype=np.float64)
            np.nan_to_num(arr, copy=False)
            return arr

        train_ms = get_state_array(train_feat)
        valid_ms = get_state_array(valid_feat)
        test_ms = get_state_array(test_feat)

        # 拼接: [market_state | factors]
        train_X = np.hstack([train_ms, train_X])
        valid_X = np.hstack([valid_ms, valid_X])
        test_X = np.hstack([test_ms, test_X])

        grouped = build_grouped_features(feature_names, include_market_state=True)
    else:
        grouped = build_grouped_features(feature_names, include_market_state=False)

    return (train_X, train_y, valid_X, valid_y, test_X,
            test_feat.index, feature_names, grouped, market_state_computer)


def _fit_tabnet(train_X, train_y, valid_X, valid_y, grouped_features):
    """创建并训练 TabNetRegressor。"""
    import torch
    from pytorch_tabnet.tab_model import TabNetRegressor

    model = TabNetRegressor(
        n_d=16,
        n_a=16,
        n_steps=5,
        gamma=1.3,
        lambda_sparse=1e-3,
        grouped_features=grouped_features,
        mask_type='sparsemax',
        n_independent=2,
        n_shared=2,
        optimizer_fn=torch.optim.AdamW,
        optimizer_params=dict(lr=2e-3, weight_decay=1e-4),
        scheduler_fn=torch.optim.lr_scheduler.CosineAnnealingLR,
        scheduler_params=dict(T_max=100),
        seed=42,
        verbose=0,
    )

    model.fit(
        train_X, train_y.reshape(-1, 1),
        eval_set=[(valid_X, valid_y.reshape(-1, 1))],
        eval_metric=['mse'],
        max_epochs=200,
        patience=15,
        batch_size=4096,
        virtual_batch_size=512,
    )

    return model


def run_tabnet_window(w: dict, market_state_computer=None) -> tuple[pd.Series, dict]:
    """纯 TabNet + grouped_features (单窗口)。"""
    import torch

    handler, dataset = build_handler_and_dataset(w)
    (train_X, train_y, valid_X, valid_y, test_X,
     test_index, feature_names, grouped, msc) = _prepare_tabnet_data(
        handler, w, market_state_computer, include_market_state=False)

    model = _fit_tabnet(train_X, train_y, valid_X, valid_y, grouped)

    pred_arr = model.predict(test_X).ravel()
    pred = pd.Series(pred_arr, index=test_index, name='score')
    print(f"    tabnet: samples={len(pred)}")

    # 提取特征重要度
    importances = model.feature_importances_
    info = {
        'feature_importances': importances.tolist(),
        'n_groups': len(grouped),
    }

    del handler, dataset, model
    gc.collect()
    return pred, info


def run_tabnet_ms_window(w: dict, market_state_computer=None) -> tuple[pd.Series, dict]:
    """TabNet + market_state 拼接输入 (单窗口)。"""
    import torch
    from factor_lab.gated_factor_net import MarketStateComputer

    handler, dataset = build_handler_and_dataset(w)
    (train_X, train_y, valid_X, valid_y, test_X,
     test_index, feature_names, grouped, msc) = _prepare_tabnet_data(
        handler, w, market_state_computer, include_market_state=True)

    model = _fit_tabnet(train_X, train_y, valid_X, valid_y, grouped)

    pred_arr = model.predict(test_X).ravel()
    pred = pd.Series(pred_arr, index=test_index, name='score')
    print(f"    tabnet_ms: samples={len(pred)}")

    importances = model.feature_importances_
    info = {
        'feature_importances': importances.tolist(),
        'n_groups': len(grouped),
        'market_state_computer': msc,
    }

    del handler, dataset, model
    gc.collect()
    return pred, info


def run_gate_tabnet_window(w: dict, market_state_computer=None) -> tuple[pd.Series, dict]:
    """Gate 预加权 + TabNet 精细选择 (单窗口, 两阶段)。

    Step 1: 训练 GatedFactorNet 获取 gate weights
    Step 2: apply gate weights 到因子
    Step 3: 训练 TabNet on weighted factors
    """
    import torch
    from factor_lab.gated_factor_net import (
        MarketStateComputer, build_group_mask, train_gated_model,
    )

    handler, dataset = build_handler_and_dataset(w)

    train_feat, train_label = extract_features_labels(handler, w['train_start'], w['train_end'])
    valid_feat, valid_label = extract_features_labels(handler, w['valid_start'], w['valid_end'])
    test_feat, test_label = extract_features_labels(handler, w['pred_start'], w['pred_end'])

    feature_names = list(train_feat.columns)

    if market_state_computer is None:
        market_state_computer = MarketStateComputer(feature_names)

    all_feat = pd.concat([train_feat, valid_feat, test_feat])
    all_label = pd.concat([train_label, valid_label, test_label])
    market_states = market_state_computer.compute_all_dates_fast(all_feat, all_label)

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

    # Step 1: 训练 GatedFactorNet 学习 gate weights
    print("    [gate_tabnet] Step 1: Training gate...")
    gate_model, gate_info = train_gated_model(
        train_state, train_factors, train_labels,
        valid_state, valid_factors, valid_labels,
        group_mask, verbose=True,
    )

    # Step 2: apply gate weights 到因子
    device = next(gate_model.parameters()).device
    gate_model.eval()

    def apply_gate_weights(factors: np.ndarray, state_arr: np.ndarray, feat_df) -> np.ndarray:
        dates = feat_df.index.get_level_values(0)
        unique_dates = dates.unique()
        weighted = factors.copy()

        with torch.no_grad():
            for dt in unique_dates:
                dt_state = market_states.loc[dt].values.astype(np.float32)
                np.nan_to_num(dt_state, copy=False)
                state_t = torch.tensor(dt_state, dtype=torch.float32).unsqueeze(0).to(device)
                gw = gate_model.gate(state_t)  # (1, n_groups)
                pfw = (gw @ gate_model.group_mask.T).cpu().numpy()[0]  # (n_features,)

                mask = np.array(dates == dt)
                weighted[mask] *= pfw

        return weighted

    train_weighted = apply_gate_weights(train_factors, train_state, train_feat).astype(np.float64)
    valid_weighted = apply_gate_weights(valid_factors, valid_state, valid_feat).astype(np.float64)
    test_weighted = apply_gate_weights(test_factors, test_state, test_feat).astype(np.float64)

    # Step 3: 训练 TabNet on weighted factors
    print("    [gate_tabnet] Step 2: Training TabNet on gate-weighted factors...")
    grouped = build_grouped_features(feature_names, include_market_state=False)
    tabnet_model = _fit_tabnet(
        train_weighted, train_labels.astype(np.float64).ravel(),
        valid_weighted, valid_labels.astype(np.float64).ravel(),
        grouped,
    )

    pred_arr = tabnet_model.predict(test_weighted).ravel()
    pred = pd.Series(pred_arr, index=test_feat.index, name='score')
    print(f"    gate_tabnet: gate_best={gate_info['best_epoch']}, samples={len(pred)}")

    # Gate 权重分析
    gate_analysis = {}
    test_dates = test_feat.index.get_level_values(0).unique()
    for dt in test_dates:
        dt_state = market_states.loc[dt].values.astype(np.float32)
        np.nan_to_num(dt_state, copy=False)
        state_t = torch.tensor(dt_state, dtype=torch.float32).unsqueeze(0).to(device)
        gw = gate_model.get_gate_weights(state_t).cpu().numpy()[0]
        gate_analysis[dt.strftime('%Y-%m-%d')] = gw.tolist()

    info = {
        'gate_best_epoch': gate_info['best_epoch'],
        'gate_best_val_loss': gate_info['best_val_loss'],
        'gate_weights': gate_analysis,
        'tabnet_importances': tabnet_model.feature_importances_.tolist(),
    }

    del handler, dataset, gate_model, tabnet_model, train_feat, valid_feat, test_feat
    gc.collect()
    return pred, info


def print_tabnet_analysis(all_info: list[dict], feature_names: list[str], strategy_name: str):
    """打印 TabNet 特征重要度分析 (按组汇总)。"""
    from factor_lab.gated_factor_net import classify_all_factors, GROUP_NAMES

    mapping = classify_all_factors(feature_names)

    # 合并所有窗口的 importances
    all_imp = []
    for info in all_info:
        key = 'tabnet_importances' if 'tabnet_importances' in info else 'feature_importances'
        if key in info:
            imp = info[key]
            # 跳过 market_state 前缀 (tabnet_ms 会有 13 个额外维度)
            if len(imp) > len(feature_names):
                imp = imp[len(imp) - len(feature_names):]
            if len(imp) == len(feature_names):
                all_imp.append(imp)

    if not all_imp:
        return

    avg_imp = np.mean(all_imp, axis=0)

    # 按组汇总
    group_imp = {}
    for gname in GROUP_NAMES:
        group_imp[gname] = 0.0

    for i, fname in enumerate(feature_names):
        gidx = mapping[fname]
        gname = GROUP_NAMES[gidx]
        group_imp[gname] += avg_imp[i]

    total = sum(group_imp.values())
    if total <= 0:
        return

    print(f"\n  --- {strategy_name} TabNet 特征重要度 (按组) ---")
    sorted_groups = sorted(group_imp.items(), key=lambda x: x[1], reverse=True)
    for gname, imp in sorted_groups:
        pct = imp / total * 100
        bar = '#' * int(pct / 2)
        print(f"  {gname:<20} {pct:>6.1f}%  {bar}")

    # Top 10 单因子
    top_idx = np.argsort(avg_imp)[::-1][:10]
    print(f"\n  Top 10 单因子:")
    for rank, idx in enumerate(top_idx, 1):
        print(f"    {rank:2d}. {feature_names[idx]:<30} {avg_imp[idx]:.4f}")


# ── Gate 权重分析 ──

def print_gate_analysis(all_gate_info: list[dict], strategy_name: str):
    """打印 gate 权重分析。"""
    from factor_lab.gated_factor_net import GROUP_NAMES

    print(f"\n  --- {strategy_name} Gate 权重分析 ---")
    print(f"  {'日期':<12} " + " ".join(f"{g[:8]:>9}" for g in GROUP_NAMES))
    print("  " + "-" * (12 + 9 * len(GROUP_NAMES)))

    # 合并所有窗口的 gate weights
    all_weights = {}
    for info in all_gate_info:
        if 'gate_weights' in info:
            all_weights.update(info['gate_weights'])

    # 采样显示 (每月第一天)
    sorted_dates = sorted(all_weights.keys())
    shown_months = set()
    for d in sorted_dates:
        month = d[:7]
        if month not in shown_months:
            shown_months.add(month)
            weights = all_weights[d]
            print(f"  {d:<12} " + " ".join(f"{w:>9.3f}" for w in weights))

    # 统计
    if all_weights:
        arr = np.array(list(all_weights.values()))
        print(f"\n  {'均值':<12} " + " ".join(f"{m:>9.3f}" for m in arr.mean(axis=0)))
        print(f"  {'标准差':<12} " + " ".join(f"{s:>9.3f}" for s in arr.std(axis=0)))
        print(f"  {'最小值':<12} " + " ".join(f"{m:>9.3f}" for m in arr.min(axis=0)))
        print(f"  {'最大值':<12} " + " ".join(f"{m:>9.3f}" for m in arr.max(axis=0)))


# ── 主流程 ──

def print_comparison_table(results: dict[str, dict]):
    """打印对比表。"""
    print("\n\n")
    print("=" * 100)
    print("              Exp 015b/c: Gate + TabNet 因子择时 — 对比表")
    print("=" * 100)
    print(f"测试区间: {TEST_START} ~ {TEST_END} | Rolling: {CONFIG_NAME} | Preset: {PRESET}")
    print(f"回测: TopkDropout (topk={TOPK}, n_drop={N_DROP}) | 成交价: 次日开盘 (T+1)")
    print()

    header = (f"{'排名':<4} {'策略':<16} {'Sharpe':>8} {'总收益':>10} {'年化':>10} "
              f"{'超额':>10} {'最大回撤':>10} {'耗时(s)':>8}")
    print(header)
    print("-" * 100)

    sorted_items = sorted(results.items(), key=lambda x: x[1].get('sharpe', 0), reverse=True)
    baseline_sharpe = results.get('baseline', {}).get('sharpe', 0)

    for rank, (name, r) in enumerate(sorted_items, 1):
        sharpe = r.get('sharpe', 0)
        delta = f"({sharpe - baseline_sharpe:+.3f})" if name != 'baseline' else ""
        print(f"{rank:<4} {name:<16} {sharpe:>7.3f} {delta:>8} "
              f"{r.get('total_return', 0):>9.2%} "
              f"{r.get('annual_return', 0):>9.2%} "
              f"{r.get('excess_return', 0):>9.2%} "
              f"{r.get('max_drawdown', 0):>9.2%} "
              f"{r.get('total_time', 0):>7.1f}")

    print("\n" + "=" * 100)


def main():
    parser = argparse.ArgumentParser(description="Exp 015b/c: Gate + TabNet Benchmark")
    parser.add_argument('--strategies', nargs='+', default=ALL_STRATEGIES,
                        choices=ALL_STRATEGIES, help="要运行的策略")
    parser.add_argument('--windows', type=int, default=0,
                        help="限制窗口数 (0=全部)")
    parser.add_argument('--report-only', action='store_true', help="只打印对比表")
    parser.add_argument('--force', action='store_true', help="强制重跑")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.report_only:
        result_file = RESULTS_DIR / "gated_benchmark.json"
        if result_file.exists():
            with open(result_file) as f:
                results = json.load(f)
            print_comparison_table(results)
        else:
            print("没有找到结果文件")
        return

    # 初始化 Qlib
    import multiprocessing
    multiprocessing.set_start_method('fork', force=True)
    import qlib
    from qlib.constant import REG_CN
    qlib.init(provider_uri='~/.qlib/qlib_data/cn_data_bs', region=REG_CN)

    # 生成 rolling windows
    from factor_lab.run_rolling_benchmark import ROLLING_CONFIGS, generate_rolling_windows
    config = ROLLING_CONFIGS[CONFIG_NAME]
    windows = generate_rolling_windows(CONFIG_NAME, config, TEST_START, TEST_END)
    if args.windows > 0:
        windows = windows[:args.windows]

    print(f"\n{'='*70}")
    print(f"Exp 015b: Gated Factor Timing Benchmark")
    print(f"Strategies: {args.strategies}")
    print(f"Windows: {len(windows)}")
    print(f"{'='*70}")

    for w in windows:
        print(f"  Window {w['window_num']}: "
              f"Train [{w['train_start']}, {w['train_end']}]  "
              f"Valid [{w['valid_start']}, {w['valid_end']}]  "
              f"Pred [{w['pred_start']}, {w['pred_end']}]")

    all_results = {}

    # ── Baseline + Timing V2 共享 LGB 预测 ──
    baseline_combined = None
    need_lgb = 'baseline' in args.strategies or 'timing_v2' in args.strategies

    if need_lgb:
        print(f"\n\n{'='*70}")
        print("  策略: baseline + timing_v2 (共享 LGB 预测)")
        print(f"{'='*70}")
        t0_lgb = time.time()
        all_preds = []
        for w in windows:
            print(f"\n  Window {w['window_num']}/{len(windows)}")
            pred = run_baseline_window(w)
            all_preds.append(pred)

        baseline_combined = pd.concat(all_preds)
        if baseline_combined.index.duplicated().any():
            baseline_combined = baseline_combined[~baseline_combined.index.duplicated(keep='last')]
        lgb_time = round(time.time() - t0_lgb, 1)

    if 'baseline' in args.strategies and baseline_combined is not None:
        bt = run_backtest(baseline_combined)
        bt['total_time'] = lgb_time
        all_results['baseline'] = bt
        print(f"\n  baseline: Sharpe={bt['sharpe']:.3f}  Return={bt['total_return']:.2%}  "
              f"MDD={bt['max_drawdown']:.2%}  Time={bt['total_time']:.1f}s")

    if 'timing_v2' in args.strategies and baseline_combined is not None:
        t0 = time.time()
        from qlib.data import D
        index_data = D.features(["SH000300"], ["$close"],
                                start_time="2016-01-01", end_time="2027-01-01")
        index_close = index_data.droplevel(0)["$close"]

        pred_dates = baseline_combined.index.get_level_values(0).unique().sort_values()
        regimes = {}
        for dt in pred_dates:
            regimes[dt] = detect_regime_for_date(index_close, dt)

        regime_counts = pd.Series(list(regimes.values())).value_counts()
        avg_topk = sum(TOPK_BY_REGIME[r] * c for r, c in regime_counts.items()) / len(regimes)
        print(f"\n  Regime 分布: {dict(regime_counts)}, 平均 TopK={avg_topk:.1f}")

        adaptive_topk = int(round(avg_topk))
        n_drop_adaptive = max(1, adaptive_topk // 4)
        bt = run_backtest(baseline_combined, topk=adaptive_topk, n_drop=n_drop_adaptive)
        bt['total_time'] = round(lgb_time + time.time() - t0, 1)
        bt['avg_topk'] = avg_topk
        all_results['timing_v2'] = bt
        print(f"\n  timing_v2: Sharpe={bt['sharpe']:.3f}  Return={bt['total_return']:.2%}  "
              f"MDD={bt['max_drawdown']:.2%}  Time={bt['total_time']:.1f}s")

    # 策略间强制 GC 释放内存
    baseline_combined = None  # 释放大对象
    gc.collect()

    # ── Gate MLP: 端到端 ──
    if 'gate_mlp' in args.strategies:
        print(f"\n\n{'='*70}")
        print("  策略: gate_mlp (可学习 Gate + MLP)")
        print(f"{'='*70}")
        t0 = time.time()
        all_preds = []
        all_gate_info = []
        for w in windows:
            print(f"\n  Window {w['window_num']}/{len(windows)}")
            pred, info = run_gate_mlp_window(w)
            all_preds.append(pred)
            all_gate_info.append(info)

        combined_gate = pd.concat(all_preds)
        if combined_gate.index.duplicated().any():
            combined_gate = combined_gate[~combined_gate.index.duplicated(keep='last')]

        bt = run_backtest(combined_gate)
        bt['total_time'] = round(time.time() - t0, 1)
        all_results['gate_mlp'] = bt
        print(f"\n  gate_mlp: Sharpe={bt['sharpe']:.3f}  Return={bt['total_return']:.2%}  "
              f"MDD={bt['max_drawdown']:.2%}  Time={bt['total_time']:.1f}s")

        print_gate_analysis(all_gate_info, 'gate_mlp')

    gc.collect()

    # ── Gate LGB: 混合 ──
    if 'gate_lgb' in args.strategies:
        print(f"\n\n{'='*70}")
        print("  策略: gate_lgb (Gate 权重 + LGB)")
        print(f"{'='*70}")
        t0 = time.time()
        all_preds = []
        all_gate_info = []
        for w in windows:
            print(f"\n  Window {w['window_num']}/{len(windows)}")
            pred, info = run_gate_lgb_window(w)
            all_preds.append(pred)
            all_gate_info.append(info)

        combined_gate_lgb = pd.concat(all_preds)
        if combined_gate_lgb.index.duplicated().any():
            combined_gate_lgb = combined_gate_lgb[~combined_gate_lgb.index.duplicated(keep='last')]

        bt = run_backtest(combined_gate_lgb)
        bt['total_time'] = round(time.time() - t0, 1)
        all_results['gate_lgb'] = bt
        print(f"\n  gate_lgb: Sharpe={bt['sharpe']:.3f}  Return={bt['total_return']:.2%}  "
              f"MDD={bt['max_drawdown']:.2%}  Time={bt['total_time']:.1f}s")

        print_gate_analysis(all_gate_info, 'gate_lgb')

    gc.collect()

    # ── MLP No Gate: 消融 ──
    if 'mlp_no_gate' in args.strategies:
        print(f"\n\n{'='*70}")
        print("  策略: mlp_no_gate (MLP 无 Gate, 消融)")
        print(f"{'='*70}")
        t0 = time.time()
        all_preds = []
        for w in windows:
            print(f"\n  Window {w['window_num']}/{len(windows)}")
            pred, info = run_mlp_no_gate_window(w)
            all_preds.append(pred)

        combined_mlp = pd.concat(all_preds)
        if combined_mlp.index.duplicated().any():
            combined_mlp = combined_mlp[~combined_mlp.index.duplicated(keep='last')]

        bt = run_backtest(combined_mlp)
        bt['total_time'] = round(time.time() - t0, 1)
        all_results['mlp_no_gate'] = bt
        print(f"\n  mlp_no_gate: Sharpe={bt['sharpe']:.3f}  Return={bt['total_return']:.2%}  "
              f"MDD={bt['max_drawdown']:.2%}  Time={bt['total_time']:.1f}s")

    # 获取 feature_names (用于 TabNet 分析, 只需获取一次)
    _feature_names = None
    if any(s in args.strategies for s in ['tabnet', 'tabnet_ms', 'gate_tabnet']):
        handler, _ = build_handler_and_dataset(windows[0])
        feat, _ = extract_features_labels(handler, windows[0]['train_start'], windows[0]['train_end'])
        _feature_names = list(feat.columns)
        del handler, feat
        gc.collect()

    # ── TabNet: 纯 TabNet ──
    if 'tabnet' in args.strategies:
        print(f"\n\n{'='*70}")
        print("  策略: tabnet (纯 TabNet + grouped_features)")
        print(f"{'='*70}")
        t0 = time.time()
        all_preds = []
        all_tabnet_info = []
        for w in windows:
            print(f"\n  Window {w['window_num']}/{len(windows)}")
            pred, info = run_tabnet_window(w)
            all_preds.append(pred)
            all_tabnet_info.append(info)

        combined = pd.concat(all_preds)
        if combined.index.duplicated().any():
            combined = combined[~combined.index.duplicated(keep='last')]

        bt = run_backtest(combined)
        bt['total_time'] = round(time.time() - t0, 1)
        all_results['tabnet'] = bt
        print(f"\n  tabnet: Sharpe={bt['sharpe']:.3f}  Return={bt['total_return']:.2%}  "
              f"MDD={bt['max_drawdown']:.2%}  Time={bt['total_time']:.1f}s")

        if _feature_names:
            print_tabnet_analysis(all_tabnet_info, _feature_names, 'tabnet')

    gc.collect()

    # ── TabNet MS: TabNet + Market State ──
    if 'tabnet_ms' in args.strategies:
        print(f"\n\n{'='*70}")
        print("  策略: tabnet_ms (TabNet + Market State)")
        print(f"{'='*70}")
        t0 = time.time()
        all_preds = []
        all_tabnet_info = []
        for w in windows:
            print(f"\n  Window {w['window_num']}/{len(windows)}")
            pred, info = run_tabnet_ms_window(w)
            all_preds.append(pred)
            all_tabnet_info.append(info)

        combined = pd.concat(all_preds)
        if combined.index.duplicated().any():
            combined = combined[~combined.index.duplicated(keep='last')]

        bt = run_backtest(combined)
        bt['total_time'] = round(time.time() - t0, 1)
        all_results['tabnet_ms'] = bt
        print(f"\n  tabnet_ms: Sharpe={bt['sharpe']:.3f}  Return={bt['total_return']:.2%}  "
              f"MDD={bt['max_drawdown']:.2%}  Time={bt['total_time']:.1f}s")

        if _feature_names:
            print_tabnet_analysis(all_tabnet_info, _feature_names, 'tabnet_ms')

    gc.collect()

    # ── Gate TabNet: Gate + TabNet 两阶段 ──
    if 'gate_tabnet' in args.strategies:
        print(f"\n\n{'='*70}")
        print("  策略: gate_tabnet (Gate 预加权 + TabNet)")
        print(f"{'='*70}")
        t0 = time.time()
        all_preds = []
        all_gate_tabnet_info = []
        for w in windows:
            print(f"\n  Window {w['window_num']}/{len(windows)}")
            pred, info = run_gate_tabnet_window(w)
            all_preds.append(pred)
            all_gate_tabnet_info.append(info)

        combined = pd.concat(all_preds)
        if combined.index.duplicated().any():
            combined = combined[~combined.index.duplicated(keep='last')]

        bt = run_backtest(combined)
        bt['total_time'] = round(time.time() - t0, 1)
        all_results['gate_tabnet'] = bt
        print(f"\n  gate_tabnet: Sharpe={bt['sharpe']:.3f}  Return={bt['total_return']:.2%}  "
              f"MDD={bt['max_drawdown']:.2%}  Time={bt['total_time']:.1f}s")

        print_gate_analysis(all_gate_tabnet_info, 'gate_tabnet')
        if _feature_names:
            print_tabnet_analysis(all_gate_tabnet_info, _feature_names, 'gate_tabnet')

    gc.collect()

    # ── 汇总 ──
    print_comparison_table(all_results)

    # 保存 (合并已有结果)
    result_file = RESULTS_DIR / "gated_benchmark.json"
    existing = {}
    if result_file.exists():
        with open(result_file) as f:
            existing = json.load(f)
    # Convert non-serializable values
    for k, v in all_results.items():
        existing[k] = {sk: sv for sk, sv in v.items() if isinstance(sv, (int, float, str, list))}
    with open(result_file, 'w') as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {result_file}")


if __name__ == '__main__':
    main()
