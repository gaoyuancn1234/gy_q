#!/usr/bin/env python3
"""
Qlib 模型大比武 - 统一 Benchmark
在相同数据集/验证集/测试集上比较所有可用模型

数据: CSI300, BaoStock, 2019-01 ~ 2026-02
训练: 2019-01-01 ~ 2023-06-30
验证: 2023-07-01 ~ 2023-12-31
测试: 2024-01-01 ~ 2026-02-05
回测: TopkDropoutStrategy, topk=12, n_drop=3, deal_price=close (T+1)
"""
import sys
import time
import warnings
import traceback
import faulthandler
from pathlib import Path
import pandas as pd

warnings.filterwarnings('ignore')
faulthandler.enable()  # 防止 PyTorch + macOS SIGSEGV


# ============ 时间划分 (所有模型共用) ============
TRAIN_START = '2019-01-01'
TRAIN_END = '2023-06-30'
VALID_START = '2023-07-01'
VALID_END = '2023-12-31'
TEST_START = '2024-01-01'
TEST_END = '2026-02-05'

TOPK = 12
N_DROP = 3

# ============ 因子预设 ============
# 可选: 'alpha158', 'alpha158_ext', 'alpha158_val', 'full'
HANDLER_PRESET = 'alpha158'


def init_qlib():
    import qlib
    from qlib.constant import REG_CN
    qlib.init(provider_uri='~/.qlib/qlib_data/cn_data_bs', region=REG_CN)


def build_alpha158_handler():
    from qlib.contrib.data.handler import Alpha158
    return Alpha158(
        start_time=TRAIN_START,
        end_time=TEST_END,
        fit_start_time=TRAIN_START,
        fit_end_time=TRAIN_END,
        instruments='csi300',
    )


def build_handler_for_benchmark():
    """根据 HANDLER_PRESET 构建 handler（可插拔因子集）"""
    if HANDLER_PRESET == 'alpha158':
        return build_alpha158_handler()

    from factor_lab.factors.presets import build_handler
    return build_handler(
        HANDLER_PRESET,
        start_time=TRAIN_START, end_time=TEST_END,
        fit_start_time=TRAIN_START, fit_end_time=TRAIN_END,
    )


def get_d_feat() -> int:
    """动态获取因子数量（DL 模型 d_feat 参数）"""
    if HANDLER_PRESET == 'alpha158':
        return 158
    from factor_lab.factors.presets import get_preset_factor_count
    return get_preset_factor_count(HANDLER_PRESET)


def build_dataset_h(handler):
    from qlib.data.dataset import DatasetH
    return DatasetH(handler=handler, segments={
        "train": (TRAIN_START, TRAIN_END),
        "valid": (VALID_START, VALID_END),
        "test": (TEST_START, TEST_END),
    })


def build_ts_dataset_h(handler, step_len=8):
    from qlib.data.dataset import TSDatasetH
    return TSDatasetH(handler=handler, segments={
        "train": (TRAIN_START, TRAIN_END),
        "valid": (VALID_START, VALID_END),
        "test": (TEST_START, TEST_END),
    }, step_len=step_len)


def filter_test_predictions(pred):
    """只保留测试区间的预测"""
    import pandas as pd
    if isinstance(pred.index, pd.MultiIndex):
        dates = pred.index.get_level_values(0)
        mask = (dates >= pd.Timestamp(TEST_START)) & (dates <= pd.Timestamp(TEST_END))
        return pred[mask]
    return pred


def run_backtest(pred):
    """统一回测逻辑"""
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
            "freq": "day", "limit_threshold": 0.095, "deal_price": "close",  # 2026-08-30 由 open 改为 close (对齐生产 signal_config)
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


# ============ 传统 ML 模型 (DatasetH + Alpha158) ============

def run_lightgbm():
    from qlib.contrib.model.gbdt import LGBModel
    dataset = build_dataset_h(build_handler_for_benchmark())
    model = LGBModel(
        loss="mse", learning_rate=0.01, num_leaves=64,
        num_boost_round=500, early_stopping_rounds=80,
        feature_fraction=0.75, bagging_fraction=0.75, bagging_freq=5,
        lambda_l1=0.1, lambda_l2=0.1, min_data_in_leaf=80,
    )
    model.fit(dataset)
    return filter_test_predictions(model.predict(dataset))


def run_double_ensemble():
    from qlib.contrib.model.double_ensemble import DEnsembleModel
    dataset = build_dataset_h(build_handler_for_benchmark())
    model = DEnsembleModel(
        base_model="gbm", loss="mse",
        num_models=6, num_boost_round=150,
        enable_sr=True, enable_fs=True, decay=0.5,
    )
    model.fit(dataset)
    return filter_test_predictions(model.predict(dataset))


def run_xgboost():
    from qlib.contrib.model.xgboost import XGBModel
    dataset = build_dataset_h(build_handler_for_benchmark())
    model = XGBModel(
        learning_rate=0.01, max_depth=6,
        n_estimators=500, early_stopping_rounds=80,
        reg_alpha=0.1, reg_lambda=0.1,
        subsample=0.75, colsample_bytree=0.75,
    )
    model.fit(dataset)
    return filter_test_predictions(model.predict(dataset))


def run_catboost():
    from qlib.contrib.model.catboost_model import CatBoostModel
    dataset = build_dataset_h(build_handler_for_benchmark())
    # 不传 verbose/logging_level 避免跟 fit 内部 verbose_eval 冲突
    model = CatBoostModel(
        loss="RMSE", learning_rate=0.01, depth=6,
        iterations=500, l2_leaf_reg=3.0, subsample=0.75,
    )
    model.fit(dataset, early_stopping_rounds=80)
    return filter_test_predictions(model.predict(dataset))


def run_linear():
    """Ridge regression (sklearn 直接实现, 绕过 Qlib LinearModel DK_L 空数据问题)"""
    import numpy as np
    from sklearn.linear_model import Ridge
    from qlib.data.dataset.handler import DataHandlerLP

    handler = build_handler_for_benchmark()
    dataset = build_dataset_h(handler)

    # 用 DK_I (inference processors) 获取数据, NaN 直接填 0
    df_train = dataset.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_I)
    X_train = np.nan_to_num(df_train["feature"].values, nan=0.0)
    y_train = np.nan_to_num(df_train["label"].values.ravel(), nan=0.0)
    print(f"  Ridge: training on {len(X_train)} samples")

    model = Ridge(alpha=0.05)
    model.fit(X_train, y_train)

    df_test = dataset.prepare("test", col_set=["feature", "label"], data_key=DataHandlerLP.DK_I)
    X_test = np.nan_to_num(df_test["feature"].values, nan=0.0)
    pred = model.predict(X_test)
    return pd.Series(pred.ravel(), index=df_test.index)


# ============ DL 模型 (TSDatasetH + Alpha158, _ts 变体) ============
# CPU 优化: step_len=8, hidden=32, epochs=30, batch=4096

DL_STEP = 8
DL_HIDDEN = 32
DL_EPOCHS = 30
DL_BATCH = 4096
DL_EARLY = 8


def patch_dl_nan(model):
    """通用 NaN 处理: 包装 DataLoader 对所有浮点 Tensor 做 nan_to_num
    支持所有 DL 模型 (LSTM/GRU/ALSTM/GATs/Transformer/Localformer)"""
    import types
    import torch

    orig_train = model.train_epoch
    orig_test = model.test_epoch

    class NanCleanLoader:
        """DataLoader 包装器: 清洗所有浮点 Tensor 中的 NaN"""
        def __init__(self, loader):
            self._loader = loader
        def __iter__(self):
            for item in self._loader:
                if isinstance(item, (list, tuple)):
                    yield tuple(
                        torch.nan_to_num(x, nan=0.0) if isinstance(x, torch.Tensor) and x.is_floating_point() else x
                        for x in item
                    )
                elif isinstance(item, torch.Tensor):
                    yield torch.nan_to_num(item, nan=0.0)
                else:
                    yield item
        def __len__(self):
            return len(self._loader)

    def wrapped_train(self, data_loader):
        return orig_train(NanCleanLoader(data_loader))

    def wrapped_test(self, data_loader):
        return orig_test(NanCleanLoader(data_loader))

    model.train_epoch = types.MethodType(wrapped_train, model)
    model.test_epoch = types.MethodType(wrapped_test, model)
    return model


def safe_dl_fit(model, dataset):
    """安全调用 DL 模型 fit（处理 NaN 导致的 UnboundLocalError）"""
    try:
        model.fit(dataset)
    except UnboundLocalError:
        # best_param 未赋值 = 所有 epoch 分数为 nan → 用最后 epoch 的参数
        if model.fitted:
            print("  警告: 训练分数为 NaN，使用最后 epoch 的模型参数")
        else:
            raise


def safe_dl_predict(model, dataset):
    """通用 DL 预测: 自动检测 NN 属性，NaN 清洗，n_jobs=0"""
    import torch
    import numpy as np
    from qlib.data.dataset.handler import DataHandlerLP
    from torch.utils.data import DataLoader

    # 自动检测 neural network 属性
    nn_model = None
    for attr in ['LSTM_model', 'GRU_model', 'ALSTM_model', 'GAT_model', 'model', 'tabnet_model']:
        candidate = getattr(model, attr, None)
        if candidate is not None and hasattr(candidate, 'eval') and callable(candidate.eval):
            nn_model = candidate
            break
    if nn_model is None:
        raise RuntimeError(f"Cannot find NN attribute for {type(model).__name__}")

    batch_size = getattr(model, 'batch_size', DL_BATCH)
    dl_test = dataset.prepare("test", col_set=["feature", "label"], data_key=DataHandlerLP.DK_I)
    dl_test.config(fillna_type="ffill+bfill")
    test_loader = DataLoader(dl_test, batch_size=batch_size, num_workers=0)
    nn_model.eval()
    preds = []
    for data in test_loader:
        if isinstance(data, (list, tuple)):
            data = data[0]
        feature = torch.nan_to_num(data[:, :, 0:-1], nan=0.0).to(model.device)
        with torch.no_grad():
            pred = nn_model(feature.float()).detach().cpu().numpy()
        preds.append(pred.ravel())
    return pd.Series(np.concatenate(preds), index=dl_test.get_index())


def run_lstm():
    from qlib.contrib.model.pytorch_lstm_ts import LSTM
    dataset = build_ts_dataset_h(build_handler_for_benchmark(), step_len=DL_STEP)
    model = LSTM(
        d_feat=get_d_feat(), hidden_size=DL_HIDDEN, num_layers=2, dropout=0.0,
        n_epochs=DL_EPOCHS, lr=0.001, batch_size=DL_BATCH, early_stop=DL_EARLY,
        optimizer="adam", GPU=-1, n_jobs=0,
    )
    patch_dl_nan(model)
    safe_dl_fit(model, dataset)
    return filter_test_predictions(safe_dl_predict(model, dataset))


def run_gru():
    from qlib.contrib.model.pytorch_gru_ts import GRU
    dataset = build_ts_dataset_h(build_handler_for_benchmark(), step_len=DL_STEP)
    model = GRU(
        d_feat=get_d_feat(), hidden_size=DL_HIDDEN, num_layers=2, dropout=0.0,
        n_epochs=DL_EPOCHS, lr=0.001, batch_size=DL_BATCH, early_stop=DL_EARLY,
        optimizer="adam", GPU=-1, n_jobs=0,
    )
    patch_dl_nan(model)
    safe_dl_fit(model, dataset)
    return filter_test_predictions(safe_dl_predict(model, dataset))


def run_alstm():
    from qlib.contrib.model.pytorch_alstm_ts import ALSTM
    dataset = build_ts_dataset_h(build_handler_for_benchmark(), step_len=DL_STEP)
    model = ALSTM(
        d_feat=get_d_feat(), hidden_size=DL_HIDDEN, num_layers=2, dropout=0.0,
        n_epochs=DL_EPOCHS, lr=0.001, batch_size=DL_BATCH, early_stop=DL_EARLY,
        optimizer="adam", GPU=-1, n_jobs=0,
    )
    patch_dl_nan(model)
    safe_dl_fit(model, dataset)
    return filter_test_predictions(safe_dl_predict(model, dataset))


def run_transformer():
    from qlib.contrib.model.pytorch_transformer_ts import TransformerModel
    dataset = build_ts_dataset_h(build_handler_for_benchmark(), step_len=DL_STEP)
    model = TransformerModel(
        d_feat=get_d_feat(), d_model=32, nhead=2, num_layers=2, dropout=0.1,
        n_epochs=DL_EPOCHS, lr=0.0001, batch_size=DL_BATCH, early_stop=DL_EARLY,
    )
    patch_dl_nan(model)
    safe_dl_fit(model, dataset)
    return filter_test_predictions(safe_dl_predict(model, dataset))


def run_gats():
    from qlib.contrib.model.pytorch_gats_ts import GATs
    dataset = build_ts_dataset_h(build_handler_for_benchmark(), step_len=DL_STEP)
    model = GATs(
        d_feat=get_d_feat(), hidden_size=DL_HIDDEN, num_layers=2, dropout=0.0,
        n_epochs=DL_EPOCHS, lr=0.001, batch_size=DL_BATCH, early_stop=DL_EARLY,
        optimizer="adam", GPU=-1, n_jobs=0,
    )
    patch_dl_nan(model)
    safe_dl_fit(model, dataset)
    return filter_test_predictions(safe_dl_predict(model, dataset))


def run_localformer():
    from qlib.contrib.model.pytorch_localformer_ts import LocalformerModel
    dataset = build_ts_dataset_h(build_handler_for_benchmark(), step_len=DL_STEP)
    model = LocalformerModel(
        d_feat=get_d_feat(), d_model=32, nhead=2, num_layers=2, dropout=0.1,
        n_epochs=DL_EPOCHS, lr=0.0001, batch_size=DL_BATCH, early_stop=DL_EARLY,
        optimizer="adam", GPU=-1, n_jobs=0,
    )
    patch_dl_nan(model)
    safe_dl_fit(model, dataset)
    return filter_test_predictions(safe_dl_predict(model, dataset))


def run_tabnet():
    from qlib.contrib.model.pytorch_tabnet import TabnetModel
    dataset = build_dataset_h(build_handler_for_benchmark())  # TabNet 用 DatasetH
    model = TabnetModel(
        d_feat=get_d_feat(), n_d=32, n_a=32,
        n_shared=2, n_ind=2, n_steps=3,
        n_epochs=DL_EPOCHS, lr=0.001, batch_size=DL_BATCH, early_stop=DL_EARLY,
        optimizer="adam", GPU=-1,
        pretrain=False,  # 关闭 pretrain（我们的 dataset 没有 pretrain segment）
    )
    model.fit(dataset)
    return filter_test_predictions(model.predict(dataset))


# ============ 主流程 ============

ALL_MODELS = [
    # 传统 ML (快)
    ("LightGBM",       "Tree",       run_lightgbm),
    ("XGBoost",        "Tree",       run_xgboost),
    ("CatBoost",       "Tree",       run_catboost),
    ("DoubleEnsemble", "Ensemble",   run_double_ensemble),
    ("Ridge",          "Linear",     run_linear),
    # DL 序列模型
    ("LSTM",           "DL-RNN",     run_lstm),
    ("GRU",            "DL-RNN",     run_gru),
    ("ALSTM",          "DL-Attn",    run_alstm),
    ("GATs",           "DL-Graph",   run_gats),
    ("Transformer",    "DL-Attn",    run_transformer),
    ("Localformer",    "DL-Attn",    run_localformer),
    # 特殊架构
    ("TabNet",         "DL-Tab",     run_tabnet),
]


RESULTS_DIR = Path(__file__).parent / "benchmark_results"
RESULTS_FILE = RESULTS_DIR / "results.json"


def load_results():
    """加载已有结果（支持断点续跑）"""
    import json
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE, encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_result(name, result):
    """每跑完一个模型立即保存"""
    import json
    RESULTS_DIR.mkdir(exist_ok=True)
    results = load_results()
    results[name] = result
    with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def print_report(results):
    """打印结论报告"""
    print("\n\n")
    print("=" * 100)
    print("                         Benchmark 结论报告")
    print("=" * 100)
    print(f"\n测试区间: {TEST_START} ~ {TEST_END} | 基准: 沪深300")
    print(f"策略: TopkDropout (topk={TOPK}, n_drop={N_DROP}) | 成交价: 次日开盘 (T+1)\n")

    valid_results = {k: v for k, v in results.items() if 'total_return' in v}
    failed_results = {k: v for k, v in results.items() if 'error' in v}
    sorted_models = sorted(valid_results.items(), key=lambda x: x[1].get('sharpe', 0), reverse=True)

    header = f"{'排名':<4} {'模型':<16} {'类别':<12} {'总收益':>10} {'年化':>10} {'超额':>10} {'Sharpe':>8} {'最大回撤':>10} {'训练(s)':>8}"
    print(header)
    print("-" * 100)

    for rank, (name, r) in enumerate(sorted_models, 1):
        print(f"{rank:<4} {name:<16} {r['category']:<12} "
              f"{r['total_return']:>9.2%} {r['annual_return']:>9.2%} "
              f"{r.get('excess_return', 0):>9.2%} {r.get('sharpe', 0):>7.3f} "
              f"{r.get('max_drawdown', 0):>9.2%} {r.get('train_time', 0):>7.1f}")

    if failed_results:
        print(f"\n失败模型:")
        for name, r in failed_results.items():
            print(f"  {name} ({r['category']}): {r['error'][:80]}")

    if sorted_models:
        bench = sorted_models[0][1].get('bench_return', 0)
        print(f"\n基准 (沪深300): {bench:.2%}")
        best_name, best_r = sorted_models[0]
        print(f"\n最佳模型 (Sharpe): {best_name} ({best_r.get('sharpe', 0):.3f})")

    print("\n" + "=" * 100)


DL_CATEGORIES = {"LSTM", "GRU", "ALSTM", "GATs", "Transformer", "Localformer", "TabNet"}


def run_single_model(name):
    """子进程入口：单独跑一个 DL 模型"""
    import os, torch
    # 限制线程数，防止 macOS + Python 3.13 + PyTorch SIGSEGV
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    torch.set_num_threads(1)

    sys.path.insert(0, str(Path(__file__).parent))
    init_qlib()

    model_map = {m[0]: m for m in ALL_MODELS}
    if name not in model_map:
        print(f"Unknown model: {name}")
        return

    _, category, run_fn = model_map[name]
    print(f"\n[子进程] 运行 {name} ({category})")

    t0 = time.time()
    try:
        pred = run_fn()
        elapsed_train = time.time() - t0
        print(f"  训练+预测完成 ({elapsed_train:.1f}s), 预测样本: {len(pred)}")

        t1 = time.time()
        bt_result = run_backtest(pred)
        elapsed_bt = time.time() - t1

        bt_result['train_time'] = elapsed_train
        bt_result['category'] = category
        save_result(name, bt_result)

        print(f"  回测完成 ({elapsed_bt:.1f}s)")
        print(f"  总收益: {bt_result.get('total_return', 0):.2%}")
        print(f"  年化: {bt_result.get('annual_return', 0):.2%}")
        print(f"  Sharpe: {bt_result.get('sharpe', 0):.3f}")
        print(f"  最大回撤: {bt_result.get('max_drawdown', 0):.2%}")

    except Exception as e:
        elapsed = time.time() - t0
        print(f"  FAILED ({elapsed:.1f}s): {e}")
        traceback.print_exc()
        save_result(name, {'error': str(e), 'category': category, 'train_time': elapsed})


def main():
    import subprocess

    sys.path.insert(0, str(Path(__file__).parent))
    init_qlib()

    existing = load_results()
    done = {k for k, v in existing.items() if 'total_return' in v}

    # 清除之前的错误记录，允许重试
    error_keys = [k for k, v in existing.items() if 'error' in v]
    if error_keys:
        import json
        print(f"清除失败记录，将重试: {', '.join(error_keys)}")
        for k in error_keys:
            del existing[k]
        RESULTS_DIR.mkdir(exist_ok=True)
        with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)

    print("=" * 80)
    print("Qlib 模型 Benchmark")
    print(f"因子预设: {HANDLER_PRESET} (d_feat={get_d_feat()})")
    print(f"训练: {TRAIN_START} ~ {TRAIN_END}")
    print(f"验证: {VALID_START} ~ {VALID_END}")
    print(f"测试: {TEST_START} ~ {TEST_END}")
    print(f"策略: TopK={TOPK}, N_drop={N_DROP}, deal_price=close (T+1)")
    print(f"DL参数: step={DL_STEP}, hidden={DL_HIDDEN}, epochs={DL_EPOCHS}, batch={DL_BATCH}")
    if done:
        print(f"已完成: {', '.join(sorted(done))} (跳过)")
    print("=" * 80)

    for i, (name, category, run_fn) in enumerate(ALL_MODELS):
        if name in done:
            print(f"\n[{i+1}/{len(ALL_MODELS)}] {name} -- 已有结果，跳过")
            continue

        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(ALL_MODELS)}] {name} ({category})")
        print(f"{'='*60}")

        # DL 模型在子进程中运行（避免 tree 模型残留状态导致 PyTorch SIGSEGV）
        if name in DL_CATEGORIES:
            print(f"  [子进程模式]...")
            import os
            script_path = str(Path(__file__).resolve())
            env = os.environ.copy()
            env['OMP_NUM_THREADS'] = '1'
            env['MKL_NUM_THREADS'] = '1'
            cmd = [sys.executable, script_path, "--model", name]
            if HANDLER_PRESET != 'alpha158':
                cmd.extend(["--preset", HANDLER_PRESET])
            result = subprocess.run(
                cmd, capture_output=False, timeout=3600, env=env,
            )
            # 结果已由子进程直接写入 results.json
            new_results = load_results()
            if name in new_results and 'total_return' in new_results[name]:
                r = new_results[name]
                print(f"  [子进程完成] 总收益: {r['total_return']:.2%}, Sharpe: {r.get('sharpe', 0):.3f}")
            elif name in new_results and 'error' in new_results[name]:
                print(f"  [子进程失败] {new_results[name]['error'][:80]}")
            else:
                print(f"  [子进程异常] 未产生结果 (exit code: {result.returncode})")
                save_result(name, {'error': f'subprocess exit {result.returncode}', 'category': category, 'train_time': 0})
            continue

        # 非 DL 模型直接在主进程运行
        t0 = time.time()
        try:
            pred = run_fn()
            elapsed_train = time.time() - t0
            print(f"  训练+预测完成 ({elapsed_train:.1f}s), 预测样本: {len(pred)}")

            t1 = time.time()
            bt_result = run_backtest(pred)
            elapsed_bt = time.time() - t1

            bt_result['train_time'] = elapsed_train
            bt_result['category'] = category
            save_result(name, bt_result)

            print(f"  回测完成 ({elapsed_bt:.1f}s)")
            print(f"  总收益: {bt_result.get('total_return', 0):.2%}")
            print(f"  年化: {bt_result.get('annual_return', 0):.2%}")
            print(f"  Sharpe: {bt_result.get('sharpe', 0):.3f}")
            print(f"  最大回撤: {bt_result.get('max_drawdown', 0):.2%}")

        except Exception as e:
            elapsed = time.time() - t0
            print(f"  FAILED ({elapsed:.1f}s): {e}")
            traceback.print_exc()
            save_result(name, {'error': str(e), 'category': category, 'train_time': elapsed})

    # 最终报告
    print_report(load_results())


if __name__ == '__main__':
    # 支持 --preset 参数切换因子集
    if '--preset' in sys.argv:
        idx = sys.argv.index('--preset')
        if idx + 1 < len(sys.argv):
            HANDLER_PRESET = sys.argv[idx + 1]
            sys.argv.pop(idx)  # remove --preset
            sys.argv.pop(idx)  # remove value

    if len(sys.argv) >= 3 and sys.argv[1] == '--model':
        # 子进程模式: python benchmark_models.py --model LSTM
        run_single_model(sys.argv[2])
    else:
        main()
