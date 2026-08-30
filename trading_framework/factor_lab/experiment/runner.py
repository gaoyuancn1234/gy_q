"""实验运行器 — 因子集 × 模型 的网格实验

在统一的数据划分下，对不同因子预设和模型组合进行回测对比。
"""
import json
import time
import traceback
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path(__file__).parent.parent / "results" / "experiments"


def run_experiment(preset_name: str, model_name: str = "lightgbm",
                   train_start: str = "2019-01-01", train_end: str = "2023-06-30",
                   valid_start: str = "2023-07-01", valid_end: str = "2023-12-31",
                   test_start: str = "2024-01-01", test_end: str = "2026-02-05",
                   topk: int = 12, n_drop: int = 3) -> dict:
    """运行单个实验

    Args:
        preset_name: 因子预设名
        model_name: 模型名 (lightgbm, xgboost, catboost, ridge)
        *时间参数*

    Returns:
        结果字典
    """
    from factor_lab.factors.presets import build_handler, get_preset_factor_count

    import qlib
    from qlib.data.dataset import DatasetH
    from qlib.contrib.evaluate import backtest_daily
    from qlib.utils import init_instance_by_config

    print(f"\n{'='*60}")
    print(f"实验: {preset_name} × {model_name}")
    print(f"  因子数: {get_preset_factor_count(preset_name)}")
    print(f"  测试区间: {test_start} ~ {test_end}")
    print(f"{'='*60}")

    t0 = time.time()

    # 构建 handler 和 dataset
    handler = build_handler(
        preset_name, start_time=train_start, end_time=test_end,
        fit_start_time=train_start, fit_end_time=train_end,
    )
    dataset = DatasetH(handler=handler, segments={
        "train": (train_start, train_end),
        "valid": (valid_start, valid_end),
        "test": (test_start, test_end),
    })

    # 训练模型
    model = _build_model(model_name)
    model.fit(dataset)

    # 预测
    pred = model.predict(dataset)
    if isinstance(pred.index, pd.MultiIndex):
        dates = pred.index.get_level_values(0)
        mask = (dates >= pd.Timestamp(test_start)) & (dates <= pd.Timestamp(test_end))
        pred = pred[mask]

    elapsed_train = time.time() - t0
    print(f"  训练+预测完成 ({elapsed_train:.1f}s), 预测样本: {len(pred)}")

    # 回测
    strategy_config = {
        "class": "TopkDropoutStrategy",
        "module_path": "qlib.contrib.strategy",
        "kwargs": {"signal": pred, "topk": topk, "n_drop": n_drop},
    }
    backtest_config = {
        "start_time": test_start, "end_time": test_end,
        "account": 100_000_000, "benchmark": "SH000300",
        "exchange_kwargs": {
            "freq": "day", "limit_threshold": 0.095, "deal_price": "close",  # 2026-08-30 由 open 改为 close (对齐生产 signal_config)
            "open_cost": 0.0005, "close_cost": 0.0015, "min_cost": 5, "trade_unit": 100,
        },
    }
    strategy = init_instance_by_config(strategy_config)
    report, positions = backtest_daily(strategy=strategy, executor=None, **backtest_config)

    result = {"preset": preset_name, "model": model_name, "train_time": elapsed_train}
    if "return" in report.columns:
        returns = report["return"]
        total_ret = (1 + returns).prod() - 1
        result["total_return"] = float(total_ret)
        n_days = len(returns)
        result["annual_return"] = float((1 + total_ret) ** (252 / max(n_days, 1)) - 1)
        cumulative = (1 + returns).cumprod()
        drawdown = (cumulative - cumulative.cummax()) / cumulative.cummax()
        result["max_drawdown"] = float(drawdown.min())
        daily_std = float(returns.std())
        result["sharpe"] = float(returns.mean() / daily_std * (252 ** 0.5)) if daily_std > 0 else 0.0
    if "bench" in report.columns:
        result["bench_return"] = float((1 + report["bench"]).prod() - 1)
        result["excess_return"] = result.get("total_return", 0) - result["bench_return"]

    # 保存
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result_file = RESULTS_DIR / f"{preset_name}_{model_name}.json"
    with open(result_file, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"  总收益: {result.get('total_return', 0):.2%}")
    print(f"  年化: {result.get('annual_return', 0):.2%}")
    print(f"  Sharpe: {result.get('sharpe', 0):.3f}")
    print(f"  最大回撤: {result.get('max_drawdown', 0):.2%}")

    return result


def _build_model(model_name: str):
    """构建模型实例"""
    if model_name == "lightgbm":
        from qlib.contrib.model.gbdt import LGBModel
        return LGBModel(
            loss="mse", learning_rate=0.01, num_leaves=64,
            num_boost_round=500, early_stopping_rounds=80,
            feature_fraction=0.75, bagging_fraction=0.75, bagging_freq=5,
            lambda_l1=0.1, lambda_l2=0.1, min_data_in_leaf=80,
        )
    elif model_name == "xgboost":
        from qlib.contrib.model.xgboost import XGBModel
        return XGBModel(
            learning_rate=0.01, max_depth=6,
            n_estimators=500, early_stopping_rounds=80,
            reg_alpha=0.1, reg_lambda=0.1,
            subsample=0.75, colsample_bytree=0.75,
        )
    elif model_name == "catboost":
        from qlib.contrib.model.catboost_model import CatBoostModel
        return CatBoostModel(
            loss="RMSE", learning_rate=0.01, depth=6,
            iterations=500, l2_leaf_reg=3.0, subsample=0.75,
        )
    elif model_name == "double_ensemble":
        from qlib.contrib.model.double_ensemble import DEnsembleModel
        return DEnsembleModel(
            base_model="gbm", loss="mse",
            num_models=6, num_boost_round=150,
            enable_sr=True, enable_fs=True, decay=0.5,
        )
    elif model_name == "ridge":
        # 使用 sklearn Ridge (同 benchmark)
        raise NotImplementedError("Ridge 需要特殊处理，参考 benchmark_models.py")
    else:
        raise ValueError(f"未知模型: {model_name}")


def run_grid(presets: list[str] = None, models: list[str] = None, **kwargs) -> pd.DataFrame:
    """网格实验: presets × models

    Args:
        presets: 因子预设列表
        models: 模型列表
        **kwargs: 传给 run_experiment 的参数

    Returns:
        结果汇总 DataFrame
    """
    if presets is None:
        presets = ["alpha158", "alpha158_ext"]
    if models is None:
        models = ["lightgbm", "xgboost"]

    all_results = []
    for preset in presets:
        for model in models:
            try:
                result = run_experiment(preset, model, **kwargs)
                all_results.append(result)
            except Exception as e:
                print(f"  失败 {preset}×{model}: {e}")
                traceback.print_exc()
                all_results.append({
                    "preset": preset, "model": model, "error": str(e)
                })

    df = pd.DataFrame(all_results)
    return df
