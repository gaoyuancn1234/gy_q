"""全量 Rolling 回测对比 — alpha158_val + new_factors vs M01 baseline

复用 run_rolling_benchmark 的 rolling 逻辑，baseline 从缓存读。
"""
import gc
import json
import time
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_DIR / "factor_lab" / "results" / "rolling"

# 回测参数从 run_rolling_benchmark 导入 (保持一致)
ROLLING_CONFIG_NAME = "D_expand_3v_3r"
ROLLING_CONFIG = {
    "train_years": 4,
    "valid_months": 3,
    "retrain_months": 3,
    "expanding": True,
}


def _load_baseline() -> dict | None:
    """从缓存读取 M01 baseline 结果"""
    baseline_file = RESULTS_DIR / "D_expand_3v_3r_alpha158_val_LightGBM.json"
    if baseline_file.exists():
        with open(baseline_file) as f:
            data = json.load(f)
        overall = data.get("overall", {})
        if overall and overall.get("sharpe"):
            return overall
    return None


def run_comparison(new_factors: list[tuple[str, str]],
                   timeout: int = 1800) -> dict:
    """alpha158_val + new_factors vs M01 baseline 全量 rolling 对比

    Args:
        new_factors: [(name, expr), ...]
        timeout: 回测总时间限制 (秒)

    Returns:
        {baseline, candidate, improvement}
    """
    from factor_lab.run_rolling_benchmark import (
        generate_rolling_windows, build_model, run_backtest,
    )
    from factor_lab.factors.custom_handler import build_handler_from_exprs
    from factor_lab.factors.presets import FACTOR_PRESETS

    # 1. 加载 baseline
    baseline = _load_baseline()
    if not baseline:
        return {"error": "baseline 缓存不存在或 overall 为空，需运行: python -m factor_lab.run_rolling_benchmark --configs D_expand_3v_3r --presets alpha158_val --models LightGBM --force"}

    # 2. 构建扩展因子列表: alpha158_val 的 extra + new_factors
    preset = FACTOR_PRESETS["alpha158_val"]
    extra = preset["extra_factors"]
    if callable(extra):
        extra = extra()
    extended_factors = extra + list(new_factors)

    # 3. 生成 rolling windows
    windows = generate_rolling_windows(
        ROLLING_CONFIG_NAME, ROLLING_CONFIG, TEST_START, TEST_END,
    )

    print(f"  [backtest] Rolling 回测: {len(windows)} 个窗口, "
          f"{len(extended_factors)} 扩展因子 (含 {len(new_factors)} 新因子)")

    # 4. 逐窗口训练预测
    all_preds = []
    t_total = time.time()

    for w in windows:
        wnum = w["window_num"]
        if time.time() - t_total > timeout:
            print(f"  [backtest] 超时 ({timeout}s), 在 window {wnum} 停止")
            break

        t0 = time.time()

        handler, _ = build_handler_from_exprs(
            factor_exprs=extended_factors,
            start_time=w["train_start"],
            end_time=w["pred_end"],
            fit_start_time=w["train_start"],
            fit_end_time=w["train_end"],
            instruments="csi300",
            include_alpha158=True,
        )

        from qlib.data.dataset import DatasetH
        dataset = DatasetH(handler=handler, segments={
            "train": (w["train_start"], w["train_end"]),
            "valid": (w["valid_start"], w["valid_end"]),
            "test": (w["pred_start"], w["pred_end"]),
        })

        model, fit_kwargs = build_model("LightGBM")
        model.fit(dataset, **fit_kwargs)

        pred = model.predict(dataset)
        if isinstance(pred.index, pd.MultiIndex):
            dates = pred.index.get_level_values(0)
            mask = (dates >= pd.Timestamp(w["pred_start"])) & \
                   (dates <= pd.Timestamp(w["pred_end"]))
            pred = pred[mask]

        all_preds.append(pred)
        elapsed = time.time() - t0
        best_iter = getattr(model.model, "best_iteration", None)
        print(f"    Window {wnum}: best_iter={best_iter}, "
              f"samples={len(pred)}, time={elapsed:.1f}s")

        del handler, dataset, model, pred
        gc.collect()

    if not all_preds:
        return {"error": "无预测结果"}

    combined = pd.concat(all_preds)
    if combined.index.duplicated().any():
        combined = combined[~combined.index.duplicated(keep="last")]

    # 5. 回测
    print(f"  [backtest] 回测 {len(combined)} 条预测...")
    candidate = run_backtest(combined)

    total_time = time.time() - t_total

    # 6. 对比
    sharpe_delta = candidate.get("sharpe", 0) - baseline.get("sharpe", 0)
    is_better = sharpe_delta >= 0.05

    result = {
        "baseline": {
            "sharpe": baseline.get("sharpe", 0),
            "mdd": baseline.get("max_drawdown", 0),
            "return": baseline.get("total_return", 0),
        },
        "candidate": {
            "sharpe": candidate.get("sharpe", 0),
            "mdd": candidate.get("max_drawdown", 0),
            "return": candidate.get("total_return", 0),
        },
        "improvement": {
            "sharpe_delta": round(sharpe_delta, 4),
            "is_better": is_better,
        },
        "total_time": round(total_time, 1),
        "new_factors_count": len(new_factors),
    }

    print(f"  [backtest] Baseline Sharpe={baseline.get('sharpe', 0):.3f} | "
          f"Candidate Sharpe={candidate.get('sharpe', 0):.3f} | "
          f"Delta={sharpe_delta:+.4f} | "
          f"{'BETTER' if is_better else 'no improvement'}")

    return result
