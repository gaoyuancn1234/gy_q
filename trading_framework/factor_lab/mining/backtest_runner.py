"""全量 Rolling 回测对比 — alpha158_val + new_factors vs M01 baseline

复用 run_rolling_benchmark 的 rolling 逻辑，baseline 从缓存读。
"""
import gc
import json
import time
from pathlib import Path

import yaml
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_DIR / "factor_lab" / "results" / "rolling"


def _get_current_preset() -> str:
    """从 signal_config.yaml 读取当前生产 preset"""
    config_path = PROJECT_DIR / "config" / "signal_config.yaml"
    try:
        with open(config_path, encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        return cfg.get('preset', 'alpha158_val')
    except Exception:
        return 'alpha158_val'


def _detect_effective_baseline() -> str:
    """智能检测实际生产 baseline preset

    逻辑:
    1. 读 signal_config.yaml 的 preset (如 alpha158_val)
    2. 检查 mined.py 是否有因子 → 有则升级到 {preset}_mined
    3. 检查升级后的缓存文件是否存在 → 不存在则回退原 preset
    """
    base_preset = _get_current_preset()

    # 检查 mined.py 是否有因子
    try:
        from factor_lab.factors.mined import MINED_FACTORS
        has_mined = bool(MINED_FACTORS)
    except (ImportError, AttributeError):
        has_mined = False

    if has_mined:
        mined_preset = f"{base_preset}_mined"
        mined_file = RESULTS_DIR / f"D_expand_3v_3r_{mined_preset}_LightGBM.json"
        if mined_file.exists():
            return mined_preset

    return base_preset


# 回测参数 (与 run_rolling_benchmark 一致)
ROLLING_CONFIG_NAME = "D_expand_3v_3r"
ROLLING_CONFIG = {
    "train_years": 4,
    "valid_months": 3,
    "retrain_months": 3,
    "expanding": True,
}
TEST_START = '2024-01-01'
TEST_END = '2026-02-05'


def _load_baseline() -> dict | None:
    """从缓存读取 baseline 结果 (智能检测实际生产 preset)"""
    preset = _detect_effective_baseline()
    baseline_file = RESULTS_DIR / f"D_expand_3v_3r_{preset}_LightGBM.json"
    if baseline_file.exists():
        with open(baseline_file, encoding='utf-8') as f:
            data = json.load(f)
        overall = data.get("overall", {})
        if overall and overall.get("sharpe"):
            print(f"  [baseline] 使用 {preset} (Sharpe={overall['sharpe']:.3f})")
            return overall
    return None


def screen_by_importance(
    candidate_factors: list[tuple[str, str]],
    min_importance: float = 0,
) -> tuple[list[tuple[str, str]], dict[str, float]]:
    """用 LightGBM feature importance 筛选因子

    训练单窗口 LightGBM (alpha158_val + 全部候选因子)，
    提取 gain importance，只保留 importance > min_importance 的候选因子。

    Args:
        candidate_factors: [(name, expr), ...] 候选因子列表
        min_importance: importance 阈值 (默认 0, 即模型至少用过一次)

    Returns:
        (filtered_factors, importance_dict)
        - filtered_factors: 通过筛选的 [(name, expr), ...]
        - importance_dict: {factor_name: importance_value} (所有候选)
    """
    from factor_lab.run_rolling_benchmark import (
        generate_rolling_windows, build_model, fit_with_degradation_guard,
    )
    from factor_lab.factors.custom_handler import build_handler_from_exprs
    from factor_lab.factors.presets import FACTOR_PRESETS

    if not candidate_factors:
        return [], {}

    # 1. 构建因子列表: 使用当前生产 baseline 的 extra + candidate_factors (去重)
    effective_preset = _detect_effective_baseline()
    base_preset_cfg = FACTOR_PRESETS.get(effective_preset,
                                         FACTOR_PRESETS["alpha158_val"])
    extra = base_preset_cfg["extra_factors"]
    if callable(extra):
        extra = extra()
    extra_names = {name for name, _ in extra}
    # 去除与 baseline extras 名字重复的候选因子，避免 LightGBM 列名冲突
    deduped_candidates = [(n, e) for n, e in candidate_factors if n not in extra_names]
    if len(deduped_candidates) < len(candidate_factors):
        n_dup = len(candidate_factors) - len(deduped_candidates)
        print(f"  [importance] 去除 {n_dup} 个与 {effective_preset} 重名的候选因子")
    if not deduped_candidates:
        print(f"  [importance] 所有候选因子与 {effective_preset} 重名，跳过训练")
        return [], {}
    extended_factors = extra + deduped_candidates

    # 2. 取最后一个 rolling 窗口 (最新数据)
    windows = generate_rolling_windows(
        ROLLING_CONFIG_NAME, ROLLING_CONFIG, TEST_START, TEST_END,
    )
    if not windows:
        print("  [importance] 无可用 rolling 窗口")
        return list(candidate_factors), {n: -1 for n, _ in candidate_factors}

    w = windows[-1]  # 最后一个窗口 = 最新数据

    print(f"  [importance] 训练单窗口 LightGBM ({len(candidate_factors)} 候选因子 + alpha158)...")
    print(f"    窗口: train={w['train_start']}~{w['train_end']}, "
          f"valid={w['valid_start']}~{w['valid_end']}")

    # 3. 构建 handler + dataset + 训练
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

    # 这里需要模型对象本身(取 feature_importance)，不能用只返回预测的
    # train_window，故直接用兜底版 fit。单种子即可 —— importance 只用于粗筛。
    model, fit_kwargs = build_model("LightGBM")
    model, best_iter, used_fb = fit_with_degradation_guard(
        model, dataset, fit_kwargs, prev_best_iters=[], model_name="LightGBM")

    elapsed = time.time() - t0
    print(f"    训练完成: best_iter={best_iter}"
          f"{' [兜底]' if used_fb else ''}, time={elapsed:.1f}s")

    # 4. 提取 feature importance (gain)
    # 注意: Qlib LGBModel 用 x.values (numpy) 训练, lightgbm feature name 为 Column_N
    # 必须按位置索引而非名字查找
    raw_imp = model.model.feature_importance(importance_type='gain')

    from factor_lab.factors.custom_handler import get_alpha158_feature_count
    alpha_count = get_alpha158_feature_count()
    candidate_start = alpha_count + len(extra)  # alpha158 base + extras 之后

    # 5. 筛选候选因子的 importance (按位置映射)
    importance_dict: dict[str, float] = {}
    for i, (name, _) in enumerate(deduped_candidates):
        pos = candidate_start + i
        importance_dict[name] = float(raw_imp[pos]) if pos < len(raw_imp) else 0.0

    # 按 importance 降序排列打印
    sorted_imp = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    print(f"  [importance] 候选因子 importance (gain):")
    for name, imp_val in sorted_imp:
        marker = "+" if imp_val > min_importance else "-"
        print(f"    {marker} {name}: {imp_val:.1f}")

    # 6. 过滤: 只保留去重后候选中 importance > 阈值 的因子
    #    与 alpha158_val extras 重名的因子排除 (已在基线中，不需要重复加入)
    filtered = [
        (name, expr) for name, expr in deduped_candidates
        if importance_dict.get(name, 0) > min_importance
    ]

    n_baseline_dup = len(candidate_factors) - len(deduped_candidates)
    n_imp_zero = len(deduped_candidates) - len(filtered)
    print(f"  [importance] {len(filtered)}/{len(candidate_factors)} 因子通过筛选"
          f" (基线重名 {n_baseline_dup}, importance=0 过滤 {n_imp_zero})")

    del handler, dataset, model
    gc.collect()

    return filtered, importance_dict


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
        generate_rolling_windows, build_model, run_backtest, train_window,
    )
    from factor_lab.factors.custom_handler import build_handler_from_exprs
    from factor_lab.factors.presets import FACTOR_PRESETS

    # 1. 加载 baseline
    baseline = _load_baseline()
    if not baseline:
        return {"error": "baseline 缓存不存在或 overall 为空，需运行: python -m factor_lab.run_rolling_benchmark --configs D_expand_3v_3r --presets alpha158_val --models LightGBM --force"}

    # 2. 构建扩展因子列表: 使用与 baseline 一致的因子集 + new_factors (去重)
    effective_preset = _detect_effective_baseline()
    base_preset_cfg = FACTOR_PRESETS.get(effective_preset,
                                         FACTOR_PRESETS["alpha158_val"])
    extra = base_preset_cfg["extra_factors"]
    if callable(extra):
        extra = extra()
    extra_names = {name for name, _ in extra}
    deduped_new = [(n, e) for n, e in new_factors if n not in extra_names]
    extended_factors = extra + deduped_new

    # 3. 生成 rolling windows
    windows = generate_rolling_windows(
        ROLLING_CONFIG_NAME, ROLLING_CONFIG, TEST_START, TEST_END,
    )

    print(f"  [backtest] Rolling 回测: {len(windows)} 个窗口, "
          f"{len(extended_factors)} 扩展因子 (含 {len(new_factors)} 新因子)")

    # 4. 逐窗口训练预测
    all_preds = []
    window_details = []          # 供 train_window 推导兜底轮数
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

        # 与 run_rolling_benchmark / retrain_pipeline 共用同一训练路径
        # (退化兜底 + 多种子集成)。2026-09-03 修复:
        # 此前这里是裸的 model.fit()，既无兜底也无种子控制 ——
        # 而挖掘正是靠这个回测判断"因子有没有增量"。判据本身带 ±0.3 的随机
        # 波动时，选出来的因子很可能只是噪声(run_006/run_010 那 22 个即如此)。
        pred, best_iter, used_fb, n_models = train_window(
            dataset, "LightGBM",
            prev_best_iters=[d.get("best_iteration") for d in window_details],
        )
        if isinstance(pred.index, pd.MultiIndex):
            dates = pred.index.get_level_values(0)
            mask = (dates >= pd.Timestamp(w["pred_start"])) & \
                   (dates <= pd.Timestamp(w["pred_end"]))
            pred = pred[mask]

        all_preds.append(pred)
        elapsed = time.time() - t0
        window_details.append({"window_num": wnum, "best_iteration": best_iter,
                               "fallback_rounds": bool(used_fb)})
        print(f"    Window {wnum}: best_iter={best_iter}"
              f"{' [兜底]' if used_fb else ''}, models={n_models}, "
              f"samples={len(pred)}, time={elapsed:.1f}s")

        del handler, dataset, pred
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
        # 供 DSR/PBO 做多重检验校正 (只有汇总指标算不了)
        "candidate_returns": candidate.get("daily_returns", []),
        "baseline_returns": baseline.get("daily_returns", []),
    }

    print(f"  [backtest] Baseline Sharpe={baseline.get('sharpe', 0):.3f} | "
          f"Candidate Sharpe={candidate.get('sharpe', 0):.3f} | "
          f"Delta={sharpe_delta:+.4f} | "
          f"{'BETTER' if is_better else 'no improvement'}")

    return result
