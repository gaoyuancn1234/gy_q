#!/usr/bin/env python3
"""QuantaAlpha 核心方法复现 — v4 论文规模实验 (Experiment 013)

严格按论文方法复现核心挖掘流程 + 5 步循环:
- Phase A: 多样化初始规划 (5 步: Propose -> Construct(regen) -> Calculate -> Backtest -> Feedback)
- Phase B: 轨迹级自我演化 (Mutation + Crossover, MAX_ROUNDS 轮)
- Phase C: 最终评估 (Alpha158 + 挖掘因子 + 纯挖掘因子 -> LightGBM -> TopK=50 回测)

用法:
  # 论文规模运行 (10方向, 5轮)
  python -m factor_lab.run_quanta_alpha

  # 快速测试 (1方向, 2轮)
  python -m factor_lab.run_quanta_alpha --directions 1 --max-rounds 2

  # Dry-run (只调 Claude, 不跑 Qlib)
  python -m factor_lab.run_quanta_alpha --dry-run

  # 跳过每轮组合回测 (加速)
  python -m factor_lab.run_quanta_alpha --no-backtest

  # 从中断恢复
  python -m factor_lab.run_quanta_alpha --resume

  # 只跑最终评估 (已有因子池)
  python -m factor_lab.run_quanta_alpha --eval-only

  # 只看报告
  python -m factor_lab.run_quanta_alpha --report-only
"""
import gc
import json
import sys
import time
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from factor_lab.utils import json_default as _json_default

from factor_lab.quanta.config import (
    N_DIRECTIONS, MAX_ROUNDS, N_CANDIDATES, CROSSOVER_N, CROSSOVER_SIZE,
    TOPK, N_DROP, BACKTEST_ENABLED, HISTORY_LIMIT, MAX_REGEN_ATTEMPTS,
    TRAIN_START, TRAIN_END, VALID_START, VALID_END, TEST_START, TEST_END,
)
from factor_lab.quanta.trajectory import Trajectory, TrajectoryPool
from factor_lab.quanta.factor_pool import FactorPool
from factor_lab.quanta import idea_agent, factor_agent, eval_agent, evolution

RESULTS_DIR = PROJECT_DIR / "factor_lab" / "results" / "quanta_alpha"


# ============ Phase A: 多样化初始规划 (v3: 5 步循环) ============

def run_phase_a(n_directions: int, pool: TrajectoryPool,
                factor_pool: FactorPool, dry_run: bool = False,
                no_backtest: bool = False) -> list[Trajectory]:
    """Phase A: 5 步循环 — Propose -> Construct(regen) -> Calculate -> Backtest -> Feedback"""
    print(f"\n{'='*70}")
    print(f"Phase A: 多样化初始规划 ({n_directions} 个方向, v3 5步循环)")
    print(f"{'='*70}")

    # Step 1: 生成多样化假说 (1 次 Claude 调用)
    print(f"\n[A.1] 生成 {n_directions} 个多样化假说...")
    t0 = time.time()
    hypotheses = idea_agent.generate_diverse_hypotheses(n=n_directions)
    print(f"  生成了 {len(hypotheses)} 个假说 ({time.time()-t0:.1f}s)")

    if not hypotheses:
        print("  [错误] 未能生成任何假说, 终止 Phase A")
        return []

    trajectories = []

    # 对每个假说执行 5 步循环
    for i, hyp in enumerate(hypotheses):
        print(f"\n[A.2.{i+1}/{len(hypotheses)}] 方向: {hyp.get('direction', '?')} | "
              f"机制: {hyp.get('mechanism', '?')}")
        print(f"  假说: {hyp.get('hypothesis', '')[:100]}...")

        direction_trace = pool.get_direction_trace(i)

        traj = Trajectory(
            direction_id=i,
            iteration=0,
            phase="init",
            hypothesis=hyp.get('hypothesis', ''),
            direction=hyp.get('direction', ''),
            mechanism=hyp.get('mechanism', ''),
        )
        traj.claude_calls = 1  # generate_diverse_hypotheses 算一次

        # --- Step 2: Construct (with regen) ---
        print(f"  [Step 2] 构造 {N_CANDIDATES} 个候选因子 (with regen)...")
        t1 = time.time()
        trace_text = direction_trace.render_for_prompt()
        factor_list_text = direction_trace.render_factor_list_for_prompt()

        candidates, regen_count = factor_agent.construct_factors_with_regen(
            hypothesis=hyp,
            n=N_CANDIDATES,
            trace_text=trace_text,
            factor_list_text=factor_list_text,
        )
        traj.claude_calls += 1 + regen_count
        traj.regen_attempts = regen_count
        print(f"  获得 {len(candidates)} 个候选 (regen={regen_count}, {time.time()-t1:.1f}s)")

        if not candidates:
            traj.failure_step = 1
            traj.error_msg = f"再生循环 {regen_count} 次后仍无候选"
            _finalize_phase_a_traj(traj, pool, factor_pool, direction_trace, no_backtest)
            trajectories.append(traj)
            continue

        traj.factor_candidates = candidates

        if dry_run:
            for c in candidates:
                print(f"    {c.get('name', '?')}: {c.get('expr', '')[:80]}")
            pool.add(traj)
            trajectories.append(traj)
            continue

        # --- Step 3: Calculate (validate + evaluate) ---
        print(f"  [Step 3] Validate + Evaluate")
        eval_agent.run_validation_pipeline(traj, candidates)

        # 一致性验证 (仅对最佳因子)
        if traj.best_factor and traj.hypothesis:
            print(f"  一致性验证...")
            ok, feedback = factor_agent.verify_consistency(
                traj.hypothesis,
                traj.best_factor.get('description', ''),
                traj.best_factor['expr'],
            )
            traj.consistency_ok = ok
            traj.claude_calls += 1
            if not ok:
                print(f"    不一致: {feedback}")

        # --- Step 4: Backtest ---
        if not no_backtest and BACKTEST_ENABLED and traj.best_factor and traj.failure_step == -1:
            new_factors = [(traj.best_factor['name'], traj.best_factor['expr'])]
            sota_factors = evolution._get_sota_factors(direction_trace, factor_pool=factor_pool)
            print(f"  [Step 4] Backtest (combined, sota={len(sota_factors)} factors)")
            bt_metrics = eval_agent.run_combined_backtest(new_factors, sota_factors)
            traj.backtest_metrics = bt_metrics
        else:
            print(f"  [Step 4] Backtest (skipped)")

        # --- Step 5: Feedback + finalize ---
        _finalize_phase_a_traj(traj, pool, factor_pool, direction_trace, no_backtest)
        trajectories.append(traj)
        _print_traj_summary(traj)

    # 保存中间结果
    pool.save()
    factor_pool.save()

    print(f"\n[Phase A 完成] 轨迹: {pool.size}, 成功: {len(pool.get_successful())}, "
          f"因子池: {factor_pool.size}")

    return trajectories


def _finalize_phase_a_traj(traj: Trajectory, pool: TrajectoryPool,
                           factor_pool: FactorPool,
                           direction_trace, no_backtest: bool):
    """Phase A 轨迹的 Step 5: Feedback + 因子池入池"""
    # 复用 evolution 的标准 finalize (compute_reward + LLM feedback + trace append)
    evolution._finalize_trace(traj, pool, direction_trace, no_backtest)

    # Phase A 额外步骤: 因子池入池
    if traj.best_factor and traj.failure_step == -1:
        admitted, reason = factor_pool.try_admit(
            name=traj.best_factor['name'],
            expr=traj.best_factor['expr'],
            rank_ic=traj.rank_ic,
            icir=traj.icir,
            hypothesis=traj.hypothesis,
            direction=traj.direction,
            source_traj_id=traj.id,
            iteration=0,
        )
        status = "ADMITTED" if admitted else "REJECTED"
        print(f"  因子池: {status} — {reason}")


# ============ Phase B: 自我演化 ============

def run_phase_b(max_rounds: int, pool: TrajectoryPool,
                factor_pool: FactorPool, dry_run: bool = False,
                no_backtest: bool = False):
    """Phase B: 轨迹级自我演化 — v3 5 步循环"""
    print(f"\n{'='*70}")
    print(f"Phase B: 自我演化 ({max_rounds} 轮, crossover_n={CROSSOVER_N}, v3 5步循环)")
    print(f"  Flow: Original -> Mutation -> Crossover -> Mutation -> Crossover -> ...")
    print(f"{'='*70}")

    for rnd in range(1, max_rounds):
        is_mutation = (rnd % 2 == 1)
        phase_name = "Mutation" if is_mutation else "Crossover"

        print(f"\n{'─'*60}")
        print(f"Round {rnd}/{max_rounds-1}: {phase_name}")
        print(f"{'─'*60}")
        rnd_start = time.time()

        if is_mutation:
            targets = evolution.get_mutation_targets(pool, current_round=rnd)
            print(f"  Mutation targets: {len(targets)} (from prev round)")

            if not targets:
                print(f"  [跳过] 没有可 mutate 的轨迹")
                continue

            for j, parent in enumerate(targets):
                print(f"\n  [M.{j+1}/{len(targets)}] 父轨迹 {parent.id} "
                      f"({parent.phase}, reward={parent.reward:.4f})")
                new_traj = evolution.mutate_trajectory(
                    parent, iteration=rnd, pool=pool,
                    dry_run=dry_run, no_backtest=no_backtest,
                    factor_pool=factor_pool,
                )
                _try_admit_to_pool(new_traj, factor_pool, rnd)
                _print_traj_summary(new_traj, indent=4)

        else:
            candidates = evolution.get_crossover_candidates(pool, current_round=rnd)
            print(f"  Crossover candidates: {len(candidates)}")

            if len(candidates) < CROSSOVER_SIZE:
                print(f"  [跳过] 候选不足 ({len(candidates)} < {CROSSOVER_SIZE})")
                continue

            groups = evolution.select_crossover_groups(
                candidates, n_groups=CROSSOVER_N, group_size=CROSSOVER_SIZE,
            )
            print(f"  Selected {len(groups)} crossover groups")

            for gi, group in enumerate(groups):
                parent_ids = [p.id for p in group]
                print(f"\n  [X.{gi+1}/{len(groups)}] 父轨迹: {parent_ids}")
                for p in group:
                    print(f"      {p.id}: reward={p.reward:.4f} dir={p.direction[:30]}")

                new_traj = evolution.crossover_trajectories(
                    group, iteration=rnd, pool=pool,
                    group_idx=gi, dry_run=dry_run, no_backtest=no_backtest,
                    factor_pool=factor_pool,
                )
                _try_admit_to_pool(new_traj, factor_pool, rnd)
                _print_traj_summary(new_traj, indent=4)

        # 保存中间结果
        pool.save()
        factor_pool.save()

        elapsed = time.time() - rnd_start
        stats = pool.stats()
        print(f"\n  [Round {rnd} 完成] {elapsed:.1f}s | "
              f"轨迹={stats['total']} 成功={stats['successful']} "
              f"因子池={factor_pool.size}/{factor_pool.capacity}")

    print(f"\n[Phase B 完成] 轨迹: {pool.size}, 因子池: {factor_pool.size}")


def _try_admit_to_pool(traj: 'Trajectory', factor_pool: FactorPool,
                       iteration: int):
    """尝试将轨迹的最佳因子加入池"""
    if traj.best_factor and traj.failure_step == -1:
        admitted, reason = factor_pool.try_admit(
            name=traj.best_factor['name'],
            expr=traj.best_factor['expr'],
            rank_ic=traj.rank_ic,
            icir=traj.icir,
            hypothesis=traj.hypothesis,
            direction=traj.direction,
            source_traj_id=traj.id,
            iteration=iteration,
        )
        status = "ADMITTED" if admitted else "REJECTED"
        print(f"    因子池: {status} — {reason}")


# ============ Phase C: 最终评估 ============

def run_phase_c(factor_pool: FactorPool, pool: TrajectoryPool):
    """Phase C: Alpha158 + 挖掘因子 -> LightGBM -> 回测"""
    print(f"\n{'='*70}")
    print(f"Phase C: 最终评估")
    print(f"{'='*70}")

    mined_factors = factor_pool.get_exprs()
    print(f"\n挖掘因子池: {len(mined_factors)} 个因子")
    for name, expr in mined_factors:
        print(f"  {name}: {expr[:80]}")

    results = {}

    # 1. 纯 Alpha158 baseline
    print(f"\n[C.1] Alpha158 baseline (LightGBM)...")
    pred_base = _train_and_predict_lightgbm(preset='alpha158', extra_factors=[])
    if pred_base is not None:
        metrics_base = _compute_full_metrics(pred_base)
        results['alpha158_baseline'] = metrics_base
        _print_metrics("Alpha158 baseline", metrics_base)

    # 2. Alpha158 + 挖掘因子
    if mined_factors:
        print(f"\n[C.2] Alpha158 + {len(mined_factors)} 挖掘因子...")
        pred_mined = _train_and_predict_lightgbm(preset='alpha158', extra_factors=mined_factors)
        if pred_mined is not None:
            metrics_mined = _compute_full_metrics(pred_mined)
            results['alpha158_mined'] = metrics_mined
            _print_metrics("Alpha158 + mined", metrics_mined)

    # 3. alpha158_val + 挖掘因子
    if mined_factors:
        print(f"\n[C.3] alpha158_val + {len(mined_factors)} 挖掘因子...")
        pred_val = _train_and_predict_lightgbm(preset='alpha158_val', extra_factors=mined_factors)
        if pred_val is not None:
            metrics_val = _compute_full_metrics(pred_val)
            results['alpha158_val_mined'] = metrics_val
            _print_metrics("alpha158_val + mined", metrics_val)

    # 4. 消融: mutation-only vs crossover-only
    print(f"\n[C.4] 消融分析...")
    mutation_factors = _get_factors_by_phase(pool, factor_pool, "mutation")
    crossover_factors = _get_factors_by_phase(pool, factor_pool, "crossover")
    init_factors = _get_factors_by_phase(pool, factor_pool, "init")

    for label, factors in [
        ("init_only", init_factors),
        ("mutation_only", mutation_factors),
        ("crossover_only", crossover_factors),
    ]:
        if factors:
            print(f"\n  [{label}] {len(factors)} 因子")
            pred = _train_and_predict_lightgbm(preset='alpha158', extra_factors=factors)
            if pred is not None:
                metrics = _compute_full_metrics(pred)
                results[label] = metrics
                _print_metrics(f"  {label}", metrics)

    # 5. 纯挖掘因子评估 (对齐论文: 不混入 Alpha158)
    if mined_factors:
        print(f"\n[C.5] 纯挖掘因子评估 ({len(mined_factors)} 因子, 无 Alpha158)...")
        pred_pure = _train_and_predict_lightgbm(
            preset='alpha158', extra_factors=mined_factors,
            include_alpha158=False,
        )
        if pred_pure is not None:
            metrics_pure = _compute_full_metrics(pred_pure)
            results['mined_only_pure'] = metrics_pure
            _print_metrics("mined_only_pure", metrics_pure)

    # 保存结果
    _save_results(results, factor_pool, pool)
    _print_comparison_table(results)

    return results


# ============ LightGBM 训练 ============

def _train_and_predict_lightgbm(preset: str, extra_factors: list[tuple[str, str]],
                                include_alpha158: bool = True):
    """训练 LightGBM 并返回测试集预测

    Args:
        include_alpha158: 是否包含 Alpha158 基础因子。False 时只用 extra_factors (纯挖掘因子评估)。
    """
    from factor_lab.run_paper_replication import (
        _get_valid_instruments, filter_test_predictions,
    )

    try:
        if extra_factors:
            from factor_lab.factors.custom_handler import build_handler_from_exprs
            from factor_lab.factors.presets import FACTOR_PRESETS

            preset_config = FACTOR_PRESETS.get(preset, FACTOR_PRESETS['alpha158'])

            if include_alpha158:
                extra_preset = preset_config.get('extra_factors', [])
                if callable(extra_preset):
                    extra_preset = extra_preset()
            else:
                extra_preset = []

            seen_names = set()
            all_extra = []
            for name, expr in list(extra_preset) + list(extra_factors):
                if name not in seen_names:
                    seen_names.add(name)
                    all_extra.append((name, expr))

            handler, n_feat = build_handler_from_exprs(
                factor_exprs=all_extra,
                start_time=TRAIN_START, end_time=TEST_END,
                fit_start_time=TRAIN_START, fit_end_time=TRAIN_END,
                instruments=_get_valid_instruments(),
                include_alpha158=include_alpha158,
            )
        else:
            from factor_lab.factors.presets import build_handler
            handler = build_handler(
                preset,
                start_time=TRAIN_START, end_time=TEST_END,
                fit_start_time=TRAIN_START, fit_end_time=TRAIN_END,
            )

        from qlib.data.dataset import DatasetH
        dataset = DatasetH(handler=handler, segments={
            "train": (TRAIN_START, TRAIN_END),
            "valid": (VALID_START, VALID_END),
            "test": (TEST_START, TEST_END),
        })

        from qlib.contrib.model.gbdt import LGBModel
        model = LGBModel(
            loss="mse", learning_rate=0.01, num_leaves=64,
            num_boost_round=500, early_stopping_rounds=80,
            feature_fraction=0.75, bagging_fraction=0.75, bagging_freq=5,
            lambda_l1=0.1, lambda_l2=0.1, min_data_in_leaf=80,
        )
        model.fit(dataset)
        pred = filter_test_predictions(model.predict(dataset))

        del handler, dataset, model
        gc.collect()

        return pred
    except Exception as e:
        print(f"  训练失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def _compute_full_metrics(pred) -> dict:
    """计算因子指标 + 回测指标"""
    from factor_lab.run_paper_replication import compute_factor_metrics, run_backtest
    factor_metrics = compute_factor_metrics(pred)
    bt_result = run_backtest(pred)
    return {**factor_metrics, **bt_result}


def _get_factors_by_phase(pool: TrajectoryPool, factor_pool: FactorPool,
                          phase: str) -> list[tuple[str, str]]:
    """获取特定 phase 产生的因子"""
    pool_factors = factor_pool.get_all()
    traj_ids = {t.id for t in pool.get_by_phase(phase)}
    return [(f.name, f.expr) for f in pool_factors if f.source_traj_id in traj_ids]


# ============ 输出 ============

def _print_traj_summary(traj: Trajectory, indent: int = 2):
    pad = " " * indent
    status = "OK" if traj.failure_step == -1 else f"FAIL@step{traj.failure_step}"
    factor_name = traj.best_factor.get('name', 'N/A') if traj.best_factor else 'N/A'
    regen_str = f" regen={traj.regen_attempts}" if traj.regen_attempts else ""
    bt_str = ""
    if traj.backtest_metrics:
        bt_str = f" Sharpe={traj.backtest_metrics.get('sharpe', 0):.3f}"
    print(f"{pad}[{traj.id}] {status} | ICIR={traj.icir:.3f} reward={traj.reward:.4f}{bt_str} | "
          f"factor={factor_name} | calls={traj.claude_calls}{regen_str}")


def _print_metrics(label: str, m: dict):
    print(f"  {label}: IC={m.get('IC', 0):.4f} ICIR={m.get('ICIR', 0):.4f} "
          f"RankIC={m.get('RankIC', 0):.4f} ARR={m.get('ARR', 0):.2f}% "
          f"MDD={m.get('MDD', 0):.2f}% Sharpe={m.get('sharpe', 0):.3f}")


def _print_comparison_table(results: dict):
    """打印对比表"""
    print(f"\n{'='*100}")
    print(f"  QuantaAlpha 最终结果 (Experiment 013 v4)")
    print(f"  Train: {TRAIN_START}~{TRAIN_END} | Valid: {VALID_START}~{VALID_END} | Test: {TEST_START}~{TEST_END}")
    print(f"  Strategy: TopK={TOPK}, N_drop={N_DROP}")
    print(f"  Config: MAX_ROUNDS={MAX_ROUNDS}, N_CANDIDATES={N_CANDIDATES}, CROSSOVER_N={CROSSOVER_N}")
    print(f"  Trace: HISTORY_LIMIT={HISTORY_LIMIT}, MAX_REGEN={MAX_REGEN_ATTEMPTS}, BACKTEST={BACKTEST_ENABLED}")
    print(f"{'='*100}")

    header = (f"  {'Config':<25} {'IC':>7} {'ICIR':>7} {'RkIC':>7} {'RkICIR':>7} "
              f"{'ARR%':>7} {'MDD%':>7} {'Sharpe':>7}")
    print(f"\n{header}")
    print(f"  {'-'*90}")

    for name, m in results.items():
        if 'IC' in m:
            print(f"  {name:<25} {m.get('IC', 0):>6.4f} {m.get('ICIR', 0):>6.4f} "
                  f"{m.get('RankIC', 0):>6.4f} {m.get('RankICIR', 0):>6.4f} "
                  f"{m.get('ARR', 0):>6.2f} {m.get('MDD', 0):>6.2f} "
                  f"{m.get('sharpe', 0):>6.3f}")

    # 加载 Exp 012 结果作为参考
    exp012_path = PROJECT_DIR / "factor_lab" / "results" / "paper_replication" / "results.json"
    if exp012_path.exists():
        with open(exp012_path) as f:
            exp012 = json.load(f)
        print(f"\n  --- Exp 012 参考 ---")
        for name in ['LightGBM', 'alpha158_val_LGB', 'rolling_ours']:
            if name in exp012 and 'IC' in exp012[name]:
                m = exp012[name]
                print(f"  {name:<25} {m.get('IC', 0):>6.4f} {m.get('ICIR', 0):>6.4f} "
                      f"{m.get('RankIC', 0):>6.4f} {m.get('RankICIR', 0):>6.4f} "
                      f"{m.get('ARR', 0):>6.2f} {m.get('MDD', 0):>6.2f} "
                      f"{m.get('sharpe', 0):>6.3f}")

    print(f"\n{'='*100}")


def _save_results(results: dict, factor_pool: FactorPool, pool: TrajectoryPool):
    """保存所有结果"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 主结果
    with open(RESULTS_DIR / "results.json", 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=_json_default)

    # 因子池摘要
    pool_summary = {
        "factors": [
            {"name": fp.name, "expr": fp.expr,
             "rank_ic": fp.rank_ic, "icir": fp.icir,
             "direction": fp.direction, "iteration": fp.iteration}
            for fp in factor_pool.get_all()
        ],
        "stats": factor_pool.stats(),
    }
    with open(RESULTS_DIR / "factor_pool_summary.json", 'w') as f:
        json.dump(pool_summary, f, indent=2, ensure_ascii=False, default=_json_default)

    # 轨迹统计
    with open(RESULTS_DIR / "trajectory_stats.json", 'w') as f:
        json.dump(pool.stats(), f, indent=2, ensure_ascii=False, default=_json_default)


# ============ 报告 ============

def print_report():
    """只打印已有结果"""
    results_path = RESULTS_DIR / "results.json"
    if not results_path.exists():
        print("没有找到结果文件")
        return

    with open(results_path) as f:
        results = json.load(f)
    _print_comparison_table(results)

    pool_path = RESULTS_DIR / "factor_pool_summary.json"
    if pool_path.exists():
        with open(pool_path) as f:
            pool_summary = json.load(f)
        print(f"\n因子池 ({len(pool_summary.get('factors', []))} 个因子):")
        for fp in pool_summary.get('factors', []):
            print(f"  {fp['name']:<30} ICIR={fp.get('icir', 0):.3f} "
                  f"RankIC={fp.get('rank_ic', 0):.4f} dir={fp.get('direction', '')}")

    stats_path = RESULTS_DIR / "trajectory_stats.json"
    if stats_path.exists():
        with open(stats_path) as f:
            stats = json.load(f)
        print(f"\n轨迹统计: {json.dumps(stats, indent=2)}")


# ============ Qlib 初始化 ============

def init_qlib():
    import multiprocessing
    multiprocessing.set_start_method('fork', force=True)
    import qlib
    from qlib.constant import REG_CN
    qlib.init(provider_uri='~/.qlib/qlib_data/cn_data_bs', region=REG_CN)


# ============ CLI ============

def main():
    parser = argparse.ArgumentParser(description="QuantaAlpha 多智能体因子挖掘 (Exp 013 v4)")
    parser.add_argument('--directions', type=int, default=N_DIRECTIONS,
                        help=f"Phase A 方向数 (默认 {N_DIRECTIONS})")
    parser.add_argument('--max-rounds', type=int, default=MAX_ROUNDS,
                        help=f"Phase B 演化轮数 (默认 {MAX_ROUNDS})")
    parser.add_argument('--dry-run', action='store_true',
                        help="只调 Claude, 不跑 Qlib")
    parser.add_argument('--no-backtest', action='store_true',
                        help="跳过每轮组合回测 (加速)")
    parser.add_argument('--resume', action='store_true',
                        help="从中断恢复")
    parser.add_argument('--eval-only', action='store_true',
                        help="只跑 Phase C (已有因子池)")
    parser.add_argument('--report-only', action='store_true',
                        help="只打印报告")
    parser.add_argument('--skip-phase-c', action='store_true',
                        help="跳过 Phase C")
    args = parser.parse_args()

    if args.report_only:
        print_report()
        return

    # 初始化
    pool = TrajectoryPool(RESULTS_DIR)
    factor_pool = FactorPool(RESULTS_DIR)

    if args.resume or args.eval_only:
        print("加载已有状态...")
        pool.load()
        factor_pool.load()
        print(f"  轨迹: {pool.size}, 因子池: {factor_pool.size}")
        # 验证 DirectionTrace 恢复
        for did, dt in pool._direction_traces.items():
            print(f"  DirectionTrace[{did}]: {len(dt.entries)} entries, "
                  f"SOTA={'yes' if dt.get_sota() else 'no'}")

    if not args.dry_run:
        init_qlib()

    t_total = time.time()

    print(f"\n{'#'*70}")
    print(f"  QuantaAlpha 多智能体因子挖掘 (Experiment 013 v4)")
    print(f"  方向: {args.directions} | 最大轮数: {args.max_rounds} | "
          f"crossover_n: {CROSSOVER_N} | dry-run: {args.dry_run}")
    print(f"  v3 新增: Trace={HISTORY_LIMIT} | Regen={MAX_REGEN_ATTEMPTS} | "
          f"Backtest={'ON' if BACKTEST_ENABLED and not args.no_backtest else 'OFF'}")
    print(f"  Flow: Original({args.directions}) -> [Mutation -> Crossover] x "
          f"{(args.max_rounds-1)//2}")
    print(f"{'#'*70}")

    if not args.eval_only:
        # Phase A (= Round 0: Original, v3 5步循环)
        run_phase_a(args.directions, pool, factor_pool,
                    dry_run=args.dry_run, no_backtest=args.no_backtest)

        # Phase B (Round 1~max_rounds: Mutation <-> Crossover, v3 5步循环)
        if args.max_rounds > 1:
            run_phase_b(args.max_rounds, pool, factor_pool,
                        dry_run=args.dry_run, no_backtest=args.no_backtest)

    # Phase C
    if not args.dry_run and not args.skip_phase_c:
        run_phase_c(factor_pool, pool)
    elif args.dry_run:
        print("\n[Dry-run] 跳过 Phase C")
        pool.save()
        factor_pool.save()
        print(f"\n轨迹统计: {json.dumps(pool.stats(), indent=2)}")

    elapsed = time.time() - t_total
    print(f"\n总耗时: {elapsed/60:.1f} 分钟")


if __name__ == '__main__':
    main()
