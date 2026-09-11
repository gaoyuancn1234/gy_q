"""Agent Ae: 评估包装 + LLM 反馈 + 组合回测 (v3)

论文 Section 4.1 — Eval Agent 负责:
1. 对候选因子计算 IC/ICIR (复用 evaluate_candidates)
2. 选出最佳候选
3. 生成失败步骤诊断和反馈
4. [v3] LLM 驱动反馈 (对齐论文 factor_feedback_generation)
5. [v3] 组合回测 (对齐论文 QlibFactorRunner.develop)
"""
import gc
import json
import re
import subprocess
import time
from typing import Optional

from .config import (
    VALID_START, VALID_END, TRAIN_START, TRAIN_END, TEST_START, TEST_END,
    TOPK, N_DROP, BACKTEST_ENABLED, FEEDBACK_PROMPT,
    CLAUDE_CLI, CLAUDE_TIMEOUT, MAX_RETRY, RETRY_WAIT, get_claude_env,
    ROLLING_EVAL_LITE, ROLLING_EVAL_WINDOWS, ROLLING_EVAL_CONFIG,
    ROLLING_EVAL_TEST_START, ROLLING_EVAL_TEST_END,
    FAST_IC_THRESHOLD, FAST_IC_PERIOD, CORRELATION_THRESHOLD,
    REPLACE_IC_MIN, REPLACE_RATIO, BATCH_DEDUP_AST,
)
from .trajectory import Trajectory, HypothesisFeedback, TraceEntry, DirectionTrace


def _zero_metrics(**extra) -> dict:
    """构建零值指标结果 (用于评估失败或无数据时)"""
    result = {'ic': 0.0, 'icir': 0.0, 'rank_ic': 0.0, 'rank_icir': 0.0}
    result.update(extra)
    return result


def _extract_metrics(row) -> dict:
    """从 evaluate_candidates 的结果行中提取指标"""
    return {
        'ic': float(row.get('mean_IC', 0)),
        'icir': float(row.get('ICIR', 0)),
        'rank_ic': float(row.get('mean_IC', 0)),  # evaluate_candidates 返回的是 rank IC
        'rank_icir': float(row.get('ICIR', 0)),
        'is_promising': bool(row.get('is_promising', False)),
        'is_excellent': bool(row.get('is_excellent', False)),
    }


def evaluate_factor(name: str, expr: str,
                    start_time: str = VALID_START,
                    end_time: str = VALID_END) -> dict:
    """评估单个因子的 IC/ICIR

    使用 Valid 期 (2021) 评估, 与论文一致。

    Returns:
        {ic, icir, rank_ic, rank_icir, eval_time, ...}
    """
    from factor_lab.mining.evaluator import evaluate_candidates

    t0 = time.time()
    try:
        df = evaluate_candidates(
            [(name, expr)],
            start_time=start_time,
            end_time=end_time,
        )
        elapsed = time.time() - t0

        if df.empty:
            return _zero_metrics(nan_ratio=1.0, eval_time=elapsed, error='empty result')

        result = _extract_metrics(df.iloc[0])
        result['eval_time'] = elapsed
        return result
    except Exception as e:
        return _zero_metrics(eval_time=time.time() - t0, error=str(e))


def evaluate_candidates_batch(factors: list[tuple[str, str]],
                              start_time: str = VALID_START,
                              end_time: str = VALID_END) -> list[dict]:
    """批量评估因子"""
    from factor_lab.mining.evaluator import evaluate_candidates

    t0 = time.time()
    try:
        df = evaluate_candidates(factors, start_time=start_time, end_time=end_time)
        elapsed = time.time() - t0
        per_factor_time = elapsed / len(factors)

        results = []
        for name, expr in factors:
            row = df[df['factor'] == name]
            if row.empty:
                entry = _zero_metrics(name=name, expr=expr, eval_time=per_factor_time)
            else:
                entry = _extract_metrics(row.iloc[0])
                entry.update(name=name, expr=expr, eval_time=per_factor_time)
            results.append(entry)
        return results
    except Exception as e:
        print(f"  [eval_agent] 批量评估失败: {e}")
        return [_zero_metrics(name=n, expr=e_, error=str(e)) for n, e_ in factors]


def select_best_candidate(candidates: list[dict]) -> Optional[dict]:
    """从评估结果中选出 |ICIR| 最大的因子"""
    if not candidates:
        return None
    valid = [c for c in candidates if abs(c.get('icir', 0)) > 0]
    if not valid:
        return candidates[0]
    return max(valid, key=lambda c: abs(c.get('icir', 0)))


def diagnose_failure(traj: Trajectory) -> tuple[int, str]:
    """诊断轨迹的失败步骤

    Returns:
        (failure_step, feedback)
        - 0: idea 层面失败 (假说太模糊/不可操作)
        - 1: factor 层面失败 (表达式无效/复杂度超标)
        - 2: eval 层面失败 (IC 太弱/冗余)
    """
    # Step 1: 因子构造失败
    if not traj.constraint_ok:
        return 1, (
            f"因子表达式未通过约束检查。"
            f"尝试: 简化表达式结构、减少嵌套深度、使用更少的字段。"
        )

    if not traj.qlib_ok:
        return 1, (
            f"因子在 Qlib 中计算失败 (可能是 NaN 过多或 shape mismatch)。"
            f"尝试: 避免 Corr/Cov 跨源字段、加除零保护、检查字段是否存在。"
            f"错误: {traj.error_msg[:200]}"
        )

    if not traj.factor_candidates:
        return 1, "未能生成任何候选因子表达式。请简化假说使其更容易转化为数学公式。"

    if traj.consistency_ok is False:  # None=未检查, 仅 False 时报告
        return 1, (
            f"因子表达式与假说不一致。请重新审视假说的核心逻辑，"
            f"确保表达式真正捕捉了假说描述的市场现象。"
        )

    # Step 2: 评估失败
    if abs(traj.icir) < 0.1:
        return 2, (
            f"因子 IC 极弱 (ICIR={traj.icir:.3f})。"
            f"假说可能方向错误或信号已被市场定价。"
            f"尝试: 换时间窗口 (如 5->20 或 20->60)，或添加条件 (If 算子)。"
        )

    if abs(traj.icir) < 0.3:
        return 2, (
            f"因子有微弱信号但不够强 (ICIR={traj.icir:.3f})。"
            f"尝试: 加 Rank 截面标准化、组合多时间尺度、或用 regime 条件增强。"
        )

    if abs(traj.icir) < 0.5:
        return 2, (
            f"因子信号中等 (ICIR={traj.icir:.3f})，接近可用但未达标。"
            f"微调建议: 调整窗口参数 (+-5)、换 Mean<->Sum、或加入波动率条件。"
        )

    # Step 0: 如果一切通过但 reward 仍然低, 归因于 idea
    if traj.reward < 0.01:
        return 0, (
            f"因子通过所有检查但预测力极弱。"
            f"假说的核心直觉可能在 A 股市场不成立。"
            f"建议: 完全换一个投资逻辑。"
        )

    return -1, ""  # 未失败


def run_validation_pipeline(traj: Trajectory, candidates: list[dict]):
    """验证候选因子 + 选最佳 + 评估 (共享逻辑, 供 evolution.py 和 factor_miner.py 调用)"""
    from factor_lab.mining.validator import validate_expression, validate_with_qlib
    from . import factor_agent

    valid_candidates = []

    for cand in candidates:
        name = cand.get('name', '')
        expr = cand.get('expr', '')

        # 论文复杂度约束
        ok, reason = factor_agent.check_paper_complexity(expr)
        if not ok:
            print(f"    [{name}] 复杂度不通过: {reason}")
            continue
        traj.constraint_ok = True

        # 静态验证
        ok, reason = validate_expression(name, expr)
        if not ok:
            fixed_name = name.upper().replace(' ', '_').replace('-', '_')
            if not fixed_name.startswith('QA_'):
                fixed_name = 'QA_' + fixed_name
            ok, reason = validate_expression(fixed_name, expr)
            if not ok:
                print(f"    [{name}] 静态验证: {reason}")
                continue
            cand['name'] = fixed_name

        # 动态验证
        ok, reason = validate_with_qlib(cand['name'], expr)
        if not ok:
            print(f"    [{cand['name']}] Qlib验证: {reason}")
            traj.error_msg = reason
            continue
        traj.qlib_ok = True

        valid_candidates.append(cand)
        print(f"    [{cand['name']}] PASS")

    if not valid_candidates:
        traj.failure_step = 1
        traj.error_msg = traj.error_msg or "所有候选验证失败"
        return

    # 批量评估
    factors = [(c['name'], c['expr']) for c in valid_candidates]
    eval_results = evaluate_candidates_batch(factors)

    # 选最佳
    best = select_best_candidate(eval_results)
    if best:
        traj.best_factor = {
            'name': best['name'],
            'expr': best['expr'],
            'description': next(
                (c.get('description', '') for c in valid_candidates
                 if c['name'] == best['name']),
                '',
            ),
        }
        traj.ic = best.get('ic', 0)
        traj.icir = best.get('icir', 0)
        traj.rank_ic = best.get('rank_ic', 0)
        traj.rank_icir = best.get('rank_icir', 0)
        print(f"  最佳: {best['name']} (ICIR={best.get('icir', 0):.3f})")
    else:
        traj.failure_step = 2
        traj.error_msg = "所有候选 IC 为 0"


# ============ v3 新增: LLM 反馈 ============

def generate_llm_feedback(traj: Trajectory, sota_entry: Optional[TraceEntry],
                          direction_trace: DirectionTrace) -> HypothesisFeedback:
    """LLM 驱动反馈 (对齐论文 AlphaAgentQlibFactorHypothesisExperiment2Feedback)

    用 Claude CLI 分析当前结果 vs SOTA，生成结构化反馈。
    Fallback: 调用 diagnose_failure() 生成规则反馈。
    """
    if traj.failure_step >= 0 or not traj.best_factor:
        # 失败的轨迹用规则反馈
        failure_step, feedback_text = diagnose_failure(traj)
        return HypothesisFeedback(
            observations=f"因子构建/评估失败 (step={failure_step})",
            hypothesis_evaluation=feedback_text,
            new_hypothesis="需要重新设计假说或因子表达式",
            reasoning=feedback_text,
            decision=False,
        )

    # 构建 SOTA section
    if sota_entry:
        sota_section = (
            f"- 假说: {sota_entry.hypothesis[:100]}\n"
            f"- 因子: {sota_entry.factor_name} = {sota_entry.factor_expr[:100]}\n"
            f"- IC={sota_entry.ic:.4f}, ICIR={sota_entry.icir:.4f}, RankIC={sota_entry.rank_ic:.4f}"
        )
        if sota_entry.backtest_metrics:
            bt = sota_entry.backtest_metrics
            sota_section += (f"\n- 回测: Sharpe={bt.get('sharpe', 0):.3f}, "
                             f"ARR={bt.get('ARR', 0):.2f}%, MDD={bt.get('MDD', 0):.2f}%")
    else:
        sota_section = "(暂无 SOTA — 这是第一轮)"

    # 构建回测 section
    backtest_section = ""
    if traj.backtest_metrics:
        bt = traj.backtest_metrics
        backtest_section = (
            f"- 组合回测: Sharpe={bt.get('sharpe', 0):.3f}, "
            f"ARR={bt.get('ARR', 0):.2f}%, MDD={bt.get('MDD', 0):.2f}%"
        )

    prompt = FEEDBACK_PROMPT.format(
        hypothesis=traj.hypothesis[:200],
        factor_name=traj.best_factor.get('name', ''),
        factor_expr=traj.best_factor.get('expr', '')[:150],
        ic=traj.ic, icir=traj.icir, rank_ic=traj.rank_ic,
        backtest_section=backtest_section,
        sota_section=sota_section,
    )

    try:
        from pathlib import Path
        work_dir = Path(__file__).resolve().parent.parent.parent.parent

        cmd = [
            CLAUDE_CLI,
            '--print', '--dangerously-skip-permissions',
            '--output-format', 'text',
        ]
        # prompt 走 stdin (Windows .cmd 包装器按换行截断 argv, 见 llm_backend._invoke)
        result = subprocess.run(
            cmd, input=prompt, capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=CLAUDE_TIMEOUT, cwd=str(work_dir),
            env=get_claude_env(),
        )
        output = result.stdout.strip()
        if output:
            parsed = _parse_feedback_json(output)
            if parsed:
                return parsed

        print("  [eval_agent] LLM 反馈解析失败, fallback 到规则反馈")
    except Exception as e:
        print(f"  [eval_agent] LLM 反馈调用失败: {e}, fallback 到规则反馈")

    # Fallback: 规则反馈
    if sota_entry is None:
        return HypothesisFeedback(
            observations=f"第一轮, IC={traj.ic:.4f}, ICIR={traj.icir:.4f}",
            hypothesis_evaluation="作为首轮结果自动接受为 SOTA",
            new_hypothesis="后续轮次需要超越此基线",
            reasoning="首轮无对比基准，自动接受",
            decision=True,
        )

    # 与 SOTA 比较
    is_better = abs(traj.icir) > abs(sota_entry.icir)
    failure_step, feedback_text = diagnose_failure(traj)
    return HypothesisFeedback(
        observations=f"ICIR={traj.icir:.4f} vs SOTA={sota_entry.icir:.4f}",
        hypothesis_evaluation=feedback_text if failure_step >= 0 else "因子有效",
        new_hypothesis="继续探索正交方向" if is_better else "需要改进信号强度",
        reasoning=f"{'优于' if is_better else '不及'} SOTA",
        decision=is_better,
    )


def _parse_feedback_json(output: str) -> Optional[HypothesisFeedback]:
    """从 Claude 输出解析 HypothesisFeedback"""
    for attempt in [
        lambda: json.loads(output),
        lambda: json.loads(re.search(r'```(?:json)?\s*\n(\{[\s\S]*?\})\s*\n```', output).group(1)),
        lambda: json.loads(re.search(r'\{[\s\S]*\}', output).group()),
    ]:
        try:
            data = attempt()
            if isinstance(data, dict) and 'decision' in data:
                return HypothesisFeedback(
                    observations=str(data.get('observations', '')),
                    hypothesis_evaluation=str(data.get('hypothesis_evaluation', '')),
                    new_hypothesis=str(data.get('new_hypothesis', '')),
                    reasoning=str(data.get('reasoning', '')),
                    decision=bool(data.get('decision', False)),
                )
        except (json.JSONDecodeError, AttributeError, TypeError):
            continue
    return None


# ============ FactorMiner: 多阶段评估管道 ============

def run_multistage_pipeline(traj, candidates: list[dict], factor_pool) -> None:
    """FactorMiner Algorithm 1, Stage 1-4 多阶段评估

    Stage 1: Fast IC screen (短时间窗口, tau_IC)
    Stage 2: Correlation check vs factor_pool (theta)
    Stage 2.5: Replacement check (被相关性拒绝但 IC 够强 → 替换)
    Stage 3: Intra-batch dedup (AST similarity)
    Stage 4: Full validation (现有逻辑)

    仅在 --daily 模式且 USE_MULTISTAGE=True 时启用。
    """
    from factor_lab.mining.validator import validate_expression, validate_with_qlib
    from . import factor_agent
    from .ast_dedup import ast_similarity

    # --- 预验证: 复杂度 + 静态 + Qlib 动态 ---
    valid_candidates = []
    for cand in candidates:
        name = cand.get('name', '')
        expr = cand.get('expr', '')

        ok, reason = factor_agent.check_paper_complexity(expr)
        if not ok:
            print(f"    [{name}] 复杂度不通过: {reason}")
            continue
        traj.constraint_ok = True

        ok, reason = validate_expression(name, expr)
        if not ok:
            fixed_name = name.upper().replace(' ', '_').replace('-', '_')
            if not fixed_name.startswith('QA_'):
                fixed_name = 'QA_' + fixed_name
            ok, reason = validate_expression(fixed_name, expr)
            if not ok:
                print(f"    [{name}] 静态验证: {reason}")
                continue
            cand['name'] = fixed_name

        ok, reason = validate_with_qlib(cand['name'], expr)
        if not ok:
            print(f"    [{cand['name']}] Qlib验证: {reason}")
            traj.error_msg = reason
            continue
        traj.qlib_ok = True
        valid_candidates.append(cand)

    if not valid_candidates:
        traj.failure_step = 1
        traj.error_msg = traj.error_msg or "所有候选验证失败"
        return

    print(f"    [multistage] {len(valid_candidates)} 候选通过预验证")

    # --- Stage 1: Fast IC screen ---
    fast_start, fast_end = FAST_IC_PERIOD
    fast_factors = [(c['name'], c['expr']) for c in valid_candidates]
    fast_results = evaluate_candidates_batch(fast_factors, start_time=fast_start, end_time=fast_end)

    stage1_pass = []
    for cand, result in zip(valid_candidates, fast_results):
        ic_abs = abs(result.get('rank_ic', 0))
        if ic_abs >= FAST_IC_THRESHOLD:
            cand['_fast_ic'] = result.get('rank_ic', 0)
            cand['_fast_icir'] = result.get('icir', 0)
            stage1_pass.append(cand)
        else:
            print(f"    [Stage 1] {cand['name']} 淘汰 (|IC|={ic_abs:.4f} < {FAST_IC_THRESHOLD})")

    print(f"    [Stage 1] Fast IC: {len(stage1_pass)}/{len(valid_candidates)} 通过")
    if not stage1_pass:
        traj.failure_step = 2
        traj.error_msg = "Stage 1: 所有候选 Fast IC 低于阈值"
        return

    # --- Stage 2: Correlation check vs factor_pool ---
    stage2_pass = []
    stage2_corr_rejected = []

    pool_exprs = factor_pool.get_exprs() if factor_pool else []
    for cand in stage1_pass:
        if not pool_exprs:
            stage2_pass.append(cand)
            continue

        max_corr = 0.0
        max_corr_factor = ""
        corr_count = 0  # 超过阈值的因子数

        for pf_name, pf_expr in pool_exprs:
            sim = ast_similarity(cand['expr'], pf_expr)
            if sim > max_corr:
                max_corr = sim
                max_corr_factor = pf_name
            if sim >= CORRELATION_THRESHOLD:
                corr_count += 1

        cand['_max_corr'] = max_corr
        cand['_max_corr_factor'] = max_corr_factor
        cand['_corr_count'] = corr_count

        if max_corr < CORRELATION_THRESHOLD:
            stage2_pass.append(cand)
        else:
            stage2_corr_rejected.append(cand)
            print(f"    [Stage 2] {cand['name']} 相关性过高 "
                  f"({max_corr:.2f} vs {max_corr_factor})")

    print(f"    [Stage 2] Correlation: {len(stage2_pass)} 通过, "
          f"{len(stage2_corr_rejected)} 被拒")

    # --- Stage 2.5: Replacement check ---
    replacements = []
    if factor_pool and stage2_corr_rejected:
        for cand in stage2_corr_rejected:
            fast_ic = abs(cand.get('_fast_ic', 0))
            corr_count = cand.get('_corr_count', 0)

            if fast_ic >= REPLACE_IC_MIN and corr_count == 1:
                max_corr_factor = cand['_max_corr_factor']
                replaced = factor_pool.try_replace(
                    name=cand['name'],
                    expr=cand['expr'],
                    rank_ic=cand.get('_fast_ic', 0),
                    icir=cand.get('_fast_icir', 0),
                    max_corr_factor=max_corr_factor,
                    max_corr_value=cand.get('_max_corr', 0),
                )
                if replaced:
                    replacements.append(cand)
                    print(f"    [Stage 2.5] {cand['name']} 替换 {max_corr_factor}")

    combined = stage2_pass + replacements
    if not combined:
        traj.failure_step = 2
        traj.error_msg = "Stage 2: 所有候选被相关性拒绝且不满足替换条件"
        return

    # --- Stage 3: Intra-batch dedup (AST similarity) ---
    stage3_pass = []
    for cand in combined:
        is_dup = False
        for existing in stage3_pass:
            sim = ast_similarity(cand['expr'], existing['expr'])
            if sim >= BATCH_DEDUP_AST:
                print(f"    [Stage 3] {cand['name']} 批内重复 "
                      f"(AST sim={sim:.2f} vs {existing['name']})")
                is_dup = True
                break
        if not is_dup:
            stage3_pass.append(cand)

    print(f"    [Stage 3] Batch dedup: {len(stage3_pass)}/{len(combined)} 通过")
    if not stage3_pass:
        traj.failure_step = 2
        traj.error_msg = "Stage 3: 所有候选批内重复"
        return

    # --- Stage 4: Full validation (使用与回测一致的时间窗口) ---
    factors = [(c['name'], c['expr']) for c in stage3_pass]
    eval_results = evaluate_candidates_batch(
        factors,
        start_time=ROLLING_EVAL_TEST_START,
        end_time=ROLLING_EVAL_TEST_END,
    )

    best = select_best_candidate(eval_results)
    if best:
        traj.best_factor = {
            'name': best['name'],
            'expr': best['expr'],
            'description': next(
                (c.get('description', '') for c in stage3_pass
                 if c['name'] == best['name']),
                '',
            ),
        }
        traj.ic = best.get('ic', 0)
        traj.icir = best.get('icir', 0)
        traj.rank_ic = best.get('rank_ic', 0)
        traj.rank_icir = best.get('rank_icir', 0)
        print(f"    [Stage 4] 最佳: {best['name']} (ICIR={best.get('icir', 0):.3f})")
    else:
        traj.failure_step = 2
        traj.error_msg = "Stage 4: 所有候选 IC 为 0"


# ============ v3 新增: 组合回测 ============

def run_combined_backtest(new_factors: list[tuple[str, str]],
                          sota_factors: list[tuple[str, str]],
                          use_rolling: bool = False) -> dict:
    """合并新因子 + SOTA 因子 + Alpha158 -> LightGBM 训练 -> 回测

    对齐论文 QlibFactorRunner.develop() 的组合回测。

    Args:
        new_factors: 新挖掘因子
        sota_factors: 当前 SOTA 因子
        use_rolling: True=使用精简版 rolling eval (4窗口~100s), False=单次训练 (原始逻辑)

    Returns:
        {IC, ICIR, RankIC, sharpe, ARR, MDD} 或空 dict (失败时)
    """
    if not BACKTEST_ENABLED:
        return {}

    # 合并因子 (去重)
    seen = set()
    all_factors = []
    for name, expr in list(new_factors) + list(sota_factors):
        if name not in seen:
            seen.add(name)
            all_factors.append((name, expr))

    if not all_factors:
        return {}

    if use_rolling:
        return _run_rolling_eval_lite(all_factors)

    return _run_single_shot_backtest(all_factors)


def _run_single_shot_backtest(all_factors: list[tuple[str, str]]) -> dict:
    """单次训练回测 (原始逻辑)"""
    try:
        from factor_lab.run_paper_replication import (
            _get_valid_instruments, filter_test_predictions,
            compute_factor_metrics, run_backtest,
        )
        from factor_lab.factors.custom_handler import build_handler_from_exprs

        print(f"  [backtest] 组合回测: {len(all_factors)} 挖掘因子 + Alpha158")
        t0 = time.time()

        handler, n_feat = build_handler_from_exprs(
            factor_exprs=all_factors,
            start_time=TRAIN_START, end_time=TEST_END,
            fit_start_time=TRAIN_START, fit_end_time=TRAIN_END,
            instruments=_get_valid_instruments(),
            include_alpha158=True,
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

        factor_metrics = compute_factor_metrics(pred)
        bt_result = run_backtest(pred)
        result = {**factor_metrics, **bt_result}

        elapsed = time.time() - t0
        print(f"  [backtest] Sharpe={result.get('sharpe', 0):.3f}, "
              f"ARR={result.get('ARR', 0):.2f}%, MDD={result.get('MDD', 0):.2f}% "
              f"({elapsed:.1f}s)")

        del handler, dataset, model
        gc.collect()

        return result

    except Exception as e:
        print(f"  [backtest] 组合回测失败: {e}")
        import traceback
        traceback.print_exc()
        return {}


def _run_rolling_eval_lite(all_factors: list[tuple[str, str]]) -> dict:
    """Rolling eval (对齐生产 SOTA: D_expand_3v_3r 全窗口)

    复用 run_rolling_benchmark 的核心逻辑:
    - 使用 D_expand_3v_3r 扩展窗口 (对齐生产 SOTA)
    - 全部窗口 (~9, 由 ROLLING_EVAL_WINDOWS 上限控制)
    - 返回 {sharpe, total_return, max_drawdown, calmar}
    """
    try:
        import pandas as pd
        from factor_lab.run_rolling_benchmark import (
            generate_rolling_windows, build_model, run_backtest,
            ROLLING_CONFIGS,
        )
        from factor_lab.factors.custom_handler import build_handler_from_exprs

        config = ROLLING_CONFIGS.get(ROLLING_EVAL_CONFIG, ROLLING_CONFIGS["D_expand_3v_3r"])
        test_start = ROLLING_EVAL_TEST_START
        test_end = ROLLING_EVAL_TEST_END

        # 生成窗口, 只取最近 N 个
        all_windows = generate_rolling_windows(
            ROLLING_EVAL_CONFIG, config, test_start, test_end,
        )
        windows = all_windows[-ROLLING_EVAL_WINDOWS:] if len(all_windows) > ROLLING_EVAL_WINDOWS else all_windows

        print(f"  [rolling-lite] {len(windows)} 窗口, "
              f"{len(all_factors)} 挖掘因子 + Alpha158")
        t0 = time.time()

        all_preds = []
        for w in windows:
            handler, _ = build_handler_from_exprs(
                factor_exprs=all_factors,
                start_time=w['train_start'],
                end_time=w['pred_end'],
                fit_start_time=w['train_start'],
                fit_end_time=w['train_end'],
                instruments='csi300',
                include_alpha158=True,
            )

            from qlib.data.dataset import DatasetH
            dataset = DatasetH(handler=handler, segments={
                "train": (w['train_start'], w['train_end']),
                "valid": (w['valid_start'], w['valid_end']),
                "test": (w['pred_start'], w['pred_end']),
            })

            model, fit_kwargs = build_model("LightGBM")
            model.fit(dataset, **fit_kwargs)

            pred = model.predict(dataset)
            if isinstance(pred.index, pd.MultiIndex):
                dates = pred.index.get_level_values(0)
                mask = (dates >= pd.Timestamp(w['pred_start'])) & \
                       (dates <= pd.Timestamp(w['pred_end']))
                pred = pred[mask]

            all_preds.append(pred)

            del handler, dataset, model, pred
            gc.collect()

        if not all_preds:
            return {}

        combined_pred = pd.concat(all_preds)
        if combined_pred.index.duplicated().any():
            combined_pred = combined_pred[~combined_pred.index.duplicated(keep='last')]

        # 回测: 使用 rolling benchmark 的 run_backtest —— 参数来自
        # signal_config.yaml (当前 topk=8 / n_drop=2)，与生产一致。
        # 注意本模块 import 的 TOPK/N_DROP (论文值 50/5) 在这条路径上不生效。
        bt = run_backtest(combined_pred)

        elapsed = time.time() - t0
        result = {
            'sharpe': bt.get('sharpe', 0),
            'total_return': bt.get('total_return', 0),
            'ARR': bt.get('annual_return', 0) * 100 if bt.get('annual_return') else 0,
            'MDD': bt.get('max_drawdown', 0) * 100 if bt.get('max_drawdown') else 0,
            'calmar': abs(bt.get('annual_return', 0) / bt.get('max_drawdown', -1)) if bt.get('max_drawdown', 0) != 0 else 0,
            'n_windows': len(windows),
            'eval_time': round(elapsed, 1),
        }

        print(f"  [rolling-lite] Sharpe={result['sharpe']:.3f}, "
              f"ARR={result['ARR']:.2f}%, MDD={result['MDD']:.2f}% "
              f"({elapsed:.1f}s)")

        return result

    except Exception as e:
        print(f"  [rolling-lite] 精简 rolling eval 失败: {e}")
        import traceback
        traceback.print_exc()
        return {}


def generate_summary(traj: Trajectory) -> str:
    """生成轨迹摘要 (用于 prompt 中提供历史上下文)"""
    status = "SUCCESS" if traj.failure_step == -1 else f"FAILED@step{traj.failure_step}"
    factor_str = ""
    if traj.best_factor:
        factor_str = f" expr={traj.best_factor.get('expr', '')[:80]}"
    return (
        f"[{traj.id}] {status} | {traj.phase} iter={traj.iteration} | "
        f"hyp={traj.hypothesis[:60]}... | ICIR={traj.icir:.3f} reward={traj.reward:.4f}"
        f"{factor_str}"
    )
