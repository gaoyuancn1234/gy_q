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
)
from .trajectory import Trajectory, HypothesisFeedback, TraceEntry, DirectionTrace


def evaluate_factor(name: str, expr: str,
                    start_time: str = VALID_START,
                    end_time: str = VALID_END) -> dict:
    """评估单个因子的 IC/ICIR

    使用 Valid 期 (2021) 评估, 与论文一致。

    Returns:
        {ic, icir, rank_ic, rank_icir, nan_ratio, eval_time}
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
            return {
                'ic': 0.0, 'icir': 0.0, 'rank_ic': 0.0, 'rank_icir': 0.0,
                'nan_ratio': 1.0, 'eval_time': elapsed, 'error': 'empty result',
            }

        row = df.iloc[0]
        return {
            'ic': float(row.get('mean_IC', 0)),
            'icir': float(row.get('ICIR', 0)),
            'rank_ic': float(row.get('mean_IC', 0)),  # evaluate_candidates 返回的是 rank IC
            'rank_icir': float(row.get('ICIR', 0)),
            'is_promising': bool(row.get('is_promising', False)),
            'is_excellent': bool(row.get('is_excellent', False)),
            'eval_time': elapsed,
        }
    except Exception as e:
        return {
            'ic': 0.0, 'icir': 0.0, 'rank_ic': 0.0, 'rank_icir': 0.0,
            'eval_time': time.time() - t0, 'error': str(e),
        }


def evaluate_candidates_batch(factors: list[tuple[str, str]],
                              start_time: str = VALID_START,
                              end_time: str = VALID_END) -> list[dict]:
    """批量评估因子"""
    from factor_lab.mining.evaluator import evaluate_candidates

    t0 = time.time()
    try:
        df = evaluate_candidates(factors, start_time=start_time, end_time=end_time)
        elapsed = time.time() - t0

        results = []
        for name, expr in factors:
            row = df[df['factor'] == name]
            if row.empty:
                results.append({
                    'name': name, 'expr': expr,
                    'ic': 0.0, 'icir': 0.0, 'rank_ic': 0.0, 'rank_icir': 0.0,
                    'eval_time': elapsed / len(factors),
                })
            else:
                r = row.iloc[0]
                results.append({
                    'name': name, 'expr': expr,
                    'ic': float(r.get('mean_IC', 0)),
                    'icir': float(r.get('ICIR', 0)),
                    'rank_ic': float(r.get('mean_IC', 0)),
                    'rank_icir': float(r.get('ICIR', 0)),
                    'is_promising': bool(r.get('is_promising', False)),
                    'is_excellent': bool(r.get('is_excellent', False)),
                    'eval_time': elapsed / len(factors),
                })
        return results
    except Exception as e:
        print(f"  [eval_agent] 批量评估失败: {e}")
        return [{
            'name': n, 'expr': e_,
            'ic': 0.0, 'icir': 0.0, 'rank_ic': 0.0, 'rank_icir': 0.0,
            'error': str(e),
        } for n, e_ in factors]


def select_best_candidate(candidates: list[dict]) -> Optional[dict]:
    """从评估结果中选出最佳因子

    优先选 |ICIR| 最大的
    """
    if not candidates:
        return None
    valid = [c for c in candidates if abs(c.get('icir', 0)) > 0]
    if not valid:
        return candidates[0] if candidates else None
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
    """验证候选因子 + 选最佳 + 评估 (共享逻辑, 供 evolution.py 和 run_quanta_alpha.py 调用)"""
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
            '-p', prompt,
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True,
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
    is_first = sota_entry is None
    if is_first:
        return HypothesisFeedback(
            observations=f"第一轮, IC={traj.ic:.4f}, ICIR={traj.icir:.4f}",
            hypothesis_evaluation="作为首轮结果自动接受为 SOTA",
            new_hypothesis="后续轮次需要超越此基线",
            reasoning="首轮无对比基准，自动接受",
            decision=True,
        )

    # 与 SOTA 比较
    is_better = abs(traj.icir) > abs(sota_entry.icir) if sota_entry else True
    failure_step, feedback_text = diagnose_failure(traj)
    return HypothesisFeedback(
        observations=f"ICIR={traj.icir:.4f} vs SOTA={sota_entry.icir:.4f if sota_entry else 0:.4f}",
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


# ============ v3 新增: 组合回测 ============

def run_combined_backtest(new_factors: list[tuple[str, str]],
                          sota_factors: list[tuple[str, str]]) -> dict:
    """合并新因子 + SOTA 因子 + Alpha158 -> LightGBM 训练 -> 回测

    对齐论文 QlibFactorRunner.develop() 的组合回测。

    Returns:
        {IC, ICIR, RankIC, sharpe, ARR, MDD} 或空 dict (失败时)
    """
    if not BACKTEST_ENABLED:
        return {}

    try:
        from factor_lab.run_paper_replication import (
            _get_valid_instruments, filter_test_predictions,
            compute_factor_metrics, run_backtest,
        )
        from factor_lab.factors.custom_handler import build_handler_from_exprs

        # 合并因子 (去重)
        seen = set()
        all_factors = []
        for name, expr in list(new_factors) + list(sota_factors):
            if name not in seen:
                seen.add(name)
                all_factors.append((name, expr))

        if not all_factors:
            return {}

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
