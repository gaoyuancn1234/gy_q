"""轨迹级自我演化 — v3 对齐论文 AlphaAgentLoop.run()

5 步循环 (论文完整流程):
  1. Propose: 生成 mutation/crossover suffix -> trace-aware 假说
  2. Construct: 因子再生循环 (construct_factors_with_regen)
  3. Calculate: 验证 + 评估 (run_validation_pipeline)
  4. Backtest: 组合回测 (run_combined_backtest)
  5. Feedback: LLM 反馈 -> DirectionTrace.append

论文流程: Original -> Mutation -> Crossover -> Mutation -> Crossover -> ...
"""
import random
from .config import (
    N_CANDIDATES, CROSSOVER_N, CROSSOVER_SIZE, BACKTEST_ENABLED, HISTORY_LIMIT,
    ROLLING_EVAL_LITE, ACCUMULATED_EVAL,
)
from .trajectory import (
    Trajectory, TrajectoryPool, TraceEntry, HypothesisFeedback, DirectionTrace,
)
from . import idea_agent, factor_agent, eval_agent


# ============ Mutation (5 步循环) ============

def get_mutation_targets(pool: TrajectoryPool, current_round: int) -> list[Trajectory]:
    """获取当前轮 mutation 目标 (论文: 取前一轮的输出)"""
    if current_round <= 1:
        targets = pool.get_by_phase("init")
    else:
        crossover_trajs = pool.get_by_phase("crossover")
        if crossover_trajs:
            max_iter = max(t.iteration for t in crossover_trajs)
            targets = [t for t in crossover_trajs if t.iteration == max_iter]
        else:
            mutation_trajs = pool.get_by_phase("mutation")
            if mutation_trajs:
                max_iter = max(t.iteration for t in mutation_trajs)
                targets = [t for t in mutation_trajs if t.iteration == max_iter]
            else:
                targets = pool.get_by_phase("init")

    targets.sort(key=lambda t: t.direction_id)
    return targets


def mutate_trajectory(parent: Trajectory, iteration: int,
                      pool: TrajectoryPool,
                      dry_run: bool = False,
                      no_backtest: bool = False,
                      factor_pool=None) -> Trajectory:
    """Mutation: 5 步循环 (对齐论文 AlphaAgentLoop.run())

    Step 1: Propose — generate_mutation_suffix + generate_hypothesis_with_trace
    Step 2: Construct — construct_factors_with_regen
    Step 3: Calculate — run_validation_pipeline
    Step 4: Backtest — run_combined_backtest
    Step 5: Feedback — generate_llm_feedback + direction_trace.append
    """
    direction_trace = pool.get_direction_trace(parent.direction_id)

    new_traj = Trajectory(
        direction_id=parent.direction_id,
        iteration=iteration,
        phase="mutation",
        parent_ids=[parent.id],
    )

    # --- Step 1: Propose ---
    print(f"    [Step 1] Propose (mutation suffix + hypothesis)")
    suffix = idea_agent.generate_mutation_suffix(parent, direction_trace)
    new_traj.claude_calls += 1

    new_hyp = idea_agent.generate_hypothesis_with_trace(
        direction_trace=direction_trace,
        direction=parent.direction,
        suffix=suffix,
        history_limit=HISTORY_LIMIT,
    )
    new_traj.hypothesis = new_hyp.get('hypothesis', parent.hypothesis)
    new_traj.direction = new_hyp.get('direction', parent.direction)
    new_traj.mechanism = new_hyp.get('mechanism', parent.mechanism)
    new_traj.claude_calls += 1

    if dry_run:
        print(f"    [dry-run] 假说: {new_traj.hypothesis[:80]}...")
        new_traj.compute_reward()
        pool.add(new_traj)
        return new_traj

    # --- Step 2: Construct (with regen) ---
    print(f"    [Step 2] Construct (with regen loop)")
    trace_text = direction_trace.render_for_prompt()
    factor_list_text = direction_trace.render_factor_list_for_prompt()

    candidates, regen_count = factor_agent.construct_factors_with_regen(
        hypothesis=new_hyp,
        n=N_CANDIDATES,
        trace_text=trace_text,
        factor_list_text=factor_list_text,
    )
    new_traj.claude_calls += 1 + regen_count
    new_traj.regen_attempts = regen_count

    if not candidates:
        new_traj.failure_step = 1
        new_traj.error_msg = f"再生循环 {regen_count} 次后仍无有效候选"
        _finalize_trace(new_traj, pool, direction_trace, no_backtest)
        return new_traj

    new_traj.factor_candidates = candidates

    # --- Step 3: Calculate (validate + evaluate) ---
    print(f"    [Step 3] Calculate (validate + evaluate)")
    eval_agent.run_validation_pipeline(new_traj, candidates)

    # --- Step 4: Backtest ---
    if not no_backtest and BACKTEST_ENABLED and new_traj.best_factor and new_traj.failure_step == -1:
        use_rolling = ROLLING_EVAL_LITE
        sota_factors = _get_sota_factors(direction_trace, factor_pool=factor_pool)
        print(f"    [Step 4] Backtest ({'rolling-lite' if use_rolling else 'single-shot'}, "
              f"sota={len(sota_factors)} factors)")
        new_factors = [(new_traj.best_factor['name'], new_traj.best_factor['expr'])]
        bt_metrics = eval_agent.run_combined_backtest(new_factors, sota_factors, use_rolling=use_rolling)
        new_traj.backtest_metrics = bt_metrics
    else:
        print(f"    [Step 4] Backtest (skipped)")

    # --- Step 5: Feedback ---
    _finalize_trace(new_traj, pool, direction_trace, no_backtest)
    return new_traj


# ============ Crossover (5 步循环) ============

def get_crossover_candidates(pool: TrajectoryPool, current_round: int) -> list[Trajectory]:
    """获取 crossover 候选 (论文: 从最近 2 轮选)"""
    original_trajs = pool.get_by_phase("init")
    mutation_trajs = pool.get_by_phase("mutation")
    crossover_trajs = pool.get_by_phase("crossover")

    latest_mut_iter = max((t.iteration for t in mutation_trajs), default=-1)
    latest_mut = [t for t in mutation_trajs if t.iteration == latest_mut_iter] if latest_mut_iter >= 0 else []

    latest_cross_iter = max((t.iteration for t in crossover_trajs), default=-1)
    latest_cross = [t for t in crossover_trajs if t.iteration == latest_cross_iter] if latest_cross_iter >= 0 else []

    if not latest_cross:
        candidates = original_trajs + latest_mut
    else:
        candidates = latest_mut + latest_cross

    candidates = [t for t in candidates if t.failure_step == -1 and t.reward > 0]
    return candidates


def select_crossover_groups(candidates: list[Trajectory],
                            n_groups: int = CROSSOVER_N,
                            group_size: int = CROSSOVER_SIZE) -> list[list[Trajectory]]:
    """从候选中选择 n_groups 个多样化组合"""
    if len(candidates) < group_size:
        return []

    groups = []
    used_pairs = set()

    by_direction: dict[str, list[Trajectory]] = {}
    for t in candidates:
        key = t.direction or str(t.direction_id)
        by_direction.setdefault(key, []).append(t)

    directions = list(by_direction.keys())

    # 策略 1: 不同方向间交叉
    if len(directions) >= group_size:
        for i in range(min(n_groups, len(directions))):
            group = []
            dir_indices = [(i + j) % len(directions) for j in range(group_size)]
            for di in dir_indices:
                trajs = by_direction[directions[di]]
                best = max(trajs, key=lambda t: t.reward)
                group.append(best)

            pair_key = frozenset(t.id for t in group)
            if pair_key not in used_pairs:
                groups.append(group)
                used_pairs.add(pair_key)

    # 补充: 随机选
    attempts = 0
    while len(groups) < n_groups and attempts < 50:
        attempts += 1
        sample = random.sample(candidates, min(group_size, len(candidates)))
        pair_key = frozenset(t.id for t in sample)
        if pair_key not in used_pairs:
            groups.append(sample)
            used_pairs.add(pair_key)

    return groups[:n_groups]


def crossover_trajectories(parents: list[Trajectory], iteration: int,
                           pool: TrajectoryPool,
                           group_idx: int = 0,
                           dry_run: bool = False,
                           no_backtest: bool = False,
                           factor_pool=None) -> Trajectory:
    """Crossover: 5 步循环 (对齐论文 AlphaAgentLoop.run())

    Step 1: Propose — generate_crossover_suffix + generate_hypothesis_with_trace
    Step 2: Construct — construct_factors_with_regen
    Step 3: Calculate — run_validation_pipeline
    Step 4: Backtest — run_combined_backtest
    Step 5: Feedback — generate_llm_feedback + direction_trace.append
    """
    # crossover 使用第一个 parent 的 direction_id
    direction_id = parents[0].direction_id if parents else group_idx
    direction_trace = pool.get_direction_trace(direction_id)

    new_traj = Trajectory(
        direction_id=direction_id,
        iteration=iteration,
        phase="crossover",
        parent_ids=[p.id for p in parents],
    )

    # --- Step 1: Propose ---
    print(f"    [Step 1] Propose (crossover suffix + hypothesis)")
    suffix = idea_agent.generate_crossover_suffix(parents)
    new_traj.claude_calls += 1

    # 合并 parents 的 direction 描述
    combined_direction = " + ".join(set(p.direction for p in parents if p.direction))

    new_hyp = idea_agent.generate_hypothesis_with_trace(
        direction_trace=direction_trace,
        direction=combined_direction or "量价联合",
        suffix=suffix,
        history_limit=HISTORY_LIMIT,
    )
    new_traj.hypothesis = new_hyp.get('hypothesis', '')
    new_traj.direction = new_hyp.get('direction', combined_direction)
    new_traj.mechanism = new_hyp.get('mechanism', '')
    new_traj.claude_calls += 1

    if not new_traj.hypothesis:
        new_traj.failure_step = 0
        new_traj.error_msg = "crossover 未生成有效假说"
        new_traj.compute_reward()
        pool.add(new_traj)
        return new_traj

    if dry_run:
        print(f"    [dry-run] 假说: {new_traj.hypothesis[:80]}...")
        new_traj.compute_reward()
        pool.add(new_traj)
        return new_traj

    # --- Step 2: Construct (with regen) ---
    print(f"    [Step 2] Construct (with regen loop)")
    trace_text = direction_trace.render_for_prompt()
    factor_list_text = direction_trace.render_factor_list_for_prompt()

    candidates, regen_count = factor_agent.construct_factors_with_regen(
        hypothesis=new_hyp,
        n=N_CANDIDATES,
        trace_text=trace_text,
        factor_list_text=factor_list_text,
    )
    new_traj.claude_calls += 1 + regen_count
    new_traj.regen_attempts = regen_count

    if not candidates:
        new_traj.failure_step = 1
        new_traj.error_msg = f"再生循环 {regen_count} 次后仍无有效候选"
        _finalize_trace(new_traj, pool, direction_trace, no_backtest)
        return new_traj

    new_traj.factor_candidates = candidates

    # --- Step 3: Calculate ---
    print(f"    [Step 3] Calculate (validate + evaluate)")
    eval_agent.run_validation_pipeline(new_traj, candidates)

    # --- Step 4: Backtest ---
    if not no_backtest and BACKTEST_ENABLED and new_traj.best_factor and new_traj.failure_step == -1:
        use_rolling = ROLLING_EVAL_LITE
        sota_factors = _get_sota_factors(direction_trace, factor_pool=factor_pool)
        print(f"    [Step 4] Backtest ({'rolling-lite' if use_rolling else 'single-shot'}, "
              f"sota={len(sota_factors)} factors)")
        new_factors = [(new_traj.best_factor['name'], new_traj.best_factor['expr'])]
        bt_metrics = eval_agent.run_combined_backtest(new_factors, sota_factors, use_rolling=use_rolling)
        new_traj.backtest_metrics = bt_metrics
    else:
        print(f"    [Step 4] Backtest (skipped)")

    # --- Step 5: Feedback ---
    _finalize_trace(new_traj, pool, direction_trace, no_backtest)
    return new_traj


# ============ 内部辅助 ============

def _finalize_trace(traj: Trajectory, pool: TrajectoryPool,
                    direction_trace: DirectionTrace,
                    no_backtest: bool = False):
    """Step 5: Feedback + compute_reward + add to pool + append trace"""
    traj.compute_reward()
    pool.add(traj)

    # LLM 反馈
    print(f"    [Step 5] Feedback (LLM)")
    sota_entry = direction_trace.get_sota()
    feedback = eval_agent.generate_llm_feedback(traj, sota_entry, direction_trace)
    traj.llm_feedback = feedback.to_dict()
    traj.claude_calls += 1

    # 构建 TraceEntry 并追加
    factor_name = traj.best_factor.get('name', '') if traj.best_factor else ''
    factor_expr = traj.best_factor.get('expr', '') if traj.best_factor else ''

    entry = TraceEntry(
        hypothesis=traj.hypothesis,
        factor_name=factor_name,
        factor_expr=factor_expr,
        ic=traj.ic,
        icir=traj.icir,
        rank_ic=traj.rank_ic,
        backtest_metrics=traj.backtest_metrics or {},
        feedback=feedback,
        traj_id=traj.id,
    )
    direction_trace.append(entry)

    decision_str = "ACCEPT (新 SOTA)" if feedback.decision else "REJECT"
    print(f"    [Feedback] {decision_str} | {feedback.observations[:80]}")


def _get_sota_factors(direction_trace: DirectionTrace,
                      factor_pool=None) -> list[tuple[str, str]]:
    """从 direction trace 中获取所有 SOTA 因子 + FactorPool 全局因子

    当 ACCUMULATED_EVAL=True 且传入 factor_pool 时，
    Step 4 Backtest 会包含 FactorPool 中所有已入池因子（跨方向累积）。
    """
    seen = set()
    factors = []
    # 1. 当前方向的 SOTA 因子
    for entry in direction_trace.entries:
        if entry.feedback and entry.feedback.decision and entry.factor_name and entry.factor_expr:
            if entry.factor_name not in seen:
                factors.append((entry.factor_name, entry.factor_expr))
                seen.add(entry.factor_name)
    # 2. 跨方向累积: FactorPool 全局因子
    if ACCUMULATED_EVAL and factor_pool is not None:
        for name, expr in factor_pool.get_exprs():
            if name not in seen:
                factors.append((name, expr))
                seen.add(name)
    return factors
