"""Agent Ai: 假说生成 — v3 Trace-aware (多 LLM 后端)

论文 Section 4.1 — Idea Agent 负责:
1. 生成多样化初始假说 (Phase A)
2. 基于失败反馈修改假说 (Mutation)
3. 融合成功假说的互补段 (Crossover)

v3 新增 (对齐论文 AlphaAgentHypothesisGen + MutationOperator + CrossoverOperator):
4. Trace-aware 假说生成 (带历史上下文)
5. 两阶段 mutation suffix (分析 + 引导)
6. 两阶段 crossover suffix (分析 + 融合)

v4: 假说生成改用 LLMBackend 多 LLM 轮询 (Claude/Gemini/Codex)。
    mutation/crossover suffix 也走 LLMBackend (仍是假说生成范畴)。
"""
import json
import re
import time
from pathlib import Path

from .config import (
    MAX_RETRY, RETRY_WAIT, HISTORY_LIMIT,
    BASE_FEATURES, QLIB_OPERATORS, QLIB_CONSTRAINTS,
)
from .trajectory import DirectionTrace, Trajectory
from factor_lab.mining.llm_backend import LLMBackend


def _get_llm() -> LLMBackend:
    return LLMBackend.shared()


# ============ v3: Trace-aware 假说生成 ============

def generate_hypothesis_with_trace(direction_trace: DirectionTrace,
                                   direction: str,
                                   suffix: str = "",
                                   history_limit: int = HISTORY_LIMIT,
                                   experience_memory=None) -> dict:
    """核心: 将 trace 历史 + evolution suffix 全部序列化到 Claude CLI prompt

    对齐论文 AlphaAgentHypothesisGen.gen(trace) + prepare_context() 的完整逻辑。
    - 无历史 -> 使用 direction (论文 potential_direction_transformation)
    - 有历史 -> 渲染 hypothesis_and_feedback + 提示关注最后一条反馈
    - suffix 来自 mutation/crossover 的 Stage 2 输出
    """
    features_str = ", ".join(BASE_FEATURES)
    trace_text = direction_trace.render_for_prompt(limit=history_limit)
    factor_list = direction_trace.render_factor_list_for_prompt()
    has_history = len(direction_trace.entries) > 0

    if has_history:
        # 有历史: 渲染 hypothesis_and_feedback
        history_section = f"""## 历史假说与反馈 (最近 {history_limit} 轮)
{trace_text}

**请特别关注最后一轮的反馈**，根据反馈改进假说方向。

## 已尝试因子列表 (避免重复)
{factor_list}"""
    else:
        # 无历史: 使用 direction 作为探索方向
        history_section = f"""## 探索方向
{direction}

这是首轮探索，请基于上述方向提出原创假说。"""

    # suffix 来自 mutation/crossover
    suffix_section = ""
    if suffix:
        suffix_section = f"""
## 演化引导
{suffix}"""

    # experience memory (FactorMiner Section 3.3)
    memory_section = ""
    if experience_memory is not None:
        memory_text = experience_memory.format_for_prompt()
        if memory_text:
            memory_section = f"\n{memory_text}\n"

    prompt = f"""你是一位资深量化研究员。请基于以下上下文生成一个投资假说，用于 A 股 CSI300 因子挖掘。

{history_section}
{suffix_section}
{memory_section}
## 可用基础特征
{features_str}
(VWAP 需写为 Div($amount, $volume + 1e-8), 不能直接用 $vwap)

{QLIB_OPERATORS}

{QLIB_CONSTRAINTS}

## 输出要求
请输出一个 JSON 对象:
```json
{{
  "hypothesis": "详细的投资假说 (2-3 句话，包含经济学逻辑)",
  "direction": "信号源类型",
  "mechanism": "市场机制",
  "time_scale": "时间尺度"
}}
```"""

    result = _call_llm(prompt, _parse_single_hypothesis)
    if result:
        return result
    # Fallback: 返回基于 direction 的默认假说
    return {
        "hypothesis": f"基于 {direction} 的量价因子",
        "direction": direction,
        "mechanism": "动量",
        "time_scale": "中期",
    }


def generate_mutation_suffix(parent: Trajectory,
                             direction_trace: DirectionTrace) -> str:
    """两阶段 mutation (对齐论文 MutationOperator)

    Stage 1: Claude CLI 分析父轨迹，生成 mutation 引导
    Stage 2: 格式化为 suffix 文本
    """
    parent_summary = _format_parent_summary(parent)
    trace_summary = direction_trace.render_for_prompt(limit=3)  # 最近 3 条作为上下文

    prompt = f"""你是一位量化研究策略师。请分析以下因子挖掘轨迹，提出一个**正交方向**的 mutation 建议。

## 父轨迹
{parent_summary}

## 近期历史
{trace_summary}

## 任务
1. 识别父轨迹的核心策略模式
2. 提出一个与之正交的新探索方向 (不同的信号源、时间尺度、或市场机制)
3. 说明为什么这个新方向值得探索

请输出 JSON:
```json
{{
  "new_hypothesis": "新探索方向的假说描述",
  "exploration_direction": "正交方向类型 (量价/波动/动量/反转/regime/...)",
  "orthogonality_reason": "与父轨迹正交的理由 (1 句话)"
}}
```"""

    result = _call_llm(prompt, _parse_single_hypothesis_flexible)
    if not result:
        return ""

    # Stage 2: 格式化为 suffix
    suffix = (
        f"### Mutation 引导\n"
        f"父策略: {parent.hypothesis[:100]}\n"
        f"父策略 ICIR: {parent.icir:.4f}\n"
        f"正交方向: {result.get('exploration_direction', '')}\n"
        f"引导: {result.get('new_hypothesis', '')}\n"
        f"理由: {result.get('orthogonality_reason', '')}"
    )
    return suffix


def generate_crossover_suffix(parents: list[Trajectory]) -> str:
    """两阶段 crossover (对齐论文 CrossoverOperator)

    Stage 1: Claude CLI 分析多个父轨迹，生成融合引导
    Stage 2: 格式化为 suffix 文本
    """
    parent_blocks = []
    for i, p in enumerate(parents):
        parent_blocks.append(_format_parent_summary(p, prefix=f"父轨迹 {i+1}"))

    prompt = f"""你是一位量化研究策略师。请分析以下多个因子挖掘轨迹，提出一个**融合创新**的 crossover 策略。

{chr(10).join(parent_blocks)}

## 任务
1. 识别每个父轨迹的核心优势
2. 设计一个融合策略，取各家之长
3. 说明融合的创新点

请输出 JSON:
```json
{{
  "hybrid_hypothesis": "融合后的策略假说",
  "fusion_logic": "融合逻辑 (取 A 的什么 + B 的什么)",
  "innovation_points": "创新点说明"
}}
```"""

    result = _call_llm(prompt, _parse_single_hypothesis_flexible)
    if not result:
        return ""

    # Stage 2: 格式化为 suffix
    parent_summaries = []
    for i, p in enumerate(parents):
        parent_summaries.append(
            f"  {i+1}. {p.hypothesis[:80]} (ICIR={p.icir:.4f})"
        )

    suffix = (
        f"### Crossover 引导\n"
        f"父策略:\n{chr(10).join(parent_summaries)}\n"
        f"融合假说: {result.get('hybrid_hypothesis', '')}\n"
        f"融合逻辑: {result.get('fusion_logic', '')}\n"
        f"创新点: {result.get('innovation_points', '')}"
    )
    return suffix


def _format_parent_summary(traj: Trajectory, prefix: str = "父轨迹") -> str:
    """辅助函数: 格式化轨迹摘要"""
    factor_expr = ""
    if traj.best_factor:
        factor_expr = traj.best_factor.get('expr', '')[:100]
    return (
        f"## {prefix}\n"
        f"- 假说: {traj.hypothesis[:150]}\n"
        f"- 方向: {traj.direction} | 机制: {traj.mechanism}\n"
        f"- 最佳因子: {factor_expr}\n"
        f"- IC={traj.ic:.4f}, ICIR={traj.icir:.4f}, RankIC={traj.rank_ic:.4f}\n"
        f"- 状态: {'成功' if traj.failure_step == -1 else f'失败@step{traj.failure_step}'}"
    )


# ============ Phase A: 初始假说生成 (保持原有) ============

def generate_diverse_hypotheses(n: int = 10) -> list[dict]:
    """Phase A: 一次生成 n 个多样化投资假说"""
    features_str = ", ".join(BASE_FEATURES)

    prompt = f"""你是一位资深量化研究员。请生成 {n} 个**多样化的**投资假说，用于 A 股 CSI300 因子挖掘。

## 可用基础特征
{features_str}
(VWAP 需写为 Div($amount, $volume + 1e-8), 不能直接用 $vwap)

## 多样性要求
你的 {n} 个假说必须覆盖以下维度的不同组合:

**信号源** (至少覆盖 4 种):
- 价格结构 (open/high/low/close 之间的关系)
- 量能模式 (volume 的变化规律)
- 量价联合 (价格变化与成交量的交互)
- 波动率特征 (价格振幅、标准差)
- VWAP 偏离 (价格 vs 成交均价)

**时间尺度** (至少覆盖 3 种):
- 短期 (1-5 日)
- 中期 (10-20 日)
- 长期 (40-60 日)

**市场机制** (至少覆盖 4 种):
- 动量 (趋势延续)
- 反转 (均值回归)
- Regime 切换 (波动率状态转换)
- 流动性冲击 (异常交易量)
- 隔夜效应 (open vs 前 close)
- 机构行为 (大单/分时特征的代理)

## 输出格式
严格输出 JSON 数组:
```json
[
  {{
    "id": 0,
    "hypothesis": "详细描述投资假说 (2-3 句话，包含经济学逻辑)",
    "direction": "信号源类型 (价格结构/量能模式/量价联合/波动率/VWAP偏离)",
    "mechanism": "市场机制 (动量/反转/regime/流动性/隔夜/机构行为)",
    "time_scale": "时间尺度 (短期/中期/长期/多尺度)"
  }},
  ...
]
```

注意: 每个假说的 (direction, mechanism, time_scale) 三元组应尽量不同。"""

    return _call_llm(prompt, _parse_hypotheses)


# ============ 保持原有 (v3 仍保留但不在 5 步循环中使用) ============

def mutate_hypothesis(original: dict, feedback: str) -> dict:
    """Mutation: 基于反馈修改失败的假说 (v2 遗留, 5 步循环不直接调用)"""
    prompt = f"""你是一位量化研究员。以下投资假说在因子构造或评估阶段失败了，请修改它。

## 原始假说
- 假说: {original.get('hypothesis', '')}
- 方向: {original.get('direction', '')}
- 机制: {original.get('mechanism', '')}

## 失败反馈
{feedback}

## 可用基础特征
{', '.join(BASE_FEATURES)}

## 修改要求
1. 保留原始假说的核心直觉
2. 针对失败原因做定向修改
3. 如果信号太弱，考虑换时间尺度或增加条件
4. 如果表达式有问题，简化逻辑或换算子

请输出一个修改后的假说 (JSON 对象):
```json
{{
  "hypothesis": "修改后的投资假说",
  "direction": "方向",
  "mechanism": "机制",
  "time_scale": "时间尺度",
  "mutation_rationale": "修改理由 (1 句话)"
}}
```"""

    results = _call_llm(prompt, _parse_single_hypothesis)
    if results:
        return results
    return original


def crossover_hypotheses(parents: list[dict], rewards: list[float]) -> dict:
    """Crossover: 融合成功假说 (v2 遗留, 5 步循环不直接调用)"""
    parent_blocks = []
    for i, (p, r) in enumerate(zip(parents, rewards)):
        parent_blocks.append(
            f"### 假说 {i+1} (reward={r:.4f})\n"
            f"- 假说: {p.get('hypothesis', '')}\n"
            f"- 方向: {p.get('direction', '')}\n"
            f"- 机制: {p.get('mechanism', '')}\n"
            f"- 最佳因子: {p.get('best_expr', 'N/A')}"
        )

    prompt = f"""你是一位量化研究员。以下是几个已验证有效的投资假说，请**融合它们的互补优势**创造一个新假说。

## 父代假说
{chr(10).join(parent_blocks)}

## 可用基础特征
{', '.join(BASE_FEATURES)}

## 融合策略
- 取 A 的时间结构 + B 的截面逻辑
- 取 A 的信号源 + B 的机制
- 组合多个尺度的信号

请输出一个融合后的假说 (JSON 对象):
```json
{{
  "hypothesis": "融合后的投资假说 (说明来自哪些父代的哪些元素)",
  "direction": "方向",
  "mechanism": "机制",
  "time_scale": "时间尺度",
  "crossover_rationale": "融合理由"
}}
```"""

    results = _call_llm(prompt, _parse_single_hypothesis)
    if results:
        return results
    return parents[0] if parents else {}


# ============ Claude CLI 调用 ============

def _call_llm(prompt: str, parser):
    """通用 LLM 调用 (通过 LLMBackend 多 provider 轮询 + 重试)"""
    empty = [] if parser == _parse_hypotheses else {}

    for attempt in range(1, MAX_RETRY + 1):
        try:
            output = _get_llm().call(prompt)
            if not output:
                print(f"  [idea_agent] LLM 返回空输出 (attempt {attempt}/{MAX_RETRY})")
                if attempt < MAX_RETRY:
                    time.sleep(RETRY_WAIT)
                    continue
                return empty

            parsed = parser(output)
            if parsed:
                return parsed
            print(f"  [idea_agent] 解析失败, 重试 (attempt {attempt}/{MAX_RETRY})")
            if attempt < MAX_RETRY:
                time.sleep(RETRY_WAIT)
                continue
            return empty

        except Exception as e:
            print(f"  [idea_agent] LLM 调用失败: {e} (attempt {attempt}/{MAX_RETRY})")
            if attempt < MAX_RETRY:
                time.sleep(RETRY_WAIT)
                continue
            return empty
    return empty


# ============ JSON 解析 ============

def _parse_hypotheses(output: str) -> list[dict]:
    """从 Claude 输出中提取 JSON 数组"""
    for attempt in [
        lambda: json.loads(output),
        lambda: json.loads(re.search(r'```(?:json)?\s*\n(\[[\s\S]*?\])\s*\n```', output).group(1)),
        lambda: json.loads(re.search(r'\[[\s\S]*\]', output).group()),
    ]:
        try:
            data = attempt()
            if isinstance(data, list):
                return [h for h in data if isinstance(h, dict) and 'hypothesis' in h]
        except (json.JSONDecodeError, AttributeError, TypeError):
            continue
    print(f"  [idea_agent] 无法解析假说 (输出长度={len(output)})")
    return []


def _parse_single_hypothesis(output: str) -> dict:
    """从 Claude 输出中提取单个 JSON 对象 (要求 hypothesis 字段)"""
    for attempt in [
        lambda: json.loads(output),
        lambda: json.loads(re.search(r'```(?:json)?\s*\n(\{[\s\S]*?\})\s*\n```', output).group(1)),
        lambda: json.loads(re.search(r'\{[\s\S]*\}', output).group()),
    ]:
        try:
            data = attempt()
            if isinstance(data, dict) and 'hypothesis' in data:
                return data
        except (json.JSONDecodeError, AttributeError, TypeError):
            continue
    print(f"  [idea_agent] 无法解析单个假说")
    return {}


def _parse_single_hypothesis_flexible(output: str) -> dict:
    """从 Claude 输出中提取单个 JSON 对象 (不要求特定字段)"""
    for attempt in [
        lambda: json.loads(output),
        lambda: json.loads(re.search(r'```(?:json)?\s*\n(\{[\s\S]*?\})\s*\n```', output).group(1)),
        lambda: json.loads(re.search(r'\{[\s\S]*\}', output).group()),
    ]:
        try:
            data = attempt()
            if isinstance(data, dict) and len(data) > 0:
                return data
        except (json.JSONDecodeError, AttributeError, TypeError):
            continue
    print(f"  [idea_agent] 无法解析 JSON 对象")
    return {}
