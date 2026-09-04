"""Agent Af: 因子构造 + 一致性验证 + 再生循环 (v3)

论文 Section 4.1 — Factor Agent 负责:
1. 将投资假说转化为 Qlib 表达式 (每个假说生成 N_CANDIDATES 个候选)
2. 一致性验证: 假说 <-> 描述 <-> 表达式语义对齐
3. 论文复杂度约束检查

v3 新增 (对齐论文 _convert_with_history_limit):
4. 因子再生循环: 被拒因子累积反馈后重试
"""
import json
import re
import subprocess
import time
from pathlib import Path

from .config import (
    CLAUDE_CLI, CLAUDE_TIMEOUT, MAX_RETRY, RETRY_WAIT,
    N_CANDIDATES, MAX_SYMBOL_LENGTH, MAX_BASE_FEATURES,
    MAX_FREE_PARAM_RATIO, MAX_NESTING_DEPTH, BASE_FEATURES,
    QLIB_OPERATORS, QLIB_CONSTRAINTS, MAX_REGEN_ATTEMPTS,
    REGEN_FEEDBACK_TMPL, get_claude_env,
    CONSISTENCY_CHECK_ENABLED, CONSISTENCY_MIN_SCORE,
    CONSISTENCY_MAX_REPAIR, CONSISTENCY_STRICT,
)

WORK_DIR = Path(__file__).resolve().parent.parent.parent.parent


# ============ v3: 因子再生循环 ============

def construct_factors_with_regen(hypothesis: dict, n: int = N_CANDIDATES,
                                 max_regen: int = MAX_REGEN_ATTEMPTS,
                                 trace_text: str = "",
                                 factor_list_text: str = "") -> tuple[list[dict], int]:
    """因子再生循环 (对齐论文 _convert_with_history_limit)

    1. 构建 prompt (含 trace_text + factor_list_text + 累积拒绝反馈)
    2. Claude CLI 生成因子
    3. check_paper_complexity() 检查每个候选
    4. 不通过 -> 渲染 REGEN_FEEDBACK_TMPL，累积到 regen_feedback -> 重试
    5. 全通过 -> 返回

    Args:
        hypothesis: {hypothesis, direction, mechanism, time_scale}
        n: 候选数量
        max_regen: 最大重试次数
        trace_text: 历史轨迹文本 (来自 DirectionTrace.render_for_prompt)
        factor_list_text: 已尝试因子列表 (来自 DirectionTrace.render_factor_list_for_prompt)

    Returns:
        (candidates, regen_count)
    """
    regen_feedback = ""
    all_tried_names = set()
    regen_count = 0

    for attempt in range(max_regen + 1):
        # 构建 prompt
        candidates = _construct_with_context(
            hypothesis, n,
            trace_text=trace_text,
            factor_list_text=factor_list_text,
            regen_feedback=regen_feedback,
        )

        if not candidates:
            if attempt < max_regen:
                regen_count += 1
                regen_feedback += "\n\n[再生尝试] Claude 未返回任何候选，请重试。"
                print(f"  [regen] 第 {regen_count} 次重试 (无候选)")
                continue
            return [], regen_count

        # 逐个候选: 复杂度检查 -> 语义一致性闸门
        valid = []
        rejected_details = []
        for cand in candidates:
            name = cand.get('name', '')
            expr = cand.get('expr', '')
            all_tried_names.add(name)

            ok, reason = check_paper_complexity(expr)
            if not ok:
                rejected_details.append(f"- {name}: {expr[:80]}... -> 拒绝原因: {reason}")
                continue

            accepted, reason = _gate_consistency(cand, hypothesis)
            if accepted is None:
                rejected_details.append(f"- {name}: {expr[:80]}... -> 拒绝原因: {reason}")
                continue
            all_tried_names.add(accepted.get('name', name))
            valid.append(accepted)

        if valid:
            # 有通过的候选
            if rejected_details:
                print(f"  [regen] {len(valid)}/{len(candidates)} 通过复杂度+一致性检查")
            return valid, regen_count

        # 全部被拒 -> 累积反馈后重试
        if attempt < max_regen:
            regen_count += 1
            tried_list = "\n".join(f"- {n}" for n in sorted(all_tried_names))
            regen_feedback = REGEN_FEEDBACK_TMPL.format(
                rejection_details="\n".join(rejected_details),
                tried_factors=tried_list or "(无)",
            )
            print(f"  [regen] 第 {regen_count} 次重试 (全部不通过)")
        else:
            print(f"  [regen] 达到最大重试次数 ({max_regen}), 返回空列表")

    return [], regen_count


def _gate_consistency(cand: dict, hypothesis: dict) -> tuple[dict | None, str]:
    """语义一致性闸门 (对齐论文 FactorConsistencyChecker.check_and_correct)

    判定不一致时先做定向重构 (reconstruct_factor)，重构结果需同时通过复杂度
    与一致性检查才接受; 重构耗尽仍不一致则拒绝，交由外层再生循环处理。

    候选会被标注 consistency_ok / consistency_feedback 供 Trajectory 记录。

    Returns:
        (accepted_candidate, reason) — accepted 为 None 表示拒绝，reason 说明原因
    """
    if not CONSISTENCY_CHECK_ENABLED:
        cand['consistency_ok'] = None
        cand['consistency_feedback'] = "一致性检查已关闭"
        return cand, ""

    hyp_text = hypothesis.get('hypothesis', '')
    current = cand

    for repair in range(CONSISTENCY_MAX_REPAIR + 1):
        verdict, feedback = verify_consistency(
            hyp_text, current.get('description', ''), current.get('expr', ''))

        if verdict is True:
            current['consistency_ok'] = True
            current['consistency_feedback'] = feedback
            if repair:
                print(f"  [一致性] {current.get('name','')} 第 {repair} 次重构后通过")
            return current, ""

        if verdict is None:
            # 未能验证: 绝不伪装成通过
            if CONSISTENCY_STRICT:
                return None, f"一致性未能验证 (strict): {feedback}"
            print(f"  [一致性] ⚠ {current.get('name','')} 未能验证，按非严格模式放行: {feedback}")
            current['consistency_ok'] = None
            current['consistency_feedback'] = feedback
            return current, ""

        # verdict is False -> 定向重构
        if repair >= CONSISTENCY_MAX_REPAIR:
            return None, f"假说与表达式语义不一致 (已重构 {repair} 次): {feedback}"

        print(f"  [一致性] {current.get('name','')} 不一致，第 {repair+1} 次定向重构")
        repaired = reconstruct_factor(hypothesis, feedback,
                                      failed_expr=current.get('expr', ''))
        repaired = [r for r in repaired if check_paper_complexity(r.get('expr', ''))[0]]
        if not repaired:
            return None, f"语义不一致且重构未产出合法表达式: {feedback}"
        current = repaired[0]

    return None, "一致性闸门未得出结论"


def _construct_with_context(hypothesis: dict, n: int,
                            trace_text: str = "",
                            factor_list_text: str = "",
                            regen_feedback: str = "") -> list[dict]:
    """带上下文的因子构造 (内部函数)"""
    features_str = ", ".join(BASE_FEATURES)

    # 历史上下文
    context_section = ""
    if trace_text and trace_text != "(暂无历史记录)":
        context_section += f"\n## 历史上下文\n{trace_text}\n"
    if factor_list_text and factor_list_text != "(暂无已尝试因子)":
        context_section += f"\n## 已尝试因子 (避免重复)\n{factor_list_text}\n"
    if regen_feedback:
        context_section += f"\n{regen_feedback}\n"

    prompt = f"""你是一位量化因子工程师。请将以下投资假说转化为 {n} 个 Qlib 表达式。

## 投资假说
{hypothesis.get('hypothesis', '')}
- 方向: {hypothesis.get('direction', '')}
- 机制: {hypothesis.get('mechanism', '')}
- 时间尺度: {hypothesis.get('time_scale', '')}
{context_section}
## 可用基础特征
{features_str}
(注意: 没有 $vwap 字段, VWAP 需写为 Div($amount, $volume + 1e-8))

{QLIB_OPERATORS}

{QLIB_CONSTRAINTS}

## 复合模式参考
- EMA 近似: Div(Mean($close, 5), Mean($close, 20)) — 短/长均线比
- RSI 近似: Div(Mean(Greater($close - Ref($close, 1), 0), 14), Mean(Abs($close - Ref($close, 1)), 14) + 1e-8)
- ATR: Mean(Greater(Greater($high - $low, Abs($high - Ref($close, 1))), Abs($low - Ref($close, 1))), 14)
- 布林带宽: Div(Std($close, 20), Mean($close, 20) + 1e-8)
- VWAP 偏离: Div($close - Div($amount, $volume + 1e-8), Std($close, 20) + 1e-8)

## 输出要求
每个因子的:
- name: 全大写下划线格式 (如 VOL_REGIME_SHIFT)，以 QA_ 开头
- expr: 有效的 Qlib 表达式
- description: 描述这个表达式如何实现假说 (1-2 句话)

{n} 个候选应使用不同的算子组合或时间窗口，但都要忠实于同一假说。

严格输出 JSON 数组:
```json
[
  {{
    "name": "QA_FACTOR_NAME",
    "expr": "Qlib expression",
    "description": "如何实现假说的说明"
  }},
  ...
]
```"""

    return _call_claude_factors(prompt)


# ============ 原有: 因子构造 ============

def construct_factors(hypothesis: dict, n: int = N_CANDIDATES) -> list[dict]:
    """将投资假说转化为 n 个候选 Qlib 表达式 (v2 兼容接口)"""
    return _construct_with_context(hypothesis, n)


def verify_consistency(hypothesis: str, description: str, expr: str
                       ) -> tuple[bool | None, str]:
    """一致性验证: 假说 <-> 描述 <-> 表达式是否语义对齐

    对应论文四大组件之一 (enforces semantic consistency across hypothesis,
    factor expression, and executable code)，用于抑制"假说说的是动量、表达式
    其实在算别的"这类因子进入池子 —— 它们靠样本内噪声取胜，样本外必然崩。

    返回 (verdict, feedback):
      True  = 一致
      False = 不一致，应拒绝并重构
      None  = 未能验证 (调用/解析失败)，由调用方按 CONSISTENCY_STRICT 决定
    """
    prompt = f"""你是一位量化因子审核员。请验证以下三者之间的一致性:

## 投资假说
{hypothesis}

## 因子描述
{description}

## Qlib 表达式
{expr}

## 验证标准
1. 表达式是否正确实现了描述中的逻辑?
2. 描述是否忠实于投资假说?
3. 表达式中使用的字段和时间窗口是否合理?

请输出 JSON:
```json
{{
  "is_consistent": true/false,
  "score": 0.0-1.0,
  "issues": ["问题1", "问题2"],
  "suggestion": "如何修复不一致 (若有)"
}}
```"""

    try:
        cmd = [
            CLAUDE_CLI,
            '--print', '--dangerously-skip-permissions',
            '--output-format', 'text',
        ]
        # prompt 走 stdin (Windows .cmd 包装器按换行截断 argv, 见 llm_backend._invoke)
        result = subprocess.run(
            cmd, input=prompt, capture_output=True, text=True,
            timeout=CLAUDE_TIMEOUT, cwd=str(WORK_DIR),
            env=get_claude_env(),
        )
        output = result.stdout.strip()
        data = _extract_json_obj(output)
        if data:
            is_ok = data.get('is_consistent', False)
            score = data.get('score', 0)
            issues = data.get('issues', [])
            suggestion = data.get('suggestion', '')
            feedback = f"score={score:.1f}"
            if issues:
                feedback += f", issues: {'; '.join(issues[:3])}"
            if suggestion:
                feedback += f", suggestion: {suggestion}"
            return bool(is_ok or score >= CONSISTENCY_MIN_SCORE), feedback
        # 无法解析 != 通过。返回 None 表示"未能验证"，由调用方决定放行还是拦截，
        # 不再伪装成 True —— 那会让这道闸在解析失败时静默失效。
        return None, "无法解析验证输出"
    except Exception as e:
        print(f"  [factor_agent] 一致性验证异常: {e}")
        return None, f"验证异常: {e}"


def check_paper_complexity(expr: str) -> tuple[bool, str]:
    """论文复杂度约束检查"""
    # 长度
    if len(expr) > MAX_SYMBOL_LENGTH:
        return False, f"表达式过长 ({len(expr)} > {MAX_SYMBOL_LENGTH} 字符)"

    # 基础字段数
    fields = set(re.findall(r'\$[a-z_]+', expr))
    if len(fields) > MAX_BASE_FEATURES:
        return False, f"字段过多 ({len(fields)} > {MAX_BASE_FEATURES}: {fields})"

    # 嵌套深度
    max_depth = 0
    depth = 0
    for ch in expr:
        if ch == '(':
            depth += 1
            max_depth = max(max_depth, depth)
        elif ch == ')':
            depth -= 1
    if max_depth > MAX_NESTING_DEPTH:
        return False, f"嵌套过深 ({max_depth} > {MAX_NESTING_DEPTH})"

    # 自由参数比例: 常量数 / 总 token 数
    tokens = re.findall(r'[A-Za-z$_]+\w*|[+-]?\d+\.?\d*(?:[eE][+-]?\d+)?', expr)
    if tokens:
        num_constants = sum(1 for t in tokens if re.match(r'^[+-]?\d', t))
        ratio = num_constants / len(tokens)
        if ratio > MAX_FREE_PARAM_RATIO:
            return False, f"自由参数比例过高 ({ratio:.0%} > {MAX_FREE_PARAM_RATIO:.0%})"

    return True, ""


def reconstruct_factor(hypothesis: dict, feedback: str,
                       failed_expr: str = "") -> list[dict]:
    """Mutation: 基于反馈重新构造因子 (v2 兼容接口)"""
    features_str = ", ".join(BASE_FEATURES)

    prompt = f"""你是一位量化因子工程师。之前的因子构造失败了，请重新构造。

## 投资假说
{hypothesis.get('hypothesis', '')}

## 之前失败的表达式
{failed_expr or '(无)'}

## 失败原因
{feedback}

## 可用基础特征
{features_str}

{QLIB_OPERATORS}

{QLIB_CONSTRAINTS}

## 修改要求
1. 针对失败原因做定向修改
2. 保留假说的核心逻辑
3. 确保表达式 <= 200 字符

请输出 2 个修改后的因子 (JSON 数组):
```json
[
  {{
    "name": "QA_FACTOR_NAME",
    "expr": "修改后的 Qlib expression",
    "description": "修改说明"
  }}
]
```"""

    return _call_claude_factors(prompt)


# ============ 内部函数 ============

def _call_claude_factors(prompt: str) -> list[dict]:
    """调用 Claude 生成因子列表 (带重试)"""
    for attempt in range(1, MAX_RETRY + 1):
        try:
            cmd = [
                CLAUDE_CLI,
                '--print', '--dangerously-skip-permissions',
                '--output-format', 'text',
            ]
            # prompt 走 stdin。原先作为 argv 传递，在 Windows 上被 .cmd 包装器
            # 按第一个换行截断 —— Claude 只收到 "请将以下投资假说转化为 N 个
            # Qlib 表达式。"，假说/算子表/约束/输出格式全部丢失。
            result = subprocess.run(
                cmd, input=prompt, capture_output=True, text=True,
                timeout=CLAUDE_TIMEOUT, cwd=str(WORK_DIR),
                env=get_claude_env(),
            )
            output = result.stdout.strip()
            if not output:
                print(f"  [factor_agent] 空输出 (attempt {attempt}/{MAX_RETRY})")
                if attempt < MAX_RETRY:
                    time.sleep(RETRY_WAIT)
                    continue
                return []
            parsed = _parse_factors(output)
            if parsed:
                return parsed
            print(f"  [factor_agent] 解析失败 (attempt {attempt}/{MAX_RETRY})")
            if attempt < MAX_RETRY:
                time.sleep(RETRY_WAIT)
                continue
            return []
        except subprocess.TimeoutExpired:
            print(f"  [factor_agent] 超时 (attempt {attempt}/{MAX_RETRY})")
            if attempt < MAX_RETRY:
                time.sleep(RETRY_WAIT)
                continue
            return []
        except Exception as e:
            print(f"  [factor_agent] 调用失败: {e} (attempt {attempt}/{MAX_RETRY})")
            if attempt < MAX_RETRY:
                time.sleep(RETRY_WAIT)
                continue
            return []
    return []


def _parse_factors(output: str) -> list[dict]:
    """解析因子列表"""
    for attempt in [
        lambda: json.loads(output),
        lambda: json.loads(re.search(r'```(?:json)?\s*\n(\[[\s\S]*?\])\s*\n```', output).group(1)),
        lambda: json.loads(re.search(r'\[[\s\S]*\]', output).group()),
    ]:
        try:
            data = attempt()
            if isinstance(data, list):
                valid = []
                for item in data:
                    if isinstance(item, dict) and 'name' in item and 'expr' in item:
                        item.setdefault('description', '')
                        valid.append(item)
                return valid
        except (json.JSONDecodeError, AttributeError, TypeError):
            continue
    print(f"  [factor_agent] 无法解析因子 (输出长度={len(output)})")
    return []


def _extract_json_obj(output: str) -> dict:
    """从输出中提取 JSON 对象"""
    for attempt in [
        lambda: json.loads(output),
        lambda: json.loads(re.search(r'```(?:json)?\s*\n(\{[\s\S]*?\})\s*\n```', output).group(1)),
        lambda: json.loads(re.search(r'\{[\s\S]*\}', output).group()),
    ]:
        try:
            data = attempt()
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, AttributeError, TypeError):
            continue
    return {}
