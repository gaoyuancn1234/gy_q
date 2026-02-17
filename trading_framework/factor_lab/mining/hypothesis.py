"""Claude CLI 因子假说生成

调用 Claude CLI agent 生成新的量化因子假说。
支持: 多样化种子初始化 / Mutation 靶向修复 / Crossover 模式重组
"""
import json
import re
import subprocess
from pathlib import Path

WORK_DIR = Path(__file__).resolve().parent.parent.parent.parent  # repo root

from factor_lab.quanta.config import (
    CLAUDE_CLI, CLAUDE_TIMEOUT, get_claude_env,
)

# 多样化种子方向 — cycle 轮转使用
SEED_CATEGORIES = [
    {
        "focus": "波动率结构",
        "hint": "短期/长期波动率比、日内振幅变化、波动率聚集/发散",
    },
    {
        "focus": "量价背离",
        "hint": "价格创新高但成交量萎缩、放量滞涨、缩量阴跌",
    },
    {
        "focus": "动量与反转",
        "hint": "动量加速/衰减的二阶导、不同时间尺度的动量交叉",
    },
    {
        "focus": "估值动态",
        "hint": "PE/PB 的变化率、估值分位数的均值回归、市值动量",
    },
    {
        "focus": "流动性微结构",
        "hint": "异常换手率、成交额集中度、Amihud 非流动性因子",
    },
    {
        "focus": "多尺度交叉",
        "hint": "短周期(5日)与长周期(60日)信号的交叉、regime 切换",
    },
]


def generate_hypotheses(context: str, existing_names: list[str],
                        batch_size: int = 10, focus: str = "",
                        seed: dict | None = None) -> list[dict]:
    """调用 Claude CLI 生成因子假说

    Args:
        context: Agent 记忆文本 (AGENT_CONTEXT.md)
        existing_names: 已有因子名 (防重复)
        batch_size: 每批生成数量
        focus: 本次探索方向 (旧参数，seed 优先)
        seed: 种子方向 {"focus": "...", "hint": "..."}

    Returns:
        [{name, expr, hypothesis, category, confidence}, ...]
    """
    existing_str = ", ".join(existing_names[-100:]) if existing_names else "(无)"

    focus_section = ""
    if seed:
        focus_section = f"""
## 本次探索方向
**{seed['focus']}**
提示: {seed['hint']}
请围绕上述方向设计因子。每个因子的经济逻辑要有差异，避免同质化。
"""
    elif focus:
        focus_section = f"""
## 本次探索方向
{focus}
请优先围绕上述方向设计因子。如果方向已被充分探索（见已有因子和失败记录），可换新方向。
"""

    prompt = f"""你是一位资深量化研究员，正在为 A 股 CSI300 股票池设计新的 alpha 因子。

## 可用字段 (Qlib 日频)
$open, $high, $low, $close, $volume, $amount, $turn, $pe_ttm, $pb, $total_mv, $circ_mv

## 可用算子
- 时序: Ref(x,N), Mean(x,N), Std(x,N), Sum(x,N), Delta(x,N), Min(x,N), Max(x,N), Slope(x,N), Rsquare(x,N)
- 截面: Rank(x,N) — 注意需要 window 参数 N
- 运算: Abs(x), Log(x), Sign(x), Power(x,N), Div(x,y), Greater(x,y), Less(x,y), If(cond,x,y)
- 统计: Corr(x,y,N), Cov(x,y,N) — 要求 x,y 日期范围一致 (同源字段)
- 注意: Max/Min 是时序滚动的; 截面比较用 Greater(x,y)/Less(x,y)

## 约束
1. 除零保护: 分母加 1e-8，如 Div(x, y + 1e-8)
2. Corr/Cov 的两个字段必须同源 (都来自日频 OHLCV)，不能混合不同数据范围
3. 不得使用未来数据 (Ref($close, -N) 禁止 N<0 用于因子，仅标签可用)
4. 因子名全大写下划线，如 VOL_REGIME_RATIO
5. 表达式必须是有效的 Qlib 表达式
6. 不要重复已有因子

## 已有因子名 (避免重复)
{existing_str}

## Agent 历史记忆
{context[:3000]}
{focus_section}
## 任务
请设计 {batch_size} 个新的 alpha 因子。每个因子需要:
- 明确的经济学/市场微结构假说
- 可执行的 Qlib 表达式
- 对预测能力的置信度 (0.0~1.0)

请严格输出 JSON 数组，不要输出其他文字：
```json
[
  {{
    "name": "FACTOR_NAME",
    "expr": "Qlib expression",
    "hypothesis": "简述因子背后的经济逻辑",
    "category": "volatility/momentum/liquidity/value/reversal/microstructure",
    "confidence": 0.6
  }},
  ...
]
```"""

    try:
        cmd = [
            CLAUDE_CLI,
            '--print',
            '--dangerously-skip-permissions',
            '--output-format', 'text',
            '-p', prompt,
        ]
        result = subprocess.run(
            cmd,
            capture_output=True, text=True,
            timeout=CLAUDE_TIMEOUT,
            cwd=str(WORK_DIR),
            env=get_claude_env(),
        )
        output = result.stdout.strip()
        return _parse_hypotheses(output)

    except subprocess.TimeoutExpired:
        print(f"  [hypothesis] Claude CLI 超时 ({CLAUDE_TIMEOUT}s)")
        return []
    except Exception as e:
        print(f"  [hypothesis] Claude CLI 调用失败: {e}")
        return []


def _parse_json_list(output: str) -> list | None:
    """从 Claude 输出中提取 JSON 数组 (尝试多种格式)"""
    for extract in [
        lambda: json.loads(output),
        lambda: json.loads(re.search(r'```(?:json)?\s*\n(\[[\s\S]*?\])\s*\n```', output).group(1)),
        lambda: json.loads(re.search(r'\[[\s\S]*\]', output).group()),
    ]:
        try:
            data = extract()
            if isinstance(data, list):
                return data
        except (json.JSONDecodeError, AttributeError, TypeError):
            continue
    return None


def _parse_hypotheses(output: str) -> list[dict]:
    """从 Claude 输出中提取并验证假说列表"""
    data = _parse_json_list(output)
    if data is not None:
        return _validate_hypotheses(data)
    print(f"  [hypothesis] 无法解析 Claude 输出 (长度={len(output)})")
    return []


def _call_claude_for_hypotheses(prompt: str, tag: str) -> list[dict]:
    """调用 Claude CLI 生成假说 (共享逻辑)"""
    try:
        cmd = [
            CLAUDE_CLI,
            '--print',
            '--dangerously-skip-permissions',
            '--output-format', 'text',
            '-p', prompt,
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=CLAUDE_TIMEOUT, cwd=str(WORK_DIR),
            env=get_claude_env(),
        )
        return _parse_hypotheses(result.stdout.strip())
    except subprocess.TimeoutExpired:
        print(f"  {tag} Claude CLI 超时 ({CLAUDE_TIMEOUT}s)")
        return []
    except Exception as e:
        print(f"  {tag} 调用失败: {e}")
        return []


def _validate_hypotheses(items: list) -> list[dict]:
    """验证假说列表的基本结构"""
    valid = []
    required_keys = {"name", "expr", "hypothesis", "category"}
    for item in items:
        if not isinstance(item, dict):
            continue
        if not required_keys.issubset(item.keys()):
            continue
        if not item["name"] or not item["expr"]:
            continue
        item.setdefault("confidence", 0.5)
        valid.append(item)
    return valid


# ---------------------------------------------------------------------------
# Mutation — 靶向修复 "差一点" 的因子
# ---------------------------------------------------------------------------

def _diagnose_failure(factor: dict) -> str:
    """根据评估结果生成诊断建议"""
    icir = abs(factor.get("icir", 0))
    is_redundant = factor.get("is_redundant", False)
    max_corr = factor.get("max_corr", 0)
    most_correlated = factor.get("most_correlated", "")

    if is_redundant:
        return (f"与已有因子 {most_correlated} 高度相关 (corr={max_corr:.2f})。"
                f"请增加差异化: 换时间窗口、加入新字段、或改变信号逻辑。")

    if icir < 0.3:
        return "IC 方向正确但信号极弱。尝试: 调整滚动窗口 (如 5→20 或 20→60)，或换用更敏感的算子。"

    # 0.3 <= icir < 0.5 — near miss
    return "信号方向对但强度不足。尝试: 微调窗口参数、加入 Rank 截面标准化、或用 regime 条件增强。"


def mutate_factors(near_miss: list[dict], context: str,
                   batch_size: int = 5) -> list[dict]:
    """对 near-miss 因子做靶向修复

    Args:
        near_miss: 评估结果列表，每个包含 name/expr/hypothesis/icir/is_redundant/max_corr
        context: Agent 记忆文本
        batch_size: 最多修复几个

    Returns:
        [{name, expr, hypothesis, category, confidence}, ...]
    """
    targets = near_miss[:batch_size]

    factor_blocks = []
    for f in targets:
        diag = _diagnose_failure(f)
        factor_blocks.append(
            f"### {f['name']}\n"
            f"- 假说: {f.get('hypothesis', '')}\n"
            f"- 表达式: {f.get('expr', '')}\n"
            f"- ICIR: {f.get('icir', 0):.3f}\n"
            f"- 诊断: {diag}"
        )

    prompt = f"""你是一位资深量化研究员。以下因子接近有效但未达标，请做**靶向修复**。

## 可用字段 (Qlib 日频)
$open, $high, $low, $close, $volume, $amount, $turn, $pe_ttm, $pb, $total_mv, $circ_mv

## 可用算子
- 时序: Ref(x,N), Mean(x,N), Std(x,N), Sum(x,N), Delta(x,N), Min(x,N), Max(x,N), Slope(x,N), Rsquare(x,N)
- 截面: Rank(x,N)
- 运算: Abs(x), Log(x), Sign(x), Power(x,N), Div(x,y), Greater(x,y), Less(x,y), If(cond,x,y)
- 统计: Corr(x,y,N), Cov(x,y,N) — 要求 x,y 同源

## 约束
1. 除零保护: 分母加 1e-8
2. 名称全大写下划线，以 MUT_ 开头
3. 只调整诊断指出的问题部分，保留原始因子的核心逻辑
4. 表达式长度 ≤ 200 字符, 嵌套 ≤ 5 层, 字段 ≤ 6 种

## 待修复因子
{chr(10).join(factor_blocks)}

## Agent 记忆
{context[:2000]}

请为每个因子输出 1-2 个修复变体。严格输出 JSON 数组:
```json
[
  {{
    "name": "MUT_FACTOR_NAME",
    "expr": "修复后的 Qlib 表达式",
    "hypothesis": "修复逻辑说明",
    "category": "volatility/momentum/liquidity/value/reversal/microstructure",
    "confidence": 0.6
  }}
]
```"""

    return _call_claude_for_hypotheses(prompt, "[mutation]")


# ---------------------------------------------------------------------------
# Crossover — 重组历史成功因子的模式
# ---------------------------------------------------------------------------

def crossover_hypotheses(discoveries: list[dict], context: str,
                         batch_size: int = 10) -> list[dict]:
    """将历史成功因子的模式进行重组

    Args:
        discoveries: 历史成功因子列表 [{name, expr, hypothesis, icir}, ...]
        context: Agent 记忆文本
        batch_size: 生成数量

    Returns:
        [{name, expr, hypothesis, category, confidence}, ...]
    """
    if len(discoveries) < 3:
        return []

    # 取 top 因子
    sorted_disc = sorted(discoveries, key=lambda d: abs(d.get("icir", 0)), reverse=True)[:8]

    disc_blocks = []
    for d in sorted_disc:
        disc_blocks.append(
            f"- **{d['name']}** (ICIR={d.get('icir', 0):.3f}): "
            f"`{d.get('expr', '')[:100]}`\n"
            f"  假说: {d.get('hypothesis', '')[:80]}"
        )

    prompt = f"""你是一位资深量化研究员。以下是历史挖掘中表现最好的因子，请**重组成功模式**创造新因子。

## 可用字段 (Qlib 日频)
$open, $high, $low, $close, $volume, $amount, $turn, $pe_ttm, $pb, $total_mv, $circ_mv

## 可用算子
- 时序: Ref(x,N), Mean(x,N), Std(x,N), Sum(x,N), Delta(x,N), Min(x,N), Max(x,N), Slope(x,N), Rsquare(x,N)
- 截面: Rank(x,N)
- 运算: Abs(x), Log(x), Sign(x), Power(x,N), Div(x,y), Greater(x,y), Less(x,y), If(cond,x,y)
- 统计: Corr(x,y,N), Cov(x,y,N) — 要求 x,y 同源

## 约束
1. 除零保护: 分母加 1e-8
2. 名称全大写下划线，以 CX_ 开头
3. 表达式长度 ≤ 200 字符, 嵌套 ≤ 5 层, 字段 ≤ 6 种

## 历史最佳因子
{chr(10).join(disc_blocks)}

## 重组策略
- 组合 A 的时间结构 + B 的截面逻辑
- 用 A 的信号源替换 B 中的字段
- 将成功的窗口参数迁移到不同的算子组合
- 融合不同类别因子的信号 (如波动率 + 动量)

## Agent 记忆
{context[:2000]}

请设计 {batch_size} 个重组因子。严格输出 JSON 数组:
```json
[
  {{
    "name": "CX_FACTOR_NAME",
    "expr": "重组后的 Qlib 表达式",
    "hypothesis": "重组逻辑: 来自 A 的 xxx + B 的 yyy",
    "category": "volatility/momentum/liquidity/value/reversal/microstructure",
    "confidence": 0.6
  }}
]
```"""

    return _call_claude_for_hypotheses(prompt, "[crossover]")
