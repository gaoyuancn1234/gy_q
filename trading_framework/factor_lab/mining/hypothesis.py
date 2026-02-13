"""Claude CLI 因子假说生成

调用 Claude CLI agent 生成新的量化因子假说。
"""
import json
import re
import subprocess
from pathlib import Path

WORK_DIR = Path(__file__).resolve().parent.parent.parent.parent  # repo root


def generate_hypotheses(context: str, existing_names: list[str],
                        batch_size: int = 10, focus: str = "") -> list[dict]:
    """调用 Claude CLI 生成因子假说

    Args:
        context: Agent 记忆文本 (AGENT_CONTEXT.md)
        existing_names: 已有因子名 (防重复)
        batch_size: 每批生成数量
        focus: 本次探索方向

    Returns:
        [{name, expr, hypothesis, category, confidence}, ...]
    """
    existing_str = ", ".join(existing_names[-100:]) if existing_names else "(无)"

    focus_section = ""
    if focus:
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
            '/usr/local/bin/claude',
            '--print',
            '--dangerously-skip-permissions',
            '--output-format', 'text',
            '-p', prompt,
        ]
        result = subprocess.run(
            cmd,
            capture_output=True, text=True,
            timeout=300,
            cwd=str(WORK_DIR),
        )
        output = result.stdout.strip()
        return _parse_hypotheses(output)

    except subprocess.TimeoutExpired:
        print("  [hypothesis] Claude CLI 超时 (300s)")
        return []
    except Exception as e:
        print(f"  [hypothesis] Claude CLI 调用失败: {e}")
        return []


def _parse_hypotheses(output: str) -> list[dict]:
    """从 Claude 输出中提取 JSON 数组"""
    # 尝试直接解析
    try:
        data = json.loads(output)
        if isinstance(data, list):
            return _validate_hypotheses(data)
    except json.JSONDecodeError:
        pass

    # 尝试从 markdown code block 中提取
    match = re.search(r'```(?:json)?\s*\n(\[[\s\S]*?\])\s*\n```', output)
    if match:
        try:
            data = json.loads(match.group(1))
            if isinstance(data, list):
                return _validate_hypotheses(data)
        except json.JSONDecodeError:
            pass

    # 尝试提取最大的 JSON 数组
    match = re.search(r'\[[\s\S]*\]', output)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, list):
                return _validate_hypotheses(data)
        except json.JSONDecodeError:
            pass

    print(f"  [hypothesis] 无法解析 Claude 输出 (长度={len(output)})")
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
