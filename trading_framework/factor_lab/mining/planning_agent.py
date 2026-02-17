"""搜索方向规划 — Claude 生成多样化挖掘方向

用于演化式挖掘 (run_evolved_mining) 的第一步:
Claude 分析现有因子库状况 + 近期表现 + 失败方向，生成 N 个多样化搜索方向。
"""
import json
import re
import subprocess
import time
from pathlib import Path

WORK_DIR = Path(__file__).resolve().parent.parent.parent.parent  # repo root

# 复用 quanta 的 Claude 配置
from factor_lab.quanta.config import (
    CLAUDE_CLI, CLAUDE_TIMEOUT, RETRY_WAIT, MAX_RETRY,
    BASE_FEATURES, get_claude_env,
)


def plan_directions(context: str, n_directions: int = 5,
                    explored_summary: str = "") -> list[dict]:
    """Claude 规划搜索方向

    Args:
        context: Agent 记忆文本 (AGENT_CONTEXT.md)
        n_directions: 方向数量
        explored_summary: 已探索方向摘要 (来自 DirectionRegistry)

    Returns:
        [{direction, hypothesis, mechanism, time_scale}, ...]
    """
    features_str = ", ".join(BASE_FEATURES)

    explored_section = ""
    if explored_summary and explored_summary != "(无已探索方向)":
        explored_section = f"""
## 已探索方向 (避免重复!)
以下方向已经被探索过，请勿与它们重复:
{explored_summary}

"""

    prompt = f"""你是一位资深量化研究主管。请为 A 股 CSI300 因子挖掘规划 {n_directions} 个**多样化的搜索方向**。

## 可用基础特征 (Qlib 日频)
{features_str}
(VWAP 需写为 Div($amount, $volume + 1e-8))

## 当前因子库概况
{context[:3000]}
{explored_section}
## 规划要求
1. 每个方向必须是一个**具体的投资假说**，包含:
   - 明确的经济学/市场微结构逻辑
   - 预期捕捉的市场现象 (动量/反转/流动性/波动率/regime...)
   - 适用的时间尺度 (短期 1-5日 / 中期 10-20日 / 长期 40-60日)

2. **多样性**: {n_directions} 个方向必须覆盖:
   - 至少 3 种不同的信号源 (价格结构/量能/量价联合/波动率/估值)
   - 至少 3 种不同的市场机制 (动量/反转/regime/流动性/隔夜/机构)
   - 至少 2 种不同的时间尺度

3. **可操作性**: 每个方向应该能直接指导因子表达式设计

4. 避免与已有因子库重复 (参见上方概况)

5. **避免与已探索方向重复** (参见上方已探索方向列表)

请输出 JSON 数组:
```json
[
  {{
    "direction": "方向简称 (如: 日内波动非对称性)",
    "hypothesis": "详细假说 (2-3句话, 含经济逻辑)",
    "mechanism": "市场机制 (动量/反转/regime/流动性/隔夜/机构行为)",
    "time_scale": "时间尺度 (短期/中期/长期/多尺度)"
  }},
  ...
]
```"""

    for attempt in range(1, MAX_RETRY + 1):
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
            output = result.stdout.strip()
            if output:
                parsed = _parse_directions(output)
                if parsed:
                    return parsed[:n_directions]
                print(f"  [planning] 解析失败 (attempt {attempt}/{MAX_RETRY})")
            else:
                print(f"  [planning] Claude 返回空输出 (attempt {attempt}/{MAX_RETRY})")

        except subprocess.TimeoutExpired:
            print(f"  [planning] Claude CLI 超时 (attempt {attempt}/{MAX_RETRY})")
        except Exception as e:
            print(f"  [planning] Claude CLI 调用失败: {e} (attempt {attempt}/{MAX_RETRY})")

        if attempt < MAX_RETRY:
            time.sleep(RETRY_WAIT)

    return _fallback_directions(n_directions)


def _parse_directions(output: str) -> list[dict]:
    """从 Claude 输出解析方向列表"""
    for attempt in [
        lambda: json.loads(output),
        lambda: json.loads(re.search(r'```(?:json)?\s*\n(\[[\s\S]*?\])\s*\n```', output).group(1)),
        lambda: json.loads(re.search(r'\[[\s\S]*?\]', output).group()),
    ]:
        try:
            data = attempt()
            if isinstance(data, list):
                valid = []
                for item in data:
                    if isinstance(item, dict) and 'direction' in item and 'hypothesis' in item:
                        item.setdefault('mechanism', '动量')
                        item.setdefault('time_scale', '中期')
                        valid.append(item)
                return valid
        except (json.JSONDecodeError, AttributeError, TypeError):
            continue
    print(f"  [planning] 无法解析方向 (输出长度={len(output)})")
    return []


def _fallback_directions(n: int) -> list[dict]:
    """Claude 调用失败时的备选方向"""
    defaults = [
        {
            "direction": "日内波动非对称性",
            "hypothesis": "上涨日与下跌日的日内波动结构不同。上涨日high-open较大，下跌日open-low较大。这种非对称性反映了市场情绪和机构行为模式。",
            "mechanism": "regime",
            "time_scale": "短期",
        },
        {
            "direction": "量价时序背离",
            "hypothesis": "价格创新高但成交量持续萎缩，或放量滞涨，预示趋势即将反转。量价背离的持续性越强，反转信号越可靠。",
            "mechanism": "反转",
            "time_scale": "中期",
        },
        {
            "direction": "波动率聚集效应",
            "hypothesis": "波动率具有聚集性——高波动后往往持续高波动。短期波动率与长期波动率的比值反映了当前波动率regime，可用于预测风险调整后收益。",
            "mechanism": "regime",
            "time_scale": "多尺度",
        },
        {
            "direction": "换手率异常检测",
            "hypothesis": "换手率突然偏离历史均值(2倍标准差以上)预示信息事件发生。异常换手率后的价格动量在5-10日内具有显著预测力。",
            "mechanism": "流动性",
            "time_scale": "短期",
        },
        {
            "direction": "动量衰减加速",
            "hypothesis": "动量因子的二阶导(加速度)比一阶导(速度)有更好的预测力。动量加速时趋势延续概率高，动量减速时反转概率高。",
            "mechanism": "动量",
            "time_scale": "中期",
        },
        {
            "direction": "隔夜收益异象",
            "hypothesis": "开盘价相对前收盘价的跳空方向和幅度蕴含信息。持续正向跳空反映机构增持，反向跳空反映抛压。",
            "mechanism": "隔夜",
            "time_scale": "短期",
        },
        {
            "direction": "估值动量交叉",
            "hypothesis": "PE/PB的变化率(估值动量)与价格动量的交叉信号有效。估值改善+价格上涨比单纯价格动量有更强的持续性。",
            "mechanism": "动量",
            "time_scale": "长期",
        },
    ]
    return defaults[:n]
