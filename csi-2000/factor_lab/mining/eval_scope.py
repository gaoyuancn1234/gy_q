"""挖掘评估口径。股票池和标签必须跟生产 yaml 走。

中证2000 上写死 csi300 / 隔日收益会挖到错池错标签。
"""
from __future__ import annotations


def mining_instruments() -> str:
    """当前 yaml 里的股票池。"""
    from qlib_paths import current_universe
    return current_universe()


def mining_label() -> str:
    """与生产一致的隔夜标签。"""
    from factor_lab.factors.open_source import OVN_LABEL
    return OVN_LABEL


# 股票池特征描述 —— 直接进 LLM 提示词。
# 2026-09-13: 此前 4 处提示词写死"A 股 CSI300"(hypothesis.py / planning_agent.py /
# idea_agent.py x2), dry-run 实测生成的假说全是"在CSI300中..."。
# 这不是措辞问题: 大盘与微盘的机制根本不同(流动性、散户占比、涨跌停动态、
# 指数纳入效应), LLM 会按大盘的逻辑设计因子, 然后在微盘上评估 —— 方向就错了。
_UNIVERSE_DESC = {
    "csi2000": (
        "A 股中证2000（微盘/小盘股，约 2000 只，日均成交额中位数量级在千万，"
        "散户参与度高、流动性薄、涨跌停与停牌频繁；"
        "主板 ±10%、创业板/科创板 ±20%、北交所 ±30%）"
    ),
    "csi300": "A 股 CSI300（大盘蓝筹，流动性充裕，机构主导）",
    "csi500": "A 股中证500（中盘）",
    "csi1000": "A 股中证1000（小盘）",
}


def universe_desc(universe: str | None = None) -> str:
    """给 LLM 的股票池描述。未知池子退回名字本身, 不编造特征。"""
    u = universe or mining_instruments()
    return _UNIVERSE_DESC.get(u, f"A 股 {u}")


def label_desc() -> str:
    """给 LLM 的预测目标描述 —— 标签变了提示词必须跟着变。"""
    from qlib_paths import CONFIG_FILE
    import yaml
    with open(CONFIG_FILE, encoding="utf-8") as f:
        preset = (yaml.safe_load(f) or {}).get("preset", "")
    if preset.startswith("alpha158_ovn"):
        return ("**隔夜收益**（T 日收盘买入 -> T+1 日开盘的涨跌幅）。"
                "注意预测目标不是日内收益, 也不是多日收益")
    return "次日收益"
