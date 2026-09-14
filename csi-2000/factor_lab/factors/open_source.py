"""开源量价因子 — 只用 OHLCV + amount, 不用 turn/基本面。

来源: WorldQuant 101 (Kakushadze 2016) 里能写成 Qlib 表达式的,
以及 Amihud / 隔夜缺口等公开定义。中证2000 的 turn 是 NaN, 全部跳过。
"""
from __future__ import annotations

from factor_lab.factors import alpha158_ext

# 隔夜标签, 与 presets._OVN_LABEL 保持一致。
OVN_LABEL = (
    "If($close/Ref($close,1)-1>0.095, "
    "($close-$close)/($close-$close), "
    "Ref($open,-1)/$close-1)"
)

# 额外开源因子。不要用 $turn。
_EXTRA = [
    ("OS_AMIHUD20",
     "Mean(Abs($close/Ref($close,1)-1)/($amount+1), 20)"),
    ("OS_ILLIQ",
     "Abs($close/Ref($close,1)-1)/($amount+1)"),
    ("OS_INTRADAY",
     "$close/$open-1"),
    ("OS_OVN_GAP",
     "$open/Ref($close,1)-1"),
    ("OS_CLOSE_POS",
     "($close-$low)/($high-$low+1e-8)"),
    ("OS_VOL_SHOCK",
     "$volume/(Mean($volume,20)+1)"),
    ("OS_AMT_SHOCK",
     "$amount/(Mean($amount,20)+1)"),
    ("OS_NEAR_LIMIT",
     "$close/Ref($close,1)-1"),
    ("OS_WQ44",
     "-1*Corr($high,Rank($volume,5),5)"),
    ("OS_WQ53",
     "-1*Delta((($close-$low)-($high-$close))"
     "/($close-$low+1e-8),9)"),
    ("OS_WQ101",
     "($close-$open)/(($high-$low)+1e-8)"),
    ("OS_REVERSAL1",
     "-1*($close/Ref($close,1)-1)"),
    ("OS_VOL_PRICE",
     "Corr($close,$volume,5)"),
]


def _needs_turn(expr: str) -> bool:
    """换手在中证2000 供给层是空的。"""
    return "$turn" in expr


def get_extra_exprs() -> list[tuple[str, str]]:
    """开源增量因子 (不含 Alpha158 本体)。"""
    seen = set()
    out: list[tuple[str, str]] = []
    for name, expr in alpha158_ext.get_all_exprs():
        if _needs_turn(expr) or name in seen:
            continue
        seen.add(name)
        out.append((name, expr))
    for name, expr in _EXTRA:
        if name in seen or _needs_turn(expr):
            continue
        seen.add(name)
        out.append((name, expr))
    return out


def get_all_exprs() -> list[tuple[str, str]]:
    """与 get_extra_exprs 相同, 给评价脚本用。"""
    return get_extra_exprs()
