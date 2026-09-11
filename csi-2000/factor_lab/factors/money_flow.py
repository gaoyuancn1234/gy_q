"""资金流因子 — 主力资金 / 北向资金 / 融资融券

依赖字段:
- main_net_inflow: 主力净流入 (AKShare)
- super_large_net: 超大单净流入
- large_net: 大单净流入
- medium_net: 中单净流入
- small_net: 小单净流入
- north_money: 北向资金总额 (Tushare, 市场级)
- margin_balance: 融资余额 (AKShare)
- short_balance: 融券余额 (AKShare)
"""
from .registry import FactorMeta, FactorCategory, REGISTRY

C = FactorCategory.MONEY_FLOW


def _mf(name, expr, desc="", req=None):
    return FactorMeta(
        name=name, expr=expr, category=C,
        description=desc,
        required_fields=req or [],
    )


# ============================================================
# 主力资金流向
# ============================================================
MAIN_FLOW_FACTORS = [
    _mf("MAIN_NET_5", "Mean($main_net_inflow, 5)",
        "5日主力净流入均值", ["main_net_inflow"]),
    _mf("MAIN_NET_10", "Mean($main_net_inflow, 10)",
        "10日主力净流入均值", ["main_net_inflow"]),
    _mf("MAIN_NET_20", "Mean($main_net_inflow, 20)",
        "20日主力净流入均值", ["main_net_inflow"]),
    _mf("MAIN_NET_ACC5", "Sum($main_net_inflow, 5)",
        "5日主力净流入累计", ["main_net_inflow"]),
    _mf("MAIN_NET_ACC20", "Sum($main_net_inflow, 20)",
        "20日主力净流入累计", ["main_net_inflow"]),
    _mf("MAIN_NET_CHANGE", "Mean($main_net_inflow, 5) - Mean($main_net_inflow, 20)",
        "主力净流入短-长差异", ["main_net_inflow"]),
]

# ============================================================
# 大单/超大单/小单分析
# ============================================================
ORDER_SIZE_FACTORS = [
    _mf("BIG_ORDER_RATIO", "($super_large_net + $large_net) / (Abs($super_large_net) + Abs($large_net) + Abs($medium_net) + Abs($small_net) + 1e-8)",
        "大单占比 (大单+超大单净额/总净额)", ["super_large_net", "large_net", "medium_net", "small_net"]),
    _mf("SUPER_LARGE_5", "Mean($super_large_net, 5)",
        "5日超大单净流入均值", ["super_large_net"]),
    _mf("SMALL_NET_5", "Mean($small_net, 5)",
        "5日小单净流入均值 (散户行为)", ["small_net"]),
    _mf("BIG_SMALL_RATIO", "Sum($super_large_net + $large_net, 5) / (Sum(Abs($small_net), 5) + 1e-8)",
        "5日大单/小单净流入比", ["super_large_net", "large_net", "small_net"]),
]

# ============================================================
# 北向资金 (市场级指标，所有股票共享)
# ============================================================
NORTH_MONEY_FACTORS = [
    _mf("NORTH_MA5", "Mean($north_money, 5)",
        "5日北向资金均值", ["north_money"]),
    _mf("NORTH_MA20", "Mean($north_money, 20)",
        "20日北向资金均值", ["north_money"]),
    _mf("NORTH_ACC5", "Sum($north_money, 5)",
        "5日北向资金累计", ["north_money"]),
    _mf("NORTH_CHANGE", "Mean($north_money, 5) - Mean($north_money, 20)",
        "北向资金短-长差异", ["north_money"]),
    _mf("NORTH_MOM", "$north_money / (Ref($north_money, 5) + 1e-8) - 1",
        "北向资金5日动量", ["north_money"]),
]

# ============================================================
# 融资融券
# ============================================================
MARGIN_FACTORS = [
    _mf("MARGIN_BAL", "$margin_balance",
        "融资余额", ["margin_balance"]),
    _mf("MARGIN_CHANGE_5", "$margin_balance / (Ref($margin_balance, 5) + 1e-8) - 1",
        "融资余额5日变化率", ["margin_balance"]),
    _mf("MARGIN_CHANGE_20", "$margin_balance / (Ref($margin_balance, 20) + 1e-8) - 1",
        "融资余额20日变化率", ["margin_balance"]),
    _mf("SHORT_BAL", "$short_balance",
        "融券余额", ["short_balance"]),
    _mf("MARGIN_SHORT_RATIO", "$margin_balance / ($short_balance + 1e-8)",
        "融资/融券余额比", ["margin_balance", "short_balance"]),
    _mf("NET_MARGIN", "$margin_balance - $short_balance",
        "融资融券净额", ["margin_balance", "short_balance"]),
]

# ============================================================
# 资金流 × 量价交叉
# ============================================================
CROSS_FACTORS = [
    _mf("MAIN_PRICE_CORR", "Corr($main_net_inflow, $close, 10)",
        "10日主力资金-价格相关性", ["main_net_inflow", "close"]),
    _mf("MAIN_VOL_CORR", "Corr($main_net_inflow, $volume, 10)",
        "10日主力资金-成交量相关性", ["main_net_inflow", "volume"]),
    _mf("MARGIN_PRICE_CORR", "Corr($margin_balance, $close, 20)",
        "20日融资余额-价格相关性", ["margin_balance", "close"]),
]


def register_all():
    """注册所有资金流因子"""
    all_factors = (MAIN_FLOW_FACTORS + ORDER_SIZE_FACTORS +
                   NORTH_MONEY_FACTORS + MARGIN_FACTORS + CROSS_FACTORS)
    REGISTRY.register_many(all_factors)
    return len(all_factors)


def get_all_names() -> list[str]:
    all_factors = (MAIN_FLOW_FACTORS + ORDER_SIZE_FACTORS +
                   NORTH_MONEY_FACTORS + MARGIN_FACTORS + CROSS_FACTORS)
    return [f.name for f in all_factors]


def get_all_exprs() -> list[tuple[str, str]]:
    """获取所有因子表达式（含交叉因子）"""
    all_factors = (MAIN_FLOW_FACTORS + ORDER_SIZE_FACTORS +
                   NORTH_MONEY_FACTORS + MARGIN_FACTORS + CROSS_FACTORS)
    return [(f.name, f.expr) for f in all_factors]


def get_safe_exprs() -> list[tuple[str, str]]:
    """获取安全因子表达式（排除交叉因子）

    交叉因子（如 Corr($main_net_inflow, $close, 10)）混合了
    不同日期范围的字段，Qlib Corr 操作符无法处理长度不一致的 bin 文件。
    仅在因子数据覆盖完整日期范围时才使用 get_all_exprs()。
    """
    safe_factors = (MAIN_FLOW_FACTORS + ORDER_SIZE_FACTORS +
                    NORTH_MONEY_FACTORS + MARGIN_FACTORS)
    return [(f.name, f.expr) for f in safe_factors]
