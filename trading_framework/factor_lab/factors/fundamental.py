"""基本面因子 — PE/PB 及其衍生因子

依赖字段 (通过 Tushare daily_basic 注入):
- pe_ttm: 滚动市盈率
- pb: 市净率
- ps_ttm: 滚动市销率
- total_mv: 总市值
- circ_mv: 流通市值
- turnover_rate: 换手率 (Tushare)
"""
from .registry import FactorMeta, FactorCategory, REGISTRY

C = FactorCategory.VALUATION
F = FactorCategory.FUNDAMENTAL


def _val(name, expr, desc="", req=None, cat=C):
    return FactorMeta(
        name=name, expr=expr, category=cat,
        description=desc,
        required_fields=req or [],
    )


# ============================================================
# 估值因子 — 原始 + 衍生
# ============================================================
VALUATION_FACTORS = [
    # 原始估值
    _val("PE_TTM", "$pe_ttm", "滚动市盈率", ["pe_ttm"]),
    _val("PB", "$pb", "市净率", ["pb"]),
    _val("PS_TTM", "$ps_ttm", "滚动市销率", ["ps_ttm"]),

    # 倒数形式 (EP/BP 更适合排序)
    _val("EP", "1 / ($pe_ttm + 1e-8)", "盈利收益率 (1/PE)", ["pe_ttm"]),
    _val("BP", "1 / ($pb + 1e-8)", "账面价值率 (1/PB)", ["pb"]),
    _val("SP", "1 / ($ps_ttm + 1e-8)", "销售收益率 (1/PS)", ["ps_ttm"]),

    # 市值
    _val("LOG_MV", "Log($total_mv + 1)", "对数总市值", ["total_mv"]),
    _val("LOG_CIRC_MV", "Log($circ_mv + 1)", "对数流通市值", ["circ_mv"]),
    _val("MV_RATIO", "$circ_mv / ($total_mv + 1e-8)", "流通/总市值比", ["total_mv", "circ_mv"]),

    # 估值动态变化
    _val("PE_MA20", "Mean($pe_ttm, 20)", "20日PE均值", ["pe_ttm"]),
    _val("PE_STD20", "Std($pe_ttm, 20)", "20日PE标准差", ["pe_ttm"]),
    _val("PE_ZSCORE", "($pe_ttm - Mean($pe_ttm, 60)) / (Std($pe_ttm, 60) + 1e-8)",
         "PE 60日Z-score (估值分位)", ["pe_ttm"]),
    _val("PB_ZSCORE", "($pb - Mean($pb, 60)) / (Std($pb, 60) + 1e-8)",
         "PB 60日Z-score", ["pb"]),

    # 估值变化率
    _val("PE_CHANGE_20", "$pe_ttm / (Ref($pe_ttm, 20) + 1e-8) - 1",
         "20日PE变化率", ["pe_ttm"]),
    _val("PB_CHANGE_20", "$pb / (Ref($pb, 20) + 1e-8) - 1",
         "20日PB变化率", ["pb"]),

    # 估值-价格背离
    _val("PE_PRICE_CORR", "Corr($pe_ttm, $close, 20)",
         "20日PE与价格相关性", ["pe_ttm", "close"]),
]

# ============================================================
# 市值因子 — 小市值效应
# ============================================================
MARKET_CAP_FACTORS = [
    _val("MV_RANK", "Rank($total_mv, 20)", "市值20日滚动排名", ["total_mv"]),
    _val("NEG_LOG_MV", "-1 * Log($total_mv + 1)", "负对数市值 (小市值因子)", ["total_mv"]),
    _val("MV_CHANGE_20", "$total_mv / (Ref($total_mv, 20) + 1e-8) - 1",
         "20日市值变化率", ["total_mv"]),
]

# ============================================================
# 估值+量价交叉
# ============================================================
CROSS_FACTORS = [
    _val("EP_MOM20", "(1/($pe_ttm+1e-8)) * ($close/Ref($close,20)-1)",
         "EP × 20日动量 (价值动量)", ["pe_ttm", "close"]),
    _val("BP_TURN", "(1/($pb+1e-8)) * $turn",
         "BP × 换手率", ["pb", "turn"]),
    _val("MV_VOL", "Log($total_mv+1) * Std($close/Ref($close,1)-1, 20)",
         "市值 × 波动率", ["total_mv", "close"]),
]


def register_all():
    """注册所有基本面因子"""
    all_factors = VALUATION_FACTORS + MARKET_CAP_FACTORS + CROSS_FACTORS
    REGISTRY.register_many(all_factors)
    return len(all_factors)


def get_all_names() -> list[str]:
    all_factors = VALUATION_FACTORS + MARKET_CAP_FACTORS + CROSS_FACTORS
    return [f.name for f in all_factors]


def _check_fields_available(required_fields: list[str]) -> bool:
    """检查 Qlib bin 数据中是否存在所需字段"""
    import os
    from pathlib import Path
    data_dir = Path(os.path.expanduser('~/.qlib/qlib_data/cn_data_bs/features'))
    if not data_dir.exists():
        return False
    # 检查第一个股票目录
    for stock_dir in data_dir.iterdir():
        if stock_dir.is_dir() and stock_dir.name.startswith('s'):
            for field in required_fields:
                if field in ('close', 'open', 'high', 'low', 'volume',
                             'amount', 'turn', 'pctchg'):
                    continue  # 基础字段肯定存在
                bin_file = stock_dir / f'{field}.day.bin'
                if not bin_file.exists():
                    return False
            return True
    return False


_BASE_FIELDS = ('close', 'open', 'high', 'low', 'volume',
                'amount', 'turn', 'pctchg', 'isst')


def _field_available(field: str) -> bool:
    """单个字段是否存在于 Qlib bin 数据中"""
    return _check_fields_available([field])


def get_all_exprs(verbose: bool = False) -> list[tuple[str, str]]:
    """返回可用的基本面因子表达式

    按【单个字段】筛选，而不是全有或全无。

    2026-09-05 修复: 原实现只要有任一字段缺失就 `return []`，整批 22 个
    因子一起消失。后果是 alpha158_val 名义上是"alpha158_ext + 估值/基本面"，
    实测 188 个因子 100% 是量价 —— 基本面占比 0%，而 preset 文档和名字
    都在宣称含基本面。典型的"看起来有、实则没有"。

    BaoStock 能提供 pe_ttm / total_mv，pb / ps_ttm / circ_mv 需要 Tushare。
    逐字段筛选后，只注入 BaoStock 数据也能用上其中一部分因子。
    """
    all_factors = VALUATION_FACTORS + MARKET_CAP_FACTORS + CROSS_FACTORS

    needed = set()
    for f in all_factors:
        for rf in f.required_fields:
            if rf.lower() not in _BASE_FIELDS:
                needed.add(rf)

    avail = {fld for fld in needed if _field_available(fld)}
    missing = needed - avail

    out = []
    dropped = []
    for f in all_factors:
        req = {rf for rf in f.required_fields if rf.lower() not in _BASE_FIELDS}
        if req <= avail:
            out.append((f.name, f.expr))
        else:
            dropped.append((f.name, sorted(req - avail)))

    if verbose or (missing and not out):
        print(f"[fundamental] 可用字段 {sorted(avail) or '无'} | "
              f"缺失 {sorted(missing) or '无'}")
        print(f"[fundamental] 启用 {len(out)}/{len(all_factors)} 个因子")
        if dropped and verbose:
            for n, m in dropped[:10]:
                print(f"    跳过 {n:22s} (缺 {','.join(m)})")
    return out
