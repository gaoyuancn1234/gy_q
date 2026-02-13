"""扩展量价因子 — 在 Alpha158 基础上新增 ~50 个因子

依赖字段: open, high, low, close, volume, amount, turn, pctChg
其中 amount/turn/pctChg 需要先通过 data_setup.py 重新下载

因子分组:
1. VWAP 系列 (5)
2. 换手率系列 (8)
3. 波动率增强 (8)
4. 量价背离 (6)
5. 动量增强 (8)
6. Alpha101 精选 (15)
"""
from .registry import FactorMeta, FactorCategory, REGISTRY

C = FactorCategory.PRICE_VOLUME
_BASE = ["open", "high", "low", "close", "volume"]
_EXT = ["amount", "turn", "pctChg"]


def _pv(name, expr, desc="", req=None):
    """快捷创建量价因子"""
    return FactorMeta(
        name=name, expr=expr, category=C,
        description=desc,
        required_fields=req or _BASE,
    )


# ============================================================
# 1. VWAP 系列 — 日内成交均价及其衍生
# ============================================================
VWAP_FACTORS = [
    _pv("VWAP", "Div($amount, $volume+1)",
        "日内成交均价 (amount/volume)", _BASE + ["amount"]),
    _pv("VWAP_RATIO", "Div($amount, $volume+1) / $close",
        "VWAP/收盘价比值", _BASE + ["amount"]),
    _pv("VWAP_MA5_RATIO", "Mean(Div($amount, $volume+1), 5) / $close",
        "5日VWAP均值/收盘价", _BASE + ["amount"]),
    _pv("VWAP_MA10_RATIO", "Mean(Div($amount, $volume+1), 10) / $close",
        "10日VWAP均值/收盘价", _BASE + ["amount"]),
    _pv("VWAP_MA20_RATIO", "Mean(Div($amount, $volume+1), 20) / $close",
        "20日VWAP均值/收盘价", _BASE + ["amount"]),
]

# ============================================================
# 2. 换手率系列 — 流动性与交易活跃度
# ============================================================
TURNOVER_FACTORS = [
    _pv("TURN_MA5", "Mean($turn, 5)",
        "5日平均换手率", _BASE + ["turn"]),
    _pv("TURN_MA10", "Mean($turn, 10)",
        "10日平均换手率", _BASE + ["turn"]),
    _pv("TURN_MA20", "Mean($turn, 20)",
        "20日平均换手率", _BASE + ["turn"]),
    _pv("TURN_MA60", "Mean($turn, 60)",
        "60日平均换手率", _BASE + ["turn"]),
    _pv("TURN_STD20", "Std($turn, 20)",
        "20日换手率标准差", _BASE + ["turn"]),
    _pv("TURN_SURGE", "$turn / (Mean($turn, 20) + 1e-8)",
        "换手率突增比 (当日/20日均值)", _BASE + ["turn"]),
    _pv("TURN_RATIO_5_20", "Mean($turn, 5) / (Mean($turn, 20) + 1e-8)",
        "短期/长期换手率比", _BASE + ["turn"]),
    _pv("TURN_RANK_5", "Rank(Mean($turn, 5), 20)",
        "5日均换手率20日时序排名", _BASE + ["turn"]),
]

# ============================================================
# 3. 波动率增强 — ATR, Garman-Klass, 振幅
# ============================================================
VOLATILITY_FACTORS = [
    _pv("ATR_14", "Mean(Greater(Greater($high-$low, Abs($high-Ref($close,1))), Abs($low-Ref($close,1))), 14)",
        "14日 ATR"),
    _pv("ATR_RATIO", "Mean(Greater(Greater($high-$low, Abs($high-Ref($close,1))), Abs($low-Ref($close,1))), 14) / ($close + 1e-8)",
        "ATR / 收盘价 (标准化波动率)"),
    _pv("GK_VOL_20", "Std(Log($high/$low+1e-8), 20)",
        "20日 Garman-Klass 近似波动率 (log(H/L) 的标准差)"),
    _pv("INTRADAY_RANGE", "($high - $low) / ($close + 1e-8)",
        "日内振幅"),
    _pv("INTRADAY_RANGE_MA5", "Mean(($high - $low) / ($close + 1e-8), 5)",
        "5日平均振幅"),
    _pv("VOL_RATIO_5_20", "Std($close/Ref($close,1)-1, 5) / (Std($close/Ref($close,1)-1, 20) + 1e-8)",
        "短期/长期波动率比"),
    _pv("HIGH_LOW_CORR_10", "Corr($high, $low, 10)",
        "10日高低价相关性"),
    _pv("CLOSE_RANGE_POS", "($close - $low) / ($high - $low + 1e-8)",
        "收盘价在日内范围中的位置"),
]

# ============================================================
# 4. 量价背离 — 价量相关性变化
# ============================================================
DIVERGENCE_FACTORS = [
    _pv("PV_CORR_10", "Corr($close, $volume, 10)",
        "10日价量相关性"),
    _pv("PV_CORR_20", "Corr($close, $volume, 20)",
        "20日价量相关性"),
    _pv("PV_CORR_CHANGE", "Corr($close, $volume, 10) - Corr($close, $volume, 20)",
        "价量相关性变化 (短-长)"),
    _pv("AMT_PRICE_CORR_10", "Corr(Div($amount, $volume+1), $close, 10)",
        "10日 VWAP-收盘价相关性", _BASE + ["amount"]),
    _pv("VOLUME_SURPRISE", "$volume / (Mean($volume, 20) + 1)",
        "成交量惊喜 (当日/20日均量)"),
    _pv("AMT_SURPRISE", "$amount / (Mean($amount, 20) + 1)",
        "成交额惊喜 (当日/20日均额)", _BASE + ["amount"]),
]

# ============================================================
# 5. 动量增强 — skip-1 动量, 多周期, 反转
# ============================================================
MOMENTUM_FACTORS = [
    _pv("MOM_SKIP1_5", "Ref($close, 1) / Ref($close, 6) - 1",
        "跳过1天的5日动量 (避免短期反转)"),
    _pv("MOM_SKIP1_10", "Ref($close, 1) / Ref($close, 11) - 1",
        "跳过1天的10日动量"),
    _pv("MOM_SKIP1_20", "Ref($close, 1) / Ref($close, 21) - 1",
        "跳过1天的20日动量"),
    _pv("MOM_3M_1M", "$close / Ref($close, 60) - $close / Ref($close, 20)",
        "3个月动量 - 1个月动量 (中期加速)"),
    _pv("REVERSION_5", "-1 * ($close / Ref($close, 5) - 1)",
        "5日反转因子"),
    _pv("REVERSION_20", "-1 * ($close / Ref($close, 20) - 1)",
        "20日反转因子"),
    _pv("UP_RATIO_20", "Mean(If($close>Ref($close,1), 1, 0), 20)",
        "20日上涨比例"),
    _pv("UP_VOL_RATIO_20", "Mean(If($close>Ref($close,1), $volume, 0), 20) / (Mean($volume, 20) + 1)",
        "20日上涨日成交量占比"),
]

# ============================================================
# 6. Alpha101 精选 — WorldQuant 101 中适合 Qlib 表达式的因子
# ============================================================
ALPHA101_FACTORS = [
    # Alpha#6: 成交额与开盘价的相关性
    _pv("WQ_ALPHA6", "-1 * Corr($open, $amount, 10)",
        "WQ Alpha#6: -corr(open, amount, 10)", _BASE + ["amount"]),
    # Alpha#12: 成交量方向 × 价格变化
    _pv("WQ_ALPHA12", "Sign(Ref($volume, 1) - $volume) * ($close - Ref($close, 1))",
        "WQ Alpha#12"),
    # Alpha#15: 高低价时序排名的相关性
    _pv("WQ_ALPHA15", "-1 * Corr(Rank($high, 20), Rank($volume, 20), 3)",
        "WQ Alpha#15"),
    # Alpha#22: 高价与收盘价关系
    _pv("WQ_ALPHA22", "-1 * (Corr($high, $volume, 5) * Std($close, 5))",
        "WQ Alpha#22"),
    # Alpha#26: 成交额排名移动相关
    _pv("WQ_ALPHA26", "-1 * Corr(Rank($amount, 20), Rank($volume, 20), 5)",
        "WQ Alpha#26: 成交额排名-量排名相关性", _BASE + ["amount"]),
    # Alpha#28
    _pv("WQ_ALPHA28", "Corr(Mean($volume, 20), $low, 5)",
        "WQ Alpha#28"),
    # Alpha#33: 开盘收盘差异时序排名
    _pv("WQ_ALPHA33", "Rank(-1 + ($open / $close), 20)",
        "WQ Alpha#33"),
    # Alpha#38
    _pv("WQ_ALPHA38", "-1 * Rank(($close - $open) / ($high - $low + 1e-8), 20)",
        "WQ Alpha#38"),
    # Alpha#41: 高低差异排名
    _pv("WQ_ALPHA41", "Rank(($high - $low) / (Mean($close, 5) + 1e-8), 20)",
        "WQ Alpha#41"),
    # Alpha#45: 收盘价与成交量排名相关
    _pv("WQ_ALPHA45", "-1 * Corr($close, Rank($volume, 20), 2) * Rank(Mean($close, 5) - $close, 20)",
        "WQ Alpha#45"),
    # Alpha#54: 开盘与收盘的负相关
    _pv("WQ_ALPHA54", "-1 * ($low - $close) * ($open - $close) / ($high - $close + 1e-8)",
        "WQ Alpha#54"),
    # Alpha#68: 高价排名与成交量均值排名的相关性排名
    _pv("WQ_ALPHA68", "Rank(Corr(Rank($high, 20), Rank(Mean($volume, 15), 20), 9), 20)",
        "WQ Alpha#68"),
    # Alpha#73: 价量相关排名 × 价格/开盘排名
    _pv("WQ_ALPHA73", "-1 * Rank(Corr($close, $volume, 10), 20) * Rank($close / $open, 20)",
        "WQ Alpha#73"),
    # Alpha#84: 价量动量排名交叉
    _pv("WQ_ALPHA84", "Rank(Ref($close, 5) - $close, 20) * Rank(Ref($volume, 5) - $volume, 20)",
        "WQ Alpha#84"),
    # Alpha#101: 开盘收盘与高低价关系
    _pv("WQ_ALPHA101", "($close - $open) / (($high - $low) + 1e-8)",
        "WQ Alpha#101"),
]


def register_all():
    """注册所有扩展量价因子"""
    all_factors = (
        VWAP_FACTORS + TURNOVER_FACTORS + VOLATILITY_FACTORS +
        DIVERGENCE_FACTORS + MOMENTUM_FACTORS + ALPHA101_FACTORS
    )
    REGISTRY.register_many(all_factors)
    return len(all_factors)


# 获取所有扩展因子的名称列表
def get_all_names() -> list[str]:
    all_factors = (
        VWAP_FACTORS + TURNOVER_FACTORS + VOLATILITY_FACTORS +
        DIVERGENCE_FACTORS + MOMENTUM_FACTORS + ALPHA101_FACTORS
    )
    return [f.name for f in all_factors]


def get_all_exprs() -> list[tuple[str, str]]:
    """返回 [(name, expr), ...] 用于直接构建 DataLoader"""
    all_factors = (
        VWAP_FACTORS + TURNOVER_FACTORS + VOLATILITY_FACTORS +
        DIVERGENCE_FACTORS + MOMENTUM_FACTORS + ALPHA101_FACTORS
    )
    return [(f.name, f.expr) for f in all_factors]
