"""因子组合预设 — 预定义的因子集合

预设:
- alpha158:     原始 Alpha158 (158因子)
- alpha158_ext: Alpha158 + 扩展量价因子 (~208因子)
- alpha158_val: alpha158_ext + 估值/基本面因子 (~231因子)
- full:         alpha158_val + 资金流/北向/融资融券 (~261因子)
"""
from . import alpha158_ext, fundamental, money_flow, mined
from .custom_handler import build_handler_from_exprs, get_alpha158_feature_count

# 去冗余后保留的因子名单 (corr > 0.7 阈值)
_SELECTED_FACTORS_07 = {
    "VWAP", "VWAP_RATIO", "VWAP_MA20_RATIO",
    "TURN_MA10", "TURN_MA60", "TURN_RATIO_5_20",
    "ATR_14", "INTRADAY_RANGE_MA5", "VOL_RATIO_5_20", "HIGH_LOW_CORR_10",
    "PV_CORR_10", "PV_CORR_20", "PV_CORR_CHANGE", "AMT_PRICE_CORR_10", "AMT_SURPRISE",
    "MOM_SKIP1_5", "MOM_SKIP1_10", "MOM_3M_1M", "REVERSION_20",
    "WQ_ALPHA6", "WQ_ALPHA12", "WQ_ALPHA15", "WQ_ALPHA22", "WQ_ALPHA26",
    "WQ_ALPHA28", "WQ_ALPHA38", "WQ_ALPHA41", "WQ_ALPHA68", "WQ_ALPHA73", "WQ_ALPHA84",
}


def _get_selected_exprs():
    """获取去冗余后的因子表达式"""
    return [(n, e) for n, e in alpha158_ext.get_all_exprs() if n in _SELECTED_FACTORS_07]


# Phase 3 全因子去冗余后保留的因子名单 (初始=全部，Step 6 更新)
_FULL_SELECTED_FACTORS = None  # None 表示尚未做去冗余，回退到 full


def _get_full_selected_exprs():
    """获取全因子去冗余后的表达式"""
    if _FULL_SELECTED_FACTORS is None:
        # 尚未做去冗余，回退到 full 的全量
        return (
            _get_selected_exprs() +
            fundamental.get_all_exprs() +
            money_flow.get_safe_exprs()
        )
    all_exprs = (
        alpha158_ext.get_all_exprs() +
        fundamental.get_all_exprs() +
        money_flow.get_safe_exprs()
    )
    return [(n, e) for n, e in all_exprs if n in _FULL_SELECTED_FACTORS]


def update_full_selected(factor_names: set[str]):
    """更新 full_selected 的因子名单 (由 Step 6 调用)"""
    global _FULL_SELECTED_FACTORS
    _FULL_SELECTED_FACTORS = factor_names


# 预设定义
FACTOR_PRESETS = {
    "alpha158": {
        "description": "原始 Alpha158 (158因子)",
        "include_alpha158": True,
        "extra_factors": [],
    },
    "alpha158_ext": {
        "description": "Alpha158 + 扩展量价因子 (~208因子)",
        "include_alpha158": True,
        "extra_factors": lambda: alpha158_ext.get_all_exprs(),
    },
    "alpha158_selected": {
        "description": "Alpha158 + 去冗余扩展因子 (~188因子)",
        "include_alpha158": True,
        "extra_factors": _get_selected_exprs,
    },
    "alpha158_val": {
        "description": "Alpha158 + 去冗余量价 + 基本面 (~210因子)",
        "include_alpha158": True,
        "extra_factors": lambda: _get_selected_exprs() + fundamental.get_all_exprs(),
    },
    "full": {
        "description": "精选量价 + 基本面 + 资金流 (~231因子)",
        "include_alpha158": True,
        "extra_factors": lambda: (
            _get_selected_exprs() +
            fundamental.get_all_exprs() +
            money_flow.get_safe_exprs()
        ),
    },
    "full_selected": {
        "description": "全因子去冗余后最优子集",
        "include_alpha158": True,
        "extra_factors": lambda: _get_full_selected_exprs(),
    },
    "alpha158_val_mined": {
        "description": "alpha158_val + 挖掘因子",
        "include_alpha158": True,
        "extra_factors": lambda: _get_selected_exprs() + fundamental.get_all_exprs() + mined.get_all_exprs(),
    },
}


def list_presets() -> dict[str, str]:
    """列出所有可用预设及说明"""
    return {k: v["description"] for k, v in FACTOR_PRESETS.items()}


def get_preset_factor_count(preset_name: str) -> int:
    """获取预设的因子总数"""
    if preset_name not in FACTOR_PRESETS:
        raise ValueError(f"未知预设: {preset_name}, 可选: {list(FACTOR_PRESETS.keys())}")

    preset = FACTOR_PRESETS[preset_name]
    base = get_alpha158_feature_count() if preset["include_alpha158"] else 0
    extra = preset["extra_factors"]
    if callable(extra):
        extra = extra()
    return base + len(extra)


def build_handler(preset_name: str, start_time: str, end_time: str,
                  fit_start_time: str, fit_end_time: str,
                  instruments: str = "csi300"):
    """根据预设名构建 DataHandler

    Args:
        preset_name: 预设名 (alpha158, alpha158_ext, alpha158_val, full)
        start_time, end_time: 数据时间范围
        fit_start_time, fit_end_time: 标准化拟合范围
        instruments: 股票池

    Returns:
        DataHandlerLP 实例
    """
    if preset_name not in FACTOR_PRESETS:
        raise ValueError(f"未知预设: {preset_name}, 可选: {list(FACTOR_PRESETS.keys())}")

    preset = FACTOR_PRESETS[preset_name]

    # alpha158 直接用原始 handler (保持回归兼容)
    if preset_name == "alpha158":
        from qlib.contrib.data.handler import Alpha158
        return Alpha158(
            start_time=start_time,
            end_time=end_time,
            fit_start_time=fit_start_time,
            fit_end_time=fit_end_time,
            instruments=instruments,
        )

    # 其他预设: Alpha158 + 扩展因子
    extra = preset["extra_factors"]
    if callable(extra):
        extra = extra()

    handler, _ = build_handler_from_exprs(
        factor_exprs=extra,
        start_time=start_time,
        end_time=end_time,
        fit_start_time=fit_start_time,
        fit_end_time=fit_end_time,
        instruments=instruments,
        include_alpha158=preset["include_alpha158"],
    )
    return handler
