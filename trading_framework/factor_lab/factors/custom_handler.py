"""通用 DataHandlerLP 子类 — 支持动态因子集

根据因子注册表或预设构建 DataHandler，可直接传给 Qlib DatasetH。
"""
from qlib.data.dataset.handler import DataHandlerLP


def _get_alpha158_config() -> tuple[list[str], list[str]]:
    """获取 Alpha158 的因子表达式配置（不实例化 handler）

    Returns:
        (fields, names) 两个等长列表
    """
    from qlib.contrib.data.loader import Alpha158DL

    conf = {
        "kbar": {},
        "price": {
            "windows": [0],
            "feature": ["OPEN", "HIGH", "LOW", "VWAP"],
        },
        "rolling": {},
    }
    fields, names = Alpha158DL.get_feature_config(conf)
    return fields, names


def build_handler_from_exprs(
    factor_exprs: list[tuple[str, str]],
    start_time: str,
    end_time: str,
    fit_start_time: str,
    fit_end_time: str,
    instruments: str = "csi300",
    include_alpha158: bool = True,
) -> tuple[DataHandlerLP, int]:
    """根据因子表达式列表构建 DataHandler

    Returns:
        (handler, n_features) 元组
    """
    from qlib.contrib.data.handler import check_transform_proc

    if include_alpha158:
        alpha_fields, alpha_names = _get_alpha158_config()
        ext_fields = [expr for _, expr in factor_exprs]
        ext_names = [name for name, _ in factor_exprs]
        all_fields = list(alpha_fields) + ext_fields
        all_names = list(alpha_names) + ext_names
    else:
        all_fields = [expr for _, expr in factor_exprs]
        all_names = [name for name, _ in factor_exprs]

    # 与 Alpha158 一致的 processor 配置
    infer_processors = [
        {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
        {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
    ]
    learn_processors = [
        {"class": "DropnaLabel"},
        {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}},
    ]

    # 用 Qlib 内置的 check_transform_proc 注入 fit 时间
    infer_processors = check_transform_proc(infer_processors, fit_start_time, fit_end_time)
    learn_processors = check_transform_proc(learn_processors, fit_start_time, fit_end_time)

    handler = DataHandlerLP(
        instruments=instruments,
        start_time=start_time,
        end_time=end_time,
        data_loader={
            "class": "QlibDataLoader",
            "kwargs": {"config": {
                "feature": (all_fields, all_names),
                "label": (["Ref($close, -2)/Ref($close, -1) - 1"], ["LABEL0"]),
            }},
        },
        infer_processors=infer_processors,
        learn_processors=learn_processors,
        process_type=DataHandlerLP.PTYPE_A,
    )
    return handler, len(all_fields)


def get_alpha158_feature_count() -> int:
    """获取 Alpha158 因子数量（不实例化 handler）"""
    fields, _ = _get_alpha158_config()
    return len(fields)
