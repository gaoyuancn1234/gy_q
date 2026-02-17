"""被接受的挖掘因子

由 factor_miner --accept 命令自动写入。
不要手动编辑此文件。
"""

MINED_FACTORS = [
    ('QA_SELLPRESS_RATIO', 'Div(Sum(Mul($volume, Less($close, Ref($close, 1))), 10), Sum($volume, 10) + 1e-8)'),  # accepted from run_006
    ('QA_VOL_PRICE_DIVERGENCE_SLOPE', 'Sub(Div(Slope($volume, 10), Std($volume, 10) + 1e-8), Div(Slope($close, 10), Std($close, 10) + 1e-8))'),  # accepted from run_006
    ('QA_SHADOW_ASYMMETRY', 'Mean(Div($close - $low, $high - $low + 1e-8), 10)'),  # accepted from run_006
    ('QA_CHANNEL_TURN_MOMENTUM', 'Mul(Div(Sub($close, Min($low, 10)), Sub(Max($high, 10), Min($low, 10)) + 1e-8), Div(Mean($turn, 5), Mean($turn, 10) + 1e-8))'),  # accepted from run_006
    ('QA_VOL_CONCENTRATION_DRIFT', 'Mul(Sub(Div(Max($volume, 5), Mean($volume, 5) + 1e-8), Div(Max($volume, 20), Mean($volume, 20) + 1e-8)), Slope($close, 5))'),  # accepted from run_006
    ('QA_QUIET_VOL_SURGE_20', 'Div(Corr(Div(Sub($high,$low),$close+1e-8),Div($volume,Mean($volume,20)+1e-8),20),Std(Div(Sub($high,$low),$close+1e-8),20)+1e-8)'),  # accepted from run_006
    ('QA_PVCORR_DELTA_VOL', 'Mul(Delta(Corr($close, $volume, 5), 10), Div(Std($close, 5), Std($close, 20) + 1e-8))'),  # accepted from run_006
]


def get_all_exprs():
    return MINED_FACTORS


def get_quantaalpha_exprs():
    """获取 QuantaAlpha 全部挖掘因子表达式 (已注入 Qlib bin)

    只返回 bin 文件实际存在的因子，避免引用不存在的字段。
    field_name 需与 rolling_runner 的命名一致: qa_{name}.lower().replace(" ", "_")
    """
    import json
    from pathlib import Path

    library_path = (
        Path(__file__).resolve().parents[3]
        / "papers" / "QuantaAlpha" / "data" / "factorlib"
        / "all_factors_library.json"
    )
    if not library_path.exists():
        return []

    with open(library_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Check bin existence using a reference instrument
    bin_dir = Path("~/.qlib/qlib_data/cn_data_bs/features/sh600000").expanduser()

    seen = set()
    result = []
    for fid, info in data["factors"].items():
        name = info["factor_name"]
        field_name = f"qa_{name}".lower().replace(" ", "_")
        if field_name in seen:
            continue
        seen.add(field_name)
        if bin_dir.exists() and not (bin_dir / f"{field_name}.day.bin").exists():
            continue
        result.append((name, f"${field_name}"))
    return result


def get_quantaalpha_selected_exprs():
    """获取 QuantaAlpha 筛选后的因子表达式 (高 ICIR + 低相关)

    由 select_and_inject.py 生成 selected_factors.json。
    """
    import json
    from pathlib import Path

    selected_path = (
        Path(__file__).resolve().parents[3]
        / "papers" / "QuantaAlpha" / "data" / "factorlib"
        / "selected_factors.json"
    )
    if not selected_path.exists():
        return []

    with open(selected_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return [
        (f["factor_name"], f["qlib_expr"])
        for f in data["selected_factors"]
    ]
