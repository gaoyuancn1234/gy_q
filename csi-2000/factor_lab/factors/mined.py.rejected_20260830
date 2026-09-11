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
    ('QA_UPDAY_VOL_RATIO', 'Div(Mean(Mul(Greater($close,Ref($close,1)),$volume),20),Mean($volume,20)+1e-8)'),  # accepted from run_010
    ('QA_UPDOWN_VOL_DIFF_10', 'Mean(Mul(Greater(Delta($close,1),0),Power(Delta($close,1),2)),10)'),  # accepted from run_010
    ('QA_UPVOL_SPREAD_NORM', 'Div(Mean(If(Greater($close, Ref($close, 1)), Sub($high, $low), 0), 20), Mean(Sub($high, $low), 20) + 1e-8)'),  # accepted from run_010
    ('QA_VOL_HIGHLOW_ASYM', 'Mean(Mul($volume, Div($close - $low, $high - $low + 1e-8)), 10)'),  # accepted from run_010
    ('QA_MULTI_PERIOD_SIGN_CONSENSUS', 'Add(Add(Sign(Div($close, Ref($close, 5)) - 1), Sign(Div($close, Ref($close, 10)) - 1)), Sign(Div($close, Ref($close, 20)) - 1))'),  # accepted from run_010
    ('QA_COMPRESSION_BREAKOUT', 'Mul(Div(Sub(Min($high, 20), Max($low, 20)), Max(Sub($high, $low), 20) + 1e-8), Div(Sub($close, Mean($close, 20)), Std($close, 20) + 1e-8))'),  # accepted from run_010
    ('QA_ASYMMETRIC_REBOUND_STRENGTH', 'Div(Div($close - Min($low, 20), Ref($close, 20) + 1e-8), Div(Max($high, 20) - $close, Ref($close, 20) + 1e-8) + 1e-8)'),  # accepted from run_010
    ('QA_VWAP_ESCAPE_VELOCITY', 'Div(Sub($close, Mean(Div($amount, Add($volume, 1e-8)), 15)), Std(Sub($close, Div($amount, Add($volume, 1e-8))), 15) + 1e-8)'),  # accepted from run_010
    ('QA_VWAP_RANGE_POS', 'Mean(Div(Sub(Div($amount, $volume + 1e-8), $low), Sub($high, $low) + 1e-8), 20)'),  # accepted from run_010
    ('QA_INTRADAY_PATH_ASYM', 'Mean(Div(Sub(IdxMax($high,10), IdxMin($low,10)), 10 + 1e-8), 5)'),  # accepted from run_010
    ('QA_MOMENTUM_DECAY_NORMED', 'Div(Sub(Mean($close, 5), Mean($close, 20)), Std($close, 20) + 1e-8)'),  # accepted from run_010
    ('QA_VOL_DISPERSION_MEAN_REV', 'Mul(Div(Std($volume, 5), Mean($volume, 5) + 1e-8), Mul(Div(Sub(Mean($close, 20), $close), Std($close, 20) + 1e-8), Div(Mean($volume, 5), Mean($volume, 20) + 1e-8)))'),  # accepted from run_010
    ('QA_VWAP_GRAVITY_DRIFT', 'Sub(Div(Sum($amount,5),Sum($volume,5)+1e-8),Div(Sum($amount,20),Sum($volume,20)+1e-8))'),  # accepted from run_010
    ('QA_VOLCONC_PRICEMOM_DIV', 'Mul(Div(Sum(Greater($volume, Mean($volume, 20)), 5), 5 + 1e-8), Mul(Sign(Div($close, Ref($close, 10) + 1e-8) - 1), -1))'),  # accepted from run_010
    ('QA_LIQUIDITY_RESILIENCE_RATIO', 'Div(Div($high - $low, $close + 1e-8), Ref(Div($high - $low, $close + 1e-8), Sub(20, IdxMax($volume, 20))) + 1e-8)'),  # accepted from run_010
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
