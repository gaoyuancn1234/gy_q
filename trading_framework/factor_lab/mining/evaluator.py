"""IC/ICIR 评估 + 冗余检测

复用 single_factor.evaluate_with_qlib() 和 correlation 模块。
"""
import pandas as pd


def evaluate_candidates(factors: list[tuple[str, str]],
                        start_time: str = "2024-01-01",
                        end_time: str = "2026-02-05",
                        ) -> pd.DataFrame:
    """批量评估因子的 IC/ICIR

    每个因子独立 try/except, 一个坏因子不影响整批。

    Args:
        factors: [(name, expr), ...]
        start_time, end_time: 评估区间

    Returns:
        DataFrame with columns: factor, mean_IC, ICIR, is_promising, is_excellent
    """
    from factor_lab.evaluation.single_factor import evaluate_with_qlib

    results = []
    batch_size = 5

    for i in range(0, len(factors), batch_size):
        batch = factors[i:i + batch_size]
        try:
            df = evaluate_with_qlib(
                batch,
                instruments="csi300",
                start_time=start_time,
                end_time=end_time,
            )
            results.append(df)
        except Exception as e:
            print(f"  [eval] 批次 {i//batch_size + 1} 失败: {e}")
            # 逐个重试
            for name, expr in batch:
                try:
                    df = evaluate_with_qlib(
                        [(name, expr)],
                        instruments="csi300",
                        start_time=start_time,
                        end_time=end_time,
                    )
                    results.append(df)
                except Exception as e2:
                    print(f"    [eval] {name} 失败: {e2}")

    if not results:
        return pd.DataFrame(columns=["factor", "mean_IC", "ICIR", "is_promising", "is_excellent"])

    combined = pd.concat(results, ignore_index=True)
    combined["is_promising"] = combined["ICIR"].abs() >= 0.5
    combined["is_excellent"] = combined["ICIR"].abs() >= 1.0
    return combined.sort_values("ICIR", key=abs, ascending=False).reset_index(drop=True)


def check_redundancy(new_factors: list[tuple[str, str]],
                     baseline_preset: str = "alpha158_val",
                     threshold: float = 0.7,
                     start_time: str = "2025-01-01",
                     end_time: str = "2025-12-31",
                     baseline_factors: list[tuple[str, str]] | None = None,
                     ) -> dict:
    """检测新因子 vs 已有因子的冗余性

    Args:
        new_factors: [(name, expr), ...]
        baseline_preset: 已有因子预设 (baseline_factors 为 None 时使用)
        threshold: 相关性阈值
        start_time, end_time: 计算相关性的区间
        baseline_factors: 显式指定对照因子集，用于"与因子池已有成员比对"
                          (论文 Section 4.3 的准入规则是 vs 池成员，不是 vs 预设)。
                          给空列表表示无对照，直接判定不冗余。

    Returns:
        {name: {is_redundant, max_corr, most_correlated}}
    """
    from qlib.data import D
    from factor_lab.factors.presets import FACTOR_PRESETS

    if baseline_factors is not None:
        extra = baseline_factors
        if not extra:                      # 池为空: 无可比对象
            return {n: {"is_redundant": False, "max_corr": 0.0,
                        "most_correlated": ""} for n, _ in new_factors}
    else:
        preset = FACTOR_PRESETS[baseline_preset]
        extra = preset["extra_factors"]
        if callable(extra):
            extra = extra()
    baseline_names = [n for n, _ in extra]
    baseline_exprs = [e for _, e in extra]

    # 新因子
    new_names = [n for n, _ in new_factors]
    new_exprs = [e for _, e in new_factors]

    # 合并加载 (baseline 加前缀防重名)
    prefixed_bl_names = [f"_bl_{n}" for n in baseline_names]
    all_names = new_names + prefixed_bl_names
    all_exprs = new_exprs + baseline_exprs

    inst = D.instruments("csi300")

    try:
        df = D.features(
            instruments=inst,
            fields=all_exprs,
            start_time=start_time,
            end_time=end_time,
        )
        df.columns = all_names
    except Exception as e:
        print(f"  [redundancy] 加载因子失败: {e}")
        return {n: {"is_redundant": False, "max_corr": 0.0, "most_correlated": ""} for n in new_names}

    # 计算每个新因子 vs 所有 baseline 因子的最大相关性
    result = {}
    for new_name in new_names:
        max_corr = 0.0
        most_correlated = ""

        new_col = df[new_name]
        for bl_name in prefixed_bl_names:
            try:
                bl_col = df[bl_name]
                # 按日截面 rank 相关性取均值
                dates = df.index.get_level_values(0).unique()
                corrs = []
                for dt in dates:
                    try:
                        a = new_col.loc[dt].dropna()
                        b = bl_col.loc[dt].dropna()
                        common = a.index.intersection(b.index)
                        if len(common) < 20:
                            continue
                        c = abs(a[common].rank().corr(b[common].rank()))
                        if pd.notna(c):
                            corrs.append(c)
                    except Exception:
                        continue

                avg_corr = sum(corrs) / len(corrs) if corrs else 0.0
                if avg_corr > max_corr:
                    max_corr = avg_corr
                    most_correlated = bl_name.removeprefix("_bl_")
            except Exception:
                continue

        result[new_name] = {
            "is_redundant": max_corr > threshold,
            "max_corr": round(max_corr, 4),
            "most_correlated": most_correlated,
        }

    return result
