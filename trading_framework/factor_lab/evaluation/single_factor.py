"""单因子评价 — IC/RankIC/ICIR/衰减分析

纯 pandas 实现，不依赖 Alphalens。
使用 Qlib 表达式引擎计算因子值，逐日截面 Spearman/Pearson 相关。

用法:
    python -m factor_lab.evaluation.single_factor --preset alpha158_ext
"""
import numpy as np
import pandas as pd


def compute_ic(factor_values: pd.DataFrame, returns: pd.Series,
               method: str = "spearman") -> pd.Series:
    """计算逐日截面 IC

    Args:
        factor_values: MultiIndex (datetime, instrument) → factor value
        returns: MultiIndex (datetime, instrument) → forward return
        method: "spearman" (RankIC) 或 "pearson" (IC)

    Returns:
        Series indexed by date, value = IC
    """
    # 确保对齐
    common = factor_values.index.intersection(returns.index)
    fv = factor_values.loc[common]
    ret = returns.loc[common]

    dates = fv.index.get_level_values(0).unique()
    ic_values = {}

    for dt in dates:
        try:
            f = fv.loc[dt]
            r = ret.loc[dt]
            # 去 NaN
            mask = f.notna() & r.notna()
            if mask.sum() < 10:
                continue
            if method == "spearman":
                ic_values[dt] = f[mask].rank().corr(r[mask].rank())
            else:
                ic_values[dt] = f[mask].corr(r[mask])
        except Exception:
            continue

    return pd.Series(ic_values, name="IC")


def compute_icir(ic_series: pd.Series) -> float:
    """计算 ICIR = mean(IC) / std(IC)"""
    if len(ic_series) < 5:
        return 0.0
    std = ic_series.std()
    if std == 0:
        return 0.0
    return float(ic_series.mean() / std)


def compute_ic_decay(factor_values: pd.DataFrame, price_data: pd.DataFrame,
                     periods: list[int] = None) -> dict[int, float]:
    """计算不同持有期的 IC 衰减

    Args:
        factor_values: MultiIndex (datetime, instrument) → factor value
        price_data: MultiIndex (datetime, instrument) → close price
        periods: 持有期列表，默认 [1, 2, 3, 5, 10, 20]

    Returns:
        {period: mean_IC}
    """
    if periods is None:
        periods = [1, 2, 3, 5, 10, 20]

    results = {}
    dates = factor_values.index.get_level_values(0).unique().sort_values()

    for period in periods:
        ic_values = []
        for i, dt in enumerate(dates):
            if i + period >= len(dates):
                break
            future_dt = dates[i + period]
            try:
                fv = factor_values.loc[dt]
                p_now = price_data.loc[dt]
                p_future = price_data.loc[future_dt]

                common = fv.index.intersection(p_now.index).intersection(p_future.index)
                if len(common) < 10:
                    continue

                ret = (p_future[common] / p_now[common] - 1)
                f = fv[common]
                mask = f.notna() & ret.notna()
                if mask.sum() < 10:
                    continue

                ic = f[mask].rank().corr(ret[mask].rank())
                if not np.isnan(ic):
                    ic_values.append(ic)
            except Exception:
                continue

        results[period] = np.mean(ic_values) if ic_values else 0.0

    return results


def batch_evaluate(factor_dict: dict[str, pd.DataFrame],
                   returns: pd.Series,
                   method: str = "spearman") -> pd.DataFrame:
    """批量评估多个因子

    Args:
        factor_dict: {因子名: MultiIndex DataFrame}
        returns: 收益率 Series

    Returns:
        DataFrame with columns: [factor, mean_IC, std_IC, ICIR, IC>0_ratio, abs_IC]
    """
    results = []

    for name, fv in factor_dict.items():
        # 如果是 DataFrame 取第一列
        if isinstance(fv, pd.DataFrame):
            fv = fv.iloc[:, 0] if fv.shape[1] == 1 else fv

        if isinstance(fv, pd.DataFrame):
            fv = fv.iloc[:, 0]

        ic = compute_ic(fv, returns, method=method)
        if len(ic) < 5:
            continue

        results.append({
            "factor": name,
            "mean_IC": ic.mean(),
            "std_IC": ic.std(),
            "ICIR": compute_icir(ic),
            "IC>0_ratio": (ic > 0).mean(),
            "abs_IC": ic.abs().mean(),
            "n_days": len(ic),
        })

    df = pd.DataFrame(results)
    if len(df) > 0:
        df = df.sort_values("abs_IC", ascending=False).reset_index(drop=True)
    return df


def evaluate_with_qlib(factor_exprs: list[tuple[str, str]],
                       instruments: str = "csi300",
                       start_time: str = "2024-01-01",
                       end_time: str = "2026-02-05",
                       label_expr: str = "Ref($close, -2)/Ref($close, -1) - 1") -> pd.DataFrame:
    """使用 Qlib 表达式引擎直接评估因子

    Args:
        factor_exprs: [(name, expr), ...]
        instruments: 股票池
        start_time, end_time: 评估区间
        label_expr: 标签表达式 (未来收益)

    Returns:
        因子评价 DataFrame
    """
    from qlib.data import D

    names = [n for n, _ in factor_exprs]
    exprs = [e for _, e in factor_exprs]

    # 字符串 instruments 需要转为 Qlib instruments 对象
    inst = D.instruments(instruments)

    # 加载因子值
    print(f"  [eval] 加载 {len(factor_exprs)} 个因子...")
    factor_df = D.features(
        instruments=inst,
        fields=exprs,
        start_time=start_time,
        end_time=end_time,
    )
    factor_df.columns = names

    # 加载标签
    label_df = D.features(
        instruments=inst,
        fields=[label_expr],
        start_time=start_time,
        end_time=end_time,
    )
    label_series = label_df.iloc[:, 0]

    # 批量评估
    factor_dict = {name: factor_df[name] for name in names}
    return batch_evaluate(factor_dict, label_series)


def main():
    """命令行入口"""
    import argparse
    import sys
    from pathlib import Path

    parser = argparse.ArgumentParser(description="单因子评价")
    parser.add_argument("--preset", type=str, default="alpha158_ext",
                        help="因子预设 (alpha158_ext, fundamental, money_flow)")
    parser.add_argument("--start", type=str, default="2024-01-01")
    parser.add_argument("--end", type=str, default="2026-02-05")
    args = parser.parse_args()

    # 初始化 Qlib
    import qlib
    from qlib.constant import REG_CN
    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data_bs", region=REG_CN)

    # 获取因子表达式
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from factor_lab.factors import alpha158_ext, fundamental, money_flow

    preset_map = {
        "alpha158_ext": alpha158_ext.get_all_exprs,
        "fundamental": fundamental.get_all_exprs,
        "money_flow": money_flow.get_all_exprs,
    }

    if args.preset not in preset_map:
        print(f"未知预设: {args.preset}, 可选: {list(preset_map.keys())}")
        return

    factor_exprs = preset_map[args.preset]()
    print(f"评估 {args.preset}: {len(factor_exprs)} 个因子")
    print(f"区间: {args.start} ~ {args.end}")

    result = evaluate_with_qlib(factor_exprs, start_time=args.start, end_time=args.end)
    print(f"\n{'='*80}")
    print(f"因子 IC 排名表 (共 {len(result)} 个因子)")
    print(f"{'='*80}")
    print(result.to_string(index=False))

    # 保存结果
    out_dir = Path(__file__).parent.parent / "results" / "factor_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.preset}_ic_report.csv"
    result.to_csv(out_path, index=False)
    print(f"\n结果保存至: {out_path}")


if __name__ == "__main__":
    main()
