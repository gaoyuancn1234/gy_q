#!/usr/bin/env python3
"""Step 6: 全因子去冗余与最终筛选

对全部 ~76 个扩展因子 (30 精选量价 + 22 基本面 + 24 资金流) 做:
1. IC/ICIR 评价
2. 相关性矩阵分析
3. 去冗余 (corr > 0.7) 保留 ICIR 更高的
4. 输出 full_selected 因子名单
5. 跑 full_selected benchmark 对比

用法:
    cd trading_framework
    python -m factor_lab.run_full_dedup
    python -m factor_lab.run_full_dedup --corr_threshold 0.65
    python -m factor_lab.run_full_dedup --benchmark  # 去冗余后跑benchmark
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import multiprocessing
multiprocessing.set_start_method('fork', force=True)

import qlib
from qlib.constant import REG_CN
qlib.init(provider_uri='~/.qlib/qlib_data/cn_data_bs', region=REG_CN)

import pandas as pd
from qlib.data import D

from factor_lab.factors import alpha158_ext, fundamental, money_flow
from factor_lab.factors.presets import _get_selected_exprs, update_full_selected
from factor_lab.evaluation.single_factor import evaluate_with_qlib
from factor_lab.evaluation.correlation import compute_factor_correlation, remove_redundant, find_high_corr_pairs

RESULTS_DIR = Path(__file__).parent / "results" / "factor_eval"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def get_all_extra_exprs() -> list[tuple[str, str]]:
    """获取所有扩展因子 (量价精选 + 基本面 + 资金流)"""
    selected = _get_selected_exprs()
    fund = fundamental.get_all_exprs()
    mf = money_flow.get_all_exprs()
    all_exprs = selected + fund + mf
    print(f"  量价精选: {len(selected)}")
    print(f"  基本面: {len(fund)}")
    print(f"  资金流: {len(mf)}")
    print(f"  合计: {len(all_exprs)}")
    return all_exprs


def _get_all_factor_metas():
    """从所有因子模块收集 FactorMeta 列表"""
    all_metas = []
    for module in [alpha158_ext, fundamental, money_flow]:
        # 收集模块中所有以 _FACTORS 结尾的列表
        for attr_name in dir(module):
            if attr_name.endswith("_FACTORS") and attr_name.isupper():
                val = getattr(module, attr_name)
                if isinstance(val, list) and val and hasattr(val[0], "required_fields"):
                    all_metas.extend(val)
    return all_metas


def check_evaluable(all_exprs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """检查哪些因子可以评价 (底层字段已存在)"""
    from factor_lab.data.qlib_injector import verify_field

    all_metas = _get_all_factor_metas()

    # 检查底层字段
    all_required = set()
    for f in all_metas:
        all_required.update(f.required_fields)

    # 基础字段 (OHLCV + baostock ext) 假设已存在
    base_fields = {"open", "high", "low", "close", "volume", "amount", "turn", "pctChg"}
    extra_fields = all_required - base_fields

    available_extra = {}
    for field in extra_fields:
        ok = verify_field(field, instrument="sh600519",
                         qlib_dir="~/.qlib/qlib_data/cn_data_bs", n_samples=1)
        available_extra[field] = ok

    all_available = base_fields | {f for f, ok in available_extra.items() if ok}
    missing = {f for f, ok in available_extra.items() if not ok}
    if missing:
        print(f"\n  缺失字段: {missing}")

    # 构建因子 → 所需字段 的映射
    factor_deps = {}
    for f in all_metas:
        factor_deps[f.name] = set(f.required_fields)

    evaluable = []
    skipped = []
    for name, expr in all_exprs:
        deps = factor_deps.get(name, set())
        missing_deps = deps - all_available
        if missing_deps:
            skipped.append((name, missing_deps))
        else:
            evaluable.append((name, expr))

    if skipped:
        print(f"\n  跳过 {len(skipped)} 个因子:")
        for name, miss in skipped[:10]:
            print(f"    {name}: 缺 {miss}")
        if len(skipped) > 10:
            print(f"    ... 及另外 {len(skipped)-10} 个")

    return evaluable


def run_dedup(start_time: str = "2024-01-01", end_time: str = "2026-02-05",
              corr_threshold: float = 0.7):
    """执行去冗余流程"""
    print(f"\n{'='*60}")
    print(f"Step 6: 全因子去冗余")
    print(f"  区间: {start_time} ~ {end_time}")
    print(f"  相关性阈值: {corr_threshold}")
    print(f"{'='*60}")

    # 1. 获取所有扩展因子
    print("\n[1] 收集因子...")
    all_exprs = get_all_extra_exprs()

    # 2. 检查可评价的因子
    print("\n[2] 检查底层字段...")
    evaluable = check_evaluable(all_exprs)
    print(f"\n  可评价因子: {len(evaluable)} / {len(all_exprs)}")

    if len(evaluable) < 5:
        print("ERROR: 可评价因子太少，请先运行数据下载")
        return None

    # 3. IC 评价
    print(f"\n[3] IC 评价 ({len(evaluable)} 个因子)...")
    ic_result = evaluate_with_qlib(evaluable, start_time=start_time, end_time=end_time)

    if ic_result.empty:
        print("ERROR: 所有因子评估失败，请检查数据完整性")
        return None

    print(f"\n因子 IC 排名 (top 20):")
    print(ic_result.head(20).to_string(index=False))

    # 保存全量 IC 报告
    ic_path = RESULTS_DIR / "full_all_ic_report.csv"
    ic_result.to_csv(ic_path, index=False)
    print(f"\nIC 报告保存至: {ic_path}")

    # 4. 相关性矩阵
    print(f"\n[4] 计算相关性矩阵...")
    names = [n for n, _ in evaluable]
    exprs = [e for _, e in evaluable]

    inst = D.instruments("csi300")
    factor_df = D.features(
        instruments=inst,
        fields=exprs,
        start_time=start_time,
        end_time=end_time,
    )
    factor_df.columns = names

    corr_matrix = compute_factor_correlation(factor_df)

    # 高相关对
    high_pairs = find_high_corr_pairs(corr_matrix, corr_threshold)
    print(f"  高相关因子对 (corr > {corr_threshold}): {len(high_pairs)}")
    for a, b, c in high_pairs[:15]:
        print(f"    {a:25s} <-> {b:25s}: {c:.3f}")
    if len(high_pairs) > 15:
        print(f"    ... 及另外 {len(high_pairs)-15} 对")

    # 5. 去冗余
    print(f"\n[5] 去冗余 (保留 ICIR 更高的)...")
    ic_scores = dict(zip(ic_result["factor"], ic_result["ICIR"].abs()))
    kept = remove_redundant(corr_matrix, ic_scores, corr_threshold)

    removed = set(names) - set(kept)
    print(f"  原始: {len(names)}")
    print(f"  保留: {len(kept)}")
    print(f"  移除: {len(removed)}")

    if removed:
        print(f"\n  移除的因子:")
        for r in sorted(removed):
            print(f"    {r} (ICIR={ic_scores.get(r, 0):.3f})")

    # 6. 保留的因子 IC
    kept_ic = ic_result[ic_result["factor"].isin(set(kept))].copy()
    print(f"\n保留因子 IC 排名:")
    print(kept_ic.to_string(index=False))

    kept_path = RESULTS_DIR / "full_selected_ic_report.csv"
    kept_ic.to_csv(kept_path, index=False)
    print(f"\n去冗余结果保存至: {kept_path}")

    # 7. 更新 preset
    kept_set = set(kept)
    update_full_selected(kept_set)
    print(f"\n[6] 已更新 full_selected preset: {len(kept_set)} 个扩展因子")

    # 8. 输出因子名单 (方便硬编码到 presets.py)
    names_path = RESULTS_DIR / "full_selected_names.txt"
    with open(names_path, "w") as f:
        for name in sorted(kept):
            f.write(f"{name}\n")
    print(f"因子名单保存至: {names_path}")

    # 打印可粘贴到 presets.py 的代码
    print(f"\n# 可粘贴到 presets.py 的 _FULL_SELECTED_FACTORS:")
    print(f"_FULL_SELECTED_FACTORS = {{")
    for name in sorted(kept):
        print(f'    "{name}",')
    print(f"}}")

    return kept


def main():
    parser = argparse.ArgumentParser(description="全因子去冗余")
    parser.add_argument("--start", type=str, default="2024-01-01")
    parser.add_argument("--end", type=str, default="2026-02-05")
    parser.add_argument("--corr_threshold", type=float, default=0.7)
    parser.add_argument("--benchmark", action="store_true",
                        help="去冗余后跑 full_selected benchmark")
    args = parser.parse_args()

    print("=" * 60)
    print("Phase 3 Step 6: 全因子去冗余与最终筛选")
    print("=" * 60)

    kept = run_dedup(args.start, args.end, args.corr_threshold)

    if kept is not None and args.benchmark:
        print(f"\n\n{'='*60}")
        print(f"跑 full_selected benchmark...")
        print(f"{'='*60}")

        # 重设 preset 后跑 benchmark
        from factor_lab.run_full_benchmark import run_preset_benchmark, print_comparison
        run_preset_benchmark("full_selected", ["LightGBM", "XGBoost", "CatBoost"], force=True)
        print_comparison(
            ["alpha158_selected", "alpha158_val", "full", "full_selected"],
            ["LightGBM", "XGBoost", "CatBoost"],
        )


if __name__ == "__main__":
    main()
