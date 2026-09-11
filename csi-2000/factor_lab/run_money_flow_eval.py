#!/usr/bin/env python3
"""Step 4: 资金流因子 IC 评价

对 24 个 money_flow 因子做 IC/ICIR 评价。
如果个股资金流/融资融券数据未就绪，可先只评价北向资金 5 个因子。

用法:
    cd trading_framework
    python -m factor_lab.run_money_flow_eval                    # 全部资金流因子
    python -m factor_lab.run_money_flow_eval --group north      # 只评价北向资金因子
    python -m factor_lab.run_money_flow_eval --group main_flow  # 只评价主力资金因子
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import multiprocessing
try:
    multiprocessing.set_start_method('fork', force=True)
except (ValueError, RuntimeError):
    pass  # Windows 无 fork，使用默认 spawn

import qlib
from qlib.constant import REG_CN
qlib.init(provider_uri='~/.qlib/qlib_data/cn_data_bs', region=REG_CN)

from factor_lab.factors import money_flow
from factor_lab.evaluation.single_factor import evaluate_with_qlib

RESULTS_DIR = Path(__file__).parent / "results" / "factor_eval"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 因子分组
FACTOR_GROUPS = {
    "main_flow": money_flow.MAIN_FLOW_FACTORS,
    "order_size": money_flow.ORDER_SIZE_FACTORS,
    "north": money_flow.NORTH_MONEY_FACTORS,
    "margin": money_flow.MARGIN_FACTORS,
    "cross": money_flow.CROSS_FACTORS,
    "all": (money_flow.MAIN_FLOW_FACTORS + money_flow.ORDER_SIZE_FACTORS +
            money_flow.NORTH_MONEY_FACTORS + money_flow.MARGIN_FACTORS +
            money_flow.CROSS_FACTORS),
}


def check_field_availability():
    """检查哪些底层字段已在 Qlib 中可用"""
    from factor_lab.data.qlib_injector import verify_field

    # 基础 OHLCV 字段（Qlib 自带，不需要检查）
    base_fields = {"open", "high", "low", "close", "volume", "amount", "turn", "pctChg"}
    available = {f: True for f in base_fields}

    extra_fields = {
        "north_money": "北向资金",
        "main_net_inflow": "主力净流入",
        "super_large_net": "超大单净流入",
        "large_net": "大单净流入",
        "medium_net": "中单净流入",
        "small_net": "小单净流入",
        "margin_balance": "融资余额",
        "short_balance": "融券余额",
    }
    print("[检查] 底层字段可用性:")
    for field, desc in extra_fields.items():
        ok = verify_field(field, instrument="sh600519", qlib_dir="~/.qlib/qlib_data/cn_data_bs", n_samples=2)
        available[field] = ok
        status = "OK" if ok else "缺失"
        print(f"  {field:20s} ({desc}): {status}")
    return available


def get_evaluable_factors(available_fields: dict[str, bool], group: str = "all"):
    """根据可用字段过滤可评价的因子"""
    factors = FACTOR_GROUPS.get(group)
    if factors is None:
        print(f"未知分组: {group}, 可选: {list(FACTOR_GROUPS.keys())}")
        return []

    evaluable = []
    skipped = []
    for f in factors:
        missing = [r for r in f.required_fields if not available_fields.get(r, False)]
        if missing:
            skipped.append((f.name, missing))
        else:
            evaluable.append((f.name, f.expr))

    if skipped:
        print(f"\n[跳过] {len(skipped)} 个因子 (底层数据缺失):")
        for name, missing in skipped:
            print(f"  {name}: 缺 {missing}")

    return evaluable


def main():
    parser = argparse.ArgumentParser(description="资金流因子 IC 评价")
    parser.add_argument("--group", type=str, default="all",
                        help=f"因子分组 ({', '.join(FACTOR_GROUPS.keys())})")
    parser.add_argument("--start", type=str, default="2024-01-01")
    parser.add_argument("--end", type=str, default="2026-02-05")
    args = parser.parse_args()

    print("=" * 60)
    print("Phase 3 Step 4: 资金流因子 IC 评价")
    print("=" * 60)

    # 1. 检查字段可用性
    available = check_field_availability()

    # 2. 获取可评价的因子
    evaluable = get_evaluable_factors(available, args.group)
    if not evaluable:
        print("\n无可评价的因子。请先运行数据下载脚本:")
        print("  python -m factor_lab.data.download_north_money")
        print("  python -m factor_lab.data.download_akshare_fund_flow")
        print("  python -m factor_lab.data.download_akshare_margin")
        return

    print(f"\n[评价] {len(evaluable)} 个因子, 区间 {args.start} ~ {args.end}")

    # 3. 评估
    result = evaluate_with_qlib(evaluable, start_time=args.start, end_time=args.end)

    # 4. 输出
    print(f"\n{'='*80}")
    print(f"资金流因子 IC 排名表 (共 {len(result)} 个)")
    print(f"{'='*80}")
    if len(result) > 0:
        print(result.to_string(index=False))
    else:
        print("无有效结果")

    # 5. 保存
    out_path = RESULTS_DIR / f"money_flow_{args.group}_ic_report.csv"
    result.to_csv(out_path, index=False)
    print(f"\n结果保存至: {out_path}")

    # 6. 分组汇总
    if args.group == "all" and len(result) > 0:
        print(f"\n[分组汇总]")
        # 按因子分组统计
        group_map = {}
        for gname, factors in FACTOR_GROUPS.items():
            if gname == "all":
                continue
            for f in factors:
                group_map[f.name] = gname

        result["group"] = result["factor"].map(group_map)
        summary = result.groupby("group").agg(
            count=("factor", "count"),
            mean_abs_IC=("abs_IC", "mean"),
            mean_ICIR=("ICIR", lambda x: x.abs().mean()),
        ).round(4)
        print(summary.to_string())


if __name__ == "__main__":
    main()
