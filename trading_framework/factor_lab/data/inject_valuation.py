#!/usr/bin/env python3
"""把 akshare 日频估值注入 Qlib bin，并验证因子确实能取到值

为什么带验证
------------
CLAUDE.md 记过: qlib 对缺失字段不报错，返回**全 NaN 列** —— `'$isST' in
columns` 判定为 True，ST 过滤形同虚设。注入这件事同样如此: 写完 bin 不代表
因子能用，必须实测 `D.features` 取回来的列非空。

而 fundamental.get_all_exprs() 此前还有 all-or-nothing 缺陷: 任一字段缺失
就返回 []，导致 alpha158_val 里基本面因子数为 0，而名字和文档都说有。
所以这里逐项验证并打印真实可用的因子数。

用法
----
    python -m factor_lab.data.inject_valuation
    python -m factor_lab.data.inject_valuation --dry-run   # 只验证不写
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd

CACHE = (Path(__file__).resolve().parent.parent / "results" / ".cache"
         / "daily_valuation.parquet")
FIELDS = ["pe_ttm", "pb", "ps_ttm", "total_mv", "circ_mv"]


def verify(qlib_dir: str, instruments: list) -> int:
    """实测字段与因子是否真的可用。返回可用因子数。"""
    from qlib.data import D

    print("\n=== 验证注入结果 ===")
    probe = instruments[:30]
    cols = [f"${f}" for f in FIELDS]
    df = D.features(probe, cols, start_time='2025-01-01', end_time='2026-09-04')

    ok_fields = []
    for f in FIELDS:
        c = f"${f}"
        if df is None or df.empty or c not in df.columns:
            print(f"  ✗ {c:12s} 列不存在")
            continue
        # 关键: 列存在不等于有数据，qlib 缺字段时返回全 NaN
        nn = df[c].notna().sum()
        if nn == 0:
            print(f"  ✗ {c:12s} 列存在但**全 NaN** (qlib 缺字段的典型表现)")
            continue
        print(f"  ✓ {c:12s} 非空 {nn:>7}/{len(df)} ({nn/len(df):.1%})  "
              f"中位数 {df[c].median():.3f}")
        ok_fields.append(f)

    from factor_lab.factors.fundamental import get_all_exprs
    exprs = get_all_exprs(verbose=True)
    print(f"\n  基本面因子可用数: {len(exprs)}")
    if exprs:
        for name, e in exprs[:5]:
            print(f"    {name:14s} {e}")
        if len(exprs) > 5:
            print(f"    ... 共 {len(exprs)} 个")
    return len(exprs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true', help='只验证，不写入')
    ap.add_argument('--qlib-dir',
                    default=str(Path.home() / '.qlib/qlib_data/cn_data_bs'))
    args = ap.parse_args()

    import qlib
    from qlib.constant import REG_CN
    qlib.init(provider_uri=args.qlib_dir, region=REG_CN)
    from qlib.data import D

    inst = D.instruments("csi300")
    instruments = sorted(D.list_instruments(instruments=inst, as_list=True))

    if not args.dry_run:
        if not CACHE.exists():
            print(f"✗ 估值数据不存在: {CACHE}")
            print("  请先运行: python -m factor_lab.data.akshare_valuation")
            return 2
        df = pd.read_parquet(CACHE)
        print(f"读入 {len(df):,} 行 / {df['instrument'].nunique()} 只  "
              f"{df['date'].min().date()} ~ {df['date'].max().date()}")

        missing = [f for f in FIELDS if f not in df.columns]
        if missing:
            print(f"✗ 缺列 {missing}")
            return 2

        from factor_lab.data.qlib_injector import inject_dataframe
        print(f"\n注入 {len(FIELDS)} 个字段...")
        inject_dataframe(df, FIELDS, qlib_dir=args.qlib_dir)

        # 注入后必须重新 init，否则读到的是旧的字段缓存
        qlib.init(provider_uri=args.qlib_dir, region=REG_CN)

    n = verify(args.qlib_dir, instruments)
    if n == 0:
        print("\n✗ 注入后基本面因子数仍为 0")
        return 1
    print(f"\n✓ {n} 个基本面因子可用")
    return 0


if __name__ == '__main__':
    sys.exit(main())
