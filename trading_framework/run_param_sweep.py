"""参数配对比较 — 同一组相位下算逐样本差值，报配对 t 与胜出相位数

为什么必须配对
--------------
CLAUDE.md: 相位间 Sharpe 标准差 0.11~0.33，而候选参数之间的差异只有
0.1~0.16 —— 直接比两个配置的均值，差异完全淹没在噪声里。

配对比较把相位当作区组: 同一相位下算 (候选 − 基线) 的差值，噪声中"这一相位
恰好好/坏"的成分在相减时抵消。TopK 8→16 就是这样定的(配对 t=2.74，
12/16 相位胜出)；用单相位对比会得出相反结论。

用法
----
    # vol_target 扫描 (两段)
    python run_param_sweep.py --param vol_target --values 0 0.06 0.08 0.10
    python run_param_sweep.py --param vol_target --values 0 0.06 0.08 0.10 \\
        --start 2022-05-04 --end 2023-12-29 --tag 段1 --pred-tag pre2024

基线取 --values 的第一个。
"""

import argparse
import json
import math
import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import qlib_compat  # noqa: F401

PROJECT_DIR = Path(__file__).parent


def paired_t(diffs: list) -> tuple:
    """配对 t 统计量与自由度。全部差值为 0 时返回 (0.0, n-1)。"""
    n = len(diffs)
    if n < 2:
        return 0.0, 0
    m = statistics.mean(diffs)
    sd = statistics.stdev(diffs)
    if sd < 1e-12:
        return (0.0 if abs(m) < 1e-12 else math.inf), n - 1
    return m / (sd / math.sqrt(n)), n - 1


def main() -> int:
    ap = argparse.ArgumentParser(description='参数配对比较')
    ap.add_argument('--param', required=True,
                    help='signal_config 里的键名，如 vol_target / n_drop / topk')
    ap.add_argument('--values', nargs='+', required=True,
                    help='候选值，第一个作为基线。0 或 none 表示关闭')
    ap.add_argument('--phases', type=int, default=8)
    ap.add_argument('--start', default='2024-01-02')
    ap.add_argument('--end', default='2026-09-04')
    ap.add_argument('--tag', default='主段')
    ap.add_argument('--pred-tag', default=None)
    ap.add_argument('--preset', default=None)
    args = ap.parse_args()

    import qlib
    from qlib.constant import REG_CN
    qlib.init(provider_uri=str(Path.home() / '.qlib/qlib_data/cn_data_bs'),
              region=REG_CN)
    from factor_lab.paper_trader import PaperTrader

    def parse(v):
        if str(v).lower() in ('none', 'off', '0', '关闭'):
            return None
        return float(v) if '.' in str(v) else int(v)

    values = [parse(v) for v in args.values]

    print("=" * 74)
    print(f"参数配对比较  {args.param}  {args.tag}  {args.start} ~ {args.end}  "
          f"{args.phases} 相位")
    print("=" * 74)

    t0 = time.time()
    results = {}          # {value: [每相位指标 dict]}
    for val in values:
        rows = []
        for ph in range(args.phases):
            t = PaperTrader(state_dir=tempfile.mkdtemp(),
                            pred_tag=args.pred_tag,
                            preset=args.preset)
            t.config[args.param] = val
            perf = t.replay(args.start, args.end, verbose=False,
                            phase=ph, save=False)
            if 'error' in perf:
                print(f"  {args.param}={val} 相位{ph}: 失败")
                continue
            rows.append(perf)
        results[str(val)] = rows
        sh = [r['sharpe'] for r in rows]
        dd = [r['max_drawdown'] for r in rows]
        ex = [r['excess_return'] for r in rows if r['excess_return'] is not None]
        print(f"  {args.param}={str(val):<6} Sharpe 均值 {statistics.mean(sh):6.3f}"
              f"  最差 {min(sh):6.3f}  回撤均值 {statistics.mean(dd):7.2%}"
              f"  超额均值 {statistics.mean(ex) if ex else float('nan'):+7.2%}",
              flush=True)

    base_key = str(values[0])
    base = results[base_key]
    print()
    print("-" * 74)
    print(f"配对比较 (基线 {args.param}={base_key})")
    print(f"{'候选':<12} {'ΔSharpe':>9} {'配对t':>8} {'胜出':>7} "
          f"{'Δ回撤':>9} {'Δ超额':>9}")
    for val in values[1:]:
        cand = results[str(val)]
        n = min(len(base), len(cand))
        d_sh = [cand[i]['sharpe'] - base[i]['sharpe'] for i in range(n)]
        d_dd = [cand[i]['max_drawdown'] - base[i]['max_drawdown']
                for i in range(n)]
        d_ex = [(cand[i]['excess_return'] or 0) - (base[i]['excess_return'] or 0)
                for i in range(n)]
        tstat, _ = paired_t(d_sh)
        wins = sum(1 for d in d_sh if d > 0)
        print(f"{str(val):<12} {statistics.mean(d_sh):>+9.3f} {tstat:>8.2f} "
              f"{wins:>4}/{n:<2} {statistics.mean(d_dd):>+9.2%} "
              f"{statistics.mean(d_ex):>+9.2%}")
    print("-" * 74)
    print("|t| < 2 表示差异在噪声范围内，不足以据此改参数。")
    print(f"耗时 {time.time()-t0:.0f}s")

    out = PROJECT_DIR / 'factor_lab' / 'results' / f'sweep_{args.param}_{args.tag}.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        json.dump({k: [{'sharpe': r['sharpe'],
                        'max_drawdown': r['max_drawdown'],
                        'excess_return': r['excess_return'],
                        'total_return': r['total_return']} for r in v]
                   for k, v in results.items()}, f, ensure_ascii=False, indent=1)
    print(f"结果已写入 {out}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
