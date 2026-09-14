"""参数配对比较。同一相位下算 (候选 − 基线), 报配对 t。

    python run_param_sweep.py --param vol_target --values 0 0.16 0.25
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
    ap.add_argument('--allow-lookahead', action='store_true',
                    help='exec_lag=0 且无截断日线时仍跑。结果有前视，不得写进验收表')
    ap.add_argument('--pred-tag', default=None)
    ap.add_argument('--preset', default=None)
    args = ap.parse_args()

    import qlib
    from qlib.constant import REG_CN
    from qlib_paths import qlib_provider_uri
    qlib.init(provider_uri=qlib_provider_uri(), region=REG_CN)
    from factor_lab.paper_trader import PaperTrader

    def parse(v):
        if '+' in str(v):
            return str(v)
        if str(v).lower() in ('none', 'off', '关闭'):
            return None
        # '0' 对数值参数是"关闭"，但对字符串参数(如 adaptive_strategy)
        # 没有这个语义，所以只在能转成数字时才当 0 处理。
        try:
            return float(v) if '.' in str(v) else int(v)
        except ValueError:
            return str(v)      # 字符串参数原样传入，如 A_topk_adaptive

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
            # 支持复合参数: --param topk+n_drop --values 100+20 30+6
            # 2026-09-13: topk 和 n_drop 是耦合的 —— n_drop/topk 才是换手率,
            # 单独扫 topk 会把换手成本的变化算进 topk 头上。
            # 实测: topk 100->30 而 n_drop 固定 20, 换手从 20% 跳到 67%,
            # 配对 t=-8.05 看着是 topk 的锅, 其实是全量换手的锅。
            if '+' in str(args.param):
                keys = str(args.param).split('+')
                vals = str(val).split('+')
                for k, v in zip(keys, vals):
                    t.config[k] = int(v) if '.' not in v else float(v)
            else:
                t.config[args.param] = val
            perf = t.replay(args.start, args.end, verbose=False,
                            allow_lookahead=args.allow_lookahead,
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
