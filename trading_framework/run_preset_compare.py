"""因子集配对比较 — 同一组相位下比两份预测缓存

与 run_param_sweep 的区别: 那个比的是同一份预测下的不同风控参数，这个比的是
**不同因子集训练出的不同预测**。因子集变了必须重训，所以先跑
run_rolling_benchmark 产出两份 pkl，再用本脚本做配对。

为什么必须配对且必须两段
------------------------
CLAUDE.md 记着一次一模一样的翻车: 挖掘出的 22 个因子样本内 +7.4%、
**样本外 -12.9%(符号反转)**，已全部清除。这次同样是 22 个基本面因子，
不能只看单段好看就放行。

用法
----
    python run_preset_compare.py \\
        --base alpha158_selected:fund188 --cand alpha158_val:fund210

    # 段1
    python run_preset_compare.py \\
        --base alpha158_selected:fund188 --cand alpha158_val:fund210 \\
        --start 2022-05-04 --end 2023-12-29 --tag 段1

格式为 preset:tag，tag 可省略。
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


def paired_t(diffs: list) -> float:
    n = len(diffs)
    if n < 2:
        return 0.0
    sd = statistics.stdev(diffs)
    if sd < 1e-12:
        return 0.0 if abs(statistics.mean(diffs)) < 1e-12 else math.inf
    return statistics.mean(diffs) / (sd / math.sqrt(n))


def parse_spec(spec: str) -> tuple:
    """'alpha158_val:fund210' -> ('alpha158_val', 'fund210')"""
    if ':' in spec:
        p, t = spec.split(':', 1)
        return p, (t or None)
    return spec, None


def run_phases(preset, tag, start, end, phases):
    from factor_lab.paper_trader import PaperTrader
    rows = []
    for ph in range(phases):
        t = PaperTrader(state_dir=tempfile.mkdtemp(),
                        pred_tag=tag, preset=preset)
        perf = t.replay(start, end, verbose=False, phase=ph, save=False)
        if 'error' in perf:
            print(f"    相位{ph} 失败: {perf['error']}")
            continue
        rows.append(perf)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description='因子集配对比较')
    ap.add_argument('--base', required=True, help='基线 preset[:tag]')
    ap.add_argument('--cand', required=True, help='候选 preset[:tag]')
    ap.add_argument('--phases', type=int, default=8)
    ap.add_argument('--start', default='2024-01-02')
    ap.add_argument('--end', default='2026-09-04')
    ap.add_argument('--tag', default='主段')
    args = ap.parse_args()

    import qlib
    from qlib.constant import REG_CN
    qlib.init(provider_uri=str(Path.home() / '.qlib/qlib_data/cn_data_bs'),
              region=REG_CN)
    from factor_lab.signal_generator import SignalGenerator
    import pandas as pd

    bp, bt = parse_spec(args.base)
    cp, ct = parse_spec(args.cand)

    # 覆盖性检查 —— 两份缓存都必须覆盖请求区间，否则会静默跑出纯现金结果
    for label, (pre, tg) in (('基线', (bp, bt)), ('候选', (cp, ct))):
        sg = SignalGenerator(pred_tag=tg, preset=pre)
        try:
            sd = pd.Index(sg.load_predictions().index.get_level_values(0).unique())
        except FileNotFoundError as e:
            print(f"✗ {label} {pre}:{tg} 预测缓存不存在\n  {e}")
            return 2
        cov = sd[(sd >= pd.Timestamp(args.start)) & (sd <= pd.Timestamp(args.end))]
        print(f"{label} {pre}:{tg or '-'}  覆盖 {sd.min().date()}~{sd.max().date()}"
              f"  区间内 {len(cov)} 个信号日")
        if len(cov) < 20:
            print(f"✗ {label} 只覆盖请求区间 {len(cov)} 天，无法比较")
            return 2

    print("=" * 76)
    print(f"因子集配对比较  {args.tag}  {args.start} ~ {args.end}  "
          f"{args.phases} 相位")
    print("=" * 76)

    t0 = time.time()
    base = run_phases(bp, bt, args.start, args.end, args.phases)
    cand = run_phases(cp, ct, args.start, args.end, args.phases)
    n = min(len(base), len(cand))
    if n < 2:
        print("✗ 有效相位不足")
        return 1

    def summ(name, rows):
        sh = [r['sharpe'] for r in rows]
        ex = [r['excess_return'] or 0 for r in rows]
        dd = [r['max_drawdown'] for r in rows]
        print(f"  {name:<26} Sharpe 均值 {statistics.mean(sh):6.3f}  "
              f"最差 {min(sh):6.3f}  回撤 {statistics.mean(dd):7.2%}  "
              f"超额 {statistics.mean(ex):+7.2%}")

    print()
    summ(f"基线 {bp}", base)
    summ(f"候选 {cp}", cand)

    d_sh = [cand[i]['sharpe'] - base[i]['sharpe'] for i in range(n)]
    d_dd = [cand[i]['max_drawdown'] - base[i]['max_drawdown'] for i in range(n)]
    d_ex = [(cand[i]['excess_return'] or 0) - (base[i]['excess_return'] or 0)
            for i in range(n)]
    t = paired_t(d_sh)
    wins = sum(1 for d in d_sh if d > 0)

    print()
    print("-" * 76)
    print(f"配对差值 (候选 − 基线)，{n} 个相位")
    print(f"  ΔSharpe  均值 {statistics.mean(d_sh):+.3f}   配对 t = {t:.2f}   "
          f"胜出 {wins}/{n}")
    print(f"  Δ回撤    均值 {statistics.mean(d_dd):+.2%}")
    print(f"  Δ超额    均值 {statistics.mean(d_ex):+.2%}")
    print(f"  逐相位 ΔSharpe: {[round(d, 3) for d in d_sh]}")
    print("-" * 76)
    if abs(t) < 2:
        print("判定: |t| < 2 —— 差异在噪声范围内，不足以据此换因子集。")
    elif t > 0:
        print("判定: 候选显著更优。仍须另一段样本同向，且经影子验证后才可上线。")
    else:
        print("判定: 候选显著更差。")
    print(f"耗时 {time.time()-t0:.0f}s")

    out = PROJECT_DIR / 'factor_lab' / 'results' / f'preset_cmp_{args.tag}.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        json.dump({'base': args.base, 'cand': args.cand,
                   'start': args.start, 'end': args.end,
                   'd_sharpe': d_sh, 'paired_t': t, 'wins': wins, 'n': n,
                   'd_drawdown': d_dd, 'd_excess': d_ex},
                  f, ensure_ascii=False, indent=1)
    print(f"结果已写入 {out}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
