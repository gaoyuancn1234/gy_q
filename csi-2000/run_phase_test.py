"""相位扫描 — paper_trader 多起点, 报均值/最差和多维指标。

单相位 Sharpe 噪声经常大于参数差。不要用 run_vol_target 报绩效,
那个引擎没进 reconcile。

    python run_phase_test.py
    python run_phase_test.py --start 2022-05-01 --end 2023-12-31 --tag 段1
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import qlib_compat  # noqa: F401

PROJECT_DIR = Path(__file__).parent


def main() -> int:
    ap = argparse.ArgumentParser(description='相位扫描')
    ap.add_argument('--start', default='2024-01-02')
    ap.add_argument('--end', default='2026-09-04')
    ap.add_argument('--phases', type=int, default=8)
    ap.add_argument('--tag', default='主段')
    ap.add_argument('--allow-lookahead', action='store_true',
                    help='exec_lag=0 且无截断日线时仍跑。结果有前视，不得写进验收表')
    ap.add_argument('--pred-tag', default=None,
                    help='预测缓存后缀，如 pre2024 (段1)。默认用实盘那份')
    ap.add_argument('--preset', default=None,
                    help='预测缓存的 preset 名，如 alpha158_selected。默认用配置里的')
    ap.add_argument('--out', help='结果 JSON 输出路径')
    ap.add_argument(
        '--vol-target',
        default=None,
        help='覆盖 yaml 的 vol_target。off 表示关闭。不写回配置。',
    )
    ap.add_argument(
        '--vol-unknown',
        default=None,
        help='覆盖估不出波动率时的敞口。不写回配置。',
    )
    ap.add_argument(
        '--n-trials',
        type=int,
        default=6,
        help='已尝试独立配置数, 进 DSR。默认含本轮扫描。',
    )
    args = ap.parse_args()
    overrides = {}
    if args.vol_target is not None:
        raw = str(args.vol_target).strip().lower()
        if raw in ("off", "none", "null"):
            overrides["vol_target"] = None
        else:
            overrides["vol_target"] = float(args.vol_target)
    if args.vol_unknown is not None:
        overrides["vol_unknown_exposure"] = float(args.vol_unknown)

    import qlib
    from qlib.constant import REG_CN
    from qlib_paths import qlib_provider_uri
    qlib.init(provider_uri=qlib_provider_uri(), region=REG_CN)

    from factor_lab.paper_trader import PaperTrader

    print("=" * 66)
    print(f"相位扫描  {args.tag}  {args.start} ~ {args.end}  "
          f"{args.phases} 个相位")
    if overrides:
        print(f"覆盖: {overrides}")
    print("=" * 66)

    # 覆盖性检查 —— 预测缓存不覆盖请求区间时必须立刻失败。
    # 否则每个交易日都 `date not in signal_dates`、一次调仓都不发生，
    # 回测照样跑完并给出一个纯粹由现金构成的 Sharpe。这正是本项目
    # 反复出现的沉默失败: 看起来跑完了，实际什么都没做。
    import pandas as pd
    from factor_lab.signal_generator import SignalGenerator
    _sg = SignalGenerator(pred_tag=args.pred_tag, preset=args.preset)
    _sd = pd.Index(_sg.load_predictions().index.get_level_values(0).unique())
    _lo, _hi = _sd.min(), _sd.max()
    _req_lo, _req_hi = pd.Timestamp(args.start), pd.Timestamp(args.end)
    _covered = _sd[(_sd >= _req_lo) & (_sd <= _req_hi)]
    print(f"预测缓存 {'(默认)' if not args.pred_tag else args.pred_tag}: "
          f"{_lo.date()} ~ {_hi.date()}，区间内 {len(_covered)} 个信号日")
    if len(_covered) < 20:
        print(f"\n✗ 预测缓存只覆盖请求区间 {args.start}~{args.end} 的 "
              f"{len(_covered)} 天，无法回测。")
        print(f"  可用缓存区间是 {_lo.date()} ~ {_hi.date()}；"
              f"段1 请加 --pred-tag pre2024")
        return 2

    import tempfile
    rows = []
    t0 = time.time()
    for ph in range(args.phases):
        # 每个相位一个独立临时目录 —— 绝不写真实模拟盘状态
        trader = PaperTrader(
            state_dir=tempfile.mkdtemp(prefix=f'phase{ph}_'),
            pred_tag=args.pred_tag,
            preset=args.preset,
            config_overrides=overrides or None,
            n_trials=args.n_trials,
        )
        # save=False: 相位扫描不能覆盖真实模拟盘状态
        perf = trader.replay(args.start, args.end, verbose=False,
                             allow_lookahead=args.allow_lookahead,
                             phase=ph, save=False)
        if 'error' in perf:
            print(f"  相位{ph}: 失败 — {perf['error']}")
            continue
        rows.append({
            'phase': ph,
            'sharpe': perf['sharpe'],
            'sortino': perf.get('sortino'),
            'calmar': perf.get('calmar'),
            'dsr': perf.get('dsr'),
            'psr': perf.get('psr'),
            'ir': perf.get('ir'),
            'beta': perf.get('beta'),
            'cvar': perf.get('cvar'),
            'omega': perf.get('omega'),
            'martin': perf.get('martin'),
            'ann_vol': perf.get('ann_vol'),
            'dd_days': perf.get('dd_days'),
            'total_return': perf['total_return'],
            'max_drawdown': perf['max_drawdown'],
            'excess_return': perf.get('excess_return'),
            'bench_return': perf.get('bench_return'),
            'n_trades': perf.get('n_trades'),
            'n_trials': perf.get('n_trials'),
        })
        ex = perf.get("excess_return")
        ex_s = f"  超额 {ex:+7.2%}" if ex is not None else ""
        dsr = perf.get("dsr")
        dsr_s = f"  DSR {dsr:5.3f}" if dsr is not None else ""
        so = perf.get("sortino")
        so_s = f"  So {so:5.3f}" if so is not None else ""
        print(f"  相位{ph}: Sharpe {perf['sharpe']:6.3f}  "
              f"收益 {perf['total_return']:+7.2%}  "
              f"回撤 {perf['max_drawdown']:7.2%}  "
              f"交易 {perf.get('n_trades', 0):4d}{ex_s}{so_s}{dsr_s}",
              flush=True)

    if not rows:
        print("\n✗ 全部相位失败")
        return 1

    def _vals(key):
        return [r[key] for r in rows if r.get(key) is not None]

    def _line(name, xs, pct=False, higher=True):
        if not xs:
            return
        worst = min(xs) if higher else max(xs)
        if pct:
            print(f"  {name:<8} 均值 {statistics.mean(xs):+7.2%}   "
                  f"最差 {worst:+7.2%}")
            return
        print(f"  {name:<8} 均值 {statistics.mean(xs):6.3f}   "
              f"最差 {worst:6.3f}")

    sh = _vals('sharpe')
    ret = _vals('total_return')
    dd = _vals('max_drawdown')
    ex = _vals('excess_return')
    # 超额全缺 = 基准没取到。此时 Sharpe/收益/回撤照样能打印, 报告看着正常,
    # 只是少了一行 —— 这种"看起来跑完了"的结果不能进验收。
    if rows and not ex:
        print()
        print("✗ 全部相位都没有超额 —— 基准未取到, 本次结果不可用于验收。")
        print("  csi2000 基准 932000.CSI 只能走 Tushare, 检查网络/配额后重跑。")
        return 2

    print()
    print("-" * 66)
    print(f"{args.tag}  {len(rows)} 个相位  (耗时 {time.time()-t0:.0f}s)"
          f"  DSR试次={args.n_trials}")
    print(f"  Sharpe   均值 {statistics.mean(sh):6.3f}   "
          f"标准差 {statistics.pstdev(sh):5.3f}   "
          f"最差 {min(sh):6.3f}   最好 {max(sh):6.3f}")
    _line("紧缩DSR", _vals('dsr'), higher=True)
    _line("PSR", _vals('psr'), higher=True)
    _line("Sortino", _vals('sortino'), higher=True)
    _line("Calmar", _vals('calmar'), higher=True)
    _line("IR", _vals('ir'), higher=True)
    print(f"  总收益   均值 {statistics.mean(ret):+7.2%}   "
          f"最差 {min(ret):+7.2%}   最好 {max(ret):+7.2%}")
    print(f"  最大回撤 均值 {statistics.mean(dd):7.2%}   "
          f"最差 {min(dd):7.2%}")
    _line("日CVaR5", _vals('cvar'), pct=True, higher=True)
    if ex:
        print(f"  超额     均值 {statistics.mean(ex):+7.2%}   "
              f"最差 {min(ex):+7.2%}")
    print("-" * 66)
    print("选参看均值+最差, 以及 DSR/Sortino/Calmar/IR, 不用单相位。")

    out = args.out or str(PROJECT_DIR / 'factor_lab' / 'results' /
                          f'phase_{args.tag}.json')
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    summary = {
        'tag': args.tag,
        'start': args.start,
        'end': args.end,
        'n_trials': args.n_trials,
        'rows': rows,
        'sharpe_mean': statistics.mean(sh),
        'sharpe_std': statistics.pstdev(sh),
        'sharpe_worst': min(sh),
    }
    for key in ('dsr', 'psr', 'sortino', 'calmar', 'ir', 'cvar'):
        xs = _vals(key)
        if xs:
            summary[f'{key}_mean'] = statistics.mean(xs)
            summary[f'{key}_worst'] = min(xs)
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(f"结果已写入 {out}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
