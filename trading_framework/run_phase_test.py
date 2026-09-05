"""相位扫描 — 用模拟盘引擎跑 N 个起始相位，报均值而非单次结果

为什么必须这样报
----------------
8 日调仓在 2.5 年样本上只有约 81 次调仓。起始日错开 1 天(相位 0~7)就是一组
同样合理、但样本不同的结果。CLAUDE.md 记录: 相位间 Sharpe 标准差 0.11~0.32，
而候选参数之间的差异只有 0.1~0.16 —— **差异完全淹没在噪声里**。

具体翻车例: 段1 的 TopK=8 曾录得 Sharpe 1.341(相位0)，8 相位均值只有 0.840、
最差相位 0.29。据此认为"8 只在熊市很强"是错的。

为什么用 paper_trader
---------------------
它是被 reconcile.py 逐日证明与实盘决策一致的引擎 (82 个调仓日 0 分叉)。
run_vol_target.py 是另一个引擎，没有进对账 —— 它的数字未必描述实盘策略。
报绩效应当用能证明"跑的就是实盘那套规则"的那个引擎。

用法
----
    python run_phase_test.py                              # 主段，8 相位
    python run_phase_test.py --start 2022-05-01 --end 2023-12-31 --tag 段1
    python run_phase_test.py --phases 4                   # 少跑几个相位
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
    ap.add_argument('--out', help='结果 JSON 输出路径')
    args = ap.parse_args()

    import qlib
    from qlib.constant import REG_CN
    qlib.init(provider_uri=str(Path.home() / '.qlib/qlib_data/cn_data_bs'),
              region=REG_CN)

    from factor_lab.paper_trader import PaperTrader

    print("=" * 66)
    print(f"相位扫描  {args.tag}  {args.start} ~ {args.end}  "
          f"{args.phases} 个相位")
    print("=" * 66)

    import tempfile
    rows = []
    t0 = time.time()
    for ph in range(args.phases):
        # 每个相位一个独立临时目录 —— 绝不写真实模拟盘状态
        trader = PaperTrader(state_dir=tempfile.mkdtemp(prefix=f'phase{ph}_'))
        # save=False: 相位扫描不能覆盖真实模拟盘状态
        perf = trader.replay(args.start, args.end, verbose=False,
                             phase=ph, save=False)
        if 'error' in perf:
            print(f"  相位{ph}: 失败 — {perf['error']}")
            continue
        rows.append({
            'phase': ph,
            'sharpe': perf['sharpe'],
            'total_return': perf['total_return'],
            'max_drawdown': perf['max_drawdown'],
            'excess_return': perf.get('excess_return'),
            'n_trades': perf.get('n_trades'),
        })
        print(f"  相位{ph}: Sharpe {perf['sharpe']:6.3f}  "
              f"收益 {perf['total_return']:+7.2%}  "
              f"回撤 {perf['max_drawdown']:7.2%}  "
              f"交易 {perf.get('n_trades', 0):4d}", flush=True)

    if not rows:
        print("\n✗ 全部相位失败")
        return 1

    sh = [r['sharpe'] for r in rows]
    ret = [r['total_return'] for r in rows]
    dd = [r['max_drawdown'] for r in rows]
    ex = [r['excess_return'] for r in rows if r['excess_return'] is not None]

    print()
    print("-" * 66)
    print(f"{args.tag}  {len(rows)} 个相位  (耗时 {time.time()-t0:.0f}s)")
    print(f"  Sharpe   均值 {statistics.mean(sh):6.3f}   "
          f"标准差 {statistics.pstdev(sh):5.3f}   "
          f"最差 {min(sh):6.3f}   最好 {max(sh):6.3f}")
    print(f"  总收益   均值 {statistics.mean(ret):+7.2%}   "
          f"最差 {min(ret):+7.2%}   最好 {max(ret):+7.2%}")
    print(f"  最大回撤 均值 {statistics.mean(dd):7.2%}   最差 {min(dd):7.2%}")
    if ex:
        print(f"  超额     均值 {statistics.mean(ex):+7.2%}   "
              f"最差 {min(ex):+7.2%}")
    print("-" * 66)
    print("报绩效请用均值与最差相位，不要用单个相位的数字。")

    out = args.out or str(PROJECT_DIR / 'factor_lab' / 'results' /
                          f'phase_{args.tag}.json')
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        json.dump({'tag': args.tag, 'start': args.start, 'end': args.end,
                   'rows': rows,
                   'sharpe_mean': statistics.mean(sh),
                   'sharpe_std': statistics.pstdev(sh),
                   'sharpe_worst': min(sh)}, f, ensure_ascii=False, indent=1)
    print(f"结果已写入 {out}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
