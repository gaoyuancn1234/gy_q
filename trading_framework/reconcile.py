"""双路径对账 — 实盘决策与回测决策必须逐日一致

为什么需要这个
--------------
本系统反复出现同一类 bug: **同一条规则被写了两遍，改了一处忘了另一处**。
已发生过的实例:

    n_drop / vol_target      回测有、实盘有、模拟盘没有      (2026-09-03)
    _exposure                回测一份、实盘一份，回退方向相反  (2026-09-05)
    deal_price               研究用 open、生产用 close        (2026-08-30)
    调仓判定                  回测按交易日、实盘按运行次数      (2026-09-05)
    pending_orders           推送走 n_drop、存盘走全量换手     (2026-09-05)

最后一条最能说明问题: 18:00 飞书推"卖4买4"，次日 9:30 盘中监控读
pending_orders 提醒"待执行 卖12买12" —— 同一次调仓两条互相矛盾的指令，
且后者正是回测中吃掉本金约 14% 的全量换手。它安静地跑了两天没人发现。

这些 bug 的共同点是**不需要被预测，只需要被对账**。逐条去猜"下一处会在哪
分叉"是猜不完的; 让两条路径每天对同一份输入各算一遍并比对，则不管分叉出现
在代码、配置还是数据里，都会当场暴露。

做法
----
驱动回测引擎逐日重放。每个调仓日:
  1. 调用 update_daily **之前**，用当时的持仓状态 + 当日信号，走**实盘**
     的 compute_rebalance_orders 算一份订单
  2. 调用 update_daily **之后**，读回测器排进 pending_orders 的订单
  3. 两者的 sells / buys 必须完全相同

这是行为对比而非结构对比: 即便分叉源自配置读取、数据对齐或迭代顺序，
而不是重复代码，也一样会被抓到。

用法
----
    python reconcile.py                    # 对账最近 120 个交易日
    python reconcile.py --days 400         # 更长区间
    python reconcile.py --start 2025-01-01 --end 2025-06-30
    python reconcile.py --push             # 不一致时推送飞书

退出码: 0 = 一致; 1 = 发现分叉; 2 = 无法完成对账(数据缺失等)
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import qlib_compat  # noqa: F401  设置 MLFLOW_ALLOW_FILE_STORE

PROJECT_DIR = Path(__file__).parent


# ---------------------------------------------------------------- 配置对账

def check_config_consistency() -> list[str]:
    """两条路径读到的风控参数必须相同

    实盘经 live_portfolio._get_vol_target_config() 读，回测直接读 yaml。
    两者若对同一个 key 给出不同默认值，后面的订单对账会一路正常、
    实际策略却不同。
    """
    import yaml
    from portfolio.live_portfolio import _get_vol_target_config

    problems = []
    with open(PROJECT_DIR / 'config/signal_config.yaml', encoding='utf-8') as f:
        raw = yaml.safe_load(f)
    live = _get_vol_target_config()

    for key in ('n_drop', 'vol_target'):
        bt_val = raw.get(key)
        live_val = live.get(key)
        # 类型可以不同 (int vs float)，值必须相等
        same = (bt_val is None and live_val is None) or (
            bt_val is not None and live_val is not None
            and float(bt_val) == float(live_val))
        if not same:
            problems.append(
                f"配置分叉 {key}: 回测读到 {bt_val!r}，实盘读到 {live_val!r}")
    return problems


# ---------------------------------------------------------------- 订单对账

def reconcile_orders(start: str, end: str, verbose: bool = True) -> dict:
    """逐日比对实盘决策与回测决策"""
    from qlib.data import D
    from factor_lab.paper_trader import PaperTrader
    from portfolio.live_portfolio import compute_rebalance_orders

    # 临时状态目录 —— 对账是只读检查，不能碰真实模拟盘的持仓与历史。
    # replay()/reset() 会写 state.json 并删除 trades.csv / daily_nav.csv。
    import tempfile
    _tmp = tempfile.mkdtemp(prefix='reconcile_')
    trader = PaperTrader(state_dir=_tmp)
    cfg = trader.config
    trader.reset()

    instruments = D.instruments(cfg.get('instruments', 'csi300'))
    prices = D.features(instruments, ['$close'], start_time=start, end_time=end)
    if prices is None or prices.empty:
        return {'status': 'error', 'reason': f'{start}~{end} 无行情数据'}
    prices = prices.swaplevel().sort_index()

    signal = trader.sg.load_predictions()
    quality = trader.sg.load_quality_score()
    trading_days = sorted(prices.index.get_level_values(0).unique())
    if not trading_days:
        return {'status': 'error', 'reason': '交易日列表为空'}

    diffs = []
    decisions = []          # 逐个调仓日的订单，供跨进程复现性比对
    n_rebalance = 0
    prev_close_map = {}

    for day_idx, date in enumerate(trading_days):
        trader.last_decision = None
        trader.update_daily(date, prices, prev_close_map, day_idx, signal, quality)

        # 回测在"执行完昨日挂单之后"才做决策，所以只能用它自己记下的
        # 决策时刻快照来比 —— 从外部取快照会差一步状态，比出来的是
        # 测量偏差而非真分叉。
        dec = getattr(trader, 'last_decision', None)
        if dec:
            n_rebalance += 1
            # 用回测决策时刻的完全相同输入，走实盘那条路径再算一遍
            sig = {
                'target_stocks': dec['target_stocks'],
                'effective_topk': dec['effective_topk'],
                'scores': dec['scores'],
            }
            holdings_view = {
                'positions': {c: {} for c in dec['positions']},
                # pending_sells 是决策**之后**的并集，需还原成决策前的止损单:
                # 即已挂但不是本次调仓选出来的那些
                'pending_orders': {
                    'sells': [c for c in dec['pending_sells']
                              if c not in set(dec['sells'])],
                    'buys': {},
                },
            }
            expected = compute_rebalance_orders(sig, holdings_view)

            exp_sells = sorted(expected['sells'])
            exp_buys = sorted(expected['buys'])
            got_sells = sorted(dec['pending_sells'])
            got_buys = sorted(dec['buys'])
            decisions.append({'date': dec['date'],
                              'sells': got_sells, 'buys': got_buys})
            if got_sells != exp_sells or got_buys != exp_buys:
                diffs.append({
                    'date': dec['date'],
                    'live_sells': exp_sells, 'bt_sells': got_sells,
                    'live_buys': exp_buys, 'bt_buys': got_buys,
                })

        day_close = prices.loc[date]['$close'].dropna() \
            if date in prices.index.get_level_values(0) else None
        if day_close is not None:
            for inst in day_close.index:
                prev_close_map[inst] = day_close[inst]

        if verbose and day_idx and day_idx % 50 == 0:
            print(f"  ... {day_idx}/{len(trading_days)} 天，"
                  f"已比对 {n_rebalance} 个调仓日，分叉 {len(diffs)}", flush=True)

    return {
        'status': 'ok',
        'days': len(trading_days),
        'rebalances': n_rebalance,
        'diffs': diffs,
        'decisions': decisions,
    }


# ---------------------------------------------------------------- 入口

def main() -> int:
    ap = argparse.ArgumentParser(description='实盘/回测双路径对账')
    ap.add_argument('--days', type=int, default=120, help='回溯交易日数')
    ap.add_argument('--start', help='起始日 (覆盖 --days)')
    ap.add_argument('--end', help='结束日')
    ap.add_argument('--push', action='store_true', help='发现分叉时推送飞书')
    ap.add_argument('--dump', help='把逐日订单写入该 JSON，用于跨进程复现性比对')
    args = ap.parse_args()

    import qlib
    from qlib.constant import REG_CN
    qlib.init(provider_uri=str(Path.home() / '.qlib/qlib_data/cn_data_bs'),
              region=REG_CN)

    from factor_lab.signal_generator import SignalGenerator
    import pandas as pd

    sg = SignalGenerator()
    sig_dates = sorted(pd.Index(sg.load_predictions()
                                .index.get_level_values(0).unique()))
    if not sig_dates:
        print("✗ 无预测缓存，无法对账")
        return 2

    end = args.end or sig_dates[-1].strftime('%Y-%m-%d')
    if args.start:
        start = args.start
    else:
        idx = max(0, len(sig_dates) - args.days)
        start = sig_dates[idx].strftime('%Y-%m-%d')

    print("=" * 60)
    print(f"双路径对账  {start} ~ {end}")
    print("=" * 60)

    print("\n[1/2] 配置一致性")
    cfg_problems = check_config_consistency()
    if cfg_problems:
        for p in cfg_problems:
            print(f"  ✗ {p}")
    else:
        print("  ✓ 实盘与回测读到的 n_drop / vol_target 一致")

    print("\n[2/2] 逐日订单一致性")
    res = reconcile_orders(start, end)
    if res['status'] != 'ok':
        print(f"  ✗ 对账无法完成: {res['reason']}")
        return 2

    diffs = res['diffs']
    print(f"\n  交易日 {res['days']}，调仓日 {res['rebalances']}，"
          f"分叉 {len(diffs)}")

    if args.dump:
        import json
        with open(args.dump, 'w', encoding='utf-8') as f:
            json.dump(res['decisions'], f, ensure_ascii=False, indent=1)
        print(f"  逐日订单已写入 {args.dump}")

    if diffs:
        print("\n  分叉明细 (最多列 5 个):")
        for d in diffs[:5]:
            print(f"    {d['date']}")
            print(f"      卖出  实盘 {d['live_sells']}")
            print(f"           回测 {d['bt_sells']}")
            print(f"      买入  实盘 {d['live_buys']}")
            print(f"           回测 {d['bt_buys']}")

    ok = not diffs and not cfg_problems
    print()
    if ok:
        print("✓ 两条路径逐日一致")
    else:
        print(f"✗ 发现 {len(cfg_problems)} 处配置分叉、{len(diffs)} 个调仓日订单分叉")
        print("  实盘与回测跑的不是同一个策略 —— 回测绩效不代表实盘。")

    if args.push and not ok:
        try:
            from send_signal import push_feishu
            msg = (f"🚨 双路径对账失败 ({start}~{end})\n"
                   f"配置分叉 {len(cfg_problems)} 处，订单分叉 {len(diffs)} 个调仓日\n"
                   f"实盘与回测跑的不是同一个策略，回测绩效不代表实盘。")
            push_feishu(msg, dry_run=False)
        except Exception as e:
            print(f"  (飞书推送失败: {e})")

    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
