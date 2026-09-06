#!/usr/bin/env python3
"""掘金仿真桥接 — 把 pending_orders 送进掘金仿真账户，并回读真实成交

为什么要接这个
--------------
整套回测的核心假设是"**次日收盘价成交**"。这个假设已被证明是敏感的:
one-switch 实验里改成开盘价，超额从 27.11% 掉到 11.90% —— 腰斩。
但收盘价本身能不能兑现，回测无法自证:它没有盘口、没有排队、没有部分成交。

掘金仿真的**精准撮合**按实时盘口、价格优先 + 时间优先撮合，委托价落在本方
盘口时进入排队(前面的量 = 同价位盘口挂单量)。接上之后能直接量出:
同一批单子的实际成交价与收盘价差多少、有没有干脆没成交。

已知限制: 仿真不模拟**交易冲击成本**，大额委托的成交价会优于实际。
对 10 万资金 / 16 只 / 每只约 6 千元买 CSI300 成分股而言，真实冲击本就接近零，
这条限制不影响结论。

架构
----
必须作为**掘金策略**运行 —— 账户接口(get_cash/get_position/order_volume)
只在 run(strategy_id=..., mode=MODE_LIVE) 启动的策略里返回数据，独立调用
一律返回空。所以本文件是一个独立进程，与 daily_runner 隔离:

    daily_runner (18:00)  ->  holdings['pending_orders']  ->  gm_bridge  ->  掘金仿真
                                                                  |
                                              成交回报  ->  gm_state.json

隔离的理由: 掘金终端必须常驻，它掉线/退出登录时下单会失败。把它放进
daily_runner 会让"推送"这条关键路径依赖一个额外的常驻进程。

**凡是对外发生效果的动作，必须检查返回值再报成功。** CLAUDE.md 记过一次
教训: 飞书推送连续数周从未生效，因为代码只 log 一行就 return，退出码仍是 0。
本模块下单后一律回查委托状态，不拿"没抛异常"当成交。

用法
----
    python gm_bridge.py --check          # 只查账户，不下单
    python gm_bridge.py --place          # 读 pending_orders 并下单
    python gm_bridge.py --sync           # 回读成交，写入 gm_state.json

.env 需要: GM_TOKEN / GM_ACCOUNT_ID / GM_STRATEGY_ID
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

STATE_FILE = PROJECT_DIR / 'gm_state.json'
HOLDINGS_FILE = PROJECT_DIR / 'portfolio' / 'live_holdings.json'

# 掘金的标的写法是 SHSE.600036 / SZSE.000001，本项目内部是 SH600036 / SZ000001
_EX = {'SH': 'SHSE', 'SZ': 'SZSE'}


def to_gm_symbol(code: str) -> str:
    """SH600036 -> SHSE.600036"""
    pre, num = code[:2].upper(), code[2:]
    if pre not in _EX:
        raise ValueError(f'无法识别的标的代码: {code}')
    return f'{_EX[pre]}.{num}'


def from_gm_symbol(sym: str) -> str:
    """SHSE.600036 -> SH600036"""
    ex, num = sym.split('.')
    rev = {v: k for k, v in _EX.items()}
    return f'{rev[ex]}{num}'


def env(key: str, required: bool = True):
    f = PROJECT_DIR / '.env'
    if f.exists():
        for line in f.read_text(encoding='utf-8').splitlines():
            if line.startswith(key + '='):
                v = line.split('=', 1)[1].strip()
                if v:
                    return v
    if required:
        raise RuntimeError(f'.env 缺少 {key}')
    return None


def _load_holdings() -> dict:
    if not HOLDINGS_FILE.exists():
        return {}
    return json.loads(HOLDINGS_FILE.read_text(encoding='utf-8'))


LOG_FILE = PROJECT_DIR / 'logs' / 'gm_bridge.log'
_LOG_BUF = []


def log(msg: str = ''):
    """写日志文件而不是 print

    掘金策略上下文里 print 不输出(实测 --place 全程无任何 stdout，
    连异常都看不到)，只能靠落盘。凡是对外发生效果的动作必须留下痕迹，
    否则失败与"没跑"无法区分 —— 本项目已在飞书推送、任务退出码、
    数据下载三处栽过同一个坑。
    """
    _LOG_BUF.append(str(msg))
    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        LOG_FILE.write_text('\n'.join(_LOG_BUF), encoding='utf-8')
    except OSError:
        pass


def _save_state(payload: dict):
    payload['updated'] = datetime.now().isoformat(timespec='seconds')
    STATE_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                          encoding='utf-8')


# ---------------------------------------------------------------- 策略回调

def init(context):
    """掘金策略入口。run() 启动后由掘金调用。"""
    import traceback
    from datetime import datetime as _dt
    _LOG_BUF.clear()
    # 用环境变量传参 —— gm.run(filename=...) 会**按文件名重新导入本模块**，
    # main() 里对 sys.modules[__name__].init 的替换在新副本里不存在，
    # context 上挂的属性也读不到。实测因此一直走 'check' 分支:
    # 传 --place --dry-run，日志却打印 "gm_bridge check"。
    import os as _os
    action = _os.environ.get('GM_BRIDGE_ACTION', 'check')
    dry = _os.environ.get('GM_BRIDGE_DRY') == '1'
    log(f'=== gm_bridge {action}{" (空跑)" if dry else ""} '
        f'{_dt.now():%Y-%m-%d %H:%M:%S} ===')

    from gm.api import set_account_id
    set_account_id(env('GM_ACCOUNT_ID'))
    log(f'account_id 已设置')

    try:
        if action == 'check':
            _do_check()
        elif action == 'place':
            _do_place(dry_run=dry)
        elif action == 'sync':
            _do_sync()
    except Exception as e:
        # 掘金会吞掉策略回调里的异常 —— 不写下来就完全看不到
        log(f'✗ 异常: {type(e).__name__}: {e}')
        log(traceback.format_exc())
        _save_state({'status': 'error', 'action': action,
                     'error': f'{type(e).__name__}: {e}'})
    finally:
        # run() 会一直阻塞等行情事件。本桥接是一次性动作，做完必须主动停，
        # 否则进程挂住 —— 实测被 timeout 杀掉时退出码 124，看起来像失败。
        log('=== 结束 ===')
        from gm.api import stop
        stop()


def _account_snapshot() -> dict:
    from gm.api import get_cash, get_position
    cash = get_cash() or {}
    pos = get_position() or []
    return {
        'nav': cash.get('nav'),
        'available': cash.get('available'),
        'market_value': cash.get('market_value'),
        'positions': [
            {'code': from_gm_symbol(p['symbol']), 'volume': p.get('volume'),
             'vwap': p.get('vwap'), 'price': p.get('price')}
            for p in pos
        ],
    }


def _do_check():
    snap = _account_snapshot()
    if snap['nav'] is None:
        log('✗ 账户无数据 — 检查 GM_ACCOUNT_ID 是否与该策略绑定的仿真账户一致')
        _save_state({'status': 'no_account', 'snapshot': snap})
        return
    log(f"✓ 总资产 {snap['nav']:,.0f}  可用 {snap['available']:,.0f}  "
          f"持仓 {len(snap['positions'])} 只")
    for p in snap['positions']:
        log(f"    {p['code']} {p['volume']}股 成本 {p['vwap']} 现价 {p['price']}")
    _save_state({'status': 'ok', 'snapshot': snap})


def _do_place(dry_run: bool = False):
    """把 pending_orders 送进仿真账户，逐笔回查委托状态

    dry_run=True 时走完全部计算(取价、敞口、分配、股数)但不提交委托。
    """
    from gm.api import order_volume, get_orders, OrderSide_Buy, OrderSide_Sell, \
        OrderType_Limit, PositionEffect_Open, PositionEffect_Close

    holdings = _load_holdings()
    pending = holdings.get('pending_orders') or {}
    sells = list(pending.get('sells') or [])
    buys = dict(pending.get('buys') or {})

    if not sells and not buys:
        log('无待执行订单')
        _save_state({'status': 'no_orders'})
        return

    log(f'待执行: 卖 {len(sells)} 只，买 {len(buys)} 只')

    # 卖出用现有持仓股数; 买入股数由 live_portfolio 的分配决定，
    # 这里从 holdings 里取不到，改为按当前可用资金等分 —— 与实盘同一函数
    from portfolio.rebalance_rules import allocate_buys
    from portfolio.live_portfolio import _get_open_cost

    snap = _account_snapshot()
    if snap['nav'] is None:
        log('✗ 账户无数据，拒绝下单')
        _save_state({'status': 'no_account'})
        return

    held = {p['code']: p['volume'] for p in snap['positions']}
    placed = []

    # 波动率目标 —— 必须与实盘同一口径。
    # 2026-09-06: 初版直接用 snap['available'] 全额分配，而 daily_runner
    # 生成指令时已按 vol_target 把可用资金缩到 60%。两边不一致就等于在
    # 仿真里跑另一个策略，测出来的滑点无法与回测比较 —— 正是 reconcile.py
    # 专门在防的那类分叉。
    from portfolio.live_portfolio import compute_exposure as _live_exposure
    from portfolio.live_portfolio import _get_vol_target_config
    _vt = _get_vol_target_config()
    cash_for_buy = float(snap['available'] or 0)
    exposure, realized = 1.0, None
    if _vt.get('vol_target'):
        exposure, realized = _live_exposure(
            holdings, _vt['vol_target'],
            window=_vt.get('vol_window', 20),
            min_exposure=_vt.get('vol_min_exposure', 0.2))
        if exposure < 1.0:
            cash_for_buy *= exposure
    _rz = f'{realized:.1%}' if realized is not None else '无法估计(净值历史不足)'
    log(f'  敞口 {exposure:.0%} (实现波动 {_rz})，'
          f'可用 {snap["available"]:,.0f} -> 买入预算 {cash_for_buy:,.0f}')

    # --- 卖出 ---
    for code in sells:
        vol = held.get(code, 0)
        if vol <= 0:
            log(f'  跳过卖出 {code}: 仿真账户无持仓')
            continue
        if dry_run:
            log(f'  [空跑] 卖 {code} {int(vol)}股')
            placed.append({'code': code, 'side': 'SELL', 'volume': int(vol),
                           'resp': None, 'dry_run': True})
            continue
        o = order_volume(symbol=to_gm_symbol(code), volume=int(vol),
                         side=OrderSide_Sell, order_type=OrderType_Limit,
                         position_effect=PositionEffect_Close, price=0)
        placed.append({'code': code, 'side': 'SELL', 'volume': int(vol),
                       'resp': _order_id(o)})

    # --- 买入 ---
    if buys:
        from gm.api import current
        codes = list(buys.keys())
        quotes = current(symbols=[to_gm_symbol(c) for c in codes]) or []
        px = {from_gm_symbol(q['symbol']): q.get('price') for q in quotes}
        noquote = [c for c in codes if not px.get(c)]
        if noquote:
            log(f'  ⚠ {len(noquote)} 只取不到实时价，会被分配函数跳过: {noquote}')
        alloc = allocate_buys(codes, px, cash_for_buy,
                              open_cost=_get_open_cost())
        skipped = [c for c in codes if c not in alloc]
        if skipped:
            log(f'  跳过 {len(skipped)} 只(买不起或无价): {skipped}')
        for code, a in alloc.items():
            if dry_run:
                log(f'  [空跑] 买 {code} {a["shares"]}股 ×{a["price"]:.2f}'
                      f' ≈ {a["amount"]:,.0f}')
                placed.append({'code': code, 'side': 'BUY',
                               'volume': int(a['shares']), 'price': a['price'],
                               'amount': a['amount'], 'resp': None,
                               'dry_run': True})
                continue
            o = order_volume(symbol=to_gm_symbol(code), volume=int(a['shares']),
                             side=OrderSide_Buy, order_type=OrderType_Limit,
                             position_effect=PositionEffect_Open, price=0)
            placed.append({'code': code, 'side': 'BUY', 'volume': int(a['shares']),
                           'price': a['price'], 'resp': _order_id(o)})

    if dry_run:
        total = sum(p.get('amount', 0) for p in placed if p['side'] == 'BUY')
        log(f'\n[空跑] 共 {len(placed)} 笔，买入金额合计 {total:,.0f}，'
              f'未提交任何委托')
        _save_state({'status': 'dry_run', 'orders': placed,
                     'exposure': exposure, 'cash_for_buy': cash_for_buy})
        return

    # --- 必须回查，不拿"没抛异常"当成交 ---
    orders = get_orders() or []
    by_id = {o.get('cl_ord_id'): o for o in orders}
    n_ok = 0
    for p in placed:
        o = by_id.get(p['resp'])
        p['status'] = o.get('status') if o else None
        p['filled'] = o.get('filled_volume') if o else None
        p['filled_vwap'] = o.get('filled_vwap') if o else None
        if o:
            n_ok += 1
        log(f"  {p['side']} {p['code']} {p['volume']}股 -> "
              f"status={p['status']} 成交 {p['filled']} @ {p['filled_vwap']}")

    log(f'已下 {len(placed)} 笔，回查到 {n_ok} 笔委托')
    if n_ok < len(placed):
        log(f'⚠ 有 {len(placed) - n_ok} 笔在委托列表里查不到 —— 不要当成已下单')
    _save_state({'status': 'placed', 'orders': placed,
                 'n_placed': len(placed), 'n_confirmed': n_ok})


def _order_id(resp):
    """order_volume 返回委托列表，取 cl_ord_id"""
    if isinstance(resp, list) and resp:
        return resp[0].get('cl_ord_id')
    if isinstance(resp, dict):
        return resp.get('cl_ord_id')
    return None


def _do_sync():
    """回读当日成交，用于与回测假设的收盘价比对"""
    from gm.api import get_execution_reports
    reps = get_execution_reports() or []
    rows = [{'code': from_gm_symbol(r['symbol']), 'side': r.get('side'),
             'volume': r.get('volume'), 'price': r.get('price'),
             'created_at': str(r.get('created_at'))} for r in reps]
    log(f'当日成交 {len(rows)} 笔')
    for r in rows[:20]:
        log(f"    {r['code']} {r['side']} {r['volume']}股 @ {r['price']}")
    _save_state({'status': 'synced', 'executions': rows,
                 'snapshot': _account_snapshot()})


# ---------------------------------------------------------------- 入口

def main() -> int:
    ap = argparse.ArgumentParser(description='掘金仿真桥接')
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--check', action='store_true', help='只查账户')
    g.add_argument('--place', action='store_true', help='读 pending_orders 下单')
    g.add_argument('--sync', action='store_true', help='回读成交')
    ap.add_argument('--dry-run', action='store_true',
                    help='与 --place 合用: 走完全部计算但不提交委托')
    args = ap.parse_args()

    action = 'check' if args.check else ('place' if args.place else 'sync')

    # gm.run() 内部用 optparse 解析 sys.argv，会把本脚本的 --check/--place
    # 当成自己的选项并报 "no such option"。解析完就把 argv 清干净。
    sys.argv = [sys.argv[0]]

    import os
    os.environ['GM_BRIDGE_ACTION'] = action
    os.environ['GM_BRIDGE_DRY'] = '1' if args.dry_run else '0'

    from gm.api import run, MODE_LIVE
    # run() 会阻塞直到策略停止; 本策略只在 init 里干活
    run(strategy_id=env('GM_STRATEGY_ID'), filename=Path(__file__).name,
        mode=MODE_LIVE, token=env('GM_TOKEN'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
