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


def _save_state(payload: dict):
    payload['updated'] = datetime.now().isoformat(timespec='seconds')
    STATE_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                          encoding='utf-8')


# ---------------------------------------------------------------- 策略回调

def init(context):
    """掘金策略入口。run() 启动后由掘金调用。"""
    from gm.api import set_account_id
    set_account_id(env('GM_ACCOUNT_ID'))
    action = getattr(context, 'gm_bridge_action', 'check')

    if action == 'check':
        _do_check()
    elif action == 'place':
        _do_place()
    elif action == 'sync':
        _do_sync()


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
        print('✗ 账户无数据 — 检查 GM_ACCOUNT_ID 是否与该策略绑定的仿真账户一致')
        _save_state({'status': 'no_account', 'snapshot': snap})
        return
    print(f"✓ 总资产 {snap['nav']:,.0f}  可用 {snap['available']:,.0f}  "
          f"持仓 {len(snap['positions'])} 只")
    for p in snap['positions']:
        print(f"    {p['code']} {p['volume']}股 成本 {p['vwap']} 现价 {p['price']}")
    _save_state({'status': 'ok', 'snapshot': snap})


def _do_place():
    """把 pending_orders 送进仿真账户，逐笔回查委托状态"""
    from gm.api import order_volume, get_orders, OrderSide_Buy, OrderSide_Sell, \
        OrderType_Limit, PositionEffect_Open, PositionEffect_Close

    holdings = _load_holdings()
    pending = holdings.get('pending_orders') or {}
    sells = list(pending.get('sells') or [])
    buys = dict(pending.get('buys') or {})

    if not sells and not buys:
        print('无待执行订单')
        _save_state({'status': 'no_orders'})
        return

    print(f'待执行: 卖 {len(sells)} 只，买 {len(buys)} 只')

    # 卖出用现有持仓股数; 买入股数由 live_portfolio 的分配决定，
    # 这里从 holdings 里取不到，改为按当前可用资金等分 —— 与实盘同一函数
    from portfolio.rebalance_rules import allocate_buys
    from portfolio.live_portfolio import _get_open_cost

    snap = _account_snapshot()
    if snap['nav'] is None:
        print('✗ 账户无数据，拒绝下单')
        _save_state({'status': 'no_account'})
        return

    held = {p['code']: p['volume'] for p in snap['positions']}
    placed = []

    # --- 卖出 ---
    for code in sells:
        vol = held.get(code, 0)
        if vol <= 0:
            print(f'  跳过卖出 {code}: 仿真账户无持仓')
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
        alloc = allocate_buys(codes, px, float(snap['available'] or 0),
                              open_cost=_get_open_cost())
        for code, a in alloc.items():
            o = order_volume(symbol=to_gm_symbol(code), volume=int(a['shares']),
                             side=OrderSide_Buy, order_type=OrderType_Limit,
                             position_effect=PositionEffect_Open, price=0)
            placed.append({'code': code, 'side': 'BUY', 'volume': int(a['shares']),
                           'price': a['price'], 'resp': _order_id(o)})

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
        print(f"  {p['side']} {p['code']} {p['volume']}股 -> "
              f"status={p['status']} 成交 {p['filled']} @ {p['filled_vwap']}")

    print(f'已下 {len(placed)} 笔，回查到 {n_ok} 笔委托')
    if n_ok < len(placed):
        print(f'⚠ 有 {len(placed) - n_ok} 笔在委托列表里查不到 —— 不要当成已下单')
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
    print(f'当日成交 {len(rows)} 笔')
    for r in rows[:20]:
        print(f"    {r['code']} {r['side']} {r['volume']}股 @ {r['price']}")
    _save_state({'status': 'synced', 'executions': rows,
                 'snapshot': _account_snapshot()})


# ---------------------------------------------------------------- 入口

def main() -> int:
    ap = argparse.ArgumentParser(description='掘金仿真桥接')
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--check', action='store_true', help='只查账户')
    g.add_argument('--place', action='store_true', help='读 pending_orders 下单')
    g.add_argument('--sync', action='store_true', help='回读成交')
    args = ap.parse_args()

    action = 'check' if args.check else ('place' if args.place else 'sync')

    from gm.api import run, MODE_LIVE
    import gm.api as gmapi

    # 掘金通过模块级 init(context) 回调进入，动作用模块全局传递
    global _ACTION
    _ACTION = action

    def _init(context):
        context.gm_bridge_action = action
        init(context)

    # run() 会阻塞直到策略停止; 这里的策略只在 init 里干活然后退出
    sys.modules[__name__].init = _init
    run(strategy_id=env('GM_STRATEGY_ID'), filename=Path(__file__).name,
        mode=MODE_LIVE, token=env('GM_TOKEN'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
