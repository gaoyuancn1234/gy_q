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

下单时点: 14:50 —— 必须与训练标签同一计价基准
--------------------------------------------
模型的标签以 signal_config.yaml 的 preset / exec_lag 为准:
    exec_lag=1  旧口径: T 收盘信号, T+1 收盘成交
    exec_lag=0  路线 B: T 14:45 出分, T 14:50 收盘竞价成交
下单日必须等于 holdings['execute_on'], 且落在 14:50~15:05。

14:50 下限价单，落在 14:57~15:00 的收盘集合竞价里成交，逼近收盘价。

2026-09-08 曾短暂改为 09:40 (直觉: "正常人开盘就买")。当天改回。
改回的理由不是开盘价不好 —— CLAUDE.md 的 one-switch 实验里开盘口径超额
27.11%、收盘 11.90%，开盘看起来好一倍多。但那个数字建立在"能拿到开盘
集合竞价撮合价"的假设上，而实盘要和所有人抢同一价位、还有隔夜跳空；
多出来的那一倍很可能正是拿不到的部分。更关键的是模型没在那个口径上训练过。

要真的改用早盘口径，必须**同时**改标签(如 Ref($close,-1)/Ref($open,-1)-1)
并重训重验，而不是只改下单时间 —— 只改一边等于让模型预测 A、你去交易 B。

架构
----
必须作为**掘金策略**运行 —— 账户接口(get_cash/get_position/order_volume)
只在 run(strategy_id=..., mode=MODE_LIVE) 启动的策略里返回数据，独立调用
一律返回空。所以本文件是一个独立进程，与 daily_runner 隔离:

    daily_runner / daily auction  ->  holdings['pending_orders']
         execute_on + exec_lag     ->  gm_bridge 14:50  ->  掘金仿真
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

# 用户指定: CSI2000 专用掘金仿真账户, 区别于 CSI1000/CSI300。
GM_ACCOUNT_ID = "21761e6d-ad82-11f1-8afe-00163e022aa6"

STATE_FILE = PROJECT_DIR / 'gm_state.json'
HOLDINGS_FILE = PROJECT_DIR / 'portfolio' / 'live_holdings.json'
PLACE_FROM = "14:50"
PLACE_TO = "15:05"
QUOTE_BATCH = 40

# 掘金的标的写法是 SHSE.600036 / SZSE.000001，本项目内部是 SH600036 / SZ000001
_EX = {'SH': 'SHSE', 'SZ': 'SZSE', 'BJ': 'BJSE'}


def to_gm_symbol(code: str) -> str:
    """SH600036 -> SHSE.600036"""
    pre, num = code[:2].upper(), code[2:]
    if pre not in _EX:
        raise ValueError(f'无法识别的标的代码: {code}')
    return f'{_EX[pre]}.{num}'


def from_gm_symbol(sym: str) -> str:
    """SHSE.600036 -> SH600036"""
    if "." not in str(sym):
        raise ValueError(f"无法识别的掘金代码: {sym}")
    ex, num = sym.split(".", 1)
    rev = {v: k for k, v in _EX.items()}
    if ex not in rev:
        raise ValueError(f"无法识别的交易所: {sym}")
    return f"{rev[ex]}{num}"


def _quotes_for(codes: list) -> dict:
    """分批取现价。中证2000 有北交所/无行情票, 一锅端会让整批买单落空。"""
    from gm.api import current
    want = []
    for code in codes:
        try:
            want.append((code, to_gm_symbol(code)))
        except ValueError as e:
            log(f"  跳过代码 {code}: {e}")
    px = {}
    for i in range(0, len(want), QUOTE_BATCH):
        part = want[i:i + QUOTE_BATCH]
        try:
            qs = current(symbols=[s for _, s in part]) or []
        except Exception as e:
            log(f"  取价批次失败 {i}: {type(e).__name__}: {e}")
            continue
        for q in qs:
            try:
                code = from_gm_symbol(q.get("symbol"))
            except (TypeError, ValueError):
                continue
            price = q.get("price")
            if price:
                px[code] = price
    return px


def env(key: str, required: bool = True):
    if key == "GM_ACCOUNT_ID":
        return GM_ACCOUNT_ID
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


def _push_feishu(msg: str) -> None:
    """推飞书。失败不能影响主流程 —— 但也不能静默吞掉。"""
    try:
        from daily_runner import push_feishu
        push_feishu(msg)
    except Exception as e:
        log(f'  飞书推送失败: {type(e).__name__}: {e}')


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


def _load_state() -> dict:
    if not STATE_FILE.exists():
        return {}
    try:
        return json.loads(STATE_FILE.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return {}


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
    force = _os.environ.get('GM_BRIDGE_FORCE') == '1'
    log(f'=== gm_bridge {action}{" (空跑)" if dry else ""} '
        f'{_dt.now():%Y-%m-%d %H:%M:%S} ===')

    from gm.api import set_account_id
    set_account_id(env('GM_ACCOUNT_ID'))
    log(f'account_id={env("GM_ACCOUNT_ID")} (CSI2000 仿真)')

    try:
        if action == 'check':
            _do_check()
        elif action == 'place':
            _do_place(dry_run=dry, force=force)
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


def _account_snapshot(retries: int = 10, wait: float = 1.0) -> dict:
    """取账户快照，等到账户数据真正同步过来为止

    2026-09-07: 首次 --check 报 "✓ 总资产 0 可用 0"，4 分钟后同一账户报
    100,000。日志显示 init() 在"连接交易服务成功"的同一秒就调了 get_cash()
    —— 连接建立与账户数据下发不是同一件事，查早了拿到的是空壳。

    危险的不是慢，是 **0 被当成有效值**: nav=0 会一路写进 gm_state.json 的
    "status": "ok"，下游据此判断就会错，而下单必然失败。这与本项目栽过的
    几次是同一类 —— 把"还没拿到"错当成"拿到了，值是空"。

    所以这里重试到 nav 为正数为止；始终取不到就如实返回 None，让上层报错。
    """
    import time as _t
    from gm.api import get_cash, get_position

    cash, pos = {}, []
    for i in range(retries):
        cash = get_cash() or {}
        pos = get_position() or []
        if (cash.get('nav') or 0) > 0:
            if i:
                log(f'  账户数据在第 {i + 1} 次查询到位 (等待 {i * wait:.0f}s)')
            break
        _t.sleep(wait)
    else:
        log(f'  ⚠ 等待 {retries * wait:.0f}s 后账户总资产仍为 '
            f'{cash.get("nav")!r} — 可能是账户未入金，也可能是终端未同步')
        cash.setdefault('nav', None)

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
    # nav 为 0 与 None 一样不可用: 零余额账户下什么单都会失败。原先只判 None，
    # 于是"查早了拿到 0"被报成 ✓ 并写进 status: ok —— 见 _account_snapshot 注释。
    if not snap['nav']:
        log(f'✗ 账户不可用 (总资产 {snap["nav"]!r}) — 检查: '
            f'GM_ACCOUNT_ID 是否与该策略绑定的仿真账户一致 / 该账户是否已入金')
        _save_state({'status': 'no_account', 'snapshot': snap})
        return
    log(f"✓ 总资产 {snap['nav']:,.0f}  可用 {snap['available']:,.0f}  "
          f"持仓 {len(snap['positions'])} 只")
    for p in snap['positions']:
        log(f"    {p['code']} {p['volume']}股 成本 {p['vwap']} 现价 {p['price']}")
    # 2026-09-14: _save_state 原先缩在 for 里, **持仓为 0 时一次都不执行** ——
    # 账户查得好好的(日志写着 ✓ 总资产 100,000), gm_state.json 却还停在上一次
    # 的 terminal_offline。空仓恰恰是首次建仓前的状态, 也就是最需要这条状态的
    # 时候。任何读 gm_state 判断"终端通不通"的地方都会被误导。
    _save_state({'status': 'ok', 'snapshot': snap,
                 'account_id': GM_ACCOUNT_ID})


def _do_place(dry_run: bool = False, force: bool = False):
    """把 pending_orders 送进仿真账户，逐笔回查委托状态

    dry_run=True 时走完全部计算(取价、敞口、分配、股数)但不提交委托。

    幂等保护 (2026-09-09 新增)
    --------------------------
    2026-09-09 早上 GmPlace 在非预定时间(10:18，预定 14:50)意外触发并
    真实下了 13 笔买单，根因未查清(疑似改动 Set-ScheduledTask 触发器后
    Windows 的 StartWhenAvailable 追赶行为)。危险的不是"为什么提前跑了"，
    是**如果 14:50 按计划再跑一次会发生什么**: 买入那段完全不看 `held`
    (当前持仓)，只看 pending_orders 和当下可用现金——重复触发会用剩余
    现金对同一批股票再买一轮，仓位不知不觉翻倍，而 rc 依然是 0。

    与其去猜清楚 Windows 为什么会双发(这类问题没有能验证"以后不会再发生"
    的修法)，不如让下单本身对重复触发免疫: 记录"这批 pending_orders 对应
    哪个信号日期已经下过单"，同一信号日期的重复触发直接跳过。
    """
    from gm.api import order_volume, get_orders, OrderSide_Buy, OrderSide_Sell, \
        OrderType_Limit, PositionEffect_Open, PositionEffect_Close

    holdings = _load_holdings()
    pending = holdings.get('pending_orders') or {}
    sells = list(pending.get('sells') or [])
    buys = dict(pending.get('buys') or {})
    signal_date = holdings.get('last_signal_date')
    from portfolio.live_portfolio import get_exec_lag
    lag = get_exec_lag()
    today = datetime.now().strftime("%Y-%m-%d")
    hhmm = datetime.now().strftime("%H:%M")
    execute_on = holdings.get("execute_on")
    if not force:
        if hhmm < PLACE_FROM or hhmm > PLACE_TO:
            log(
                f"非收盘竞价窗口 {hhmm} "
                f"(允许 {PLACE_FROM}~{PLACE_TO}), 不下单。"
                " --force 才强制。"
            )
            return
        if execute_on and today != str(execute_on)[:10]:
            log(
                f"exec_lag={lag} 成交日是 {execute_on}, "
                f"今天 {today}, 不下单"
            )
            return
        if execute_on is None:
            sd = str(signal_date or "")[:10]
            if lag == 0 and sd and today != sd:
                log(f"exec_lag=0 只能在信号日 {sd} 下单, 今天 {today}")
                return
            if lag == 1 and sd and today == sd:
                log("exec_lag=1 信号日不当天成交, 等下一个交易日 14:50")
                return
    log(f"成交时钟 exec_lag={lag} execute_on={execute_on or '?'} today={today}")

    if not force and not dry_run:
        prev = _load_state()
        if (prev.get('status') in ('placed', 'dry_run')
                and prev.get('signal_date') == signal_date
                and signal_date is not None):
            log(f'信号日期 {signal_date} 已下过单 (状态={prev.get("status")}, '
               f'{prev.get("updated","?")})，本次跳过。传 --force 强制重下。')
            return

    if not sells and not buys:
        log('无待执行订单')
        _save_state({'status': 'no_orders', 'signal_date': signal_date})
        return

    log(f'待执行: 卖 {len(sells)} 只，买 {len(buys)} 只 (信号日期 {signal_date})')

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
            min_exposure=_vt.get('vol_min_exposure', 0.2),
            unknown_exposure=_vt.get('vol_unknown_exposure'),
        )
        if exposure < 1.0:
            cash_for_buy *= exposure
    _rz = f'{realized:.1%}' if realized is not None else '无法估计(净值历史不足)'
    log(f'  敞口 {exposure:.0%} (实现波动 {_rz})，'
          f'可用 {snap["available"]:,.0f} -> 买入预算 {cash_for_buy:,.0f}')

    # --- 卖出 ---
    #
    # 2026-09-07: 原先传 price=0 且 order_type=OrderType_Limit。限价单必须有
    # 价格，price=0 会被拒单 —— 而此前只做过空跑验证(空跑根本不调 order_volume)，
    # 所以这条路径从未真正下出过一笔。
    #
    # 限价单成交在**对手价**而非自己的限价，所以留一点缓冲只影响"能不能成交"，
    # 不抬高成本: 买单挂 现价×1.01 仍按卖一价成交。缓冲取 1%(A股涨跌停 ±10%，
    # 不会触板)。实际成交价由 --sync 回读，滑点照样能量。
    _LIMIT_BUFFER = 0.01

    sell_px = _quotes_for(sells) if sells else {}

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
        ref = sell_px.get(code)
        if not ref:
            log(f'  跳过卖出 {code}: 取不到实时价，无法定限价(停牌?)')
            continue
        limit = round(ref * (1 - _LIMIT_BUFFER), 2)
        o = order_volume(symbol=to_gm_symbol(code), volume=int(vol),
                         side=OrderSide_Sell, order_type=OrderType_Limit,
                         position_effect=PositionEffect_Close, price=limit)
        placed.append({'code': code, 'side': 'SELL', 'volume': int(vol),
                       'limit': limit, 'ref_price': ref, 'resp': _order_id(o)})

    # --- 买入 ---
    if buys:
        codes = list(buys.keys())
        px = _quotes_for(codes)
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
            # 见上方卖出处的说明: price=0 的限价单会被拒，且成交在对手价，
            # 挂 1% 缓冲只保证成交、不抬高实际成本。
            limit = round(a['price'] * (1 + _LIMIT_BUFFER), 2)
            o = order_volume(symbol=to_gm_symbol(code), volume=int(a['shares']),
                             side=OrderSide_Buy, order_type=OrderType_Limit,
                             position_effect=PositionEffect_Open, price=limit)
            placed.append({'code': code, 'side': 'BUY', 'volume': int(a['shares']),
                           'price': a['price'], 'limit': limit,
                           'resp': _order_id(o)})

    if dry_run:
        total = sum(p.get('amount', 0) for p in placed if p['side'] == 'BUY')
        log(f'\n[空跑] 共 {len(placed)} 笔，买入金额合计 {total:,.0f}，'
              f'未提交任何委托')
        _save_state({'status': 'dry_run', 'signal_date': signal_date, 'orders': placed,
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
    _save_state({'status': 'placed', 'signal_date': signal_date, 'orders': placed,
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
    snap = _account_snapshot()
    _save_state({'status': 'synced', 'executions': rows, 'snapshot': snap})
    _mirror_to_live_holdings(snap)


def _mirror_to_live_holdings(snap: dict) -> None:
    """把掘金仿真账户的真实持仓回写到 live_holdings.json

    2026-09-08: 补这一步之前存在一个数据源断层 ——

      盘中监控(每5分钟)、飞书"持仓"命令、止损检查、自我反思，
      读的全是 portfolio/live_holdings.json。而那个文件原本只能靠
      "截图 + 回复已执行" 更新。改成掘金自动下单之后没人再截图，
      于是 live_holdings 永远是空的。

    实测后果: 监控当天 09:25~15:05 醒来 68 次，68 次都记
    "空仓且无待执行订单" —— 而掘金账户里是有仓位的。
    也就是说**真实持仓完全没有止损保护**: 模型让持有 8 个交易日，
    中间某只跌 20% 也没有任何人在看。

    掘金是成交的唯一真相来源(它按真实盘口撮合)，所以以它为准回写，
    而不是反过来。cost_price 用 vwap(持仓均价)，与 live_holdings 里
    "实际成交价" 的语义一致。
    """
    if not snap or snap.get('nav') is None:
        log('  账户快照无效，跳过回写 live_holdings(不写入不可信数据)')
        return
    try:
        from portfolio.live_portfolio import load_live_holdings, save_live_holdings
    except Exception as e:
        log(f'  回写 live_holdings 失败(导入): {type(e).__name__}: {e}')
        return

    h = load_live_holdings()
    old_codes = set((h.get('positions') or {}).keys())

    positions = {}
    for p in snap.get('positions') or []:
        vol = int(p.get('volume') or 0)
        if vol <= 0:
            continue
        positions[p['code']] = {
            'name': p.get('name') or p['code'],
            'shares': vol,
            'cost_price': float(p.get('vwap') or p.get('price') or 0),
            'entry_date': (h.get('positions', {}).get(p['code'], {})
                           .get('entry_date') or datetime.now().strftime('%Y-%m-%d')),
        }

    h['positions'] = positions
    h['cash'] = float(snap.get('available') or 0)
    h['last_update'] = datetime.now().isoformat(timespec='seconds')
    h['synced_from'] = 'gm_bridge'      # 标明这份持仓的来源，便于排查

    # 净值序列 —— vol_target 靠它估已实现波动率。
    #
    # 2026-09-12: 此前这里只写 positions/cash，从不写 nav_history；而
    # record_nav() 只在 daily_runner.generate_and_push 里调用，那条路
    # (daily signal) 在 exec_lag=0 时是被挡掉的。于是 csi2000 的净值序列
    # **永远是空的**，compute_exposure 每次都走"估不出来"分支，敞口被
    # 永久钉在 vol_unknown_exposure=0.40。
    #
    # 后果不只是"保守一点": 0.40 敞口下 topk=100 每坑位只有 400 元，
    # allocate_buys 的价格上限降到 6 元，实测 Top100 里只买得起 18.4 只、
    # 买入中位价 4.58 元(应买 12.28)。即实盘跑的是一个"18 只低价股"的
    # 组合，而不是回测里那个。vol_target=0.25 那套配对检验也从未生效。
    #
    # 掘金账户按真实盘口撮合，它的 nav 是这条路上唯一可信的权益值，
    # 所以直接记它，不用本地价格重算。
    _nav = snap.get('nav')
    if _nav is not None and float(_nav) > 0:
        _today = datetime.now().strftime('%Y-%m-%d')
        _hist = h.setdefault('nav_history', [])
        _rec = {'date': _today, 'nav': float(_nav), 'src': 'gm'}
        if _hist and _hist[-1].get('date') == _today:
            _hist[-1] = _rec            # 同日重复 sync 只留最后一次
        else:
            _hist.append(_rec)
        if len(_hist) > 500:
            del _hist[:-500]
        log(f'  净值 {float(_nav):,.0f} 已记入 nav_history '
            f'(共 {len(_hist)} 条; vol_target 需要 >= 21 条才生效)')
    else:
        log('  账户 nav 无效，本次不记净值(不写入不可信数据)')

    save_live_holdings(h)

    new_codes = set(positions)
    log(f'  已回写 live_holdings: {len(positions)} 只持仓, 现金 '
        f'{h["cash"]:,.0f} (新增 {len(new_codes - old_codes)}, '
        f'移除 {len(old_codes - new_codes)})')


# ---------------------------------------------------------------- 入口

def main() -> int:
    ap = argparse.ArgumentParser(description='掘金仿真桥接')
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--check', action='store_true', help='只查账户')
    g.add_argument('--place', action='store_true', help='读 pending_orders 下单')
    g.add_argument('--sync', action='store_true', help='回读成交')
    ap.add_argument('--dry-run', action='store_true',
                    help='与 --place 合用: 走完全部计算但不提交委托')
    ap.add_argument('--force', action='store_true',
                    help='与 --place 合用: 忽略"同一信号日期已下过单"的幂等跳过')
    args = ap.parse_args()

    action = 'check' if args.check else ('place' if args.place else 'sync')

    # gm.run() 内部用 optparse 解析 sys.argv，会把本脚本的 --check/--place
    # 当成自己的选项并报 "no such option"。解析完就把 argv 清干净。
    sys.argv = [sys.argv[0]]

    import os
    os.environ['GM_BRIDGE_ACTION'] = action
    os.environ['GM_BRIDGE_DRY'] = '1' if args.dry_run else '0'
    os.environ['GM_BRIDGE_FORCE'] = '1' if args.force else '0'

    from gm.api import run, MODE_LIVE
    # run() 会阻塞直到策略停止; 本策略只在 init 里干活
    try:
        run(strategy_id=env('GM_STRATEGY_ID'), filename=Path(__file__).name,
            mode=MODE_LIVE, token=env('GM_TOKEN'))
    except Exception as e:
        # 终端没开时 gm 抛 {"status": 1001, "message": "无法连接到终端服务"}。
        # 2026-09-08: 裸抛的话定时任务只留下一个非零退出码和一段
        # 英文堆栈，看不出"是掘金终端没启动"这个唯一需要人干预的原因。
        # SDK 连的是本机终端(127.0.0.1:7001)，终端不在就什么都做不了。
        msg = str(e)
        if '1001' in msg or '无法连接到终端' in msg:
            log('✗ 连不上掘金终端 —— 请启动掘金量化终端并保持登录。')
            log('  SDK 通过本机 127.0.0.1:7001 与终端通信，终端不在则无法下单/回读。')
            _save_state({'status': 'terminal_offline',
                         'action': action, 'error': msg})
            _push_feishu('⚠ 掘金终端未启动，今日下单/回读跳过。'
                         '请打开掘金量化终端并登录，否则仿真盘不会有任何成交。')
            return 2
        raise
    return 0


if __name__ == '__main__':
    sys.exit(main())
