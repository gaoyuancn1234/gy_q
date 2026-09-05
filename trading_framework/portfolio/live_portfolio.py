#!/usr/bin/env python3
"""实盘持仓管理 — 10万资金约束感知

核心功能:
1. 持仓 JSON 读写 + 历史追踪
2. 资金约束分配 (跳过买不起的股票)
3. 止损检查
4. 调仓指令生成 (人类可读)
5. 截图解析后 apply trades

不依赖 qlib.init() — 价格通过 BaoStock 获取。
"""
import json
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path

import baostock as bs
import pandas as pd

# 调仓风控规则与模拟盘共用同一实现
from portfolio.rebalance_rules import (
    select_sells, compute_exposure as _compute_exposure)
from net_guard import run_with_timeout

# 单次取价的总预算 (秒)。超时即降级到本地缓存，绝不允许无限期挂住。
PRICE_FETCH_TIMEOUT = 90

PROJECT_DIR = Path(__file__).resolve().parent.parent
HOLDINGS_FILE = PROJECT_DIR / "portfolio" / "live_holdings.json"

# BaoStock 使用全局 TCP socket，不可并发
_bs_lock = threading.Lock()

# ============ 代码转换 ============

_stock_names_cache = {}


def qlib_to_display(code: str) -> str:
    """SH600036 → 招商银行(600036)"""
    names = get_stock_names()
    num = code[2:]  # 600036
    name = names.get(code, code)
    return f"{name}({num})"


def qlib_to_bao(code: str) -> str:
    """SH600036 → sh.600036"""
    prefix = code[:2].lower()
    num = code[2:]
    return f"{prefix}.{num}"


def bao_to_qlib(code: str) -> str:
    """sh.600036 → SH600036"""
    return code.replace(".", "").upper()


def get_stock_names() -> dict:
    """获取 CSI300 股票中文名映射 {SH600036: '招商银行'}"""
    global _stock_names_cache
    if _stock_names_cache:
        return _stock_names_cache

    with _bs_lock:
        try:
            lg = bs.login()
            if lg.error_code != '0':
                return _stock_names_cache

            rs = bs.query_hs300_stocks()
            while rs.error_code == '0' and rs.next():
                row = rs.get_row_data()
                bao_code = row[1]
                name = row[2]
                qlib_code = bao_to_qlib(bao_code)
                _stock_names_cache[qlib_code] = name
        except Exception as e:
            print(f"[stock_names] 获取失败: {e}")
        finally:
            try:
                bs.logout()
            except Exception:
                pass

    return _stock_names_cache


# ============ 价格获取 ============

PRICE_CACHE_FILE = PROJECT_DIR / "portfolio" / "price_cache.json"


def _save_price_cache(prices: dict):
    """保存价格到本地缓存 (原子写入)"""
    try:
        cache = {
            'date': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'prices': prices,
        }
        tmp = PRICE_CACHE_FILE.with_suffix('.tmp')
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(cache, f)
        import os
        os.replace(tmp, PRICE_CACHE_FILE)
    except Exception:
        pass


def _load_price_cache(instruments: list) -> dict:
    """从缓存加载价格 (仅当天有效)"""
    if not PRICE_CACHE_FILE.exists():
        return {}
    try:
        with open(PRICE_CACHE_FILE, 'r', encoding='utf-8') as f:
            cache = json.load(f)
        cache_date = cache.get('date', '')[:10]
        today = datetime.now().strftime('%Y-%m-%d')
        if cache_date != today:
            return {}
        cached = cache.get('prices', {})
        return {k: v for k, v in cached.items() if k in instruments}
    except Exception:
        return {}


def _fetch_prices_sina(instruments: list) -> dict:
    """新浪实时行情 —— 一次请求取回全部代码

    2026-09-04 改为主源，两个原因:

    1. 可用性: BaoStock 已连续多日连不上(login 无超时，只会一直挂)。
    2. 口径: cost_price 来自成交截图解析，是**实际成交价(原始价)**。
       BaoStock 那条路用的是 adjustflag=2 前复权价，两者在除权日之后会
       系统性错位 —— 止损判据 current/cost-1 会凭空多出或抹掉一截收益。
       新浪返回原始价，与 cost_price 同口径；仓位市值、可买手数也都该用
       原始价算。所以这不只是换个源，是把口径改对了。
    """
    import requests

    codes = [f"{c[:2].lower()}{c[2:]}" for c in instruments]
    prices = {}
    # 单次 URL 不宜过长，分批
    for i in range(0, len(codes), 60):
        batch = codes[i:i + 60]
        r = requests.get("https://hq.sinajs.cn/list=" + ",".join(batch),
                         timeout=15,
                         headers={"Referer": "https://finance.sina.com.cn"})
        r.raise_for_status()
        r.encoding = "gbk"
        for line in r.text.strip().split("\n"):
            if '="' not in line:
                continue
            code = line.split("hq_str_")[1].split("=")[0]
            fields = line.split('"')[1].split(",")
            if len(fields) < 4:
                continue
            try:
                px = float(fields[3])          # 当前价; 收盘后即为收盘价
            except ValueError:
                continue
            if px <= 0:                        # 停牌返回 0，跳过而不是记 0
                continue
            prices[f"{code[:2].upper()}{code[2:]}"] = px
    return prices


def get_current_prices(instruments: list) -> tuple:
    """获取指定股票的最新价格

    新浪实时行情 → BaoStock → 本地当日缓存，三级降级，每级都有硬超时。

    Args:
        instruments: Qlib 格式代码列表 ['SH600036', ...]

    Returns:
        (prices_dict, from_cache) — prices_dict: {instrument: price}, from_cache: bool
    """
    if not instruments:
        return {}, False

    try:
        prices = run_with_timeout(lambda: _fetch_prices_sina(instruments),
                                  PRICE_FETCH_TIMEOUT, "新浪取价")
        if prices:
            _save_price_cache(prices)
            return prices, False
        print("[get_prices] 新浪返回空，转 BaoStock")
    except Exception as e:
        print(f"[get_prices] 新浪失败 ({type(e).__name__}): {e}, 转 BaoStock")

    def _fetch() -> dict:
        lg = bs.login()
        if lg.error_code != '0':
            raise ConnectionError(f"BaoStock 登录失败: {lg.error_msg}")

        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=10)).strftime('%Y-%m-%d')
        prices = {}

        for inst in instruments:
            bao_code = qlib_to_bao(inst)
            rs = bs.query_history_k_data_plus(
                bao_code,
                "date,close",
                start_date=start_date,
                end_date=end_date,
                frequency="d",
                adjustflag="2",  # 前复权
            )
            rows = []
            while rs.next():
                rows.append(rs.get_row_data())
            if rows:
                prices[inst] = float(rows[-1][1])
        return prices

    with _bs_lock:
        try:
            # 2026-09-04: 必须加硬超时。bs.login() 没有超时参数，源挂了就永久
            # 阻塞 —— 挂起不抛异常，下面写好的缓存降级分支永远轮不到执行。
            # 当天 18:00 的 daily_runner 就卡死在这一行一个多小时。
            prices = run_with_timeout(_fetch, PRICE_FETCH_TIMEOUT, "BaoStock 取价")
            if prices:
                _save_price_cache(prices)
            return prices, False
        except Exception as e:
            print(f"[get_prices] BaoStock 失败 ({type(e).__name__}): {e}, 尝试本地缓存...")
            cached = _load_price_cache(instruments)
            if cached:
                print(f"[get_prices] 使用缓存价格 ({len(cached)}只)")
            return cached, bool(cached)
        finally:
            try:
                bs.logout()
            except Exception:
                pass


# ============ 资金约束分配 ============

def calculate_affordable_allocation(targets: list, prices: dict,
                                    available_cash: float,
                                    min_lot: int = 100) -> dict:
    """跳过买不起的股票，重新分配资金

    2026-09-05: 实现移到 portfolio/rebalance_rules.allocate_buys，与模拟盘
    共用。此前两边各写一份、四条语义都不同(太贵剔除/不足一手/交易成本/
    跳过的钱是否重分)，同一份买入清单会买出不同股数与不同持仓只数。
    本函数保留是为了不改调用点。

    交易成本此前完全没进预算 —— 下单金额刚好等于可用资金时，实际会因佣金
    透支。现在按 signal_config.yaml 的 open_cost 计入。
    """
    from portfolio.rebalance_rules import allocate_buys
    return allocate_buys(targets, prices, available_cash,
                         open_cost=_get_open_cost(), min_lot=min_lot)


def _get_open_cost() -> float:
    """买入费率 (signal_config.yaml)，读不到时按 0 处理"""
    try:
        import yaml
        with open(PROJECT_DIR / 'config' / 'signal_config.yaml',
                  encoding='utf-8') as f:
            return float(yaml.safe_load(f).get('open_cost', 0.0) or 0.0)
    except Exception:
        return 0.0


# ============ 净值记录与波动率目标 ============

TRADING_DAYS = 242


def compute_nav(holdings: dict, prices: dict) -> tuple:
    """按当前价计算组合总净值 (现金 + 持仓市值)

    Returns:
        (nav, n_stale) —— n_stale 是取不到实时价、退回成本价的持仓数。

    2026-09-05: 原实现 `prices.get(code) or pos.get('cost_price', 0)` 有两个
    问题:
      1. `or` 在价格为 0.0 时也会回退 —— 0.0 是 falsy，停牌股取到 0 会被
         当成"没取到"，虽然结果碰巧一样，但掩盖了真实状态。
      2. 更要命: 回退是**静默**的。取价整体失败时 NAV 变成成本价合计、
         逐日恒定，vol_target 据此算出实现波动率 ≈ 0，exposure 直接拉满
         100%。数据源故障时风控自动失效并放大仓位 —— 失败方向完全反了。
    现在把 stale 数量返回给调用方，由其决定是否记录/告警/收缩。
    """
    nav = holdings.get('cash', 0.0)
    n_stale = 0
    for code, pos in holdings.get('positions', {}).items():
        px = prices.get(code)
        if px is None or px <= 0:
            px = pos.get('cost_price', 0)
            n_stale += 1
        nav += pos.get('shares', 0) * px
    return float(nav), n_stale


def record_nav(holdings: dict, prices: dict, date: str = None) -> float:
    """记录每日净值 (vol targeting 需要净值序列来估计已实现波动率)

    同一天重复调用只保留最后一次。序列上限 500 个交易日。

    取价不全的日子会打上 stale 标记并记录 stale 持仓数。compute_exposure
    会跳过这些点 —— 否则用成本价冒充市价的"假净值"会压低实现波动率，
    让 vol_target 在数据故障时反而放大敞口。
    """
    nav, n_stale = compute_nav(holdings, prices)
    n_pos = len(holdings.get('positions', {}))
    date = date or datetime.now().strftime('%Y-%m-%d')
    rec = {'date': date, 'nav': nav}
    if n_stale:
        rec['stale'] = n_stale
        print(f"[live_portfolio] 警告: {n_stale}/{n_pos} 只持仓取不到实时价，"
              f"已用成本价估值; 该日净值标记为 stale，不参与波动率估计")
    hist = holdings.setdefault('nav_history', [])
    if hist and hist[-1].get('date') == date:
        hist[-1] = rec
    else:
        hist.append(rec)
    if len(hist) > 500:
        del hist[:-500]
    return nav


def _get_vol_target_config() -> dict:
    """从 signal_config.yaml 读取波动率目标配置 (缺失/损坏时安全关闭)"""
    try:
        import yaml
        cfg_path = PROJECT_DIR / "config" / "signal_config.yaml"
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        nd = cfg.get('n_drop')
        return {
            'vol_target': cfg.get('vol_target'),
            'vol_window': int(cfg.get('vol_window', 20)),
            'vol_min_exposure': float(cfg.get('vol_min_exposure', 0.2)),
            'n_drop': int(nd) if nd is not None else None,
        }
    except Exception as e:
        print(f"[live_portfolio] 读取风控配置失败，已回退到无风控行为: {e}")
        return {'vol_target': None, 'n_drop': None}


def compute_exposure(holdings: dict, target_vol: float,
                     window: int = 20, min_exposure: float = 0.2) -> tuple:
    """按已实现波动率计算权益敞口 (薄封装，实现见 rebalance_rules)

    保留本函数是为了不改动既有调用方的签名 (传 holdings 而非净值列表)。
    算法本身与模拟盘共用 portfolio/rebalance_rules.compute_exposure，
    避免两份实现再次分叉 —— 这正是 n_drop/vol_target 在模拟盘缺失的成因。
    """
    # 剔除 stale 净值点: 那些日子有持仓取不到实时价、用成本价充数，
    # 净值人为平滑。留着会压低实现波动率 -> exposure 被算高 -> 数据故障时
    # 风控反而放大仓位。宁可样本不足退回保守值，也不能用假净值。
    hist = holdings.get('nav_history') or []
    navs = [h['nav'] for h in hist if not h.get('stale')]
    n_dropped = len(hist) - len(navs)
    if n_dropped:
        print(f"[live_portfolio] 波动率估计剔除 {n_dropped} 个 stale 净值点")
    return _compute_exposure(navs, target_vol, window=window,
                             min_exposure=min_exposure)


# ============ 持仓操作 ============

def load_live_holdings() -> dict:
    """加载实盘持仓"""
    if HOLDINGS_FILE.exists():
        with open(HOLDINGS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return init_live_portfolio()


def save_live_holdings(holdings: dict):
    """保存实盘持仓 (原子写入)

    2026-09-03: 原先直接 open(...,'w') + json.dump —— 写到一半被中断
    (断电、进程被杀、磁盘满) 会留下截断的 JSON。这个文件记录真实持仓与现金，
    损坏等于账本丢失，而下次读取只会抛 JSONDecodeError，无从恢复。
    改用先写临时文件再 os.replace 的原子写法。
    """
    holdings['last_update'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    from factor_lab.utils import atomic_json_dump
    atomic_json_dump(HOLDINGS_FILE, holdings, ensure_ascii=False, indent=2)


def init_live_portfolio(capital: float = 100000) -> dict:
    """初始化实盘持仓"""
    holdings = {
        "initial_capital": capital,
        "cash": capital,
        "positions": {},
        "pending_orders": {"sells": [], "buys": {}},
        "rebalance_day_count": 0,
        "last_signal_date": None,
        "last_update": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "history": [],
    }
    save_live_holdings(holdings)
    return holdings


def add_capital(holdings: dict, amount: float) -> dict:
    """追加资金 — 只增加 cash 和 initial_capital，不动持仓

    下次调仓日由 calculate_affordable_allocation() 自动分配。

    Args:
        holdings: live_holdings 数据
        amount: 追加金额 (元)

    Returns:
        更新后的 holdings
    """
    holdings['cash'] = holdings.get('cash', 0) + amount
    holdings['initial_capital'] = holdings.get('initial_capital', 0) + amount
    save_live_holdings(holdings)
    return holdings


def clear_positions(holdings: dict) -> dict:
    """清空所有持仓 — positions 清零，cash 重置为 initial_capital

    Returns:
        更新后的 holdings
    """
    holdings['positions'] = {}
    holdings['cash'] = holdings.get('initial_capital', 100000)
    holdings['pending_orders'] = {"sells": [], "buys": {}}
    save_live_holdings(holdings)
    return holdings


def apply_trades(parsed_trades: list, holdings: dict = None) -> str:
    """应用截图解析后的成交记录

    Args:
        parsed_trades: [
            {"action": "buy", "code": "SH600036", "shares": 200, "price": 38.50},
            {"action": "sell", "code": "SH601166", "shares": 300, "price": 16.80},
        ]
        holdings: 持仓数据 (None 则自动加载)

    Returns:
        执行结果文本
    """
    if holdings is None:
        holdings = load_live_holdings()

    results = []

    for trade in parsed_trades:
        action = trade.get('action', '')
        code = trade.get('code', '')
        shares = int(trade.get('shares', 0))
        price = float(trade.get('price', 0))

        if not code or shares <= 0 or price <= 0:
            results.append(f"跳过无效交易: {trade}")
            continue

        name = qlib_to_display(code)
        amount = shares * price

        if action == 'sell':
            if code in holdings['positions']:
                pos = holdings['positions'][code]
                if pos['shares'] >= shares:
                    pos['shares'] -= shares
                    holdings['cash'] += amount
                    if pos['shares'] == 0:
                        del holdings['positions'][code]
                    results.append(f"卖出 {name} {shares}股 ×{price:.2f} = {amount:.0f}")
                else:
                    results.append(f"股数不足 {name}: 持有{pos['shares']}股, 要卖{shares}股")
            else:
                results.append(f"未持有 {name}")

        elif action == 'buy':
            if amount > holdings['cash']:
                results.append(f"现金不足买入 {name}: 需{amount:.0f}, 可用{holdings['cash']:.0f}")
                continue
            holdings['cash'] -= amount
            if code in holdings['positions']:
                old = holdings['positions'][code]
                old_total = old['shares'] * old['cost_price']
                new_total = old_total + amount
                old['shares'] += shares
                old['cost_price'] = round(new_total / old['shares'], 4)
            else:
                holdings['positions'][code] = {
                    'name': get_stock_names().get(code, code),
                    'shares': shares,
                    'cost_price': price,
                    'entry_date': datetime.now().strftime('%Y-%m-%d'),
                }
            results.append(f"买入 {name} {shares}股 ×{price:.2f} = {amount:.0f}")

    # 记录历史
    holdings['history'].append({
        'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'trades': parsed_trades,
        'results': results,
    })

    # 清除待执行订单
    holdings['pending_orders'] = {"sells": [], "buys": {}}

    save_live_holdings(holdings)
    return "\n".join(results) if results else "无有效交易"


# ============ 模型版本 ============

def get_model_version() -> str:
    """从 signal_config.yaml 读取模型版本号，如 M01-LGB-D3v3r-v2602"""
    config_file = PROJECT_DIR / "config" / "signal_config.yaml"
    try:
        import yaml
        with open(config_file, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        tag = cfg.get('model_tag', 'M01')
        retrain = cfg.get('last_retrain', '')  # '2026-02'
        if retrain:
            parts = retrain.split('-')
            ver = parts[0][2:] + parts[1] if len(parts) == 2 else retrain.replace('-', '')
            return f"{tag}-v{ver}"
        return tag
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning("get_current_model_id 失败，回退到默认 M01: %s", e)
        return "M01"


# ============ 调仓订单计算 (单一实现) ============

def compute_rebalance_orders(signal: dict, holdings: dict) -> dict:
    """由信号与当前持仓算出本次调仓的买卖清单

    2026-09-05 抽出。此前同一件事有两份实现且结论不同:

        generate_live_instructions (推给用户的消息)
            select_sells(..., n_drop=4)      -> 最多卖 4 只
            to_buy[:free]                    -> 只买腾出的坑位

        daily_runner 存进 holdings['pending_orders']
            sorted(current_set - target_set) -> 全量换手，无 n_drop
            to_buy 无上限

    后果是同一次调仓给出两条互相矛盾的指令: 18:00 飞书推"卖4买4"，次日
    9:30 盘中监控读 pending_orders 提醒"待执行 卖12买12"。若照后者执行，
    就是回测中吃掉本金约 14% 的全量换手。daily_runner 的日志行
    "调仓指令: 卖N 买M" 报的也是这份错数。

    两处现在共用本函数。新增消费方请调用它，不要再各算一遍。

    Returns:
        {'sells': [...], 'buys': [...], 'holds': set, 'n_drop': int|None}
        sells/buys 均已排序，保证可复现 (集合迭代顺序随哈希随机化变化)。
    """
    target_set = set(signal['target_stocks'])
    current_set = set(holdings.get('positions', {}).keys())
    topk = signal['effective_topk']

    n_drop = _get_vol_target_config().get('n_drop')
    sells = select_sells(current_set, target_set,
                         scores=signal.get('scores') or {},
                         n_drop=n_drop)

    # 已挂但未执行的止损单也要算作"即将卖出"，否则腾出的坑位数会少算，
    # 买入数量与回测对不上 (回测的 current_holds 就是扣掉 pending sells 的)。
    pending_sells = set(holdings.get('pending_orders', {}).get('sells', []))
    sells = sells | (pending_sells & current_set)

    holds = current_set - sells
    # 只买入被腾出的坑位数，保持持仓数稳定
    free = max(0, topk - len(holds))
    buys = [c for c in signal['target_stocks'] if c not in current_set][:free]

    return {
        'sells': sorted(sells),
        'buys': buys,
        'holds': holds,
        'n_drop': n_drop,
    }


# ============ 指令生成 ============

def generate_live_instructions(signal: dict, holdings: dict, prices: dict) -> str:
    """生成人类可读的调仓指令 (飞书格式)

    Args:
        signal: SignalGenerator.get_signal() 的输出
        holdings: live_holdings 数据
        prices: 当前价格

    Returns:
        飞书消息文本
    """
    today = signal['date']
    regime = signal['regime']
    topk = signal['effective_topk']
    quality = signal.get('quality_score')
    target_set = set(signal['target_stocks'])
    current_set = set(holdings['positions'].keys())
    names = get_stock_names()

    # 计算卖出 — 受 n_drop 换手限制
    #
    # 原实现是全量换手 (current_set - target_set)，在 5 日调仓下几乎每次换掉
    # 全部持仓。回测显示这会吃掉绝大部分收益: 10万资金 2.5 年约 1734 笔交易，
    # 双边成本 0.05%+0.15% 累积 ≈ 1.4 万 (本金的 14%)。
    #   全量换手  Sharpe 0.24(段1) / 0.48(2024-26)
    #   n_drop=2  Sharpe 1.27(段1) / 1.16(2024-26)
    # 规则实现在 portfolio/rebalance_rules.py，与模拟盘共用同一份
    _orders = compute_rebalance_orders(signal, holdings)
    to_sell = set(_orders['sells'])
    to_hold = _orders['holds']
    to_buy_codes = _orders['buys']

    model_ver = get_model_version()
    lines = [
        "━━━━━━━━━━━━━━━━━━━",
        f"📊 ML调仓信号 ({today})",
        f"模型: {model_ver}",
        "",
        f"信号状态: {regime} | TopK: {topk} | 质量: {quality:.2f}" if quality else f"信号状态: {regime} | TopK: {topk}",
    ]

    # 模型新鲜度 (简化)
    window = signal.get('current_window')
    if window:
        lines.append(f"当前 Window: {window}")
    lines.append("")

    # 卖出
    sell_total = 0
    if to_sell:
        lines.append(f"【卖出】({len(to_sell)}只)")
        for code in to_sell:
            pos = holdings['positions'].get(code, {})
            shares = pos.get('shares', 0)
            price = prices.get(code, 0)
            amount = shares * price
            sell_total += amount
            lines.append(f"  {qlib_to_display(code)} {shares}股 ×{price:.2f} ≈ {amount:,.0f}")
        lines.append("")

    # 买入 (考虑资金约束 + 波动率目标)
    available_cash = holdings['cash'] + sell_total

    # 波动率目标: 市场波动放大时按比例收缩敞口，其余留现金
    _vt = _get_vol_target_config()
    if _vt.get('vol_target'):
        exposure, realized = compute_exposure(
            holdings, _vt['vol_target'],
            window=_vt.get('vol_window', 20),
            min_exposure=_vt.get('vol_min_exposure', 0.2))
        if exposure < 1.0:
            available_cash *= exposure
            lines.append(f"【风控】实现波动 {realized:.1%} > 目标 "
                         f"{_vt['vol_target']:.0%}，敞口降至 {exposure:.0%}，"
                         f"可用资金 {available_cash:,.0f}")
            lines.append("")

    if to_buy_codes:
        alloc = calculate_affordable_allocation(
            to_buy_codes, prices, available_cash
        )
        buy_total = 0
        lines.append(f"【买入】({len(to_buy_codes)}只)")
        for code in to_buy_codes:
            if code in alloc:
                a = alloc[code]
                buy_total += a['amount']
                lines.append(f"  {qlib_to_display(code)} {a['shares']}股 ×{a['price']:.2f} ≈ {a['amount']:,.0f}")
            else:
                price = prices.get(code, 0)
                lines.append(f"  ⚠️ {qlib_to_display(code)} 跳过 (股价{price:.0f} > 预算上限)")
        lines.append("")

    # 持仓不变
    if to_hold:
        hold_names = [names.get(c, c) for c in to_hold]
        lines.append(f"【持仓不变】({len(to_hold)}只)")
        lines.append(f"  {', '.join(hold_names)}")
        lines.append("")

    # 无变动
    if not to_sell and not to_buy_codes:
        lines.append("本期无需调仓，继续持有")
        lines.append("")

    # 资金概览
    # 保留的持仓市值
    hold_value = sum(
        holdings['positions'].get(c, {}).get('shares', 0) * prices.get(c, 0)
        for c in (current_set - to_sell)
    )
    # 新买入市值
    buy_value = sum(alloc.get(c, {}).get('amount', 0) for c in to_buy_codes) if to_buy_codes else 0
    total_position = hold_value + buy_value
    estimated_cash = available_cash - buy_value
    lines.append(f"💰 预计: 持仓 {total_position:,.0f} | 现金 {estimated_cash:,.0f}")
    lines.append("请于明日收盘前执行，完成后发截图确认")
    lines.append("━━━━━━━━━━━━━━━━━━━")

    return "\n".join(lines)


# ============ 止损检查 ============

def check_stop_loss(holdings: dict, prices: dict,
                    threshold: float | None = None) -> list:
    """检查止损

    2026-09-03: threshold 原先硬编码默认 0.08，而唯一的调用处不传参 ——
    改 signal_config.yaml 的 stop_loss 对实盘止损不生效，
    与盘中监控(读 settings.STOP_LOSS)用的是两个值。改为默认读配置。

    Returns:
        [{'code': 'SH600036', 'name': ..., 'loss_pct': -0.12, ...}]
    """
    if threshold is None:
        from config.settings import STOP_LOSS
        threshold = STOP_LOSS
    alerts = []
    for code, pos in holdings.get('positions', {}).items():
        cost = pos.get('cost_price', 0)
        current = prices.get(code, 0)
        if cost > 0 and current > 0:
            pnl_pct = (current - cost) / cost
            if pnl_pct <= -threshold:
                alerts.append({
                    'code': code,
                    'name': pos.get('name', code),
                    'cost_price': cost,
                    'current_price': current,
                    'loss_pct': round(pnl_pct, 4),
                    'shares': pos.get('shares', 0),
                })
    return alerts


# ============ 组合概览 ============

def get_portfolio_summary(holdings: dict, prices: dict) -> str:
    """生成持仓概览"""
    total_market = 0
    lines = []

    positions = holdings.get('positions', {})
    if not positions:
        cash = holdings.get('cash', 0)
        return f"📋 实盘持仓: 空仓\n现金: {cash:,.0f}"

    lines.append("📋 实盘持仓概览")
    lines.append("")

    for code, pos in positions.items():
        price = prices.get(code, 0)
        shares = pos.get('shares', 0)
        cost = pos.get('cost_price', 0)
        market = shares * price
        total_market += market
        pnl = (price - cost) / cost * 100 if cost > 0 else 0
        sign = "+" if pnl >= 0 else ""
        lines.append(f"  {qlib_to_display(code)} {shares}股 {price:.2f} {sign}{pnl:.1f}%")

    cash = holdings.get('cash', 0)
    total = total_market + cash
    initial = holdings.get('initial_capital', 100000)
    total_pnl = (total - initial) / initial * 100
    sign = "+" if total_pnl >= 0 else ""

    lines.append("")
    lines.append(f"持仓市值: {total_market:,.0f}")
    lines.append(f"现金: {cash:,.0f}")
    lines.append(f"总资产: {total:,.0f} ({sign}{total_pnl:.1f}%)")

    return "\n".join(lines)


def track_empty_position_days(holdings: dict) -> int:
    """追踪连续空仓天数，返回当前连续空仓天数

    在 holdings 中维护 'empty_position_days' 和 'last_empty_check_date' 字段。
    每个交易日调用一次，有持仓则重置为 0。
    """
    today = datetime.now().strftime('%Y-%m-%d')
    last_check = holdings.get('last_empty_check_date', '')
    positions = holdings.get('positions', {})

    if today == last_check:
        # 同一天重复调用，不累加
        return holdings.get('empty_position_days', 0)

    if not positions:
        holdings['empty_position_days'] = holdings.get('empty_position_days', 0) + 1
    else:
        holdings['empty_position_days'] = 0

    holdings['last_empty_check_date'] = today
    return holdings['empty_position_days']


def get_daily_report(holdings: dict, prices: dict) -> str:
    """生成非调仓日日报"""
    positions = holdings.get('positions', {})
    cash = holdings.get('cash', 0)
    initial = holdings.get('initial_capital', 100000)

    total_market = sum(
        pos.get('shares', 0) * prices.get(code, 0)
        for code, pos in positions.items()
    )
    total = total_market + cash
    total_pnl = (total - initial) / initial * 100

    today = datetime.now().strftime('%Y-%m-%d')
    sign = "+" if total_pnl >= 0 else ""

    # 止损检查
    alerts = check_stop_loss(holdings, prices)
    alert_text = "无" if not alerts else ", ".join(
        f"{a['name']}({a['loss_pct']:.1%})" for a in alerts
    )

    model_ver = get_model_version()
    report = (
        f"📈 持仓日报 ({today})\n"
        f"模型: {model_ver}\n\n"
        f"总资产: {total:,.0f} ({sign}{total_pnl:.1f}%) | 现金: {cash:,.0f}\n"
        f"持仓: {len(positions)}只 | 市值: {total_market:,.0f}\n\n"
        f"⚠️ 止损预警: {alert_text}"
    )

    # 连续空仓告警
    empty_days = track_empty_position_days(holdings)
    if empty_days >= 5:
        report += f"\n\n🔔 已连续空仓 {empty_days} 个交易日，请检查是否需要开始建仓"

    return report
