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

def get_current_prices(instruments: list) -> dict:
    """获取指定股票的最新价格 (BaoStock)

    Args:
        instruments: Qlib 格式代码列表 ['SH600036', ...]

    Returns:
        {instrument: price} 字典
    """
    with _bs_lock:
        try:
            lg = bs.login()
            if lg.error_code != '0':
                return {}

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
        except Exception as e:
            print(f"[get_prices] 失败: {e}")
            return {}
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

    Args:
        targets: 目标股票列表 (Qlib code)
        prices: {instrument: price} 字典
        available_cash: 可用资金
        min_lot: 最小交易单位 (A股 100)

    Returns:
        {instrument: {'shares': 200, 'amount': 7700, 'price': 38.50}}
        跳过的股票不在结果中
    """
    if not targets:
        return {}

    # 第一轮: 筛出买得起的
    affordable = []
    skipped = []
    budget_per = available_cash / len(targets)

    for inst in targets:
        price = prices.get(inst, 0)
        if price <= 0:
            skipped.append((inst, 'no_price'))
            continue
        min_cost = price * min_lot
        if min_cost > budget_per * 1.5:
            # 预算的 1.5 倍都买不了 100 股，跳过
            skipped.append((inst, 'too_expensive'))
            continue
        affordable.append(inst)

    if not affordable:
        return {}

    # 第二轮: 等权分配给可买的
    budget_per = available_cash / len(affordable)
    allocation = {}
    total_used = 0

    for inst in affordable:
        price = prices[inst]
        shares = int(budget_per / price / min_lot) * min_lot
        if shares < min_lot:
            shares = min_lot
        amount = shares * price
        if total_used + amount > available_cash:
            # 资金不够了
            shares = int((available_cash - total_used) / price / min_lot) * min_lot
            if shares < min_lot:
                continue
            amount = shares * price
        allocation[inst] = {
            'shares': shares,
            'amount': round(amount, 2),
            'price': price,
        }
        total_used += amount

    return allocation


# ============ 持仓操作 ============

def load_live_holdings() -> dict:
    """加载实盘持仓"""
    if HOLDINGS_FILE.exists():
        with open(HOLDINGS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return init_live_portfolio()


def save_live_holdings(holdings: dict):
    """保存实盘持仓"""
    holdings['last_update'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(HOLDINGS_FILE, 'w', encoding='utf-8') as f:
        json.dump(holdings, f, ensure_ascii=False, indent=2)


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
        with open(config_file, 'r') as f:
            cfg = yaml.safe_load(f)
        tag = cfg.get('model_tag', 'M01')
        retrain = cfg.get('last_retrain', '')  # '2026-02'
        if retrain:
            parts = retrain.split('-')
            ver = parts[0][2:] + parts[1] if len(parts) == 2 else retrain.replace('-', '')
            return f"{tag}-v{ver}"
        return tag
    except Exception:
        return "M01"


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

    # 计算卖出
    to_sell = current_set - target_set
    to_hold = current_set & target_set
    to_buy_codes = [c for c in signal['target_stocks'] if c not in current_set]

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

    # 买入 (考虑资金约束)
    available_cash = holdings['cash'] + sell_total
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

def check_stop_loss(holdings: dict, prices: dict, threshold: float = 0.08) -> list:
    """检查止损

    Returns:
        [{'code': 'SH600036', 'name': ..., 'loss_pct': -0.12, ...}]
    """
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
