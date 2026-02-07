#!/usr/bin/env python3
"""
交易执行器
- 生成调仓指令
- 更新持仓记录
- 与飞书集成
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.data_loader import DataLoader

HOLDINGS_FILE = Path(__file__).parent / 'holdings.json'

# 股票池和名称
STOCK_POOL = [
    'sh.601318', 'sh.600036', 'sh.601166', 'sh.600016', 'sh.601328',
    'sh.600000', 'sh.601288', 'sh.601398', 'sh.601939', 'sh.601988',
    'sh.600519', 'sz.000858', 'sh.600887', 'sz.000568', 'sz.002304',
    'sh.600809', 'sz.000651', 'sz.000333', 'sz.000895', 'sh.600298',
    'sz.000596', 'sh.603288', 'sh.600600', 'sz.000725', 'sh.600104',
    'sh.600276', 'sz.000538', 'sh.600196', 'sz.002415', 'sh.600436',
    'sz.300015', 'sh.600085', 'sz.000963',
    'sz.002475', 'sz.002230', 'sh.600588', 'sh.601012', 'sz.000977',
    'sh.600031', 'sh.600050', 'sz.000338', 'sh.601899', 'sh.600900',
    'sh.600690', 'sh.601888',
    'sz.300274', 'sz.002129', 'sh.601633', 'sz.002460', 'sh.600438',
]

STOCK_NAMES = {
    'sh.601318': '中国平安', 'sh.600036': '招商银行', 'sh.601166': '兴业银行',
    'sh.600016': '民生银行', 'sh.601328': '交通银行', 'sh.600000': '浦发银行',
    'sh.601288': '农业银行', 'sh.601398': '工商银行', 'sh.601939': '建设银行',
    'sh.601988': '中国银行', 'sh.600519': '贵州茅台', 'sz.000858': '五粮液',
    'sh.600887': '伊利股份', 'sz.000568': '泸州老窖', 'sz.002304': '洋河股份',
    'sh.600809': '山西汾酒', 'sz.000651': '格力电器', 'sz.000333': '美的集团',
    'sz.000895': '双汇发展', 'sh.600298': '安琪酵母', 'sz.000596': '古井贡酒',
    'sh.603288': '海天味业', 'sh.600600': '青岛啤酒', 'sz.000725': '京东方A',
    'sh.600104': '上汽集团', 'sh.600276': '恒瑞医药', 'sz.000538': '云南白药',
    'sh.600196': '复星医药', 'sz.002415': '海康威视', 'sh.600436': '片仔癀',
    'sz.300015': '爱尔眼科', 'sh.600085': '同仁堂', 'sz.000963': '华东医药',
    'sz.002475': '立讯精密', 'sz.002230': '科大讯飞', 'sh.600588': '用友网络',
    'sh.601012': '隆基绿能', 'sz.000977': '浪潮信息', 'sh.600031': '三一重工',
    'sh.600050': '中国联通', 'sz.000338': '潍柴动力', 'sh.601899': '紫金矿业',
    'sh.600900': '长江电力', 'sh.600690': '海尔智家', 'sh.601888': '中国中免',
    'sz.300274': '阳光电源', 'sz.002129': '中环股份', 'sh.601633': '长城汽车',
    'sz.002460': '赣锋锂业', 'sh.600438': '通威股份',
}


def load_holdings():
    """加载持仓"""
    if HOLDINGS_FILE.exists():
        with open(HOLDINGS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {
        'last_updated': None,
        'total_capital': None,
        'cash': None,
        'positions': {},
        'history': []
    }


def save_holdings(holdings):
    """保存持仓"""
    with open(HOLDINGS_FILE, 'w', encoding='utf-8') as f:
        json.dump(holdings, f, ensure_ascii=False, indent=2)


def get_current_prices():
    """获取当前价格"""
    from datetime import timedelta
    loader = DataLoader(cache_dir='./cache')

    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')

    data = loader.get_all_data(STOCK_POOL, start_date, end_date)
    benchmark = loader.get_index_data('sh.000300', start_date, end_date)

    if data is None:
        return None, None

    latest_date = data['date'].max()
    prices = {}

    for code in data['code'].unique():
        code_data = data[(data['code'] == code) & (data['date'] == latest_date)]
        if len(code_data) > 0:
            prices[code] = code_data['close'].iloc[0]

    return prices, latest_date, data, benchmark


def calculate_rs(stock_data, benchmark, date, lookback=15):
    """计算相对强度"""
    sh = stock_data[stock_data['date'] <= date].tail(lookback + 1)
    bh = benchmark[benchmark['date'] <= date].tail(lookback + 1)
    if len(sh) < lookback or len(bh) < lookback:
        return None
    s_ret = sh['close'].iloc[-1] / sh['close'].iloc[0] - 1
    b_ret = bh['close'].iloc[-1] / bh['close'].iloc[0] - 1
    return s_ret - b_ret


def get_top12_signals(data, benchmark, latest_date):
    """获取Top12信号"""
    all_rs = []
    for code in data['code'].unique():
        sd = data[data['code'] == code]
        rs = calculate_rs(sd, benchmark, latest_date, lookback=15)
        if rs is not None:
            all_rs.append({'code': code, 'rs': rs})

    all_rs = sorted(all_rs, key=lambda x: x['rs'], reverse=True)
    return [s['code'] for s in all_rs[:12]]


def generate_trade_instructions():
    """生成调仓指令"""
    holdings = load_holdings()

    if holdings['total_capital'] is None:
        return "❌ 错误: 请先设置总资金\n运行: python trade_executor.py --init <资金>"

    print("获取市场数据...")
    result = get_current_prices()
    if result[0] is None:
        return "❌ 错误: 无法获取市场数据"

    prices, latest_date, data, benchmark = result

    # 获取新的Top12
    new_top12 = get_top12_signals(data, benchmark, latest_date)

    # 当前持仓
    current_positions = set(holdings['positions'].keys())
    new_positions = set(new_top12)

    # 计算买卖
    to_sell = current_positions - new_positions
    to_buy = new_positions - current_positions
    to_hold = current_positions & new_positions

    # 计算每只股票目标金额
    total_capital = holdings['total_capital']
    target_per_stock = total_capital / 12

    # 生成指令
    lines = []
    lines.append(f"📊 调仓指令 ({latest_date.strftime('%Y-%m-%d')})")
    lines.append("")

    # 卖出
    if to_sell:
        lines.append("【卖出】")
        sell_total = 0
        for i, code in enumerate(to_sell, 1):
            pos = holdings['positions'].get(code, {})
            shares = pos.get('shares', 0)
            price = prices.get(code, 0)
            amount = shares * price
            sell_total += amount
            name = STOCK_NAMES.get(code, code)
            code_short = code.replace('sh.', '').replace('sz.', '')
            lines.append(f"{i}. {name}({code_short}) 卖{shares}股 × {price:.2f}元 ≈ {amount:.0f}元")
        lines.append(f"卖出合计: {sell_total:.0f}元")
        lines.append("")
    else:
        lines.append("【卖出】无")
        lines.append("")

    # 买入
    if to_buy:
        lines.append("【买入】")
        buy_total = 0
        for i, code in enumerate(to_buy, 1):
            price = prices.get(code, 0)
            if price > 0:
                shares = int(target_per_stock / price / 100) * 100  # 取整到100股
                if shares < 100:
                    shares = 100
                amount = shares * price
                buy_total += amount
                name = STOCK_NAMES.get(code, code)
                code_short = code.replace('sh.', '').replace('sz.', '')
                lines.append(f"{i}. {name}({code_short}) 买{shares}股 × {price:.2f}元 ≈ {amount:.0f}元")
        lines.append(f"买入合计: {buy_total:.0f}元")
        lines.append("")
    else:
        lines.append("【买入】无")
        lines.append("")

    # 持仓不变
    if to_hold:
        hold_names = [STOCK_NAMES.get(c, c) for c in to_hold]
        lines.append(f"【持仓不变】({len(to_hold)}只)")
        lines.append(", ".join(hold_names))
    else:
        lines.append("【持仓不变】无")

    return "\n".join(lines)


def show_current_holdings():
    """显示当前持仓"""
    holdings = load_holdings()

    if holdings['total_capital'] is None:
        return "❌ 尚未初始化，请先设置总资金"

    lines = []
    lines.append(f"📋 当前持仓 (更新于 {holdings['last_updated'] or '未知'})")
    lines.append(f"总资金: {holdings['total_capital']:,.0f}元")
    lines.append(f"现金: {holdings['cash']:,.0f}元")
    lines.append("")

    if holdings['positions']:
        lines.append("持仓明细:")
        total_value = 0
        for code, pos in holdings['positions'].items():
            name = STOCK_NAMES.get(code, code)
            shares = pos.get('shares', 0)
            cost = pos.get('cost', 0)
            lines.append(f"  {name}: {shares}股 成本{cost:.2f}")
            total_value += shares * cost
        lines.append(f"\n持仓市值: {total_value:,.0f}元")
    else:
        lines.append("当前空仓")

    return "\n".join(lines)


def init_capital(amount):
    """初始化资金"""
    holdings = load_holdings()
    holdings['total_capital'] = float(amount)
    holdings['cash'] = float(amount)
    holdings['positions'] = {}
    holdings['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M')
    save_holdings(holdings)
    return f"✅ 已设置总资金: {amount:,.0f}元"


def update_position(code, shares, price, action='buy'):
    """更新单个持仓"""
    holdings = load_holdings()

    code_full = code
    if not code.startswith('sh.') and not code.startswith('sz.'):
        # 自动补全代码
        if code.startswith('6'):
            code_full = f'sh.{code}'
        else:
            code_full = f'sz.{code}'

    if action == 'buy':
        amount = shares * price
        if holdings['cash'] < amount:
            return f"❌ 现金不足: 需要{amount:.0f}元, 现有{holdings['cash']:.0f}元"

        holdings['cash'] -= amount
        if code_full in holdings['positions']:
            old = holdings['positions'][code_full]
            total_shares = old['shares'] + shares
            total_cost = old['shares'] * old['cost'] + shares * price
            holdings['positions'][code_full] = {
                'shares': total_shares,
                'cost': total_cost / total_shares
            }
        else:
            holdings['positions'][code_full] = {
                'shares': shares,
                'cost': price
            }

        holdings['history'].append({
            'time': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'action': 'buy',
            'code': code_full,
            'shares': shares,
            'price': price
        })

    elif action == 'sell':
        if code_full not in holdings['positions']:
            return f"❌ 没有持仓: {code_full}"

        pos = holdings['positions'][code_full]
        if pos['shares'] < shares:
            return f"❌ 股数不足: 持有{pos['shares']}股, 要卖{shares}股"

        amount = shares * price
        holdings['cash'] += amount

        if pos['shares'] == shares:
            del holdings['positions'][code_full]
        else:
            holdings['positions'][code_full]['shares'] -= shares

        holdings['history'].append({
            'time': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'action': 'sell',
            'code': code_full,
            'shares': shares,
            'price': price
        })

    holdings['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M')
    save_holdings(holdings)

    name = STOCK_NAMES.get(code_full, code_full)
    return f"✅ 已{'买入' if action == 'buy' else '卖出'}: {name} {shares}股 × {price:.2f}元"


def main():
    import argparse
    parser = argparse.ArgumentParser(description='交易执行器')
    parser.add_argument('--action', type=str, default='signal',
                       choices=['signal', 'holdings', 'init', 'buy', 'sell'],
                       help='操作类型')
    parser.add_argument('--capital', type=float, help='初始资金')
    parser.add_argument('--code', type=str, help='股票代码')
    parser.add_argument('--shares', type=int, help='股数')
    parser.add_argument('--price', type=float, help='价格')

    args = parser.parse_args()

    if args.action == 'signal':
        print(generate_trade_instructions())
    elif args.action == 'holdings':
        print(show_current_holdings())
    elif args.action == 'init':
        if args.capital:
            print(init_capital(args.capital))
        else:
            print("❌ 请指定资金: --capital <金额>")
    elif args.action == 'buy':
        if args.code and args.shares and args.price:
            print(update_position(args.code, args.shares, args.price, 'buy'))
        else:
            print("❌ 请指定: --code <代码> --shares <股数> --price <价格>")
    elif args.action == 'sell':
        if args.code and args.shares and args.price:
            print(update_position(args.code, args.shares, args.price, 'sell'))
        else:
            print("❌ 请指定: --code <代码> --shares <股数> --price <价格>")


if __name__ == '__main__':
    main()
