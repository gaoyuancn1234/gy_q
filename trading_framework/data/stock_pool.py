#!/usr/bin/env python3
"""
动态股票池管理
- 获取指数成分股
- 根据条件筛选
- 定期更新缓存
"""

import pandas as pd
import baostock as bs
from datetime import datetime, timedelta
from pathlib import Path
import json

# 缓存目录
CACHE_DIR = Path(__file__).parent / 'cache'
CACHE_DIR.mkdir(exist_ok=True)

# 缓存有效期（天）
CACHE_DAYS = 7


def get_index_stocks(index_code: str = 'sh.000300', use_cache: bool = True) -> list:
    """
    获取指数成分股

    参数:
        index_code: 指数代码
            - sh.000300: 沪深300
            - sh.000905: 中证500
            - sh.000016: 上证50
            - sh.000852: 中证1000
        use_cache: 是否使用缓存

    返回:
        成分股代码列表
    """
    cache_file = CACHE_DIR / f'index_{index_code.replace(".", "_")}.json'

    # 检查缓存
    if use_cache and cache_file.exists():
        with open(cache_file, 'r') as f:
            data = json.load(f)
            cache_date = datetime.fromisoformat(data['date'])
            if (datetime.now() - cache_date).days < CACHE_DAYS:
                print(f"使用缓存: {index_code} ({len(data['stocks'])}只)")
                return data['stocks']

    # 从baostock获取
    print(f"获取指数成分股: {index_code}")
    lg = bs.login()

    # 获取最新交易日
    today = datetime.now().strftime('%Y-%m-%d')
    rs = bs.query_stock_industry()  # 先查询激活连接

    # 获取成分股
    rs = bs.query_hs300_stocks() if index_code == 'sh.000300' else \
         bs.query_zz500_stocks() if index_code == 'sh.000905' else \
         bs.query_sz50_stocks() if index_code == 'sh.000016' else None

    if rs is None:
        # 通用方法
        rs = bs.query_stock_industry()

    stocks = []
    while (rs.error_code == '0') & rs.next():
        row = rs.get_row_data()
        if len(row) > 0:
            code = row[1] if len(row) > 1 else row[0]
            if code.startswith('sh.') or code.startswith('sz.'):
                stocks.append(code)

    bs.logout()

    # 保存缓存
    if stocks:
        with open(cache_file, 'w') as f:
            json.dump({
                'date': datetime.now().isoformat(),
                'index': index_code,
                'stocks': stocks
            }, f)
        print(f"获取成功: {len(stocks)}只")

    return stocks


def get_stock_info(stock_codes: list) -> pd.DataFrame:
    """获取股票基本信息"""
    lg = bs.login()

    infos = []
    for code in stock_codes:
        rs = bs.query_stock_basic(code=code)
        while (rs.error_code == '0') & rs.next():
            row = rs.get_row_data()
            infos.append({
                'code': code,
                'name': row[1] if len(row) > 1 else '',
                'ipoDate': row[2] if len(row) > 2 else '',
                'type': row[4] if len(row) > 4 else '',
                'status': row[5] if len(row) > 5 else ''
            })

    bs.logout()
    return pd.DataFrame(infos)


def filter_stocks(stocks: list,
                  min_market_cap: float = None,
                  exclude_st: bool = True,
                  exclude_new: int = 60,
                  industry_filter: list = None) -> list:
    """
    筛选股票

    参数:
        stocks: 股票列表
        min_market_cap: 最小市值（亿）
        exclude_st: 排除ST股票
        exclude_new: 排除上市不足N天的新股
        industry_filter: 只保留这些行业（None=不筛选）
    """
    lg = bs.login()
    filtered = []

    cutoff_date = (datetime.now() - timedelta(days=exclude_new)).strftime('%Y-%m-%d')

    for code in stocks:
        # 获取基本信息
        rs = bs.query_stock_basic(code=code)
        if rs.next():
            row = rs.get_row_data()
            name = row[1] if len(row) > 1 else ''
            ipo_date = row[2] if len(row) > 2 else ''
            status = row[5] if len(row) > 5 else '1'

            # 排除ST
            if exclude_st and ('ST' in name or 'st' in name):
                continue

            # 排除新股
            if ipo_date and ipo_date > cutoff_date:
                continue

            # 排除停牌
            if status != '1':
                continue

            filtered.append(code)

    bs.logout()
    print(f"筛选后: {len(filtered)}只 (原{len(stocks)}只)")
    return filtered


def get_industry_stocks(industry: str) -> list:
    """获取行业股票"""
    lg = bs.login()

    rs = bs.query_stock_industry()
    stocks = []

    while (rs.error_code == '0') & rs.next():
        row = rs.get_row_data()
        if len(row) >= 4 and industry in row[3]:
            stocks.append(row[1])

    bs.logout()
    return stocks


def build_stock_pool(strategy: str = 'balanced') -> list:
    """
    构建股票池

    策略:
        - 'conservative': 保守型（消费+医药为主）
        - 'balanced': 均衡型（沪深300筛选）
        - 'aggressive': 激进型（中证500+科技）
    """
    print(f"\n构建股票池: {strategy}")

    if strategy == 'conservative':
        # 保守型：消费+医药
        base_stocks = [
            # 白酒
            'sh.600519', 'sz.000858', 'sz.000568', 'sh.600809',
            'sz.002304', 'sz.000596', 'sh.603288',
            # 消费
            'sh.600887', 'sz.000651', 'sz.000333', 'sz.000895', 'sh.600298',
            # 医药
            'sh.600276', 'sz.000538', 'sh.600196', 'sz.002415',
            'sh.600436', 'sz.300122',
            # 金融龙头
            'sh.601318', 'sh.600036',
        ]
        stocks = filter_stocks(base_stocks, exclude_st=True)

    elif strategy == 'aggressive':
        # 激进型：科技+新能源
        hs300 = get_index_stocks('sh.000300')
        zz500 = get_index_stocks('sh.000905')

        # 合并并筛选
        all_stocks = list(set(hs300 + zz500))
        stocks = filter_stocks(all_stocks, exclude_st=True, exclude_new=120)

    else:  # balanced
        # 均衡型：沪深300筛选
        hs300 = get_index_stocks('sh.000300')
        stocks = filter_stocks(hs300, exclude_st=True, exclude_new=60)

    return stocks


def update_stock_pool():
    """更新股票池配置文件"""
    stocks = build_stock_pool('conservative')

    # 格式化输出
    print("\n# 更新后的股票池配置:")
    print("STOCK_POOL = [")
    for i, code in enumerate(stocks):
        print(f"    '{code}',")
    print("]")

    return stocks


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=str, default='sh.000300',
                       help='指数代码')
    parser.add_argument('--strategy', type=str, default='conservative',
                       choices=['conservative', 'balanced', 'aggressive'],
                       help='策略类型')
    parser.add_argument('--update', action='store_true',
                       help='更新股票池')

    args = parser.parse_args()

    if args.update:
        update_stock_pool()
    else:
        stocks = build_stock_pool(args.strategy)
        print(f"\n股票池: {len(stocks)}只")
        for s in stocks[:10]:
            print(f"  {s}")
        if len(stocks) > 10:
            print(f"  ... 共{len(stocks)}只")
