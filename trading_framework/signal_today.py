#!/usr/bin/env python3
"""
生成今日交易信号
策略: 15日相对强度, Top12
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.data_loader import DataLoader

FIXED_STOCK_POOL = [
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

# 股票名称映射
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


def calculate_rs(stock_data, benchmark, date, lookback=15):
    sh = stock_data[stock_data['date'] <= date].tail(lookback + 1)
    bh = benchmark[benchmark['date'] <= date].tail(lookback + 1)
    if len(sh) < lookback or len(bh) < lookback:
        return None
    s_ret = sh['close'].iloc[-1] / sh['close'].iloc[0] - 1
    b_ret = bh['close'].iloc[-1] / bh['close'].iloc[0] - 1
    return s_ret - b_ret, s_ret, b_ret


def main():
    print("=" * 60)
    print("交易信号生成")
    print(f"日期: {datetime.now().strftime('%Y-%m-%d')}")
    print("策略: 15日相对强度 Top12")
    print("=" * 60)

    loader = DataLoader(cache_dir='./cache')

    print("\n加载数据...")
    start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')

    data = loader.get_all_data(FIXED_STOCK_POOL, start_date, end_date)
    benchmark = loader.get_index_data('sh.000300', start_date, end_date)

    if data is None:
        print("数据加载失败")
        return

    # 最新日期
    latest_date = data['date'].max()
    print(f"最新数据日期: {latest_date.strftime('%Y-%m-%d')}")

    # 计算所有股票的相对强度
    all_rs = []
    for code in data['code'].unique():
        sd = data[data['code'] == code]
        result = calculate_rs(sd, benchmark, latest_date, lookback=15)
        if result:
            rs, stock_ret, bench_ret = result
            # 获取最新价格
            latest_price = sd[sd['date'] == latest_date]['close'].iloc[-1] if len(sd[sd['date'] == latest_date]) > 0 else 0

            all_rs.append({
                'code': code,
                'name': STOCK_NAMES.get(code, code),
                'rs': rs,
                'stock_ret': stock_ret,
                'price': latest_price
            })

    # 排序
    all_rs = sorted(all_rs, key=lambda x: x['rs'], reverse=True)

    # 基准15日收益
    bm_hist = benchmark[benchmark['date'] <= latest_date].tail(16)
    bm_ret = bm_hist['close'].iloc[-1] / bm_hist['close'].iloc[0] - 1

    print(f"\n沪深300 15日收益: {bm_ret*100:+.2f}%")

    print(f"\n{'='*60}")
    print("推荐持仓 (Top 12)")
    print(f"{'='*60}")
    print(f"{'排名':<4} {'代码':<12} {'名称':<10} {'相对强度':>10} {'15日涨幅':>10} {'现价':>8}")
    print("-" * 60)

    for i, s in enumerate(all_rs[:12], 1):
        print(f"{i:<4} {s['code']:<12} {s['name']:<10} {s['rs']*100:>+9.2f}% {s['stock_ret']*100:>+9.2f}% {s['price']:>8.2f}")

    print(f"\n{'='*60}")
    print("配置建议")
    print(f"{'='*60}")
    print("• 仓位: 满仓 (100%)")
    print("• 每只股票: 等权 (约8.3%)")
    print("• 调仓频率: 每周五")

    # 输出简洁版本
    print(f"\n{'='*60}")
    print("简洁版 - 复制用")
    print(f"{'='*60}")
    codes = [s['code'].replace('sh.', '').replace('sz.', '') for s in all_rs[:12]]
    print("股票代码:", ", ".join(codes))

    print("\n股票列表:")
    for s in all_rs[:12]:
        print(f"  {s['name']} ({s['code']})")


if __name__ == '__main__':
    main()
