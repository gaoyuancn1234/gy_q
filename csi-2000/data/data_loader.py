"""
数据加载模块
- 从baostock获取A股日线数据
- 支持增量更新
"""

import pandas as pd
import numpy as np
import baostock as bs
from datetime import datetime, timedelta
import os
import pickle

class DataLoader:
    def __init__(self, cache_dir='./cache'):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self._login()

    def _login(self):
        """登录baostock"""
        self.lg = bs.login()
        if self.lg.error_code != '0':
            raise Exception(f"登录失败: {self.lg.error_msg}")

    def _logout(self):
        """登出"""
        bs.logout()

    def get_stock_list(self, date=None):
        """获取指定日期的沪深300成分股"""
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')

        rs = bs.query_hs300_stocks(date)
        stocks = []
        while rs.next():
            stocks.append(rs.get_row_data()[1])  # 股票代码
        return stocks

    def get_daily_data(self, stock_code, start_date, end_date):
        """获取单只股票日线数据"""
        fields = "date,code,open,high,low,close,volume,amount,turn,pctChg"

        rs = bs.query_history_k_data_plus(
            stock_code,
            fields,
            start_date=start_date,
            end_date=end_date,
            frequency="d",
            adjustflag="2"  # 前复权
        )

        data = []
        while rs.next():
            data.append(rs.get_row_data())

        if not data:
            return None

        df = pd.DataFrame(data, columns=fields.split(','))

        # 转换数据类型
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'amount', 'turn', 'pctChg']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df['date'] = pd.to_datetime(df['date'])
        df = df.dropna()

        return df

    def get_all_data(self, stock_list, start_date, end_date, show_progress=True):
        """获取多只股票数据"""
        all_data = []

        for i, code in enumerate(stock_list):
            if show_progress and (i + 1) % 10 == 0:
                print(f"  加载进度: {i+1}/{len(stock_list)}")

            df = self.get_daily_data(code, start_date, end_date)
            if df is not None and len(df) > 0:
                all_data.append(df)

        if not all_data:
            return None

        return pd.concat(all_data, ignore_index=True)

    def get_index_data(self, index_code='sh.000300', start_date=None, end_date=None):
        """获取指数数据（用于基准对比）"""
        fields = "date,code,open,high,low,close,volume"

        rs = bs.query_history_k_data_plus(
            index_code,
            fields,
            start_date=start_date,
            end_date=end_date,
            frequency="d"
        )

        data = []
        while rs.next():
            data.append(rs.get_row_data())

        df = pd.DataFrame(data, columns=fields.split(','))
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df['date'] = pd.to_datetime(df['date'])

        return df

    def save_cache(self, data, name):
        """保存数据缓存"""
        path = os.path.join(self.cache_dir, f'{name}.pkl')
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def load_cache(self, name):
        """加载数据缓存"""
        path = os.path.join(self.cache_dir, f'{name}.pkl')
        if os.path.exists(path):
            with open(path, 'rb') as f:
                return pickle.load(f)
        return None

    def update_data(self, stock_list, cache_name='stock_data'):
        """增量更新数据"""
        cached = self.load_cache(cache_name)

        if cached is not None:
            last_date = cached['date'].max()
            start_date = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
        else:
            start_date = (datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d')

        end_date = datetime.now().strftime('%Y-%m-%d')

        print(f"更新数据: {start_date} ~ {end_date}")
        new_data = self.get_all_data(stock_list, start_date, end_date)

        if new_data is not None and len(new_data) > 0:
            if cached is not None:
                data = pd.concat([cached, new_data], ignore_index=True)
                data = data.drop_duplicates(subset=['date', 'code'], keep='last')
            else:
                data = new_data

            self.save_cache(data, cache_name)
            return data

        return cached


def get_trade_calendar(start_date, end_date):
    """获取交易日历"""
    lg = bs.login()
    rs = bs.query_trade_dates(start_date=start_date, end_date=end_date)

    dates = []
    while rs.next():
        row = rs.get_row_data()
        if row[1] == '1':  # is_trading_day
            dates.append(row[0])

    bs.logout()
    return dates


if __name__ == '__main__':
    # 测试
    loader = DataLoader()
    stocks = loader.get_stock_list()
    print(f"沪深300成分股: {len(stocks)}只")

    # 获取示例数据
    df = loader.get_daily_data('sh.600519', '2024-01-01', '2024-12-31')
    print(f"\n贵州茅台数据: {len(df)}条")
    print(df.head())
