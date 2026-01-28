"""
数据库模块
支持SQLite和MySQL，用于存储历史数据和交易记录
"""

import sqlite3
import os
from datetime import date, datetime
from typing import List, Dict, Any, Optional
import pandas as pd
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class Database:
    """数据库管理类"""

    def __init__(self, db_path: str = "data/quant_trading.db"):
        """
        初始化数据库

        Args:
            db_path: SQLite数据库文件路径
        """
        self.db_path = db_path
        self._ensure_dir()
        self._init_tables()

    def _ensure_dir(self):
        """确保数据目录存在"""
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir)

    @contextmanager
    def get_connection(self):
        """获取数据库连接"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    def _init_tables(self):
        """初始化数据库表"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # K线数据表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS kline_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    code VARCHAR(20) NOT NULL,
                    market VARCHAR(10) NOT NULL,
                    date DATE NOT NULL,
                    kline_type VARCHAR(10) NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    amount REAL,
                    turnover REAL,
                    adjust_flag INTEGER DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(code, date, kline_type, adjust_flag)
                )
            ''')

            # 股票基本信息表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS stock_info (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    code VARCHAR(20) NOT NULL,
                    name VARCHAR(100),
                    market VARCHAR(10) NOT NULL,
                    industry VARCHAR(100),
                    sector VARCHAR(100),
                    list_date DATE,
                    total_shares REAL,
                    float_shares REAL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(code, market)
                )
            ''')

            # 交易记录表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trade_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    order_id VARCHAR(50) NOT NULL UNIQUE,
                    code VARCHAR(20) NOT NULL,
                    market VARCHAR(10) NOT NULL,
                    direction VARCHAR(10) NOT NULL,  -- BUY/SELL
                    price REAL NOT NULL,
                    quantity INTEGER NOT NULL,
                    amount REAL NOT NULL,
                    commission REAL DEFAULT 0,
                    stamp_duty REAL DEFAULT 0,
                    status VARCHAR(20) DEFAULT 'PENDING',
                    strategy_name VARCHAR(100),
                    trade_time TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # 持仓记录表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    code VARCHAR(20) NOT NULL,
                    market VARCHAR(10) NOT NULL,
                    quantity INTEGER NOT NULL,
                    avg_cost REAL NOT NULL,
                    current_price REAL,
                    profit_loss REAL,
                    profit_loss_pct REAL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(code, market)
                )
            ''')

            # 账户资金表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS account (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    total_assets REAL NOT NULL,
                    available_cash REAL NOT NULL,
                    frozen_cash REAL DEFAULT 0,
                    market_value REAL DEFAULT 0,
                    total_profit_loss REAL DEFAULT 0,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # 策略记录表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS strategy_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_name VARCHAR(100) NOT NULL,
                    signal_type VARCHAR(20) NOT NULL,  -- BUY/SELL/HOLD
                    code VARCHAR(20) NOT NULL,
                    market VARCHAR(10) NOT NULL,
                    price REAL,
                    reason TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # 回测结果表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS backtest_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_name VARCHAR(100) NOT NULL,
                    start_date DATE NOT NULL,
                    end_date DATE NOT NULL,
                    initial_capital REAL NOT NULL,
                    final_capital REAL NOT NULL,
                    total_return REAL,
                    annual_return REAL,
                    max_drawdown REAL,
                    sharpe_ratio REAL,
                    win_rate REAL,
                    profit_factor REAL,
                    total_trades INTEGER,
                    parameters TEXT,  -- JSON格式的策略参数
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # 创建索引
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_kline_code_date ON kline_data(code, date)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_trade_code ON trade_records(code)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_trade_time ON trade_records(trade_time)')

            logger.info("数据库表初始化完成")

    def save_kline_data(self, df: pd.DataFrame, market: str = "SH"):
        """保存K线数据"""
        if df.empty:
            return

        with self.get_connection() as conn:
            for _, row in df.iterrows():
                try:
                    conn.execute('''
                        INSERT OR REPLACE INTO kline_data
                        (code, market, date, kline_type, open, high, low, close, volume, amount, adjust_flag)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        row['code'],
                        market,
                        row['date'],
                        row.get('kline_type', '1d'),
                        row['open'],
                        row['high'],
                        row['low'],
                        row['close'],
                        row['volume'],
                        row.get('amount', 0),
                        row.get('adjust_flag', 1)
                    ))
                except Exception as e:
                    logger.error(f"保存K线数据失败: {e}")

    def get_kline_data(
        self,
        code: str,
        start_date: date,
        end_date: date,
        kline_type: str = "1d"
    ) -> pd.DataFrame:
        """获取K线数据"""
        with self.get_connection() as conn:
            query = '''
                SELECT code, date, open, high, low, close, volume, amount
                FROM kline_data
                WHERE code = ? AND date >= ? AND date <= ? AND kline_type = ?
                ORDER BY date
            '''
            df = pd.read_sql_query(
                query,
                conn,
                params=(code, start_date, end_date, kline_type)
            )
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
            return df

    def save_trade_record(self, trade: Dict[str, Any]):
        """保存交易记录"""
        with self.get_connection() as conn:
            conn.execute('''
                INSERT INTO trade_records
                (order_id, code, market, direction, price, quantity, amount,
                 commission, stamp_duty, status, strategy_name, trade_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade['order_id'],
                trade['code'],
                trade['market'],
                trade['direction'],
                trade['price'],
                trade['quantity'],
                trade['amount'],
                trade.get('commission', 0),
                trade.get('stamp_duty', 0),
                trade.get('status', 'PENDING'),
                trade.get('strategy_name', ''),
                trade.get('trade_time', datetime.now())
            ))

    def update_trade_status(self, order_id: str, status: str):
        """更新交易状态"""
        with self.get_connection() as conn:
            conn.execute('''
                UPDATE trade_records
                SET status = ?, updated_at = CURRENT_TIMESTAMP
                WHERE order_id = ?
            ''', (status, order_id))

    def get_trade_records(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        code: Optional[str] = None
    ) -> pd.DataFrame:
        """获取交易记录"""
        with self.get_connection() as conn:
            query = 'SELECT * FROM trade_records WHERE 1=1'
            params = []

            if start_date:
                query += ' AND trade_time >= ?'
                params.append(start_date)
            if end_date:
                query += ' AND trade_time <= ?'
                params.append(end_date)
            if code:
                query += ' AND code = ?'
                params.append(code)

            query += ' ORDER BY trade_time DESC'

            return pd.read_sql_query(query, conn, params=params)

    def save_position(self, position: Dict[str, Any]):
        """保存或更新持仓"""
        with self.get_connection() as conn:
            conn.execute('''
                INSERT OR REPLACE INTO positions
                (code, market, quantity, avg_cost, current_price, profit_loss, profit_loss_pct, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', (
                position['code'],
                position['market'],
                position['quantity'],
                position['avg_cost'],
                position.get('current_price', 0),
                position.get('profit_loss', 0),
                position.get('profit_loss_pct', 0)
            ))

    def get_positions(self) -> pd.DataFrame:
        """获取所有持仓"""
        with self.get_connection() as conn:
            return pd.read_sql_query(
                'SELECT * FROM positions WHERE quantity > 0',
                conn
            )

    def delete_position(self, code: str, market: str):
        """删除持仓记录"""
        with self.get_connection() as conn:
            conn.execute(
                'DELETE FROM positions WHERE code = ? AND market = ?',
                (code, market)
            )

    def save_account(self, account: Dict[str, Any]):
        """保存账户信息"""
        with self.get_connection() as conn:
            conn.execute('''
                INSERT INTO account
                (total_assets, available_cash, frozen_cash, market_value, total_profit_loss)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                account['total_assets'],
                account['available_cash'],
                account.get('frozen_cash', 0),
                account.get('market_value', 0),
                account.get('total_profit_loss', 0)
            ))

    def get_latest_account(self) -> Optional[Dict[str, Any]]:
        """获取最新账户信息"""
        with self.get_connection() as conn:
            cursor = conn.execute(
                'SELECT * FROM account ORDER BY updated_at DESC LIMIT 1'
            )
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None

    def save_backtest_result(self, result: Dict[str, Any]):
        """保存回测结果"""
        import json
        with self.get_connection() as conn:
            conn.execute('''
                INSERT INTO backtest_results
                (strategy_name, start_date, end_date, initial_capital, final_capital,
                 total_return, annual_return, max_drawdown, sharpe_ratio, win_rate,
                 profit_factor, total_trades, parameters)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                result['strategy_name'],
                result['start_date'],
                result['end_date'],
                result['initial_capital'],
                result['final_capital'],
                result.get('total_return', 0),
                result.get('annual_return', 0),
                result.get('max_drawdown', 0),
                result.get('sharpe_ratio', 0),
                result.get('win_rate', 0),
                result.get('profit_factor', 0),
                result.get('total_trades', 0),
                json.dumps(result.get('parameters', {}))
            ))

    def get_backtest_results(self, strategy_name: Optional[str] = None) -> pd.DataFrame:
        """获取回测结果"""
        with self.get_connection() as conn:
            query = 'SELECT * FROM backtest_results'
            params = []

            if strategy_name:
                query += ' WHERE strategy_name = ?'
                params.append(strategy_name)

            query += ' ORDER BY created_at DESC'

            return pd.read_sql_query(query, conn, params=params)
