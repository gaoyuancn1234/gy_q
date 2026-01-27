"""
全局配置设置
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum


class Market(Enum):
    """市场类型"""
    A_STOCK = "a_stock"      # A股
    HK_STOCK = "hk_stock"    # 港股
    US_STOCK = "us_stock"    # 美股


class TradingMode(Enum):
    """交易模式"""
    LIVE = "live"            # 实盘交易
    PAPER = "paper"          # 模拟交易
    BACKTEST = "backtest"    # 回测模式


@dataclass
class DatabaseConfig:
    """数据库配置"""
    host: str = "localhost"
    port: int = 3306
    user: str = "root"
    password: str = ""
    database: str = "quant_trading"

    # SQLite配置（默认使用）
    sqlite_path: str = "data/quant_trading.db"
    use_sqlite: bool = True


@dataclass
class RedisConfig:
    """Redis配置（用于缓存和消息队列）"""
    host: str = "localhost"
    port: int = 6379
    password: str = ""
    db: int = 0


@dataclass
class LogConfig:
    """日志配置"""
    level: str = "INFO"
    log_dir: str = "logs"
    max_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


@dataclass
class RiskConfig:
    """风险控制配置"""
    # 单只股票最大持仓比例
    max_position_ratio: float = 0.3
    # 单日最大亏损比例
    max_daily_loss: float = 0.05
    # 总资产最大回撤
    max_drawdown: float = 0.2
    # 单笔交易最大金额
    max_trade_amount: float = 100000.0
    # 止损比例
    stop_loss_ratio: float = 0.08
    # 止盈比例
    take_profit_ratio: float = 0.15
    # 最大持仓股票数
    max_positions: int = 10


@dataclass
class TradingConfig:
    """交易配置"""
    # A股交易时间
    a_stock_morning_start: str = "09:30"
    a_stock_morning_end: str = "11:30"
    a_stock_afternoon_start: str = "13:00"
    a_stock_afternoon_end: str = "15:00"

    # 港股交易时间
    hk_stock_morning_start: str = "09:30"
    hk_stock_morning_end: str = "12:00"
    hk_stock_afternoon_start: str = "13:00"
    hk_stock_afternoon_end: str = "16:00"

    # 交易手续费率
    a_stock_commission: float = 0.0003  # 万三
    hk_stock_commission: float = 0.0005  # 万五

    # 印花税（卖出时收取）
    a_stock_stamp_duty: float = 0.001   # 千一
    hk_stock_stamp_duty: float = 0.001  # 千一

    # 最低手续费
    min_commission: float = 5.0


@dataclass
class Settings:
    """全局设置"""
    # 基础配置
    app_name: str = "QuantTradingSystem"
    version: str = "1.0.0"
    debug: bool = False

    # 交易模式
    trading_mode: TradingMode = TradingMode.PAPER

    # 支持的市场
    markets: List[Market] = field(default_factory=lambda: [Market.A_STOCK, Market.HK_STOCK])

    # 数据目录
    data_dir: str = "data"
    cache_dir: str = "cache"

    # 子配置
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    log: LogConfig = field(default_factory=LogConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)

    @classmethod
    def from_env(cls) -> 'Settings':
        """从环境变量加载配置"""
        settings = cls()

        # 从环境变量覆盖配置
        if os.getenv('TRADING_MODE'):
            settings.trading_mode = TradingMode(os.getenv('TRADING_MODE'))
        if os.getenv('DEBUG'):
            settings.debug = os.getenv('DEBUG').lower() == 'true'

        # 数据库配置
        if os.getenv('DB_HOST'):
            settings.database.host = os.getenv('DB_HOST')
        if os.getenv('DB_PORT'):
            settings.database.port = int(os.getenv('DB_PORT'))
        if os.getenv('DB_USER'):
            settings.database.user = os.getenv('DB_USER')
        if os.getenv('DB_PASSWORD'):
            settings.database.password = os.getenv('DB_PASSWORD')

        return settings

    @classmethod
    def from_file(cls, config_path: str) -> 'Settings':
        """从配置文件加载"""
        import json

        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)

        settings = cls()
        # 递归更新配置
        cls._update_from_dict(settings, config_dict)
        return settings

    @staticmethod
    def _update_from_dict(obj, config_dict: dict):
        """从字典更新对象属性"""
        for key, value in config_dict.items():
            if hasattr(obj, key):
                attr = getattr(obj, key)
                if hasattr(attr, '__dataclass_fields__'):
                    Settings._update_from_dict(attr, value)
                else:
                    setattr(obj, key, value)

    def to_dict(self) -> dict:
        """转换为字典"""
        from dataclasses import asdict
        result = asdict(self)
        # 转换枚举类型
        result['trading_mode'] = self.trading_mode.value
        result['markets'] = [m.value for m in self.markets]
        return result

    def save(self, config_path: str):
        """保存配置到文件"""
        import json

        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


# 全局配置实例
settings = Settings.from_env()
