"""
日志模块
"""

import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from typing import Optional


def setup_logger(
    name: str = "quant_trading",
    level: str = "INFO",
    log_dir: str = "logs",
    max_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    console: bool = True,
    file: bool = True
) -> logging.Logger:
    """
    配置日志

    Args:
        name: 日志名称
        level: 日志级别
        log_dir: 日志目录
        max_size: 单个日志文件最大大小
        backup_count: 备份文件数量
        console: 是否输出到控制台
        file: 是否输出到文件

    Returns:
        Logger对象
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # 避免重复添加handler
    if logger.handlers:
        return logger

    # 日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 控制台输出
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # 文件输出
    if file:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # 普通日志文件（按大小轮转）
        log_file = os.path.join(log_dir, f"{name}.log")
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # 错误日志单独文件
        error_file = os.path.join(log_dir, f"{name}_error.log")
        error_handler = RotatingFileHandler(
            error_file,
            maxBytes=max_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        logger.addHandler(error_handler)

        # 交易日志（按日期轮转）
        trade_file = os.path.join(log_dir, f"{name}_trade.log")
        trade_handler = TimedRotatingFileHandler(
            trade_file,
            when='D',
            interval=1,
            backupCount=30,
            encoding='utf-8'
        )
        trade_handler.setFormatter(formatter)
        trade_handler.addFilter(TradeLogFilter())
        logger.addHandler(trade_handler)

    return logger


def get_logger(name: str = "quant_trading") -> logging.Logger:
    """获取logger"""
    return logging.getLogger(name)


class TradeLogFilter(logging.Filter):
    """交易日志过滤器"""

    def filter(self, record):
        # 只记录包含交易相关关键词的日志
        keywords = ['买入', '卖出', '成交', '委托', 'BUY', 'SELL', 'TRADE', 'ORDER']
        return any(kw in record.getMessage() for kw in keywords)


class LoggerAdapter(logging.LoggerAdapter):
    """日志适配器，支持额外上下文"""

    def process(self, msg, kwargs):
        # 添加额外上下文信息
        extra = self.extra.copy()
        extra.update(kwargs.get('extra', {}))
        kwargs['extra'] = extra
        return msg, kwargs
