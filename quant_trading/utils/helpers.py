"""
辅助函数
"""

from datetime import datetime, date, timedelta
from typing import List, Optional
import math


# 中国节假日列表（需要每年更新）
CHINESE_HOLIDAYS_2024 = [
    # 元旦
    date(2024, 1, 1),
    # 春节
    date(2024, 2, 10), date(2024, 2, 11), date(2024, 2, 12),
    date(2024, 2, 13), date(2024, 2, 14), date(2024, 2, 15),
    date(2024, 2, 16), date(2024, 2, 17),
    # 清明节
    date(2024, 4, 4), date(2024, 4, 5), date(2024, 4, 6),
    # 劳动节
    date(2024, 5, 1), date(2024, 5, 2), date(2024, 5, 3),
    date(2024, 5, 4), date(2024, 5, 5),
    # 端午节
    date(2024, 6, 8), date(2024, 6, 9), date(2024, 6, 10),
    # 中秋节
    date(2024, 9, 15), date(2024, 9, 16), date(2024, 9, 17),
    # 国庆节
    date(2024, 10, 1), date(2024, 10, 2), date(2024, 10, 3),
    date(2024, 10, 4), date(2024, 10, 5), date(2024, 10, 6),
    date(2024, 10, 7),
]

CHINESE_HOLIDAYS_2025 = [
    # 元旦
    date(2025, 1, 1),
    # 春节
    date(2025, 1, 28), date(2025, 1, 29), date(2025, 1, 30),
    date(2025, 1, 31), date(2025, 2, 1), date(2025, 2, 2),
    date(2025, 2, 3), date(2025, 2, 4),
    # 清明节
    date(2025, 4, 4), date(2025, 4, 5), date(2025, 4, 6),
    # 劳动节
    date(2025, 5, 1), date(2025, 5, 2), date(2025, 5, 3),
    date(2025, 5, 4), date(2025, 5, 5),
    # 端午节
    date(2025, 5, 31), date(2025, 6, 1), date(2025, 6, 2),
    # 中秋节+国庆节
    date(2025, 10, 1), date(2025, 10, 2), date(2025, 10, 3),
    date(2025, 10, 4), date(2025, 10, 5), date(2025, 10, 6),
    date(2025, 10, 7), date(2025, 10, 8),
]

CHINESE_HOLIDAYS = CHINESE_HOLIDAYS_2024 + CHINESE_HOLIDAYS_2025


def is_trading_day(d: date = None, market: str = "A_STOCK") -> bool:
    """
    判断是否为交易日

    Args:
        d: 日期，默认今天
        market: 市场类型 (A_STOCK/HK_STOCK)

    Returns:
        是否为交易日
    """
    if d is None:
        d = date.today()

    # 周末不交易
    if d.weekday() >= 5:
        return False

    # A股检查中国节假日
    if market == "A_STOCK":
        if d in CHINESE_HOLIDAYS:
            return False

    # 港股检查香港节假日（简化处理）
    if market == "HK_STOCK":
        # 香港节假日与中国大陆有重叠但不完全相同
        # 这里简化处理，实际使用应该维护完整的香港交易日历
        if d in CHINESE_HOLIDAYS:
            return False

    return True


def is_trading_time(market: str = "A_STOCK") -> bool:
    """
    判断是否在交易时间

    Args:
        market: 市场类型

    Returns:
        是否在交易时间
    """
    if not is_trading_day(market=market):
        return False

    now = datetime.now()
    current_time = now.time()

    if market == "A_STOCK":
        # A股交易时间: 9:30-11:30, 13:00-15:00
        morning_start = datetime.strptime("09:30", "%H:%M").time()
        morning_end = datetime.strptime("11:30", "%H:%M").time()
        afternoon_start = datetime.strptime("13:00", "%H:%M").time()
        afternoon_end = datetime.strptime("15:00", "%H:%M").time()

    elif market == "HK_STOCK":
        # 港股交易时间: 9:30-12:00, 13:00-16:00
        morning_start = datetime.strptime("09:30", "%H:%M").time()
        morning_end = datetime.strptime("12:00", "%H:%M").time()
        afternoon_start = datetime.strptime("13:00", "%H:%M").time()
        afternoon_end = datetime.strptime("16:00", "%H:%M").time()

    else:
        return False

    return (morning_start <= current_time <= morning_end or
            afternoon_start <= current_time <= afternoon_end)


def get_next_trading_day(d: date = None, market: str = "A_STOCK") -> date:
    """
    获取下一个交易日

    Args:
        d: 日期，默认今天
        market: 市场类型

    Returns:
        下一个交易日
    """
    if d is None:
        d = date.today()

    next_day = d + timedelta(days=1)
    while not is_trading_day(next_day, market):
        next_day += timedelta(days=1)

    return next_day


def get_trading_days(start_date: date, end_date: date, market: str = "A_STOCK") -> List[date]:
    """
    获取指定日期范围内的交易日列表

    Args:
        start_date: 开始日期
        end_date: 结束日期
        market: 市场类型

    Returns:
        交易日列表
    """
    trading_days = []
    current = start_date

    while current <= end_date:
        if is_trading_day(current, market):
            trading_days.append(current)
        current += timedelta(days=1)

    return trading_days


def format_price(price: float, decimal: int = 2) -> str:
    """格式化价格"""
    if price >= 1000000:
        return f"{price/1000000:.2f}M"
    elif price >= 1000:
        return f"{price/1000:.2f}K"
    else:
        return f"{price:.{decimal}f}"


def format_volume(volume: float) -> str:
    """格式化成交量"""
    if volume >= 100000000:
        return f"{volume/100000000:.2f}亿"
    elif volume >= 10000:
        return f"{volume/10000:.2f}万"
    else:
        return f"{volume:.0f}"


def format_amount(amount: float) -> str:
    """格式化金额"""
    if amount >= 100000000:
        return f"{amount/100000000:.2f}亿"
    elif amount >= 10000:
        return f"{amount/10000:.2f}万"
    else:
        return f"{amount:,.2f}"


def format_change(change: float, with_sign: bool = True) -> str:
    """格式化涨跌幅"""
    sign = '+' if change > 0 and with_sign else ''
    return f"{sign}{change:.2%}"


def calculate_position_size(
    total_capital: float,
    price: float,
    risk_per_trade: float = 0.02,
    stop_loss_pct: float = 0.08,
    max_position_pct: float = 0.2
) -> int:
    """
    计算建议仓位大小

    Args:
        total_capital: 总资金
        price: 当前价格
        risk_per_trade: 每笔交易风险比例
        stop_loss_pct: 止损比例
        max_position_pct: 最大仓位比例

    Returns:
        建议买入股数（A股100股整数倍）
    """
    # 根据风险计算
    risk_amount = total_capital * risk_per_trade
    risk_based_shares = risk_amount / (price * stop_loss_pct)

    # 根据最大仓位计算
    max_position_amount = total_capital * max_position_pct
    max_based_shares = max_position_amount / price

    # 取较小值
    shares = min(risk_based_shares, max_based_shares)

    # A股100股整数倍
    return int(shares / 100) * 100


def calculate_annual_return(total_return: float, days: int) -> float:
    """
    计算年化收益率

    Args:
        total_return: 总收益率
        days: 交易天数

    Returns:
        年化收益率
    """
    if days <= 0:
        return 0
    return (1 + total_return) ** (252 / days) - 1


def calculate_sharpe_ratio(
    returns: List[float],
    risk_free_rate: float = 0.03
) -> float:
    """
    计算夏普比率

    Args:
        returns: 日收益率列表
        risk_free_rate: 无风险利率（年化）

    Returns:
        夏普比率
    """
    import numpy as np

    if len(returns) < 2:
        return 0

    returns = np.array(returns)
    daily_rf = risk_free_rate / 252
    excess_returns = returns - daily_rf

    std = excess_returns.std()
    if std == 0:
        return 0

    return excess_returns.mean() / std * math.sqrt(252)


def calculate_max_drawdown(equity_curve: List[float]) -> tuple:
    """
    计算最大回撤

    Args:
        equity_curve: 权益曲线

    Returns:
        (最大回撤, 回撤开始位置, 回撤结束位置)
    """
    if len(equity_curve) < 2:
        return 0, 0, 0

    max_drawdown = 0
    peak_idx = 0
    trough_idx = 0
    peak = equity_curve[0]

    for i, value in enumerate(equity_curve):
        if value > peak:
            peak = value
            peak_idx = i

        drawdown = (peak - value) / peak
        if drawdown > max_drawdown:
            max_drawdown = drawdown
            trough_idx = i

    return max_drawdown, peak_idx, trough_idx


def round_price(price: float, tick_size: float = 0.01) -> float:
    """
    按最小价格单位取整

    Args:
        price: 价格
        tick_size: 最小价格单位

    Returns:
        取整后的价格
    """
    return round(price / tick_size) * tick_size


def round_quantity(quantity: int, lot_size: int = 100) -> int:
    """
    按最小交易单位取整

    Args:
        quantity: 数量
        lot_size: 最小交易单位

    Returns:
        取整后的数量
    """
    return (quantity // lot_size) * lot_size
