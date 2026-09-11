"""A 股交易日判断 —— Tushare trade_cal, 不依赖新浪/BaoStock

查询顺序:
  1. 周末 → False
  2. Tushare 日历缓存 (data_hub, 含未来已公布休市)
  3. 都不可用 → 按工作日处理 (fail-open)

fail-open: 节假日多跑一次只是查不到数据, 把交易日误判成休市
会让当天信号全部静默跳过。
"""
from __future__ import annotations

from datetime import date


def _lines() -> list[str]:
    """截至今天的开市日期, 给间隔/新鲜度用, 不含未来已公布交易日。"""
    from data_hub.calendar import open_days
    today = date.today().strftime("%Y-%m-%d")
    return [d for d in open_days() if d <= today]


def is_trading_day(day: date | None = None) -> bool:
    """今天(或指定日)是否为 A 股交易日。"""
    day = day or date.today()
    if day.weekday() >= 5:
        return False
    from data_hub.calendar import is_open
    try:
        hit = is_open(day)
    except Exception:
        hit = None
    if hit is not None:
        return bool(hit)
    return True


def trading_days_between(start: str, end: str) -> int | None:
    """本地日历上 (start, end] 之间的交易日数。答不了返回 None。"""
    try:
        lines = _lines()
    except Exception:
        return None
    if not lines or start > end:
        return None
    if start < lines[0] or end > lines[-1]:
        return None
    return sum(1 for d in lines if start < d <= end)


def next_open_day(day: str) -> str:
    """day 之后的下一个交易日 (含已公布的未来休市日历)。"""
    from data_hub.calendar import open_days
    key = str(day)[:10]
    for d in open_days():
        if d > key:
            return d
    raise RuntimeError(f"交易日历里找不到 {key} 之后的开市日")


def calendar_status(signal_date: str, today: str = None) -> dict:
    """信号相对交易日历的新鲜度。"""
    from datetime import datetime as _dt
    try:
        lines = _lines()
    except Exception:
        return {
            "trading_gap": None,
            "calendar_lag": None,
            "calendar_end": None,
        }
    if not lines:
        return {
            "trading_gap": None,
            "calendar_lag": None,
            "calendar_end": None,
        }
    cal_end = lines[-1]
    today = today or date.today().strftime("%Y-%m-%d")
    gap = None
    if signal_date >= lines[0]:
        gap = sum(1 for d in lines if signal_date < d <= cal_end)
    try:
        lag = (
            _dt.strptime(today, "%Y-%m-%d").date()
            - _dt.strptime(cal_end, "%Y-%m-%d").date()
        ).days
    except ValueError:
        lag = None
    return {
        "trading_gap": gap,
        "calendar_lag": lag,
        "calendar_end": cal_end,
    }


if __name__ == "__main__":
    import sys
    d = (
        date.fromisoformat(sys.argv[1])
        if len(sys.argv) > 1
        else date.today()
    )
    print(
        f"{d} tushare={is_trading_day(d)} "
        f"status={calendar_status(d.strftime('%Y-%m-%d'))}"
    )
