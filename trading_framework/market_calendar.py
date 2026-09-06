"""A 股交易日判断 —— 带硬超时，不依赖 BaoStock

2026-09-04: daily_runner 和 intraday_monitor 原先各自用 `baostock.login()`
判断交易日。BaoStock 长期连不上，而 `bs.login()` 没有超时参数，会一直空转
—— 实测盘中监控的一个实例从 09:35 卡到 11:05，烧掉 1394 秒 CPU 且一行日志
都没写出来，后续每 5 分钟的触发全部被"上一个实例还在跑"拒绝。任务看起来
是 Running，实际什么都没做。

查询顺序:
  1. 周末 → False (无需联网)
  2. 本地 qlib 日历 (离线, 只在日期落在已下载区间内时可用)
  3. akshare 新浪交易日历 (联网, 放在子线程里做硬超时)
  4. 都不可用 → 按工作日处理 (fail-open)

fail-open 是刻意的: 节假日多跑一次只是查不到数据，而把真实交易日误判成
休市会让当天的信号、监控、模拟盘全部静默跳过 —— 后者危险得多。
"""
from __future__ import annotations

import os
from datetime import date
from pathlib import Path

QLIB_CALENDAR = Path(
    os.path.expanduser("~/.qlib/qlib_data/cn_data_bs/calendars/day.txt"))
NET_TIMEOUT = 20  # 秒


def _from_local_calendar(day: date):
    """本地 qlib 日历。返回 True/False，日期超出已下载区间则返回 None。"""
    try:
        lines = [l.strip() for l in
                 QLIB_CALENDAR.read_text(encoding="utf-8").splitlines() if l.strip()]
    except OSError:
        return None
    if not lines:
        return None
    s = day.strftime("%Y-%m-%d")
    if s > lines[-1] or s < lines[0]:
        return None          # 超出区间，本地日历答不了
    return s in set(lines)


def _from_akshare(day: date):
    """新浪交易日历。放子线程里跑，超时就放弃(线程留给解释器回收)。"""
    import threading
    box = {}

    def work():
        try:
            import akshare as ak
            df = ak.tool_trade_date_hist_sina()
            box["v"] = day in set(df["trade_date"])
        except Exception as e:
            box["err"] = e

    t = threading.Thread(target=work, daemon=True)
    t.start()
    t.join(NET_TIMEOUT)
    return box.get("v")


def is_trading_day(day: date | None = None) -> bool:
    day = day or date.today()
    if day.weekday() >= 5:
        return False
    for src in (_from_local_calendar, _from_akshare):
        try:
            r = src(day)
        except Exception:
            r = None
        if r is not None:
            return bool(r)
    return True  # fail-open: 见模块开头


if __name__ == "__main__":
    import sys
    d = date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else date.today()
    print(f"{d} 本地日历={_from_local_calendar(d)} "
          f"akshare={_from_akshare(d)} → is_trading_day={is_trading_day(d)}")


def trading_days_between(start: str, end: str) -> int | None:
    """本地日历上 (start, end] 之间的交易日数。答不了返回 None。

    2026-09-05 新增。用于替换 daily_runner 里的 `rebalance_day_count += 1`
    —— 那个计数器名为"交易日计数"，实为"daily_runner 跑了几次":
      - 手动补跑一次        -> +2，调仓相位永久前移一天
      - 任务失败/机器关机     -> 不增，相位后移
      - 非交易日 fail-open 跑 -> +1，相位前移
    而回测 (paper_trader) 用的是真实交易日索引 `day_idx % rebal`。两者会
    随时间漂移到不同的调仓日 —— 与 n_drop/vol_target 曾在模拟盘缺失同类。
    改为按信号日在日历上的真实间隔判定，与回测同口径且对补跑幂等。
    """
    try:
        lines = [l.strip() for l in
                 QLIB_CALENDAR.read_text(encoding="utf-8").splitlines() if l.strip()]
    except OSError:
        return None
    if not lines or start > end:
        return None
    # 两端都得在已下载区间内，否则数出来的间隔是残缺的
    if start < lines[0] or end > lines[-1]:
        return None
    return sum(1 for d in lines if start < d <= end)


def calendar_status(signal_date: str, today: str = None) -> dict:
    """信号相对交易日历的新鲜度

    2026-09-06 新增，取代直接用 trading_days_between(signal, today) 判时效。

    原做法要求 today 也落在已下载的交易日历区间内，而**日历只含交易日** ——
    周六/周日/节假日跑就必然落到区间外，被判成"无法判定 -> 按过期处理"。
    实测 2026-09-06(周六) 的 dry run 即因此取消了调仓，而信号日 2026-09-04
    正是最近的交易日、完全新鲜。日历不含非交易日是正常的，不是数据故障。

    改为分开看两件事:

      trading_gap    日历上 (signal_date, 日历末端] 的交易日数。
                     衡量"有行情但没信号"的天数 —— 这才是信号本身的滞后。
      calendar_lag   今天 − 日历末端，按**日历日**算。
                     衡量数据管线的滞后。周末是 2，国庆最长约 9。
                     它变大说明刷新没跑成，此时 trading_gap 会假性为 0
                     (日历不再前进，看起来"没有新的交易日")，必须单独看。

    Returns:
        {'trading_gap': int|None, 'calendar_lag': int|None,
         'calendar_end': str|None}
        日历读不到或 signal_date 早于日历起点时相应字段为 None。
    """
    from datetime import date as _date, datetime as _dt

    try:
        lines = [l.strip() for l in
                 QLIB_CALENDAR.read_text(encoding="utf-8").splitlines() if l.strip()]
    except OSError:
        return {'trading_gap': None, 'calendar_lag': None, 'calendar_end': None}
    if not lines:
        return {'trading_gap': None, 'calendar_lag': None, 'calendar_end': None}

    cal_end = lines[-1]
    today = today or _date.today().strftime("%Y-%m-%d")

    # signal_date 早于日历起点 -> 数不出来
    gap = None
    if signal_date >= lines[0]:
        gap = sum(1 for d in lines if signal_date < d <= cal_end)

    try:
        lag = (_dt.strptime(today, "%Y-%m-%d").date()
               - _dt.strptime(cal_end, "%Y-%m-%d").date()).days
    except ValueError:
        lag = None

    return {'trading_gap': gap, 'calendar_lag': lag, 'calendar_end': cal_end}
