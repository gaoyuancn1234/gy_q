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
