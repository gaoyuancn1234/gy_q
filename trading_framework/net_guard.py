"""网络调用的硬超时护栏

2026-09-04 一天之内在三个地方踩到同一个坑: 第三方库(baostock / akshare)的
网络调用没有超时参数，源一挂就永久阻塞，而调用方写好的 try/except 降级
分支永远不会触发 —— 挂起不抛异常。

实际后果:
  - intraday_monitor.is_trading_day  一个实例从 09:35 卡到 11:05，烧 1394s CPU、
    一行日志没写，后续每 5 分钟的触发全被"上一实例仍在运行"拒绝
  - daily_runner.is_trading_day      同一份代码，18:00 任务同样会卡死
  - live_portfolio.get_current_prices 18:00 任务卡在这里 1 小时以上，
    而它下面就写着"失败时降级到本地缓存"
  - data_setup_sina 的逐只下载        卡在第 1 只上 8.5 小时

统一用这里的 run_with_timeout 包一层。超时的线程是 daemon，不 join ——
join 就等于把无限等待搬了个位置。
"""
from __future__ import annotations

import threading


class NetTimeout(TimeoutError):
    """网络调用超过预算 —— 单独一个类型，方便调用方与真实异常区分"""


def run_with_timeout(fn, seconds: float, what: str = "网络调用"):
    """在子线程里执行 fn，超时抛 NetTimeout

    fn 内部抛的异常会原样重新抛出，调用方原有的 except 分支照常工作。
    """
    box: dict = {}

    def work():
        try:
            box["v"] = fn()
        except BaseException as e:      # noqa: BLE001 - 原样转交给调用方
            box["e"] = e

    t = threading.Thread(target=work, daemon=True)
    t.start()
    t.join(seconds)
    if "e" in box:
        raise box["e"]
    if "v" not in box:
        raise NetTimeout(f"{what} 超过 {seconds}s 未返回")
    return box["v"]


def install_default_request_timeout(seconds: float = 20) -> None:
    """给 requests 装默认超时 —— 只在调用方没显式传 timeout 时生效

    akshare 内部的 requests.get 全都不传 timeout，改库不现实，
    在进程级打一个补丁最省事。幂等。
    """
    import requests

    if getattr(requests.Session, "_default_timeout_installed", False):
        return
    orig = requests.Session.request

    def patched(self, method, url, **kw):
        kw.setdefault("timeout", seconds)
        return orig(self, method, url, **kw)

    requests.Session.request = patched
    requests.Session._default_timeout_installed = True
