"""Tushare 客户端: token、代理、限流重试只有这一份。"""
from __future__ import annotations

import os
import re
import time

from data_hub.paths import PROJECT_DIR

# 接口方建议间隔。用户可改。
SLEEP = 0.85
TRIES = 6
# Tushare 限流文案既有"后重试"也有"后再试"。
RATE_RE = re.compile(r"请\s*(\d+)\s*秒后(?:再试|重试)")


def pro_api():
    """构造 Tushare 客户端。token 与代理从 .env 读。"""
    import tushare as ts
    from dotenv import load_dotenv
    load_dotenv(PROJECT_DIR / ".env", override=True)
    token = os.environ.get("TUSHARE_TOKEN", "").strip()
    if not token:
        raise SystemExit("TUSHARE_TOKEN 未配置，见 .env")
    client = ts.pro_api(token)
    url = os.environ.get("TUSHARE_HTTP_URL", "").strip()
    if url:
        client._DataApi__http_url = url
    return client


def call(fn, tries: int = TRIES, **kwargs):
    """调接口。限流按提示等待后重试同一请求, 不跳过。"""
    for k in range(1, tries + 1):
        try:
            return fn(**kwargs)
        except Exception as exc:
            msg = str(exc)
            matched = RATE_RE.search(msg)
            if matched:
                wait = int(matched.group(1)) + 3
            elif _is_transient(msg):
                wait = 20 * k
            else:
                raise
            if k == tries:
                raise
            print(f"[hub] 限流/上游异常，等待 {wait}s "
                  f"({k}/{tries})", flush=True)
            time.sleep(wait)
    raise RuntimeError("unreachable")


def _is_transient(msg: str) -> bool:
    """502 / 超时视为可重试。"""
    low = msg.lower()
    return "502" in msg or "超时" in msg or "timeout" in low
