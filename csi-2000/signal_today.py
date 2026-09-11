#!/usr/bin/env python3
"""已废弃。CSI2000 不走 15 日 RS / 沪深300 蓝筹池。

请用: python -m daily status|run|signal
"""
from __future__ import annotations

import sys

if __name__ == "__main__":
    sys.exit(
        "signal_today.py 是 CSI300 RS 遗留入口, 已停用。"
        "CSI2000 用 python -m daily"
    )
