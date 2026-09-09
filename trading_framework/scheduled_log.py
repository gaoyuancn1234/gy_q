"""定时任务的自重定向日志

2026-09-08/09 教训: 给 FactorMiner / Reconcile 两个定时任务配的
`cmd.exe /c "python ..." > "日志" 2>&1` 重定向，手动在 shell 里跑完全正常，
但**通过 Task Scheduler 触发时进程根本没起来**(LastTaskResult=1，
无 cmd.exe/python.exe 进程，无日志文件)——同一条命令字符串在两种调用
路径下行为不同，具体差异没有查清，但没必要继续猜引号写法。

这个项目里真正稳定的任务(DailyRunner/SelfReflect/gm_bridge)有一个共同点:
**从不依赖 shell 重定向**，Task Scheduler 直接调 python.exe，日志由 Python
自己写文件。DailyRunner 用 logging.basicConfig，gm_bridge 自己攒 buffer
写文件；这两个脚本(factor_miner/reconcile)从头到尾是裸 print()，不值得为
了这点事把几百处 print 都改成 logging，所以选择更省事的等价方案:
进程一启动就把 sys.stdout/stderr 接到文件上，之后所有 print() 原样生效。

用法: 在 `if __name__ == '__main__':` 的第一行调用
    from scheduled_log import redirect_to_file
    redirect_to_file('factor_miner')
"""
from __future__ import annotations

import sys
from pathlib import Path


def redirect_to_file(name: str, mode: str = "w") -> Path:
    """把 stdout/stderr 接到 logs/{name}.log

    固定文件名(不带日期戳): 上一版用 cmd 的 %date% 做文件名，在中文 locale
    下 %date% 是"周三 2026/09/09"而非西式格式，子串截取拿到乱码文件名
    ——固定名反而更可靠，且与 daily_runner.log 等既有文件同一套约定。

    mode='w' 每次覆盖(挖掘/对账都是"这次跑的完整记录"，不需要累积)；
    需要保留历史时调用方自己在覆盖前 rename 旧文件。
    """
    log_dir = Path(__file__).resolve().parent / "logs"
    log_dir.mkdir(exist_ok=True)
    path = log_dir / f"{name}.log"
    f = open(path, mode, encoding="utf-8", buffering=1)
    sys.stdout = f
    sys.stderr = f
    return path
