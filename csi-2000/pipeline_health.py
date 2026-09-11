#!/usr/bin/env python3
"""管线自检 —— 把"能自动判定的不变量"检查一遍

为什么要有这个
--------------
2026-09-07 一天内查出 6 个缺陷，全是**静默失败**:

    飞书主动推送从未生效        收件人环境变量没配，push 只 log 一行就 return
    交易日判断卡死整条链路      baostock.login() 无超时，挂起不抛异常
    预测扩展抛 NameError        用了不存在的变量名，每天失败但只 log 一句
    因子指纹闭不上环            检查方要求的字段，没有任何代码在写
    原子替换回滚失效            只捕 PermissionError，实际抛 FileNotFoundError
    掘金下单 price=0            限价单没有价格，必然拒单(只验过空跑)

共同点: **退出码都是 0，日志都"看起来正常"**。靠人读日志发现不了，
靠 LLM 读日志也发现不了 —— 因为日志里没有异常，缺的是"本该发生却没发生"。

所以这里检查的不是"有没有报错"，而是**状态有没有按预期推进**:
数据日历是否跟得上今天、预测是否跟得上数据、模拟盘是否跟得上预测、
定时任务上次退出码是不是 0。任何一环停滞都会在这里显形。

用法
----
    python pipeline_health.py            # 打印
    python pipeline_health.py --json     # 供 self_reflect 消费
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path

BOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BOT_DIR))

# 单项检查的容忍阈值(交易日)。超过即判为停滞。
MAX_DATA_LAG = 1        # 数据日历落后今天
MAX_PRED_LAG = 2        # 预测落后数据
MAX_PAPER_LAG = 2       # 模拟盘落后预测


def _checks() -> list[dict]:
    out: list[dict] = []

    def add(name, ok, detail, severity="error"):
        out.append({"name": name, "ok": bool(ok), "detail": detail,
                    "severity": severity if not ok else "info"})

    # --- 1. 数据日历是否跟得上今天 ---
    try:
        from market_calendar import is_trading_day
        cal = Path.home() / ".qlib/qlib_data/cn_data_bs/calendars/day.txt"
        days = [l.strip() for l in cal.read_text(encoding="utf-8").splitlines() if l.strip()]
        last = days[-1]
        today = date.today()
        gap = sum(1 for d in days_between(last, today.isoformat())
                  if is_trading_day(date.fromisoformat(d)))
        add("数据日历", gap <= MAX_DATA_LAG,
            f"末日 {last}，距今 {gap} 个交易日 (阈值 {MAX_DATA_LAG})")
    except Exception as e:
        add("数据日历", False, f"检查失败: {type(e).__name__}: {e}")

    # --- 2. 预测是否跟得上数据 ---
    try:
        import pandas as pd
        import yaml
        cfg = yaml.safe_load((BOT_DIR / "config/signal_config.yaml").read_text(encoding="utf-8"))
        pkl = (BOT_DIR / cfg["model_cache_dir"] /
               f"{cfg['rolling_config']}_{cfg['preset']}_{cfg['model']}.pkl")
        pred_last = pd.read_pickle(pkl).index.get_level_values(0).max().date().isoformat()
        cal = Path.home() / ".qlib/qlib_data/cn_data_bs/calendars/day.txt"
        days = [l.strip() for l in cal.read_text(encoding="utf-8").splitlines() if l.strip()]
        lag = len([d for d in days if d > pred_last])
        add("预测覆盖", lag <= MAX_PRED_LAG,
            f"末日 {pred_last}，落后数据 {lag} 个交易日 (阈值 {MAX_PRED_LAG})")
    except Exception as e:
        add("预测覆盖", False, f"检查失败: {type(e).__name__}: {e}")

    # --- 3. 因子指纹是否写入(不写则每天的增量扩展都会被拒) ---
    try:
        import yaml
        cfg = yaml.safe_load((BOT_DIR / "config/signal_config.yaml").read_text(encoding="utf-8"))
        j = (BOT_DIR / cfg["rolling_json_dir"] /
             f"{cfg['rolling_config']}_{cfg['preset']}_{cfg['model']}.json")
        fp = json.loads(j.read_text(encoding="utf-8")).get("factor_fingerprint")
        from factor_lab.factors.presets import factor_fingerprint
        cur = factor_fingerprint(cfg["preset"])
        if not fp:
            add("因子指纹", False, "预测缓存无指纹 —— 增量扩展会被拒，每天都得全量重训")
        else:
            same = fp.get("hash") == cur["hash"]
            add("因子指纹", same,
                f"缓存 {fp.get('n_factors')} 因子 vs 当前 {cur['n_factors']} 因子"
                + ("" if same else " —— 不一致，增量扩展会被拒"))
    except Exception as e:
        add("因子指纹", False, f"检查失败: {type(e).__name__}: {e}")

    # --- 4. 掘金仿真账户 ---
    #
    # 2026-09-08 由"模拟盘重放"换成这个。重放每天把 2024 年至今重算一遍,
    # 649 天里 648 天的结果和昨天完全相同,只为多算 1 天 —— 检查它等于
    # 每天确认历史没变。而掘金是真正在成交的账户,按真实盘口撮合,
    # 它有没有仓位、有没有成交才是"这套系统今天到底干活了没有"的答案。
    # paper_trader 引擎本身保留: 它是唯一过了对账的回测引擎,
    # run_phase_test / 参数配对检验都靠它,只是不再每天跑。
    try:
        gm = json.loads((BOT_DIR / "gm_state.json").read_text(encoding="utf-8"))
        snap = gm.get("snapshot") or {}
        st = gm.get("status")
        npos = len(snap.get("positions") or [])
        nav = snap.get("nav")
        upd = (gm.get("updated") or "")[:10]
        if st == "terminal_offline":
            add("掘金仿真", False, f"终端未启动 (最后更新 {upd}) —— 不会有任何成交")
        elif nav is None:
            add("掘金仿真", False, f"账户快照无效 (status={st})")
        else:
            add("掘金仿真", True,
                f"nav {nav:,.0f}，持仓 {npos} 只，最后更新 {upd}")
    except FileNotFoundError:
        add("掘金仿真", False, "gm_state.json 不存在 —— 桥接从未成功运行过")
    except Exception as e:
        add("掘金仿真", False, f"检查失败: {type(e).__name__}: {e}")

    # --- 5. 定时任务上次退出码 ---
    try:
        ps = ("Get-ScheduledTask -TaskName 'TradingSystem*' | ForEach-Object { "
              "$i=$_|Get-ScheduledTaskInfo; "
              "\"$($_.TaskName)|$($i.LastRunTime)|$($i.LastTaskResult)\" }")
        r = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                           capture_output=True, text=True, timeout=60,
                           encoding="utf-8", errors="replace")
        bad, busy = [], []
        for line in (r.stdout or "").strip().splitlines():
            parts = line.strip().split("|")
            if len(parts) != 3:
                continue
            name, lastrun, rc = parts
            # 这些不是失败，别报警:
            #   0          成功
            #   267011     0x41303 从未运行过(刚注册)
            #   267009     0x41301 当前正在运行(长任务如挖掘会跑 5 小时)
            #   2147946720 0x800710E0 上一实例仍在运行被拒(5分钟一次的监控常见)
            #              —— 它本身不是脚本错误; 真正卡死会由"数据/预测停滞"
            #              那几项检查暴露出来，不需要在这里重复判定
            benign = {"0", "267011", "267009", "2147946720"}
            if rc not in benign:
                bad.append(f"{name} rc={rc} (上次 {lastrun})")
            elif rc == "2147946720":
                busy.append(name)
        detail = "全部退出码正常" if not bad else "; ".join(bad)
        if busy:
            detail += f" | 曾因上一实例未结束被跳过: {', '.join(busy)}"
        add("定时任务", not bad, detail)
    except Exception as e:
        add("定时任务", False, f"检查失败: {type(e).__name__}: {e}")

    # --- 6. 飞书收件人是否可解析(否则所有主动推送静默丢弃) ---
    try:
        from dotenv import load_dotenv
        load_dotenv(BOT_DIR / ".env", override=True)
        from feishu_target import resolve_open_id
        uid = resolve_open_id()
        add("飞书收件人", bool(uid),
            f"已解析 {uid[:8]}****" if uid else "无法解析 —— 所有主动推送会被静默丢弃")
    except Exception as e:
        add("飞书收件人", False, f"检查失败: {type(e).__name__}: {e}")

    return out


def days_between(start: str, end: str) -> list[str]:
    """(start, end] 之间的日历日"""
    from datetime import timedelta
    s, e = date.fromisoformat(start), date.fromisoformat(end)
    return [(s + timedelta(days=i)).isoformat() for i in range(1, (e - s).days + 1)]


def run() -> dict:
    cs = _checks()
    failed = [c for c in cs if not c["ok"]]
    return {"ts": datetime.now().isoformat(timespec="seconds"),
            "passed": len(cs) - len(failed), "total": len(cs),
            "checks": cs, "failed": [c["name"] for c in failed]}


if __name__ == "__main__":
    res = run()
    if "--json" in sys.argv:
        print(json.dumps(res, ensure_ascii=False, indent=2))
    else:
        for c in res["checks"]:
            print(f"  {'OK ' if c['ok'] else 'BAD'}  {c['name']:10} {c['detail']}")
        print(f"\n  {res['passed']}/{res['total']} 通过")
    sys.exit(0 if not res["failed"] else 1)
