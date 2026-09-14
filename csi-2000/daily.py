#!/usr/bin/env python3
"""CSI2000 每日例行唯一入口。

收盘后 (15:05+):
    python -m daily run

查询:
    python -m daily status

14:45 截断出分 (路线 B, 需要已落盘模型):
    python -m daily auction

不刷新 CSI300, 不走新浪, 不注册 TradingSystem-* 任务。
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from qlib_paths import parse_exec_lag

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

CLOSE_HHMM = "15:05"
AUCTION_HHMM = "14:45"
PLACE_HHMM = "14:50"
PREFETCH_HHMM = "14:31"
SYNC_HHMM = "15:05"
OVN_PKL_NAME = "D_expand_3v_3r_alpha158_ovn_LightGBM.pkl"
OVN_JSON_NAME = "D_expand_3v_3r_alpha158_ovn_LightGBM.json"


def _cfg() -> dict:
    """读 signal_config。"""
    import yaml
    path = PROJECT_DIR / "config" / "signal_config.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _require_csi2000() -> dict:
    """本入口只服务中证2000。"""
    from qlib_paths import current_universe
    cfg = _cfg()
    uni = current_universe()
    if uni != "csi2000":
        raise SystemExit(f"配置是 {uni}, 不是 csi2000, 停")
    return cfg


def _now_hhmm() -> str:
    """本地时分。"""
    return datetime.now().strftime("%H:%M")


def cmd_status() -> int:
    """打印配置、日历、预测覆盖。"""
    cfg = _require_csi2000()
    from data_hub.validate import report
    from qlib_paths import qlib_dir, data_refresh_chain
    info = report("csi2000")
    pred = (
        PROJECT_DIR / cfg["model_cache_dir"]
        / f"{cfg['rolling_config']}_{cfg['preset']}_{cfg['model']}.pkl"
    )
    print(f"preset={cfg.get('preset')} exec_lag={cfg.get('exec_lag', 1)}")
    print(f"topk={cfg.get('topk')} n_drop={cfg.get('n_drop')}")
    print(f"qlib={qlib_dir('csi2000')}")
    print(f"刷新链={data_refresh_chain('csi2000')}")
    print(f"日线文件={info.get('daily_files')} "
          f"{info.get('daily_first')}~{info.get('daily_last')}")
    print(f"Qlib日历={info.get('qlib_days')} "
          f"{info.get('qlib_first')}~{info.get('qlib_last')}")
    print(f"预测pkl={pred.name} 存在={pred.exists()}")
    if pred.exists():
        import pandas as pd
        s = pd.read_pickle(pred)
        s = s.iloc[:, 0] if getattr(s, "ndim", 1) > 1 else s
        d = s.index.get_level_values(0)
        print(f"预测区间={d.min().date()}~{d.max().date()} 行={len(s)}")
    ovn = (
        PROJECT_DIR / "factor_lab" / "results" / "rolling"
        / "predictions" / OVN_PKL_NAME
    )
    print(f"ovn_pkl={ovn.name} 存在={ovn.exists()}")
    from factor_lab.lgb_store import models_ready
    print(f"booster_ready={models_ready(cfg.get('preset') or 'alpha158')}")
    gm = PROJECT_DIR / "gm_state.json"
    if gm.exists():
        import json
        st = json.loads(gm.read_text(encoding="utf-8"))
        snap = st.get("snapshot") or {}
        print(
            f"gm_account={st.get('account_id', '?')} "
            f"nav={snap.get('nav')} pos={len(snap.get('positions') or [])}"
        )
    fc = info.get("field_coverage") or {}
    bad = [k for k, v in fc.items()
           if v.get("nan_ratio") is not None and v["nan_ratio"] > 0.99]
    if fc:
        ok = [k for k in fc if k not in bad
              and (fc[k].get("nan_ratio") or 1) <= 0.99]
        print(f"字段可用: {', '.join(ok) or '无'}")
        if bad:
            print(f"字段全 NaN(依赖它的过滤会静默失效): {', '.join(bad)}")

    return 0


def cmd_refresh(*, force: bool = False) -> int:
    """15:05 后刷新 Tushare 日线到 Qlib 供给层。"""
    _require_csi2000()
    if not force and _now_hhmm() < CLOSE_HHMM:
        print(f"未到 {CLOSE_HHMM}, 日线可能不齐。加 --force 才刷新。")
        return 2
    from daily_runner import refresh_daily_data
    ok = refresh_daily_data()
    if not ok:
        raise SystemExit("日线刷新失败")
    return 0


def cmd_extend() -> int:
    """把 rolling 预测追到 Qlib 日历末日。"""
    cfg = _require_csi2000()
    from daily_runner import _update_daily_predictions
    if not _update_daily_predictions():
        raise SystemExit("增量预测失败 (检查 pkl/指纹是否匹配 preset)")
    print(f"extend 完成 preset={cfg.get('preset')}")
    return 0


def cmd_signal(*, dry_run: bool = False) -> int:
    """用最新预测写 pending_orders, 可选推飞书。"""
    cfg = _require_csi2000()
    if parse_exec_lag(cfg) == 0:
        # 2026-09-12: 原先只 print 不 return, 警告完照样往下跑 ——
        # exec_lag=0 时本命令会按收盘预测改写 pending, 覆盖 14:50 已下的单。
        print(
            "exec_lag=0 请用 python -m daily auction 写 pending。"
            "本命令走收盘预测, 14:50 已经过了。"
        )
        return 2
    from daily_runner import generate_and_push
    generate_and_push(dry_run=dry_run)
    return 0


def cmd_run(*, force: bool = False, dry_run: bool = False) -> int:
    """收盘后例行: 刷新 → 增量预测; exec_lag=0 时不改写当天 pending。"""
    cmd_refresh(force=force)
    cmd_extend()
    cfg = _cfg()
    lag = parse_exec_lag(cfg)
    if lag == 0:
        print("exec_lag=0: 下单在 14:50, 收盘后不再写 pending")
        return 0
    cmd_signal(dry_run=dry_run)
    return 0


def cmd_prefetch(*, day: str = "", limit: int = 0) -> int:
    """盘中预取当日分钟, 给 14:45 出分铺路。"""
    _require_csi2000()
    from qlib_engine.fetch_minute import fetch_today_minutes
    n = fetch_today_minutes(day=day or None, limit=limit)
    print(f"prefetch 成功 {n} 只")
    return 0 if n else 2


def cmd_auction(*, force: bool = False, dry_run: bool = False,
                day: str = "") -> int:
    """14:45 截断出分。没有落盘模型则明确失败, 不准静默用隔日预测。"""
    from datetime import date as _date
    from factor_lab.lgb_store import models_ready
    from factor_lab.auction_infer import score_day
    from factor_lab.signal_generator import SignalGenerator
    from daily_runner import generate_and_push

    cfg = _require_csi2000()
    lag = parse_exec_lag(cfg)
    if lag != 0 and not force:
        print(
            "exec_lag!=0, 14:45 出分会和成交时钟错位。"
            "先切 exec_lag=0, 或加 --force 只做演练。"
        )
        return 2
    if not force and _now_hhmm() < AUCTION_HHMM:
        print(f"未到 {AUCTION_HHMM}, 14:45 K 线还没有。")
        return 2
    preset = cfg.get("preset") or "alpha158"
    if not models_ready(preset):
        print(
            "14:45 出分需要落盘 LightGBM。"
            "跑 python -m factor_lab.persist_boosters"
            f"  preset={preset} exec_lag={lag}"
        )
        return 2
    target = day or _date.today().strftime("%Y-%m-%d")
    scores, px = score_day(target, preset)
    sg = SignalGenerator()
    signal = sg.signal_from_scores(scores, target)
    generate_and_push(
        dry_run=dry_run, signal=signal, prices=px or None,
    )
    print(f"auction 完成 {target} 分数 {len(scores)}")
    return 0


def _ovn_paths() -> tuple[Path, Path]:
    """隔夜重训产物路径。"""
    root = PROJECT_DIR / "factor_lab" / "results" / "rolling"
    return root / "predictions" / OVN_PKL_NAME, root / OVN_JSON_NAME


def cmd_activate_ovn() -> int:
    """pkl 落地后把 yaml 切到 alpha158_ovn + exec_lag=0。"""
    import re
    pkl, _json = _ovn_paths()
    if not pkl.exists():
        print(f"ovn pkl 还没有: {pkl.name}")
        return 2
    path = PROJECT_DIR / "config" / "signal_config.yaml"
    text = path.read_text(encoding="utf-8")
    ntext = re.sub(
        r"^preset:\s*alpha158\s*$",
        "preset: alpha158_ovn",
        text,
        count=1,
        flags=re.M,
    )
    ntext = re.sub(
        r"^exec_lag:\s*1\s*$",
        "exec_lag: 0",
        ntext,
        count=1,
        flags=re.M,
    )
    if ntext == text:
        cfg = _cfg()
        if (cfg.get("preset") == "alpha158_ovn"
                and parse_exec_lag(cfg) == 0):
            print("yaml 已经是 alpha158_ovn / exec_lag=0")
            return 0
        print("yaml 没有匹配到 preset/exec_lag 行, 未改")
        return 2
    path.write_text(ntext, encoding="utf-8")
    print("已切 preset=alpha158_ovn exec_lag=0")
    return 0


def cmd_place(*, force: bool = False, dry_run: bool = False) -> int:
    """14:50 把 pending 送到 CSI2000 掘金仿真账户。"""
    _require_csi2000()
    import subprocess
    cmd = [sys.executable, "-X", "utf8",
           str(PROJECT_DIR / "gm_bridge.py"), "--place"]
    if force:
        cmd.append("--force")
    if dry_run:
        cmd.append("--dry-run")
    return subprocess.call(cmd, cwd=str(PROJECT_DIR))


def cmd_sync() -> int:
    """回读掘金成交。"""
    _require_csi2000()
    import subprocess
    return subprocess.call(
        [sys.executable, "-X", "utf8",
         str(PROJECT_DIR / "gm_bridge.py"), "--sync"],
        cwd=str(PROJECT_DIR),
    )


def _sleep_until(hhmm: str) -> None:
    """等到本地时分。已经过了就直接返回。"""
    import time
    while _now_hhmm() < hhmm:
        print(f"等待 {hhmm}, 现在 {_now_hhmm()}", flush=True)
        time.sleep(30)


def _ovn_retrain_running() -> bool:
    """全量隔夜重训是否还在。pkl 写出后还会跑 Qlib 回测, 这时不能 persist。"""
    import subprocess
    ps = (
        "Get-CimInstance Win32_Process | "
        "Where-Object { $_.CommandLine -like "
        "'*run_rolling_benchmark*' -and $_.CommandLine -like "
        "'*alpha158_ovn*' } | Measure-Object | "
        "Select-Object -ExpandProperty Count"
    )
    try:
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command", ps],
            text=True,
        )
        return int((out or "0").strip() or "0") > 0
    except Exception:
        return False


def cmd_wait_live() -> int:
    """等隔夜 pkl → 落盘 booster → 切 yaml → 14:45 出分 → 14:50 下单。"""
    import time
    import subprocess
    from factor_lab.lgb_store import models_ready

    pkl, js = _ovn_paths()
    log_path = PROJECT_DIR / "logs" / "wait_live.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def wlog(msg: str) -> None:
        print(msg, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")

    wlog(f"等待 ovn pkl: {pkl.name}")
    while not pkl.exists():
        wlog(f"ovn pkl 未到 {_now_hhmm()}")
        time.sleep(45)
    wlog("LIVE_STEP pkl 已落地")
    wlog("等待重训进程退出 (含回测, 避免和 persist 抢内存)")
    while _ovn_retrain_running():
        wlog(f"重训仍在跑 {_now_hhmm()}")
        time.sleep(30)
    wlog("LIVE_STEP 重训进程已退出")
    if not models_ready("alpha158_ovn"):
        rc = 1
        for attempt in range(1, 3):
            wlog(f"LIVE_STEP persist_boosters 开始 try={attempt}")
            rc = subprocess.call(
                [sys.executable, "-X", "utf8", "-u",
                 "-m", "factor_lab.persist_boosters",
                 "--preset", "alpha158_ovn"],
                cwd=str(PROJECT_DIR),
            )
            if rc == 0:
                break
            wlog(f"LIVE_WARN persist rc={rc} try={attempt}")
            time.sleep(30)
        if rc != 0:
            wlog(f"LIVE_FAIL persist rc={rc}")
            return rc
        wlog("LIVE_STEP persist 完成")
    else:
        wlog("LIVE_STEP booster 已在")
    rc = cmd_activate_ovn()
    if rc != 0:
        wlog(f"LIVE_FAIL activate_ovn rc={rc}")
        return rc
    _sleep_until(PREFETCH_HHMM)
    wlog("LIVE_STEP prefetch")
    prc = cmd_prefetch()
    if prc != 0:
        wlog(f"LIVE_WARN prefetch rc={prc}, 仍尝试出分")
    _sleep_until(AUCTION_HHMM)
    wlog("LIVE_STEP auction")
    rc = cmd_auction(force=False, dry_run=False)
    if rc != 0:
        wlog(f"LIVE_FAIL auction rc={rc}")
        return rc
    _sleep_until(PLACE_HHMM)
    wlog("LIVE_STEP place")
    rc = cmd_place(force=False, dry_run=False)
    wlog(f"LIVE_STEP place rc={rc}")
    _sleep_until(SYNC_HHMM)
    wlog("LIVE_STEP sync")
    rc2 = cmd_sync()
    wlog(f"LIVE_DONE place={rc} sync={rc2}")
    return 0 if rc == 0 and rc2 == 0 else 1


def main(argv: list[str] | None = None) -> int:
    """入口。"""
    ap = argparse.ArgumentParser(description="CSI2000 每日例行")
    ap.add_argument(
        "cmd",
        nargs="?",
        default="status",
        choices=["status", "refresh", "extend", "signal",
                 "run", "auction", "prefetch", "place", "sync",
                 "activate_ovn", "wait_live"],
    )
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--day", default="", help="auction/prefetch 指定日期")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)
    if args.cmd == "status":
        return cmd_status()
    if args.cmd == "refresh":
        return cmd_refresh(force=args.force)
    if args.cmd == "extend":
        return cmd_extend()
    if args.cmd == "signal":
        return cmd_signal(dry_run=args.dry_run)
    if args.cmd == "run":
        return cmd_run(force=args.force, dry_run=args.dry_run)
    if args.cmd == "prefetch":
        return cmd_prefetch(day=args.day, limit=args.limit)
    if args.cmd == "auction":
        return cmd_auction(
            force=args.force, dry_run=args.dry_run, day=args.day,
        )
    if args.cmd == "place":
        return cmd_place(force=args.force, dry_run=args.dry_run)
    if args.cmd == "sync":
        return cmd_sync()
    if args.cmd == "activate_ovn":
        return cmd_activate_ovn()
    if args.cmd == "wait_live":
        return cmd_wait_live()
    return 2


# 定时任务入口的命令。Task Scheduler 直接调 python.exe, stdout 无处可去,
# 这几个跑起来一行记录都不留。
_SCHEDULED_CMDS = {"prefetch", "auction", "place", "sync"}


def _redirect_scheduled_log(cmd: str) -> None:
    """把定时任务的输出接到 logs/daily_<cmd>.log。

    2026-09-14: 14:00 预取被 Tushare 限流, 26 分钟一只没抓到, 而报错一行
    都没留下 —— 只能靠进程 CPU 时间和 parquet 的 mtime 反推出在限流。
    prefetch 一天跑两遍(14:00/14:25), redirect_to_file 是 mode='w', 直接用
    会让第二遍冲掉第一遍, 所以覆盖前把上一份留成 .prev。
    重定向本身失败不许影响主流程: 没有日志是难查, 不跑才是事故。
    """
    try:
        from scheduled_log import redirect_to_file
        p = PROJECT_DIR / "logs" / f"daily_{cmd}.log"
        if p.exists():
            p.replace(p.with_name(p.name + ".prev"))
        redirect_to_file(f"daily_{cmd}")
    except Exception as e:
        print(f"[daily] 日志重定向失败, 继续裸跑: {e}")


if __name__ == "__main__":
    _cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if _cmd in _SCHEDULED_CMDS:
        _redirect_scheduled_log(_cmd)
    sys.exit(main())
