#!/usr/bin/env python3
"""冒烟套件 —— 每个生产入口用极小规模真跑一遍。

为什么要这个 (2026-09-13)
--------------------------
这套系统这两天查出的真缺陷，**几乎全是"跑起来"发现的，不是读代码发现的**：

    对账跑通            -> 逐日订单一致性、配置分叉
    _trunc_feat 跑完    -> 推翻"前视让数字整体乐观"的推断
    score_day 真跑      -> last_tod 用 .median() 当场崩 (字符串列)
    全量建截断日线       -> 量纲断言抓到 x100 的过度推广

而纯靠读代码得出的判断错了好几次。`daily auction` 那个崩溃尤其说明问题：
它在 score_day 的必经路径上，此前只做过**空跑**（空跑走不到那一步），
于是一个"实盘一跑就崩"的 bug 安静地躺着。

本套件与 reconcile.py 分工不同：
    reconcile  两条路径算出来的**结果**是否一致
    smoke      每条路径**还能不能跑**

所以这里只看退出码和异常，不判断数值对不对。

用法
----
    python smoke.py              # 全部（约 3-6 分钟）
    python smoke.py --list       # 只列出检查项
    python smoke.py -k auction   # 只跑名字含 auction 的

退出码: 0 = 全通过; 1 = 有失败
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJ = Path(__file__).resolve().parent
PY = [sys.executable, "-X", "utf8"]

# (名字, 命令, 说明, 允许的退出码)
# 允许多个退出码: 有些入口在"条件不满足"时返回 2 是**正确行为**，
# 不该算失败。真正要抓的是崩溃(非 0 非预期)和异常栈。
CHECKS: list[tuple[str, list[str], str, tuple[int, ...]]] = [
    ("daily-status", ["-m", "daily", "status"],
     "读配置/数据/预测/字段可用性", (0,)),
    ("reconcile", ["reconcile.py", "--days", "20"],
     "双路径对账(短窗)", (0,)),
    ("phase-test", ["run_phase_test.py", "--phases", "1",
                    "--start", "2026-06-01", "--end", "2026-09-11",
                    "--tag", "smoke"],
     "验收引擎单相位", (0,)),
    ("param-sweep", ["run_param_sweep.py", "--param", "vol_target",
                     "--values", "0.25", "0.20", "--phases", "1",
                     "--start", "2026-06-01", "--end", "2026-09-11",
                     "--tag", "smoke"],
     "参数配对工具", (0,)),
    ("gm-check", ["gm_bridge.py", "--check"],
     "掘金仿真账户只读查询", (0,)),
    ("auction-score", ["-c", (
        "import sys;sys.path.insert(0,'.');"
        "from factor_lab.auction_infer import score_day;"
        "s,px=score_day('2026-09-11','alpha158_ovn');"
        "assert len(s)>50, f'只出 {len(s)} 个分数';"
        "print(f'[smoke] score_day 出分 {len(s)} 只')")],
     "14:45 截断出分(实盘必经路径)", (0,)),
    ("trunc-daily", ["-m", "data_hub.trunc_daily", "--limit", "2"],
     "分钟->14:45 截断日线", (0,)),
    ("fetch-minute", ["-m", "qlib_engine.fetch_minute", "--limit", "1"],
     "分钟数据下载(1 只)", (0, 1)),
]


# 故意不放进冒烟的入口，写明理由，避免下次有人以为漏了
EXCLUDED = [
    ("daily refresh", "会触发全量 Tushare 下载(小时级)，且会覆盖生产数据"),
    ("daily place", "会向掘金仿真账户真实下单；--dry-run 亦依赖盘中时段"),
    ("factor_miner", "一轮 5 小时、多轮 LLM 调用，且会写因子池"),
    ("persist_boosters", "会覆盖落盘模型，需在重训之后单独跑"),
    ("retrain_pipeline", "会重写生产预测缓存"),
]

# 2026-09-13 运维约束: **两个 qlib 重任务不能并发**。
# 实测同时跑 run_phase_test 与 run_param_sweep, 两者的 joblib memmap 临时目录
# 互删, 后启动的那个在跑完第一个配置前就死于退出码 127:
#   resource_tracker: joblib_memmapping_folder_...: FileNotFoundError(2)
# 表现是莫名其妙失败而非明确报错, 所以记在这里。
# 冒烟套件本身是串行的, 但手动跑扫描/相位测试时要排队。
CONCURRENCY_NOTE = (
    "qlib 重任务(run_phase_test / run_param_sweep / run_rolling_benchmark / "
    "reconcile)必须串行, 并发会因 joblib memmap 目录互删而失败"
)



# 纯库不变量 —— 不需要子进程, 进程内直接断言。
# 放进冒烟是因为这些守卫本身也会失灵(改坏了就静默放行), 必须每次验证它们还在拦。

def _check_validator() -> str:
    from factor_lab.mining.validator import validate_expression as v
    ok, _ = v('SMOKE_OK', 'Div($close, Ref($close, 5))')
    if not ok:
        raise AssertionError('正常表达式被拒')
    bad, msg = v('SMOKE_LOOKAHEAD', 'Ref(Mean($close, 5), -3)')
    if bad:
        raise AssertionError('前视表达式未被拦')
    dead, _ = v('SMOKE_DEADFIELD', 'Div($turn, Mean($turn, 20))')
    if dead:
        raise AssertionError('全 NaN 字段($turn)未被拦')
    return '正常通过/前视拦下/死字段拦下'


def _check_bar() -> str:
    from data_hub.bar_check import check_daily_bar, BarUnitError
    check_daily_bar(code='x', date='d', open_=10, high=11, low=9,
                    close=10, volume=1e6, amount=1e7)
    for vol, why in ((1e4, '量纲错100倍'), (1e8, '量纲反向错')):
        try:
            check_daily_bar(code='x', date='d', open_=10, high=11, low=9,
                            close=10, volume=vol, amount=1e7)
        except BarUnitError:
            continue
        raise AssertionError(f'{why} 未被拦')
    return '正常通过/两种量纲错拦下'


def _check_exec_lag() -> str:
    from qlib_paths import parse_exec_lag
    cases = [({'exec_lag': 0}, 0), ({'exec_lag': 1}, 1),
             ({'exec_lag': None}, 1), ({}, 1), ({'exec_lag': '0'}, 0)]
    for cfg, want in cases:
        got = parse_exec_lag(cfg)
        if got != want:
            raise AssertionError(f'{cfg} -> {got}, 期望 {want}')
    return '0/1/None/缺失/字符串 五种输入判对'


LIB_CHECKS = [
    ("validator", _check_validator, "因子表达式验证(前视/死字段)"),
    ("bar-check", _check_bar, "日线量纲断言"),
    ("exec-lag", _check_exec_lag, "exec_lag 解析(0 不能被 or 吃掉)"),
]

def run_one(name: str, args: list[str], desc: str,
            ok_codes: tuple[int, ...]) -> tuple[bool, str, float]:
    t0 = time.time()
    try:
        r = subprocess.run(PY + args, cwd=str(PROJ), capture_output=True,
                           text=True, encoding="utf-8", errors="replace",
                           timeout=900)
    except subprocess.TimeoutExpired:
        return False, "超时 (>900s)", time.time() - t0
    el = time.time() - t0
    out = (r.stdout or "") + (r.stderr or "")
    # 退出码对但输出里有 Traceback 也算失败 —— 有些入口吞异常后照样返回 0
    if "Traceback (most recent call last)" in out:
        tail = [l for l in out.splitlines() if l.strip()][-1:]
        return False, f"退出码 {r.returncode} 但有 Traceback: {tail[0][:90]}", el
    if r.returncode not in ok_codes:
        tail = [l for l in out.splitlines() if l.strip()][-1:]
        last = tail[0][:90] if tail else ""
        return False, f"退出码 {r.returncode} (允许 {ok_codes}): {last}", el
    return True, f"退出码 {r.returncode}", el


def main() -> int:
    ap = argparse.ArgumentParser(description="生产入口冒烟")
    ap.add_argument("--list", action="store_true", help="只列出检查项")
    ap.add_argument("-k", default="", help="只跑名字含该子串的项")
    args = ap.parse_args()

    checks = [c for c in CHECKS if args.k in c[0]]
    if args.list:
        for n, _, d, ok in CHECKS:
            print(f"  {n:<16} {d}  (允许退出码 {ok})")
        for n, _, d in LIB_CHECKS:
            print(f"  {n:<16} {d}  (进程内断言)")
        print("\n故意排除:")
        for n, why in EXCLUDED:
            print(f"  {n:<20} {why}")
        return 0

    print("=" * 70)
    print(f"冒烟套件 — {len(checks)} 个入口真跑 + {len(LIB_CHECKS)} 项守卫自检")
    print("=" * 70)
    fails, t0 = [], time.time()
    for name, a, desc, ok_codes in checks:
        print(f"  {name:<16} {desc} ...", end=" ", flush=True)
        ok, msg, el = run_one(name, a, desc, ok_codes)
        print(f"{'✓' if ok else '✗'} {msg}  ({el:.0f}s)", flush=True)
        if not ok:
            fails.append((name, msg))

    for name, fn, desc in LIB_CHECKS:
        if args.k and args.k not in name:
            continue
        print(f"  {name:<16} {desc} ...", end=" ", flush=True)
        try:
            msg = fn()
            print(f"✓ {msg}", flush=True)
        except Exception as e:
            print(f"✗ {type(e).__name__}: {str(e)[:70]}", flush=True)
            fails.append((name, f"{type(e).__name__}: {str(e)[:70]}"))

    print("-" * 70)
    if fails:
        print(f"✗ {len(fails)}/{len(checks) + len(LIB_CHECKS)} 项失败:")
        for n, m in fails:
            print(f"    {n}: {m}")
    else:
        print(f"✓ {len(checks) + len(LIB_CHECKS)} 项全部通过  (耗时 {time.time()-t0:.0f}s)")
    print("注: 本套件只验证「路径能不能跑」。数值是否正确由 reconcile.py 负责。")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
