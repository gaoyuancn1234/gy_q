#!/usr/bin/env python3
"""中证2000 全量 15 分钟数据下载

目的
----
现在的信号有整整一天延迟: T 日收盘算信号 -> T+1 买入。而实测模型的预测力
几乎全在**隔夜跳空**上 (IC +0.1654, 96% 交易日为正)，而跳空发生在
T 收盘到 T+1 开盘之间 —— T+1 才买，等于全错过。

有分钟数据后可以改成: T 日 14:45 算信号 -> 14:50 下单 -> 收盘竞价成交，
买在跳空之前。日线上做的上限估算是 TopK50 从 +50.5% 提到 +84.1%。

已验证的前提 (2026-09-10, 6 个调仓日 × 200 只抽样):
  - 尾盘 14:45-15:00 成交额中位 637 万元，单笔买 5000 元不超尾盘量 10%
    的可行比例 99.7% —— 流动性不是约束
  - 14:45 价格与收盘价差: 均值 +0.112%, 标准差 0.414% —— 截断代价很小

接口约束 (实测)
--------------
  - stk_mins **必须逐只查**，不支持按交易日批量取全市场
  - 单次返回上限 **8000 行** (约 2 年 15min 数据)，超出只返回最近的部分。
    所以按 2 年分段, 每只 4 次调用覆盖 2019-10 ~ 2026-09。

断点续传
--------
全量约 8.6 小时。每只股票单独落盘 parquet, 重跑时已存在的直接跳过 ——
2026-09-09 日线下载就是跑到最后一步才因成功率不足中止, 3 小时全废。

用法
    python -m qlib_engine.fetch_minute                 # 全量(断点续传)
    python -m qlib_engine.fetch_minute --limit 20      # 冒烟
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PROJECT_DIR = Path(__file__).resolve().parent.parent
CACHE = PROJECT_DIR / "factor_lab" / "results" / ".cache" / "minute_csi2000"

SLEEP = 0.85
FREQ = "15min"
# 单次上限 8000 行, 15min 每天 16 根 -> 500 天 ≈ 2 年。留余量按 20 个月分段。
CHUNKS = [
    ("2019-10-01", "2021-05-31"),
    ("2021-06-01", "2023-01-31"),
    ("2023-02-01", "2024-09-30"),
    ("2024-10-01", "2026-09-30"),
]


def _pro():
    import tushare as ts
    from dotenv import load_dotenv
    load_dotenv(PROJECT_DIR / ".env", override=True)
    token = os.environ.get("TUSHARE_TOKEN", "").strip()
    if not token:
        raise SystemExit("TUSHARE_TOKEN 未配置")
    pro = ts.pro_api(token)
    url = os.environ.get("TUSHARE_HTTP_URL", "").strip()
    if url:
        pro._DataApi__http_url = url
    return pro


def _call(fn, _tries: int = 6, **kw):
    """限流/上游异常按提示退避重试

    限流是"要求等待"，不是"这只没有数据" —— 直接跳过会一路撞限流撞到底
    (2026-09-09 日线下载踩过, 连报上百次后成功率不足而中止)。
    """
    for k in range(1, _tries + 1):
        try:
            return fn(**kw)
        except Exception as e:
            msg = str(e)
            m = re.search(r"请\s*(\d+)\s*秒后(?:再试|重试)", msg)
            if m:
                wait = int(m.group(1)) + 3
            elif "502" in msg or "timeout" in msg.lower() or "超时" in msg:
                wait = 20 * k
            else:
                raise
            if k == _tries:
                raise
            print(f"    限流/上游异常，等待 {wait}s 重试 ({k}/{_tries})", flush=True)
            time.sleep(wait)


def _to_ts(code: str) -> str:
    """sh.600000 -> 600000.SH; 北交所 43/82/83/87/92 -> .BJ"""
    if "." in code:
        ex, num = code.split(".")
    else:
        ex, num = code[:2], code[2:]
    if num[:2] in ("43", "82", "83", "87", "92"):
        return f"{num}.BJ"
    return f"{num}.{ex.upper()}"


def fetch_today_minutes(day: str | None = None, limit: int = 0) -> int:
    """只拉某一日分钟线, 写入 minute_raw_dir (与截断聚合同一目录)。"""
    from datetime import date as _date
    from data_hub.paths import minute_raw_dir

    day = day or _date.today().strftime("%Y-%m-%d")
    cache = minute_raw_dir()
    cache.mkdir(parents=True, exist_ok=True)
    mem = json.loads(
        (PROJECT_DIR / "data" / "csi2000_membership.json")
        .read_text(encoding="utf-8")
    )
    latest = max(mem)
    codes = sorted(set(mem[latest]))
    if limit:
        codes = codes[:limit]
    pro = _pro()
    t0 = time.time()
    ok = fail = 0
    start = f"{day} 09:30:00"
    end = f"{day} 15:00:00"
    sleep_s = 0.2
    print(
        f"[minute-today] {day} 成分 {len(codes)} 只 -> {cache}",
        flush=True,
    )
    for i, code in enumerate(codes, 1):
        ts_code = _to_ts(code)
        try:
            d = _call(
                pro.stk_mins,
                ts_code=ts_code,
                freq=FREQ,
                start_date=start,
                end_date=end,
            )
        except Exception as e:
            print(
                f"  {ts_code} 失败: {type(e).__name__}: {str(e)[:60]}",
                flush=True,
            )
            fail += 1
            time.sleep(sleep_s)
            continue
        time.sleep(sleep_s)
        if d is None or len(d) == 0:
            fail += 1
            continue
        out = cache / f"{code.replace('.', '')}.parquet"
        if out.exists():
            old = pd.read_parquet(out)
            d = pd.concat(
                [old, d], ignore_index=True,
            ).drop_duplicates("trade_time")
        d.to_parquet(out, index=False)
        ok += 1
        if i % 20 == 0 or i == len(codes):
            el = time.time() - t0
            eta = el / i * (len(codes) - i) if i else 0
            print(
                f"[minute-today] [{i}/{len(codes)}] 成功 {ok} "
                f"空/失败 {fail} {el/60:.1f}m ETA {eta/60:.1f}m",
                flush=True,
            )
    print(
        f"[minute-today] 完成 成功 {ok} 空/失败 {fail}",
        flush=True,
    )
    return ok


def main():
    ap = argparse.ArgumentParser(description="中证2000 15分钟数据下载")
    ap.add_argument("--limit", type=int, default=0, help="只下前 N 只(冒烟)")
    ap.add_argument("--sample", type=int, default=0,
                    help="随机抽 N 只(固定种子)。--limit 取代码序前 N 个, "
                         "全是 sh600xxx 主板老票, 做 IC 门禁会偏离薄票")
    ap.add_argument(
        "--today",
        action="store_true",
        help="只拉某一日 15 分钟, 合并进已有 parquet",
    )
    ap.add_argument("--day", default="", help="配合 --today, 默认今天")
    args = ap.parse_args()

    if args.today:
        fetch_today_minutes(day=args.day or None, limit=args.limit)
        return

    mem = json.loads((PROJECT_DIR / "data" / "csi2000_membership.json")
                     .read_text(encoding="utf-8"))
    codes = sorted({c for v in mem.values() for c in v})
    if args.sample:
        import random
        random.Random(20260913).shuffle(codes)
        codes = sorted(codes[:args.sample])
    elif args.limit:
        codes = codes[:args.limit]
    CACHE.mkdir(parents=True, exist_ok=True)

    done = {p.stem for p in CACHE.glob("*.parquet")}
    todo = [c for c in codes if c.replace(".", "") not in done]
    print(f"[minute] 目标 {len(codes)} 只，已缓存 {len(codes)-len(todo)} 只，待下 {len(todo)} 只")
    if not todo:
        print("[minute] 全部已缓存")
        return

    pro = _pro()
    t0 = time.time()
    ok = fail = 0
    for i, code in enumerate(todo, 1):
        ts_code = _to_ts(code)
        frames = []
        bad = False
        for a, b in CHUNKS:
            try:
                d = _call(pro.stk_mins, ts_code=ts_code, freq=FREQ,
                          start_date=f"{a} 09:30:00", end_date=f"{b} 15:00:00")
                if len(d):
                    frames.append(d)
            except Exception as e:
                print(f"  {ts_code} {a} 失败: {type(e).__name__}: {str(e)[:60]}",
                      flush=True)
                bad = True
            time.sleep(SLEEP)
        if frames and not bad:
            df = pd.concat(
                frames, ignore_index=True
            ).drop_duplicates("trade_time")
            out = CACHE / f"{code.replace('.', '')}.parquet"
            df.to_parquet(out, index=False)
            ok += 1
        else:
            fail += 1
            if frames and bad:
                print(
                    f"  {ts_code} 分段失败, 不落盘以便续传",
                    flush=True,
                )
        if i % 10 == 0 or i == len(todo):
            el = time.time() - t0
            print(f"[minute] [{i}/{len(todo)}] {ts_code} 成功 {ok} 失败 {fail} "
                  f"已用 {el/3600:.2f}h ETA {el/i*(len(todo)-i)/3600:.2f}h", flush=True)

    print(f"[minute] 完成: 成功 {ok} 失败 {fail}，耗时 {(time.time()-t0)/3600:.2f}h")
    print(f"[minute] 缓存目录 {CACHE}")
    # 不中止、不回滚 —— 2026-09-09 那次"跑到最后因成功率不足中止"白费 3 小时,
    # 教训是**已下好的必须留在盘上**。但退出码要如实反映, 否则管道里
    # 一看退出码 0 就当成功了 (失败的下次重跑会自动续传)。
    if ok + fail and fail > (ok + fail) * 0.1:
        print(f"[minute] 失败率 {fail/(ok+fail):.0%} 偏高, 重跑本命令续传剩余",
              flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
