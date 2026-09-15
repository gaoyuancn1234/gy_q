#!/usr/bin/env python3
"""Qlib 数据下载 — Tushare 源 (支持中证2000 等宽基股票池)

为什么新加这一路
----------------
2026-09-09: 要把策略切到中证2000，卡在两件事上，新浪/baostock 都解决不了:

  1. **时点成分股拿不到**。baostock 只封装了 hs300/zz500/sz50 三个指数;
     akshare 的几个成分股函数(index_stock_cons / _csindex / _weight_csindex)
     实测全部只返回"最新"快照，源码里连日期参数都没有。而直接拿今天的成分股
     铺满历史，正是 CLAUDE.md 记载过的幸存者偏差 —— CSI300 当年这么算，
     虚高的 Sharpe 2.055 到修正后跌回 1.643。
  2. **逐只下载太慢**。新浪源 549 只要 45 分钟，中证2000 的历史并集
     2500+ 只，按同样速度要 3~4 小时。

Tushare 两条都能解:
  - index_weight 给月度时点成分股(中证2000 从 2024 年起数据齐全)
  - daily / adj_factor 支持**按交易日批量取全市场**(单日 5549 只仅 1.9 秒)，
    652 个交易日约 18 分钟拉完，且拿到全市场后任何股票池都能切

复权口径
--------
Tushare daily 返回**未复权**价，adj_factor 单独给复权因子。本项目的约定
(见 CLAUDE.md)是: 训练用前复权价，`$factor` 存复权因子供回测还原整手约束。
这里价格字段写前复权值，factor 写 adj_factor/最新adj_factor。

用法
    python -m qlib_engine.data_setup_tushare --universe csi2000
    python -m qlib_engine.data_setup_tushare --universe csi2000 \
        --limit-days 20 --target_dir ~/.qlib/qlib_data/_smoke   # 冒烟
"""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from .data_setup import (
    snapshots_to_intervals, _bao_to_qlib_instrument,
    _build_calendar, _build_instruments, _build_features,
)

PROJECT_DIR = Path(__file__).resolve().parent.parent

# 每次请求之间的休眠。接口方说明: 每分钟请求次数有上限，建议间隔 0.85 秒。
SLEEP = 0.85

UNIVERSE_INDEX = {
    "csi2000": "932000.CSI",     # 中证2000 — 小微盘
    "csi1000": "000852.SH",      # 中证1000 — 小盘
    "csi500": "000905.SH",
    "csi300": "000300.SH",
}


def _pro():
    """构造 Tushare 客户端。token 与代理地址都从 .env 读。"""
    import tushare as ts
    from dotenv import load_dotenv
    load_dotenv(PROJECT_DIR / ".env", override=True)
    token = os.environ.get("TUSHARE_TOKEN", "").strip()
    if not token:
        raise SystemExit("TUSHARE_TOKEN 未配置，见 .env")
    pro = ts.pro_api(token)
    # 走第三方代理时需要覆盖 http_url; 官方源留空即可
    url = os.environ.get("TUSHARE_HTTP_URL", "").strip()
    if url:
        pro._DataApi__http_url = url
    return pro


def _ts_to_bao(ts_code: str) -> str:
    """000001.SZ -> sz.000001 (与既有成分股缓存/instruments 命名一致)"""
    num, ex = ts_code.split(".")
    return f"{'sh' if ex == 'SH' else 'sz'}.{num}"


def fetch_membership(pro, universe: str, start: str, end: str) -> dict:
    """月度时点成分股快照 {YYYY-MM-DD: [sh.600000, ...]}"""
    idx = UNIVERSE_INDEX[universe]
    frames = []
    # index_weight 单次返回行数有上限，按半年分段拉
    for y in range(int(start[:4]), int(end[:4]) + 1):
        for a, b in ((f"{y}0101", f"{y}0630"), (f"{y}0701", f"{y}1231")):
            if b < start.replace("-", "") or a > end.replace("-", ""):
                continue
            w = _call(pro.index_weight, index_code=idx, start_date=a, end_date=b)
            if len(w):
                frames.append(w)
            time.sleep(SLEEP)
    if not frames:
        raise SystemExit(f"{universe}({idx}) 没有取到任何成分股数据")
    df = pd.concat(frames, ignore_index=True)
    snaps = {}
    for d, g in df.groupby("trade_date"):
        key = pd.Timestamp(d).strftime("%Y-%m-%d")
        snaps[key] = sorted(_ts_to_bao(c) for c in g["con_code"].unique())
    sizes = [len(v) for v in snaps.values()]
    print(f"[tushare] 成分股快照 {len(snaps)} 个时点 "
          f"({min(snaps)} ~ {max(snaps)})，每期 {min(sizes)}~{max(sizes)} 只")
    return snaps


def _call(fn, _tries: int = 6, **kw):
    """调 Tushare 接口，遇限流按提示退避重试

    2026-09-09: 首版只是 print 一句就跳到下一天，于是一路撞限流撞到底 ——
    "Token 请求频繁，请 99 秒后重试" 连报上百次，最终成功率 1864/2955
    触发中止，3 小时下载全废。限流是**要求等待**，不是"这天没有数据"，
    必须等完重试同一天。
    """
    import re
    for k in range(1, _tries + 1):
        try:
            return fn(**kw)
        except Exception as e:
            msg = str(e)
            m = re.search(r"请\s*(\d+)\s*秒后重试", msg)
            if m:
                wait = int(m.group(1)) + 3
            elif "502" in msg or "超时" in msg or "timeout" in msg.lower():
                wait = 20 * k
            else:
                raise
            if k == _tries:
                raise
            print(f"[tushare] 限流/上游异常，等待 {wait}s 后重试 ({k}/{_tries})",
                  flush=True)
            time.sleep(wait)
    raise RuntimeError("unreachable")


def fetch_daily(pro, dates: list, wanted: set, cache_dir: Path | None = None) -> dict:
    """按交易日批量拉全市场，切出 wanted 里的股票

    cache_dir: 每个交易日的结果单独落盘。整段下载要 2~3 小时，中途被限流
    或断网时不必从头再来 —— 2026-09-09 首次全量跑到最后一步才因成功率不足
    中止，等于白跑 3 小时。

    返回 {qlib_instrument: DataFrame(date + 价量字段 + factor)}
    """
    failed: list[str] = []
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    t0 = time.time()
    for i, d in enumerate(dates, 1):
        ymd = d.replace("-", "")
        cached = cache_dir / f"{ymd}.parquet" if cache_dir else None
        if cached is not None and cached.exists():
            rows.append(pd.read_parquet(cached))
            continue
        try:
            px = _call(pro.daily, trade_date=ymd)
            adj = _call(pro.adj_factor, trade_date=ymd)
        except Exception as e:
            print(f"[tushare] {d} 拉取失败: {type(e).__name__}: {e}", flush=True)
            failed.append(d)
            continue
        if not len(px):
            continue
        m = px.merge(adj[["ts_code", "adj_factor"]], on="ts_code", how="left")
        m = m[m["ts_code"].map(lambda c: _ts_to_bao(c) in wanted)]
        if len(m):
            m = m.assign(date=d)
            rows.append(m)
            if cached is not None:
                m.to_parquet(cached, index=False)
        if i % 20 == 0 or i == len(dates):
            el = time.time() - t0
            print(f"[tushare] 下载中 [{i}/{len(dates)}] {d} "
                  f"已用 {el/60:.1f}min ETA {el/i*(len(dates)-i)/60:.1f}min "
                  f"失败 {len(failed)}", flush=True)

    if failed:
        print(f"[tushare] ⚠ {len(failed)} 个交易日最终失败: {failed[:5]}"
              f"{' ...' if len(failed) > 5 else ''}")

    if not rows:
        raise SystemExit("没有取到任何行情数据")
    df = pd.concat(rows, ignore_index=True)

    # 复权: Tushare daily 是未复权价，前复权 = 原价 * adj_factor / 最新adj_factor
    df["adj_factor"] = df["adj_factor"].fillna(1.0)
    latest = df.sort_values("date").groupby("ts_code")["adj_factor"].last()
    df["_ratio"] = df["adj_factor"] / df["ts_code"].map(latest)

    out = {}
    for code, g in df.groupby("ts_code"):
        inst = _bao_to_qlib_instrument(_ts_to_bao(code))
        r = g["_ratio"].values
        o = pd.DataFrame({
            "date": pd.to_datetime(g["date"].values),
            "open": g["open"].values * r,
            "close": g["close"].values * r,
            "high": g["high"].values * r,
            "low": g["low"].values * r,
            "volume": g["vol"].values * 100.0,       # Tushare 单位: 手
            "amount": g["amount"].values * 1000.0,   # Tushare 单位: 千元
            "turn": np.nan,
            "pctChg": g["pct_chg"].values,
            "isST": np.nan,                           # daily 接口无 ST 标记
            "factor": r,
            # vwap = 成交额/成交量，同样按前复权比例还原。
            #
            # 2026-09-09: 必须写这个字段。qlib 原生 Alpha158 handler(preset
            # "alpha158" 走的就是它)有多个因子引用 $vwap，字段缺失时 qlib
            # 不报错、返回**长度为 0 的序列**，到二元运算才炸出
            # "operands could not be broadcast together with shapes (0,) (1680,)"
            # —— 又一次"qlib 对缺失字段不报错"的沉默失败。
            # 项目此前一直用 alpha158_val(走自定义表达式，不引用 vwap)，
            # 所以旧数据集没有这个字段也没出过问题。
            "vwap": np.where(g["vol"].values > 0,
                             g["amount"].values * 1000.0
                             / np.maximum(g["vol"].values * 100.0, 1e-9) * r,
                             np.nan),
        }).sort_values("date").reset_index(drop=True)
        o = o[(o["volume"] > 0) & (o["close"] > 0)]
        if len(o):
            out[inst] = o
    return out


def main():
    ap = argparse.ArgumentParser(description="Qlib 数据下载 (Tushare 源)")
    ap.add_argument("--universe", default="csi2000", choices=list(UNIVERSE_INDEX))
    ap.add_argument("--start_date", default="2024-01-01")
    ap.add_argument("--end_date", default=None)
    ap.add_argument("--target_dir", default=None)
    ap.add_argument("--limit-days", type=int, default=0,
                    help="只拉最近 N 个交易日(冒烟测试，须配 --target_dir)")
    args = ap.parse_args()

    end = args.end_date or pd.Timestamp.today().strftime("%Y-%m-%d")
    default_target = Path(f"~/.qlib/qlib_data/cn_data_{args.universe}").expanduser()
    target = Path(args.target_dir).expanduser() if args.target_dir else default_target
    if args.limit_days and target == default_target:
        raise SystemExit("--limit-days 会写出残缺数据集，请另指定 --target_dir")

    t_all = time.time()
    pro = _pro()
    print(f"[tushare] 目标 {target}")
    print(f"[tushare] 区间 {args.start_date} ~ {end} | 股票池 {args.universe}")

    snaps = fetch_membership(pro, args.universe, args.start_date, end)

    # 训练窗口要 4 年历史(D_expand_3v_3r)，但中证2000 是 2023-08 才发布的指数，
    # index_weight 最早只到 2024-02。缺口段用**最早那期快照**回填。
    #
    # 这是一个已知妥协，必须说清楚: 回填段带幸存者偏差(用 2024-02 的成分股
    # 去代表 2019~2023 的股票池，期间已退市/被调出的票不在其中)。
    #
    # 为什么可以接受: 偏差只落在**训练期**，影响模型学到什么;
    # 而回测/绩效评估期(2024-01 起)用的是真实时点成分股，报出来的收益
    # 不会因此虚高 —— 后者才是 CLAUDE.md 记载 CSI300 那次
    # "Sharpe 2.055 -> 1.643" 的病根。
    #
    # 想彻底消除: 用 daily_basic 的历史市值按中证2000 的选样规则重建时点池。
    # 2026-09-09 当天该接口在代理侧整体 502，未能实施。
    first = min(snaps)
    if pd.Timestamp(args.start_date) < pd.Timestamp(first):
        snaps[args.start_date] = list(snaps[first])
        print(f"[tushare] ⚠ 成分股最早只到 {first}，{args.start_date} 起用该期回填 "
              f"—— 训练期带幸存者偏差，回测期({first} 后)不受影响")

    # snapshots_to_intervals 要 set (内部做 |= 求并集); 落盘时再转回 list
    stocks, intervals = snapshots_to_intervals({k: set(v) for k, v in snaps.items()})
    wanted = set(stocks)
    print(f"[tushare] 历史并集 {len(wanted)} 只")

    cal = _call(pro.trade_cal, exchange="SSE",
                start_date=args.start_date.replace("-", ""),
                end_date=end.replace("-", ""), is_open="1")
    time.sleep(SLEEP)
    dates = sorted(pd.Timestamp(d).strftime("%Y-%m-%d") for d in cal["cal_date"])
    if args.limit_days:
        dates = dates[-args.limit_days:]
    print(f"[tushare] 交易日 {len(dates)} 个")

    cache = PROJECT_DIR / 'factor_lab' / 'results' / '.cache' / f'tushare_{args.universe}'
    stock_map = fetch_daily(pro, dates, wanted, cache_dir=cache)
    print(f"[tushare] 成功 {len(stock_map)}/{len(wanted)} 只")
    if len(stock_map) < len(wanted) * 0.9:
        raise SystemExit(
            f"成功率过低 ({len(stock_map)}/{len(wanted)})，已中止，未改动 {target}")

    all_dates = [d for df in stock_map.values() for d in df["date"]]

    # 建到旁路目录再原子替换 —— 重建期间目录为空会让读数据的任务拿到全 NaN
    build = target.with_name(target.name + ".building")
    if build.exists():
        shutil.rmtree(build)
    build.mkdir(parents=True)
    calendar = _build_calendar(all_dates, build)
    _build_instruments(stock_map, build, intervals, args.universe)
    _build_features(stock_map, calendar, build)

    old = target.with_name(target.name + ".old")
    if old.exists():
        shutil.rmtree(old, ignore_errors=True)
    if target.exists():
        os.replace(target, old)
    os.replace(build, target)
    shutil.rmtree(old, ignore_errors=True)

    # 成分股缓存落盘，与既有 csi300_membership.json 同格式
    mem = PROJECT_DIR / "data" / f"{args.universe}_membership.json"
    mem.write_text(json.dumps(snaps, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[tushare] 成分股缓存 -> {mem}")
    print(f"[tushare] 完成! 总耗时 {(time.time()-t_all)/60:.1f}min")
    print("[tushare] ⚠ isST/turn 为 NaN (daily 接口不提供) —— ST 过滤在本数据集不生效")


if __name__ == "__main__":
    main()
