#!/usr/bin/env python3
"""14:45 截断 IC — 杀/走, 不做收益外推、不重训。

同一份模型分数, 比较:
  close 口径  隔夜 = T+1 开盘 / T 收盘 - 1
  14:45 口径  隔夜 = T+1 开盘 / T 14:45 - 1
若 14:45 口径 IC 明显塌掉, 隔夜 alpha 住在最后 15 分钟, 路线 B 停。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qlib_paths import prediction_pkl

# 2026-09-12: 原先写死 alpha158 那份 pkl, 而生产是 alpha158_ovn,
# 文件不存在 -> 本脚本(路线 B 的杀/走判据)从来跑不起来。
PRED = str(prediction_pkl())
HOLD = 8
TOPK = 100
START = "2024-01-02"
END = "2026-09-09"
N_DAYS = 16


def inst_stem(inst: str) -> str:
    """SH600012 -> sh600012"""
    return inst[:2].lower() + inst[2:]


def inst_ts(inst: str) -> str:
    """SH600012 -> 600012.SH; 北交所 -> .BJ"""
    num, ex = inst[2:], inst[:2].upper()
    if num[:2] in ("43", "82", "83", "87", "92"):
        return f"{num}.BJ"
    return f"{num}.{ex}"


def _bars(minute, day: str):
    """当日 14:45 收盘和 15:00 收盘。没有则 (None, None)。"""
    col = minute["trade_time"].astype(str)
    g = minute.loc[col.str.startswith(day)].copy()
    if g.empty:
        return None, None
    g = g.sort_values("trade_time")
    t = g["trade_time"].astype(str)
    a = g.loc[t.str.endswith("14:45:00"), "close"]
    b = g.loc[t.str.endswith("15:00:00"), "close"]
    px45 = float(a.iloc[-1]) if len(a) else None
    pxcl = float(b.iloc[-1]) if len(b) else None
    return px45, pxcl


def _mu(xs) -> float:
    import numpy as np
    ys = [x for x in xs if x == x]
    if not ys:
        return float("nan")
    return float(np.mean(ys))


def main() -> None:
    import numpy as np
    import pandas as pd
    from data_hub.paths import daily_raw_dir, minute_raw_dir
    from portfolio.rebalance_rules import price_limit

    pred = pd.read_pickle(PRED)
    s = (pred.iloc[:, 0] if pred.ndim > 1 else pred).dropna()
    dates = sorted(
        d for d in s.index.get_level_values(0).unique()
        if START <= str(d)[:10] <= END
    )
    rebal = dates[::HOLD]
    if len(rebal) > N_DAYS:
        idx = np.linspace(0, len(rebal) - 1, N_DAYS).astype(int)
        rebal = [rebal[i] for i in idx]
    nxt = {dates[i]: dates[i + 1] for i in range(len(dates) - 1)}
    print(f"调仓日 {len(rebal)} 个  {str(rebal[0])[:10]} ~ "
          f"{str(rebal[-1])[:10]}")

    mdir = minute_raw_dir()
    ddir = daily_raw_dir()
    have = {p.stem for p in mdir.glob("*.parquet")}
    cache = {}
    ic_c, ic_t, px_corr, last_move = [], [], [], []
    lu_rate, lu_tail = [], []
    n_min = n_miss = 0

    for d in rebal:
        d0 = str(d)[:10]
        d1 = nxt.get(d)
        if d1 is None:
            continue
        d1s = str(d1)[:10]
        day = s.loc[d].sort_values(ascending=False)
        names = list(day.index)
        top = set(names[:TOPK])
        f0 = ddir / f"{d0.replace('-', '')}.parquet"
        f1 = ddir / f"{d1s.replace('-', '')}.parquet"
        if not f0.exists() or not f1.exists():
            continue
        px0 = pd.read_parquet(f0).set_index("ts_code")
        px1 = pd.read_parquet(f1).set_index("ts_code")
        sc, r_c, r_t, t45, tcl = [], [], [], [], []
        lu_n = 0
        tail = []
        for inst in names:
            ts = inst_ts(inst)
            if ts not in px0.index or ts not in px1.index:
                continue
            close_t = float(px0.loc[ts, "close"])
            open_n = float(px1.loc[ts, "open"])
            if close_t <= 0 or open_n <= 0:
                continue
            stem = inst_stem(inst)
            if stem not in have:
                n_miss += 1
                continue
            if stem not in cache:
                p = mdir / f"{stem}.parquet"
                if not p.exists():
                    cache[stem] = None
                else:
                    raw = pd.read_parquet(
                        p, columns=["trade_time", "close"]
                    )
                    tt = raw["trade_time"].astype(str)
                    keep = tt.str.endswith("14:45:00") | tt.str.endswith(
                        "15:00:00"
                    )
                    cache[stem] = raw.loc[keep].copy()
            m = cache[stem]
            a, b = (None, None) if m is None else _bars(m, d0)
            if a is None or b is None or a <= 0 or b <= 0:
                n_miss += 1
                continue
            n_min += 1
            sc.append(float(day[inst]))
            r_c.append(open_n / close_t - 1)
            r_t.append(open_n / a - 1)
            t45.append(a)
            tcl.append(b)
            if inst not in top:
                continue
            prev = float(px1.loc[ts, "pre_close"])
            hi = float(px1.loc[ts, "high"])
            lim = price_limit(inst)
            if prev > 0 and hi / prev - 1 >= lim - 0.005:
                lu_n += 1
                tail.append(b / a - 1)
        if len(sc) < 50:
            continue
        ic_c.append(pd.Series(sc).corr(pd.Series(r_c), method="spearman"))
        ic_t.append(pd.Series(sc).corr(pd.Series(r_t), method="spearman"))
        px_corr.append(
            pd.Series(t45).corr(pd.Series(tcl), method="spearman")
        )
        last_move.append(float(np.mean(np.abs(
            np.array(tcl) / np.array(t45) - 1
        ))))
        lu_rate.append(lu_n / TOPK)
        if tail:
            lu_tail.append(float(np.mean(tail)))
        print(f"  {d0} n={len(sc)} ic_c={ic_c[-1]:.3f} "
              f"ic_t={ic_t[-1]:.3f}", flush=True)

    print(f"有分钟样本 {n_min}  缺分钟 {n_miss}")
    print(f"14:45 vs 15:00 价格秩相关  {_mu(px_corr):.4f}")
    print(f"|最后15分钟涨跌|均值        {_mu(last_move):.4%}")
    print(f"IC 隔夜(从收盘)             {_mu(ic_c):.4f}")
    print(f"IC 隔夜(从14:45)            {_mu(ic_t):.4f}")
    print(f"Top{TOPK} 次日涨停率        {_mu(lu_rate):.2%}")
    print(f"涨停票当日最后15min涨跌     {_mu(lu_tail):.4%}")
    gap = _mu(ic_t) - _mu(ic_c)
    print(f"IC(14:45成交) - IC(收盘成交) {gap:+.4f}")
    print("说明: 路线B是14:45出信号、14:50收盘成交, 前向收益仍是隔夜.")
    print("     价格秩相关~1 且涨停票尾盘无异动 → 最后15分钟不是涨停识别源.")
    print("     IC变差只说明不能在14:45成交; 截断特征是否还能出分另测.")


if __name__ == "__main__":
    main()
