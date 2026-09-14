#!/usr/bin/env python3
"""TopK 扫描 —— 严格按真实执行口径

与之前那个 _naive_csi2000.py 的关键区别: **没有前视**。

2026-09-10 教训: 上一版重建把收益记成 open(T)->open(T+1) / close(T)->close(T+1)，
而信号是 T 日**收盘**才产生的 —— 等于用当天收盘才知道的信息去买当天早盘，
凭空多出一天。那版跑出 +254%(收盘) / +928%(开盘)，修正后变成 +26% / -31%。
两次都是"验证脚本比被验证的系统更宽松"，差点据此认定 paper_trader 有 bug。

本脚本的口径:
    T 日收盘出信号 -> T+1 日买入 -> 持有到下一调仓日的 T'+1 卖出
    买入日涨停(按板块阈值)的票买不进，直接剔除
    整手约束: 每仓资金买不起 100 股则跳过

用法: python -m factor_lab._topk_tradable
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

HOLD = 8
CAPITAL = 100_000
LOT = 100
KS = [16, 30, 50, 70, 100, 150, 200]
from qlib_paths import prediction_pkl

# 2026-09-12: 原先写死的两份 pkl 都不存在 (生产是 alpha158_ovn)。
PRED = str(prediction_pkl())
FALLBACK = PRED


def main():
    import numpy as np
    import pandas as pd
    import qlib
    from qlib.constant import REG_CN
    from qlib_paths import qlib_provider_uri
    from portfolio.rebalance_rules import price_limit, allocate_buys

    qlib.init(provider_uri=qlib_provider_uri(), region=REG_CN, kernels=2)
    from qlib.data import D

    path = PRED if Path(PRED).exists() else FALLBACK
    print(f"  预测文件: {Path(path).name}")
    s = pd.read_pickle(path)
    s = (s.iloc[:, 0] if s.ndim > 1 else s).dropna()

    dates = sorted(set(s.index.get_level_values(0)))
    rebal = dates[::HOLD]

    # 只加载会被选中的股票，省内存(全市场 2934 只会把 16GB 打满)
    picks = set()
    for t in rebal:
        picks |= set(s.loc[t].nlargest(max(KS)).index)
    picks = sorted(picks)
    print(f"  候选股票 {len(picks)} 只，调仓 {len(rebal)} 次")

    f = D.features(picks, ["$open", "$close", "$factor"],
                   start_time=str(dates[0].date()),
                   end_time=str((dates[-1] + pd.Timedelta(days=30)).date()))
    if f.index.names[0] == "instrument":
        f = f.swaplevel().sort_index()
    cl = f["$close"].unstack(level=1)
    fac = f["$factor"].unstack(level=1)
    prev = cl.shift(1)
    cret = cl / prev - 1                       # 当日涨幅，判涨停

    print(f"\n  {'TopK':>6}{'总收益':>10}{'年化':>9}{'Sharpe':>9}"
          f"{'最大回撤':>10}{'平均持仓':>10}{'涨停剔除':>10}{'买不起':>9}")

    for k in KS:
        segs, held, blocked, unaff = [], [], 0, 0
        cand_total = 0
        for i, t in enumerate(rebal):
            j = dates.index(t)
            if j + 1 >= len(dates):
                break
            buy = dates[j + 1]
            nt = rebal[i + 1] if i + 1 < len(rebal) else None
            sell = (dates[-1] if nt is None else
                    dates[min(dates.index(nt) + 1, len(dates) - 1)])
            if sell <= buy or buy not in cl.index:
                continue

            cols = [c for c in s.loc[t].nlargest(k).index if c in cl.columns]
            cand_total += len(cols)

            # 涨停买不进
            ok = []
            for c in cols:
                u = cret.loc[buy, c] if buy in cret.index else np.nan
                if pd.notna(u) and u > price_limit(c):
                    blocked += 1
                    continue
                ok.append(c)

            # 资金分配走实盘同一函数: 买不起的剔除后，预算**重分**给买得起的。
            # 上一版是"买不起就跳过、钱闲置"，于是 TopK=200 名义 200 只、
            # 实际只建 30 只仓，剩下的钱空转 —— 那个 +64.1% 不能当成
            # "TopK=200 可行"的证据。
            raw_px = {}
            for c in ok:
                p, fa = cl.loc[buy, c], fac.loc[buy, c]
                if pd.isna(p) or pd.isna(fa) or fa <= 0:
                    continue
                raw_px[c] = float(p / fa)       # 前复权价 -> 原始价(实际报价)
            if not raw_px:
                continue
            alloc = allocate_buys(list(raw_px), raw_px, CAPITAL)
            unaff += len(raw_px) - len(alloc)
            if not alloc:
                continue
            held.append(len(alloc))

            # 按实际投入金额加权(不是等权) —— 整手导致各仓金额本就不等
            tot_amt = sum(a['amount'] for a in alloc.values())
            acc = 0.0
            for c, a in alloc.items():
                r = cl.loc[sell, c] / cl.loc[buy, c] - 1
                if pd.notna(r):
                    acc += r * a['amount'] / tot_amt
            segs.append(acc)

        if not segs:
            print(f"  {k:>6}   (无有效持仓段)")
            continue
        eq = np.cumprod([1 + x for x in segs])
        tot = eq[-1] - 1
        yrs = (dates[-1] - dates[0]).days / 365.25
        ann = (1 + tot) ** (1 / yrs) - 1
        vol = np.std(segs, ddof=1) * np.sqrt(242 / HOLD)
        mdd = float((eq / np.maximum.accumulate(eq) - 1).min())
        print(f"  {k:>6}{tot:>10.1%}{ann:>9.1%}{ann/vol if vol else 0:>9.3f}"
              f"{mdd:>10.1%}{np.mean(held):>10.1f}"
              f"{blocked/max(cand_total,1):>10.1%}{unaff/max(cand_total,1):>9.1%}")


if __name__ == "__main__":
    main()
