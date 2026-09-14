#!/usr/bin/env python3
"""独立重建 TopK 组合 —— 判定 paper_trader 有没有 bug

矛盾: 中证2000 的预测 IC 强正(日均 +0.0575, ICIR 0.429, 70% 交易日为正)、
价格与 Tushare 权威 pct_chg 逐行一致、交易成本仅占本金 4.6%，
但 paper_trader 回放出来是 -28.28%，而中证2000 指数同期 +32.47%。

这里用纯 pandas 重建: 每 8 个交易日按预测分取 TopK 等权、持满 8 日、
无成本、无止损、无敞口缩放、无 n_drop 换手限制。

  - 若重建为正 -> paper_trader 里有 bug
  - 若重建也是负 -> IC 虽正但不可变现(集中在流动性差的极端票等)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

HOLD = 8
KS = [16, 30, 50, 100]



def _pred_pkl():
    """2026-09-12: 原先写死 alpha158 那份, 生产是 alpha158_ovn, 文件不存在。"""
    from qlib_paths import prediction_pkl
    return str(prediction_pkl())

def main():
    import numpy as np
    import pandas as pd
    import qlib
    from qlib.constant import REG_CN
    from qlib_paths import qlib_provider_uri

    qlib.init(provider_uri=qlib_provider_uri(), region=REG_CN)
    from qlib.data import D

    s = pd.read_pickle(_pred_pkl())
    s = (s.iloc[:, 0] if s.ndim > 1 else s).dropna()

    d0 = s.index.get_level_values(0).min()
    d1 = s.index.get_level_values(0).max()
    px = D.features(D.instruments("all"), ["$close"],
                    start_time=d0, end_time=d1 + pd.Timedelta(days=20))["$close"]
    if px.index.names[0] == "instrument":
        px = px.swaplevel().sort_index()
    wide = px.unstack(level=1).sort_index()
    # fill_method=None: 默认的前向填充会把停牌/退市的缺口补成假收益,
    # 一只票停牌半年后复牌, padding 会凭空造出一段平滑走势 —— 结论会虚高
    rets = wide.pct_change(fill_method=None).shift(-1)  # T+1 收盘买入

    dates = sorted(set(s.index.get_level_values(0)) & set(wide.index))
    rebal = dates[::HOLD]
    print(f"  区间 {dates[0].date()} ~ {dates[-1].date()}，{len(rebal)} 次调仓")
    print(f"  {'TopK/nd':>8}{'年化':>9}{'年化波动':>10}{'Sharpe':>9}"
          f"{'总收益':>10}{'最大回撤':>10}{'平均持仓':>10}")

    for k, nd in [(16, None), (16, 4), (16, 8), (30, None), (30, 8), (50, None)]:
        daily = pd.Series(0.0, index=pd.Index(dates, name="datetime"))
        held = []
        cur: list = []
        for i, t in enumerate(rebal):
            nxt = rebal[i + 1] if i + 1 < len(rebal) else None
            seg = (dates[dates.index(t):dates.index(nxt)] if nxt
                   else dates[dates.index(t):])
            top = [c for c in s.loc[t].nlargest(k).index if c in rets.columns]
            if nd is None:
                cur = top                      # 全量换手: 直接持有当期 TopK
            else:
                # n_drop 换手限制: 每次最多换 nd 只
                out = [c for c in cur if c not in top][:nd]
                cur = [c for c in cur if c not in out]
                room = min(k - len(cur), nd)
                cur += [c for c in top if c not in cur][:max(room, 0)]
            cols = cur
            held.append(len(cols))
            if cols and seg:
                daily.loc[seg] = rets.loc[seg, cols].mean(axis=1).fillna(0).values
        dr = daily.values
        tot = float(np.prod(1 + dr) - 1)
        vol = dr.std(ddof=1) * np.sqrt(242)
        ann = (1 + tot) ** (242 / len(dr)) - 1
        eq = np.cumprod(1 + dr)
        mdd = float((eq / np.maximum.accumulate(eq) - 1).min())
        print(f"  {str(k)+"/"+str(nd):>8}{ann:>9.2%}{vol:>10.2%}{ann/vol if vol else 0:>9.3f}"
              f"{tot:>10.2%}{mdd:>10.2%}{np.mean(held):>10.1f}")
        if nd is None and k == 16:
            yr = daily.groupby(daily.index.year).apply(lambda x: (1+x).prod()-1)
            print("           逐年:", {int(a): f"{b:.1%}" for a, b in yr.items()})

    # 全池等权基准: 模型无技能时该拿到的收益
    eqw = rets.reindex(columns=wide.columns).loc[dates].mean(axis=1).fillna(0)
    tot = float((1 + eqw).prod() - 1)
    print(f"\n  全市场等权基准: 总收益 {tot:>7.2%}")


if __name__ == "__main__":
    main()
