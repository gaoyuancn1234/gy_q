#!/usr/bin/env python3
"""路线B验证 —— 分钟数据能否把延迟消掉

要回答两个问题, 在投入 18 小时全量下载之前:

  1. **特征截断代价**: 现在信号用 T 日完整收盘价算。改成 14:45 截断后
     (14:50 下单需要), IC 掉多少? 这是最大的未知数。
  2. **尾盘流动性**: 14:45-15:00 这 15 分钟, 小盘股的成交额够不够
     一次买 30~50 只、每只一两千元?

做法: 抽若干调仓日, 只拉当天 Top200 候选的 15min 数据 (约 1.2 小时全量,
这里先抽样几天做快速验证)。

2026-09-10 教训在先: 本轮已两次被前视造出的漂亮数字带偏(+254% / +928%)。
所以这里**只测可直接观测的量**(IC 变化、成交额), 不做收益外推。
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

N_DAYS = 6          # 抽样调仓日数
TOPN = 200          # 每天取前 N 名候选
FREQ = "15min"
SLEEP = 0.85



def _pred_pkl():
    """2026-09-12: 原先写死 alpha158 那份, 生产是 alpha158_ovn, 文件不存在。"""
    from qlib_paths import prediction_pkl
    return str(prediction_pkl())

def _pro():
    import tushare as ts
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)
    pro = ts.pro_api(os.environ["TUSHARE_TOKEN"])
    url = os.environ.get("TUSHARE_HTTP_URL", "").strip()
    if url:
        pro._DataApi__http_url = url
    return pro


def _to_ts(code: str) -> str:
    """SH600000 -> 600000.SH"""
    return f"{code[2:]}.{code[:2].upper()}"


def main():
    import numpy as np
    import pandas as pd

    pred = pd.read_pickle(_pred_pkl())
    s = (pred.iloc[:, 0] if pred.ndim > 1 else pred).dropna()
    dates = sorted(set(s.index.get_level_values(0)))
    rebal = dates[::8]
    # 均匀抽样, 覆盖不同市况
    idx = np.linspace(0, len(rebal) - 1, N_DAYS).astype(int)
    sample_days = [rebal[i] for i in idx]
    print(f"  抽样调仓日: {[str(d.date()) for d in sample_days]}")

    pro = _pro()
    rows, liq = [], []
    for d in sample_days:
        day = d.strftime("%Y-%m-%d")
        codes = list(s.loc[d].nlargest(TOPN).index)
        got = 0
        for c in codes:
            try:
                m = pro.stk_mins(ts_code=_to_ts(c), freq=FREQ,
                                 start_date=f"{day} 09:30:00",
                                 end_date=f"{day} 15:00:00")
            except Exception:
                time.sleep(SLEEP)
                continue
            time.sleep(SLEEP)
            if not len(m):
                continue
            m = m.sort_values("trade_time")
            closes = m["close"].values
            # 14:45 那根的收盘 = 14:45 时点价; 最后一根 15:00 = 全日收盘
            if len(closes) < 2:
                continue
            rows.append({"date": day, "code": c,
                         "px_1445": float(closes[-2]),
                         "px_close": float(closes[-1]),
                         "score": float(s.loc[d, c])})
            liq.append({"date": day, "code": c,
                        "amt_tail": float(m["amount"].values[-1]),
                        "amt_day": float(m["amount"].sum())})
            got += 1
        print(f"  {day}: 取到 {got}/{len(codes)} 只", flush=True)

    if not rows:
        print("  没有取到数据")
        return

    df = pd.DataFrame(rows)
    lq = pd.DataFrame(liq)
    out = Path("factor_lab/results/.cache/minute_verify.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.merge(lq, on=["date", "code"]).to_parquet(out, index=False)
    print(f"\n  已落盘 {out} ({len(df)} 行)")

    print("\n  === 1. 尾盘流动性 (14:45-15:00 成交额) ===")
    print(f"  中位 {lq.amt_tail.median()/1e4:.0f} 万元  "
          f"25%分位 {lq.amt_tail.quantile(.25)/1e4:.0f} 万元  "
          f"最小 {lq.amt_tail.min()/1e4:.1f} 万元")
    print(f"  占全日成交额比例: 中位 {(lq.amt_tail/lq.amt_day).median():.1%}")
    for amt in (2000, 5000):
        share = (lq.amt_tail * 0.1 >= amt).mean()   # 只吃尾盘 10% 成交额
        print(f"  单笔买 {amt} 元且不超过尾盘成交额10%的可行比例: {share:.1%}")

    print("\n  === 2. 14:45 截断 vs 收盘价 的价差 ===")
    dv = (df.px_close / df.px_1445 - 1)
    print(f"  14:45->收盘 涨跌幅: 均值 {dv.mean():+.3%}  "
          f"标准差 {dv.std():.3%}  |>1%| 占比 {(dv.abs()>0.01).mean():.1%}")
    print("  (这段价差就是'14:50下单、按收盘价成交'的滑点上限)")


if __name__ == "__main__":
    main()
