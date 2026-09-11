#!/usr/bin/env python3
"""akshare 日频估值 → Qlib bin

生成字段: pe_ttm, pb, ps_ttm, total_mv, circ_mv, turn

为什么换成这个源
----------------
2026-09-05~06 用 baostock 抓季报跑了两轮 (合计约 3.5 小时): 第一轮因不检查
error_code 空转两个半小时，第二轮跑到 360/549 被限流拉黑 (10001011)。

但更要紧的问题是**取错了字段**。`factor_lab/factors/fundamental.py` 的 23 个
因子只用到:

    pe_ttm  pb  ps_ttm  total_mv  circ_mv  close  turn

而 baostock 那个脚本产出的是 eps_ttm / roe_avg / net_profit / total_share /
yoy_ni —— 一个都对不上。即便下载成功，get_all_exprs() 仍会因缺 pe_ttm 返回
空，基本面因子数依旧是 0。开跑前没有核对"下载的字段"与"因子要的字段"，
是同一类"两处各写一份、没人对过"的问题，只不过跨的是数据层与因子层。

akshare `stock_value_em` 一次调用返回单只股票的完整历史日频估值:

    数据日期 当日收盘价 总市值 流通市值 总股本 流通股本
    PE(TTM) PE(静) 市净率 PEG值 市现率 市销率

600036 实测 2106 行，覆盖 2018-01-02 ~ 2026-09-04。五个字段全有，且是按
交易日的日频数据 —— 不需要按公告日 forward-fill，比季报那条路更简单，也
没有"用最新报表回填历史"的前视风险。

调用量: 一只一次，549 只约 10~20 分钟。比 baostock 的 4 万次小两个数量级，
不会触发限流。

前视说明
--------
日频估值由当日价格与"截至当日已披露"的报表算出，本身是时点数据。残余风险
是数据商可能用重述后的财报重算历史 PE —— 这一点所有商业数据源都一样，
无法从消费端消除。

用法
----
    python -m factor_lab.data.akshare_valuation              # 全量
    python -m factor_lab.data.akshare_valuation --limit 5    # 冒烟
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd

# 断点文件与最终产物同目录
CACHE_DIR = Path(__file__).resolve().parent.parent / "results" / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CKPT = CACHE_DIR / "daily_valuation.partial.parquet"
FINAL = CACHE_DIR / "daily_valuation.parquet"

# 每只之间的停顿。baostock 那次是持续高频查询后被拉黑的，这里主动摊开。
REQUEST_PAUSE = 0.3

# akshare 中文列 -> qlib 字段
COLMAP = {
    "数据日期": "date",
    "PE(TTM)": "pe_ttm",
    "市净率": "pb",
    "市销率": "ps_ttm",
    "总市值": "total_mv",
    "流通市值": "circ_mv",
}


def _ak_symbol(inst: str) -> str:
    """SH600036 / SZ000001 -> 600036 / 000001"""
    return inst[2:] if inst[:2] in ("SH", "SZ") else inst


def fetch_one(inst: str, timeout: float = 60):
    """取单只股票的历史估值。失败抛异常，由调用方决定跳过还是中止。"""
    from net_guard import run_with_timeout
    import akshare as ak

    df = run_with_timeout(lambda: ak.stock_value_em(symbol=_ak_symbol(inst)),
                          timeout, f"stock_value_em {inst}")
    if df is None or df.empty:
        return None
    missing = [c for c in COLMAP if c not in df.columns]
    if missing:
        # 列名变了就该立刻报出来，不要静默产出缺列的表
        raise RuntimeError(f"{inst}: akshare 返回缺列 {missing}；"
                           f"实际列 {list(df.columns)}")
    out = df[list(COLMAP)].rename(columns=COLMAP)
    out["instrument"] = inst
    out["date"] = pd.to_datetime(out["date"])
    return out


def download(instruments: list, max_consecutive_fail: int = 20) -> pd.DataFrame:
    """逐只下载，支持断点续传

    连续 max_consecutive_fail 只失败即中止 —— 正常情况下不可能，几乎必然是
    源侧出了问题。宁可现在失败，也不要跑完全程交出一张残表。
    """
    from net_guard import install_default_request_timeout
    install_default_request_timeout(25)

    frames, done = [], set()
    if CKPT.exists():
        prev = pd.read_parquet(CKPT)
        frames.append(prev)
        done = set(prev["instrument"].unique())
        print(f"  断点续传: 已有 {len(done)} 只 / {len(prev)} 行", flush=True)

    todo = [i for i in instruments if i not in done]
    if not todo:
        print("  全部已在断点文件中")
        return pd.concat(frames, ignore_index=True)

    t0 = time.time()
    fail_streak, n_fail = 0, 0
    for i, inst in enumerate(todo):
        try:
            one = fetch_one(inst)
            if one is None or one.empty:
                raise RuntimeError("返回空表")
            frames.append(one)
            fail_streak = 0
        except Exception as e:
            n_fail += 1
            fail_streak += 1
            print(f"  [{i+1}/{len(todo)}] {inst} 失败: "
                  f"{type(e).__name__} {str(e)[:80]}", flush=True)
            if fail_streak >= max_consecutive_fail:
                raise RuntimeError(
                    f"连续 {fail_streak} 只失败 (已处理 {i+1}/{len(todo)})，"
                    f"判定数据源异常，中止")

        if (i + 1) % 20 == 0:
            el = time.time() - t0
            eta = el / (i + 1) * (len(todo) - i - 1)
            rows = sum(len(f) for f in frames)
            print(f"  [{i+1}/{len(todo)}] {inst} 累计 {rows} 行  "
                  f"失败 {n_fail}  已用 {el/60:.1f}min ETA {eta/60:.1f}min",
                  flush=True)
            pd.concat(frames, ignore_index=True).to_parquet(CKPT)

        time.sleep(REQUEST_PAUSE)

    df = pd.concat(frames, ignore_index=True)
    df.to_parquet(CKPT)
    print(f"  下载完成: {len(df)} 行 / {df['instrument'].nunique()} 只，"
          f"失败 {n_fail} 只")
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description='akshare 日频估值下载')
    ap.add_argument('--limit', type=int, help='只跑前 N 只 (冒烟)')
    ap.add_argument('--qlib-dir',
                    default=str(Path.home() / '.qlib/qlib_data/cn_data_bs'))
    args = ap.parse_args()

    import qlib
    from qlib.constant import REG_CN
    qlib.init(provider_uri=args.qlib_dir, region=REG_CN)
    from qlib.data import D

    inst = D.instruments("csi300")
    instruments = sorted(D.list_instruments(instruments=inst, as_list=True))
    if args.limit:
        instruments = instruments[:args.limit]
    print(f"股票池: {len(instruments)} 只")

    df = download(instruments)
    if df.empty:
        print("✗ 空表")
        return 1

    df = df.sort_values(["instrument", "date"]).reset_index(drop=True)
    df.to_parquet(FINAL)
    print(f"\n已写入 {FINAL}")
    print(f"  {len(df)} 行 / {df['instrument'].nunique()} 只")
    print(f"  日期 {df['date'].min().date()} ~ {df['date'].max().date()}")
    for c in ("pe_ttm", "pb", "ps_ttm", "total_mv", "circ_mv"):
        nn = df[c].notna().sum()
        print(f"  {c:10s} 非空 {nn:>8} ({nn/len(df):.1%})")
    return 0


if __name__ == '__main__':
    sys.exit(main())
