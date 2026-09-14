#!/usr/bin/env python3
"""全市场行情快照 — 逐只取数的替代路径 (2026-09-14 新增, 暂不接入实盘)。

为什么要这个
------------
原路径 fetch_today_minutes 逐只调 Tushare stk_mins, 2000 只 = 2000 次调用。
2026-09-14 第一次全量真跑暴露两个问题:

  1. 限流。数据源(第三方镜像)约每分钟 20 次, 2000 只需要 100 分钟, 而
     14:00->14:45 只有 45 分钟。跑 26 分钟 0 只入库, 且每只失败前要退避
     215 秒, 全程跑完要 119 小时。不是调参能解决的, 是算术不够。
  2. 截面不同步。逐只串行意味着第 1 只拿到 14:00 的数据、第 2000 只拿到
     14:16 的 —— 横截面特征建在 2000 个不同时刻上。auction_infer 里早有
     注释指出这点, 说"根治要换批量取数"。

快照一次返回全市场当前的 今开/最高/最低/现价/成交量/成交额, 两个问题
同时解决: 2000 只 4 批 1.5 秒, 且所有票是同一时刻(实测前后 27 秒)。

口径 (2026-09-14 实测, 勿凭记忆改)
----------------------------------
  - volume 单位是**股**, 与 qlib bin 的 $volume 一致。
    注意 stk_mins 的 vol 是**手**(差 100 倍) —— 原路径某处要乘 100,
    这条路**不能乘**。这种错不报错, 只会让特征悄悄算错。
  - amount 单位是**元**, 与 qlib $amount 一致 (实测 9-11 逐笔对上)。
  - 价格是**不复权**。auction_infer._ratio_row 用昨日官方收盘换算到
    qlib 尺度, 与数据来源无关, 这条路可直接复用。
  - pre_close **不可信**: 除息日(名称 XD 开头)新浪会按分红调低它。
    实测 1957 只里 3 只偏离, 全是 XD。要昨收请读官方日线。
  - 单次请求有长度上限: 2000 只一次会 HTTP 431 (Request Header Too Large)。
    实测 800 只可以, 取 500 留余量。
  - 停牌票的判据是 **volume > 0**, 不是 close > 0。2026-09-14 收盘快照里
    3 只停牌: 香山股份 全零; 新华传媒 close=5.31 / 金橙子 close=44.40 但
    open=high=low=0 —— 快照给的是最后成交价, 而当日没有开盘价。按
    close>0 过滤会放它们进去, 然后 close/open-1 算出 inf。实测那次
    涨跌幅分布的最大值就是 inf, 就是这两只。

未验证
------
  快照拼出的当日K与官方日线的一致性 —— 快照只能取当下, 取不到历史,
  所以只能今天收盘存一份、明天官方日线出来后对比。见 compare_to_daily()。
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent

# 单批股票数。实测 800 可以, 2000 会 HTTP 431; 取 500 留余量。
BATCH = 500
# 快照落盘目录 (与分钟缓存分开, 这条路暂不接入实盘)
SNAP_DIR = PROJECT_DIR / "factor_lab" / "results" / ".cache" / "snapshot"


def _members() -> list[str]:
    """最新时点成分, 返回 6 位代码。"""
    mem = json.loads(
        (PROJECT_DIR / "data" / "csi2000_membership.json")
        .read_text(encoding="utf-8")
    )
    out = []
    for code in mem[max(mem)]:
        out.append(code.split(".")[1] if "." in code else code[2:])
    return sorted(set(out))


def _to_qlib(code: str) -> str:
    """600020 -> SH600020; 北交所 43/82/83/87/92 -> BJ。"""
    if code[:2] in ("43", "82", "83", "87", "92"):
        return "BJ" + code
    return ("SH" if code.startswith("6") else "SZ") + code


def fetch_market_snapshot(codes: list[str] | None = None,
                          batch: int = BATCH) -> pd.DataFrame:
    """取全市场当前快照。

    Returns:
        DataFrame, 一行一只, 列:
          inst   qlib 代码 (SH600020)
          open/high/low/close  当日至此刻的 开/最高/最低/最新 (不复权, 元)
          volume 当日至此刻累计成交量 (**股**)
          amount 当日至此刻累计成交额 (元)
          tod    行情时间戳 (HH:MM:SS)
          tradable  当日是否有成交 (volume>0 且 open>0)
        停牌票保留在结果里但 tradable=False, 由调用方决定丢弃还是保留 ——
        不在这里静默删行, 否则调用方会以为成分股少了几只而去查别的地方。
    """
    import tushare as ts

    codes = codes or _members()
    parts = []
    for i in range(0, len(codes), batch):
        chunk = codes[i:i + batch]
        parts.append(ts.get_realtime_quotes(chunk))
    df = pd.concat(parts, ignore_index=True)

    num = ["open", "high", "low", "price", "volume", "amount", "pre_close"]
    for c in num:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    out = pd.DataFrame({
        "inst": df["code"].map(_to_qlib),
        "name": df["name"],
        "open": df["open"],
        "high": df["high"],
        "low": df["low"],
        "close": df["price"],     # 盘中即最新价, 收盘后即收盘价
        "volume": df["volume"],   # 股
        "amount": df["amount"],   # 元
        "tod": df["time"].astype(str),
    })
    # 判据是成交量, 不是价格 —— 见模块文档"停牌票"一段。
    out["tradable"] = (out["volume"] > 0) & (out["open"] > 0)
    return out.drop_duplicates("inst").reset_index(drop=True)


def save_snapshot(df: pd.DataFrame, day: str | None = None,
                  tag: str = "") -> Path:
    """落盘。文件名带时刻, 一天可以存多份(盘中/收盘)。"""
    SNAP_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now()
    day = day or now.strftime("%Y-%m-%d")
    stamp = now.strftime("%H%M%S")
    name = f"{day.replace('-', '')}_{stamp}{('_' + tag) if tag else ''}.parquet"
    p = SNAP_DIR / name
    df.to_parquet(p, index=False)
    return p


def trunc_from_snapshot(day: str, snap: pd.DataFrame | None = None,
                        save: bool = True) -> pd.DataFrame | None:
    """快照 -> 截断日线表, 与 data_hub.trunc_daily 同结构。

    auction_infer._patch_or_append 要的是按 stem 索引、含
    open/high/low/close/volume/vwap 的表。这里产出同样的东西, 好让
    出分那条路不用关心数据从哪来。

    只对**当天**有效: 实时行情取不到历史。day 不是今天就返回 None,
    调用方回退到分钟聚合(研究/回放路径仍走原来那条)。

    单位: volume 是股, 与 trunc_daily._agg_one 的输出一致 —— 那边要按
    amount/vol/close 的比值判断 Tushare 给的是手还是股, 这边不用判,
    新浪一直是股 (2026-09-14 实测)。

    save=True 时顺手把原始快照落盘, 出了问题能回看当时的行情。
    """
    from datetime import date as _date

    if day != _date.today().strftime("%Y-%m-%d"):
        return None

    if snap is None:
        snap = fetch_market_snapshot()
        if save:
            try:
                p = save_snapshot(snap, day=day, tag="auction")
                print(f"[snapshot] 已存 {p}", flush=True)
            except OSError as e:
                print(f"[snapshot] 落盘失败(不影响出分): {e}", flush=True)

    g = snap[snap["tradable"]].copy()
    if g.empty:
        return None
    g["stem"] = g["inst"].str.lower()
    g["date"] = day
    g["vwap"] = g["amount"] / g["volume"]
    g["n_bars"] = 1                      # 快照是一根, 不是逐根聚合
    g = g.rename(columns={"tod": "last_tod"})
    cols = ["stem", "date", "open", "high", "low", "close",
            "volume", "amount", "vwap", "n_bars", "last_tod"]
    return g[cols].drop_duplicates("stem").set_index("stem")


def compare_to_daily(snap_path: str | Path, day: str) -> pd.DataFrame:
    """把存下的收盘快照跟官方日线比 —— 切换前必须过的一关。

    官方日线要等当晚/次日才有, 所以这个函数是给"明天"用的。
    比 open/high/low/close/volume 五项, 逐只算相对误差。
    """
    from data_hub.paths import daily_raw_dir

    snap = pd.read_parquet(snap_path)
    f = daily_raw_dir() / f"{day.replace('-', '')}.parquet"
    if f.exists():
        off = pd.read_parquet(f)
    else:
        # 本地还没落盘就直接问接口。pro.daily(trade_date=...) 一次返回全市场,
        # 只一次调用。**不写进 daily_raw_dir** —— 那是正式管线的数据目录,
        # 这里只是验证, 不该往共享数据里塞东西。
        from qlib_engine.fetch_minute import _pro
        print(f"[compare] 本地无 {f.name}, 直接取接口")
        off = _pro().daily(trade_date=day.replace("-", ""))
        if off is None or len(off) == 0:
            raise SystemExit(
                f"{day} 官方日线尚未发布 (接口返回空)。"
                f"日线一般收盘后若干小时才有, 晚些再跑。")

    def _ts_to_qlib(ts: str) -> str:
        num, ex = ts.split(".")
        return ("BJ" if ex == "BJ" else ex) + num

    off["inst"] = off["ts_code"].map(_ts_to_qlib)
    if "tradable" in snap.columns:
        snap = snap[snap["tradable"]]
    # 官方列显式改名再 join。原先靠 join(rsuffix="_off") 区分, 然后仍按
    # 原名取值 —— m["open"] 两边都指向快照自己, 比出来五个字段误差恒为 0,
    # 看着像"完美一致", 实际是自己跟自己比。只有 volume/vol 没撞名, 那行
    # 才是真的。(2026-09-14 差点据此宣布验证通过。)
    off = off.rename(columns={
        "open": "o_off", "high": "h_off", "low": "l_off",
        "close": "c_off", "vol": "v_off",
    })
    m = snap.set_index("inst").join(
        off.set_index("inst")[["o_off", "h_off", "l_off", "c_off", "v_off"]],
        how="inner")
    print(f"[compare] 快照 {len(snap)} 只(仅有成交), 官方 {len(off)} 只, "
          f"可比对 {len(m)} 只")
    rows = []
    for col, offcol, scale in [
        ("open", "o_off", 1.0), ("high", "h_off", 1.0),
        ("low", "l_off", 1.0), ("close", "c_off", 1.0),
        # 官方日线 vol 是手, 快照是股
        ("volume", "v_off", 100.0),
    ]:
        if offcol not in m.columns:
            continue
        a = m[col]
        b = m[offcol] * scale
        ok = (a > 0) & (b > 0)
        rel = ((a[ok] - b[ok]) / b[ok]).abs()
        rows.append({
            "字段": col, "可比": int(ok.sum()),
            "中位相对误差": round(float(rel.median()), 8),
            "p95": round(float(rel.quantile(0.95)), 8),
            "最大": round(float(rel.max()), 8),
            "误差<0.1%占比": round(float((rel < 0.001).mean()), 4),
        })
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="全市场行情快照")
    ap.add_argument("--save", action="store_true", help="取一份并落盘")
    ap.add_argument("--tag", default="", help="文件名后缀, 如 close")
    ap.add_argument("--compare", default="", help="快照 parquet 路径")
    ap.add_argument("--day", default="", help="配合 --compare 的交易日")
    args = ap.parse_args()

    if args.compare:
        if not args.day:
            print("--compare 需要 --day")
            return 2
        print(compare_to_daily(args.compare, args.day).to_string(index=False))
        return 0

    t0 = time.time()
    df = fetch_market_snapshot()
    el = time.time() - t0
    live = int(df["tradable"].sum())
    print(f"[snapshot] {len(df)} 只, 当日有成交 {live} "
          f"({live/len(df):.1%}), 停牌 {len(df) - live}, 耗时 {el:.2f}s")
    print(f"[snapshot] 时间戳 {df['tod'].min()} ~ {df['tod'].max()}")
    if args.save:
        p = save_snapshot(df, tag=args.tag)
        print(f"[snapshot] 已存 {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
