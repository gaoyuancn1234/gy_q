#!/usr/bin/env python3
"""14:45 截断特征 + 已落盘 LightGBM 出分。

用户指定: 历史日线用 Qlib, 只覆盖/追加信号日当天的 14:45 截断 OHLCV。
没有落盘模型就失败, 不准用隔日预测冒充。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

LOOKBACK = 80
EPS = 1e-12
# 截断K 至少要覆盖成分的这个比例才允许出分。见 _patch_or_append 里的说明。
MIN_COVERAGE = 0.80


def _inst_stem(inst: str) -> str:
    """SH600012 -> sh600012"""
    return inst[:2].lower() + inst[2:]


def _inst_ts(inst: str) -> str:
    """SH600012 -> 600012.SH"""
    num, ex = inst[2:], inst[:2].upper()
    if num[:2] in ("43", "82", "83", "87", "92"):
        return f"{num}.BJ"
    return f"{num}.{ex}"


def _wide(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """MultiIndex -> date x instrument。"""
    s = df[col]
    if s.index.names[0] == "instrument":
        s = s.swaplevel().sort_index()
    return s.unstack(level=1).sort_index()


def _today_members() -> list[str]:
    """最新时点成分, Qlib 代码。"""
    import json
    mem = json.loads(
        (PROJECT_DIR / "data" / "csi2000_membership.json")
        .read_text(encoding="utf-8")
    )
    latest = max(mem)
    out = []
    for code in mem[latest]:
        if "." in code:
            ex, num = code.split(".")
            out.append(ex.upper() + num)
        else:
            out.append(code)
    return sorted(set(out))


def _trunc_table(day: str) -> pd.DataFrame:
    """取指定日的截断日线。信号日优先快照, 历史日走分钟聚合。

    2026-09-14 实测: Tushare stk_mins 对**当天**返回 0 根K, 收盘后仍是 0
    (同一只票 9-11 能返回 16 根)。也就是分钟数据要隔日才发布, 逐只预取
    那条路对"今天"永远拿不到数据 —— 这与当天同时发生的限流无关, 就算
    不限流、跑满 45 分钟, 抓到的也是 0 根。
    实时快照是目前唯一能取到当日盘中的源, 而且 2000 只 1.5 秒、所有票
    同一时刻(实测前后 27 秒), 顺带解决了逐只串行导致的"每只截断时点
    不同"(14:00~14:16 离散)那个老问题。
    历史日/回放仍走分钟聚合, 那条路没变。
    """
    from data_hub.trunc_daily import build_trunc_for_dates
    from data_hub.paths import trunc_daily_dir

    try:
        from qlib_engine.fetch_snapshot import trunc_from_snapshot
        g = trunc_from_snapshot(day)
        if g is not None and len(g):
            _tod = g["last_tod"].astype(str).sort_values()
            print(
                f"[auction] 快照截断 {len(g)} 只  时点 "
                f"{_tod.iloc[0]} ~ {_tod.iloc[-1]}",
                flush=True,
            )
            return g
    except Exception as e:
        # 快照拿不到不等于不能出分 —— 分钟缓存里若有当日数据仍可聚合。
        # 但要吼出来, 不能静默回退到一条已知对今天无效的路。
        print(
            f"[auction] 快照失败, 回退分钟聚合: {type(e).__name__}: {e}",
            flush=True,
        )

    out = trunc_daily_dir() / f"_auction_{day.replace('-', '')}.parquet"
    path = build_trunc_for_dates(
        [day],
        out=out,
        require_cutoff=False,
    )
    df = pd.read_parquet(path)
    df["date"] = df["date"].astype(str)
    g = df.loc[df["date"] == day].copy()
    if g.empty:
        raise SystemExit(f"{day} 没有截断日线 (分钟缓存缺 14:00 以后的K)")
    if "last_tod" in g.columns:
        # 截断时点是个分布不是一个点。
        # 2026-09-14: prefetch 14:00 起逐只取数(0.47s x 2000 只 = 16 分钟),
        # Tushare 每次只返回"到此刻为止"的K —— 第 1 只拿到 14:00, 第 2000 只
        # 拿到 14:16。也就是说 CUTOFF 写着 14:45, 实际拿到的是 14:00~14:16,
        # 而且**各只不一致**(截面特征建在不同时点上)。
        # require_cutoff=False 会照单全收, 原先只打一个众数, 看不出离散度。
        # 这不是前视(用的是更旧的信息, 偏保守), 但也不是回测测的那个时点。
        # 根治要换批量取数(掘金 SDK 已连着, history() 能一次取多只),
        # 逐只串行取数在 45 分钟窗口内拿不到同步的 14:45 截面。
        # .median() 对字符串列会抛 TypeError —— 2026-09-13 在这个文件里
        # 刚踩过一次, 我写这段可见化时又踩了第二次(冒烟抓到)。
        # 时刻是字符串, 取中位只能排序后取中间那个。
        _tod = g["last_tod"].astype(str).sort_values()
        _mid = _tod.iloc[len(_tod) // 2] if len(_tod) else "?"
        print(
            f"[auction] 截断K 时点 最早 {_tod.iloc[0] if len(_tod) else chr(63)} "
            f"中位 {_mid} "
            f"最晚 {_tod.iloc[-1] if len(_tod) else chr(63)} "
            f"众数 {_tod.mode().iat[0] if len(_tod) else chr(63)}"
            f"  (名义 CUTOFF {__import__('data_hub.trunc_daily', fromlist=['CUTOFF']).CUTOFF})",
            flush=True,
        )
        print(
            # 2026-09-13: 原写 .median() —— last_tod 是字符串列, 新版 pandas
            # 直接抛 TypeError。这行在 score_day 的必经路径上, 意味着
            # **实盘 14:45 出分会崩**, 而此前只做过空跑(空跑不到这一步)。
            # 字符串取众数用 mode(), 语义也更对(截断点应当一致)。
            f"[auction] 截断K last_tod 众数 "
            f"{g['last_tod'].astype(str).mode().iat[0] if len(g) else '?'} "
            f"只数 {g['stem'].nunique() if 'stem' in g.columns else len(g)}",
            flush=True,
        )
    return g.drop_duplicates("stem").set_index("stem")


def _ratio_row(
    insts: list[str],
    last_close: np.ndarray,
    last_day: str,
) -> np.ndarray:
    """未复权截断价 -> Qlib 前复权, 用昨日官方收盘对齐。"""
    from data_hub.paths import daily_raw_dir

    ratio = np.ones(len(insts), dtype=np.float64)
    f = daily_raw_dir() / f"{last_day.replace('-', '')}.parquet"
    if not f.exists():
        return ratio
    px0 = pd.read_parquet(f).set_index("ts_code")
    for j, inst in enumerate(insts):
        ts = _inst_ts(inst)
        if ts not in px0.index:
            continue
        raw_c = float(px0.loc[ts, "close"])
        q_c = float(last_close[j])
        if raw_c > 0 and np.isfinite(q_c) and q_c > 0:
            ratio[j] = q_c / raw_c
    return ratio


def _patch_or_append(
    opens, highs, lows, closes, vols, vwaps, insts, day, trunc,
    min_coverage: float = MIN_COVERAGE,
):
    """信号日已在日历里则覆盖, 否则追加一行。"""
    loc_of = {str(d)[:10]: i for i, d in enumerate(opens.index)}
    loc = loc_of.get(day)
    o = opens.to_numpy(dtype=np.float64, copy=True)
    h = highs.to_numpy(dtype=np.float64, copy=True)
    l = lows.to_numpy(dtype=np.float64, copy=True)
    c = closes.to_numpy(dtype=np.float64, copy=True)
    v = vols.to_numpy(dtype=np.float64, copy=True)
    w = vwaps.to_numpy(dtype=np.float64, copy=True)
    last_i = o.shape[0] - 1
    last_day = str(opens.index[-1])[:10]
    ratio = _ratio_row(insts, c[last_i], last_day)
    o_t = np.full(len(insts), np.nan)
    h_t = np.full(len(insts), np.nan)
    l_t = np.full(len(insts), np.nan)
    c_t = np.full(len(insts), np.nan)
    v_t = np.full(len(insts), np.nan)
    w_t = np.full(len(insts), np.nan)
    n_hit = 0
    for j, inst in enumerate(insts):
        stem = _inst_stem(inst)
        if stem not in trunc.index:
            continue
        row = trunc.loc[stem]
        r = ratio[j]
        o_t[j] = float(row["open"]) * r
        h_t[j] = float(row["high"]) * r
        l_t[j] = float(row["low"]) * r
        c_t[j] = float(row["close"]) * r
        v_t[j] = float(row["volume"])
        w_t[j] = float(row["vwap"]) * r
        n_hit += 1
    # 2026-09-14: 原先只要 >=50 只就放行。对 2000 只的池子这个下限没有意义 ——
    # 若 prefetch 只取回 300 只, 从 300 里选 top100 是"前 33%", 而回测测的是
    # "前 5%", 等于换了一套策略还照常下单。改成按比例卡。
    # 实测依据(2026-09-14 抽 40 只未缓存成分): Tushare stk_mins 可得率 100%,
    # 单只 0.47s, 全量约 16 分钟 —— 14:00 起在 14:45 前取完有充分余量。
    # 所以正常日子覆盖率应接近 1, 差的那部分基本只有停牌。
    if n_hit < min_coverage * len(insts):
        raise SystemExit(
            f"截断覆盖 {n_hit}/{len(insts)} = {n_hit/max(1,len(insts)):.1%}"
            f" < 下限 {min_coverage:.0%}, 拒绝出分。\n"
            f"  信号日正常走快照(全市场 1.5s), 覆盖率低通常意味着快照失败"
            f"并回退到了分钟聚合 —— 往上翻 [auction] 那几行看是哪一条。\n"
            f"  手查: python -m qlib_engine.fetch_snapshot")
    if loc is not None:
        o[loc], h[loc], l[loc] = o_t, h_t, l_t
        c[loc], v[loc], w[loc] = c_t, v_t, w_t
        return o, h, l, c, v, w, loc, n_hit
    o = np.vstack([o, o_t])
    h = np.vstack([h, h_t])
    l = np.vstack([l, l_t])
    c = np.vstack([c, c_t])
    v = np.vstack([v, v_t])
    w = np.vstack([w, w_t])
    return o, h, l, c, v, w, o.shape[0] - 1, n_hit


def _assert_history_fresh(day: str, cal: list[str]) -> None:
    """qlib 的最后一天必须是信号日的上一个交易日。

    2026-09-14: 发现 qlib 日线停在 9-11 而没有任何定时任务在刷新
    (刷新是 python -m qlib_engine.data_setup_tushare, 只能手动跑)。
    那天恰好 9-11 就是上一个交易日, 接上当日快照正好连续, 所以没露馅;
    但到 9-15, 历史仍停在 9-11, 当日行追加上去就变成
    ... 9-10, 9-11, 9-15 —— 中间缺了 9-14, 而 Alpha158 里所有带窗口的
    因子(动量/波动/量比)都会算在错位的序列上, 且全程不报错。
    宁可拒绝出分, 也不要用缺口序列下真单。
    """
    from market_calendar import prev_trading_day

    # 信号日已经在日历里 = 历史回放, 历史自然是全的, 这道闸不适用。
    # 只有"把新的一天追加到历史后面"时才需要担心中间缺天。
    # (冒烟套件跑 score_day('2026-09-11') 就是这种情况, 差点被误伤。)
    if day in cal:
        return

    want = prev_trading_day(day)
    if want is None:
        print("[auction] 往回 15 天找不到交易日, 跳过历史新鲜度检查",
              flush=True)
        return
    have = cal[-1] if cal else "(空)"
    if have != want:
        raise SystemExit(
            f"qlib 历史最后一天是 {have}, 而 {day} 的上一个交易日是 {want}"
            f" —— 历史有缺口, 拒绝出分。\n"
            f"  跑 python -m qlib_engine.data_setup_tushare --universe csi2000"
            f" 补齐后重试 (按日缓存, 只补缺的那天, 几分钟)。")
    print(f"[auction] 历史新鲜度 OK: qlib 到 {have}, 上一交易日 {want}",
          flush=True)


def score_day(day: str, preset: str,
              min_coverage: float = MIN_COVERAGE,
              ) -> tuple[pd.Series, dict]:
    """对某一日截断特征打分。返回 (分数, 未复权收盘价)。

    min_coverage: 截断K 至少要覆盖成分的比例。生产走默认值; 冒烟用低值
    只验路径(测试机的分钟缓存本来就只有几百只, 不代表生产覆盖率)。
    """
    import qlib
    from qlib.data import D
    from qlib.contrib.data.loader import Alpha158DL
    from qlib_paths import qlib_init_kwargs
    from factor_lab.lgb_store import (
        apply_robust_zscore,
        load_boosters,
        models_ready,
    )
    from factor_lab._trunc_feat import _feat_at, _stack

    if not models_ready(preset):
        raise SystemExit(
            f"没有可用的 {preset} booster, "
            "先跑 python -m factor_lab.persist_boosters"
        )
    qlib.init(**qlib_init_kwargs("csi2000"))
    trunc = _trunc_table(day)
    insts = _today_members()
    cal = [str(x)[:10] for x in D.calendar()]
    _assert_history_fresh(day, cal)
    if day in cal:
        loc_day = cal.index(day)
        start = cal[max(0, loc_day - LOOKBACK)]
        end = day
    else:
        start = cal[max(0, len(cal) - LOOKBACK)]
        end = cal[-1]
    raw = D.features(
        insts,
        ["$open", "$high", "$low", "$close", "$volume", "$vwap"],
        start_time=start,
        end_time=end,
    )
    if raw is None or raw.empty:
        raise SystemExit("Qlib 取不到历史日线")
    opens = _wide(raw, "$open")
    highs = _wide(raw, "$high")
    lows = _wide(raw, "$low")
    closes = _wide(raw, "$close")
    vols = _wide(raw, "$volume")
    vwaps = _wide(raw, "$vwap")
    cols = [c for c in insts if c in closes.columns]
    opens, highs = opens[cols], highs[cols]
    lows, closes = lows[cols], closes[cols]
    vols, vwaps = vols[cols], vwaps[cols]
    insts = cols
    o, h, l, c, v, w, loc, n_hit = _patch_or_append(
        opens, highs, lows, closes, vols, vwaps, insts, day, trunc,
        min_coverage=min_coverage,
    )
    if loc < 60:
        raise SystemExit("历史日线不足 60 天, 算不了 Alpha158")
    _, names = Alpha158DL.get_feature_config()
    feat = _stack(_feat_at(o, h, l, c, v, w, loc), names, insts)

    # 没有当日截断K的票, 信号日那一行是全 NaN。LightGBM 照样会吐出一个数
    # (NaN 走默认分支), 于是它们**带着分数进入排序**。
    # 2026-09-14 实测: 缓存只有 214 只时出分 2000 只, Top100 里 35 只是这种
    # ——而这批恰恰是**停牌或取数失败**的票, 是最不该买的一批。
    # 它们的分数还挤在一起(std 0.085 vs 有K的 0.237), 即模型对 NaN 行的输出
    # 几乎是个常数, 排名纯属噪声。
    # 另外 px 只收录有K的票, 选中无K的票下游也拿不到价格。
    # 一律剔除 —— 宁可少选, 不要拿 NaN 当信号。
    covered = [i for i in insts if _inst_stem(i) in trunc.index]
    n_drop_nan = len(insts) - len(covered)
    if n_drop_nan:
        print(f"[auction] 剔除无截断K {n_drop_nan} 只 (全 NaN 行不出分)",
              flush=True)
    feat = feat.loc[covered]
    insts = covered

    boosters, info = load_boosters(preset)
    x = apply_robust_zscore(feat, info)
    pred = np.mean([b.predict(x) for b in boosters], axis=0)
    scores = pd.Series(pred, index=insts, name="score")
    scores = scores.replace([np.inf, -np.inf], np.nan).dropna()
    px = {
        inst: float(trunc.loc[_inst_stem(inst), "close"])
        for inst in scores.index
        if _inst_stem(inst) in trunc.index
    }
    # 出分集合必须被价格集合覆盖 —— 否则下单时拿不到价, 只能静默丢单
    missing_px = [i for i in scores.index if i not in px]
    if missing_px:
        raise SystemExit(
            f"{len(missing_px)} 只有分数却无截断价 (例 {missing_px[:3]}), "
            "这不该发生, 拒绝出分")
    print(
        f"[auction] {day} 截断 {n_hit} 只 有效分数 {len(scores)} "
        f"覆盖率 {n_hit/max(1,len(cols)):.1%} 种子 {len(boosters)}",
        flush=True,
    )
    return scores, px
