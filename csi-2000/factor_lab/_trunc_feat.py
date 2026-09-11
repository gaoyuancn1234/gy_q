#!/usr/bin/env python3
"""14:45 截断 vs 完整日线: Alpha158 特征与 TopK。

只改信号日当天的 OHLCV (历史日线仍用收盘), 对齐 14:45 出分场景。
不重训、不改 Qlib 供给层。模型未落盘, 用轻量 LGB 拟合已有分数作代理。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PRED = (
    "factor_lab/results/rolling/predictions/"
    "D_expand_3v_3r_alpha158_LightGBM.pkl"
)
HOLD = 8
TOPK = 100
START = "2024-01-02"
END = "2026-09-09"
N_DAYS = 16
PX_START = "2023-10-01"
WINDOWS = (5, 10, 20, 30, 60)
EPS = 1e-12


def inst_stem(inst: str) -> str:
    """SH600012 -> sh600012"""
    return inst[:2].lower() + inst[2:]


def inst_ts(inst: str) -> str:
    """SH600012 -> 600012.SH; 北交所 -> .BJ"""
    num, ex = inst[2:], inst[:2].upper()
    if num[:2] in ("43", "82", "83", "87", "92"):
        return f"{num}.BJ"
    return f"{num}.{ex}"


def _mu(xs) -> float:
    import numpy as np
    ys = [x for x in xs if x == x]
    if not ys:
        return float("nan")
    return float(np.mean(ys))


def _rebal_dates(pred) -> list:
    """与 _trunc_ic 同一套 16 个调仓日。"""
    import numpy as np
    s = (pred.iloc[:, 0] if pred.ndim > 1 else pred).dropna()
    dates = sorted(
        d for d in s.index.get_level_values(0).unique()
        if START <= str(d)[:10] <= END
    )
    rebal = dates[::HOLD]
    if len(rebal) > N_DAYS:
        idx = np.linspace(0, len(rebal) - 1, N_DAYS).astype(int)
        rebal = [rebal[i] for i in idx]
    return rebal, s


def _wide(df, col: str):
    """MultiIndex (datetime, instrument) -> date x inst。"""
    s = df[col]
    if s.index.names[0] == "instrument":
        s = s.swaplevel().sort_index()
    return s.unstack(level=1).sort_index()


def _slope(y):
    import numpy as np
    w = y.shape[0]
    t = np.arange(w, dtype=np.float64)
    tm = t - t.mean()
    ym = y - np.nanmean(y, axis=0)
    den = float((tm ** 2).sum())
    return (tm[:, None] * ym).sum(axis=0) / den


def _rsqr(y):
    import numpy as np
    w = y.shape[0]
    t = np.arange(w, dtype=np.float64)
    sl = _slope(y)
    y_mean = np.nanmean(y, axis=0)
    t_mean = t.mean()
    yhat = y_mean + sl * (t[:, None] - t_mean)
    ss_res = np.nansum((y - yhat) ** 2, axis=0)
    ss_tot = np.nansum((y - y_mean) ** 2, axis=0)
    out = 1.0 - ss_res / (ss_tot + EPS)
    std = np.nanstd(y, axis=0, ddof=0)
    out[np.isclose(std, 0, atol=2e-5)] = np.nan
    return out


def _resi(y):
    import numpy as np
    w = y.shape[0]
    t = np.arange(w, dtype=np.float64)
    sl = _slope(y)
    y_mean = np.nanmean(y, axis=0)
    t_mean = t.mean()
    yhat_last = y_mean + sl * (t[-1] - t_mean)
    return y[-1] - yhat_last


def _nan_peak(a, which: str):
    """按列 nanargmax/nanargmin, 全 NaN 列返回 NaN。"""
    import numpy as np
    mask = np.all(np.isnan(a), axis=0)
    fill = -np.inf if which == "max" else np.inf
    filled = np.where(np.isnan(a), fill, a)
    if which == "max":
        idx = np.argmax(filled, axis=0)
        peak = np.max(filled, axis=0)
    else:
        idx = np.argmin(filled, axis=0)
        peak = np.min(filled, axis=0)
    peak = peak.astype(np.float64)
    loc = idx.astype(np.float64) + 1.0
    peak[mask] = np.nan
    loc[mask] = np.nan
    return peak, loc


def _corr(x, y):
    import numpy as np
    xm = np.nanmean(x, axis=0)
    ym = np.nanmean(y, axis=0)
    num = np.nansum((x - xm) * (y - ym), axis=0)
    den = np.sqrt(
        np.nansum((x - xm) ** 2, axis=0)
        * np.nansum((y - ym) ** 2, axis=0)
    )
    return num / (den + EPS)


def _feat_at(o, h, l, c, v, wap, loc: int) -> dict:
    """在 loc 这一天算完整 Alpha158。数组形状 (n_dates, n_inst)。"""
    import numpy as np
    ot, ht, lt, ct = o[loc], h[loc], l[loc], c[loc]
    vt, wt = v[loc], wap[loc]
    rng = ht - lt + EPS
    go = np.maximum(ot, ct)
    lo = np.minimum(ot, ct)
    out = {
        "KMID": (ct - ot) / ot,
        "KLEN": (ht - lt) / ot,
        "KMID2": (ct - ot) / rng,
        "KUP": (ht - go) / ot,
        "KUP2": (ht - go) / rng,
        "KLOW": (lo - lt) / ot,
        "KLOW2": (lo - lt) / rng,
        "KSFT": (2 * ct - ht - lt) / ot,
        "KSFT2": (2 * ct - ht - lt) / rng,
        "OPEN0": ot / ct,
        "HIGH0": ht / ct,
        "LOW0": lt / ct,
        "VWAP0": wt / ct,
    }
    for d in WINDOWS:
        sl = slice(loc - d + 1, loc + 1)
        cw, hw, lw, vw = c[sl], h[sl], l[sl], v[sl]
        prev = slice(loc - d, loc + 1)
        c_lag = c[prev]
        v_lag = v[prev]
        dlt = c_lag[1:] - c_lag[:-1]
        dltv = v_lag[1:] - v_lag[:-1]
        ret = c_lag[1:] / (c_lag[:-1] + EPS) - 1.0
        abs_ret_v = np.abs(ret) * vw
        logv = np.log(vw + 1.0)
        ret_c = cw / (c_lag[:-1] + EPS)
        log_dv = np.log(vw / (v_lag[:-1] + EPS) + 1.0)
        mx, imax = _nan_peak(hw, "max")
        mn, imin = _nan_peak(lw, "min")
        last = cw[-1]
        rank = (np.sum(cw <= last, axis=0).astype(np.float64) / d)
        gain = np.nansum(np.maximum(dlt, 0.0), axis=0)
        loss = np.nansum(np.maximum(-dlt, 0.0), axis=0)
        abss = np.nansum(np.abs(dlt), axis=0) + EPS
        vg = np.nansum(np.maximum(dltv, 0.0), axis=0)
        vl = np.nansum(np.maximum(-dltv, 0.0), axis=0)
        vabs = np.nansum(np.abs(dltv), axis=0) + EPS
        out[f"ROC{d}"] = c[loc - d] / ct
        out[f"MA{d}"] = np.nanmean(cw, axis=0) / ct
        out[f"STD{d}"] = np.nanstd(cw, axis=0, ddof=0) / ct
        out[f"BETA{d}"] = _slope(cw) / ct
        out[f"RSQR{d}"] = _rsqr(cw)
        out[f"RESI{d}"] = _resi(cw) / ct
        out[f"MAX{d}"] = mx / ct
        out[f"MIN{d}"] = mn / ct
        out[f"QTLU{d}"] = np.nanquantile(cw, 0.8, axis=0) / ct
        out[f"QTLD{d}"] = np.nanquantile(cw, 0.2, axis=0) / ct
        out[f"RANK{d}"] = rank
        out[f"RSV{d}"] = (ct - mn) / (mx - mn + EPS)
        out[f"IMAX{d}"] = imax / d
        out[f"IMIN{d}"] = imin / d
        out[f"IMXD{d}"] = (imax - imin) / d
        out[f"CORR{d}"] = _corr(cw, logv)
        out[f"CORD{d}"] = _corr(ret_c, log_dv)
        out[f"CNTP{d}"] = np.nanmean(dlt > 0, axis=0)
        out[f"CNTN{d}"] = np.nanmean(dlt < 0, axis=0)
        out[f"CNTD{d}"] = out[f"CNTP{d}"] - out[f"CNTN{d}"]
        out[f"SUMP{d}"] = gain / abss
        out[f"SUMN{d}"] = loss / abss
        out[f"SUMD{d}"] = (gain - loss) / abss
        out[f"VMA{d}"] = np.nanmean(vw, axis=0) / (vt + EPS)
        out[f"VSTD{d}"] = np.nanstd(vw, axis=0, ddof=0) / (vt + EPS)
        out[f"WVMA{d}"] = (
            np.nanstd(abs_ret_v, axis=0, ddof=0)
            / (np.nanmean(abs_ret_v, axis=0) + EPS)
        )
        out[f"VSUMP{d}"] = vg / vabs
        out[f"VSUMN{d}"] = vl / vabs
        out[f"VSUMD{d}"] = (vg - vl) / vabs
    return out


def _stack(feat: dict, names: list, insts) -> "pd.DataFrame":
    import pandas as pd
    cols = {n: feat[n] for n in names}
    return pd.DataFrame(cols, index=insts)


def _overlap(a, b, k: int) -> float:
    sa, sb = set(a[:k]), set(b[:k])
    return len(sa & sb) / float(k)


def main() -> None:
    import numpy as np
    import pandas as pd
    import qlib
    from qlib.contrib.data.loader import Alpha158DL
    from qlib.data import D
    from qlib_paths import qlib_init_kwargs
    from data_hub.paths import daily_raw_dir
    from data_hub.trunc_daily import build_trunc_for_dates

    pred = pd.read_pickle(PRED)
    rebal, scores = _rebal_dates(pred)
    days = [str(d)[:10] for d in rebal]
    print(f"调仓日 {len(days)} 个  {days[0]} ~ {days[-1]}")

    cache = build_trunc_for_dates(days)
    trunc = pd.read_parquet(cache)
    trunc["date"] = trunc["date"].astype(str)
    print(f"截断日线 {len(trunc)} 行")

    qlib.init(**qlib_init_kwargs("csi2000"))
    _, names = Alpha158DL.get_feature_config()
    print(f"Alpha158 因子 {len(names)}")

    insts = sorted({i for d in rebal for i in scores.loc[d].index})
    print(f"取价 {len(insts)} 只 {PX_START} ~ {END}")
    raw_px = D.features(
        insts,
        ["$open", "$high", "$low", "$close", "$volume", "$vwap"],
        start_time=PX_START,
        end_time=END,
    )
    opens = _wide(raw_px, "$open")
    highs = _wide(raw_px, "$high")
    lows = _wide(raw_px, "$low")
    closes = _wide(raw_px, "$close")
    vols = _wide(raw_px, "$volume")
    vwaps = _wide(raw_px, "$vwap")
    cols = [c for c in insts if c in closes.columns]
    opens, highs = opens[cols], highs[cols]
    lows, closes = lows[cols], closes[cols]
    vols, vwaps = vols[cols], vwaps[cols]
    insts = cols
    date_index = closes.index
    loc_of = {str(d)[:10]: i for i, d in enumerate(date_index)}

    ddir = daily_raw_dir()
    o_a = opens.to_numpy(dtype=np.float64, copy=True)
    h_a = highs.to_numpy(dtype=np.float64, copy=True)
    l_a = lows.to_numpy(dtype=np.float64, copy=True)
    c_a = closes.to_numpy(dtype=np.float64, copy=True)
    v_a = vols.to_numpy(dtype=np.float64, copy=True)
    w_a = vwaps.to_numpy(dtype=np.float64, copy=True)
    o_t, h_t = o_a.copy(), h_a.copy()
    l_t, c_t = l_a.copy(), c_a.copy()
    v_t, w_t = v_a.copy(), w_a.copy()

    n_patch = 0
    for d0 in days:
        loc = loc_of.get(d0)
        if loc is None:
            continue
        f = ddir / f"{d0.replace('-', '')}.parquet"
        if not f.exists():
            continue
        px0 = pd.read_parquet(f).set_index("ts_code")
        g = trunc.loc[trunc["date"] == d0].set_index("stem")
        for j, inst in enumerate(insts):
            ts = inst_ts(inst)
            stem = inst_stem(inst)
            if ts not in px0.index or stem not in g.index:
                continue
            raw_c = float(px0.loc[ts, "close"])
            q_c = c_a[loc, j]
            if raw_c <= 0 or not np.isfinite(q_c) or q_c <= 0:
                continue
            ratio = q_c / raw_c
            row = g.loc[stem]
            o_t[loc, j] = float(row["open"]) * ratio
            h_t[loc, j] = float(row["high"]) * ratio
            l_t[loc, j] = float(row["low"]) * ratio
            c_t[loc, j] = float(row["close"]) * ratio
            v_t[loc, j] = float(row["volume"])
            w_t[loc, j] = float(row["vwap"]) * ratio
            n_patch += 1
    print(f"覆盖截断 {n_patch} 个股票日")

    feat_c, feat_t, y_list = [], [], []
    roc_ov, kmid_sp, close_sp = [], [], []
    feat_sp = []
    per_feat = {n: [] for n in names}
    for d0, d in zip(days, rebal):
        loc = loc_of.get(d0)
        if loc is None or loc < 60:
            continue
        sc = scores.loc[d].reindex(insts)
        ok = sc.notna().to_numpy() & np.isfinite(c_a[loc])
        ok &= np.isfinite(c_t[loc]) & (c_a[loc] > 0) & (c_t[loc] > 0)
        if ok.sum() < 50:
            continue
        fc = _stack(
            _feat_at(o_a, h_a, l_a, c_a, v_a, w_a, loc),
            names, insts,
        )
        ft = _stack(
            _feat_at(o_t, h_t, l_t, c_t, v_t, w_t, loc),
            names, insts,
        )
        fc, ft = fc.iloc[ok], ft.iloc[ok]
        sc = sc.iloc[ok]
        feat_c.append(fc)
        feat_t.append(ft)
        y_list.append(sc)
        sps = []
        for n in names:
            sp = fc[n].corr(ft[n], method="spearman")
            if sp == sp:
                sps.append(float(sp))
                per_feat[n].append(float(sp))
        feat_sp.append(float(np.mean(sps)) if sps else np.nan)
        close_sp.append(pd.Series(c_a[loc, ok]).corr(
            pd.Series(c_t[loc, ok]), method="spearman",
        ))
        kmid_sp.append(fc["KMID"].corr(ft["KMID"], method="spearman"))
        r_c = fc["ROC5"].sort_values(ascending=False).index
        r_t = ft["ROC5"].sort_values(ascending=False).index
        roc_ov.append(_overlap(r_c, r_t, TOPK))
        print(f"  {d0} n={ok.sum()} feat_sp={feat_sp[-1]:.4f} "
              f"ROC5_Top{TOPK}={roc_ov[-1]:.2%}", flush=True)

    print(f"收盘 vs 14:45 价格秩相关  {_mu(close_sp):.4f}")
    print(f"KMID 秩相关                {_mu(kmid_sp):.4f}")
    print(f"158 因子截面秩相关均值     {_mu(feat_sp):.4f}")
    print(f"ROC5 Top{TOPK} 重叠        {_mu(roc_ov):.2%}")
    worst = sorted(
        ((n, _mu(vs)) for n, vs in per_feat.items() if vs),
        key=lambda x: x[1],
    )[:8]
    print("最不稳的 8 个因子:")
    for n, v in worst:
        print(f"  {n:10s} {_mu(per_feat[n]):.4f}")

    x_c = pd.concat(feat_c)
    x_t = pd.concat(feat_t)
    y = pd.concat(y_list).astype(float)
    x_c = x_c.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    x_t = x_t.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    import lightgbm as lgb
    mdl = lgb.LGBMRegressor(
        n_estimators=800,
        num_leaves=64,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        n_jobs=2,
        verbosity=-1,
    )
    mdl.fit(x_c, y)
    p_c = pd.Series(mdl.predict(x_c), index=x_c.index)
    p_t = pd.Series(mdl.predict(x_t), index=x_t.index)
    ov_surr, ov_true, sp_surr, ov_same = [], [], [], []
    off = 0
    for fc in feat_c:
        n = len(fc)
        idx = fc.index
        a = y.iloc[off:off + n]
        b = p_c.iloc[off:off + n]
        c = p_t.iloc[off:off + n]
        off += n
        ra = a.sort_values(ascending=False).index
        rb = pd.Series(b.values, index=idx).sort_values(
            ascending=False,
        ).index
        rc = pd.Series(c.values, index=idx).sort_values(
            ascending=False,
        ).index
        ov_true.append(_overlap(ra, rb, TOPK))
        ov_surr.append(_overlap(ra, rc, TOPK))
        ov_same.append(_overlap(rb, rc, TOPK))
        sp_surr.append(pd.Series(b.values).corr(
            pd.Series(c.values), method="spearman",
        ))
    print(f"代理拟合 Top{TOPK} 重叠(应高) {_mu(ov_true):.2%}")
    print(f"代理分数秩相关 截断 vs 完整  {_mu(sp_surr):.4f}")
    print(f"同一代理 Top{TOPK} 截断 vs 完整 {_mu(ov_same):.2%}")
    print(f"代理 Top{TOPK} 重叠 截断 vs 原分 {_mu(ov_surr):.2%}")
    mean_sp = _mu(feat_sp)
    mimic_ok = _mu(ov_true) >= 0.80
    mean_ov = _mu(ov_same) if mimic_ok else _mu(roc_ov)
    if mean_sp >= 0.95 and mean_ov >= 0.80:
        print("结论: 截断特征够稳, 下一步才是改标签并重训。")
    elif mean_sp < 0.85 or mean_ov < 0.50:
        print("结论: 截断后截面重排明显, 维持 T+1 收盘宽仓。")
    else:
        print("结论: 中间地带, 先报数字不重训。")
    if not mimic_ok:
        print("说明: 原 LightGBM 未落盘, 代理拟合不足, "
              "TopK 以因子重叠为准。")


if __name__ == "__main__":
    main()
