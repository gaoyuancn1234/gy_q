"""日线 bar 的语义校验 —— 量纲/一致性错误必须当场炸。

为什么要这个
------------
2026-09-12 踩到: Tushare 分钟 `vol` 的单位是**手**, Qlib `$volume` 是**股**,
`trunc_daily` 直接求和后 volume 小 100 倍、vwap 大 100 倍。这两个字段由
auction_infer 覆盖进信号日喂给 Alpha158 —— 量因子整组失真、VWAP0 从 ~1.0
变成 ~100, 而**全程不报错**。回测走 Qlib 日线, 看不到; 只有实盘会中招。

根因是 float 不带语义: 7.37e6 和 7.37e4 在类型上没区别。借 vn.py 的
BarData 思路, 在数据进模型的边界上做一次显式校验。不引入框架, 只要断言。

最有效的一条是 amount/volume ≈ close —— 它直接把单位错误暴露出来,
差 100 倍时这个比值会差 100 倍, 没法蒙混。
"""
from __future__ import annotations


class BarUnitError(ValueError):
    """bar 的量纲或内部一致性不对。"""


# amount/volume 相对 close 的容忍区间。日内均价偏离收盘价通常在几个点内,
# 放宽到 ±25% 仍能抓住 100 倍的量纲错。
VWAP_LO, VWAP_HI = 0.75, 1.25


def check_daily_bar(*, code: str, date: str, open_: float, high: float,
                    low: float, close: float, volume: float,
                    amount: float, vwap: float | None = None,
                    strict: bool = True) -> list[str]:
    """校验一根日线。strict=True 时不合格直接抛 BarUnitError。

    Returns:
        问题描述列表(strict=False 时用于汇总)。
    """
    bad = []
    px = (open_, high, low, close)
    if any(p is None or not (p == p) or p <= 0 for p in px):
        bad.append(f'价格非正或 NaN: O{open_} H{high} L{low} C{close}')
    else:
        if not (low <= min(open_, close) and max(open_, close) <= high):
            bad.append(f'OHLC 越界: O{open_} H{high} L{low} C{close}')

    if volume is None or not (volume == volume) or volume < 0:
        bad.append(f'volume 非法: {volume}')
    if amount is None or not (amount == amount) or amount < 0:
        bad.append(f'amount 非法: {amount}')

    # 核心: amount/volume 必须落在 close 附近。单位错 100 倍会立刻现形。
    if not bad and volume > 0 and amount > 0:
        implied = amount / volume
        r = implied / close
        if not (VWAP_LO <= r <= VWAP_HI):
            bad.append(
                f'量纲可疑: amount/volume={implied:.4g} 而 close={close:.4g} '
                f'(比值 {r:.4g}, 应在 {VWAP_LO}~{VWAP_HI})。'
                f'常见原因: volume 用了「手」而非「股」(差 100 倍)')

    if vwap is not None and vwap == vwap and close > 0:
        r = vwap / close
        if not (VWAP_LO <= r <= VWAP_HI):
            bad.append(
                f'vwap 可疑: {vwap:.4g} / close {close:.4g} = {r:.4g}')

    if bad and strict:
        raise BarUnitError(f'{code} {date}: ' + '; '.join(bad))
    return bad


def check_frame(df, *, code_col='stem', date_col='date',
                sample: int = 200) -> list[str]:
    """抽样校验一个 DataFrame。返回问题列表(不抛)。"""
    problems = []
    n = len(df)
    step = max(1, n // sample) if sample else 1
    for i in range(0, n, step):
        r = df.iloc[i]
        problems += [
            f"{r.get(code_col, '?')} {r.get(date_col, '?')}: {m}"
            for m in check_daily_bar(
                code=str(r.get(code_col, '?')), date=str(r.get(date_col, '?')),
                open_=float(r['open']), high=float(r['high']),
                low=float(r['low']), close=float(r['close']),
                volume=float(r['volume']), amount=float(r['amount']),
                vwap=float(r['vwap']) if 'vwap' in r else None,
                strict=False)
        ]
    return problems
