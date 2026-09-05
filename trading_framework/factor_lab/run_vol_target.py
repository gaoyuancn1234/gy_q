#!/usr/bin/env python3
"""波动率目标 (vol targeting) 实验

诊断依据 (2026-08-30):
  段2 (2026-02~08) 的选股能力并未衰减 —— TopK 超额日均 +0.0921%，与 2024 的
  +0.1020% 相当；但 TopK 组合日波动从 1.14% 涨到 2.15%，翻了一倍。
  同样的 alpha 承担两倍风险 → Sharpe 崩塌、回撤从 -13% 扩大到 -30%。

  根因: TopK 等权是固定风险敞口，市场波动翻倍时组合风险跟着翻倍，
        没有任何缓冲。现有的自适应 TopK 由 quality_score(信号质量) 驱动，
        而段2 的信号质量正常，所以不会触发。

做法: 按近期实现波动率反向缩放权益敞口，其余现金留存。
      exposure = clip(target_vol / realized_vol, min_exp, 1.0)
      不加杠杆 (上限 1.0)，符合散户 A 股账户实际。

用法:
    python -m factor_lab.run_vol_target                    # 默认扫描
    python -m factor_lab.run_vol_target --target-vols 0.15 0.20
"""
import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

RESULTS_DIR = PROJECT_DIR / "factor_lab" / "results" / "vol_target"
PRED_DIR = PROJECT_DIR / "factor_lab" / "results" / "rolling" / "predictions"

TRADING_DAYS = 242


class VolTargetBacktester:
    """收盘价回测器 + 波动率目标仓位缩放

    与 run_capital_impact.CapitalBacktester 保持相同的执行约束
    (T+1 收盘成交 / 整手 100 股 / 涨跌停 ±9.5% / 双边成本 / 止损)，
    唯一新增的是 exposure 缩放，因此两者结果可直接对比。
    """

    def __init__(self, signal: pd.Series,
                 initial_cash: float = 100_000,
                 topk: int = 12,
                 rebalance_every: int = 5,
                 stop_loss: float = 0.08,
                 open_cost: float = 0.0005,
                 close_cost: float = 0.0015,
                 target_vol: float | None = None,
                 vol_window: int = 20,
                 min_exposure: float = 0.2,
                 n_drop: int | None = None,
                 exclude_st: bool = True):
        self.signal = signal
        self.initial_cash = initial_cash
        self.topk = topk
        self.rebalance_every = rebalance_every
        self.stop_loss = stop_loss
        self.open_cost = open_cost
        self.close_cost = close_cost
        self.target_vol = target_vol      # None = 不做 vol targeting (基准)
        self.vol_window = vol_window
        self.min_exposure = min_exposure
        # 每次调仓最多替换几只 (类似 qlib TopkDropout 的 n_drop)。
        # None = 全量替换。全量换手在 5 日调仓下几乎每次换掉全部 12 只，
        # 双边成本 0.05%+0.15% 累积起来会吃掉大部分收益。
        self.n_drop = n_drop
        # 排除 ST 股。沪深300 里 ST 极罕见，但中证500/800 概率高得多:
        # ST 股涨跌停限制是 ±5% 而非 ±10%、流动性差、退市风险高。
        # 用日线 isST 字段(时点状态，无前视)在选股时剔除。
        self.exclude_st = exclude_st

    def _exposure(self, recent_returns: list[float]) -> float:
        """由近期实现波动率决定权益敞口

        2026-09-04 修复: 原先只有 target_vol is None 才视为关闭，而 0 是合法数值，
        会走到 clip(0/vol, min_exposure, 1.0) = min_exposure ——
        命令行传 --target-vols 0 想表达"关闭"，实际得到"固定 20% 仓位"。
        实测该配置敞口 21.5%、收益 21.77%，被当成"不启用"的对照组写进汇总表，
        任何与之比较的结论都是错的。

        2026-09-05: 改为复用 portfolio/rebalance_rules.compute_exposure，
        与实盘/模拟盘同一实现。此前这里是第三份独立实现，生产已改为
        "波动率估不出来时收缩到 UNKNOWN_VOL_EXPOSURE" 而这里仍是"满仓"，
        回测会系统性高估实际敞口 —— 与 n_drop/vol_target 曾在模拟盘缺失
        属同一类分叉。
        """
        from portfolio.rebalance_rules import compute_exposure as _ce
        # 共用实现吃的是净值序列，这里把日收益累乘还原成净值
        navs, v = [1.0], 1.0
        for r in recent_returns:
            v *= (1.0 + r)
            navs.append(v)
        exposure, _vol = _ce(navs, self.target_vol,
                             window=self.vol_window,
                             min_exposure=self.min_exposure)
        return float(exposure)

    def run(self, prices_df: pd.DataFrame) -> dict:
        trading_days = sorted(prices_df.index.get_level_values(0).unique())
        signal_days = set(self.signal.index.get_level_values(0).unique())

        cash = self.initial_cash
        positions = {}          # {inst: {'shares', 'cost_price'}}
        pending_sells = []
        pending_buys = {}       # {inst: target_value}

        daily_values = []
        daily_returns = []
        exposures = []
        prev_close = {}
        prev_value = self.initial_cash
        total_trades = 0

        # 注意: qlib 对缺失字段不抛异常，而是返回全 NaN 列。只判断列是否存在
        # 会导致 ST 过滤"看似启用、实则一只都没过滤"——必须检查是否有真实值。
        has_st = ('$isST' in prices_df.columns
                  and bool(prices_df['$isST'].notna().any()))
        if self.exclude_st and not has_st:
            print("  [警告] 数据集无有效 $isST 数据，ST 过滤未生效。"
                  "当前数据源为新浪(akshare)，其日线接口不提供 ST 标记，"
                  "重新下载也不会有 —— 需 baostock 恢复后补该字段。")

        for day_idx, date in enumerate(trading_days):
            day_row = prices_df.loc[date]
            day_close = day_row['$close'].dropna()
            # 当日处于 ST 状态的标的 (时点状态，无前视)
            st_today = set()
            if self.exclude_st and has_st:
                st_today = set(day_row.index[day_row['$isST'] > 0])

            # --- 1. 挂起卖出 (今日收盘) ---
            for inst in list(pending_sells):
                if inst not in positions:
                    pending_sells.remove(inst)
                    continue
                if inst not in day_close.index:
                    continue
                if inst in prev_close and prev_close[inst] > 0:
                    if day_close[inst] / prev_close[inst] - 1 < -0.095:
                        continue        # 跌停卖不出
                price = day_close[inst]
                cash += positions[inst]['shares'] * price * (1 - self.close_cost)
                del positions[inst]
                pending_sells.remove(inst)
                total_trades += 1

            # --- 2. 挂起买入 (今日收盘, 整手) ---
            if pending_buys:
                # 保持插入顺序 (= 信号分数降序)。原先按 target_value 排序，
                # 但所有值都相同 (等权)，排序退化为不确定顺序。
                for inst, target_value in pending_buys.items():
                    if inst in positions or inst not in day_close.index:
                        continue
                    if inst in prev_close and prev_close[inst] > 0:
                        if day_close[inst] / prev_close[inst] - 1 > 0.095:
                            continue    # 涨停买不进
                    price = day_close[inst]
                    shares = int(target_value / (price * (1 + self.open_cost)) / 100) * 100
                    if shares < 100:
                        continue        # 整手买不起
                    cost = shares * price * (1 + self.open_cost)
                    if cost > cash * 0.99:
                        continue        # 现金不足
                    cash -= cost
                    positions[inst] = {'shares': shares, 'cost_price': float(price)}
                    total_trades += 1
                pending_buys = {}

            # --- 3. 收盘估值 ---
            port_value = cash
            for inst, pos in positions.items():
                px = day_close[inst] if inst in day_close.index else pos['cost_price']
                port_value += pos['shares'] * px
            daily_values.append({'date': date, 'value': port_value,
                                 'n_positions': len(positions)})
            daily_returns.append(port_value / prev_value - 1 if prev_value > 0 else 0.0)
            prev_value = port_value

            # --- 4. 止损 ---
            if self.stop_loss:
                for inst, pos in list(positions.items()):
                    if inst in day_close.index:
                        if day_close[inst] / pos['cost_price'] - 1 < -self.stop_loss:
                            if inst not in pending_sells:
                                pending_sells.append(inst)

            # --- 5. 调仓日 ---
            if day_idx % self.rebalance_every == 0 and date in signal_days:
                exposure = self._exposure(daily_returns)
                exposures.append(exposure)

                day_signal = self.signal.loc[date]
                if isinstance(day_signal, pd.DataFrame):
                    day_signal = day_signal.iloc[:, 0]
                full_rank = day_signal.sort_values(ascending=False)
                if st_today:
                    full_rank = full_rank[~full_rank.index.isin(st_today)]
                ranked = full_rank.head(self.topk)
                target_stocks = set(ranked.index)

                if self.n_drop is not None and positions:
                    # 限制换手: 只卖掉当前持仓中排名最差的 n_drop 只
                    # (且必须已跌出目标名单)，其余保留
                    rank_pos = {inst: i for i, inst in enumerate(full_rank.index)}
                    held = [i for i in positions if i not in pending_sells]
                    # 排名越靠后越该卖; 不在信号里的视为最差
                    worst = sorted(held, key=lambda i: -rank_pos.get(i, 10**9))
                    drops = [i for i in worst if i not in target_stocks][:self.n_drop]
                    for inst in drops:
                        pending_sells.append(inst)
                else:
                    # 卖出不在目标内的 (全量替换)
                    for inst in list(positions.keys()):
                        if inst not in target_stocks and inst not in pending_sells:
                            pending_sells.append(inst)

                # 目标持仓市值 = 组合总值 × exposure / topk
                # 已持有的不动 (避免过度换手)，只补买缺口
                per_slot = port_value * exposure / self.topk
                current_holds = set(positions.keys()) - set(pending_sells)
                # 必须按信号分数排序: 用集合差集会带来不确定的迭代顺序
                # (Python 字符串哈希随机化)，现金耗尽时买到的股票随机变化，
                # 同一配置两次运行结果不同 —— 参数对比将建立在噪声之上。
                # 按分数降序也是经济上正确的选择: 优先买高置信度标的。
                to_buy = [i for i in ranked.index if i not in current_holds]
                if self.n_drop is not None:
                    # 只买入被腾出的坑位数，保持持仓数稳定在 topk
                    free_slots = self.topk - len(current_holds)
                    to_buy = to_buy[:max(0, free_slots)]
                if to_buy:
                    pending_buys = {inst: per_slot for inst in to_buy}

            for inst in day_close.index:
                prev_close[inst] = day_close[inst]

        return self._metrics(daily_values, total_trades, exposures)

    def _metrics(self, daily_values, n_trades, exposures) -> dict:
        df = pd.DataFrame(daily_values).set_index('date')
        rets = df['value'].pct_change().dropna()
        total_ret = df['value'].iloc[-1] / self.initial_cash - 1
        n_days = len(df)
        ann_ret = (1 + total_ret) ** (TRADING_DAYS / max(n_days, 1)) - 1
        sharpe = (rets.mean() / rets.std() * np.sqrt(TRADING_DAYS)) if rets.std() > 0 else 0.0
        cummax = df['value'].cummax()
        mdd = ((df['value'] - cummax) / cummax).min()
        return {
            'total_return': float(total_ret),
            'annual_return': float(ann_ret),
            'sharpe': float(sharpe),
            'max_drawdown': float(mdd),
            # 日收益序列: DSR 多重检验校正需要序列而非汇总指标
            'daily_returns': [float(x) for x in rets.tolist()],
            'realized_vol': float(rets.std() * np.sqrt(TRADING_DAYS)),
            'n_trades': int(n_trades),
            'avg_positions': float(df['n_positions'].mean()),
            'avg_exposure': float(np.mean(exposures)) if exposures else 1.0,
            'min_exposure_used': float(np.min(exposures)) if exposures else 1.0,
            'n_days': int(n_days),
        }


def load_predictions(tags: list[str]) -> pd.Series:
    """合并多个 tag 的预测缓存"""
    parts = []
    for t in tags:
        name = "D_expand_3v_3r_alpha158_val_LightGBM"
        p = PRED_DIR / (f"{name}_{t}.pkl" if t else f"{name}.pkl")
        if p.exists():
            parts.append(pd.read_pickle(p))
        else:
            print(f"  [跳过] 预测缓存不存在: {p.name}")
    if not parts:
        raise FileNotFoundError("没有可用的预测缓存")
    s = pd.concat(parts)
    return s[~s.index.duplicated(keep='last')].sort_index()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="波动率目标实验")
    parser.add_argument('--target-vols', nargs='+', type=float,
                        default=[0.12, 0.15, 0.20, 0.25],
                        help="年化目标波动率 (默认扫描 0.12~0.25)")
    parser.add_argument('--capital', type=float, default=100_000)
    parser.add_argument('--topk', type=int, default=12)
    parser.add_argument('--vol-window', type=int, default=20)
    parser.add_argument('--pred-tags', nargs='+', default=['', 'seg2pred'],
                        help="要合并的预测缓存 tag")
    parser.add_argument('--n-drops', nargs='+', default=['none'],
                        help="每次调仓最多替换几只; 'none'=全量替换 (默认)")
    parser.add_argument('--universe', default='csi300',
                        help="股票池 instruments 名 (csi300/csi500/csi800)")
    parser.add_argument('--provider-uri', default='~/.qlib/qlib_data/cn_data_bs',
                        help="qlib 数据目录")
    parser.add_argument('--rebalance-every', nargs='+', type=int, default=[5],
                        help="调仓间隔(交易日); 1=每日调仓 (默认 5)")
    args = parser.parse_args()

    import qlib
    from qlib.data import D
    from qlib.constant import REG_CN
    qlib.init(provider_uri=args.provider_uri, region=REG_CN)

    print("=" * 78)
    print("  波动率目标实验 — 收盘价 / T+1 / 整手 / 涨跌停 / 8%止损")
    print("=" * 78)

    signal = load_predictions(args.pred_tags)
    dts = signal.index.get_level_values(0)
    start, end = dts.min(), dts.max()
    print(f"\n预测区间: {start.date()} ~ {end.date()} ({dts.nunique()} 交易日)")
    print(f"资金: {args.capital:,.0f} | TopK: {args.topk} | vol窗口: {args.vol_window}日")

    # 尝试带上 $isST 以支持 ST 过滤; 旧数据集没有该字段时优雅降级
    fields = ['$close', '$isST']
    try:
        prices = D.features(D.instruments(args.universe), fields,
                            start_time=start.strftime('%Y-%m-%d'),
                            end_time=end.strftime('%Y-%m-%d'))
    except Exception:
        print("  [提示] 数据集无 $isST 字段，回退为仅加载 $close (ST 过滤不生效)")
        prices = D.features(D.instruments(args.universe), ['$close'],
                            start_time=start.strftime('%Y-%m-%d'),
                            end_time=end.strftime('%Y-%m-%d'))
    prices = prices.swaplevel().sort_index()

    # 分段评估: 揭示不同波动环境下的表现
    segments = [
        ('全期', None, None),
        ('2024 (低波动)', '2024-01-01', '2025-01-01'),
        ('2025', '2025-01-01', '2026-01-01'),
        ('2026-02~08 (高波动)', '2026-02-06', '2026-09-01'),
    ]

    nd_list = [None if str(x).lower() == 'none' else int(x) for x in args.n_drops]
    configs = []
    for rb in args.rebalance_every:
        rb_tag = f' 调仓{rb}日'
        for nd in nd_list:
            nd_tag = '' if nd is None else f' n_drop={nd}'
            configs.append((f'基准{nd_tag}{rb_tag}', None, nd, rb))
            configs += [(f'tv={v:.0%}{nd_tag}{rb_tag}', v, nd, rb)
                        for v in args.target_vols]

    all_results = {}
    for label, tv, nd, rb in configs:
        print(f"\n--- {label} ---")
        row = {}
        for seg_name, a, b in segments:
            sig = signal
            pr = prices
            if a:
                mask = (sig.index.get_level_values(0) >= a) & (sig.index.get_level_values(0) < b)
                sig = sig[mask]
                pm = (pr.index.get_level_values(0) >= a) & (pr.index.get_level_values(0) < b)
                pr = pr[pm]
            if len(sig) == 0:
                continue
            bt = VolTargetBacktester(sig, initial_cash=args.capital, topk=args.topk,
                                     target_vol=tv, vol_window=args.vol_window,
                                     n_drop=nd, rebalance_every=rb)
            r = bt.run(pr)
            row[seg_name] = r
            print(f"  {seg_name:22s} Sharpe={r['sharpe']:6.3f}  "
                  f"Ret={r['total_return']:8.2%}  MDD={r['max_drawdown']:8.2%}  "
                  f"敞口={r['avg_exposure']:5.1%}  交易={r['n_trades']:5d}  "
                  f"持仓={r['avg_positions']:4.1f}")
        all_results[label] = row

    # 汇总表
    print("\n" + "=" * 78)
    print("  汇总: Sharpe / 最大回撤")
    print("=" * 78)
    seg_names = [s[0] for s in segments]
    print(f"{'配置':>22} " + " ".join(f"{s:>20}" for s in seg_names))
    for label, row in all_results.items():
        cells = []
        for s in seg_names:
            r = row.get(s)
            cells.append(f"{r['sharpe']:6.2f}/{r['max_drawdown']:7.1%}" if r else " " * 14)
        print(f"{label:>22} " + " ".join(f"{c:>20}" for c in cells))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "vol_target_results.json"
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {out}")


if __name__ == '__main__':
    main()
