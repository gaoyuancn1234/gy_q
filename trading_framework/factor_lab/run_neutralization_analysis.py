#!/usr/bin/env python3
"""
实验 007: 行业/市值中性化分析

分析当前最佳策略 (D_expand_3v_3r + LightGBM + alpha158_val, Sharpe=2.202) 的收益来源:
- 行业/市值暴露度分析
- Brinson 收益归因分解 (行业配置 vs 个股选择)
- 中性化对比回测 (行业中性化 / 行业+市值中性化)

用法:
  python -m factor_lab.run_neutralization_analysis
  python -m factor_lab.run_neutralization_analysis --topk 20
  python -m factor_lab.run_neutralization_analysis --skip-backtest  # 仅分析, 不跑回测
"""
import sys
import json
import time
import pickle
import warnings
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

# 与 run_rolling_benchmark.py 一致
TEST_START = '2024-01-01'
TEST_END = '2026-02-05'
TOPK = 12
N_DROP = 3

CACHE_DIR = PROJECT_DIR / "factor_lab" / "results" / "rolling" / "predictions"
RESULTS_DIR = PROJECT_DIR / "factor_lab" / "results" / "neutralization"


# ============================================================
# Step 1: 获取行业分类数据
# ============================================================

def download_industry_map() -> pd.DataFrame:
    """从 BaoStock 下载 CSI300 成分股的申万一级行业分类

    Returns:
        DataFrame with columns: [instrument, industry, industry_code]
        instrument 格式: SH600519 / SZ000001
    """
    cache_path = RESULTS_DIR / ".cache" / "industry_map.csv"
    if cache_path.exists():
        print(f"  [缓存] 行业分类数据: {cache_path}")
        return pd.read_csv(cache_path)

    import baostock as bs

    print("  [下载] 从 BaoStock 获取行业分类...")
    lg = bs.login()
    if lg.error_code != '0':
        raise RuntimeError(f"BaoStock login failed: {lg.error_msg}")

    try:
        # 获取沪深300成分股
        rs = bs.query_hs300_stocks()
        members = []
        while rs.next():
            members.append(rs.get_row_data())
        hs300 = pd.DataFrame(members, columns=rs.fields)

        # 去重 (成分股列表可能有多条记录)
        codes = hs300['code'].unique()
        print(f"  沪深300成分股: {len(codes)} 只")

        # 查询每只股票的行业分类
        results = []
        for code in codes:
            rs = bs.query_stock_industry(code=code)
            while rs.next():
                results.append(rs.get_row_data())

        industry_df = pd.DataFrame(results, columns=rs.fields)

        # 转换为 Qlib instrument 格式: sh.600519 → SH600519
        def to_qlib_instrument(code):
            parts = code.split('.')
            return parts[0].upper() + parts[1]

        industry_df['instrument'] = industry_df['code'].apply(to_qlib_instrument)
        industry_df = industry_df[['instrument', 'industry', 'industryClassification']].copy()
        industry_df.columns = ['instrument', 'industry', 'industry_code']

        # 去重 (保留最新一条)
        industry_df = industry_df.drop_duplicates(subset='instrument', keep='last')
        print(f"  行业覆盖: {industry_df['industry'].nunique()} 个申万一级行业")

    finally:
        bs.logout()

    # 缓存
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    industry_df.to_csv(cache_path, index=False)
    print(f"  [保存] {cache_path}")

    return industry_df


# ============================================================
# Step 2: 持仓暴露分析
# ============================================================

def get_market_cap_data(instruments, start_date, end_date) -> pd.DataFrame:
    """从 Qlib 获取市值数据

    Returns:
        DataFrame with MultiIndex (datetime, instrument), column: total_mv
    """
    import qlib
    from qlib.data import D

    if not qlib.auto_init():
        qlib.init(provider_uri='~/.qlib/qlib_data/cn_data_bs')

    inst = D.instruments('csi300')
    mv_df = D.features(inst, ['$total_mv'], start_time=start_date, end_time=end_date)
    mv_df.columns = ['total_mv']
    return mv_df


def get_price_data(instruments, start_date, end_date) -> pd.DataFrame:
    """从 Qlib 获取价格数据 (open, close)"""
    import qlib
    from qlib.data import D

    if not qlib.auto_init():
        qlib.init(provider_uri='~/.qlib/qlib_data/cn_data_bs')

    inst = D.instruments('csi300')
    price_df = D.features(inst, ['$open', '$close'], start_time=start_date, end_time=end_date)
    price_df.columns = ['open', 'close']
    return price_df


def analyze_portfolio_exposure(pred: pd.Series, industry_map: pd.DataFrame,
                                mv_data: pd.DataFrame, topk: int = 12) -> dict:
    """分析每日 topk 持仓的行业/市值分布

    Args:
        pred: 预测信号, MultiIndex (datetime, instrument)
        industry_map: 行业分类 DataFrame
        mv_data: 市值数据, MultiIndex (datetime, instrument)
        topk: 选股数量

    Returns:
        dict with exposure analysis results
    """
    print("\n=== Step 2: 持仓暴露分析 ===")

    # 构建行业查找表
    ind_map = industry_map.set_index('instrument')['industry'].to_dict()

    # 每日选出 topk
    dates = pred.index.get_level_values(0).unique()
    print(f"  分析区间: {dates[0].date()} ~ {dates[-1].date()}, 共 {len(dates)} 个交易日")

    daily_industry_weights = []  # 每日行业权重
    daily_mv_stats = []          # 每日市值统计
    daily_portfolio = []         # 每日持仓

    for dt in dates:
        day_pred = pred.loc[dt].dropna().sort_values(ascending=False)
        if len(day_pred) < topk:
            continue

        top_stocks = day_pred.head(topk).index.tolist()
        daily_portfolio.append({'date': dt, 'stocks': top_stocks})

        # 行业分布
        industries = [ind_map.get(s, '未知') for s in top_stocks]
        ind_counts = pd.Series(industries).value_counts(normalize=True)
        daily_industry_weights.append({'date': dt, **ind_counts.to_dict()})

        # 市值统计
        if dt in mv_data.index.get_level_values(0):
            try:
                day_mv = mv_data.loc[dt]
                top_mv = day_mv.loc[day_mv.index.isin(top_stocks), 'total_mv']
                all_mv = day_mv['total_mv'].dropna()
                if len(top_mv) > 0 and len(all_mv) > 0:
                    daily_mv_stats.append({
                        'date': dt,
                        'portfolio_log_mv_mean': float(np.log(top_mv.clip(lower=1)).mean()),
                        'portfolio_log_mv_median': float(np.log(top_mv.clip(lower=1)).median()),
                        'universe_log_mv_mean': float(np.log(all_mv.clip(lower=1)).mean()),
                        'universe_log_mv_median': float(np.log(all_mv.clip(lower=1)).median()),
                        'portfolio_mv_mean': float(top_mv.mean()),
                        'universe_mv_mean': float(all_mv.mean()),
                    })
            except (KeyError, IndexError):
                pass

    # 汇总行业分析
    ind_weights_df = pd.DataFrame(daily_industry_weights).set_index('date').fillna(0)

    # 1. 行业 HHI 指数 (每日)
    daily_hhi = (ind_weights_df ** 2).sum(axis=1)
    avg_hhi = float(daily_hhi.mean())

    # 2. 基准行业权重 (等权近似: 每个行业按成分股数量)
    all_industries = [ind_map.get(s, '未知') for s in pred.index.get_level_values(1).unique()
                      if s in ind_map]
    benchmark_weights = pd.Series(all_industries).value_counts(normalize=True)

    # 3. 行业偏离度
    avg_portfolio_weights = ind_weights_df.mean()
    all_industries_set = set(avg_portfolio_weights.index) | set(benchmark_weights.index)
    deviation = 0.0
    industry_deviation_detail = {}
    for ind in all_industries_set:
        pw = avg_portfolio_weights.get(ind, 0)
        bw = benchmark_weights.get(ind, 0)
        dev = pw - bw
        deviation += abs(dev)
        industry_deviation_detail[ind] = {
            'portfolio_weight': round(float(pw), 4),
            'benchmark_weight': round(float(bw), 4),
            'deviation': round(float(dev), 4),
        }

    # 按偏离度排序
    industry_deviation_detail = dict(
        sorted(industry_deviation_detail.items(),
               key=lambda x: abs(x[1]['deviation']), reverse=True)
    )

    # 4. 单行业最大占比
    max_single_industry = float(ind_weights_df.max().max())
    max_single_industry_name = ind_weights_df.max().idxmax()

    # 5. 行业出现频次
    industry_frequency = {}
    for record in daily_industry_weights:
        for ind, w in record.items():
            if ind == 'date':
                continue
            if ind not in industry_frequency:
                industry_frequency[ind] = 0
            industry_frequency[ind] += 1
    industry_frequency = dict(sorted(industry_frequency.items(), key=lambda x: x[1], reverse=True))

    # 6. 市值暴露
    mv_stats_df = pd.DataFrame(daily_mv_stats)
    mv_exposure = {}
    if len(mv_stats_df) > 0:
        mv_exposure = {
            'portfolio_log_mv_mean': round(float(mv_stats_df['portfolio_log_mv_mean'].mean()), 4),
            'universe_log_mv_mean': round(float(mv_stats_df['universe_log_mv_mean'].mean()), 4),
            'log_mv_diff': round(float(mv_stats_df['portfolio_log_mv_mean'].mean()
                                       - mv_stats_df['universe_log_mv_mean'].mean()), 4),
            'portfolio_mv_mean_billion': round(float(mv_stats_df['portfolio_mv_mean'].mean()) / 1e8, 2),
            'universe_mv_mean_billion': round(float(mv_stats_df['universe_mv_mean'].mean()) / 1e8, 2),
        }

    result = {
        'hhi_index': round(avg_hhi, 4),
        'hhi_interpretation': '高集中度' if avg_hhi > 0.15 else '适度分散',
        'total_industry_deviation': round(float(deviation), 4),
        'max_single_industry_weight': round(max_single_industry, 4),
        'max_single_industry_name': max_single_industry_name,
        'n_industries_in_portfolio': len([v for v in industry_frequency.values()]),
        'n_industries_total': len(benchmark_weights),
        'industry_deviation_detail': industry_deviation_detail,
        'industry_frequency_top10': dict(list(industry_frequency.items())[:10]),
        'market_cap_exposure': mv_exposure,
    }

    # 打印摘要
    print(f"\n  行业 HHI 指数: {avg_hhi:.4f} ({result['hhi_interpretation']})")
    print(f"  行业偏离度 (L1): {deviation:.4f}")
    print(f"  单行业最大占比: {max_single_industry:.1%} ({max_single_industry_name})")
    print(f"  组合覆盖行业数: {result['n_industries_in_portfolio']}/{result['n_industries_total']}")
    if mv_exposure:
        print(f"  组合平均市值: {mv_exposure['portfolio_mv_mean_billion']:.0f} 亿")
        print(f"  全市场平均市值: {mv_exposure['universe_mv_mean_billion']:.0f} 亿")
        print(f"  log(市值)偏差: {mv_exposure['log_mv_diff']:+.4f}")

    print(f"\n  行业偏离 Top5:")
    for i, (ind, detail) in enumerate(industry_deviation_detail.items()):
        if i >= 5:
            break
        print(f"    {ind}: 组合 {detail['portfolio_weight']:.1%} vs 基准 {detail['benchmark_weight']:.1%}"
              f"  (偏离 {detail['deviation']:+.1%})")

    return result


# ============================================================
# Step 3: Brinson 收益归因
# ============================================================

def decompose_returns(pred: pd.Series, industry_map: pd.DataFrame,
                      price_data: pd.DataFrame, topk: int = 12) -> dict:
    """Brinson 收益归因分解: R_total = R_allocation + R_selection + R_interaction

    简化版 Brinson-Fachler 归因:
    - R_allocation: 行业权重偏离基准带来的收益 (超配涨的行业)
    - R_selection: 行业内选股 alpha (选到行业内更好的股票)
    - R_interaction: 交互项

    按月汇总输出
    """
    print("\n=== Step 3: Brinson 收益归因 ===")

    ind_map = industry_map.set_index('instrument')['industry'].to_dict()

    # 使用 open 价格计算 T+1 收益 (与回测一致)
    # 收益 = next_day_open / current_day_open - 1
    if price_data.index.names != ['datetime', 'instrument']:
        if price_data.index.names == ['instrument', 'datetime']:
            price_data = price_data.swaplevel()
            price_data.sort_index(inplace=True)

    dates = pred.index.get_level_values(0).unique().sort_values()

    daily_attribution = []

    for i in range(len(dates) - 1):
        dt = dates[i]
        dt_next = dates[i + 1]

        # 当日信号 → 选出 topk
        day_pred = pred.loc[dt].dropna().sort_values(ascending=False)
        if len(day_pred) < topk:
            continue

        top_stocks = day_pred.head(topk).index.tolist()
        all_stocks = day_pred.index.tolist()

        # 次日开盘价收益
        try:
            today_open = price_data.loc[dt, 'open']
            next_open = price_data.loc[dt_next, 'open']
        except KeyError:
            continue

        # 计算所有股票的日收益
        returns = (next_open / today_open - 1).dropna()

        # 构建行业分组
        stock_industries = {s: ind_map.get(s, '未知') for s in returns.index}

        # 获取所有行业列表
        all_industries = list(set(stock_industries.values()))

        # 基准: 等权所有可用股票
        benchmark_return = returns.mean()

        # 组合: 等权 topk
        portfolio_stocks_with_ret = [s for s in top_stocks if s in returns.index]
        if len(portfolio_stocks_with_ret) == 0:
            continue
        portfolio_return = returns[portfolio_stocks_with_ret].mean()

        # 按行业分解
        r_allocation = 0.0
        r_selection = 0.0
        r_interaction = 0.0

        for ind in all_industries:
            # 基准中该行业的股票
            bench_stocks_in_ind = [s for s in all_stocks if stock_industries.get(s) == ind
                                   and s in returns.index]
            # 组合中该行业的股票
            port_stocks_in_ind = [s for s in portfolio_stocks_with_ret
                                  if stock_industries.get(s) == ind]

            if len(bench_stocks_in_ind) == 0:
                continue

            # 权重
            w_bench = len(bench_stocks_in_ind) / len([s for s in all_stocks if s in returns.index])
            w_port = len(port_stocks_in_ind) / len(portfolio_stocks_with_ret) if portfolio_stocks_with_ret else 0

            # 行业收益
            r_bench_ind = returns[bench_stocks_in_ind].mean()
            r_port_ind = returns[port_stocks_in_ind].mean() if port_stocks_in_ind else 0

            # Brinson 分解
            r_allocation += (w_port - w_bench) * r_bench_ind
            r_selection += w_bench * (r_port_ind - r_bench_ind)
            r_interaction += (w_port - w_bench) * (r_port_ind - r_bench_ind)

        daily_attribution.append({
            'date': dt,
            'total_return': float(portfolio_return - benchmark_return),
            'allocation': float(r_allocation),
            'selection': float(r_selection),
            'interaction': float(r_interaction),
            'portfolio_return': float(portfolio_return),
            'benchmark_return': float(benchmark_return),
        })

    attr_df = pd.DataFrame(daily_attribution)
    attr_df['date'] = pd.to_datetime(attr_df['date'])
    attr_df.set_index('date', inplace=True)

    # 按月汇总
    monthly = attr_df.resample('M').sum()

    # 按季度汇总
    quarterly = attr_df.resample('Q').sum()

    # 总体汇总
    total = attr_df.sum()
    total_abs = attr_df[['allocation', 'selection', 'interaction']].abs().sum()

    # 使用绝对值占比 (各分项正负抵消时百分比会失真)
    abs_sum = float(total_abs.sum())
    alloc_abs_pct = round(float(total_abs['allocation'] / abs_sum * 100), 1) if abs_sum > 0 else 0
    sel_abs_pct = round(float(total_abs['selection'] / abs_sum * 100), 1) if abs_sum > 0 else 0
    inter_abs_pct = round(float(total_abs['interaction'] / abs_sum * 100), 1) if abs_sum > 0 else 0

    result = {
        'total': {
            'excess_return': round(float(total['total_return']), 6),
            'allocation': round(float(total['allocation']), 6),
            'selection': round(float(total['selection']), 6),
            'interaction': round(float(total['interaction']), 6),
            'allocation_abs_pct': alloc_abs_pct,
            'selection_abs_pct': sel_abs_pct,
            'interaction_abs_pct': inter_abs_pct,
        },
        'monthly': {str(k.date()): {
            'allocation': round(float(v['allocation']), 6),
            'selection': round(float(v['selection']), 6),
            'interaction': round(float(v['interaction']), 6),
            'total': round(float(v['total_return']), 6),
        } for k, v in monthly.iterrows()},
        'quarterly': {str(k.date()): {
            'allocation': round(float(v['allocation']), 6),
            'selection': round(float(v['selection']), 6),
            'interaction': round(float(v['interaction']), 6),
            'total': round(float(v['total_return']), 6),
        } for k, v in quarterly.iterrows()},
        'daily_stats': {
            'n_days': len(attr_df),
            'allocation_mean': round(float(attr_df['allocation'].mean()), 6),
            'selection_mean': round(float(attr_df['selection'].mean()), 6),
            'interaction_mean': round(float(attr_df['interaction'].mean()), 6),
            'allocation_std': round(float(attr_df['allocation'].std()), 6),
            'selection_std': round(float(attr_df['selection'].std()), 6),
        },
    }

    # 打印摘要
    print(f"\n  分析天数: {len(attr_df)}")
    print(f"\n  === 总体归因 ===")
    print(f"  超额收益: {total['total_return']:.4%}")
    print(f"    行业配置 (Allocation): {total['allocation']:.4%} (|abs| 占比 {alloc_abs_pct:.1f}%)")
    print(f"    个股选择 (Selection):  {total['selection']:.4%} (|abs| 占比 {sel_abs_pct:.1f}%)")
    print(f"    交互项 (Interaction):  {total['interaction']:.4%} (|abs| 占比 {inter_abs_pct:.1f}%)")
    print(f"    归因合计: {total['allocation'] + total['selection'] + total['interaction']:.4%}")

    # 验证归因完整性
    residual = abs(total['total_return'] - total['allocation'] - total['selection'] - total['interaction'])
    if residual > 0.001:
        print(f"  ⚠ 归因残差偏大: {residual:.6f}")
    else:
        print(f"  归因残差: {residual:.6f} (OK)")

    print(f"\n  === 季度归因 ===")
    for k, v in quarterly.iterrows():
        print(f"  {k.strftime('%Y-Q%q').replace('Q%q', f'Q{(k.month-1)//3+1}')}: "
              f"配置 {v['allocation']:+.4f}  选股 {v['selection']:+.4f}  "
              f"交互 {v['interaction']:+.4f}  合计 {v['total_return']:+.4f}")

    return result


# ============================================================
# Step 4: 中性化处理
# ============================================================

def neutralize_signal_industry(signal: pd.Series,
                                industry_map: pd.DataFrame) -> pd.Series:
    """方式 A: 行业内 z-score 中性化

    每日截面: 按行业分组 zscore
    signal_neutral[i] = (signal[i] - mean(signal[industry_i])) / std(signal[industry_i])
    """
    print("\n  [中性化] 行业内 z-score...")
    ind_map = industry_map.set_index('instrument')['industry']

    result = signal.copy()
    dates = signal.index.get_level_values(0).unique()

    for dt in dates:
        day_signal = signal.loc[dt].copy()
        instruments = day_signal.index
        industries = instruments.map(ind_map).fillna('未知')

        # 按行业 zscore
        for ind in industries.unique():
            mask = industries == ind
            group = day_signal[mask]
            if len(group) > 1 and group.std() > 0:
                result.loc[(dt, group.index)] = ((group - group.mean()) / group.std()).values
            elif len(group) == 1:
                result.loc[(dt, group.index)] = 0.0

    print(f"    处理 {len(dates)} 天完成")
    return result


def neutralize_signal_regression(signal: pd.Series,
                                  industry_map: pd.DataFrame,
                                  mv_data: pd.DataFrame) -> pd.Series:
    """方式 B: 行业 + 市值回归残差中性化

    每日截面: 对行业虚拟变量 + log(市值) 做 OLS 回归, 取残差
    signal_neutral = signal - X @ beta
    """
    print("\n  [中性化] 行业 + 市值 OLS 回归残差...")
    ind_map = industry_map.set_index('instrument')['industry']

    result = signal.copy()
    dates = signal.index.get_level_values(0).unique()
    n_success = 0

    for dt in dates:
        day_signal = signal.loc[dt].dropna()
        if len(day_signal) == 0:
            continue

        instruments = day_signal.index
        industries = instruments.map(ind_map).fillna('未知')

        # 构建设计矩阵
        # 行业虚拟变量 (drop_first=True 避免多重共线)
        X_industry = pd.get_dummies(industries, drop_first=True, dtype=float)
        X_industry.index = instruments

        # log(市值)
        try:
            day_mv = mv_data.loc[dt, 'total_mv']
            log_mv = np.log(day_mv.clip(lower=1))
            # 对齐
            common = instruments.intersection(log_mv.index)
            if len(common) < 10:
                continue
            X_industry = X_industry.loc[common]
            X_mv = log_mv[common].values.reshape(-1, 1)
            y = day_signal[common].values
        except (KeyError, IndexError):
            continue

        # OLS: y = X @ beta + residual
        X = np.hstack([X_industry.values, X_mv, np.ones((len(common), 1))])
        try:
            beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            residual = y - X @ beta
            result.loc[(dt, common)] = residual
            n_success += 1
        except np.linalg.LinAlgError:
            continue

    print(f"    成功处理 {n_success}/{len(dates)} 天")
    return result


def run_neutralized_backtest(signal: pd.Series, label: str) -> dict:
    """用中性化后的信号跑 TopkDropout 回测"""
    from factor_lab.run_rolling_benchmark import run_backtest

    print(f"\n  [回测] {label}...")
    t0 = time.time()
    result = run_backtest(signal)
    elapsed = time.time() - t0

    print(f"    Sharpe: {result.get('sharpe', 0):.3f}")
    print(f"    Total Return: {result.get('total_return', 0):.2%}")
    print(f"    Max Drawdown: {result.get('max_drawdown', 0):.2%}")
    print(f"    耗时: {elapsed:.1f}s")

    result['label'] = label
    result['elapsed_seconds'] = round(elapsed, 1)
    return result


# ============================================================
# Step 5: 报告生成
# ============================================================

def generate_report(exposure: dict, attribution: dict,
                    backtest_results: list, topk: int) -> str:
    """生成文本分析报告"""
    lines = []
    lines.append("=" * 70)
    lines.append("实验 007: 行业/市值中性化分析报告")
    lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"基准策略: D_expand_3v_3r + LightGBM + alpha158_val (TopK={topk})")
    lines.append("=" * 70)

    # Part 1: 暴露分析
    lines.append("\n[ 1. 行业/市值暴露分析 ]")
    lines.append(f"  行业 HHI 指数: {exposure['hhi_index']:.4f} ({exposure['hhi_interpretation']})")
    lines.append(f"  行业偏离度 (L1): {exposure['total_industry_deviation']:.4f}")
    lines.append(f"  单行业最大占比: {exposure['max_single_industry_weight']:.1%} "
                 f"({exposure['max_single_industry_name']})")
    lines.append(f"  组合覆盖行业: {exposure['n_industries_in_portfolio']}/{exposure['n_industries_total']}")

    mv = exposure.get('market_cap_exposure', {})
    if mv:
        lines.append(f"\n  市值暴露:")
        lines.append(f"    组合平均市值: {mv['portfolio_mv_mean_billion']:.0f} 亿")
        lines.append(f"    全市场平均市值: {mv['universe_mv_mean_billion']:.0f} 亿")
        lines.append(f"    log(市值)偏差: {mv['log_mv_diff']:+.4f}")
        if mv['log_mv_diff'] < -0.3:
            lines.append(f"    → 存在显著小市值偏好")
        elif mv['log_mv_diff'] > 0.3:
            lines.append(f"    → 存在显著大市值偏好")
        else:
            lines.append(f"    → 市值暴露适中")

    lines.append(f"\n  行业偏离 Top5:")
    for i, (ind, detail) in enumerate(exposure['industry_deviation_detail'].items()):
        if i >= 5:
            break
        lines.append(f"    {ind}: 组合 {detail['portfolio_weight']:.1%} "
                     f"vs 基准 {detail['benchmark_weight']:.1%} "
                     f"(偏离 {detail['deviation']:+.1%})")

    # Part 2: 归因分解
    lines.append("\n[ 2. Brinson 收益归因 ]")
    t = attribution['total']
    lines.append(f"  超额收益: {t['excess_return']:.4%}")
    lines.append(f"    行业配置: {t['allocation']:.4%} (|abs|占比 {t['allocation_abs_pct']:.1f}%)")
    lines.append(f"    个股选择: {t['selection']:.4%} (|abs|占比 {t['selection_abs_pct']:.1f}%)")
    lines.append(f"    交互项:   {t['interaction']:.4%} (|abs|占比 {t['interaction_abs_pct']:.1f}%)")
    lines.append(f"\n  注: 各分项正负抵消较大, 百分比使用绝对值占比更准确")

    if t['allocation_abs_pct'] < t['selection_abs_pct']:
        lines.append(f"  Brinson: 个股选择波动 > 行业配置波动")
    else:
        lines.append(f"  Brinson: 行业配置波动 > 个股选择波动")

    # Part 3: 中性化回测
    if backtest_results:
        lines.append("\n[ 3. 中性化回测对比 ]")
        lines.append(f"  {'配置':<30} {'Sharpe':>8} {'收益':>10} {'最大回撤':>10}")
        lines.append(f"  {'-'*30} {'-'*8} {'-'*10} {'-'*10}")
        for r in backtest_results:
            lines.append(f"  {r['label']:<30} {r.get('sharpe',0):>8.3f} "
                         f"{r.get('total_return',0):>9.2%} "
                         f"{r.get('max_drawdown',0):>9.2%}")

        baseline = backtest_results[0]
        for r in backtest_results[1:]:
            sharpe_diff = r.get('sharpe', 0) - baseline.get('sharpe', 0)
            lines.append(f"\n  {r['label']} vs 原始:")
            lines.append(f"    Sharpe 变化: {sharpe_diff:+.3f}")
            if sharpe_diff < -0.3:
                lines.append(f"    → Sharpe 显著下降, 原策略确实依赖该暴露")
            elif sharpe_diff > -0.1:
                lines.append(f"    → Sharpe 变化不大, alpha 主要来自选股能力")

    # 总结 — 以中性化回测为主要依据
    lines.append("\n[ 4. 总结 ]")

    if backtest_results and len(backtest_results) >= 3:
        baseline_sharpe = backtest_results[0].get('sharpe', 0)
        ind_neutral_sharpe = backtest_results[1].get('sharpe', 0)
        full_neutral_sharpe = backtest_results[2].get('sharpe', 0)
        ind_retain = ind_neutral_sharpe / baseline_sharpe if baseline_sharpe > 0 else 0
        full_retain = full_neutral_sharpe / baseline_sharpe if baseline_sharpe > 0 else 0

        lines.append(f"  行业中性化后 Sharpe 保留 {ind_retain:.0%} (从 {baseline_sharpe:.3f} → {ind_neutral_sharpe:.3f})")
        lines.append(f"  行业+市值中性化后 Sharpe 保留 {full_retain:.0%} (从 {baseline_sharpe:.3f} → {full_neutral_sharpe:.3f})")

        # 行业暴露贡献 = baseline - 行业中性化
        ind_contrib = (1 - ind_retain) * 100
        # 市值暴露额外贡献 = 行业中性化 - 全中性化
        mv_extra_contrib = (ind_retain - full_retain) * 100
        # 纯选股 alpha = 全中性化保留
        alpha_contrib = full_retain * 100

        lines.append(f"\n  收益来源拆解 (基于中性化回测):")
        lines.append(f"    行业暴露贡献: ~{ind_contrib:.0f}% 的 Sharpe")
        lines.append(f"    市值暴露贡献: ~{mv_extra_contrib:.0f}% 的 Sharpe")
        lines.append(f"    纯选股 alpha:  ~{alpha_contrib:.0f}% 的 Sharpe")

        if full_retain >= 0.7:
            lines.append(f"\n  结论: 选股 alpha 扎实 ({full_retain:.0%}), 行业/市值暴露影响有限。")
            lines.append(f"  后续优化有坚实基础。")
        elif full_retain >= 0.4:
            lines.append(f"\n  结论: 存在真实选股 alpha ({full_retain:.0%}), 但行业/市值暴露贡献显著。")
            lines.append(f"  策略兼具行业择时和选股能力, 风格切换时需警惕回撤。")
            lines.append(f"  建议: 可考虑增加行业约束 (如单行业上限 30%) 以提高稳健性。")
        else:
            lines.append(f"\n  结论: 选股 alpha 较弱 ({full_retain:.0%}), 收益主要来自行业/市值暴露。")
            lines.append(f"  市场风格切换时策略可能大幅回撤, 需谨慎。")
            lines.append(f"  建议: 优先解决行业集中度问题, 再考虑后续优化。")
    else:
        lines.append("  (无回测数据, 跳过总结)")

    lines.append("\n" + "=" * 70)
    return "\n".join(lines)


# ============================================================
# Main
# ============================================================

def main():
    import multiprocessing
    try:
        multiprocessing.set_start_method('fork', force=True)
    except (ValueError, RuntimeError):
        pass  # Windows 无 fork，使用默认 spawn

    parser = argparse.ArgumentParser(description='实验007: 行业/市值中性化分析')
    parser.add_argument('--topk', type=int, default=TOPK, help='TopK 选股数量')
    parser.add_argument('--skip-backtest', action='store_true', help='跳过中性化回测')
    parser.add_argument('--pred-file', type=str,
                        default='D_expand_3v_3r_alpha158_val_LightGBM.pkl',
                        help='预测结果文件名')
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("实验 007: 行业/市值中性化分析")
    print("=" * 60)
    t_start = time.time()

    # --- 加载预测结果 ---
    pred_path = CACHE_DIR / args.pred_file
    print(f"\n[加载] 预测结果: {pred_path.name}")
    pred = pd.read_pickle(pred_path)
    print(f"  shape: {pred.shape}, index levels: {pred.index.names}")
    print(f"  日期范围: {pred.index.get_level_values(0).min().date()} ~ "
          f"{pred.index.get_level_values(0).max().date()}")

    # 确保 index 顺序是 (datetime, instrument)
    if pred.index.names == ['instrument', 'datetime']:
        pred = pred.swaplevel()
        pred.sort_index(inplace=True)

    # --- Step 1: 行业数据 ---
    print(f"\n=== Step 1: 获取行业分类数据 ===")
    industry_map = download_industry_map()
    print(f"  行业覆盖: {len(industry_map)} 只股票, {industry_map['industry'].nunique()} 个行业")

    # 检查覆盖率
    pred_instruments = pred.index.get_level_values(1).unique()
    covered = pred_instruments.isin(industry_map['instrument'])
    coverage = covered.sum() / len(pred_instruments)
    print(f"  行业数据覆盖率: {coverage:.1%} ({covered.sum()}/{len(pred_instruments)})")

    # --- 获取市值和价格数据 ---
    print(f"\n[加载] 市值和价格数据...")
    mv_data = get_market_cap_data(None, TEST_START, TEST_END)
    price_data = get_price_data(None, TEST_START, TEST_END)

    # 确保 index 顺序
    if mv_data.index.names == ['instrument', 'datetime']:
        mv_data = mv_data.swaplevel()
        mv_data.sort_index(inplace=True)
    if price_data.index.names == ['instrument', 'datetime']:
        price_data = price_data.swaplevel()
        price_data.sort_index(inplace=True)

    print(f"  市值数据: {mv_data.shape[0]} 条")
    print(f"  价格数据: {price_data.shape[0]} 条")

    # --- Step 2: 暴露分析 ---
    exposure_result = analyze_portfolio_exposure(pred, industry_map, mv_data, topk=args.topk)

    # --- Step 3: 收益归因 ---
    attribution_result = decompose_returns(pred, industry_map, price_data, topk=args.topk)

    # --- Step 4: 中性化回测 ---
    backtest_results = []
    if not args.skip_backtest:
        print("\n=== Step 4: 中性化回测对比 ===")

        # Baseline
        print("\n--- 4.0 原始信号回测 (Baseline) ---")
        baseline = run_neutralized_backtest(pred, f"原始信号 (TopK={args.topk})")
        backtest_results.append(baseline)

        # 方式 A: 行业中性化
        print("\n--- 4.1 行业中性化 ---")
        pred_ind_neutral = neutralize_signal_industry(pred, industry_map)
        bt_ind = run_neutralized_backtest(pred_ind_neutral, f"行业中性化 (TopK={args.topk})")
        backtest_results.append(bt_ind)

        # 方式 B: 行业 + 市值中性化
        print("\n--- 4.2 行业+市值中性化 ---")
        pred_full_neutral = neutralize_signal_regression(pred, industry_map, mv_data)
        bt_full = run_neutralized_backtest(pred_full_neutral, f"行业+市值中性化 (TopK={args.topk})")
        backtest_results.append(bt_full)

    # --- Step 5: 保存结果和报告 ---
    print("\n=== Step 5: 保存结果 ===")

    # 保存 JSON 结果
    with open(RESULTS_DIR / 'exposure_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(exposure_result, f, ensure_ascii=False, indent=2)
    print(f"  [保存] exposure_analysis.json")

    with open(RESULTS_DIR / 'brinson_attribution.json', 'w', encoding='utf-8') as f:
        json.dump(attribution_result, f, ensure_ascii=False, indent=2)
    print(f"  [保存] brinson_attribution.json")

    if backtest_results:
        with open(RESULTS_DIR / 'neutralized_backtest.json', 'w', encoding='utf-8') as f:
            json.dump(backtest_results, f, ensure_ascii=False, indent=2)
        print(f"  [保存] neutralized_backtest.json")

    # 生成报告
    report = generate_report(exposure_result, attribution_result, backtest_results, args.topk)
    with open(RESULTS_DIR / 'analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"  [保存] analysis_report.txt")

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"实验 007 完成! 总耗时: {elapsed:.1f}s")
    print(f"结果目录: {RESULTS_DIR}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
