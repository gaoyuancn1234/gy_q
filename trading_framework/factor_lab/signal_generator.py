#!/usr/bin/env python3
"""信号生成器 — 封装 rolling 预测 → TopK 信号 pipeline

核心职责:
1. 加载/生成 rolling 预测 (复用 run_execution_benchmark 缓存)
2. 计算信号质量 (复用 evaluation.signal_quality)
3. 根据质量动态调整 topK (策略 A)
4. 返回当日交易信号 + 调仓指令

用法:
    from factor_lab.signal_generator import SignalGenerator

    sg = SignalGenerator()
    signal = sg.get_signal('2026-02-10')
    instructions = sg.get_rebalance_instructions(current_positions, '2026-02-10')
"""
import json
import pickle
import warnings
from pathlib import Path

import yaml
import pandas as pd

warnings.filterwarnings('ignore')

PROJECT_DIR = Path(__file__).resolve().parent.parent

# 策略 A topK 映射 (与实验 008 AdaptiveBacktester 一致)
TOPK_BY_REGIME = {
    'strong': 12,
    'normal': 16,
    'weak': 20,
}


def cap_topk(topk: int, config: dict) -> int:
    """按资金约束收敛 TopK — 信号生成器与模拟盘共用

    2026-09-03 新增，2026-09-04 抽为模块级函数。
    TOPK_BY_REGIME 是按 20 万资金标定的 (strong 12 / normal 16 / weak 20)，
    与 signal_config 里的 topk 无关，且会静默覆盖它。10 万资金落到 normal 档
    每仓仅 6250 元、weak 档 5000 元 —— 沪深300 里相当一部分股票一手就超过
    这个数，那些仓位根本建不起来。

    实测后果 (模拟盘回放，修复前): 持有 16 只而非配置的 8 只，
    SH600460 只买到 100 股(3285 元)、SH601916 仅 1704 元，仓位碎到无意义。

    最初只在 SignalGenerator 内部实现，模拟盘 (paper_trader) 直接用
    TOPK_BY_REGIME 绕过了它 —— 同一规则写两处必然漏改，故抽出共用。

    max_topk_by_capital 未配置时不做限制，保持原行为。
    """
    cap = (config or {}).get('max_topk_by_capital')
    if not cap:
        return topk
    capped = min(topk, int(cap))
    if capped != topk:
        cash = (config or {}).get('initial_cash')
        print(f"  [signal] TopK {topk} → {capped} "
              f"(资金约束 max_topk_by_capital={cap}"
              f"{f', 资金 {cash:,.0f}' if cash else ''})")
    return capped


class SignalGenerator:
    """封装 rolling 预测 → TopK 信号 pipeline"""

    def __init__(self, config_path: str = 'config/signal_config.yaml'):
        self.config = self._load_config(config_path)
        self._predictions = None
        self._quality_score = None
        self._rolling_info = None

    def _load_config(self, config_path: str) -> dict:
        full_path = PROJECT_DIR / config_path
        with open(full_path, encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _cap_topk(self, topk: int) -> int:
        """按资金约束收敛 TopK (实现见模块级 cap_topk)"""
        return cap_topk(topk, self.config)

    # ── 数据加载 ──

    def load_predictions(self) -> pd.Series:
        """加载缓存的 rolling 预测"""
        if self._predictions is not None:
            return self._predictions

        cfg = self.config
        cache_dir = PROJECT_DIR / cfg['model_cache_dir']
        pkl_name = f"{cfg['rolling_config']}_{cfg['preset']}_{cfg['model']}.pkl"
        pkl_path = cache_dir / pkl_name

        if not pkl_path.exists():
            raise FileNotFoundError(
                f"预测缓存不存在: {pkl_path}\n"
                "请先运行: python -m factor_lab.run_execution_benchmark --predict-only"
            )

        self._predictions = pd.read_pickle(pkl_path)
        return self._predictions

    def load_quality_score(self) -> pd.Series:
        """加载信号质量评分"""
        if self._quality_score is not None:
            return self._quality_score

        quality_dir = PROJECT_DIR / self.config['quality_cache_dir']
        pkl_path = quality_dir / "quality_score.pkl"

        if not pkl_path.exists():
            raise FileNotFoundError(
                f"质量评分缓存不存在: {pkl_path}\n"
                "请先运行: python -m factor_lab.run_signal_decay_benchmark --analysis-only"
            )

        with open(pkl_path, 'rb') as f:
            self._quality_score = pickle.load(f)
        return self._quality_score

    def load_rolling_info(self) -> dict:
        """加载 rolling window 信息"""
        if self._rolling_info is not None:
            return self._rolling_info

        cfg = self.config
        json_dir = PROJECT_DIR / cfg['rolling_json_dir']
        json_name = f"{cfg['rolling_config']}_{cfg['preset']}_{cfg['model']}.json"
        json_path = json_dir / json_name

        with open(json_path, encoding='utf-8') as f:
            self._rolling_info = json.load(f)
        return self._rolling_info

    # ── 信号生成 ──

    def _get_regime(self, date: pd.Timestamp) -> str:
        """获取 date 前一天的信号状态 (防前视偏差)"""
        quality = self.load_quality_score()
        lo, hi = self.config['adaptive_thresholds']
        valid = quality[quality.index < date]
        if len(valid) == 0:
            return 'normal'
        last_score = valid.iloc[-1]
        if last_score >= hi:
            return 'strong'
        elif last_score < lo:
            return 'weak'
        return 'normal'

    def _find_current_window(self, date: pd.Timestamp) -> int | None:
        """找到 date 所在的 rolling window"""
        info = self.load_rolling_info()
        for w in info['windows']:
            ws = pd.Timestamp(w['pred_start'])
            we = pd.Timestamp(w['pred_end'])
            if ws <= date <= we:
                return w['window_num']
        return None

    def get_signal(self, date: str = None) -> dict:
        """获取指定日期的交易信号

        Args:
            date: 日期字符串 (YYYY-MM-DD)，None 表示最新可用日期

        Returns:
            {
                'date': '2026-02-10',
                'regime': 'normal',
                'effective_topk': 16,
                'quality_score': 0.45,
                'target_stocks': ['SH600036', 'SH601318', ...],
                'scores': {'SH600036': 0.023, ...},
                'current_window': 9,
            }
        """
        signal = self.load_predictions()
        quality = self.load_quality_score()

        signal_dates = signal.index.get_level_values(0).unique().sort_values()

        if date is None:
            target_date = signal_dates[-1]
        else:
            target_date = pd.Timestamp(date)
            if target_date not in signal_dates:
                # 找最近的可用日期
                earlier = signal_dates[signal_dates <= target_date]
                if len(earlier) == 0:
                    return {'error': f'日期 {date} 无可用信号 (最早: {signal_dates[0].strftime("%Y-%m-%d")})'}
                target_date = earlier[-1]

        # 获取 regime
        regime = self._get_regime(target_date)
        effective_topk = self._cap_topk(TOPK_BY_REGIME[regime])

        # 获取当日信号
        day_signal = signal.loc[target_date]
        if isinstance(day_signal, pd.DataFrame):
            day_signal = day_signal.iloc[:, 0]
        day_signal = day_signal.sort_values(ascending=False)

        target_stocks = list(day_signal.head(effective_topk).index)
        # scores 必须是**全池**分数，不能只给 TopK。
        # 2026-09-05: 原先是 day_signal.head(effective_topk).to_dict()，
        # 而卖出候选按定义就是"不在 TopK 里"的持仓 —— 于是 select_sells 里
        # 每一个候选都 scores.get(c, -inf) 取不到值、一起并列 -inf，排序
        # 退化成按股票代码字母序。n_drop 本该"卖掉分数最低的 N 只"，实盘
        # 实际执行的是"卖掉代码最小的 N 只"。
        # 双路径对账发现: 2026-03~09 的 15 个调仓日里 14 个卖出清单与回测
        # 不同(买入清单 15/15 全同)，即 n_drop 这项 Sharpe 0.33->1.27 的
        # 改动在实盘被降级成了随机挑选。
        scores = day_signal.to_dict()

        # 质量分数
        q_val = None
        valid_q = quality[quality.index < target_date]
        if len(valid_q) > 0:
            q_val = float(valid_q.iloc[-1])

        health = self._check_signal_health(day_signal, target_date)

        return {
            'date': target_date.strftime('%Y-%m-%d'),
            'regime': regime,
            'effective_topk': effective_topk,
            'quality_score': q_val,
            'target_stocks': target_stocks,
            'scores': scores,
            'current_window': self._find_current_window(target_date),
            'health': health,
        }

    # 退化模型的判定阈值
    MIN_SCORE_STD_RATIO = 0.25   # 当日分数标准差 / 历史中位数，低于此视为退化
    MIN_UNIQUE_RATIO = 0.5       # 唯一分数占股票数比例
    MIN_BEST_ITER = 20           # LightGBM 早停轮数下限

    def _check_signal_health(self, day_signal, target_date) -> dict:
        """信号健康检查 — 拦截退化模型产生的无效信号

        背景 (2026-08-30): Window 11 的 LightGBM 在第 3 轮就早停 (阈值 80)，
        因为验证期 2026 Q2 的 IC 为负，模型学不到任何有效模式。3 棵树的模型
        产生几乎扁平的预测: 300 只股票只有 123 个不同分数、标准差 0.0023
        (历史约 0.05，仅 1/30)，TopK 里 8 只并列同分 —— 选哪只本质是随机。
        Window 8 (验证期 2025 Q3, IC 亦为负) 同样发生过 (best_iter=4)。

        原先系统对此毫无察觉，会照常把随机名单推送出去。
        """
        import numpy as np
        issues = []

        n = len(day_signal)
        std = float(day_signal.std())
        n_uniq = int(day_signal.nunique())

        # 与历史比较: 取全部预测按日标准差的中位数作基准
        try:
            all_std = self._predictions.groupby(level=0).std()
            base_std = float(all_std.median())
        except Exception:
            base_std = None

        if base_std and base_std > 0 and std < base_std * self.MIN_SCORE_STD_RATIO:
            issues.append(f"分数标准差异常收缩: {std:.5f} (历史中位数 {base_std:.5f})")
        if n and n_uniq < n * self.MIN_UNIQUE_RATIO:
            issues.append(f"分数区分度不足: {n} 只股票仅 {n_uniq} 个不同分数")

        # 该日所属窗口的 best_iteration
        best_iter = None
        try:
            info = self.load_rolling_info()
            for w in info.get('windows', []):
                if w['pred_start'] <= target_date.strftime('%Y-%m-%d') <= w['pred_end']:
                    best_iter = w.get('best_iteration')
                    break
        except Exception as e:
            # 不能静默 pass: best_iter 保持 None 会让下面的判据被整条跳过，
            # 健康检查少了一条却毫无痕迹 —— "检查失败"被当成"检查通过"。
            issues.append(f"无法读取 best_iteration (健康检查第三条判据未生效): {e}")
        if best_iter is not None and best_iter < self.MIN_BEST_ITER:
            issues.append(f"模型早停轮数过低: best_iter={best_iter} "
                          f"(阈值 {self.MIN_BEST_ITER})，模型未学到有效模式")

        return {
            'ok': not issues,
            'issues': issues,
            'score_std': std,
            'n_unique': n_uniq,
            'n_stocks': n,
            'best_iteration': best_iter,
        }

    def get_rebalance_instructions(self, current_positions: dict,
                                    date: str = None) -> dict:
        """生成调仓指令

        Args:
            current_positions: {instrument: shares} 当前持仓
            date: 调仓日期

        Returns:
            {
                'date': ...,
                'to_sell': [{'instrument': ..., 'reason': 'not_in_topk'}],
                'to_buy': [{'instrument': ..., 'weight': 0.05}],
                'to_hold': [...],
            }
        """
        sig = self.get_signal(date)
        if 'error' in sig:
            return sig

        target_set = set(sig['target_stocks'])
        current_set = set(current_positions.keys())
        topk = sig['effective_topk']
        weight = 1.0 / topk

        # 显式排序: 直接迭代集合会因 Python 字符串哈希随机化导致同一输入
        # 两次得到不同顺序。to_sell 的顺序会影响资金不足时先卖谁，
        # 进而影响实际成交 —— CLAUDE.md 记录过同类事故(Sharpe 0.318 vs 0.247)。
        # 按信号分数升序卖出(最差的先走)，与 rebalance_rules.select_sells 一致。
        scores = sig.get('scores') or {}
        to_sell = [
            {'instrument': inst, 'reason': 'not_in_topk'}
            for inst in sorted(current_set - target_set,
                               key=lambda c: (scores.get(c, float('-inf')), c))
        ]

        to_buy = []
        for inst in sig['target_stocks']:      # 已按分数降序，顺序确定
            if inst not in current_set:
                to_buy.append({'instrument': inst, 'weight': round(weight, 4)})

        to_hold = sorted(current_set & target_set)

        return {
            'date': sig['date'],
            'regime': sig['regime'],
            'effective_topk': topk,
            'to_sell': to_sell,
            'to_buy': to_buy,
            'to_hold': to_hold,
        }

    def check_model_freshness(self) -> dict:
        """检查模型是否过期 (>3个月)"""
        info = self.load_rolling_info()
        windows = info['windows']
        last_window = windows[-1]
        pred_end = pd.Timestamp(last_window['pred_end'])
        today = pd.Timestamp('today').normalize()
        gap_days = (today - pred_end).days

        return {
            'last_window': last_window['window_num'],
            'pred_end': last_window['pred_end'],
            'days_since': gap_days,
            'is_stale': gap_days > 90,
            'message': (
                f"模型覆盖至 {last_window['pred_end']}，距今 {gap_days} 天"
                + (" (已过期，建议重新训练)" if gap_days > 90 else " (有效)")
            ),
        }

    def get_available_dates(self) -> list[str]:
        """返回所有可用的信号日期"""
        signal = self.load_predictions()
        dates = signal.index.get_level_values(0).unique().sort_values()
        return [d.strftime('%Y-%m-%d') for d in dates]
