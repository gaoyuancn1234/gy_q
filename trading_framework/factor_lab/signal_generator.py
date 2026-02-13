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


class SignalGenerator:
    """封装 rolling 预测 → TopK 信号 pipeline"""

    def __init__(self, config_path: str = 'config/signal_config.yaml'):
        self.config = self._load_config(config_path)
        self._predictions = None
        self._quality_score = None
        self._rolling_info = None

    def _load_config(self, config_path: str) -> dict:
        full_path = PROJECT_DIR / config_path
        with open(full_path) as f:
            return yaml.safe_load(f)

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

        with open(json_path) as f:
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
        effective_topk = TOPK_BY_REGIME[regime]

        # 获取当日信号
        day_signal = signal.loc[target_date]
        if isinstance(day_signal, pd.DataFrame):
            day_signal = day_signal.iloc[:, 0]
        day_signal = day_signal.sort_values(ascending=False)

        target_stocks = list(day_signal.head(effective_topk).index)
        scores = day_signal.head(effective_topk).to_dict()

        # 质量分数
        q_val = None
        valid_q = quality[quality.index < target_date]
        if len(valid_q) > 0:
            q_val = float(valid_q.iloc[-1])

        return {
            'date': target_date.strftime('%Y-%m-%d'),
            'regime': regime,
            'effective_topk': effective_topk,
            'quality_score': q_val,
            'target_stocks': target_stocks,
            'scores': scores,
            'current_window': self._find_current_window(target_date),
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

        to_sell = []
        for inst in current_set:
            if inst not in target_set:
                to_sell.append({'instrument': inst, 'reason': 'not_in_topk'})

        to_buy = []
        for inst in sig['target_stocks']:
            if inst not in current_set:
                to_buy.append({'instrument': inst, 'weight': round(weight, 4)})

        to_hold = list(current_set & target_set)

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
