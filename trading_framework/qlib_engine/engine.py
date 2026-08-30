#!/usr/bin/env python3
"""
Qlib 核心引擎 - 训练/预测/生成选股信号

基于 qlib_optimized_v2.py 验证过的模式：
- Handler: Alpha158 因子
- Model: LGBModel (LightGBM)
- Strategy: TopkDropoutStrategy
"""

import warnings
warnings.filterwarnings('ignore')

import re
from dateutil.relativedelta import relativedelta
import qlib
from qlib.constant import REG_CN
from qlib.contrib.model.gbdt import LGBModel
from qlib.contrib.data.handler import Alpha158
from qlib.contrib.strategy import TopkDropoutStrategy
from qlib.contrib.evaluate import backtest_daily
from qlib.utils import init_instance_by_config
from qlib.data.dataset import DatasetH
import pandas as pd


class QlibEngine:
    """Qlib ML 选股引擎"""

    _qlib_initialized = False  # 类级别标记，避免重复初始化

    def __init__(self, config: dict):
        """
        Args:
            config: 来自 settings.py 的 QLIB_CONFIG
        """
        self.config = config
        self._pred = None

    def init_qlib(self):
        """懒初始化 qlib（全局只初始化一次）"""
        if not QlibEngine._qlib_initialized:
            qlib.init(
                provider_uri=self.config['provider_uri'],
                region=REG_CN,
            )
            QlibEngine._qlib_initialized = True

    def _build_dataset(self) -> DatasetH:
        """构建 Alpha158 数据集"""
        handler_config = {
            "start_time": self.config['train_start'],
            "end_time": self.config.get('test_end', self.config['valid_end']),
            "fit_start_time": self.config['train_start'],
            "fit_end_time": self.config['train_end'],
            "instruments": self.config['instruments'],
        }
        handler = Alpha158(**handler_config)

        segments = {
            "train": (self.config['train_start'], self.config['train_end']),
            "valid": (self.config['valid_start'], self.config['valid_end']),
        }
        # test 区间可选
        if 'test_start' in self.config and 'test_end' in self.config:
            segments["test"] = (self.config['test_start'], self.config['test_end'])

        return DatasetH(handler=handler, segments=segments)

    def _build_model(self) -> LGBModel:
        """构建 LightGBM 模型"""
        params = self.config.get('model_params', {})
        return LGBModel(
            loss="mse",
            learning_rate=params.get('learning_rate', 0.01),
            num_leaves=params.get('num_leaves', 64),
            num_boost_round=params.get('num_boost_round', 500),
            early_stopping_rounds=params.get('early_stopping_rounds', 80),
            feature_fraction=params.get('feature_fraction', 0.75),
            bagging_fraction=params.get('bagging_fraction', 0.75),
            bagging_freq=params.get('bagging_freq', 5),
            lambda_l1=params.get('lambda_l1', 0.1),
            lambda_l2=params.get('lambda_l2', 0.1),
            min_data_in_leaf=params.get('min_data_in_leaf', 80),
        )

    def _build_dataset_for_period(self, train_start, train_end,
                                   valid_start, valid_end,
                                   pred_start, pred_end) -> DatasetH:
        """构建指定时间段的 Alpha158 数据集（用于滚动训练）"""
        handler_config = {
            "start_time": train_start,
            "end_time": pred_end,
            "fit_start_time": train_start,
            "fit_end_time": train_end,
            "instruments": self.config['instruments'],
        }
        handler = Alpha158(**handler_config)

        segments = {
            "train": (train_start, train_end),
            "valid": (valid_start, valid_end),
            "test": (pred_start, pred_end),
        }
        return DatasetH(handler=handler, segments=segments)

    def train_and_predict(self) -> pd.DataFrame:
        """训练模型 + 返回预测分数

        自动检测 rolling 配置，决定使用单次训练还是滚动训练。

        Returns:
            pd.DataFrame: MultiIndex (datetime, instrument) 的预测分数
        """
        rolling_cfg = self.config.get('rolling', {})
        if rolling_cfg.get('enabled', False):
            return self.rolling_train_and_predict()

        self.init_qlib()

        dataset = self._build_dataset()
        model = self._build_model()

        print("[QlibEngine] 训练模型（单次）...")
        model.fit(dataset)

        print("[QlibEngine] 生成预测...")
        self._pred = model.predict(dataset)
        print(f"[QlibEngine] 预测样本数: {len(self._pred)}")

        return self._pred

    def rolling_train_and_predict(self) -> pd.DataFrame:
        """滚动训练（Walk-Forward）：每隔 retrain_months 重训模型

        流程：
          1. 将 test 区间按 retrain_months 切分为多个预测窗口
          2. 每个窗口独立训练模型（训练集向前滑动）
          3. 拼接所有窗口的预测结果

        Returns:
            pd.DataFrame: 拼接后的预测分数
        """
        self.init_qlib()

        rolling_cfg = self.config['rolling']
        train_window_years = rolling_cfg['train_window_years']
        valid_months = rolling_cfg['valid_months']
        retrain_months = rolling_cfg['retrain_months']

        test_start = pd.Timestamp(self.config['test_start'])
        test_end = pd.Timestamp(self.config['test_end'])

        all_preds = []
        window_num = 0
        cursor = test_start  # 当前预测窗口的起始日

        while cursor < test_end:
            window_num += 1

            # 预测区间: [cursor, cursor + retrain_months) 或到 test_end
            pred_end_candidate = cursor + relativedelta(months=retrain_months) - relativedelta(days=1)
            pred_end_actual = min(pred_end_candidate, test_end)

            # 验证区间: 预测起始日前的 valid_months
            v_end = cursor - relativedelta(days=1)
            v_start = cursor - relativedelta(months=valid_months)

            # 训练区间: 验证起始日前的 train_window_years
            t_end = v_start - relativedelta(days=1)
            t_start = v_start - relativedelta(years=train_window_years)

            # 格式化日期
            fmt = lambda d: d.strftime('%Y-%m-%d')

            print(f"\n[QlibEngine] === 滚动窗口 {window_num} ===")
            print(f"  训练: {fmt(t_start)} ~ {fmt(t_end)}")
            print(f"  验证: {fmt(v_start)} ~ {fmt(v_end)}")
            print(f"  预测: {fmt(cursor)} ~ {fmt(pred_end_actual)}")

            # 构建数据集并训练
            dataset = self._build_dataset_for_period(
                fmt(t_start), fmt(t_end),
                fmt(v_start), fmt(v_end),
                fmt(cursor), fmt(pred_end_actual),
            )
            model = self._build_model()

            print(f"  训练中...")
            model.fit(dataset)

            pred = model.predict(dataset)

            # 只保留 test segment 的预测（model.predict 会返回所有 segment）
            if isinstance(pred.index, pd.MultiIndex):
                dates = pred.index.get_level_values(0)
                mask = (dates >= cursor) & (dates <= pred_end_actual)
                pred = pred[mask]

            best_iter = getattr(model.model, 'best_iteration', 'N/A')
            print(f"  最佳迭代: {best_iter}, 预测样本: {len(pred)}")
            all_preds.append(pred)

            # 滑动到下一个窗口
            cursor = cursor + relativedelta(months=retrain_months)

        self._pred = pd.concat(all_preds)
        print(f"\n[QlibEngine] 滚动训练完成: {window_num} 个窗口, "
              f"总预测样本: {len(self._pred)}")
        return self._pred

    def get_top_signals(self, top_k: int) -> list:
        """从预测结果中取 top_k 股票代码

        Args:
            top_k: 选取前 k 只股票

        Returns:
            list[str]: baostock 格式的股票代码列表 (如 ['sh.600519', 'sz.000858'])
        """
        if self._pred is None:
            self.train_and_predict()

        pred = self._pred

        if len(pred) == 0:
            print("[QlibEngine] 警告: 预测结果为空，请检查数据时间范围")
            return []

        # 取最新一天的预测
        if isinstance(pred.index, pd.MultiIndex):
            latest_date = pred.index.get_level_values(0).max()
            if pd.isna(latest_date):
                print("[QlibEngine] 警告: 预测日期为空")
                return []
            latest_pred = pred.loc[latest_date]
        else:
            latest_date = "unknown"
            latest_pred = pred

        # 按分数降序排列，取 top_k
        top_stocks = latest_pred.sort_values(ascending=False).head(top_k)
        print(f"[QlibEngine] Top {top_k} 信号 ({latest_date}):")

        # 转换代码格式: Qlib SH600519 -> baostock sh.600519
        result = []
        for instrument in top_stocks.index:
            code = self._qlib_to_baostock(instrument)
            score = top_stocks[instrument]
            if isinstance(score, pd.Series):
                score = score.iloc[0]
            print(f"  {code} score={score:.4f}")
            result.append(code)

        return result

    def run_backtest(self) -> dict:
        """运行回测，返回绩效指标

        Returns:
            dict: 包含 total_return, bench_return, excess_return 等
        """
        if self._pred is None:
            self.train_and_predict()

        topk = self.config.get('topk', 50)
        n_drop = self.config.get('n_drop', 5)

        strategy_config = {
            "class": "TopkDropoutStrategy",
            "module_path": "qlib.contrib.strategy",
            "kwargs": {
                "signal": self._pred,
                "topk": topk,
                "n_drop": n_drop,
            },
        }

        test_start = self.config.get('test_start') or self.config.get('valid_start')
        test_end = self.config.get('test_end') or self.config.get('valid_end')

        backtest_config = {
            "start_time": test_start,
            "end_time": test_end,
            "account": 100_000_000,
            "benchmark": "SH000300",
            "exchange_kwargs": {
                "freq": "day",
                "limit_threshold": 0.095,
                "deal_price": "close",  # T+1: 信号当日收盘后生成，次日收盘价成交 (2026-08-30 由 open 改)
                "open_cost": 0.0005,
                "close_cost": 0.0015,
                "min_cost": 5,
                "trade_unit": 100,      # A股最小交易单位
            },
        }

        print("[QlibEngine] 运行回测...")
        strategy = init_instance_by_config(strategy_config)
        report, positions = backtest_daily(
            strategy=strategy,
            executor=None,
            **backtest_config,
        )

        result = {}
        if 'return' in report.columns:
            result['total_return'] = (1 + report['return']).prod() - 1
        if 'bench' in report.columns:
            result['bench_return'] = (1 + report['bench']).prod() - 1
            result['excess_return'] = result.get('total_return', 0) - result['bench_return']

        print(f"[QlibEngine] 回测结果: {result}")
        return result

    @staticmethod
    def _qlib_to_baostock(qlib_code: str) -> str:
        """Qlib 代码转 baostock 格式

        SH600519 -> sh.600519
        SZ000858 -> sz.000858
        """
        qlib_code = str(qlib_code).upper()
        match = re.match(r'^(SH|SZ)(\d{6})$', qlib_code)
        if match:
            return f"{match.group(1).lower()}.{match.group(2)}"
        return qlib_code

    @staticmethod
    def _baostock_to_qlib(bao_code: str) -> str:
        """baostock 代码转 Qlib 格式

        sh.600519 -> SH600519
        sz.000858 -> SZ000858
        """
        match = re.match(r'^(sh|sz)\.(\d{6})$', bao_code)
        if match:
            return f"{match.group(1).upper()}{match.group(2)}"
        return bao_code
