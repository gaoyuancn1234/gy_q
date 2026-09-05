#!/usr/bin/env python3
"""模拟盘引擎 — 每日更新 NAV + 周度调仓

状态持久化:
- state.json: 现金 + 持仓 + 元数据
- trades.csv: 交易记录
- daily_nav.csv: 日净值

执行模型 (T+1 收盘价):
- Day T 收盘后: 生成信号 → 产生交易指令 (pending)
- Day T+1 收盘: 以收盘价执行交易 + 更新 NAV

用法:
    # 回放历史 (验证)
    python -m factor_lab.paper_trader replay

    # 查看当前状态
    python -m factor_lab.paper_trader status

    # 重置
    python -m factor_lab.paper_trader reset
"""
import json
import csv
import sys
import time
import warnings
import argparse
from pathlib import Path
from datetime import datetime

# 直接执行 `python factor_lab/paper_trader.py` 时 sys.path[0] 是 factor_lab/，
# 顶层的 portfolio 包不可见。这里补上项目根目录，让脚本式和 -m 两种调用都能跑。
_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# 调仓风控规则与实盘共用同一实现，避免两边各写一份再次分叉
from portfolio.rebalance_rules import select_sells, compute_exposure

import yaml
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

PROJECT_DIR = Path(__file__).resolve().parent.parent

from factor_lab.signal_generator import TOPK_BY_REGIME, cap_topk


class PaperTrader:
    """模拟盘引擎 — 收盘价成交，T+1 执行"""

    def __init__(self, config_path: str = 'config/signal_config.yaml',
                 state_dir: str = None, pred_tag: str = None):
        """
        Args:
            state_dir: 覆盖配置里的状态目录。分析类脚本(对账、相位扫描)必须
                传一个临时目录 —— 2026-09-05 教训: reconcile.py 与
                run_phase_test.py 都调 replay()，而 replay() 开头的 reset()
                无条件 _save_state() 并删除 trades.csv / daily_nav.csv，
                于是每跑一次分析就把真实模拟盘的持仓与历史清空一次。
                只读的分析绝不该写生产状态。
        """
        self.config = self._load_config(config_path)
        self.state_dir = (Path(state_dir) if state_dir
                          else PROJECT_DIR / self.config['state_dir'])
        self.state_dir.mkdir(parents=True, exist_ok=True)

        # 从 SignalGenerator 加载信号和质量分数
        from factor_lab.signal_generator import SignalGenerator
        self.sg = SignalGenerator(config_path, pred_tag=pred_tag)

        self.state = self._load_state()

    def _load_config(self, config_path: str) -> dict:
        full_path = PROJECT_DIR / config_path
        with open(full_path, encoding='utf-8') as f:
            return yaml.safe_load(f)

    # ── 状态管理 ──

    def _load_state(self) -> dict:
        state_file = self.state_dir / "state.json"
        if state_file.exists():
            with open(state_file, encoding='utf-8') as f:
                return json.load(f)
        return self._initial_state()

    def _initial_state(self) -> dict:
        return {
            'cash': self.config['initial_cash'],
            'positions': {},
            'pending_orders': {'sells': [], 'buys': {}},
            # 波动率目标需要按已实现波动率缩放敞口，故在状态里保留净值序列。
            # 净值同时写入 daily_nav.csv，但那是给人看的，程序不依赖解析 CSV。
            'nav_history': [],
            'metadata': {
                'initial_cash': self.config['initial_cash'],
                'start_date': None,
                'last_update': None,
                'day_count': 0,
                'strategy': self.config['adaptive_strategy'],
            },
        }

    def _save_state(self):
        state_file = self.state_dir / "state.json"

        def _convert(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        # 原子写入: 模拟盘每个交易日都写这个文件，中断会留下截断的 JSON，
        # 下次读取直接 JSONDecodeError，整段回放历史丢失。
        from factor_lab.utils import atomic_json_dump
        atomic_json_dump(state_file, self.state, indent=2,
                         ensure_ascii=False, default=_convert)

    def _append_trade(self, date: str, action: str, instrument: str,
                      shares: int, price: float, cost: float, reason: str):
        """追加交易记录到 CSV"""
        trades_file = self.state_dir / "trades.csv"
        is_new = not trades_file.exists()
        with open(trades_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if is_new:
                writer.writerow(['date', 'action', 'instrument', 'shares',
                                 'price', 'cost', 'reason'])
            writer.writerow([date, action, instrument, shares,
                             f"{price:.4f}", f"{cost:.4f}", reason])

    def _append_nav(self, date: str, nav: float, cash: float,
                    n_positions: int):
        """追加日净值到 CSV"""
        nav_file = self.state_dir / "daily_nav.csv"
        is_new = not nav_file.exists()
        with open(nav_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if is_new:
                writer.writerow(['date', 'nav', 'cash', 'n_positions'])
            writer.writerow([date, f"{nav:.2f}", f"{cash:.2f}", n_positions])

        # 同步维护状态里的净值序列 (vol_target 计算已实现波动率要用)。
        # 只保留最近 250 个交易日: 波动率窗口默认 20 天，留一年足够且不让状态无限膨胀。
        hist = self.state.setdefault('nav_history', [])
        hist.append({'date': date, 'nav': float(nav)})
        if len(hist) > 250:
            del hist[:-250]

    def reset(self, initial_cash: float = None):
        """重置模拟盘"""
        if initial_cash:
            self.config['initial_cash'] = initial_cash
        self.state = self._initial_state()
        self._save_state()
        # 清除交易和净值记录
        for f in ['trades.csv', 'daily_nav.csv']:
            p = self.state_dir / f
            if p.exists():
                p.unlink()
        print(f"模拟盘已重置 (初始资金: {self.state['cash']:,.0f})")

    # ── 每日更新 ──

    def update_daily(self, date: pd.Timestamp, prices_df: pd.DataFrame,
                     prev_close_map: dict, day_idx: int,
                     signal: pd.Series, quality_score: pd.Series,
                     auto_save: bool = False) -> dict:
        """每日收盘后调用

        Args:
            date: 当日日期
            prices_df: 价格 DataFrame (datetime, instrument)
            prev_close_map: {instrument: 前收盘价}
            day_idx: 交易日序号 (用于判断调仓日)
            signal: 预测信号 Series
            quality_score: 信号质量评分 Series

        Returns:
            {'nav': float, 'trades': int, 'regime': str or None}
        """
        if date not in prices_df.index.get_level_values(0):
            return {'nav': 0, 'trades': 0, 'regime': None}

        day_data = prices_df.loc[date]
        day_close = day_data['$close'].dropna()

        date_str = date.strftime('%Y-%m-%d')
        n_trades = 0
        cfg = self.config

        # --- 1. 执行昨日挂起的交易指令 (以今日收盘价) ---
        pending = self.state['pending_orders']

        # 卖出
        for inst in list(pending['sells']):
            if inst not in self.state['positions']:
                pending['sells'].remove(inst)
                continue
            if inst not in day_close.index:
                continue
            # 跌停不能卖
            if inst in prev_close_map and prev_close_map[inst] > 0:
                limit_ret = day_close[inst] / prev_close_map[inst] - 1
                if limit_ret < -0.095:
                    continue

            price = day_close[inst]
            pos = self.state['positions'][inst]
            shares = pos['shares']
            proceeds = shares * price * (1 - cfg['close_cost'])
            self.state['cash'] += proceeds
            self._append_trade(date_str, 'SELL', inst, shares, price,
                               shares * price * cfg['close_cost'], 'rebalance')
            del self.state['positions'][inst]
            pending['sells'].remove(inst)
            n_trades += 1

        # 买入
        if pending['buys']:
            total_weight = sum(pending['buys'].values())
            available_cash = self.state['cash'] * 0.99  # 预留 1%

            # 波动率目标: 缩减本次可投入的资金。
            #
            # 2026-09-05 修复。原先 exposure 只存进 pending['buys'] 的 weight
            # 里，而分配式是 alloc = cash * (weight / sum(weights))。每个
            # weight 都等于 exposure/topk，权重相同 -> 归一化后 = cash / n，
            # **exposure 被整个约掉**。即 vol_target 在本引擎里完全不起作用。
            #
            # 实测: 同区间同相位跑 vol_target=8% 与关闭，Sharpe 0.976 /
            # 收益 +17.01% / 交易 331 / 终值 117,006 —— 逐个数字完全一致。
            #
            # 而实盘 (live_portfolio.py:703) 做的是 `available_cash *= exposure`，
            # 即真的少投一部分钱。于是实盘跑的策略与所有 paper_trader 回测
            # 都不是同一个。这里改为与实盘同一口径。
            #
            # exposure 与这批买单绑定存放(而非放在实例属性上) —— 挂单是隔日
            # 执行的，用实例属性会取到之后某次调仓算出的值。
            #
            # 注: reconcile.py 只比对买卖清单，仓位大小不在其覆盖范围内 ——
            # 这处分叉正是因此才藏到今天。
            _exp = float(pending.get('exposure', 1.0) or 1.0)
            if _exp < 1.0:
                available_cash *= _exp
            if total_weight > 0:
                for inst, weight in list(pending['buys'].items()):
                    if inst in self.state['positions']:
                        continue
                    if inst not in day_close.index:
                        continue
                    # 涨停不能买
                    if inst in prev_close_map and prev_close_map[inst] > 0:
                        limit_ret = day_close[inst] / prev_close_map[inst] - 1
                        if limit_ret > 0.095:
                            continue

                    price = day_close[inst]
                    alloc = available_cash * (weight / total_weight)
                    shares = int(alloc / (price * (1 + cfg['open_cost'])) / 100) * 100
                    if shares >= 100:
                        cost = shares * price * (1 + cfg['open_cost'])
                        self.state['cash'] -= cost
                        self.state['positions'][inst] = {
                            'shares': shares,
                            'cost_price': float(price),
                            'entry_date': date_str,
                        }
                        self._append_trade(date_str, 'BUY', inst, shares, price,
                                           shares * price * cfg['open_cost'], 'rebalance')
                        n_trades += 1
            pending['buys'] = {}
            pending['exposure'] = 1.0

        # --- 2. 收盘估值 ---
        #
        # 停牌股用**最后成交价**估值，不是成本价。
        # 2026-09-05: 原先取不到当日行情就回退 pos['cost_price'] —— 一只跌了
        # 30% 之后停牌的票，在净值里显示零亏损。净值虚高、回撤与波动率被低估，
        # 而已实现波动率正是 vol_target 的输入，于是敞口还会被算高。
        # 与当日在 live_portfolio.compute_nav 修的是同一类问题(用成本价冒充
        # 市价)，这里是模拟盘的那一份。
        #
        # 背景: A 股停牌常见，且卖单遇停牌会一直挂着重试。段1 相位5 实测有
        # 1180 个"待卖但当日无行情"的日次(相位7 只有 205)，被卡住的仓位同时
        # 占着 topk 名额，组合因此几乎停止轮动 —— 200 张卖单只成交 76 张。
        port_value = self.state['cash']
        for inst, pos in self.state['positions'].items():
            if inst in day_close.index:
                px = float(day_close[inst])
                pos['last_price'] = px          # 记下，供停牌期间估值
            else:
                px = float(pos.get('last_price') or pos['cost_price'])
            port_value += pos['shares'] * px

        self._append_nav(date_str, port_value, self.state['cash'],
                         len(self.state['positions']))

        # --- 3. 收盘后检查止损 ---
        if cfg['stop_loss']:
            for inst, pos in list(self.state['positions'].items()):
                if inst in day_close.index:
                    ret = day_close[inst] / pos['cost_price'] - 1
                    if ret < -cfg['stop_loss']:
                        if inst not in pending['sells']:
                            pending['sells'].append(inst)
                            # 止损卖出不等调仓日，下一日收盘执行

        # --- 4. 调仓日: 生成新的交易指令 ---
        regime = None
        rebal = cfg['rebalance_every']
        signal_dates = set(signal.index.get_level_values(0).unique())

        if day_idx % rebal == 0 and date in signal_dates:
            # 信号质量 → regime
            lo, hi = cfg['adaptive_thresholds']
            valid_q = quality_score[quality_score.index < date]
            if len(valid_q) > 0:
                last_q = valid_q.iloc[-1]
                if last_q >= hi:
                    regime = 'strong'
                elif last_q < lo:
                    regime = 'weak'
                else:
                    regime = 'normal'
            else:
                regime = 'normal'

            # 资金上限: 与 SignalGenerator 共用同一规则。
            # 此前直接用 TOPK_BY_REGIME，绕过上限 —— 实测持有 16 只而非 8 只。
            effective_topk = cap_topk(TOPK_BY_REGIME[regime], cfg)

            # 获取当日信号
            if date in signal_dates:
                day_signal = signal.loc[date]
                if isinstance(day_signal, pd.DataFrame):
                    day_signal = day_signal.iloc[:, 0]
                day_signal = day_signal.sort_values(ascending=False)
                target_stocks = set(day_signal.head(effective_topk).index)

                # 卖出 — 受 n_drop 换手限制 (2026-09-03 补齐)
                # 此前模拟盘是全量换手，与实盘/回测不一致，绩效被交易成本系统性拖低。
                current_set = set(self.state['positions'].keys())
                to_sell = select_sells(
                    current_set, target_stocks,
                    scores=day_signal.to_dict(),
                    n_drop=cfg.get('n_drop'),
                )
                for inst in sorted(to_sell):          # 显式排序，保证可复现
                    if inst not in pending['sells']:
                        pending['sells'].append(inst)

                # 规划买入 — 按信号分数降序排列，保证确定性迭代顺序。
                # 用集合差集会因 Python 字符串哈希随机化导致同配置两次结果不同。
                current_holds = current_set - set(pending['sells'])
                # 只买入被腾出的坑位数，保持持仓数稳定 (与实盘一致)
                free_slots = max(0, effective_topk - len(current_holds))
                # 候选里必须排除**全部现有持仓**，而不只是 current_holds。
                # 2026-09-05 双路径对账发现: 一只被止损排进待卖、但仍留在
                # TopK 里的股票，不在 current_holds 中，于是会被"卖掉再买回"。
                # 执行阶段该买单会因 `inst in positions` 被跳过，但它已经占掉
                # 一个 free_slot，把真正该买的股票挤出去 —— 资金被少投出去一份。
                # 实盘 (compute_rebalance_orders) 排除的是全部持仓，是对的。
                to_buy = [i for i in day_signal.head(effective_topk).index
                          if i not in current_holds and i not in current_set
                          ][:free_slots]

                # 记录决策时刻的输入与输出，供 reconcile.py 做等价比对。
                # 必须在这里记 —— 决策发生在"执行完昨日挂单"之后，从外部
                # 取快照会差一步状态，比出来的是测量偏差而非真分叉。
                self.last_decision = {
                    'date': date_str,
                    'positions': sorted(current_set),
                    'pending_sells': sorted(pending['sells']),
                    'effective_topk': effective_topk,
                    'regime': regime,
                    'target_stocks': list(day_signal.head(effective_topk).index),
                    'scores': day_signal.to_dict(),
                    'sells': sorted(to_sell),
                    'buys': list(to_buy),
                }
                if to_buy:
                    # 波动率目标: 按已实现波动率缩放本次投入的权益敞口
                    exposure, realized = compute_exposure(
                        [h['nav'] for h in self.state.get('nav_history', [])],
                        cfg.get('vol_target'),
                        window=int(cfg.get('vol_window', 20)),
                        min_exposure=float(cfg.get('vol_min_exposure', 0.2)),
                    )
                    weight = exposure / effective_topk
                    pending['buys'] = {inst: weight for inst in to_buy}
                    # 与这批买单绑定，执行日据此缩减可用资金 (见上方注释)
                    pending['exposure'] = float(exposure)
                    if exposure < 1.0:
                        self._last_exposure = (exposure, realized)

        # 更新元数据
        self.state['pending_orders'] = pending
        self.state['metadata']['last_update'] = date_str
        self.state['metadata']['day_count'] += 1
        if self.state['metadata']['start_date'] is None:
            self.state['metadata']['start_date'] = date_str

        if auto_save:
            self._save_state()

        return {'nav': port_value, 'trades': n_trades, 'regime': regime}

    # ── 批量回放 ──

    def replay(self, start_date: str = None, end_date: str = None,
               verbose: bool = True, phase: int = 0,
               save: bool = True) -> dict:
        """批量回放历史数据

        逐日调用 update_daily()，用于:
        1. 初始化模拟盘 (从 TEST_START 回放到今天)
        2. 与 AdaptiveBacktester 结果对比验证

        Returns:
            绩效指标 dict
        """
        import multiprocessing
        try:
            multiprocessing.set_start_method('fork', force=True)
        except (ValueError, RuntimeError):
            pass  # Windows 无 fork，使用默认 spawn
        import qlib
        from qlib.data import D
        from qlib.constant import REG_CN
        try:
            qlib.init(provider_uri='~/.qlib/qlib_data/cn_data_bs', region=REG_CN)
        except Exception:
            pass  # 已初始化

        cfg = self.config
        start = start_date or cfg['test_start']
        end = end_date or cfg['test_end']

        if verbose:
            print(f"\n=== 模拟盘回放: {start} ~ {end} ===")
            print(f"初始资金: {cfg['initial_cash']:,.0f}")
            print(f"成交价: 收盘价 | 策略: {cfg['adaptive_strategy']}")
            print(f"TopK: strong={TOPK_BY_REGIME['strong']}, "
                  f"normal={TOPK_BY_REGIME['normal']}, "
                  f"weak={TOPK_BY_REGIME['weak']}")

        # 重置状态
        self.reset()

        # 加载数据
        t0 = time.time()
        if verbose:
            print("\n加载数据...", end=" ", flush=True)

        instruments = D.instruments('csi300')
        prices = D.features(instruments, ['$close'],
                            start_time=start, end_time=end)
        prices = prices.swaplevel().sort_index()

        signal = self.sg.load_predictions()
        quality = self.sg.load_quality_score()

        if verbose:
            print(f"完成 ({time.time()-t0:.1f}s)")

        # 交易日列表
        trading_days = sorted(prices.index.get_level_values(0).unique())
        if verbose:
            print(f"交易日: {len(trading_days)} 天")
            print(f"信号日: {len(signal.index.get_level_values(0).unique())} 天")

        # 逐日回放
        prev_close_map = {}
        total_trades = 0
        regime_counts = {'strong': 0, 'normal': 0, 'weak': 0}
        t_start = time.time()

        for day_idx, date in enumerate(trading_days):
            # phase: 把调仓日整体平移几个交易日。
            # 8 日调仓在 2.5 年上只有约 81 次调仓，起始日错开 1 天就是一组
            # 同样合理、但样本不同的结果。实测相位间 Sharpe 标准差 0.11~0.32，
            # 而候选参数之间的差异只有 0.1~0.16 —— 单次回测的数字读不出结论。
            result = self.update_daily(
                date, prices, prev_close_map, day_idx + phase, signal, quality
            )
            total_trades += result['trades']
            if result['regime']:
                regime_counts[result['regime']] += 1

            # 更新 prev_close_map
            if date in prices.index.get_level_values(0):
                day_close = prices.loc[date]['$close'].dropna()
                for inst in day_close.index:
                    prev_close_map[inst] = day_close[inst]

            # 进度显示
            if verbose and (day_idx + 1) % 100 == 0:
                nav = result['nav']
                ret = nav / cfg['initial_cash'] - 1
                print(f"  Day {day_idx+1}/{len(trading_days)}: "
                      f"{date.strftime('%Y-%m-%d')} NAV={nav:,.0f} "
                      f"({ret:+.2%}) 持仓={len(self.state['positions'])}")

        # 保存最终状态 (相位扫描时不落盘，否则会覆盖真实模拟盘状态)
        if save:
            self._save_state()

        elapsed = time.time() - t_start

        # 计算绩效
        perf = self.get_performance()
        perf['n_trades'] = total_trades
        perf['regime_counts'] = regime_counts
        perf['replay_time'] = round(elapsed, 1)

        if verbose:
            print(f"\n=== 回放完成 ({elapsed:.1f}s) ===")
            self._print_performance(perf)

        return perf

    # ── 绩效计算 ──

    def get_performance(self) -> dict:
        """从 daily_nav.csv 计算绩效指标"""
        nav_file = self.state_dir / "daily_nav.csv"
        if not nav_file.exists():
            return {'error': '无净值数据'}

        df = pd.read_csv(nav_file)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()

        # 去重 (replay 可能追加重复行)
        df = df[~df.index.duplicated(keep='last')]

        nav = df['nav']
        returns = nav.pct_change().dropna()
        initial = self.config['initial_cash']

        total_ret = nav.iloc[-1] / initial - 1
        n_days = len(returns)
        annual_ret = (1 + total_ret) ** (252 / max(n_days, 1)) - 1

        cumulative = (1 + returns).cumprod()
        drawdown = (cumulative - cumulative.cummax()) / cumulative.cummax()
        max_dd = float(drawdown.min())

        daily_std = float(returns.std())
        sharpe = float(returns.mean() / daily_std * (252 ** 0.5)) if daily_std > 0 else 0.0

        # 基准收益
        bench_ret = self._get_benchmark_return(
            df.index[0].strftime('%Y-%m-%d'),
            df.index[-1].strftime('%Y-%m-%d'),
        )

        return {
            'total_return': float(total_ret),
            'annual_return': float(annual_ret),
            'max_drawdown': max_dd,
            'sharpe': sharpe,
            'bench_return': bench_ret,
            'excess_return': float(total_ret) - bench_ret if bench_ret else None,
            'final_nav': float(nav.iloc[-1]),
            'trading_days': n_days + 1,
            'start_date': df.index[0].strftime('%Y-%m-%d'),
            'end_date': df.index[-1].strftime('%Y-%m-%d'),
        }

    def _get_benchmark_return(self, start: str, end: str) -> float | None:
        """获取基准 (沪深300) 收益率"""
        try:
            from qlib.data import D
            bench = D.features(['SH000300'], ['$close'],
                               start_time=start, end_time=end)
            bench_close = bench['$close'].droplevel(0)
            return float(bench_close.iloc[-1] / bench_close.iloc[0] - 1)
        except Exception:
            return None

    def _print_performance(self, perf: dict):
        """打印绩效摘要"""
        print(f"\n{'='*60}")
        print(f"  模拟盘绩效 ({perf.get('start_date','?')} ~ {perf.get('end_date','?')})")
        print(f"{'='*60}")
        print(f"  最终净值:   {perf.get('final_nav', 0):>14,.0f}")
        print(f"  总收益:     {perf.get('total_return', 0):>14.2%}")
        print(f"  年化收益:   {perf.get('annual_return', 0):>14.2%}")
        print(f"  Sharpe:     {perf.get('sharpe', 0):>14.3f}")
        print(f"  最大回撤:   {perf.get('max_drawdown', 0):>14.2%}")
        if perf.get('bench_return') is not None:
            print(f"  基准收益:   {perf['bench_return']:>14.2%}")
            print(f"  超额收益:   {perf.get('excess_return', 0):>14.2%}")
        print(f"  交易次数:   {perf.get('n_trades', 0):>14}")
        print(f"  交易日:     {perf.get('trading_days', 0):>14}")
        if perf.get('regime_counts'):
            rc = perf['regime_counts']
            print(f"  信号状态:   strong={rc.get('strong',0)} "
                  f"normal={rc.get('normal',0)} weak={rc.get('weak',0)}")
        if perf.get('replay_time'):
            print(f"  回放耗时:   {perf['replay_time']:>14.1f}s")
        print(f"{'='*60}")

    # ── 持仓/指令展示 ──

    def get_current_holdings(self) -> str:
        """人类可读的持仓明细"""
        pos = self.state['positions']
        if not pos:
            return "当前无持仓"

        lines = [f"现金: {self.state['cash']:,.0f}",
                 f"持仓 ({len(pos)} 只):"]
        lines.append(f"  {'代码':<12} {'股数':>8} {'成本':>10} {'入场日期':<12}")
        lines.append("  " + "-" * 48)
        for inst, p in sorted(pos.items()):
            lines.append(f"  {inst:<12} {p['shares']:>8} "
                         f"{p['cost_price']:>10.2f} {p.get('entry_date', 'N/A'):<12}")
        return "\n".join(lines)

    def get_trade_instructions(self) -> str:
        """人类可读的调仓指令 (飞书推送格式)"""
        pending = self.state['pending_orders']
        if not pending['sells'] and not pending['buys']:
            return "无待执行指令"

        lines = [f"待执行指令 (下一交易日收盘价执行):"]
        if pending['sells']:
            lines.append(f"  卖出 ({len(pending['sells'])} 只):")
            for inst in pending['sells']:
                pos = self.state['positions'].get(inst, {})
                lines.append(f"    {inst} {pos.get('shares', '?')} 股")
        if pending['buys']:
            lines.append(f"  买入 ({len(pending['buys'])} 只):")
            for inst, weight in pending['buys'].items():
                lines.append(f"    {inst} 目标权重 {weight:.2%}")
        return "\n".join(lines)


# ── CLI 入口 ──

def main():
    parser = argparse.ArgumentParser(description="模拟盘引擎")
    parser.add_argument('action', choices=['replay', 'status', 'reset'],
                        help="replay=回放历史, status=查看状态, reset=重置")
    parser.add_argument('--start', default=None, help="回放起始日期")
    parser.add_argument('--end', default=None,
                        help="回放结束日期，可传 today 表示当天(定时任务用)")
    args = parser.parse_args()

    # 定时任务无法在命令行里算出当天日期，这里支持字面量 today。
    # replay 内部会先 reset 再全量重放，所以每日重建是幂等的。
    if args.end == 'today':
        args.end = pd.Timestamp.today().strftime('%Y-%m-%d')

    pt = PaperTrader()

    if args.action == 'replay':
        result = pt.replay(start_date=args.start, end_date=args.end)
        # 保存绩效
        perf_file = pt.state_dir / "replay_performance.json"
        with open(perf_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n绩效已保存: {perf_file}")

    elif args.action == 'status':
        print(pt.get_current_holdings())
        print()
        print(pt.get_trade_instructions())
        perf = pt.get_performance()
        if 'error' not in perf:
            pt._print_performance(perf)

    elif args.action == 'reset':
        pt.reset()


if __name__ == '__main__':
    main()
