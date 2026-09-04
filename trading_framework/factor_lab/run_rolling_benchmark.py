#!/usr/bin/env python3
"""
Rolling Training Benchmark — Walk-Forward 滚动训练对比实验

对比不同 rolling 配置 × 因子预设 × 模型的表现，与 single-shot baseline 比较。

用法:
  # 快速测试 (1 config × 1 preset × 1 model)
  python -m factor_lab.run_rolling_benchmark --configs A_4yr_3v_3r --presets alpha158_val --models LightGBM

  # 核心对比 (4 configs × alpha158_val × LightGBM)
  python -m factor_lab.run_rolling_benchmark --presets alpha158_val --models LightGBM

  # 全量 (4 configs × 3 presets × 3 models)
  python -m factor_lab.run_rolling_benchmark

  # 强制重跑已有结果
  python -m factor_lab.run_rolling_benchmark --force
"""
import sys
import gc
import json
import time
import warnings
import argparse
import traceback
from pathlib import Path
from datetime import date
from dateutil.relativedelta import relativedelta

import pandas as pd

warnings.filterwarnings('ignore')

# 项目路径
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

# 测试区间 (与 benchmark_models.py 一致)
TEST_START = '2024-01-01'
TEST_END = '2026-02-05'
RESULT_SUFFIX = ''   # 由 --tag 设置，区分不同测试期的结果文件
# 成交价: 'close'=次日收盘 (默认), 'open'=次日开盘。由 --deal-price 覆盖。
#
# 2026-08-30 由 open 改为 close，理由 (one-switch 实验，其余变量全部固定):
#     open  Sharpe 1.643  超额 27.11%
#     close Sharpe 1.332  超额 11.90%   ← 超额腰斩
# 开盘价是实盘最难兑现的价格 (集合竞价冲击大、滑点高)，按开盘价成交属乐观
# 假设；且生产配置 signal_config.yaml 本就是 deal_price: close，研究口径
# 必须与之一致，否则挖掘会用生产不采用的执行假设去评判因子。
DEAL_PRICE = 'close'
# 已使用过的 tag，用于把带 tag 的结果排除出默认对比表。
# 新增 tag 时必须登记在这里，否则不同测试期/口径的结果会混进同一张表 ——
# 表头写着当前区间、数字却来自别的区间，极易误判。
KNOWN_TAGS = ('holdout', 'closeprice', 'openprice',
              'windowcmp', 'winseg1', 'seg2pred', 'pre2024')

# 从 signal_config 读取回测参数
import yaml as _yaml
with open(PROJECT_DIR / "config" / "signal_config.yaml", encoding='utf-8') as _f:
    _sig_cfg = _yaml.safe_load(_f)
INSTRUMENTS = _sig_cfg.get('instruments', 'csi300')

# 回测参数 (2026-09-03 改为读配置)
#
# 原先硬编码 TOPK=12 / N_DROP=3，而 signal_config 是 topk=8 / n_drop=2 ——
# 回测报出的绩效不是实盘实际会执行的那个策略，两者无从对比。
# instruments 早已读配置，TOPK/N_DROP 却没有，属于半截改造。
TOPK = int(_sig_cfg.get('topk', 12))
N_DROP = int(_sig_cfg.get('n_drop') if _sig_cfg.get('n_drop') is not None else 3)
# 成交价同样改为读配置 (原先在上方硬编码 'close')，理由见上方注释:
# 研究口径必须与生产一致，硬编码迟早会和配置分叉。
DEAL_PRICE = _sig_cfg.get('deal_price', DEAL_PRICE)

# Rolling 配置矩阵
ROLLING_CONFIGS = {
    "A_4yr_3v_3r": {
        "description": "4年滑动训练, 3个月验证, 3个月重训",
        "train_years": 4,
        "valid_months": 3,
        "retrain_months": 3,
        "expanding": False,
    },
    "B_4yr_3v_6r": {
        "description": "4年滑动训练, 3个月验证, 6个月重训",
        "train_years": 4,
        "valid_months": 3,
        "retrain_months": 6,
        "expanding": False,
    },
    "F_2yr_3v_3r": {
        "description": "2年滑动训练, 3个月验证, 3个月重训 (2026-08-30 新增)",
        # 加入原因: 窗口越短表现越好的趋势 (2024-01~2026-02 测试期):
        #   D_expand 1.332 -> A_4yr 1.609 -> C_3yr 1.943
        # 需要探明拐点在哪，否则不知道 3 年是最优还是仍未触底。
        "train_years": 2,
        "valid_months": 3,
        "retrain_months": 3,
        "expanding": False,
    },
    "C_3yr_3v_3r": {
        "description": "3年滑动训练, 3个月验证, 3个月重训",
        "train_years": 3,
        "valid_months": 3,
        "retrain_months": 3,
        "expanding": False,
    },
    "D_expand_3v_3r": {
        "description": "扩展窗口训练, 3个月验证, 3个月重训",
        "train_years": 4,  # 初始窗口大小, 之后持续扩展
        "valid_months": 3,
        "retrain_months": 3,
        "expanding": True,
    },
    "E_expand_6v_3r": {
        "description": "扩展窗口训练, 6个月验证, 3个月重训",
        "train_years": 4,
        "valid_months": 6,
        "retrain_months": 3,
        "expanding": True,
    },
}

# 默认参数
DEFAULT_CONFIGS = list(ROLLING_CONFIGS.keys())
DEFAULT_PRESETS = ['alpha158_val', 'alpha158', 'full']
DEFAULT_MODELS = ['LightGBM', 'XGBoost', 'CatBoost']

# 结果目录
RESULTS_DIR = PROJECT_DIR / "factor_lab" / "results" / "rolling"


def generate_rolling_windows(config_name: str, config: dict,
                             test_start: str, test_end: str) -> list[dict]:
    """生成 walk-forward 滚动窗口列表

    窗口从 test_start 的预测期开始，向后推导训练/验证期。
    每个窗口的 pred 区间长度 = retrain_months，最后一个窗口截断到 test_end。

    Args:
        config_name: 配置名
        config: 配置字典 (train_years, valid_months, retrain_months, expanding)
        test_start: 测试起始日
        test_end: 测试结束日

    Returns:
        list[dict]: 每个窗口的时间信息
    """
    ts = pd.Timestamp(test_start)
    te = pd.Timestamp(test_end)

    train_years = config['train_years']
    valid_months = config['valid_months']
    retrain_months = config['retrain_months']
    expanding = config['expanding']

    # 扩展窗口模式的固定起始点
    expand_start = ts - relativedelta(months=valid_months) - relativedelta(years=train_years)

    windows = []
    cursor = ts
    window_num = 0

    while cursor < te:
        window_num += 1

        # 预测区间
        pred_start = cursor
        pred_end_candidate = cursor + relativedelta(months=retrain_months) - relativedelta(days=1)
        pred_end = min(pred_end_candidate, te)

        # 验证区间: pred_start 前的 valid_months
        valid_end = pred_start - relativedelta(days=1)
        valid_start = pred_start - relativedelta(months=valid_months)

        # 训练区间
        train_end = valid_start - relativedelta(days=1)
        if expanding:
            train_start = expand_start
        else:
            train_start = valid_start - relativedelta(years=train_years)

        fmt = lambda d: d.strftime('%Y-%m-%d')
        windows.append({
            "window_num": window_num,
            "train_start": fmt(train_start),
            "train_end": fmt(train_end),
            "valid_start": fmt(valid_start),
            "valid_end": fmt(valid_end),
            "pred_start": fmt(pred_start),
            "pred_end": fmt(pred_end),
        })

        cursor = cursor + relativedelta(months=retrain_months)

    return windows


def make_rank_ic_feval(dates_by_nrow: dict):
    """构造 LightGBM feval: 按日截面 rank IC 的均值 (越大越好)

    2026-09-03 修复。原实现把所有交易日的样本混在一起算一个总的 Spearman:

        ic = pd.Series(preds).corr(pd.Series(labels), method='spearman')

    IC 的定义是【每天做截面相关、再对天数取平均】。混合计算会被跨日的水平差异
    主导，得到的数不是 IC。后果是 rank_ic 早停变体形同虚设 —— 实测它与 default
    变体产出逐位相同的结果，看着能用、实则无效。

    LightGBM 的 feval 只收到 (preds, dataset)，拿不到日期索引，这是原实现只能
    混合计算的原因。这里改为工厂函数，由调用方按行数把日期索引传进来。

    Args:
        dates_by_nrow: {样本行数: 该数据集的日期索引 (array-like)}。
                       用行数区分 train / valid 两个数据集。

    Returns:
        feval 函数
    """
    def _feval(preds, eval_data):
        labels = eval_data.get_label()
        dates = dates_by_nrow.get(len(labels))
        if dates is None:
            # 认不出是哪个数据集就不要给一个错的数 —— 返回 nan 让它显式暴露，
            # 而不是退回混合计算冒充 IC。
            return 'rank_ic', float('nan'), True
        df = pd.DataFrame({'p': preds, 'y': labels}, index=dates)
        ic = df.groupby(level=0).apply(
            lambda g: g['p'].corr(g['y'], method='spearman')).mean()
        return 'rank_ic', float(0.0 if pd.isna(ic) else ic), True

    return _feval


# 早停退化判定与兜底 (2026-09-03 新增)
#
# 现象: W8/W11 的 best_iteration=1 —— 模型只长一棵树, 预测分数几乎无区分度
# (300 只股票仅 18~27 个不同分数, TopK 里大量并列同分, 选谁本质是随机)。
#
# 根因: 验证集固定取【紧邻的上一个季度】。撞上无信号的季度(2026 Q2 的验证
# rank-IC 全程 ≈0, 2025 Q3 全程 ≈-0.02)时, 早停无法区分模型优劣, 于是停在第 1 轮。
# 但这并不代表【预测期】没有信号 —— 实测 W11 训练到 200 轮时预测期 IC 从
# 0.0279 升到 0.0446(翻倍), 区分度恢复到 299.8。也就是说早停掐死了有效模型。
#
# 兜底: 判定验证集不具区分能力时, 改用【历史窗口 best_iteration 的中位数】
# 作为固定轮数重训。只使用当时已有的信息, 无泄漏 —— 绝不能拿预测期表现挑轮数,
# 那就是拿测试集调参(CLAUDE.md 记录过的"用考题当练习册")。
MIN_HEALTHY_BEST_ITER = 20      # 低于此视为早停退化
FALLBACK_ROUNDS_DEFAULT = 200   # 无历史可依时的保底轮数
FALLBACK_ROUNDS_MIN = 50
FALLBACK_ROUNDS_MAX = 500


def fallback_rounds(prev_best_iters: list) -> int:
    """由历史窗口的 best_iteration 推导兜底轮数 (中位数, 裁剪到合理区间)

    只接受已完成窗口的结果 —— 这些在当前窗口训练时已经存在, 不构成前视。
    """
    healthy = sorted(b for b in (prev_best_iters or [])
                     if b is not None and b >= MIN_HEALTHY_BEST_ITER)
    if not healthy:
        return FALLBACK_ROUNDS_DEFAULT
    mid = healthy[len(healthy) // 2] if len(healthy) % 2 else \
        (healthy[len(healthy) // 2 - 1] + healthy[len(healthy) // 2]) // 2
    return int(max(FALLBACK_ROUNDS_MIN, min(FALLBACK_ROUNDS_MAX, mid)))


def fit_with_degradation_guard(model, dataset, fit_kwargs: dict,
                               prev_best_iters: list, model_name: str,
                               build_fn=None) -> tuple:
    """训练一个窗口; 早停退化时用历史中位数轮数重训

    Returns:
        (model, best_iteration, used_fallback)
    """
    model.fit(dataset, **fit_kwargs)
    best = getattr(getattr(model, 'model', None), 'best_iteration', None)
    if best is None or best >= MIN_HEALTHY_BEST_ITER:
        return model, best, False

    rounds = fallback_rounds(prev_best_iters)
    print(f"    ⚠ 早停退化 (best_iteration={best} < {MIN_HEALTHY_BEST_ITER}): "
          f"验证集无区分能力。改用历史中位数 {rounds} 轮重训。")

    # 关掉早停的正确做法: 构造一个【不含 valid 段】的 DatasetH。
    #
    # qlib 的 LGBModel._prepare_data 按 dataset.segments 决定 valid_sets:
    #     for key in ["train", "valid"]:
    #         if key in dataset.segments: ...
    # 没有 valid 段时只剩训练集，LightGBM 会自行禁用早停
    # ("Only training set found, disabling early stopping")，从而跑满轮数。
    #
    # 不能靠把 early_stopping_rounds 设得大于 num_boost_round 来绕过 ——
    # 实测无效: 只要早停回调存在，模型仍被截断到 best_iteration=1，
    # 树数只有 1 棵、预测与修复前逐位相同，兜底看着触发了实则没生效。
    from qlib.data.dataset import DatasetH

    builder = build_fn or build_model
    model2, kw2 = builder(model_name)

    segs = {k: v for k, v in dataset.segments.items() if k != "valid"}
    ds_no_valid = DatasetH(handler=dataset.handler, segments=segs)
    model2.fit(ds_no_valid, num_boost_round=rounds, **kw2)

    trees = getattr(model2.model, "current_iteration", lambda: None)()
    print(f"      兜底完成: {trees} 棵树 "
          f"(best_iteration={getattr(model2.model, 'best_iteration', None)}, "
          f"0 表示预测使用全部树)")
    if trees is not None and trees < rounds:
        print(f"      ⚠ 实际树数 {trees} 少于预期 {rounds}，早停可能仍在生效")
    return model2, rounds, True


# 随机种子 (2026-09-03 新增)
#
# 原先一个种子都没设，而 bagging_fraction=0.75 / feature_fraction=0.75 都依赖
# 随机抽样 —— 同样的代码、同样的数据，每次训练出的模型都不同。
#
# 实测这个不确定性有多大: 主段 (2024-01~2026-09) 用两次独立训练的预测回测，
#   预测相关系数 0.9002，TopK 8 平均只有 7.1/8 只重叠 (每次调仓差一只)
#   Sharpe 0.876 vs 1.449   —— 相差 0.57
# 也就是说单次回测的 Sharpe 有 ±0.3 量级的随机波动，
# 而我们在 TopK 扫描里比较的差异只有 0.1~0.3 —— 完全淹没在噪声里。
#
# CLAUDE.md 记录过同类事故(集合迭代顺序导致 Sharpe 0.318 vs 0.247)并修了排序，
# 但模型种子这一路一直没堵上。
#
# seed 会派生 bagging_seed / feature_fraction_seed 等；deterministic=True
# 进一步关闭 LightGBM 内部依赖线程调度的非确定性路径。
_SEED_PARAMS = {"seed": 42, "deterministic": True}

# 集成种子数: 训练 N 个模型取预测均值。
# 光设固定种子只解决"可复现"，不解决"单次结果本身是一次抽样" ——
# 实盘信号会取决于某个任意选定的种子。取多个种子的均值能把这部分方差
# 按 ~1/sqrt(N) 压下去，同时让每次调仓的选股更稳定。
# 代价是训练时间线性增加，故默认 1 (与原行为一致)，由配置开启。
N_SEEDS = int(_sig_cfg.get('n_seeds', 1))


def _seed_params(seed: int | None = None) -> dict:
    """LightGBM 随机性参数; seed 为 None 时用默认种子"""
    p = dict(_SEED_PARAMS)
    if seed is not None:
        p["seed"] = int(seed)
    return p


def ensemble_seeds(n: int | None = None) -> list[int]:
    """集成使用的种子列表 (确定性派生自基准种子)"""
    n = N_SEEDS if n is None else n
    base = int(_SEED_PARAMS["seed"])
    return [base + i * 1000 for i in range(max(1, int(n)))]


def train_window(dataset, model_name: str, prev_best_iters: list,
                 variant: str = "default", dates_by_nrow: dict | None = None,
                 n_seeds: int | None = None) -> tuple:
    """训练一个窗口并返回预测 —— 回测器与重训 pipeline 共用

    2026-09-03 新增。此前两条训练路径各写各的，导致:
      - 退化兜底只接在 retrain_pipeline，run_rolling_benchmark 没有 ——
        段1/主段的验证数字是在 W8/W11 只有 1 棵树的情况下跑出来的
      - 随机性未受控，同配置两次训练 Sharpe 0.876 vs 1.449

    多种子集成: 训练 n_seeds 个模型取预测均值。固定种子只解决可复现，
    集成才能真正压低"单次抽样"的方差 (~1/sqrt(N))，并让实盘选股稳定。

    Returns:
        (pred, best_iteration, used_fallback, n_models)
    """
    seeds = ensemble_seeds(n_seeds)
    preds, bests, fb_any = [], [], False
    for i, sd in enumerate(seeds):
        model, fit_kwargs = build_model(model_name, variant=variant,
                                        dates_by_nrow=dates_by_nrow, seed=sd)
        model, best, used_fb = fit_with_degradation_guard(
            model, dataset, fit_kwargs, prev_best_iters, model_name)
        fb_any = fb_any or used_fb
        bests.append(best)
        preds.append(model.predict(dataset))
        if len(seeds) > 1:
            print(f"      [seed {i+1}/{len(seeds)}] best_iter={best}")
        del model
    pred = preds[0] if len(preds) == 1 else sum(preds) / len(preds)
    valid = [b for b in bests if b is not None]
    best_iter = int(sum(valid) / len(valid)) if valid else None   # 均值便于记录
    return pred, best_iter, fb_any, len(seeds)


def build_model(model_name: str, variant: str = "default",
                dates_by_nrow: dict | None = None,
                seed: int | None = None):
    """创建模型实例 (超参与 benchmark_models.py 一致)

    Args:
        model_name: 模型名
        variant: "default" 或 "rank_ic" (rank-IC 早停)
        dates_by_nrow: variant="rank_ic" 时必填，{样本行数: 日期索引}，
                       用于在 feval 里按日分组算截面 IC

    Returns:
        (model, fit_kwargs) 元组
    """
    if model_name == 'LightGBM':
        from qlib.contrib.model.gbdt import LGBModel
        if variant == "rank_ic":
            if not dates_by_nrow:
                raise ValueError(
                    "variant='rank_ic' 需要 dates_by_nrow 才能按日计算截面 IC。"
                    "缺少它就只能把所有交易日混在一起算，那不是 IC —— "
                    "旧实现正是如此，导致该变体形同虚设。")
            model = LGBModel(
                loss="mse", metric="None",
                learning_rate=0.01, num_leaves=64,
                num_boost_round=500, early_stopping_rounds=80,
                feature_fraction=0.75, bagging_fraction=0.75, bagging_freq=5,
                lambda_l1=0.1, lambda_l2=0.1, min_data_in_leaf=80,
                **_seed_params(seed),
            )
            return model, {"feval": make_rank_ic_feval(dates_by_nrow)}
        model = LGBModel(
            loss="mse", learning_rate=0.01, num_leaves=64,
            num_boost_round=500, early_stopping_rounds=80,
            feature_fraction=0.75, bagging_fraction=0.75, bagging_freq=5,
            lambda_l1=0.1, lambda_l2=0.1, min_data_in_leaf=80,
            **_seed_params(seed),
        )
        return model, {}

    elif model_name == 'XGBoost':
        from qlib.contrib.model.xgboost import XGBModel
        model = XGBModel(
            learning_rate=0.01, max_depth=6,
            n_estimators=500, early_stopping_rounds=80,
            reg_alpha=0.1, reg_lambda=0.1,
            subsample=0.75, colsample_bytree=0.75,
        )
        return model, {}

    elif model_name == 'CatBoost':
        from qlib.contrib.model.catboost_model import CatBoostModel
        model = CatBoostModel(
            loss="RMSE", learning_rate=0.01, depth=6,
            iterations=500, l2_leaf_reg=3.0, subsample=0.75,
        )
        return model, {"early_stopping_rounds": 80}

    else:
        raise ValueError(f"未知模型: {model_name}")


def _run_backtest_hifi(pred):
    """高保真回测 — 口径与实盘一致，参数全部来自 signal_config.yaml"""
    from qlib.data import D
    from factor_lab.run_vol_target import VolTargetBacktester

    dates = pred.index.get_level_values(0)
    start, end = dates.min(), dates.max()

    # 行情: 带 $isST 以支持 ST 过滤; 数据集没有该字段时优雅降级
    try:
        prices = D.features(D.instruments(INSTRUMENTS), ['$close', '$isST'],
                            start_time=start.strftime('%Y-%m-%d'),
                            end_time=end.strftime('%Y-%m-%d'))
    except Exception:
        prices = D.features(D.instruments(INSTRUMENTS), ['$close'],
                            start_time=start.strftime('%Y-%m-%d'),
                            end_time=end.strftime('%Y-%m-%d'))

    # D.features 返回 (instrument, datetime)，而 VolTargetBacktester 用
    # prices_df.loc[date] 按日取截面，需要 datetime 在第 0 层。
    # 漏掉这步不会报错 —— loc[date] 会按 instrument 匹配，交集为空，
    # 回测跑完但一笔交易都没有，指标全 0。属于典型的静默失败。
    prices = prices.swaplevel().sort_index()

    # 显式自检: 信号日与行情日必须有交集，否则回测会"跑完但没交易"、
    # 指标全 0 且不报错。这次正是靠新旧引擎对照才发现，不能依赖运气。
    _overlap = len(set(prices.index.get_level_values(0).unique())
                   & set(dates.unique()))
    if _overlap == 0:
        raise RuntimeError(
            f"行情与信号无共同交易日 (行情 "
            f"{prices.index.get_level_values(0).nunique()} 天 / 信号 "
            f"{dates.nunique()} 天)，回测无法进行。"
            f"常见原因: MultiIndex 层级顺序不符 (需 datetime 在第 0 层)。")

    bt = VolTargetBacktester(
        signal=pred,
        initial_cash=float(_sig_cfg.get('initial_cash', 100_000)),
        topk=TOPK,
        rebalance_every=int(_sig_cfg.get('rebalance_every', 5)),
        stop_loss=float(_sig_cfg.get('stop_loss', 0.08)),
        open_cost=float(_sig_cfg.get('open_cost', 0.0005)),
        close_cost=float(_sig_cfg.get('close_cost', 0.0015)),
        target_vol=_sig_cfg.get('vol_target'),
        vol_window=int(_sig_cfg.get('vol_window', 20)),
        min_exposure=float(_sig_cfg.get('vol_min_exposure', 0.2)),
        n_drop=N_DROP,
    )
    result = bt.run(prices)

    # 基准超额: 与旧引擎的输出字段保持一致，调用方无需改动
    try:
        bench = D.features(['SH000300'], ['$close'],
                           start_time=start.strftime('%Y-%m-%d'),
                           end_time=end.strftime('%Y-%m-%d'))['$close']
        bench = bench.droplevel(0) if bench.index.nlevels > 1 else bench
        bench = bench.dropna()
        if len(bench) > 1:
            result['bench_return'] = float(bench.iloc[-1] / bench.iloc[0] - 1)
            result['excess_return'] = result['total_return'] - result['bench_return']
    except Exception as e:
        print(f"  [backtest] 基准收益计算失败(不影响策略指标): {e}")
    return result


def run_backtest(pred, engine: str | None = None):
    """统一回测入口

    2026-09-04: 默认改用高保真引擎 (VolTargetBacktester)。
    ================================================================
    此前本函数用 qlib 原生 TopkDropoutStrategy，参数只有 topk/n_drop ——
    **不含 vol_target**。而实盘、模拟盘都应用波动率目标，于是主力验证工具
    跑的不是实盘会执行的策略。实测同一区间、同一份预测:

        qlib 原生 (无 vol_target):  Sharpe 1.005  回撤 -28.40%
        模拟盘   (有 vol_target):  Sharpe 1.143  回撤  -9.69%

    回撤差了三倍。而三段验证、TopK 扫描、因子挖掘的 beat_baseline 判据
    全部走这个函数 —— 等于用一个没开风控的引擎去评判要不要上线因子。

    高保真引擎的口径与实盘一致: 10万资金 / 整手 100 股 / 涨跌停 ±9.5% /
    双边成本 / 止损 / n_drop 换手限制 / vol_target 敞口缩放 / ST 过滤。

    engine="qlib" 可显式回到旧引擎 (仅供口径对照，不要用于验收)。
    """
    if (engine or "hifi") != "qlib":
        return _run_backtest_hifi(pred)

    from qlib.contrib.evaluate import backtest_daily
    from qlib.utils import init_instance_by_config

    strategy_config = {
        "class": "TopkDropoutStrategy",
        "module_path": "qlib.contrib.strategy",
        "kwargs": {"signal": pred, "topk": TOPK, "n_drop": N_DROP},
    }
    backtest_config = {
        "start_time": TEST_START, "end_time": TEST_END,
        # 资金规模改为读配置 (原先硬编码 1 亿)。
        # 1 亿资金下整手约束形同不存在，TopK 越大失真越严重: TopK=20 时
        # 每仓 500 万，任何股票都买得进; 而 10 万本金下每仓仅 5000 元，
        # 28% 的沪深300 成分股一手就超过这个数，根本建不了仓。
        # 回测口径必须与实际资金一致，否则得出的"最优 TopK"在实盘无法执行。
        #
        # ⚠ 仍有一层未解决: qlib 需要 `$factor` 字段才会应用 trade_unit(整手)，
        #   我们的数据没有 factor.day.bin，qlib 回退到 adjusted_price 模式并
        #   显式警告 "trade unit 100 is not supported"。也就是说当前回测仍允许
        #   买零碎股。要真正模拟整手，需在数据里补 factor 字段(见 data_setup_sina)。
        "account": float(_sig_cfg.get('initial_cash', 100_000_000)),
        "benchmark": "SH000300",
        "exchange_kwargs": {
            "freq": "day", "limit_threshold": 0.095, "deal_price": DEAL_PRICE,
            "open_cost": 0.0005, "close_cost": 0.0015, "min_cost": 5, "trade_unit": 100,
        },
    }
    strategy = init_instance_by_config(strategy_config)
    report, positions = backtest_daily(strategy=strategy, executor=None, **backtest_config)

    result = {}
    if 'return' in report.columns:
        returns = report['return']
        total_ret = (1 + returns).prod() - 1
        result['total_return'] = float(total_ret)
        n_days = len(returns)
        result['annual_return'] = float((1 + total_ret) ** (252 / max(n_days, 1)) - 1)
        cumulative = (1 + returns).cumprod()
        drawdown = (cumulative - cumulative.cummax()) / cumulative.cummax()
        result['max_drawdown'] = float(drawdown.min())
        daily_std = float(returns.std())
        result['sharpe'] = float(returns.mean() / daily_std * (252 ** 0.5)) if daily_std > 0 else 0.0
        # 保留日收益序列: DSR/PBO 需要序列而不只是汇总指标。
        # 汇总指标丢掉了偏度/峰度和时间结构，无法做多重检验校正。
        result['daily_returns'] = [float(x) for x in returns.tolist()]
    if 'bench' in report.columns:
        result['bench_return'] = float((1 + report['bench']).prod() - 1)
        result['excess_return'] = result.get('total_return', 0) - result['bench_return']
    return result


def run_rolling_single(preset: str, model_name: str,
                       config_name: str, config: dict,
                       force: bool = False,
                       variant: str = "default") -> dict | None:
    """执行一个 (preset × model × config) 的滚动训练

    Args:
        preset: 因子预设名
        model_name: 模型名
        config_name: rolling 配置名
        config: rolling 配置字典
        force: 是否强制重跑
        variant: 模型变体 ("default" 或 "rank_ic")

    Returns:
        结果字典, 或 None (已有结果且非 force)
    """
    suffix = f"_{variant}" if variant != "default" else ""
    if RESULT_SUFFIX:
        suffix += f"_{RESULT_SUFFIX}"
    result_file = RESULTS_DIR / f"{config_name}_{preset}_{model_name}{suffix}.json"

    if result_file.exists() and not force:
        print(f"  [跳过] 已有结果: {result_file.name}")
        with open(result_file, encoding='utf-8') as f:
            return json.load(f)

    windows = generate_rolling_windows(config_name, config, TEST_START, TEST_END)
    print(f"\n{'='*70}")
    print(f"  Config: {config_name} ({config['description']})")
    print(f"  Preset: {preset} | Model: {model_name}")
    print(f"  Windows: {len(windows)}")
    print(f"{'='*70}")

    # 打印窗口时间线
    for w in windows:
        print(f"  Window {w['window_num']}: "
              f"Train [{w['train_start']}, {w['train_end']}]  "
              f"Valid [{w['valid_start']}, {w['valid_end']}]  "
              f"Pred [{w['pred_start']}, {w['pred_end']}]")

    all_preds = []
    window_details = []
    total_start = time.time()

    for w in windows:
        wnum = w['window_num']
        print(f"\n  --- Window {wnum}/{len(windows)} ---")

        t0 = time.time()

        # 1. 构建 handler
        if preset == 'alpha158':
            from qlib.contrib.data.handler import Alpha158
            handler = Alpha158(
                start_time=w['train_start'],
                end_time=w['pred_end'],
                fit_start_time=w['train_start'],
                fit_end_time=w['train_end'],
                instruments=INSTRUMENTS,
            )
        else:
            from factor_lab.factors.presets import build_handler
            handler = build_handler(
                preset,
                start_time=w['train_start'],
                end_time=w['pred_end'],
                fit_start_time=w['train_start'],
                fit_end_time=w['train_end'],
                instruments=INSTRUMENTS,
            )

        # 2. 构建 DatasetH
        from qlib.data.dataset import DatasetH
        dataset = DatasetH(handler=handler, segments={
            "train": (w['train_start'], w['train_end']),
            "valid": (w['valid_start'], w['valid_end']),
            "test": (w['pred_start'], w['pred_end']),
        })

        # 3. 构建模型并训练
        dates_by_nrow = None
        if variant == "rank_ic":
            # feval 拿不到日期索引，按行数把 train/valid 的日期传进去
            from qlib.data.dataset.handler import DataHandlerLP
            dates_by_nrow = {}
            for seg in ("train", "valid"):
                d = dataset.prepare(seg, col_set=["feature", "label"],
                                    data_key=DataHandlerLP.DK_L)
                y = d["label"].iloc[:, 0]
                dates_by_nrow[len(y)] = y.index.get_level_values(0)
        pred, _win_best, _win_fb, _n_models = train_window(
            dataset, model_name,
            prev_best_iters=[d.get('best_iteration') for d in window_details],
            variant=variant, dates_by_nrow=dates_by_nrow)

        # 4. 过滤到 pred 区间
        if isinstance(pred.index, pd.MultiIndex):
            dates = pred.index.get_level_values(0)
            mask = (dates >= pd.Timestamp(w['pred_start'])) & \
                   (dates <= pd.Timestamp(w['pred_end']))
            pred = pred[mask]

        elapsed = time.time() - t0

        # 记录窗口信息 (best_iteration 由 train_window 汇总，多种子时取均值)
        detail = {
            "window_num": wnum,
            "train_start": w['train_start'],
            "train_end": w['train_end'],
            "pred_start": w['pred_start'],
            "pred_end": w['pred_end'],
            "n_samples": int(len(pred)),
            "best_iteration": _win_best,
            "fallback_rounds": bool(_win_fb),
            "n_models": _n_models,
            "train_time": round(elapsed, 1),
        }
        window_details.append(detail)
        all_preds.append(pred)

        print(f"    Best iter: {_win_best}{' [兜底]' if _win_fb else ''}, "
              f"models: {_n_models}, Samples: {len(pred)}, Time: {elapsed:.1f}s")

        # 5. 释放内存 (model 已在 train_window 内部释放)
        del handler, dataset, pred
        gc.collect()

    # 拼接所有窗口预测
    combined_pred = pd.concat(all_preds)

    # 检查无重复 index
    if combined_pred.index.duplicated().any():
        n_dup = combined_pred.index.duplicated().sum()
        print(f"  [警告] 发现 {n_dup} 个重复 index, 保留最后一个窗口的预测")
        combined_pred = combined_pred[~combined_pred.index.duplicated(keep='last')]

    print(f"\n  总预测样本: {len(combined_pred)}")

    # 落盘预测: 事后做 IC/归因分析必需 (原先只在内存中用完即弃，
    # 想查"某段时间为何表现差"就得整轮重训)
    pred_dir = RESULTS_DIR / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_path = pred_dir / f"{config_name}_{preset}_{model_name}{suffix}.pkl"
    combined_pred.to_pickle(pred_path)
    print(f"  预测已保存: {pred_path.name}")

    # 回测
    print("  运行回测...")
    bt_result = run_backtest(combined_pred)
    total_time = time.time() - total_start

    result = {
        "config_name": config_name,
        "config_description": config['description'],
        "preset": preset,
        "model": model_name,
        "variant": variant,
        "n_windows": len(windows),
        "windows": window_details,
        "overall": bt_result,
        # 参数快照 (2026-09-04 新增)
        # 回测参数在模块导入时读一次，之后改配置不影响已运行的进程。
        # 不记录快照就无法事后分辨某个结果文件是哪套配置跑出来的 ——
        # 这正是 CLAUDE.md 列为头号风险的"表头与数字不符"。
        "params": {
            "topk": TOPK,
            "n_drop": N_DROP,
            "deal_price": DEAL_PRICE,
            "instruments": INSTRUMENTS,
            "engine": "hifi",
            "initial_cash": _sig_cfg.get("initial_cash"),
            "rebalance_every": _sig_cfg.get("rebalance_every"),
            "vol_target": _sig_cfg.get("vol_target"),
            "stop_loss": _sig_cfg.get("stop_loss"),
            "n_seeds": N_SEEDS,
            "test_start": TEST_START,
            "test_end": TEST_END,
        },
        "total_time": round(total_time, 1),
        "timestamp": pd.Timestamp.now().isoformat(),
    }

    # 保存结果
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n  结果: Sharpe={bt_result.get('sharpe', 0):.3f}  "
          f"Return={bt_result.get('total_return', 0):.2%}  "
          f"MDD={bt_result.get('max_drawdown', 0):.2%}  "
          f"Time={total_time:.1f}s")
    print(f"  保存: {result_file.name}")

    return result


def print_comparison_table(all_results: list[dict]):
    """打印对比表"""
    print("\n\n")
    print("=" * 110)
    print("                     Rolling Training Benchmark 对比表")
    print("=" * 110)
    print(f"测试区间: {TEST_START} ~ {TEST_END} | 基准: 沪深300")
    _dp = "次日开盘" if DEAL_PRICE == "open" else "次日收盘"
    # 表头必须反映实际口径: 引擎已换成高保真版，旧表头写的是 qlib 原生策略名，
    # 且不显示 vol_target / 调仓频率 —— 看表的人会以为跑的是另一套配置。
    _vt = _sig_cfg.get('vol_target')
    print(f"引擎: hifi (整手/涨跌停/双边成本/止损) | "
          f"topk={TOPK} n_drop={N_DROP} 调仓{_sig_cfg.get('rebalance_every')}日 | "
          f"vol_target={f'{_vt:.0%}' if _vt else '关闭'} | "
          f"资金={_sig_cfg.get('initial_cash'):,.0f} | 种子={N_SEEDS}")
    print(f"成交价: {_dp} ({DEAL_PRICE}, T+1) | 基准: 沪深300")
    print()

    # 按 Sharpe 降序排列
    sorted_results = sorted(all_results,
                            key=lambda x: x['overall'].get('sharpe', 0),
                            reverse=True)

    header = (f"{'排名':<4} {'Config':<16} {'Preset':<16} {'Model':<12} {'Variant':<10} "
              f"{'Windows':>7} {'总收益':>10} {'年化':>10} {'超额':>10} "
              f"{'Sharpe':>8} {'最大回撤':>10} {'耗时(s)':>8}")
    print(header)
    print("-" * 120)

    for rank, r in enumerate(sorted_results, 1):
        o = r['overall']
        print(f"{rank:<4} {r['config_name']:<16} {r['preset']:<16} {r['model']:<12} "
              f"{r.get('variant', 'default'):<10} "
              f"{r['n_windows']:>7} {o.get('total_return', 0):>9.2%} "
              f"{o.get('annual_return', 0):>9.2%} "
              f"{o.get('excess_return', 0):>9.2%} "
              f"{o.get('sharpe', 0):>7.3f} "
              f"{o.get('max_drawdown', 0):>9.2%} "
              f"{r.get('total_time', 0):>7.1f}")

    # Single-shot baseline 参考
    print()
    print("--- Single-Shot Baseline (参考) ---")
    baseline_file = PROJECT_DIR / "benchmark_results" / "results.json"
    if baseline_file.exists():
        with open(baseline_file, encoding='utf-8') as f:
            baselines = json.load(f)
        for model_name in ['LightGBM', 'XGBoost', 'CatBoost']:
            if model_name in baselines and 'total_return' in baselines[model_name]:
                b = baselines[model_name]
                print(f"     {'single-shot':<16} {'alpha158_val':<16} {model_name:<12} "
                      f"{'1':>7} {b.get('total_return', 0):>9.2%} "
                      f"{b.get('annual_return', 0):>9.2%} "
                      f"{b.get('excess_return', 0):>9.2%} "
                      f"{b.get('sharpe', 0):>7.3f} "
                      f"{b.get('max_drawdown', 0):>9.2%} "
                      f"{b.get('train_time', 0):>7.1f}")
    else:
        print("  (未找到 benchmark_results/results.json)")

    print("\n" + "=" * 110)


def load_all_results() -> list[dict]:
    """加载当前测试期对应的结果

    只读与 RESULT_SUFFIX 匹配的文件：否则切换测试期(--test-start/--test-end)
    时会把旧测试期的结果混进对比表，表头写着新区间、数字却是旧的，
    极易误判。
    """
    results = []
    if not RESULTS_DIR.exists():
        return results
    for f in sorted(RESULTS_DIR.glob("*.json")):
        stem = f.stem
        if RESULT_SUFFIX:
            if not stem.endswith(f"_{RESULT_SUFFIX}"):
                continue
        elif any(stem.endswith(f"_{t}") for t in KNOWN_TAGS):
            continue  # 默认测试期不应混入带 tag 的结果
        with open(f, encoding='utf-8') as fp:
            results.append(json.load(fp))

    # 参数一致性校验 (2026-09-04 新增)
    # tag 只区分测试期，不区分回测参数。同一张表里若混入不同 topk/n_drop/
    # 调仓频率/vol_target 的结果，表头写着当前配置、数字却来自别的配置 ——
    # CLAUDE.md 把这类"表头与数字不符"列为头号风险。
    _cur = {"topk": TOPK, "n_drop": N_DROP,
            "rebalance_every": _sig_cfg.get("rebalance_every"),
            "vol_target": _sig_cfg.get("vol_target"),
            "deal_price": DEAL_PRICE}
    for r in results:
        p = r.get("params")
        if p is None:
            print(f"  [警告] {r.get('config_name')}/{r.get('preset')} "
                  f"无参数快照(旧结果)，无法确认口径是否与当前一致")
            continue
        diff = {k: (p.get(k), v) for k, v in _cur.items() if p.get(k) != v}
        if diff:
            desc = ", ".join(f"{k}: 结果={a} 当前={b}" for k, (a, b) in diff.items())
            print(f"  [警告] {r.get('config_name')}/{r.get('preset')} "
                  f"参数与当前配置不符 — {desc}")
    return results


def main():
    global TEST_START, TEST_END, RESULT_SUFFIX, DEAL_PRICE
    parser = argparse.ArgumentParser(description="Rolling Training Benchmark")
    parser.add_argument('--configs', nargs='+', default=DEFAULT_CONFIGS,
                        choices=list(ROLLING_CONFIGS.keys()),
                        help=f"Rolling 配置 (默认: 全部)")
    parser.add_argument('--presets', nargs='+', default=DEFAULT_PRESETS,
                        help=f"因子预设 (默认: {DEFAULT_PRESETS})")
    parser.add_argument('--models', nargs='+', default=DEFAULT_MODELS,
                        choices=DEFAULT_MODELS,
                        help=f"模型 (默认: {DEFAULT_MODELS})")
    parser.add_argument('--force', action='store_true',
                        help="强制重跑已有结果")
    parser.add_argument('--report-only', action='store_true',
                        help="只打印对比表, 不跑实验")
    parser.add_argument('--test-start', default=None,
                        help=f"覆盖测试期起点 (默认 {TEST_START})，用于样本外holdout检验")
    parser.add_argument('--test-end', default=None,
                        help=f"覆盖测试期终点 (默认 {TEST_END})")
    parser.add_argument('--tag', default='',
                        help="结果文件名后缀，避免覆盖默认测试期的结果")
    parser.add_argument('--deal-price', default=None, choices=['open', 'close'],
                        help="成交价 (默认 open=次日开盘)。one-switch 实验用: "
                             "固定其他所有变量，只切换此项以隔离执行假设的影响")
    args = parser.parse_args()

    if args.deal_price:
        DEAL_PRICE = args.deal_price

    # 覆盖测试区间（挖掘的评估期 = 默认测试期，故重测默认期属样本内；
    # 用 --test-start/--test-end 指定挖掘未见过的区间才是真正的样本外检验）
    if args.test_start:
        TEST_START = args.test_start
    if args.test_end:
        TEST_END = args.test_end
    RESULT_SUFFIX = args.tag

    if args.report_only:
        all_results = load_all_results()
        if all_results:
            print_comparison_table(all_results)
        else:
            print("没有找到任何结果文件")
        return

    # 初始化 Qlib
    import multiprocessing
    try:
        multiprocessing.set_start_method('fork', force=True)
    except (ValueError, RuntimeError):
        pass  # Windows 无 fork，使用默认 spawn

    import qlib
    from qlib.constant import REG_CN
    _data_dir = f"~/.qlib/qlib_data/cn_data_{'bs' if INSTRUMENTS == 'csi300' else INSTRUMENTS}"
    qlib.init(provider_uri=_data_dir, region=REG_CN)

    # 计算总任务数
    total = len(args.configs) * len(args.presets) * len(args.models)
    print(f"\n{'='*70}")
    print(f"Rolling Training Benchmark")
    print(f"Configs: {args.configs}")
    print(f"Presets: {args.presets}")
    print(f"Models:  {args.models}")
    print(f"总任务: {total}")
    print(f"{'='*70}")

    all_results = []
    task_num = 0

    for config_name in args.configs:
        config = ROLLING_CONFIGS[config_name]
        for preset in args.presets:
            for model_name in args.models:
                task_num += 1
                print(f"\n\n[Task {task_num}/{total}]")

                try:
                    result = run_rolling_single(
                        preset, model_name, config_name, config,
                        force=args.force,
                    )
                    if result:
                        all_results.append(result)
                except Exception as e:
                    print(f"  [失败] {e}")
                    traceback.print_exc()
                    error_result = {
                        "config_name": config_name,
                        "preset": preset,
                        "model": model_name,
                        "error": str(e),
                        "overall": {},
                        "n_windows": 0,
                        "total_time": 0,
                    }
                    all_results.append(error_result)

    # 本次运行的失败必须显式暴露，不能被"打印历史结果"掩盖
    failed = [r for r in all_results if r.get("error")]

    # 打印对比表 (只含当前测试期的结果)
    all_results = load_all_results()
    if all_results:
        print_comparison_table(all_results)

    if failed:
        print(f"\n⚠ 本次运行有 {len(failed)} 个任务失败，上表不含它们的结果:")
        for r in failed:
            print(f"    {r['config_name']} / {r['preset']} / {r['model']}: {r['error']}")
        sys.exit(1)


if __name__ == '__main__':
    main()
