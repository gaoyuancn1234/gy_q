"""调仓风控规则 — 回测/实盘/模拟盘共用的唯一实现

2026-09-03 新增。此前 n_drop 与 vol_target 只在 portfolio/live_portfolio.py 里
实现，factor_lab/paper_trader.py 完全没有 —— 两个引擎跑的是不同策略:

    控制项            回测器   实盘   模拟盘(修复前)
    n_drop 换手限制     有      有        无
    vol_target         有      有        无

后果是模拟盘全量换手(5 日调仓下几乎每次换掉全部持仓)，10万资金 2.5 年约 1734 笔
交易、双边成本累积约本金的 14%，绩效会系统性低于回测与实盘。而 retrain_pipeline
的 Step 5 "验证" 正是跑 PaperTrader.replay —— 等于在验证一个与实盘不同的策略。

分叉的根因是同一规则被写了两遍，所以这里抽成纯函数，两边共用。新增引擎请复用本模块，
不要再各写一份。
"""

# A 股年均交易日数。沿用 live_portfolio 原值 242 (美股惯例是 252)，
# 改动会静默改变已实现波动率的年化结果与实盘敞口。
TRADING_DAYS = 242

# 无法估计已实现波动率时使用的敞口 (净值历史不足 / 序列异常 / 取价失败)。
# 取 0.6 而非 1.0: 风控在信息缺失时必须收缩而非放开。见 compute_exposure 注释。
UNKNOWN_VOL_EXPOSURE = 0.6


def select_sells(current: set, target: set, scores: dict,
                 n_drop: int | None) -> set:
    """按换手限制挑选卖出标的

    n_drop 为 None 时退回全量换手(旧行为)。否则只卖出"已不在目标里"的持仓中
    分数最低的 n_drop 只 —— 分数越低越该先走。

    实测依据 (CLAUDE.md / signal_config.yaml):
        全量换手  Sharpe 0.24(段1) / 0.48(2024-26)
        n_drop=2  Sharpe 1.27(段1) / 1.16(2024-26)

    Args:
        current: 当前持仓 instrument 集合
        target:  目标持仓 instrument 集合
        scores:  {instrument: 信号分数}，缺失的视为最差(信号已不覆盖该股)
        n_drop:  单次调仓最多卖出几只；None = 不限制

    Returns:
        应卖出的 instrument 集合
    """
    all_out = set(current) - set(target)
    if n_drop is None or not all_out:
        return all_out
    scores = scores or {}
    # 显式排序: 直接迭代集合会因字符串哈希随机化导致同配置两次结果不同
    # (CLAUDE.md 记录过 Sharpe 0.318 vs 0.247 的复现失败)
    ranked = sorted(all_out, key=lambda c: (scores.get(c, float('-inf')), c))
    return set(ranked[:int(n_drop)])


def compute_exposure(navs: list, target_vol: float | None,
                     window: int = 20, min_exposure: float = 0.2) -> tuple:
    """按已实现波动率反向缩放权益敞口 (不加杠杆)

    exposure = clip(target_vol / realized_vol, min_exposure, 1.0)

    诊断依据: TopK 等权是固定风险敞口，市场波动翻倍时组合风险随之翻倍。
    2026-02~08 正是如此 —— 选股能力未衰减(超额日均 +0.09%，与 2024 相当)，
    但组合日波动从 1.14% 升至 2.15%，Sharpe 因此崩塌。

    Args:
        navs: 按时间升序的净值序列
        target_vol: 年化目标波动率；None/0 表示关闭
        window: 计算已实现波动率的回看天数
        min_exposure: 敞口下限

    Returns:
        (exposure, realized_vol) — 无法估计波动率时 realized_vol 为 None，
        exposure 取 UNKNOWN_VOL_EXPOSURE。

    ⚠ 关于"估不出来时给多少敞口" (2026-09-05):
    原实现四处都 `return 1.0` —— 即**不确定就满仓**。方向是反的:
      - 净值取价失败 -> 净值序列被成本价平滑 -> 波动率算成 ~0 -> 满仓
      - 刚建仓/刚重置 -> 历史不足 -> 满仓
    也就是数据故障或状态未知时，这道风控自动失效并放大仓位。
    风控在信息缺失时应当收缩。改为 UNKNOWN_VOL_EXPOSURE (默认 0.6)，
    即先用六成仓位运行，等净值序列攒够再按实测波动率调整。
    """
    if not target_vol:
        return 1.0, None                      # 显式关闭，不属于"估不出来"
    if not navs or len(navs) < window + 1:
        return UNKNOWN_VOL_EXPOSURE, None
    tail = navs[-(window + 1):]
    rets = [tail[i] / tail[i - 1] - 1
            for i in range(1, len(tail)) if tail[i - 1] > 0]
    if len(rets) < window:
        return UNKNOWN_VOL_EXPOSURE, None
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    vol = (var ** 0.5) * (TRADING_DAYS ** 0.5)
    if vol <= 1e-9:
        # 波动率为 0 在真实组合里不可能，几乎必然是净值序列有问题
        # (取价失败被成本价填充、或序列里全是重复值)。按未知处理。
        return UNKNOWN_VOL_EXPOSURE, None
    return float(min(1.0, max(min_exposure, target_vol / vol))), float(vol)
