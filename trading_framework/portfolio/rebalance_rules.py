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
    # 排序依据全缺失时必须出声 —— 否则所有候选并列 -inf，排序静默退化成
    # 按股票代码字母序，"卖掉最差的 N 只"变成"卖掉代码最小的 N 只"，
    # 而调用方看到的仍是一个长度正确的卖出清单。
    # 2026-09-05 实盘就是这样跑的: get_signal 只返回 TopK 的分数，而卖出
    # 候选按定义都在 TopK 之外，无一命中。
    n_scored = sum(1 for c in all_out if c in scores)
    if n_scored == 0:
        print(f"[rebalance_rules] 警告: {len(all_out)} 个卖出候选无一有分数，"
              f"n_drop 排序已退化为按代码字母序 —— 请检查 scores 是否只含 TopK")
    elif n_scored < len(all_out):
        print(f"[rebalance_rules] 提示: {len(all_out) - n_scored}/{len(all_out)} "
              f"个卖出候选无分数，将被排在最前(视为最差)")
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


def allocate_buys(targets: list, prices: dict, available_cash: float,
                  open_cost: float = 0.0, min_lot: int = 100,
                  expensive_ratio: float = 1.5) -> dict:
    """把可用资金分配到买入清单 — 回测/实盘共用的唯一实现

    2026-09-05 抽出。此前实盘 (live_portfolio.calculate_affordable_allocation)
    与模拟盘 (paper_trader 内联) 各写一份，四条语义都不同:

        项目          实盘                          模拟盘
        太贵的票       剔除后把预算重分给买得起的       无此逻辑
        不足一手       强制买 100 股(允许超配)         直接跳过
        交易成本       不计入预算                     计入
        跳过的钱       重新分配                       闲置

    后果是同一份买入清单在两边买出不同的股数与不同的持仓只数 —— 模拟盘会
    系统性少投一部分钱。而 reconcile.py 只比对买卖**清单**，仓位大小不在
    覆盖范围内，所以这处分叉一直没被发现 (vol_target 失效那处也藏在这里)。

    统一后的语义 = 实盘那套(它是真在交易的) + 模拟盘的成本核算:

      1. 先按等分预算筛掉"贵到 1.5 倍预算都买不了一手"的票
      2. 把全部资金重分给剩下的，避免预算闲置
      3. 不足一手时仍买一手 —— 小资金下宁可让个别标的超配，也不要凭空
         少持一只。上限由 expensive_ratio 控制(最多约 1.5 倍超配)
      4. 现金不够时按剩余资金缩减；缩到不足一手才跳过

    Args:
        targets: 买入清单，按优先级(信号分数降序)排列
        prices: {instrument: 价格}
        available_cash: 可用资金(已扣除波动率目标的缩减)
        open_cost: 买入费率，如 0.0005。计入预算，避免下单后现金为负
        min_lot: 最小交易单位，A 股 100
        expensive_ratio: 允许的最大超配倍数

    Returns:
        {instrument: {'shares': int, 'price': float, 'amount': float}}
        amount 含交易成本。买不起的股票不在结果里。
    """
    if not targets or available_cash <= 0:
        return {}

    unit = 1.0 + float(open_cost)

    # 第一轮: 筛掉买不起的
    budget_per = available_cash / len(targets)
    affordable = []
    for inst in targets:
        price = prices.get(inst, 0)
        if not price or price <= 0:
            continue                       # 无价格，跳过
        if price * min_lot * unit > budget_per * expensive_ratio:
            continue                       # 1.5 倍预算都买不了一手
        affordable.append(inst)
    if not affordable:
        return {}

    # 第二轮: 资金重分给买得起的
    budget_per = available_cash / len(affordable)
    allocation = {}
    used = 0.0
    for inst in affordable:
        price = prices[inst]
        shares = int(budget_per / (price * unit) / min_lot) * min_lot
        if shares < min_lot:
            shares = min_lot               # 不足一手仍买一手(见 docstring 3)
        amount = shares * price * unit
        if used + amount > available_cash:
            shares = int((available_cash - used) / (price * unit)
                         / min_lot) * min_lot
            if shares < min_lot:
                continue                   # 剩余资金不足一手，放弃
            amount = shares * price * unit
        allocation[inst] = {'shares': int(shares),
                            'price': float(price),
                            'amount': round(float(amount), 2)}
        used += amount
    return allocation
