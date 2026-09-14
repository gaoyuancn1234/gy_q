"""调仓风控 — 回测 / 实盘 / 模拟盘共用, 不要再各写一份。"""

# A 股年均交易日数。沿用 live_portfolio 原值 242 (美股惯例是 252)，
# 改动会静默改变已实现波动率的年化结果与实盘敞口。
TRADING_DAYS = 242

# 无法估计已实现波动率时使用的敞口 (净值历史不足 / 序列异常 / 取价失败)。
# 取 0.6 而非 1.0: 风控在信息缺失时必须收缩而非放开。见 compute_exposure 注释。
UNKNOWN_VOL_EXPOSURE = 0.6


def select_sells(current: set, target: set, scores: dict,
                 n_drop: int | None, entry_dates: dict | None = None) -> set:
    """按换手限制挑选卖出标的。

    n_drop 为 None 时全量换手。否则只卖出已不在目标里、分数最低的
    n_drop 只。分数缺失视为最差; 全缺失会退化成按代码字母序, 必须告警。
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
    # 2026-09-14: 无分数候选并列 -inf 后, 原先的次级键是**股票代码**,
    # 于是排序退化成字母序, 而 "SH..." 恒小于 "SZ..." —— 深市高代码的
    # 出局票会系统性地排在队尾, 永远轮不到卖出, 在持仓里越积越多。
    # 全量回放实测(2026-09-04 最后一个调仓日): 持仓 98 只里 31 只无分数,
    # n_drop=20 全被它们占满, 模型对"卖哪些"贡献为 0; 留下的 11 只
    # (SZ000626 ... SZ301223) 全是深市。82 个调仓日里 78% 是这种情况。
    # 改用 entry_date 做次级键 = 持有最久的先卖(FIFO), 换手量不变,
    # 只改"卖哪些", 且消除了交易所偏向。代码仍作最末位键保证可复现。
    #
    # 注意这只治了"选谁"。更根本的问题是 n_drop 本意是限制**模型驱动的
    # 换手**以控成本, 实际却被"已不在截面里的票"占满 —— 那需要把强制
    # 退出与调仓卖出分开计budget, 属于策略变更, 要重跑回测才能动。
    ed = entry_dates or {}
    _n_no = len(all_out) - n_scored
    if n_drop is not None and _n_no >= int(n_drop) and _n_no > 0:
        print(f"[rebalance_rules] 注意: 无分数候选 {_n_no} 只 >= n_drop "
              f"{n_drop}, 本次卖出名单**完全由无分数候选决定**, "
              f"模型排名不起作用")
    # 显式排序: 直接迭代集合会因字符串哈希随机化导致同配置两次结果不同
    # (CLAUDE.md 记录过 Sharpe 0.318 vs 0.247 的复现失败)
    ranked = sorted(
        all_out,
        key=lambda c: (scores.get(c, float('-inf')), str(ed.get(c, '')), c),
    )
    return set(ranked[:int(n_drop)])


def compute_exposure(navs: list, target_vol: float | None,
                     window: int = 20, min_exposure: float = 0.2,
                     unknown_exposure: float | None = None) -> tuple:
    """按已实现波动率反向缩放权益敞口, 不加杠杆。

    exposure = clip(target_vol / realized_vol, min_exposure, 1.0)
    估不出波动率时收缩到 unknown_exposure, 禁止满仓。
    """
    unk = (UNKNOWN_VOL_EXPOSURE if unknown_exposure is None
           else float(unknown_exposure))
    if not target_vol:
        return 1.0, None                      # 显式关闭，不属于"估不出来"
    if not navs or len(navs) < window + 1:
        return unk, None
    tail = navs[-(window + 1):]
    rets = [tail[i] / tail[i - 1] - 1
            for i in range(1, len(tail)) if tail[i - 1] > 0]
    if len(rets) < window:
        return unk, None
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    vol = (var ** 0.5) * (TRADING_DAYS ** 0.5)
    if vol <= 1e-9:
        # 波动率为 0 在真实组合里不可能，几乎必然是净值序列有问题
        # (取价失败被成本价填充、或序列里全是重复值)。按未知处理。
        return unk, None
    return float(min(1.0, max(min_exposure, target_vol / vol))), float(vol)


def price_limit(code: str) -> float:
    """按板块返回涨跌停幅度。留 0.5 个百分点余量。

    中证2000 约一半不是主板, 不能再用 ±9.5% 一刀切。
    """
    n = code[2:] if code[:2].isalpha() else code
    if n.startswith(('688', '300', '301')):
        return 0.195          # 创业板 / 科创板 20%
    if n.startswith(('8', '92', '43')):
        return 0.295          # 北交所 30%
    return 0.095              # 主板 10%


def pick_liquid(ranked: list, amounts: dict, k: int,
                min_amount: float | None) -> list:
    """沿分数从高到低走到凑满 k 只流动性合格的票

    机构小市值产品常用日成交额 2000 万以上做硬门槛, 10 万资金冲击可以
    忽略, 降到一两百万仍能尾盘成交 —— 这是散户相对机构的容量优势.
    不够门槛的票跳过、从更后面补, 不要让持仓缺一个坑.
    """
    if not min_amount:
        return list(ranked)[:k]
    if not amounts:
        return list(ranked)[:k]
    out = []
    for code in ranked:
        amt = amounts.get(code)
        if amt is None or amt < min_amount:
            continue
        out.append(code)
        if len(out) >= k:
            break
    return out


def prev_official_closes(prev_day: str) -> dict:
    """上一交易日官方收盘价(未复权), 涨停判定的基准。

    与 auction_infer._ratio_row 读同一个文件, 同为未复权 —— 基准和当日价
    必须同口径, 否则算出来的涨幅是假的。
    """
    from data_hub.paths import daily_raw_dir
    import pandas as pd

    f = daily_raw_dir() / f"{prev_day.replace('-', '')}.parquet"
    if not f.exists():
        return {}
    df = pd.read_parquet(f)
    out = {}
    for ts, c in zip(df["ts_code"], df["close"]):
        num, ex = str(ts).split(".")
        out[("BJ" if ex == "BJ" else ex) + num] = float(c)
    return out


def limit_up_blocked(prices: dict, prev_closes: dict) -> set:
    """当日涨幅已达板块涨跌停线、买不进的票。

    为什么必须有这个 (2026-09-14):
      训练标签 OVN_LABEL 把当日涨幅 > 9.5% 的样本标成 NaN 丢掉, 也就是
      模型**从没学过涨停日之后会发生什么**; 但出分时照样给涨停日打分,
      等于拿训练集里被刻意删掉的那类输入去做推断。当天实测 1958 只里
      47 只涨幅 > 9.5%, TopK-100 里占 4 只, Top-10 里占 2 只, 而第一名
      (中新赛克, 三连板、后两天一字板)分数 1.61, 甩开第二名 3.5 倍。
      一字板买不进: 卖盘为零, 且 gm_bridge 挂单价是 现价 x 1.01, 对已经
      封在涨停价的票来说这个价格超过涨停限价, 会被直接拒单。

    阈值按板块走 price_limit(), 不用标签里那个一刀切的 0.095 ——
    创业板涨停是 20%, 一只 300xxx 涨 12% 根本没涨停。

    只挡买入。涨停时是可以卖的(买盘充足), 所以不碰卖出路径。
    """
    out = set()
    for code, px in (prices or {}).items():
        pc = prev_closes.get(code)
        if not pc or pc <= 0 or not px or px <= 0:
            continue
        if px / pc - 1 >= price_limit(code):
            out.add(code)
    return out


def limit_down_blocked(prices: dict, prev_closes: dict) -> set:
    """当日跌幅已达板块跌停线、卖不掉的票。limit_up_blocked 的镜像。

    涨停买不进是"换一只买", 跌停卖不掉是"这只继续持有" —— 后者更要紧:
    若把卖不掉的票当成已卖出, 会以为腾出了坑位, 拿并不存在的现金去买,
    实盘表现为买单资金不足而部分落空, 且与回测对不上。
    所以调用方必须在算 holds/free **之前**把它们从 sells 里剔除。
    """
    out = set()
    for code, px in (prices or {}).items():
        pc = prev_closes.get(code)
        if not pc or pc <= 0 or not px or px <= 0:
            continue
        if px / pc - 1 <= -price_limit(code):
            out.add(code)
    return out


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
