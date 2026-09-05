"""过拟合统计检验 — 紧缩夏普比率 (DSR) 与回测过拟合概率 (PBO)

2026-09-03 新增。解决同一个问题: **你试了很多次才挑出最好的那个，而"最好的那个"
本身就会因为运气看起来很好。**

现有的验收标准只有"holdout 上 beat_baseline"，没有任何对试验次数的校正。
而一轮 --evolved 挖掘是 8 方向 × 6 轮 × 2 候选，几百次试验起步 ——
从一堆真实夏普为 0 的随机策略里挑最大值，那个最大值必然显著大于 0。
run_006/run_010 的 22 个因子(样本内 +7.4% / 样本外 -12.9%)正是这个机制。

参考:
  Bailey & López de Prado (2014), "The Deflated Sharpe Ratio"
  Bailey, Borwein, López de Prado & Zhu (2014), "The Probability of
  Backtest Overfitting" (CSCV 方法)
"""
from __future__ import annotations

import itertools
from math import sqrt, e, log
from statistics import NormalDist

_Z = NormalDist().inv_cdf
_PHI = NormalDist().cdf
EULER_GAMMA = 0.5772156649015329

TRADING_DAYS = 242          # 与 portfolio/rebalance_rules 保持一致 (A 股)

# 验收门槛
DSR_MIN = 0.95              # DSR 低于此视为未通过多重检验校正
PBO_MAX = 0.20              # PBO 高于此说明挑选流程本身不可靠


# ============ 基础统计量 ============

def sharpe(returns: list, annualize: bool = True) -> float:
    """夏普比率 (无风险利率按 0 处理)"""
    n = len(returns)
    if n < 2:
        return 0.0
    mean = sum(returns) / n
    var = sum((r - mean) ** 2 for r in returns) / (n - 1)
    sd = sqrt(var)
    if sd <= 1e-12:
        return 0.0
    sr = mean / sd
    return sr * sqrt(TRADING_DAYS) if annualize else sr


def _moments(returns: list) -> tuple:
    """返回 (偏度, 峰度)。峰度为非超额峰度，正态分布 = 3。"""
    n = len(returns)
    if n < 4:
        return 0.0, 3.0
    mean = sum(returns) / n
    m2 = sum((r - mean) ** 2 for r in returns) / n
    if m2 <= 1e-24:
        return 0.0, 3.0
    m3 = sum((r - mean) ** 3 for r in returns) / n
    m4 = sum((r - mean) ** 4 for r in returns) / n
    return m3 / m2 ** 1.5, m4 / m2 ** 2


# ============ DSR ============

def expected_max_sharpe(n_trials: int, sr_std: float) -> float:
    """纯运气可达的期望最高夏普 (与 sr_std 同频率)

    E[max SR] ≈ σ_SR · [(1-γ)·Z⁻¹(1 - 1/N) + γ·Z⁻¹(1 - 1/(N·e))]

    Args:
        n_trials: 试验次数 N
        sr_std:   各次试验夏普的标准差 (试验越分散 -> 策略越不稳 -> 惩罚越重)
    """
    if n_trials <= 1 or sr_std <= 0:
        return 0.0          # 只试过一次，无选择偏差
    return sr_std * ((1 - EULER_GAMMA) * _Z(1 - 1.0 / n_trials)
                     + EULER_GAMMA * _Z(1 - 1.0 / (n_trials * e)))


def deflated_sharpe(returns: list, n_trials: int, sr_std_ann: float,
                    skew: float | None = None,
                    kurt: float | None = None) -> dict:
    """紧缩夏普比率

    DSR = Φ[ (SR - SR₀)·√(T-1) / √(1 - γ₃·SR + (γ₄-1)/4·SR²) ]

    γ₃(偏度)/γ₄(峰度) 一项修正的是: 传统夏普假设收益正态，而动量类策略左偏肥尾，
    夏普会系统性高估其真实边际。

    Args:
        returns:     策略的周期收益序列 (日频)
        n_trials:    产生该结果前一共试了多少次
        sr_std_ann:  各次试验【年化】夏普的标准差
        skew/kurt:   不给则从 returns 估计

    Returns:
        {dsr, sharpe_ann, sr0_ann, T, n_trials, skew, kurt, passed}
    """
    T = len(returns)
    if T < 8:
        return {"dsr": 0.0, "sharpe_ann": 0.0, "sr0_ann": 0.0, "T": T,
                "n_trials": n_trials, "skew": 0.0, "kurt": 3.0,
                "passed": False, "note": "样本过短，无法检验"}

    sr_p = sharpe(returns, annualize=False)          # 每期夏普
    if skew is None or kurt is None:
        s_est, k_est = _moments(returns)
        skew = s_est if skew is None else skew
        kurt = k_est if kurt is None else kurt

    sr0_p = expected_max_sharpe(n_trials, sr_std_ann / sqrt(TRADING_DAYS))
    denom = 1 - skew * sr_p + (kurt - 1) / 4 * sr_p ** 2
    denom = sqrt(denom) if denom > 1e-12 else 1e-6
    z = (sr_p - sr0_p) * sqrt(T - 1) / denom
    dsr = _PHI(z)
    return {
        "dsr": float(dsr),
        "sharpe_ann": float(sr_p * sqrt(TRADING_DAYS)),
        "sr0_ann": float(sr0_p * sqrt(TRADING_DAYS)),
        "T": T, "n_trials": int(n_trials),
        "skew": float(skew), "kurt": float(kurt),
        "passed": bool(dsr >= DSR_MIN),
    }


# ============ PBO (CSCV) ============

def probability_of_backtest_overfitting(returns_matrix: list,
                                        n_splits: int = 16,
                                        max_combos: int = 2000) -> dict:
    """回测过拟合概率 (组合对称交叉验证)

    评价的不是某个策略，而是【挑选流程本身】:
      1. 把时间切成 n_splits 段
      2. 任取一半做样本内、其余做样本外 (共 C(S, S/2) 种组合)
      3. 每种组合里取样本内最优的策略，看它在样本外排第几
      4. PBO = 样本内最优在样本外掉到中位数以下的比例

    PBO > 0.5 意味着这套挑法比抛硬币还差。

    Args:
        returns_matrix: [[策略1的收益...], [策略2的收益...], ...]，各策略等长
        n_splits: 时间分段数 (必须为偶数)
        max_combos: 组合数上限，超出则均匀抽样 (C(16,8)=12870，全跑较慢)

    Returns:
        {pbo, n_strategies, n_combos, median_oos_rank, ...}
    """
    n_str = len(returns_matrix)
    if n_str < 2:
        return {"pbo": 0.0, "n_strategies": n_str, "n_combos": 0,
                "note": "策略数不足 2，无法评估挑选流程"}
    T = min(len(r) for r in returns_matrix)
    if n_splits % 2:
        n_splits += 1
    if T < n_splits * 2:
        return {"pbo": 0.0, "n_strategies": n_str, "n_combos": 0,
                "note": f"样本过短 (T={T} < {n_splits*2})，无法切分"}

    # 等长切分
    size = T // n_splits
    blocks = [list(range(i * size, (i + 1) * size)) for i in range(n_splits)]

    combos = list(itertools.combinations(range(n_splits), n_splits // 2))
    if len(combos) > max_combos:
        step = len(combos) / max_combos
        combos = [combos[int(i * step)] for i in range(max_combos)]

    n_below = 0
    ranks = []
    for is_idx in combos:
        is_set = set(is_idx)
        is_rows = [i for b in is_idx for i in blocks[b]]
        oos_rows = [i for b in range(n_splits) if b not in is_set
                    for i in blocks[b]]

        is_sr = [sharpe([r[i] for i in is_rows]) for r in returns_matrix]
        oos_sr = [sharpe([r[i] for i in oos_rows]) for r in returns_matrix]

        best = max(range(n_str), key=lambda k: is_sr[k])
        # 该策略在样本外的相对排名 (0=最差, 1=最好)
        worse = sum(1 for k in range(n_str) if oos_sr[k] < oos_sr[best])
        rel = worse / (n_str - 1) if n_str > 1 else 1.0
        ranks.append(rel)
        if rel < 0.5:
            n_below += 1

    ranks.sort()
    mid = ranks[len(ranks) // 2] if ranks else 0.0
    pbo = n_below / len(combos) if combos else 0.0
    return {
        "pbo": float(pbo),
        "n_strategies": n_str,
        "n_combos": len(combos),
        "n_splits": n_splits,
        "median_oos_rank": float(mid),
        "passed": bool(pbo <= PBO_MAX),
    }


# ============ 汇总 ============

def evaluate(returns: list, n_trials: int, sr_std_ann: float,
             returns_matrix: list | None = None) -> dict:
    """对一次验收同时给出 DSR 与 PBO 结论

    ⚠ 当前无调用方 (2026-09-05 核实)。factor_miner._run_overfit_check 绕过本
    函数、直接调 deflated_sharpe，所以**DSR 生效但 PBO 从未运行**，过拟合
    检验只有一半在跑。

    接上 PBO 需要 returns_matrix —— 即同一轮里多个候选因子池各自的日收益
    序列。Phase D 的漏斗本来就会评估多个池 (FUNNEL_STRATEGIES 有 6 种)，
    把它们的收益序列收集起来即可，不需要额外回测。

    PBO 回答的问题与 DSR 不同: DSR 问"这个夏普是不是运气"，PBO 问"我这套
    挑选流程本身可不可靠" —— 后者正是 run_006/run_010 那 22 个因子
    (样本内 +7.4% / 样本外 -12.9%) 暴露的问题。
    """
    out = {"dsr": deflated_sharpe(returns, n_trials, sr_std_ann)}
    if returns_matrix:
        out["pbo"] = probability_of_backtest_overfitting(returns_matrix)
    checks = [out["dsr"].get("passed")]
    if "pbo" in out:
        checks.append(out["pbo"].get("passed"))
    out["passed"] = all(c for c in checks if c is not None)
    return out


def format_report(result: dict) -> str:
    """渲染成可读文本 (用于飞书通知/日志)"""
    d = result.get("dsr", {})
    lines = ["【过拟合检验】"]
    if d:
        mark = "✅" if d.get("passed") else "❌"
        lines.append(f"  {mark} DSR={d.get('dsr', 0):.3f} (门槛 {DSR_MIN})")
        lines.append(f"     实测夏普 {d.get('sharpe_ann', 0):.3f} vs "
                     f"纯运气可达 {d.get('sr0_ann', 0):.3f} "
                     f"(试验 {d.get('n_trials', 0)} 次, T={d.get('T', 0)})")
        if d.get("note"):
            lines.append(f"     {d['note']}")
    p = result.get("pbo")
    if p:
        mark = "✅" if p.get("passed") else "❌"
        lines.append(f"  {mark} PBO={p.get('pbo', 0):.3f} (门槛 {PBO_MAX})")
        lines.append(f"     {p.get('n_strategies', 0)} 个候选 × "
                     f"{p.get('n_combos', 0)} 种切分, "
                     f"样本外中位排名 {p.get('median_oos_rank', 0):.2f}")
        if p.get("note"):
            lines.append(f"     {p['note']}")
    return "\n".join(lines)
