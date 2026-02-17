#!/usr/bin/env python3
"""实验盘管理器 — 基于同一基础信号的后处理实验

与 ShadowManager 的区别:
  - Shadow: 不同 predictions.pkl (不同因子/模型), 需独立重训
  - 实验盘: 同一基础信号 + 后处理 (过滤/调参), 不需要重训

生命周期: active → completed/rejected
  - create → daily update → expire → (人工) reject / extend
  - 无 promote/rollback (实验盘不会替代基线)

用法:
    from experiment_manager import ExperimentManager
    em = ExperimentManager()
    eid = em.create_experiment(name="情绪哨兵", experiment_type="sentiment",
                               config={...}, reason="宏观+个股过滤", days=30)
    em.update_daily(eid, live_signal, exp_signal, prices, extra={...})
"""
import json
import math
from datetime import date, datetime
from pathlib import Path

from factor_lab.utils import json_default as _json_default

PROJECT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = PROJECT_DIR / "experiment"
REGISTRY_FILE = EXPERIMENT_DIR / "registry.json"


def _ensure_dirs():
    for d in [EXPERIMENT_DIR, EXPERIMENT_DIR / "news_cache",
              EXPERIMENT_DIR / "daily_log"]:
        d.mkdir(parents=True, exist_ok=True)


def _load_registry() -> dict:
    if REGISTRY_FILE.exists():
        return json.loads(REGISTRY_FILE.read_text(encoding='utf-8'))
    return {}


def _save_registry(reg: dict):
    _ensure_dirs()
    REGISTRY_FILE.write_text(
        json.dumps(reg, ensure_ascii=False, indent=2), encoding='utf-8')


def _next_exp_id(reg: dict) -> str:
    nums = []
    for k in reg:
        if k.startswith("exp_"):
            try:
                nums.append(int(k.split("_")[1]))
            except (ValueError, IndexError):
                continue
    if nums:
        return f"exp_{max(nums) + 1:03d}"
    return "exp_001"


class ExperimentManager:
    """实验盘管理器"""

    def __init__(self):
        _ensure_dirs()
        self.registry = _load_registry()

    def create_experiment(self, name: str, experiment_type: str,
                          config: dict, reason: str,
                          duration_days: int = 30) -> str:
        """创建实验盘

        Args:
            name: 显示名 (如 "情绪哨兵")
            experiment_type: 类型标识 (如 "sentiment")
            config: 实验配置参数
            reason: 创建原因
            duration_days: 验证天数
        Returns:
            exp_id
        """
        exp_id = _next_exp_id(self.registry)
        self.registry[exp_id] = {
            "status": "active",
            "name": name,
            "experiment_type": experiment_type,
            "created_date": date.today().isoformat(),
            "duration_days": duration_days,
            "elapsed_days": 0,
            "config": config,
            "reason": reason,
        }
        _save_registry(self.registry)
        print(f"  [experiment] 创建 {exp_id} [{name}]: {reason}")
        return exp_id

    def get_active_experiments(self) -> dict:
        """返回所有 active 实验"""
        return {k: v for k, v in self.registry.items()
                if v.get('status') == 'active'}

    def update_daily(self, exp_id: str, live_signal: dict,
                     exp_signal: dict, prices: dict,
                     extra: dict = None):
        """记录每日对比数据

        Args:
            exp_id: 实验 ID
            live_signal: 基线信号 (SignalGenerator.get_signal())
            exp_signal: 实验信号 (经过后处理)
            prices: 今日价格
            extra: 额外数据 (macro_sentiment, filtered_stocks 等)
        """
        exp = self.registry.get(exp_id)
        if not exp or exp['status'] != 'active':
            return

        today = date.today().isoformat()

        # 信号重叠率
        live_stocks = set(live_signal.get('target_stocks', []))
        exp_stocks = set(exp_signal.get('target_stocks', []))
        if live_stocks or exp_stocks:
            overlap = len(live_stocks & exp_stocks) / max(
                len(live_stocks | exp_stocks), 1)
        else:
            overlap = 1.0

        # 理论日收益
        prev_prices = _load_prev_prices()
        live_return = _calc_daily_returns(
            live_signal.get('target_stocks', []), prices, prev_prices)
        exp_return = _calc_daily_returns(
            exp_signal.get('target_stocks', []), prices, prev_prices)
        _save_prev_prices(prices)

        daily_entry = {
            "exp_id": exp_id,
            "name": exp.get("name", ""),
            "target_stocks": exp_signal.get('target_stocks', []),
            "effective_topk": exp_signal.get('effective_topk', 0),
            "overlap_with_live": round(overlap, 4),
            "exp_return": round(exp_return, 6),
            "live_return": round(live_return, 6),
            "macro_sentiment": (extra or {}).get('macro_sentiment', ''),
            "filtered_stocks": (extra or {}).get('filtered_stocks', []),
            "replacement_stocks": (extra or {}).get('replacement_stocks', []),
            "topk_adjustment": (extra or {}).get('topk_adjustment', 0),
        }

        # 写入每日日志
        log_dir = EXPERIMENT_DIR / "daily_log"
        log_file = log_dir / f"{today}.json"
        if log_file.exists():
            daily_log = json.loads(log_file.read_text(encoding='utf-8'))
        else:
            daily_log = {
                "date": today,
                "live": {
                    "target_stocks": live_signal.get('target_stocks', []),
                    "regime": live_signal.get('regime', ''),
                    "topk": live_signal.get('effective_topk', 0),
                },
                "experiments": {},
            }
        daily_log["experiments"][exp_id] = daily_entry
        log_file.write_text(
            json.dumps(daily_log, ensure_ascii=False, indent=2),
            encoding='utf-8')

        # 更新 elapsed
        exp['elapsed_days'] = exp.get('elapsed_days', 0) + 1
        _save_registry(self.registry)

        # 更新汇总
        self._update_summary(exp_id)

    def get_performance(self, exp_id: str) -> dict:
        """获取累积绩效"""
        summary_file = EXPERIMENT_DIR / "daily_log" / "summary.json"
        if not summary_file.exists():
            return {}
        summaries = json.loads(summary_file.read_text(encoding='utf-8'))
        return summaries.get(exp_id, {})

    def get_status_text(self) -> str:
        """生成状态文本 (飞书展示)"""
        active = self.get_active_experiments()
        completed = {k: v for k, v in self.registry.items()
                     if v.get('status') in ('completed', 'rejected')}

        if not active and not completed:
            return "当前无实验盘"

        lines = []

        if active:
            lines.append(f"实验盘 ({len(active)}个活跃):")
            for eid, exp in active.items():
                elapsed = exp.get('elapsed_days', 0)
                duration = exp.get('duration_days', 30)
                name = exp.get('name', '未命名')
                perf = self.get_performance(eid)
                cum_ret = perf.get('exp_cumulative_return', 0)
                live_ret = perf.get('live_cumulative_return', 0)

                lines.append(
                    f"  {eid} [{name}]: {elapsed}/{duration}天")
                if perf.get('n_days', 0) > 0:
                    lines.append(
                        f"    累计: 实验{cum_ret:+.2%} vs 基线{live_ret:+.2%}")

                    # 今日摘要
                    latest = self._get_latest_daily(eid)
                    if latest:
                        sent = latest.get('macro_sentiment', '')
                        filtered = latest.get('filtered_stocks', [])
                        sent_str = f"宏观:{sent}" if sent else ""
                        filt_str = f"过滤:{len(filtered)}只" if filtered else ""
                        extra = " | ".join(
                            s for s in [sent_str, filt_str] if s)
                        if extra:
                            lines.append(f"    今日: {extra}")

        if completed:
            lines.append(f"\n已结束 ({len(completed)}个):")
            for eid, exp in list(completed.items())[-3:]:
                name = exp.get('name', '未命名')
                status = exp.get('status', '')
                lines.append(f"  {eid} [{name}]: {status}")

        return "\n".join(lines)

    def get_daily_review(self, exp_id: str) -> str:
        """生成每日复盘文本 (附在推送消息中)"""
        exp = self.registry.get(exp_id)
        if not exp:
            return ""

        latest = self._get_latest_daily(exp_id)
        if not latest:
            return ""

        perf = self.get_performance(exp_id)
        name = exp.get('name', '未命名')
        elapsed = exp.get('elapsed_days', 0)

        live_topk = latest.get('_live_topk', perf.get('_live_topk', '?'))
        exp_topk = latest.get('effective_topk', '?')
        exp_ret = latest.get('exp_return', 0)
        live_ret = latest.get('live_return', 0)
        cum_exp = perf.get('exp_cumulative_return', 0)
        cum_live = perf.get('live_cumulative_return', 0)

        filtered = latest.get('filtered_stocks', [])
        replacements = latest.get('replacement_stocks', [])
        sentiment = latest.get('macro_sentiment', 'N/A')

        # 模拟净值 (假设初始100万)
        exp_nav = 1000000 * (1 + cum_exp)
        live_nav = 1000000 * (1 + cum_live)

        filter_note = ""
        if filtered:
            try:
                from portfolio.live_portfolio import get_stock_names
                names = get_stock_names()
                filter_note = "过滤: " + ", ".join(
                    f"{names.get(c, c)}" for c in filtered)
            except Exception:
                filter_note = f"过滤{len(filtered)}只"

        lines = []
        lines.append(f"  [{name}] (模拟)")
        lines.append(
            f"    今日: TopK={exp_topk} 日收益{exp_ret:+.2%}"
            f" vs 基线 TopK={live_topk} {live_ret:+.2%}")
        lines.append(
            f"    累计({elapsed}天): "
            f"模拟{cum_exp:+.2%}(净值{exp_nav:,.0f}) "
            f"vs 基线{cum_live:+.2%}(净值{live_nav:,.0f})")
        if filter_note:
            lines.append(f"    {filter_note}")

        return "\n".join(lines)

    def get_macro_review(self, exp_id: str) -> str:
        """生成宏观分析复盘文本"""
        today = date.today().isoformat()
        analysis_file = EXPERIMENT_DIR / "news_cache" / f"{today}_macro_analysis.json"
        if not analysis_file.exists():
            return ""

        analysis = json.loads(analysis_file.read_text(encoding='utf-8'))
        sentiment = analysis.get('sentiment', 'N/A')
        summary = analysis.get('summary', '')
        impacts = analysis.get('holdings_impact', [])

        sentiment_emoji = {
            'bullish': '偏多', 'neutral': '中性', 'bearish': '偏空'
        }.get(sentiment, sentiment)

        lines = [f"宏观情绪: {sentiment_emoji}"]
        if summary:
            lines.append(f"  {summary}")
        if impacts:
            lines.append("  对持仓影响:")
            for imp in impacts[:5]:
                icon = {'positive': '+', 'negative': '-', 'neutral': '='}.get(
                    imp.get('impact', ''), '?')
                lines.append(
                    f"  {icon} {imp.get('name', '')}: {imp.get('reason', '')}")

        return "\n".join(lines)

    def check_expired(self) -> list:
        """检查到期的实验"""
        expired = []
        for eid, exp in self.registry.items():
            if exp['status'] != 'active':
                continue
            if exp.get('elapsed_days', 0) >= exp.get('duration_days', 30):
                expired.append(eid)
        return expired

    def get_expiry_report(self, exp_id: str) -> str:
        """生成到期报告"""
        exp = self.registry.get(exp_id)
        if not exp:
            return ""

        perf = self.get_performance(exp_id)
        name = exp.get('name', '未命名')
        elapsed = exp.get('elapsed_days', 0)

        exp_ret = perf.get('exp_cumulative_return', 0)
        live_ret = perf.get('live_cumulative_return', 0)
        exp_sharpe = perf.get('exp_sharpe', 0)
        live_sharpe = perf.get('live_sharpe', 0)
        exp_mdd = perf.get('exp_max_drawdown', 0)

        lines = [
            f"实验盘到期报告 - {exp_id} [{name}] ({elapsed}天)",
            f"原因: {exp.get('reason', '')}",
            f"实验收益: {exp_ret:+.2%} (vs 基线 {live_ret:+.2%})",
            f"实验 Sharpe: {exp_sharpe:.2f} (vs 基线 {live_sharpe:.2f})",
            f"实验 MDD: {exp_mdd:.2%}",
            "",
        ]

        if exp_sharpe > live_sharpe and exp_ret > live_ret:
            lines.append("结论: 实验表现优于基线")
        elif exp_sharpe > live_sharpe:
            lines.append("结论: 实验 Sharpe 更优，但收益稍低")
        else:
            lines.append("结论: 实验未明显优于基线")

        lines.append("")
        lines.append("命令:")
        lines.append(f"  python daily_runner.py --extend-experiment {exp_id} 15")
        lines.append(f"  python daily_runner.py --reject-experiment {exp_id}")

        # 标记为已到期 (仍在 active 中，等人工决定)
        exp['status'] = 'completed'
        _save_registry(self.registry)

        return "\n".join(lines)

    def reject(self, exp_id: str):
        """终止实验"""
        exp = self.registry.get(exp_id)
        if not exp:
            raise ValueError(f"Experiment {exp_id} 不存在")
        exp['status'] = 'rejected'
        exp['rejected_date'] = date.today().isoformat()
        _save_registry(self.registry)
        print(f"  [experiment] {exp_id} 已终止")

    def extend(self, exp_id: str, extra_days: int):
        """延长实验"""
        exp = self.registry.get(exp_id)
        if not exp:
            raise ValueError(f"Experiment {exp_id} 不存在")
        exp['duration_days'] = exp.get('duration_days', 30) + extra_days
        if exp['status'] == 'completed':
            exp['status'] = 'active'
        _save_registry(self.registry)
        print(f"  [experiment] {exp_id} 延长 {extra_days} 天 "
              f"(总 {exp['duration_days']} 天)")

    def _get_latest_daily(self, exp_id: str) -> dict:
        """获取最新一天的每日记录"""
        log_dir = EXPERIMENT_DIR / "daily_log"
        log_files = sorted(log_dir.glob("20*.json"))
        for lf in reversed(log_files):
            try:
                daily = json.loads(lf.read_text(encoding='utf-8'))
                entry = daily.get('experiments', {}).get(exp_id)
                if entry:
                    entry['_live_topk'] = daily.get('live', {}).get('topk', '?')
                    return entry
            except Exception:
                continue
        return {}

    def _update_summary(self, exp_id: str):
        """更新累积汇总"""
        summary_file = EXPERIMENT_DIR / "daily_log" / "summary.json"
        if summary_file.exists():
            summaries = json.loads(summary_file.read_text(encoding='utf-8'))
        else:
            summaries = {}

        exp_returns = []
        live_returns = []
        overlaps = []

        log_dir = EXPERIMENT_DIR / "daily_log"
        for log_file in sorted(log_dir.glob("20*.json")):
            try:
                daily = json.loads(log_file.read_text(encoding='utf-8'))
                entry = daily.get('experiments', {}).get(exp_id)
                if entry:
                    exp_returns.append(entry.get('exp_return', 0))
                    live_returns.append(entry.get('live_return', 0))
                    overlaps.append(entry.get('overlap_with_live', 0))
            except Exception:
                continue

        if not exp_returns:
            return

        # 实验累积收益
        exp_cum, exp_peak, exp_mdd = 1.0, 1.0, 0.0
        for r in exp_returns:
            exp_cum *= (1 + r)
            exp_peak = max(exp_peak, exp_cum)
            dd = (exp_cum - exp_peak) / exp_peak if exp_peak > 0 else 0
            exp_mdd = min(exp_mdd, dd)

        # 基线累积收益
        live_cum, live_peak, live_mdd = 1.0, 1.0, 0.0
        for r in live_returns:
            live_cum *= (1 + r)
            live_peak = max(live_peak, live_cum)
            dd = (live_cum - live_peak) / live_peak if live_peak > 0 else 0
            live_mdd = min(live_mdd, dd)

        n = len(exp_returns)

        def _sharpe(returns, n):
            if n < 2:
                return 0.0
            mean_r = sum(returns) / n
            var_r = sum((r - mean_r) ** 2 for r in returns) / (n - 1)
            std_r = math.sqrt(var_r) if var_r > 0 else 1e-9
            return (mean_r / std_r) * math.sqrt(252)

        summaries[exp_id] = {
            "exp_cumulative_return": round(exp_cum - 1.0, 6),
            "live_cumulative_return": round(live_cum - 1.0, 6),
            "exp_sharpe": round(_sharpe(exp_returns, n), 3),
            "live_sharpe": round(_sharpe(live_returns, n), 3),
            "exp_max_drawdown": round(exp_mdd, 6),
            "live_max_drawdown": round(live_mdd, 6),
            "avg_overlap": round(
                sum(overlaps) / len(overlaps) if overlaps else 0, 4),
            "n_days": n,
        }
        summary_file.write_text(
            json.dumps(summaries, ensure_ascii=False, indent=2),
            encoding='utf-8')


# ============ 价格工具 ============

PREV_PRICES_FILE = EXPERIMENT_DIR / "prev_prices.json"


def _load_prev_prices() -> dict:
    if PREV_PRICES_FILE.exists():
        return json.loads(PREV_PRICES_FILE.read_text(encoding='utf-8'))
    return {}


def _save_prev_prices(prices: dict):
    _ensure_dirs()
    PREV_PRICES_FILE.write_text(
        json.dumps(prices, ensure_ascii=False, default=_json_default),
        encoding='utf-8')


def _calc_daily_returns(target_stocks: list, prices_today: dict,
                        prices_yesterday: dict) -> float:
    """TopK 等权理论日收益"""
    if not target_stocks:
        return 0.0
    returns = []
    for c in target_stocks:
        p1 = prices_today.get(c)
        p0 = prices_yesterday.get(c)
        if p1 and p0 and p0 > 0:
            returns.append(p1 / p0 - 1)
    return sum(returns) / len(returns) if returns else 0.0
