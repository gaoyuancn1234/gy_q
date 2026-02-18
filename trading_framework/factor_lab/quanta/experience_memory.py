"""Experience Memory — FactorMiner Section 3.3

跨 session 结构化经验记忆:
- P_succ: 成功模式模板 (高成功率的算子签名/信号源)
- P_fail: 禁区 (反复被相关性拒绝的方向)
- Strategic Insights: 策略级洞察 (算子级别成功/失败比)

持久化为 JSON, 供 idea_agent 在 prompt 中注入经验引导。
"""
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path

from factor_lab.utils import atomic_json_dump

from .ast_dedup import parse_qlib_expr, ASTNode


# ============ 数据结构 ============

@dataclass
class SuccessPattern:
    """P_succ: 高成功率模式"""
    signature: str          # 算子签名 (如 "Div(Std,Mean)")
    signal_source: str      # 信号源 (价格结构/量能/波动率/...)
    example_exprs: list[str] = field(default_factory=list)
    success_count: int = 0
    total_count: int = 0
    avg_icir: float = 0.0

    @property
    def success_rate(self) -> float:
        return self.success_count / max(self.total_count, 1)


@dataclass
class ForbiddenRegion:
    """P_fail: 高相关性禁区"""
    keywords: list[str] = field(default_factory=list)
    correlated_with: str = ""       # 最相关的已有因子名
    rejection_count: int = 0
    last_corr: float = 0.0


@dataclass
class StrategicInsight:
    """策略级洞察"""
    category: str = ""       # 算子类别 (时序/截面/运算/统计)
    insight: str = ""        # 洞察描述
    confidence: float = 0.0  # 0~1
    evidence_count: int = 0


class ExperienceMemory:
    """跨 session 结构化经验记忆 (FactorMiner Section 3.3)"""

    def __init__(self, save_dir: Path):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._success_patterns: list[SuccessPattern] = []
        self._forbidden_regions: list[ForbiddenRegion] = []
        self._strategic_insights: list[StrategicInsight] = []
        self._op_stats: dict[str, dict] = {}  # {op: {success, fail, total_icir}}
        self._updated_at: float = 0.0

    # ============ 核心接口 ============

    def retrieve(self, library_size: int = 0) -> dict:
        """检索相关记忆 → 用于构造 LLM prompt

        Returns:
            {recommended, forbidden, insights}
        """
        recommended = sorted(
            [p for p in self._success_patterns if p.total_count >= 2],
            key=lambda p: p.success_rate * abs(p.avg_icir),
            reverse=True,
        )[:5]

        forbidden = sorted(
            self._forbidden_regions,
            key=lambda f: f.rejection_count,
            reverse=True,
        )[:5]

        insights = sorted(
            self._strategic_insights,
            key=lambda i: i.confidence,
            reverse=True,
        )[:5]

        return {
            "recommended": [asdict(p) for p in recommended],
            "forbidden": [asdict(f) for f in forbidden],
            "insights": [asdict(i) for i in insights],
            "library_size": library_size,
        }

    def evolve(self, trajectories: list, factor_pool) -> None:
        """从轨迹中提炼经验 (每个 session 结束后调用)

        Args:
            trajectories: 本 session 的所有 Trajectory 对象
            factor_pool: 当前 FactorPool
        """
        if not trajectories:
            return

        # 1. 提炼 P_succ: 按算子签名聚类入池因子
        self._evolve_success_patterns(trajectories)

        # 2. 提炼 P_fail: 聚类被相关性拒绝的假说
        self._evolve_forbidden_regions(trajectories)

        # 3. 提炼 Strategic Insights: 算子级别统计
        self._evolve_strategic_insights(trajectories)

        self._updated_at = time.time()

    def format_for_prompt(self) -> str:
        """格式化为 prompt 文本 (供 idea_agent 注入)"""
        lines = ["## 经验记忆 (跨 session 累积)"]

        # 推荐方向
        recommended = [p for p in self._success_patterns
                       if p.total_count >= 2 and p.success_rate >= 0.3]
        recommended.sort(key=lambda p: p.success_rate * abs(p.avg_icir), reverse=True)
        if recommended:
            lines.append("")
            lines.append("### 推荐方向 (高成功率)")
            for p in recommended[:5]:
                lines.append(
                    f"- {p.signature} ({p.signal_source}): "
                    f"成功率 {p.success_rate:.0%}, 平均|ICIR|={abs(p.avg_icir):.3f}, "
                    f"样本 {p.total_count}"
                )
                if p.example_exprs:
                    lines.append(f"  示例: {p.example_exprs[0][:80]}")

        # 禁区
        active_forbidden = [f for f in self._forbidden_regions if f.rejection_count >= 3]
        active_forbidden.sort(key=lambda f: f.rejection_count, reverse=True)
        if active_forbidden:
            lines.append("")
            lines.append("### 禁区 (避免)")
            for f in active_forbidden[:5]:
                kw_str = ", ".join(f.keywords[:3])
                lines.append(
                    f"- {kw_str}: 与 {f.correlated_with} 相关性 >{f.last_corr:.2f} "
                    f"(被拒 {f.rejection_count} 次)"
                )

        # 策略洞察
        strong_insights = [i for i in self._strategic_insights
                           if i.confidence >= 0.3 and i.evidence_count >= 3]
        strong_insights.sort(key=lambda i: i.confidence, reverse=True)
        if strong_insights:
            lines.append("")
            lines.append("### 策略洞察")
            for i in strong_insights[:5]:
                lines.append(f"- {i.insight} (置信度 {i.confidence:.0%})")

        if len(lines) <= 1:
            return ""  # 无有价值的记忆
        return "\n".join(lines)

    # ============ 内部提炼逻辑 ============

    def _evolve_success_patterns(self, trajectories: list) -> None:
        """按算子签名聚类, 统计成功率"""
        sig_groups: dict[str, list] = defaultdict(list)

        for traj in trajectories:
            if not traj.best_factor or not traj.best_factor.get('expr'):
                continue
            expr = traj.best_factor['expr']
            sig = _extract_op_signature(expr)
            if not sig:
                continue
            sig_groups[sig].append(traj)

        # 更新已有 patterns 或创建新的
        existing_sigs = {p.signature: p for p in self._success_patterns}

        for sig, trajs in sig_groups.items():
            success_count = sum(1 for t in trajs if t.failure_step == -1 and abs(t.icir) >= 0.3)
            total_count = len(trajs)
            avg_icir = sum(abs(t.icir) for t in trajs) / max(total_count, 1)
            exprs = [t.best_factor['expr'] for t in trajs
                     if t.failure_step == -1 and t.best_factor][:3]
            source = trajs[0].direction if trajs else ""

            if sig in existing_sigs:
                p = existing_sigs[sig]
                # 加权平均 ICIR
                old_weight = p.total_count
                new_weight = total_count
                total_w = old_weight + new_weight
                p.avg_icir = (p.avg_icir * old_weight + avg_icir * new_weight) / max(total_w, 1)
                p.success_count += success_count
                p.total_count += total_count
                if exprs:
                    p.example_exprs = (p.example_exprs + exprs)[:5]
            else:
                self._success_patterns.append(SuccessPattern(
                    signature=sig,
                    signal_source=source,
                    example_exprs=exprs,
                    success_count=success_count,
                    total_count=total_count,
                    avg_icir=avg_icir,
                ))

    def _evolve_forbidden_regions(self, trajectories: list) -> None:
        """聚类被相关性拒绝的假说"""
        for traj in trajectories:
            # 检查是否被相关性/冗余拒绝 (error_msg 中有线索)
            err = traj.error_msg or ""
            if not ("相似度" in err or "冗余" in err or "AST" in err
                    or "相关性" in err or "Stage 2" in err):
                continue
            if not traj.hypothesis:
                continue
            keywords = _extract_hypothesis_keywords(traj.hypothesis)
            if not keywords:
                continue

            # 从 error_msg 提取相关性数值和因子名
            corr_value, corr_factor = _parse_corr_from_error(err)

            # 查找已有禁区
            merged = False
            for fr in self._forbidden_regions:
                overlap = len(set(fr.keywords) & set(keywords))
                if overlap >= 1:
                    fr.rejection_count += 1
                    fr.keywords = list(set(fr.keywords + keywords))[:6]
                    if corr_value > fr.last_corr:
                        fr.last_corr = corr_value
                    if corr_factor and corr_factor != "(pool)":
                        fr.correlated_with = corr_factor
                    merged = True
                    break
            if not merged:
                self._forbidden_regions.append(ForbiddenRegion(
                    keywords=keywords[:4],
                    correlated_with=corr_factor or "(pool)",
                    rejection_count=1,
                    last_corr=corr_value,
                ))

    def _evolve_strategic_insights(self, trajectories: list) -> None:
        """统计算子级别成功/失败比"""
        for traj in trajectories:
            if not traj.best_factor or not traj.best_factor.get('expr'):
                continue
            ops = _extract_operators(traj.best_factor['expr'])
            is_success = traj.failure_step == -1 and abs(traj.icir) >= 0.3
            for op in ops:
                if op not in self._op_stats:
                    self._op_stats[op] = {"success": 0, "fail": 0, "total_icir": 0.0}
                stats = self._op_stats[op]
                if is_success:
                    stats["success"] += 1
                else:
                    stats["fail"] += 1
                stats["total_icir"] += abs(traj.icir)

        # 重建 insights
        self._strategic_insights = []
        for op, stats in self._op_stats.items():
            total = stats["success"] + stats["fail"]
            if total < 3:
                continue
            rate = stats["success"] / total
            avg_icir = stats["total_icir"] / total
            category = _classify_operator(op)

            if rate >= 0.5:
                insight = f"{op} 算子成功率 {rate:.0%} (n={total}), 平均|ICIR|={avg_icir:.3f}"
                confidence = min(rate * (total / 10), 1.0)
            elif rate <= 0.2 and total >= 5:
                insight = f"{op} 算子成功率仅 {rate:.0%} (n={total}), 建议避免单独使用"
                confidence = min((1 - rate) * (total / 10), 1.0)
            else:
                continue

            self._strategic_insights.append(StrategicInsight(
                category=category,
                insight=insight,
                confidence=confidence,
                evidence_count=total,
            ))

    # ============ 序列化 ============

    def save(self, filename: str = "experience_memory.json"):
        path = self.save_dir / filename
        data = {
            "updated_at": self._updated_at,
            "success_patterns": [asdict(p) for p in self._success_patterns],
            "forbidden_regions": [asdict(f) for f in self._forbidden_regions],
            "strategic_insights": [asdict(i) for i in self._strategic_insights],
            "op_stats": self._op_stats,
        }
        atomic_json_dump(path, data, indent=2, ensure_ascii=False)

    def load(self, filename: str = "experience_memory.json"):
        path = self.save_dir / filename
        if not path.exists():
            return
        with open(path) as f:
            data = json.load(f)
        self._updated_at = data.get("updated_at", 0.0)
        self._success_patterns = [
            SuccessPattern(**item) for item in data.get("success_patterns", [])
        ]
        self._forbidden_regions = [
            ForbiddenRegion(**item) for item in data.get("forbidden_regions", [])
        ]
        self._strategic_insights = [
            StrategicInsight(**item) for item in data.get("strategic_insights", [])
        ]
        self._op_stats = data.get("op_stats", {})


# ============ 辅助函数 ============

def _extract_op_signature(expr: str, max_depth: int = 2) -> str:
    """提取表达式的算子签名 (如 "Div(Std,Mean)")"""
    try:
        tree = parse_qlib_expr(expr)
        return _sig_node(tree, max_depth)
    except Exception:
        return ""


def _sig_node(node: ASTNode, depth: int) -> str:
    """递归构建签名"""
    if depth <= 0 or not node.op:
        if node.is_field:
            return "$"
        return "C"
    children = [_sig_node(c, depth - 1) for c in node.children]
    return f"{node.op}({','.join(children)})"


def _extract_operators(expr: str) -> list[str]:
    """提取表达式中使用的所有算子"""
    try:
        tree = parse_qlib_expr(expr)
        ops = []
        _collect_ops(tree, ops)
        return ops
    except Exception:
        return []


def _collect_ops(node: ASTNode, ops: list[str]):
    if node.op and node.op not in ('+', '-', '*', '/'):
        ops.append(node.op)
    for c in node.children:
        _collect_ops(c, ops)


def _parse_corr_from_error(err: str) -> tuple[float, str]:
    """从 error_msg 中提取相关性数值和因子名

    常见格式:
    - "AST 相似度过高 (0.85 >= 0.8) vs QA_FACTOR_001"
    - "Stage 2: ... 相关性过高 (0.72 vs QA_FACTOR_002)"
    """
    import re
    # 提取数值 (0.XX)
    corr_match = re.search(r'(\d\.\d+)\s*(?:>=|vs\b)', err)
    corr_value = float(corr_match.group(1)) if corr_match else 0.0
    # 提取因子名 (QA_ 开头)
    name_match = re.search(r'(QA_\w+)', err)
    corr_factor = name_match.group(1) if name_match else ""
    return corr_value, corr_factor


def _extract_hypothesis_keywords(hypothesis: str) -> list[str]:
    """从假说中提取关键词 (简单分词)"""
    import re
    # 提取中文词和英文词
    words = re.findall(r'[a-zA-Z]{3,}|[\u4e00-\u9fff]{2,4}', hypothesis)
    # 过滤通用词
    stopwords = {"the", "and", "for", "with", "that", "this", "from", "are",
                 "因子", "基于", "利用", "通过", "使用", "计算", "生成", "假说"}
    return [w for w in words if w.lower() not in stopwords][:6]


def _classify_operator(op: str) -> str:
    """算子分类"""
    ts_ops = {"Ref", "Mean", "Std", "Sum", "Delta", "Min", "Max", "Slope",
              "Rsquare", "Corr", "Cov"}
    cs_ops = {"Rank"}
    logic_ops = {"If", "Greater", "Less", "Ge", "Le"}

    if op in ts_ops:
        return "时序"
    if op in cs_ops:
        return "截面"
    if op in logic_ops:
        return "条件"
    return "运算"
