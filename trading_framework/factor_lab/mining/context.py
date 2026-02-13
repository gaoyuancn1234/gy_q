"""Agent 记忆管理 — 管理 AGENT_CONTEXT.md

跨运行持久记忆，供 Claude agent 生成因子假说时参考。
"""
import json
from datetime import datetime
from pathlib import Path

MINING_DIR = Path(__file__).resolve().parent.parent / "mining_results"
CONTEXT_FILE = MINING_DIR / "AGENT_CONTEXT.md"
INDEX_FILE = MINING_DIR / "index.json"


class MiningContext:
    """管理 AGENT_CONTEXT.md — Claude agent 的跨运行记忆"""

    def __init__(self):
        MINING_DIR.mkdir(parents=True, exist_ok=True)

    def load(self) -> str:
        """加载当前记忆文本 (给 Claude prompt 用)"""
        if CONTEXT_FILE.exists():
            return CONTEXT_FILE.read_text(encoding="utf-8")
        return self._create_initial_context()

    def _create_initial_context(self) -> str:
        text = """# Factor Mining Agent Memory

## Stats
Total runs: 0 | Factors tested: 0 | Promising: 0 | Beat M01: 0

## Best Discoveries
(none yet)

## Failed Approaches
(none yet)

## Current Directions
- 波动率结构因子 (短期/长期波动率比)
- 量价背离因子 (价格创新高但成交量萎缩)
- 动量衰减/加速因子 (动量的二阶导)
- 流动性冲击因子 (异常换手率)
- 价格形态因子 (高低点位置)

## Recent Runs
(none yet)
"""
        CONTEXT_FILE.write_text(text, encoding="utf-8")
        return text

    def update_after_run(self, run_result: dict):
        """运行后更新记忆: 统计、最佳发现、失败记录、下次方向

        保持文件 < 4000 tokens，旧运行只保留摘要。
        """
        index = self._load_index()
        run_id = run_result.get("run_id", "unknown")
        index[run_id] = {
            "date": run_result.get("date", ""),
            "factors_tested": run_result.get("factors_tested", 0),
            "promising_count": run_result.get("promising_count", 0),
            "beat_baseline": run_result.get("beat_baseline", False),
            "focus": run_result.get("focus", ""),
        }
        self._save_index(index)
        self._rebuild_context(index, run_result)

    def _rebuild_context(self, index: dict, latest_result: dict):
        """根据完整索引和最新结果重建 AGENT_CONTEXT.md"""
        total_runs = len(index)
        total_tested = sum(r.get("factors_tested", 0) for r in index.values())
        total_promising = sum(r.get("promising_count", 0) for r in index.values())
        total_beat = sum(1 for r in index.values() if r.get("beat_baseline"))

        # Best discoveries
        discoveries_dir = MINING_DIR / "discoveries"
        discoveries_lines = []
        if discoveries_dir.exists():
            for f in sorted(discoveries_dir.glob("*.json")):
                try:
                    d = json.loads(f.read_text(encoding="utf-8"))
                    name = d.get("name", "?")
                    icir = d.get("icir", 0)
                    hyp = d.get("hypothesis", "")[:60]
                    discoveries_lines.append(f"| {name} | {icir:.3f} | {hyp} |")
                except Exception:
                    continue

        discoveries_text = "(none yet)"
        if discoveries_lines:
            discoveries_text = "| Factor | ICIR | Hypothesis |\n|--------|------|------------|\n"
            discoveries_text += "\n".join(discoveries_lines)

        # Failed approaches (from latest + history)
        failed = latest_result.get("failed_approaches", [])
        failed_text = "\n".join(f"- {f}" for f in failed[-10:]) if failed else "(none yet)"

        # Current directions
        directions = latest_result.get("next_directions", [])
        if not directions:
            directions = [
                "波动率结构因子 (短期/长期波动率比)",
                "量价背离因子 (价格创新高但成交量萎缩)",
                "动量衰减/加速因子 (动量的二阶导)",
                "流动性冲击因子 (异常换手率)",
                "价格形态因子 (高低点位置)",
            ]
        directions_text = "\n".join(f"- {d}" for d in directions[:8])

        # Recent runs (last 5)
        sorted_runs = sorted(index.items(), key=lambda x: x[1].get("date", ""), reverse=True)[:5]
        recent_lines = []
        for rid, info in sorted_runs:
            dt = info.get("date", "?")
            tested = info.get("factors_tested", 0)
            prom = info.get("promising_count", 0)
            beat = "Y" if info.get("beat_baseline") else "N"
            focus = info.get("focus", "")[:40]
            recent_lines.append(f"- {rid} ({dt}): tested={tested}, promising={prom}, beat={beat} | {focus}")

        recent_text = "\n".join(recent_lines) if recent_lines else "(none yet)"

        text = f"""# Factor Mining Agent Memory

## Stats
Total runs: {total_runs} | Factors tested: {total_tested} | Promising: {total_promising} | Beat M01: {total_beat}

## Best Discoveries
{discoveries_text}

## Failed Approaches
{failed_text}

## Current Directions
{directions_text}

## Recent Runs
{recent_text}
"""
        CONTEXT_FILE.write_text(text, encoding="utf-8")

    def get_focus_areas(self) -> str:
        """根据历史确定本次探索方向 (避免重复失败)"""
        ctx = self.load()
        # 提取 Current Directions 部分
        if "## Current Directions" in ctx:
            start = ctx.index("## Current Directions") + len("## Current Directions")
            rest = ctx[start:]
            end = rest.index("##") if "##" in rest else len(rest)
            return rest[:end].strip()
        return ""

    def get_all_tried_names(self) -> list[str]:
        """已尝试过的所有因子名 (含失败的)"""
        names = set()
        runs_dir = MINING_DIR / "runs"
        if runs_dir.exists():
            for f in sorted(runs_dir.glob("*.json")):
                try:
                    run = json.loads(f.read_text(encoding="utf-8"))
                    for h in run.get("hypotheses", []):
                        names.add(h.get("name", ""))
                except Exception:
                    continue
        names.discard("")
        return sorted(names)

    def _load_index(self) -> dict:
        if INDEX_FILE.exists():
            try:
                return json.loads(INDEX_FILE.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {}

    def _save_index(self, index: dict):
        INDEX_FILE.write_text(
            json.dumps(index, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
