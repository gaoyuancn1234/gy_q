"""搜索方向注册表 — 管理方向生命周期 + 跨天累积

方向生命周期:
    pending → exploring → explored → exhausted

- pending: 已规划但未探索
- exploring: 正在探索中 (当前 session)
- explored: 已探索过，可继续深入 (mutation)
- exhausted: 连续失败 >= N 次，不再探索
"""
import json
from datetime import date
from pathlib import Path

from factor_lab.utils import atomic_json_dump


class DirectionRegistry:
    """搜索方向注册表: 管理方向生命周期 + 跨天持久化"""

    def __init__(self, save_path: Path):
        self._path = Path(save_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._directions: list[dict] = []
        self._next_id = 1

    def load(self):
        if not self._path.exists():
            return
        with open(self._path) as f:
            data = json.load(f)
        self._directions = data.get("directions", [])
        self._next_id = data.get("next_id", 1)

    def save(self):
        data = {
            "next_id": self._next_id,
            "directions": self._directions,
        }
        atomic_json_dump(self._path, data, indent=2, ensure_ascii=False)

    def add_directions(self, directions: list[dict], session_id: str):
        """批量注册新方向 (来自 planning_agent)

        Args:
            directions: [{direction, hypothesis, mechanism, time_scale}, ...]
            session_id: 创建该方向的 session run_id
        """
        for d in directions:
            entry = {
                "id": f"dir_{self._next_id:03d}",
                "direction": d.get("direction", ""),
                "hypothesis": d.get("hypothesis", ""),
                "mechanism": d.get("mechanism", ""),
                "time_scale": d.get("time_scale", ""),
                "status": "pending",
                "created_session": session_id,
                "best_icir": 0.0,
                "best_factor_name": "",
                "best_factor_expr": "",
                "best_hypothesis": "",
                "factors_admitted": 0,
                "mutations_tried": 0,
                "consecutive_failures": 0,
                "last_explored": "",
            }
            self._directions.append(entry)
            self._next_id += 1

    def get_pending(self, limit: int = 5) -> list[dict]:
        """获取未探索方向"""
        result = [d for d in self._directions if d["status"] == "pending"]
        return result[:limit]

    def get_depth_targets(self, limit: int = 5) -> list[dict]:
        """获取值得深入的方向 (有因子入池但仍有空间)

        条件: status=explored, factors_admitted > 0, consecutive_failures < 3
        按 best_icir 降序排序 (优先深入最好的方向)
        """
        targets = [
            d for d in self._directions
            if d["status"] == "explored"
            and d["factors_admitted"] > 0
            and d["consecutive_failures"] < 3
        ]
        targets.sort(key=lambda d: abs(d["best_icir"]), reverse=True)
        return targets[:limit]

    def get_explorable(self, limit: int = 5) -> list[dict]:
        """获取所有可探索方向 (explored 但未 exhausted, 无论是否有因子)

        用于深度阶段在 depth_targets 不足时补充
        """
        targets = [
            d for d in self._directions
            if d["status"] == "explored"
            and d["consecutive_failures"] < 3
        ]
        targets.sort(key=lambda d: abs(d["best_icir"]), reverse=True)
        return targets[:limit]

    def needs_new_planning(self) -> bool:
        """所有方向都 explored/exhausted 且无 pending 时触发新规划"""
        if not self._directions:
            return True
        pending = [d for d in self._directions if d["status"] == "pending"]
        if pending:
            return False
        # 有值得深入的也不需要新规划
        depth = self.get_depth_targets(limit=1)
        if depth:
            return False
        return True

    def mark_exploring(self, dir_id: str):
        """标记方向为正在探索"""
        d = self._find(dir_id)
        if d:
            d["status"] = "exploring"

    def record_result(self, dir_id: str, admitted: bool, icir: float = 0.0,
                      session_id: str = "",
                      factor_name: str = "", factor_expr: str = "",
                      hypothesis: str = ""):
        """记录一次探索结果

        Args:
            dir_id: 方向 ID
            admitted: 本次是否有因子入池
            icir: 本次最佳 ICIR
            session_id: 当前 session ID
            factor_name: 最佳因子名
            factor_expr: 最佳因子表达式
            hypothesis: 假说文本
        """
        d = self._find(dir_id)
        if not d:
            return

        d["last_explored"] = date.today().isoformat()
        d["mutations_tried"] += 1

        if admitted:
            d["factors_admitted"] += 1
            d["consecutive_failures"] = 0
            if abs(icir) > abs(d["best_icir"]):
                d["best_icir"] = icir
                if factor_name:
                    d["best_factor_name"] = factor_name
                if factor_expr:
                    d["best_factor_expr"] = factor_expr
                if hypothesis:
                    d["best_hypothesis"] = hypothesis
        else:
            d["consecutive_failures"] += 1

        # 状态转换
        from factor_lab.quanta.config import DAILY_EXHAUSTED_FAILURES
        if d["consecutive_failures"] >= DAILY_EXHAUSTED_FAILURES:
            d["status"] = "exhausted"
        elif self._is_hopeless(d):
            d["status"] = "exhausted"
        else:
            d["status"] = "explored"

    def get_explored_summary(self) -> str:
        """供 planning_agent 参考的已探索摘要，避免重复方向"""
        lines = []
        for d in self._directions:
            if d["status"] in ("explored", "exhausted"):
                status_tag = "exhausted" if d["status"] == "exhausted" else "explored"
                lines.append(
                    f"- [{status_tag}] {d['direction']}: "
                    f"ICIR={d['best_icir']:.3f}, "
                    f"admitted={d['factors_admitted']}, "
                    f"tried={d['mutations_tried']}"
                )
        return "\n".join(lines) if lines else "(无已探索方向)"

    def stats(self) -> dict:
        by_status = {}
        for d in self._directions:
            s = d["status"]
            by_status[s] = by_status.get(s, 0) + 1
        total_admitted = sum(d["factors_admitted"] for d in self._directions)
        return {
            "total": len(self._directions),
            "by_status": by_status,
            "total_admitted": total_admitted,
        }

    def cleanup_exhausted(self, min_tries: int = 5) -> int:
        """批量检查并标记无望方向为 exhausted

        条件: mutations_tried >= min_tries, best_icir <= 0, factors_admitted == 0
        Returns: 本次新标记为 exhausted 的方向数
        """
        count = 0
        for d in self._directions:
            if d["status"] in ("explored", "exploring") and self._is_hopeless(d, min_tries):
                d["status"] = "exhausted"
                count += 1
        return count

    def _is_hopeless(self, d: dict, min_tries: int = 4) -> bool:
        """判断方向是否无望 (ICIR 持续负且无改善)

        条件: 尝试 >= min_tries 次, 从未入池, best_icir <= 0
        """
        return (d["mutations_tried"] >= min_tries
                and d["factors_admitted"] == 0
                and d["best_icir"] <= 0)

    def _find(self, dir_id: str) -> dict | None:
        for d in self._directions:
            if d["id"] == dir_id:
                return d
        return None


