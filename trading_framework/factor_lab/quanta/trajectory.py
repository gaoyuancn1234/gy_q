"""轨迹数据结构 + 池管理 + Trace 系统 (v3)

每条轨迹记录一个因子从假说到评估的完整生命周期:
  idea -> factor_candidates -> best_factor -> evaluation -> reward

v3 新增 Trace 系统 (对齐论文 core/proposal.py):
  HypothesisFeedback: LLM 结构化反馈
  TraceEntry: 单轮 (假说, 实验, 反馈) 记录
  DirectionTrace: 单个 direction 的轨迹历史
"""
import json
import os
import tempfile
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from .config import HISTORY_LIMIT, TRACE_ENTRY_TMPL


# ============ Trace 系统 (对齐论文 core/proposal.py) ============

@dataclass
class HypothesisFeedback:
    """LLM 结构化反馈 (对齐论文 core/proposal.py:60-83)"""
    observations: str = ""           # 对实验结果的关键观察
    hypothesis_evaluation: str = ""  # 假说有效性评估
    new_hypothesis: str = ""         # 改进方向建议
    reasoning: str = ""              # 接受/拒绝的推理
    decision: bool = False           # True=接受(新SOTA), False=拒绝

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'HypothesisFeedback':
        if not d:
            return cls()
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})

    def summary(self) -> str:
        """简短摘要 (用于 prompt 渲染)"""
        parts = []
        if self.observations:
            parts.append(self.observations[:100])
        if self.new_hypothesis:
            parts.append(f"改进: {self.new_hypothesis[:80]}")
        return "; ".join(parts) if parts else "(无反馈)"


@dataclass
class TraceEntry:
    """单轮 (假说, 实验, 反馈) 记录

    扁平化论文的 tuple[Hypothesis, Experiment, HypothesisFeedback]
    """
    round_idx: int = 0
    hypothesis: str = ""
    factor_name: str = ""
    factor_expr: str = ""
    ic: float = 0.0
    icir: float = 0.0
    rank_ic: float = 0.0
    backtest_metrics: dict = field(default_factory=dict)  # {sharpe, ARR, MDD}
    feedback: Optional[HypothesisFeedback] = None
    traj_id: str = ""  # 关联的 Trajectory ID

    def to_dict(self) -> dict:
        d = asdict(self)
        if self.feedback:
            d['feedback'] = self.feedback.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> 'TraceEntry':
        if not d:
            return cls()
        fb_data = d.get('feedback', None)
        known = {f.name for f in cls.__dataclass_fields__.values()}
        entry = cls(**{k: v for k, v in d.items() if k in known and k != 'feedback'})
        if fb_data and isinstance(fb_data, dict):
            entry.feedback = HypothesisFeedback.from_dict(fb_data)
        return entry


class DirectionTrace:
    """单个 direction 的轨迹历史 (对齐论文 Trace)

    每个 direction 独立维护历史，支持:
    - 无限累积 entries
    - get_recent(limit) 返回最近 N 条
    - get_sota() 返回最后一个 decision=True 的 entry
    - render_for_prompt() 序列化为 prompt 文本
    """

    def __init__(self, direction_id: int):
        self.direction_id = direction_id
        self.entries: list[TraceEntry] = []
        self._sota_idx: int = -1  # SOTA entry 的 index

    def append(self, entry: TraceEntry):
        """追加记录，自动更新 SOTA"""
        entry.round_idx = len(self.entries)
        self.entries.append(entry)
        if entry.feedback and entry.feedback.decision:
            self._sota_idx = len(self.entries) - 1

    def get_recent(self, limit: int = HISTORY_LIMIT) -> list[TraceEntry]:
        """返回最近 N 条"""
        return self.entries[-limit:]

    def get_sota(self) -> Optional[TraceEntry]:
        """返回最后一个 decision=True 的 entry"""
        if 0 <= self._sota_idx < len(self.entries):
            return self.entries[self._sota_idx]
        return None

    def render_for_prompt(self, limit: int = HISTORY_LIMIT) -> str:
        """序列化历史为 prompt 文本 (对齐论文 hypothesis_and_feedback Jinja2 模板)"""
        recent = self.get_recent(limit)
        if not recent:
            return "(暂无历史记录)"

        blocks = []
        for entry in recent:
            bt_line = ""
            if entry.backtest_metrics:
                bt = entry.backtest_metrics
                bt_line = (f"  回测: Sharpe={bt.get('sharpe', 0):.3f}, "
                           f"ARR={bt.get('ARR', 0):.2f}%, MDD={bt.get('MDD', 0):.2f}%")

            fb_summary = entry.feedback.summary() if entry.feedback else "(无反馈)"
            decision = entry.feedback.decision if entry.feedback else False

            decision_text = "接受 (新 SOTA)" if decision else "拒绝"
            block = TRACE_ENTRY_TMPL.format(
                round_idx=entry.round_idx,
                hypothesis=entry.hypothesis[:150],
                factor_name=entry.factor_name,
                factor_expr=entry.factor_expr[:120],
                ic=entry.ic,
                icir=entry.icir,
                rank_ic=entry.rank_ic,
                backtest_line=bt_line,
                feedback_summary=fb_summary,
                decision_text=decision_text,
            )
            blocks.append(block)

        return "\n\n".join(blocks)

    def render_factor_list_for_prompt(self) -> str:
        """渲染已尝试因子列表 (对齐论文 target_list)"""
        if not self.entries:
            return "(暂无已尝试因子)"
        lines = []
        for entry in self.entries:
            if entry.factor_name and entry.factor_expr:
                lines.append(f"- {entry.factor_name}: {entry.factor_expr[:100]}")
        return "\n".join(lines) if lines else "(暂无已尝试因子)"

    def to_dict(self) -> dict:
        return {
            "direction_id": self.direction_id,
            "entries": [e.to_dict() for e in self.entries],
            "sota_idx": self._sota_idx,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'DirectionTrace':
        trace = cls(direction_id=d.get("direction_id", 0))
        for e_data in d.get("entries", []):
            trace.entries.append(TraceEntry.from_dict(e_data))
        trace._sota_idx = d.get("sota_idx", -1)
        return trace


# ============ 轨迹 ============

@dataclass
class Trajectory:
    """一条因子挖掘轨迹"""
    # 标识
    id: str = ""
    direction_id: int = 0          # 初始方向编号 (0~N_DIRECTIONS-1)
    iteration: int = 0             # 演化迭代轮次 (0=初始)
    phase: str = "init"            # init / mutation / crossover
    parent_ids: list[str] = field(default_factory=list)  # 父轨迹 (crossover 时多个)

    # Step 0: Idea
    hypothesis: str = ""
    direction: str = ""            # 信号方向描述 (价格/量能/联合等)
    mechanism: str = ""            # 机制 (动量/反转/regime等)

    # Step 1: Factor
    factor_candidates: list[dict] = field(default_factory=list)  # [{name, expr, desc}]
    best_factor: Optional[dict] = None  # {name, expr, desc}
    consistency_ok: Optional[bool] = None  # None=未检查, True/False=已验证
    constraint_ok: bool = False
    qlib_ok: bool = False

    # Step 2: Evaluation
    ic: float = 0.0
    icir: float = 0.0
    rank_ic: float = 0.0
    rank_icir: float = 0.0
    reward: float = 0.0            # |RankIC| 作为奖励信号

    # 反馈
    eval_feedback: str = ""
    failure_step: int = -1         # 0=idea, 1=factor, 2=eval, -1=未失败
    error_msg: str = ""

    # v3 新增: LLM 反馈 + 回测 + 再生
    llm_feedback: Optional[dict] = None     # HypothesisFeedback.to_dict()
    backtest_metrics: Optional[dict] = None  # {sharpe, ARR, MDD, ...}
    regen_attempts: int = 0                  # 再生重试次数

    # 元数据
    created_at: float = 0.0
    claude_calls: int = 0
    eval_time: float = 0.0

    def __post_init__(self):
        if not self.created_at:
            self.created_at = time.time()

    def compute_reward(self):
        """计算奖励 = |RankIC|, 失败的轨迹奖励为 0"""
        if self.failure_step >= 0:
            self.reward = 0.0
        else:
            self.reward = abs(self.rank_ic) if self.rank_ic else abs(self.ic)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'Trajectory':
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in known_fields}
        return cls(**filtered)


# ============ 轨迹池 ============

class TrajectoryPool:
    """轨迹池: 管理所有轨迹的生命周期 + DirectionTrace"""

    def __init__(self, save_dir: Path):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._trajectories: dict[str, Trajectory] = {}
        self._next_id = 1
        self._direction_traces: dict[int, DirectionTrace] = {}

    def add(self, traj: Trajectory) -> str:
        """添加轨迹, 自动分配 ID"""
        if not traj.id:
            traj.id = f"T{self._next_id:04d}"
            self._next_id += 1
        self._trajectories[traj.id] = traj
        return traj.id

    def get(self, traj_id: str) -> Optional[Trajectory]:
        return self._trajectories.get(traj_id)

    def all(self) -> list[Trajectory]:
        return list(self._trajectories.values())

    def get_direction_trace(self, direction_id: int) -> DirectionTrace:
        """获取或创建 direction 的 trace"""
        if direction_id not in self._direction_traces:
            self._direction_traces[direction_id] = DirectionTrace(direction_id)
        return self._direction_traces[direction_id]

    def get_by_iteration(self, iteration: int) -> list[Trajectory]:
        return [t for t in self._trajectories.values() if t.iteration == iteration]

    def get_by_phase(self, phase: str) -> list[Trajectory]:
        return [t for t in self._trajectories.values() if t.phase == phase]

    def get_successful(self) -> list[Trajectory]:
        """返回评估成功 (failure_step == -1) 且有正向奖励的轨迹"""
        return [t for t in self._trajectories.values()
                if t.failure_step == -1 and t.reward > 0]

    def get_top_k(self, k: int = 20) -> list[Trajectory]:
        """按 reward 降序返回 top-k 轨迹"""
        successful = self.get_successful()
        successful.sort(key=lambda t: t.reward, reverse=True)
        return successful[:k]

    def get_medium_reward(self, low_q: float = 0.3, high_q: float = 0.7) -> list[Trajectory]:
        """返回中等奖励的轨迹 (适合 mutation)"""
        successful = self.get_successful()
        if len(successful) < 3:
            return successful
        rewards = sorted([t.reward for t in successful])
        low_th = rewards[int(len(rewards) * low_q)]
        high_th = rewards[int(len(rewards) * high_q)]
        return [t for t in successful if low_th <= t.reward <= high_th]

    def get_failed(self) -> list[Trajectory]:
        """返回失败的轨迹"""
        return [t for t in self._trajectories.values() if t.failure_step >= 0]

    @property
    def size(self) -> int:
        return len(self._trajectories)

    def stats(self) -> dict:
        """统计信息"""
        all_t = self.all()
        successful = self.get_successful()
        rewards = [t.reward for t in successful] if successful else [0]
        return {
            "total": len(all_t),
            "successful": len(successful),
            "failed": len(self.get_failed()),
            "avg_reward": sum(rewards) / len(rewards),
            "max_reward": max(rewards),
            "by_phase": {
                "init": len(self.get_by_phase("init")),
                "mutation": len(self.get_by_phase("mutation")),
                "crossover": len(self.get_by_phase("crossover")),
            },
            "by_iteration": {
                i: len(self.get_by_iteration(i))
                for i in range(max((t.iteration for t in all_t), default=0) + 1)
            },
            "direction_traces": {
                str(did): len(dt.entries)
                for did, dt in self._direction_traces.items()
            },
        }

    def save(self, filename: str = "trajectories.json"):
        path = self.save_dir / filename
        data = {
            "next_id": self._next_id,
            "trajectories": {k: v.to_dict() for k, v in self._trajectories.items()},
            "direction_traces": {
                str(k): v.to_dict() for k, v in self._direction_traces.items()
            },
        }
        _atomic_json_dump(path, data, indent=2, ensure_ascii=False, default=_json_default)

    def load(self, filename: str = "trajectories.json"):
        path = self.save_dir / filename
        if not path.exists():
            return
        with open(path) as f:
            data = json.load(f)
        self._next_id = data.get("next_id", 1)
        self._trajectories = {
            k: Trajectory.from_dict(v)
            for k, v in data.get("trajectories", {}).items()
        }
        # 加载 direction traces
        for k, v in data.get("direction_traces", {}).items():
            self._direction_traces[int(k)] = DirectionTrace.from_dict(v)


def _atomic_json_dump(path: Path, data, **kwargs):
    """原子写入 JSON: 先写临时文件再 rename，防止中断导致文件损坏"""
    path = Path(path)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix='.tmp')
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(data, f, **kwargs)
        os.replace(tmp_path, str(path))
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _json_default(obj):
    """numpy/float32 等类型的 JSON 序列化"""
    import numpy as np
    if isinstance(obj, (np.floating, np.complexfloating)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
