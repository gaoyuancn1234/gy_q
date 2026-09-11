"""写入权: 中证2000 只允许 Tushare 写 Qlib 供给目录。"""
from __future__ import annotations

from pathlib import Path

from data_hub.paths import TUSHARE_ONLY, qlib_serve_dir


def refuse_non_tushare(universe: str, target_dir) -> None:
    """新浪/BaoStock 写入 CSI2000 目录时直接失败。

    2026-09-10 DailyRunner 曾按 CSI300 逻辑用新浪全量重建 2955 只,
    目标正是 cn_data_csi2000, 会盖掉 Tushare 时点成分股数据。
    """
    target = Path(str(target_dir)).expanduser().resolve()
    blocked = {u: qlib_serve_dir(u).resolve() for u in TUSHARE_ONLY}
    hit_universe = universe in blocked
    hit_path = target in blocked.values()
    name_hit = "cn_data_csi2000" in target.name
    if not (hit_universe or hit_path or name_hit):
        return
    raise SystemExit(
        f"拒绝非 Tushare 写入: universe={universe} target={target}\n"
        "中证2000 只走 python -m data_hub refresh "
        "或 qlib_engine.data_setup_tushare。"
    )
