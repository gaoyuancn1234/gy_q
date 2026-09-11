#!/usr/bin/env python3
"""只训最后一个 rolling 窗口并落盘 booster。

当前全量重训进程不会保存模型。pkl 落地后跑本脚本, 约 25 分钟,
给 `python -m daily auction` 用。不覆盖预测 pkl。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))


def main(argv: list[str] | None = None) -> int:
    """入口。"""
    import yaml
    import qlib
    from qlib.data.dataset import DatasetH
    from qlib_paths import qlib_init_kwargs
    from factor_lab.lgb_store import booster_dir
    from factor_lab.run_rolling_benchmark import (
        INSTRUMENTS,
        ROLLING_CONFIGS,
        generate_rolling_windows,
        train_window,
    )

    ap = argparse.ArgumentParser(description="落盘最后一窗 LightGBM")
    ap.add_argument("--preset", default="")
    args = ap.parse_args(argv)

    cfg_path = PROJECT_DIR / "config" / "signal_config.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    preset = args.preset or cfg.get("preset") or "alpha158"
    config_name = cfg.get("rolling_config", "D_expand_3v_3r")
    qlib.init(**qlib_init_kwargs())
    from qlib.data import D
    cal = D.calendar()
    data_end = str(cal[-1])[:10]
    test_end = min(str(cfg.get("test_end", data_end)), data_end)
    windows = generate_rolling_windows(
        config_name,
        ROLLING_CONFIGS[config_name],
        cfg.get("test_start", "2024-01-01"),
        test_end,
        data_end=data_end,
    )
    if not windows:
        raise SystemExit("没有 rolling 窗口")
    w = windows[-1]
    print(
        f"落盘窗口 {w['window_num']}/{len(windows)} "
        f"train {w['train_start']}~{w['train_end']} "
        f"preset={preset}",
        flush=True,
    )
    if preset in ("alpha158",):
        from qlib.contrib.data.handler import Alpha158
        handler = Alpha158(
            start_time=w["train_start"],
            end_time=w["pred_end"],
            fit_start_time=w["train_start"],
            fit_end_time=w["train_end"],
            instruments=INSTRUMENTS,
        )
    else:
        from factor_lab.factors.presets import build_handler
        handler = build_handler(
            preset,
            start_time=w["train_start"],
            end_time=w["pred_end"],
            fit_start_time=w["train_start"],
            fit_end_time=w["train_end"],
            instruments=INSTRUMENTS,
        )
    dataset = DatasetH(
        handler=handler,
        segments={
            "train": (w["train_start"], w["train_end"]),
            "valid": (w["valid_start"], w["valid_end"]),
            "test": (w["pred_start"], w["pred_end"]),
        },
    )
    dest = booster_dir(preset)
    train_window(
        dataset,
        cfg.get("model", "LightGBM"),
        prev_best_iters=[],
        save_dir=dest,
        save_meta={
            "window": w["window_num"],
            "preset": preset,
            "train_start": w["train_start"],
            "train_end": w["train_end"],
            "pred_start": w["pred_start"],
            "pred_end": w["pred_end"],
        },
    )
    print(f"已写入 {dest}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
