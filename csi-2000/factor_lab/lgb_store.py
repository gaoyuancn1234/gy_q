"""滚动 LightGBM 落盘, 给 14:45 截断出分用。

用户指定: 模型目录在 rolling/models/{preset}/ 。
"""
from __future__ import annotations

import json
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_ROOT = (
    PROJECT_DIR / "factor_lab" / "results" / "rolling" / "models"
)


def booster_dir(preset: str) -> Path:
    """某个预设的 booster 目录。"""
    return MODEL_ROOT / preset


def models_ready(preset: str) -> bool:
    """至少有一个 seed 文本和一个含特征统计的 meta。"""
    d = booster_dir(preset)
    if not d.exists():
        return False
    seeds = list(d.glob("seed*.txt"))
    meta = d / "meta.json"
    if not seeds or not meta.exists():
        return False
    try:
        info = json.loads(meta.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    names = info.get("feature_names") or []
    mean = info.get("mean_train")
    return bool(names) and mean is not None


def save_booster(model, dest: Path, seed_i: int) -> None:
    """把 Qlib LGBModel 的 booster 写成 txt。"""
    dest.mkdir(parents=True, exist_ok=True)
    booster = getattr(model, "model", None)
    if booster is None:
        raise RuntimeError("LGBModel.model 为空, 无法落盘")
    booster.save_model(str(dest / f"seed{seed_i}.txt"))


def write_infer_meta(dataset, dest: Path, extra: dict) -> None:
    """从已 fit 的 DatasetH 抽出 RobustZScoreNorm 统计。"""
    from qlib.data.dataset.handler import DataHandlerLP

    dest.mkdir(parents=True, exist_ok=True)
    feat = dataset.prepare(
        "test",
        col_set="feature",
        data_key=DataHandlerLP.DK_I,
    )
    names = []
    for col in feat.columns:
        if isinstance(col, tuple):
            names.append(str(col[-1]))
        else:
            names.append(str(col))
    mean_train = None
    std_train = None
    clip = True
    handler = getattr(dataset, "handler", None)
    procs = []
    if handler is not None:
        for attr in (
            "infer_processors",
            "_infer_processors",
            "shared_processors",
        ):
            got = getattr(handler, attr, None) or []
            if isinstance(got, (list, tuple)):
                procs.extend(got)
    for proc in procs:
        name = proc.__class__.__name__
        if name != "RobustZScoreNorm":
            continue
        mean_train = getattr(proc, "mean_train", None)
        std_train = getattr(proc, "std_train", None)
        clip = bool(getattr(proc, "clip_outlier", True))
        break
    if mean_train is None or std_train is None:
        names_p = [p.__class__.__name__ for p in procs]
        raise RuntimeError(
            "handler 没有 RobustZScoreNorm 统计, "
            f"截断出分会对不齐 processors={names_p}"
        )
    payload = {
        **extra,
        "feature_names": names,
        "n_features": len(names),
        "clip_outlier": clip,
        "mean_train": [float(x) for x in mean_train],
        "std_train": [float(x) for x in std_train],
    }
    (dest / "meta.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_boosters(preset: str) -> tuple[list, dict]:
    """读全部 seed 和 meta。"""
    import lightgbm as lgb

    d = booster_dir(preset)
    meta_path = d / "meta.json"
    info = json.loads(meta_path.read_text(encoding="utf-8"))
    boosters = []
    for path in sorted(d.glob("seed*.txt")):
        boosters.append(lgb.Booster(model_file=str(path)))
    if not boosters:
        raise FileNotFoundError(f"没有 booster: {d}")
    return boosters, info


def apply_robust_zscore(raw, info: dict):
    """按训练窗的 median/MAD 标准化, 与 Qlib RobustZScoreNorm 一致。"""
    import numpy as np

    names = info["feature_names"]
    x = raw.reindex(columns=names).to_numpy(dtype=np.float64)
    mean = np.asarray(info["mean_train"], dtype=np.float64)
    std = np.asarray(info["std_train"], dtype=np.float64)
    x = (x - mean) / std
    if info.get("clip_outlier", True):
        x = np.clip(x, -3.0, 3.0)
    x = np.nan_to_num(x, nan=0.0)
    return x
