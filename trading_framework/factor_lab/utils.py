"""共享工具函数 — 原子写入 + JSON 序列化"""
import json
import os
import tempfile
from pathlib import Path


def atomic_json_dump(path: Path, data, **kwargs):
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


def json_default(obj):
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
