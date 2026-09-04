"""共享工具函数 — 原子写入 + JSON 序列化"""
import json
import os
import tempfile
from pathlib import Path


def atomic_json_dump(path: Path, data, **kwargs):
    """原子写入 JSON: 先写临时文件再 rename，防止中断导致文件损坏

    必须显式指定 encoding='utf-8' (2026-09-02 修复)。
    原实现用 os.fdopen(fd, 'w')，落到 locale 编码 —— Windows 中文环境是 cp936。
    调用方普遍带 ensure_ascii=False (挖掘方向/假说都是中文)，于是文件被写成 GBK，
    而其他地方用 encoding='utf-8' 读回时抛 UnicodeDecodeError。
    更麻烦的是结果取决于进程怎么启动: 带 -X utf8 时 locale 变 utf-8、不带则是 cp936，
    daemon.py 启动 smart_bot 时并未加该参数 —— 同一份文件在不同进程间读写会互相打架。
    """
    path = Path(path)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
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
