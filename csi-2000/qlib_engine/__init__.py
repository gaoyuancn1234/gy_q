"""Qlib 引擎模块 - 基于微软 Qlib 的 ML 选股引擎"""

import qlib_compat  # noqa: F401  (设置 MLFLOW_ALLOW_FILE_STORE)
from .engine import QlibEngine

__all__ = ['QlibEngine']
