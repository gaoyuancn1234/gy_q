"""Qlib / MLflow 版本兼容垫片

导入即生效，必须在 qlib 真正跑 workflow（内部会用 mlflow）之前完成。

背景:
  qlib 0.9.7 用 MLflow 的文件系统追踪后端 (./mlruns) 记录实验。
  MLflow 3.x 起该后端进入维护模式，创建时直接抛异常:
    "The filesystem tracking backend ... is in maintenance mode"
  官方给出的опт-out 就是设置 MLFLOW_ALLOW_FILE_STORE=true。

  这与平台无关（macOS 上同版本 mlflow 一样会炸），所以固化在代码里，
  而不是依赖 shell 环境变量 —— 否则定时任务、飞书机器人子进程里
  照样会失败。

用 setdefault: 若使用者显式设了值（比如迁移到 sqlite 后端），不覆盖。
"""
import os

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
