"""外部 CLI 可执行文件路径解析（跨平台）

原先各模块硬编码 macOS 路径 /usr/local/bin/<name>，在 Windows/Linux 上不可用。
统一改为运行时探测，解析顺序:

  1. 环境变量覆盖 (CLAUDE_BIN / GEMINI_BIN / CODEX_BIN)
  2. PATH 查找 (Windows 上会匹配 claude.cmd 等)
  3. 常见安装路径兜底
  4. 裸名字 (交给系统 PATH 解析，失败时报错信息更直观)
"""
import os
import shutil
from pathlib import Path

# 各 CLI 的兜底路径（PATH 查不到时依次尝试）
_FALLBACKS = {
    "claude": [
        "/usr/local/bin/claude",
        "/opt/homebrew/bin/claude",
        str(Path.home() / "AppData/Roaming/npm/claude.cmd"),
        str(Path.home() / ".local/bin/claude"),
    ],
    "gemini": ["/usr/local/bin/gemini", "/opt/homebrew/bin/gemini",
               str(Path.home() / "AppData/Roaming/npm/gemini.cmd")],
    "codex": ["/usr/local/bin/codex", "/opt/homebrew/bin/codex",
              str(Path.home() / "AppData/Roaming/npm/codex.cmd")],
}


def resolve_cli(name: str) -> str:
    """解析 CLI 可执行文件路径

    Args:
        name: CLI 名称，如 'claude' / 'gemini' / 'codex'

    Returns:
        可执行文件路径；都找不到时返回裸名字
    """
    env_override = os.environ.get(f"{name.upper()}_BIN")
    if env_override:
        return env_override

    found = shutil.which(name)
    if found:
        return found

    for candidate in _FALLBACKS.get(name, []):
        if Path(candidate).exists():
            return candidate

    return name


CLAUDE_BIN = resolve_cli("claude")
GEMINI_BIN = resolve_cli("gemini")
CODEX_BIN = resolve_cli("codex")
