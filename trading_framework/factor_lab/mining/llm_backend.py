"""多 LLM 后端 — Claude/Gemini/Codex CLI 统一调用

仅用于假设生成 (hypothesis_only); feedback/evaluation/construct 保持 Claude。
支持: round-robin 轮询 + 每日配额 + 失败降级 + 使用量追踪。
"""
import json
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from cli_paths import CLAUDE_BIN

MINING_DIR = Path(__file__).resolve().parent.parent / "mining_results"


@dataclass
class LLMProvider:
    """单个 LLM CLI 后端"""
    name: str
    cli_path: str
    cli_args: list[str] = field(default_factory=list)
    prompt_flag: str = "-p"
    output_mode: str = "stdout"  # "stdout" | "file" (codex)
    env_remove: list[str] = field(default_factory=list)
    timeout: int = 600
    max_retry: int = 3
    daily_quota: int = 50
    enabled: bool = True


def _load_default_providers() -> list[LLMProvider]:
    """从 config 加载 provider 列表"""
    try:
        from factor_lab.quanta.config import LLM_PROVIDERS
    except ImportError:
        return [_claude_fallback_provider()]

    providers = []
    for cfg in LLM_PROVIDERS:
        providers.append(LLMProvider(
            name=cfg["name"],
            cli_path=cfg["cli_path"],
            cli_args=cfg.get("cli_args", []),
            prompt_flag=cfg.get("prompt_flag", "-p"),
            output_mode=cfg.get("output_mode", "stdout"),
            env_remove=cfg.get("env_remove", []),
            timeout=cfg.get("timeout", 600),
            max_retry=cfg.get("max_retry", 3),
            daily_quota=cfg.get("daily_quota", 50),
            enabled=cfg.get("enabled", True),
        ))
    return providers or [_claude_fallback_provider()]


def _claude_fallback_provider() -> LLMProvider:
    """Claude 默认 provider (永远可用)"""
    return LLMProvider(
        name="claude",
        cli_path=CLAUDE_BIN,
        cli_args=["--print", "--dangerously-skip-permissions",
                  "--output-format", "text"],
        prompt_flag="-p",
        env_remove=["CLAUDECODE"],
        timeout=600,
        daily_quota=999,  # 不限额
    )


class LLMBackend:
    """统一 LLM 调用接口 — round-robin 轮询 + 配额管理

    使用 shared() 获取进程级单例, 确保 round-robin 和配额计数一致。
    """
    _instance: "LLMBackend | None" = None

    @classmethod
    def shared(cls) -> "LLMBackend":
        """进程级单例 — 跨模块共享同一实例"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self, providers: list[LLMProvider] | None = None):
        self.providers = providers or _load_default_providers()
        self._usage: dict[str, int] = {}
        self._errors: dict[str, int] = {}
        self._usage_date: str = ""
        self._round_robin_idx = 0
        self._usage_file = MINING_DIR / "llm_usage.json"
        self._load_usage()

    # --- Public API ---

    def call(self, prompt: str, parser=None):
        """调用 LLM, 自动选择可用 provider (round-robin)

        Args:
            prompt: LLM prompt 文本
            parser: 可选的输出解析器 f(str) -> any

        Returns:
            解析后的结果 (parser 存在时) 或原始字符串
        """
        tried = 0
        for _ in range(len(self.providers)):
            provider = self._select_next()
            if not provider:
                break
            tried += 1
            try:
                output = self._invoke(provider, prompt)
                if output:
                    self._record_usage(provider.name)
                    return parser(output) if parser else output
            except Exception as e:
                self._record_error(provider.name)
                print(f"  [llm] {provider.name} 失败: {e}, 尝试下一个")
                continue

        # 全部失败, fallback 到 Claude (无配额限制)
        if tried == 0 or not any(p.name == "claude" for p in self.providers):
            print("  [llm] 所有 provider 不可用, 使用 Claude fallback")
        return self._invoke_claude_fallback(prompt, parser)

    def stats(self) -> dict:
        """返回今日使用统计"""
        self._check_date_rollover()
        return {
            "date": self._usage_date,
            "usage": dict(self._usage),
            "errors": dict(self._errors),
            "providers": [
                {
                    "name": p.name,
                    "enabled": p.enabled,
                    "quota": p.daily_quota,
                    "used": self._usage.get(p.name, 0),
                    "remaining": max(0, p.daily_quota - self._usage.get(p.name, 0)),
                }
                for p in self.providers
            ],
        }

    # --- Internal ---

    def _select_next(self) -> LLMProvider | None:
        """Round-robin 选择, 跳过配额用尽/未启用/不存在的"""
        self._check_date_rollover()
        n = len(self.providers)
        for _ in range(n):
            idx = self._round_robin_idx % n
            self._round_robin_idx += 1
            p = self.providers[idx]
            if not p.enabled:
                continue
            if self._usage.get(p.name, 0) >= p.daily_quota:
                continue
            # 检查 CLI 是否存在
            if not os.path.isfile(p.cli_path):
                continue
            return p
        return None

    def _invoke(self, provider: LLMProvider, prompt: str) -> str:
        """subprocess 调用 CLI, 支持 stdout 和 file 两种输出模式"""
        env = os.environ.copy()
        for key in provider.env_remove:
            env.pop(key, None)

        work_dir = Path(__file__).resolve().parent.parent.parent.parent

        if provider.output_mode == "file":
            return self._invoke_file_mode(provider, prompt, env, work_dir)

        # stdout 模式 (claude / gemini)
        #
        # prompt 走 stdin，不作为 argv 传递。Windows 上这些 CLI 都由 npm 安装为
        # .cmd 批处理包装器，CreateProcess 经 cmd.exe 执行时 argv 里的换行符会
        # 截断命令行 —— 多行 prompt 只有第一行送达，假说/算子表/约束/输出格式
        # 全部丢失，LLM 收到的是残缺指令却照常返回内容，属于静默失败。
        # stdin 是字节流，不经 cmd.exe 解析，同时规避 8191 字符命令行上限。
        cmd = [provider.cli_path] + list(provider.cli_args)

        result = subprocess.run(
            cmd, input=prompt, capture_output=True, text=True,
            timeout=provider.timeout, cwd=str(work_dir), env=env,
        )
        if result.returncode != 0 and not result.stdout.strip():
            stderr = result.stderr.strip()[:200]
            raise RuntimeError(f"退出码 {result.returncode}: {stderr}")
        return result.stdout.strip()

    def _invoke_file_mode(self, provider: LLMProvider, prompt: str,
                          env: dict, work_dir: Path) -> str:
        """Codex 等需要 -o 文件输出的 CLI (stdout 含 banner 噪声)"""
        MINING_DIR.mkdir(parents=True, exist_ok=True)
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt',
                                             delete=False, dir=str(MINING_DIR)) as tmp:
                tmp_path = tmp.name

            # codex exec ... -o tmpfile  (prompt 走 stdin，理由同 _invoke)
            cmd = ([provider.cli_path] + list(provider.cli_args)
                   + ["-o", tmp_path])

            subprocess.run(
                cmd, input=prompt, capture_output=True, text=True,
                timeout=provider.timeout, cwd=str(work_dir), env=env,
            )
            # -o 写入的是干净的最终回复 (无 banner/thinking)
            if tmp_path and os.path.exists(tmp_path):
                with open(tmp_path, encoding='utf-8') as f:
                    content = f.read().strip()
                if content:
                    return content
            return ""
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def _invoke_claude_fallback(self, prompt: str, parser=None):
        """Claude fallback — 无配额限制, 最后的保底"""
        fallback = _claude_fallback_provider()
        try:
            output = self._invoke(fallback, prompt)
            if output:
                self._record_usage("claude_fallback")
                return parser(output) if parser else output
        except Exception as e:
            print(f"  [llm] Claude fallback 也失败: {e}")
        return "" if parser is None else parser("") if callable(parser) else ""

    def _check_date_rollover(self):
        """日期切换时重置配额"""
        today = date.today().isoformat()
        if self._usage_date != today:
            self._save_usage()  # 保存昨天的
            self._usage = {}
            self._errors = {}
            self._usage_date = today

    def _record_usage(self, name: str):
        self._usage[name] = self._usage.get(name, 0) + 1
        self._save_usage()

    def _record_error(self, name: str):
        self._errors[name] = self._errors.get(name, 0) + 1

    def _load_usage(self):
        today = date.today().isoformat()
        self._usage_date = today
        if self._usage_file.exists():
            try:
                data = json.loads(self._usage_file.read_text())
                if data.get("date") == today:
                    self._usage = data.get("usage", {})
                    self._errors = data.get("errors", {})
            except (json.JSONDecodeError, KeyError):
                pass

    def _save_usage(self):
        self._usage_file.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "date": self._usage_date,
            "usage": self._usage,
            "errors": self._errors,
        }
        try:
            from factor_lab.utils import atomic_json_dump
            atomic_json_dump(self._usage_file, data, indent=2, ensure_ascii=False)
        except (ImportError, OSError):
            # fallback: 直接写
            try:
                self._usage_file.write_text(
                    json.dumps(data, ensure_ascii=False, indent=2))
            except OSError:
                pass
