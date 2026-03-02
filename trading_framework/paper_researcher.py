#!/usr/bin/env python3
"""论文调研与复现管理器

在隔离的 git worktree 分支中调研和复现 arXiv 论文，
不影响主分支上的定时任务 (daily_runner / factor_miner / intraday_monitor)。

用法:
    python paper_researcher.py --status
    python paper_researcher.py --add https://arxiv.org/abs/2602.14670
    python paper_researcher.py --run paper_001 [read|plan|replicate|evaluate]
    python paper_researcher.py --remove paper_001
"""

import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import date, datetime
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
PAPERS_DIR = PROJECT_DIR / "papers"
REGISTRY_FILE = PAPERS_DIR / "registry.json"
PDFS_DIR = PAPERS_DIR / "pdfs"
REPO_ROOT = PROJECT_DIR.parent  # qlib/
WORKTREE_ROOT = REPO_ROOT.parent / "qlib-papers"  # sibling of qlib/

# 主项目的 auto-memory 目录 (Claude CLI 按 cwd 路径隔离)
_PROJECT_KEY = str(REPO_ROOT).replace('/', '-')
MAIN_MEMORY_DIR = Path.home() / ".claude" / "projects" / _PROJECT_KEY / "memory"

# Claude CLI 调用时清除嵌套会话变量
_CLEAN_ENV = {k: v for k, v in os.environ.items()
              if k not in ('CLAUDECODE', 'CLAUDE_CODE_ENTRYPOINT')}

# 状态流转
PHASES = ["download", "read", "plan", "replicate", "evaluate"]
STATUS_MAP = {
    "download": "pending",
    "read": "reading",
    "plan": "planning",
    "replicate": "replicating",
    "evaluate": "evaluating",
}


def _ensure_dirs():
    for d in [PAPERS_DIR, PDFS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def _load_registry() -> dict:
    if REGISTRY_FILE.exists():
        return json.loads(REGISTRY_FILE.read_text(encoding='utf-8'))
    return {}


def _save_registry(reg: dict):
    _ensure_dirs()
    REGISTRY_FILE.write_text(
        json.dumps(reg, ensure_ascii=False, indent=2, default=str),
        encoding='utf-8')


def _next_paper_id(reg: dict) -> str:
    nums = []
    for k in reg:
        if k.startswith("paper_"):
            try:
                nums.append(int(k.split("_")[1]))
            except (ValueError, IndexError):
                continue
    next_num = max(nums, default=0) + 1
    return f"paper_{next_num:03d}"


def _parse_arxiv_url(url: str) -> str:
    """从 URL 中提取 arXiv ID (如 2602.14670)"""
    m = re.search(r'(\d{4}\.\d{4,5})(v\d+)?', url)
    if m:
        return m.group(1)
    raise ValueError(f"无法从 URL 中提取 arXiv ID: {url}")


def search_arxiv(query: str, max_results: int = 3) -> list:
    """搜索 arXiv，返回 [{arxiv_id, title, url, summary, authors, date}]"""
    import urllib.parse
    import xml.etree.ElementTree as ET

    encoded = urllib.parse.quote(query)
    api_url = f"https://export.arxiv.org/api/query?search_query=ti:{encoded}&max_results={max_results}"
    result = subprocess.run(
        ["curl", "-sf", "--max-time", "15", api_url],
        capture_output=True, text=True, timeout=20)

    if result.returncode != 0 or not result.stdout:
        return []

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    try:
        root = ET.fromstring(result.stdout)
    except ET.ParseError:
        return []

    papers = []
    for entry in root.findall("atom:entry", ns):
        id_text = entry.findtext("atom:id", "", ns)  # http://arxiv.org/abs/2602.14670v1
        m = re.search(r'(\d{4}\.\d{4,5})', id_text)
        if not m:
            continue
        arxiv_id = m.group(1)
        title = entry.findtext("atom:title", "", ns).strip().replace('\n', ' ')
        summary = entry.findtext("atom:summary", "", ns).strip()[:300]
        authors = [a.findtext("atom:name", "", ns)
                   for a in entry.findall("atom:author", ns)]
        published = entry.findtext("atom:published", "", ns)[:10]
        papers.append({
            "arxiv_id": arxiv_id,
            "title": title,
            "url": f"https://arxiv.org/abs/{arxiv_id}",
            "summary": summary,
            "authors": ", ".join(authors[:3]) + ("..." if len(authors) > 3 else ""),
            "date": published,
        })
    return papers


def looks_like_paper_title(text: str) -> bool:
    """启发式判断文本是否可能是论文标题"""
    if len(text) < 15 or len(text) > 300:
        return False
    # 英文字符占比 > 50%
    alpha_count = sum(1 for c in text if c.isascii() and c.isalpha())
    if alpha_count / max(len(text), 1) < 0.5:
        return False
    # 至少 4 个英文单词且含至少 1 个首字母大写的词
    words = re.findall(r'[A-Za-z]{2,}', text)
    if len(words) < 4:
        return False
    capitalized = sum(1 for w in words if w[0].isupper())
    if capitalized < 1:
        return False
    # 排除纯日常对话: 全部小写单词且无技术术语
    lower_text = text.lower()
    casual_words = {'hi', 'hello', 'hey', 'how', 'are', 'you', 'doing',
                    'thanks', 'thank', 'please', 'good', 'fine', 'what',
                    'can', 'could', 'would', 'should', 'will', 'the',
                    'is', 'am', 'was', 'were', 'been', 'have', 'has'}
    if all(w.lower() in casual_words for w in words):
        return False
    return True


class PaperResearcher:
    def __init__(self):
        _ensure_dirs()
        self.registry = _load_registry()

    def _save(self):
        _save_registry(self.registry)

    # ── 生命周期 ──────────────────────────────────────────

    def add_paper(self, url: str, notes: str = "") -> str:
        """添加论文 → 下载 PDF → 创建 worktree → 返回 paper_id"""
        url = url.strip()

        # 判断是 arXiv 还是普通 PDF URL
        arxiv_m = re.search(r'(\d{4}\.\d{4,5})', url)
        if arxiv_m:
            paper_key = arxiv_m.group(1)
            source = "arxiv"
        else:
            import hashlib
            paper_key = hashlib.md5(url.encode()).hexdigest()[:10]
            source = "url"

        # 检查是否已存在
        for pid, entry in self.registry.items():
            key_match = (entry.get("arxiv_id") == paper_key or
                         entry.get("url") == url)
            if key_match and entry.get("status") != "removed":
                print(f"论文已存在: {pid}")
                return pid

        paper_id = _next_paper_id(self.registry)
        today = date.today().isoformat()

        # 下载 PDF
        pdf_path = self._download_pdf(paper_key, url if source == "url" else None)

        # 创建 worktree
        worktree_path = self._create_worktree(paper_id, paper_key)

        self.registry[paper_id] = {
            "arxiv_id": paper_key,
            "title": "",
            "url": url,
            "source": source,
            "status": "pending",
            "phase": "download",
            "created_date": today,
            "updated_date": today,
            "branch": f"paper/{paper_key}",
            "worktree_path": str(worktree_path),
            "pdf_path": str(pdf_path.relative_to(PROJECT_DIR)),
            "progress_summary": "PDF 已下载, worktree 已创建",
            "phases_completed": ["download"],
            "notes": notes,
        }
        self._save()
        print(f"[paper] {paper_id} 已添加: {paper_key} (source={source})")
        return paper_id

    def run_phase(self, paper_id: str, phase: str) -> str:
        """执行指定阶段"""
        if paper_id not in self.registry:
            return f"未找到 {paper_id}"

        entry = self.registry[paper_id]
        entry["updated_date"] = date.today().isoformat()

        if phase in STATUS_MAP:
            entry["status"] = STATUS_MAP[phase]

        try:
            if phase == "read":
                result = self._read_paper(paper_id)
            elif phase == "plan":
                result = self._plan_replication(paper_id)
            elif phase == "replicate":
                result = self._replicate(paper_id)
            elif phase == "evaluate":
                result = self._evaluate(paper_id)
            else:
                return f"未知阶段: {phase}"

            if phase not in entry["phases_completed"]:
                entry["phases_completed"].append(phase)

            # 更新 progress_summary (取最后几行)
            if result:
                lines = result.strip().split('\n')
                entry["progress_summary"] = lines[-1][:200] if lines else ""

            self._save()
            return result

        except Exception as e:
            entry["status"] = "failed"
            entry["progress_summary"] = f"Phase {phase} 失败: {e}"
            self._save()
            return f"Phase {phase} 失败: {e}"

    def run_all(self, paper_id: str) -> str:
        """完整流程: read → plan → replicate → evaluate"""
        results = []
        failed = False
        for phase in ["read", "plan", "replicate", "evaluate"]:
            print(f"\n[paper] {paper_id} 开始 Phase: {phase}")
            result = self.run_phase(paper_id, phase)
            results.append(f"=== {phase} ===\n{result}")

            entry = self.registry.get(paper_id, {})
            if entry.get("status") == "failed":
                failed = True
                break

        if not failed:
            self.registry[paper_id]["status"] = "completed"
            self._save()
        return "\n\n".join(results)

    def chat(self, paper_id: str, message: str) -> str:
        """与论文上下文交互对话，支持 session 续接。

        首次对话: 加载 PROGRESS.md 作为上下文，创建新 session
        后续对话: --resume 上次 chat session，保持完整对话历史
        """
        if paper_id not in self.registry:
            return f"未找到 {paper_id}"

        entry = self.registry[paper_id]
        worktree = Path(entry["worktree_path"])
        chat_session = entry.get("sessions", {}).get("chat", "")

        if chat_session:
            # 续接上次 chat session
            result = self._run_claude(
                message, cwd=worktree, phase="chat",
                timeout=300, resume_session=chat_session)
        else:
            # 首次对话: 注入 PROGRESS.md 建立上下文
            progress_file = worktree / "PROGRESS.md"
            progress = progress_file.read_text(encoding='utf-8') if progress_file.exists() else "(尚未开始)"
            title = entry.get("title") or entry.get("arxiv_id", "?")

            prompt = (
                f"你是论文「{title}」(arXiv: {entry['arxiv_id']}) 的研究助手。\n"
                f"以下是当前的复现进度文档:\n\n"
                f"<progress>\n{progress}\n</progress>\n\n"
                f"用户的问题: {message}"
            )
            result = self._run_claude(
                prompt, cwd=worktree, phase="chat", timeout=300)

        return result

    def get_status(self, paper_id: str = None) -> str:
        """查询状态 (单篇或全部)"""
        if not self.registry:
            return "暂无论文调研记录"

        if paper_id:
            entry = self.registry.get(paper_id)
            if not entry:
                return f"未找到 {paper_id}"
            return self._format_entry(paper_id, entry, detail=True)

        # 全部论文
        lines = ["论文调研状态", ""]
        active = {k: v for k, v in self.registry.items()
                  if v.get("status") not in ("removed",)}
        if not active:
            return "暂无论文调研记录"

        for pid in sorted(active.keys()):
            lines.append(self._format_entry(pid, active[pid], detail=False))
        return "\n".join(lines)

    def remove_paper(self, paper_id: str):
        """清理: 删除 worktree + registry 记录"""
        entry = self.registry.get(paper_id)
        if not entry:
            print(f"未找到 {paper_id}")
            return

        # 删除 worktree
        worktree = Path(entry.get("worktree_path", ""))
        branch = entry.get("branch", "")
        if worktree.exists():
            subprocess.run(
                ["git", "worktree", "remove", str(worktree), "--force"],
                cwd=str(REPO_ROOT), capture_output=True)

        # 删除分支
        if branch:
            subprocess.run(
                ["git", "branch", "-D", branch],
                cwd=str(REPO_ROOT), capture_output=True)

        # 删除 PDF
        pdf_path = PROJECT_DIR / entry.get("pdf_path", "")
        if pdf_path.exists():
            pdf_path.unlink()

        entry["status"] = "removed"
        self._save()
        print(f"[paper] {paper_id} 已清理")

    # ── 内部方法 ──────────────────────────────────────────

    def _download_pdf(self, paper_key: str, direct_url: str = None) -> Path:
        """下载 PDF (arXiv 或任意 URL)"""
        _ensure_dirs()
        pdf_path = PDFS_DIR / f"{paper_key}.pdf"
        if pdf_path.exists():
            print(f"[paper] PDF 已存在: {pdf_path}")
            return pdf_path

        url = direct_url or f"https://arxiv.org/pdf/{paper_key}.pdf"
        print(f"[paper] 下载 PDF: {url}")
        subprocess.run(
            ["curl", "-sfL", "-o", str(pdf_path), url],
            check=True, timeout=120)

        if not pdf_path.exists() or pdf_path.stat().st_size < 1000:
            raise RuntimeError(f"PDF 下载失败或文件过小: {pdf_path}")

        print(f"[paper] PDF 已下载: {pdf_path} ({pdf_path.stat().st_size / 1024:.0f}KB)")
        return pdf_path

    def _create_worktree(self, paper_id: str, arxiv_id: str) -> Path:
        """创建 git worktree"""
        WORKTREE_ROOT.mkdir(parents=True, exist_ok=True)
        worktree_path = WORKTREE_ROOT / paper_id
        branch_name = f"paper/{arxiv_id}"

        if worktree_path.exists():
            print(f"[paper] worktree 已存在: {worktree_path}")
            return worktree_path

        print(f"[paper] 创建 worktree: {worktree_path} (branch: {branch_name})")
        subprocess.run(
            ["git", "worktree", "add", str(worktree_path), "-b", branch_name],
            cwd=str(REPO_ROOT), check=True, capture_output=True, text=True)

        return worktree_path

    def _run_claude(self, prompt: str, cwd: Path, phase: str = "",
                    timeout: int = 300, resume_session: str = "") -> str:
        """调用 Claude CLI，注入主项目记忆，保存完整对话日志。

        - 主项目 CLAUDE.md 由 worktree 中的 git 副本提供 (CLI 自动加载)
        - 主项目 MEMORY.md 通过 prompt 注入 (auto-memory 按路径隔离，worktree 路径不同)
        - --output-format stream-json 捕获完整对话 (tool_use, tool_result, 文本)
        - 日志: {cwd}/logs/{timestamp}_{phase}.jsonl (原始) + .md (可读)
        - session_id 记录到 registry，支持后续 --resume 继续对话
        """
        # 注入主项目 MEMORY.md
        full_prompt = self._build_prompt_with_memory(prompt)

        # 准备日志目录
        log_dir = cwd / "logs"
        log_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_prefix = f"{timestamp}_{phase}" if phase else timestamp

        # 构建命令
        cmd = [
            '/usr/local/bin/claude', '--print',
            '--dangerously-skip-permissions',
            '--output-format', 'stream-json',
            '--verbose',
        ]
        if resume_session:
            cmd.extend(['--resume', resume_session])
        cmd.extend(['-p', full_prompt])

        print(f"[paper] 调用 Claude (cwd={cwd}, phase={phase}, timeout={timeout}s)")
        result = subprocess.run(
            cmd, cwd=str(cwd), env=_CLEAN_ENV,
            capture_output=True, text=True, timeout=timeout)

        raw_output = result.stdout or ""

        # 保存原始 JSONL 日志 (即使失败也保存)
        jsonl_file = log_dir / f"{log_prefix}.jsonl"
        jsonl_file.write_text(raw_output, encoding='utf-8')

        if result.returncode != 0:
            stderr = result.stderr[:500] if result.stderr else ""
            raise RuntimeError(f"Claude CLI 失败 (rc={result.returncode}): {stderr}")

        # 解析 stream-json
        final_text, session_id, readable = self._parse_stream_json(raw_output)

        # 保存可读 Markdown 日志
        md_file = log_dir / f"{log_prefix}.md"
        md_content = (
            f"# Claude CLI Log: {phase}\n\n"
            f"- **Time**: {timestamp}\n"
            f"- **CWD**: {cwd}\n"
            f"- **Session**: {session_id}\n"
            f"- **JSONL**: {jsonl_file.name}\n\n"
            f"## Prompt\n\n{prompt}\n\n"
            f"## Conversation\n\n{readable}\n"
        )
        md_file.write_text(md_content, encoding='utf-8')

        # 记录 session_id 到 registry (支持 --resume 继续对话)
        self._record_session(cwd, phase, session_id)

        print(f"[paper] 日志已保存: {jsonl_file.name}, {md_file.name} (session: {session_id[:12]})")
        return final_text

    def _record_session(self, cwd: Path, phase: str, session_id: str):
        """记录 session_id 到对应 paper 的 registry"""
        if not session_id:
            return
        for entry in self.registry.values():
            if entry.get("worktree_path") == str(cwd):
                sessions = entry.setdefault("sessions", {})
                sessions[phase] = session_id
                self._save()
                break

    def _build_prompt_with_memory(self, prompt: str) -> str:
        """注入主项目的 MEMORY.md (CLAUDE.md 由 worktree git 副本自动加载)"""
        memory_md = MAIN_MEMORY_DIR / "MEMORY.md"
        if not memory_md.exists():
            return prompt

        memory_content = memory_md.read_text(encoding='utf-8')
        return (
            f"<main-project-memory>\n"
            f"以下是主项目的关键经验和记忆，供你参考：\n\n"
            f"{memory_content}\n"
            f"</main-project-memory>\n\n"
            f"{prompt}"
        )

    @staticmethod
    def _parse_stream_json(raw: str) -> tuple:
        """解析 stream-json，返回 (final_text, session_id, readable_markdown)"""
        final_text = ""
        session_id = ""
        readable_parts = []

        for line in raw.strip().split('\n'):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            evt_type = event.get("type", "")

            if evt_type == "system":
                session_id = event.get("session_id", "")
                model = event.get("model", "?")
                readable_parts.append(f"> Session: `{session_id}` | Model: `{model}`\n")

            elif evt_type == "assistant":
                msg = event.get("message", {})
                for block in msg.get("content", []):
                    bt = block.get("type")
                    if bt == "text":
                        readable_parts.append(f"### Assistant\n\n{block['text']}\n")
                    elif bt == "tool_use":
                        name = block.get("name", "?")
                        inp = json.dumps(block.get("input", {}),
                                         ensure_ascii=False, indent=2)
                        # 截断过长的输入
                        if len(inp) > 2000:
                            inp = inp[:2000] + "\n... (truncated)"
                        readable_parts.append(
                            f"### Tool Use: `{name}`\n\n"
                            f"```json\n{inp}\n```\n"
                        )

            elif evt_type == "user":
                msg = event.get("message", {})
                for block in msg.get("content", []):
                    if block.get("type") == "tool_result":
                        content = str(block.get("content", ""))
                        if len(content) > 3000:
                            content = content[:3000] + "\n... (truncated)"
                        readable_parts.append(
                            f"### Tool Result\n\n```\n{content}\n```\n"
                        )

            elif evt_type == "result":
                final_text = event.get("result", "")
                session_id = event.get("session_id", session_id)
                cost = event.get("total_cost_usd", 0)
                turns = event.get("num_turns", 0)
                duration = event.get("duration_ms", 0)
                readable_parts.append(
                    f"---\n\n"
                    f"**Result**: {turns} turns, "
                    f"{duration/1000:.1f}s, ${cost:.4f}\n"
                )

        # fallback: stream-json 解析失败
        if not final_text and raw.strip():
            final_text = raw.strip()
            readable_parts = [f"### Raw Output\n\n{raw}\n"]

        return final_text, session_id, "\n".join(readable_parts)

    def _read_paper(self, paper_id: str) -> str:
        """Phase read: 用 Claude 读论文 PDF, 输出结构化摘要"""
        entry = self.registry[paper_id]
        pdf_path = PROJECT_DIR / entry["pdf_path"]
        worktree = Path(entry["worktree_path"])

        prompt = f"""请认真阅读论文 PDF 文件: {pdf_path}

然后完成以下任务:
1. 提取论文标题
2. 提炼核心贡献 (3-5 点)
3. 关键算法/公式描述
4. 与我们的量化交易框架的关联 (参考 trading_framework/ 下的代码结构、已有的因子挖掘/rolling训练/影子验证体系)
5. 复现难度评估 (简单/中等/困难)
6. **复现价值评估** — 这是最重要的部分，请从以下维度给出判断:
   - 对我们框架的增量价值 (能否提升 Sharpe/降低 MDD/增加因子多样性?)
   - 方法新颖性 (是否有已有工具无法实现的独特方法?)
   - 实用性 (方法是否可直接集成到我们的 rolling 训练 pipeline?)
   - 最终给出: 推荐复现 / 建议部分借鉴 / 不建议复现，并说明理由

请将结构化摘要写入文件 {worktree}/PROGRESS.md，格式如下:

# 论文复现: [论文标题]
- arXiv: {entry['arxiv_id']}
- 状态: reading
- 创建: {entry['created_date']}

## 论文摘要
[你的结构化摘要]

## 核心贡献
1. ...

## 关键算法
...

## 与框架关联
...

## 复现难度
...

## 复现价值评估
- 增量价值: ...
- 方法新颖性: ...
- 实用性: ...
- **结论**: 推荐复现 / 建议部分借鉴 / 不建议复现
- **理由**: ...

同时，请输出论文标题作为最后一行，格式: TITLE: xxx
"""
        result = self._run_claude(prompt, cwd=worktree, phase="read", timeout=600)

        # 尝试从输出中提取标题
        for line in result.split('\n'):
            if line.startswith('TITLE:'):
                title = line[6:].strip()
                entry["title"] = title
                break

        return result

    def _plan_replication(self, paper_id: str) -> str:
        """Phase plan: 生成复现计划"""
        entry = self.registry[paper_id]
        worktree = Path(entry["worktree_path"])
        progress_file = worktree / "PROGRESS.md"

        progress = progress_file.read_text(encoding='utf-8') if progress_file.exists() else ""

        prompt = f"""你正在为论文复现制定计划。

当前进度文件 PROGRESS.md 内容:
{progress}

请基于论文摘要，制定详细的复现计划:
1. 分析需要实现哪些核心组件
2. 确定依赖和前置条件
3. 制定 5-10 步的复现步骤
4. 每步标注预计工作内容和验证方法

请在 PROGRESS.md 中追加「## 复现计划」部分，格式:

## 复现计划
- [ ] Step 1: [描述]
  - 文件: [将创建/修改的文件]
  - 验证: [如何验证完成]
- [ ] Step 2: ...
...

注意: 代码应在 trading_framework/ 目录下实现，复用已有的 qlib 基础设施。
"""
        return self._run_claude(prompt, cwd=worktree, phase="plan", timeout=600)

    def _replicate(self, paper_id: str) -> str:
        """Phase replicate: 在 worktree 中逐步复现"""
        entry = self.registry[paper_id]
        worktree = Path(entry["worktree_path"])
        progress_file = worktree / "PROGRESS.md"

        progress = progress_file.read_text(encoding='utf-8') if progress_file.exists() else ""

        prompt = f"""你在分支 {entry['branch']} 的 worktree 中工作。
工作目录: {worktree}

当前 PROGRESS.md:
{progress}

请继续复现工作:
1. 阅读 PROGRESS.md 中的复现计划
2. 依次完成所有未勾选的步骤 ([ ])
3. 每完成一步，将 PROGRESS.md 中对应条目标记为 [x] 并记录实现的文件和结果
4. 确保代码可运行 (python -c "import ...")
5. 完成所有步骤后，总结复现成果

注意:
- 复用 trading_framework/ 已有的 qlib 基础设施
- 不要修改主分支上的现有文件，只在 worktree 中新增文件
- 代码写在 trading_framework/ 目录下合适的位置
"""
        return self._run_claude(prompt, cwd=worktree, phase="replicate", timeout=1800)

    def _evaluate(self, paper_id: str) -> str:
        """Phase evaluate: 评估复现结果"""
        entry = self.registry[paper_id]
        worktree = Path(entry["worktree_path"])
        progress_file = worktree / "PROGRESS.md"

        progress = progress_file.read_text(encoding='utf-8') if progress_file.exists() else ""

        prompt = f"""请评估论文复现结果。

PROGRESS.md:
{progress}

请完成:
1. 对比复现结果与论文报告的指标
2. 分析差异原因
3. 给出改进建议
4. 评估是否可以集成到主框架

在 PROGRESS.md 中追加「## 评估结果」部分。
"""
        return self._run_claude(prompt, cwd=worktree, phase="evaluate", timeout=600)

    # ── 格式化 ────────────────────────────────────────────

    def get_read_summary(self, paper_id: str) -> str:
        """从 PROGRESS.md 提取论文分析摘要 (用于 bot 展示给用户)

        提取关键内容: 论文摘要、核心贡献、与框架关联、复现价值评估，
        跳过内部进度信息 (状态、worktree路径等)。
        """
        entry = self.registry.get(paper_id)
        if not entry:
            return f"未找到 {paper_id}"

        worktree = Path(entry.get("worktree_path", ""))
        progress_file = worktree / "PROGRESS.md"
        if not progress_file.exists():
            return "论文尚未阅读完成"

        content = progress_file.read_text(encoding='utf-8')

        # 按 ## 标题解析各节
        sections = self._parse_sections(content)

        title = entry.get("title") or entry.get("arxiv_id", "?")
        arxiv_id = entry.get("arxiv_id", "?")

        parts = [f"**{title}**\narXiv: {arxiv_id}\n"]

        # 按用户关心的顺序展示
        for key in ["论文摘要", "核心贡献", "关键算法",
                     "与框架关联", "复现难度", "复现价值评估"]:
            if key in sections:
                parts.append(f"**{key}**\n{sections[key]}")

        return "\n---\n".join(parts)

    @staticmethod
    def _parse_sections(markdown: str) -> dict:
        """将 Markdown 按 ## 标题解析为 {标题: 内容} 字典

        支持嵌套的 ### 小节 — 它们会归入父 ## 的内容中。
        """
        sections = {}
        current_key = None
        current_lines = []

        for line in markdown.split('\n'):
            # ## 标题 (包括 ## **加粗标题**)
            m = re.match(r'^##\s+\*{0,2}(.+?)\*{0,2}\s*$', line)
            if m:
                # 保存上一节
                if current_key:
                    sections[current_key] = '\n'.join(current_lines).strip()
                current_key = m.group(1).strip().rstrip(':')
                current_lines = []
            elif current_key is not None:
                current_lines.append(line)

        # 保存最后一节
        if current_key:
            sections[current_key] = '\n'.join(current_lines).strip()

        return sections

    def _format_entry(self, paper_id: str, entry: dict, detail: bool = False) -> str:
        """格式化单条记录"""
        status_icon = {
            "pending": "⏳", "reading": "📖", "planning": "📋",
            "replicating": "🔧", "evaluating": "📊",
            "completed": "✅", "failed": "❌", "removed": "🗑",
        }
        icon = status_icon.get(entry.get("status", ""), "❓")

        if not detail:
            title = entry.get("title") or entry.get("arxiv_id", "?")
            return (f"{icon} **{paper_id}** | {title} | "
                    f"{entry.get('status', '?')} | "
                    f"{entry.get('progress_summary', '')}")

        lines = [
            f"{icon} {paper_id}: {entry.get('title') or entry.get('arxiv_id')}",
            f"  arXiv: {entry.get('arxiv_id')}",
            f"  URL: {entry.get('url')}",
            f"  状态: {entry.get('status')}",
            f"  阶段: {', '.join(entry.get('phases_completed', []))}",
            f"  分支: {entry.get('branch')}",
            f"  Worktree: {entry.get('worktree_path')}",
            f"  日志: {entry.get('worktree_path', '')}/logs/",
            f"  进度: {entry.get('progress_summary', '')}",
            f"  创建: {entry.get('created_date')} | 更新: {entry.get('updated_date')}",
        ]
        if entry.get("sessions"):
            sessions = entry["sessions"]
            lines.append(f"  Sessions: {', '.join(f'{k}={v[:12]}' for k, v in sessions.items())}")
        if entry.get("notes"):
            lines.append(f"  备注: {entry['notes']}")
        return "\n".join(lines)


# ── CLI ────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="论文调研与复现管理器")
    parser.add_argument("--status", nargs="?", const="ALL", default=None,
                        metavar="PAPER_ID", help="查看状态 (不带参数=全部)")
    parser.add_argument("--add", metavar="URL", help="添加论文 (arXiv URL)")
    parser.add_argument("--run", nargs="+", metavar=("PAPER_ID", "PHASE"),
                        help="运行阶段: paper_id [read|plan|replicate|evaluate]")
    parser.add_argument("--remove", metavar="PAPER_ID", help="删除论文")
    parser.add_argument("--logs", metavar="PAPER_ID", help="查看论文对话日志列表")
    parser.add_argument("--notes", default="", help="备注 (配合 --add 使用)")

    args = parser.parse_args()
    pr = PaperResearcher()

    if args.add:
        paper_id = pr.add_paper(args.add, notes=args.notes)
        print(f"Added: {paper_id}")
    elif args.run:
        paper_id = args.run[0]
        phase = args.run[1] if len(args.run) > 1 else None
        if phase:
            result = pr.run_phase(paper_id, phase)
            print(result)
        else:
            result = pr.run_all(paper_id)
            print(result)
    elif args.remove:
        pr.remove_paper(args.remove)
    elif args.logs:
        entry = pr.registry.get(args.logs)
        if not entry:
            print(f"未找到 {args.logs}")
        else:
            log_dir = Path(entry["worktree_path"]) / "logs"
            if not log_dir.exists():
                print("暂无日志")
            else:
                print(f"日志目录: {log_dir}\n")
                for f in sorted(log_dir.glob("*.md")):
                    size = f.stat().st_size / 1024
                    print(f"  {f.name}  ({size:.1f}KB)")
                print(f"\n查看日志: cat {log_dir}/<filename>.md")
                print(f"原始数据: cat {log_dir}/<filename>.jsonl")
    elif args.status is not None:
        if args.status == "ALL":
            print(pr.get_status())
        else:
            print(pr.get_status(args.status))
    else:
        print(pr.get_status())


if __name__ == "__main__":
    main()
