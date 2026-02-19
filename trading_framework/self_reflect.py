#!/usr/bin/env python3
"""
每日自我反思与迭代优化
- 收集当天所有输入输出（聊天记录、操作日志、产出）
- 调用 Claude 进行自我反思
- 根据反思结果自动优化代码
- 记录反思日志
- 通过 daemon 重启 smart_bot 使改动生效
"""

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, date, timedelta
from pathlib import Path

BOT_DIR = Path(__file__).parent
WORK_DIR = BOT_DIR.parent

# 清除 Claude Code 注入的环境变量，避免嵌套会话错误
_CLEAN_ENV = {k: v for k, v in os.environ.items() if k not in ('CLAUDECODE', 'CLAUDE_CODE_ENTRYPOINT')}
MEMORY_DIR = BOT_DIR / "memory"
LOGS_DIR = BOT_DIR / "logs"
REFLECT_DIR = BOT_DIR / "reflections"
REFLECT_DIR.mkdir(exist_ok=True)

# 需要审查的源代码文件
SOURCE_FILES = [
    BOT_DIR / "smart_bot.py",
    BOT_DIR / "daemon.py",
    BOT_DIR / "daily_runner.py",
    BOT_DIR / "monitor" / "intraday_monitor.py",
]


def collect_today_data() -> dict:
    """收集当天的所有输入输出数据"""
    today = date.today().isoformat()
    data = {
        "date": today,
        "conversations": [],
        "bot_log_snippets": [],
        "daily_runner_log": "",
        "monitor_log": "",
        "errors": [],
        "daemon_log": "",
    }

    # 1. 收集所有用户的对话记录
    if MEMORY_DIR.exists():
        for user_dir in MEMORY_DIR.iterdir():
            if not user_dir.is_dir():
                continue
            short_term = user_dir / "short_term.json"
            if short_term.exists():
                try:
                    with open(short_term, 'r', encoding='utf-8') as f:
                        messages = json.load(f)
                    # 过滤今天的消息
                    today_msgs = []
                    for msg in messages:
                        ts = msg.get("timestamp", 0)
                        msg_date = datetime.fromtimestamp(ts).date().isoformat() if ts else ""
                        if msg_date == today:
                            today_msgs.append({
                                "role": msg.get("role", ""),
                                "content": msg.get("content", "")[:500],
                                "time": datetime.fromtimestamp(ts).strftime("%H:%M:%S") if ts else "",
                            })
                    if today_msgs:
                        data["conversations"].append({
                            "user_id": user_dir.name[:12] + "...",
                            "message_count": len(today_msgs),
                            "messages": today_msgs,
                        })
                except Exception as e:
                    data["errors"].append(f"读取 {user_dir.name} 对话失败: {e}")

    # 2. 收集 smart_bot.log 中今天的内容
    bot_log = BOT_DIR / "logs" / "smart_bot.log"
    if bot_log.exists():
        try:
            today_prefix = today.replace("-", "-")  # 2026-02-11
            lines = []
            with open(bot_log, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    if today_prefix in line:
                        lines.append(line.rstrip())
            # 提取错误和关键事件
            errors = [l for l in lines if 'ERROR' in l or 'Exception' in l or '❌' in l]
            events = [l for l in lines if any(kw in l for kw in ['启动', '重启', '超时', '后台任务', 'Claude', '连接'])]
            data["bot_log_snippets"] = (errors + events)[-50:]  # 最多50条
            data["errors"].extend(errors[:20])
        except Exception as e:
            data["errors"].append(f"读取 bot log 失败: {e}")

    # 3. 收集 daily_runner 日志
    dr_log = LOGS_DIR / "daily_runner.log"
    if dr_log.exists():
        try:
            with open(dr_log, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            # 只取今天的部分
            if today in content:
                idx = content.index(today)
                data["daily_runner_log"] = content[idx:][:2000]
        except Exception as e:
            data["errors"].append(f"读取 daily_runner log 失败: {e}")

    # 4. 收集盘中监控日志
    monitor_log = LOGS_DIR / "intraday_monitor.log"
    if monitor_log.exists():
        try:
            with open(monitor_log, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            if today in content:
                idx = content.index(today)
                data["monitor_log"] = content[idx:][:2000]
        except Exception as e:
            data["errors"].append(f"读取 monitor log 失败: {e}")

    # 5. 收集实验盘日志
    exp_log = BOT_DIR / "experiment" / "daily_log" / f"{today}.json"
    if exp_log.exists():
        try:
            data["experiment_log"] = json.loads(
                exp_log.read_text(encoding='utf-8'))
        except Exception as e:
            data["errors"].append(f"读取 experiment log 失败: {e}")

    # 5b. 收集宏观分析缓存
    macro_analysis = BOT_DIR / "experiment" / "news_cache" / f"{today}_macro_analysis.json"
    if macro_analysis.exists():
        try:
            data["macro_analysis"] = json.loads(
                macro_analysis.read_text(encoding='utf-8'))
        except Exception as e:
            data["errors"].append(f"读取 macro analysis 失败: {e}")

    # 6. 收集 daemon 日志
    daemon_log = BOT_DIR / "logs" / "daemon.log"
    if daemon_log.exists():
        try:
            with open(daemon_log, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            if today in content:
                idx = content.index(today)
                data["daemon_log"] = content[idx:][:1000]
        except Exception as e:
            data["errors"].append(f"读取 daemon log 失败: {e}")

    return data


def _build_reflection_prompt(today_data: dict, compact: bool = False) -> str:
    """构建反思 prompt，compact 模式下缩短内容以降低耗时"""
    if compact:
        conv_limit, log_limit, snippet_limit = 2000, 500, 15
    else:
        conv_limit, log_limit, snippet_limit = 5000, 1000, 30

    return f"""你是一个智能系统的自我反思模块。请根据今天的运行数据进行深度反思和优化建议。

## 今日运行数据

### 日期: {today_data['date']}

### 对话统计
{json.dumps(today_data['conversations'], ensure_ascii=False, indent=2)[:conv_limit]}

### 机器人日志摘要
{chr(10).join(today_data['bot_log_snippets'][:snippet_limit])}

### Daily Runner 日志
{today_data['daily_runner_log'][:log_limit]}

### 盘中监控日志
{today_data['monitor_log'][:log_limit]}

### Daemon 日志
{today_data['daemon_log'][:500]}

### 实验盘日志
{json.dumps(today_data.get('experiment_log', {}), ensure_ascii=False, indent=2)[:log_limit]}

### 宏观分析
{json.dumps(today_data.get('macro_analysis', {}), ensure_ascii=False, indent=2)[:500]}

### 错误汇总
{chr(10).join(today_data['errors'][:20])}

## 反思要求

请从以下维度进行反思，输出 JSON 格式：

1. **用户体验分析**: 用户的需求是否被很好地满足？有没有失败或超时的请求？
2. **系统稳定性**: WebSocket 断连次数？错误率？重启次数？
3. **性能分析**: Claude 调用超时情况？响应速度？
4. **代码改进建议**: 具体可以改进的代码（给出文件路径和修改建议）
5. **新功能想法**: 基于用户的使用模式，有什么新功能可以开发？
6. **自我评分**: 今天整体表现打分 (1-10)

请严格输出以下 JSON 格式（不要输出其他内容）：
```json
{{
    "date": "{today_data['date']}",
    "score": 8,
    "summary": "今日整体表现概述（1-2句话）",
    "user_experience": {{
        "total_conversations": 0,
        "failed_requests": 0,
        "highlights": ["亮点1"],
        "issues": ["问题1"]
    }},
    "stability": {{
        "ws_disconnects": 0,
        "errors_count": 0,
        "restarts": 0,
        "issues": ["稳定性问题"]
    }},
    "performance": {{
        "avg_response_quality": "good/fair/poor",
        "timeout_count": 0,
        "issues": ["性能问题"]
    }},
    "code_improvements": [
        {{
            "file": "smart_bot.py",
            "description": "改进描述",
            "priority": "high/medium/low",
            "auto_fixable": true
        }}
    ],
    "new_feature_ideas": [
        {{
            "title": "功能名",
            "description": "功能描述",
            "effort": "small/medium/large"
        }}
    ],
    "action_items": [
        "明天要做的具体事项1"
    ]
}}
```"""


def _call_claude_reflection(prompt: str, timeout: int) -> dict:
    """单次调用 Claude 反思，返回解析后的 dict 或包含 error 的 dict"""
    cmd = [
        '/usr/local/bin/claude',
        '--print',
        '--dangerously-skip-permissions',
        '--output-format', 'text',
        '-p', prompt
    ]
    result = subprocess.run(
        cmd,
        capture_output=True, text=True,
        timeout=timeout,
        cwd=str(WORK_DIR),
        env=_CLEAN_ENV
    )
    output = result.stdout.strip()

    json_match = re.search(r'\{[\s\S]*\}', output)
    if json_match:
        return json.loads(json_match.group())
    else:
        return {
            "score": 0,
            "summary": "反思输出解析失败",
            "raw_output": output[:2000],
            "error": "无法提取 JSON"
        }


def run_reflection(today_data: dict) -> dict:
    """调用 Claude 进行自我反思，超时后用精简内容重试一次"""
    max_attempts = 2
    timeouts = [10800, 3600]  # 第一次 3 小时，重试 1 小时

    for attempt in range(max_attempts):
        compact = (attempt > 0)
        prompt = _build_reflection_prompt(today_data, compact=compact)
        timeout = timeouts[attempt]

        try:
            if attempt > 0:
                print(f"  重试反思 (精简模式, 超时 {timeout // 60} 分钟)...")
            reflection = _call_claude_reflection(prompt, timeout)
            reflection["date"] = today_data["date"]
            if "error" not in reflection or reflection.get("score", 0) > 0:
                return reflection
            # 解析失败但没超时，直接返回
            return reflection
        except subprocess.TimeoutExpired:
            print(f"  反思超时 (第 {attempt + 1} 次, {timeout // 60} 分钟)")
            if attempt < max_attempts - 1:
                continue
            return {
                "date": today_data['date'],
                "score": 0,
                "summary": "反思超时 (已重试)",
                "error": f"Claude 调用超时 ({max_attempts} 次)"
            }
        except Exception as e:
            return {
                "date": today_data['date'],
                "score": 0,
                "summary": f"反思失败: {e}",
                "error": str(e)
            }

    return {
        "date": today_data['date'],
        "score": 0,
        "summary": "反思失败",
        "error": "未知错误"
    }


def apply_auto_fixes(reflection: dict) -> list:
    """根据反思结果，对 auto_fixable 的改进项调用 Claude 自动修复"""
    applied = []
    improvements = reflection.get("code_improvements", [])
    auto_fixes = [imp for imp in improvements if imp.get("auto_fixable") and imp.get("priority") in ("high", "medium")]

    if not auto_fixes:
        return applied

    # 构建修复 prompt
    fix_descriptions = "\n".join(
        f"- [{imp['priority']}] {imp['file']}: {imp['description']}"
        for imp in auto_fixes
    )

    prompt = f"""你是一个代码优化助手。请根据以下反思结果，对代码进行安全的小规模改进。

## 需要修复的项目
{fix_descriptions}

## 重要规则
1. 只做安全的、不破坏现有功能的修改
2. 不要修改核心业务逻辑
3. 主要关注：错误处理、日志改进、性能优化、代码简化
4. 每个修改都要有 git commit
5. 工作目录: {WORK_DIR}
6. smart_bot.py 位于 {BOT_DIR}/smart_bot.py

请直接执行修改，不需要确认。修改完成后输出一个摘要，说明做了哪些改动。"""

    try:
        cmd = [
            '/usr/local/bin/claude',
            '--print',
            '--dangerously-skip-permissions',
            '-p', prompt
        ]
        result = subprocess.run(
            cmd,
            capture_output=True, text=True,
            timeout=1800,  # 30 分钟超时
            cwd=str(WORK_DIR),
            env=_CLEAN_ENV
        )
        output = result.stdout.strip()
        if output:
            applied.append({
                "fixes_attempted": len(auto_fixes),
                "output_summary": output[:1000],
            })
    except Exception as e:
        applied.append({"error": str(e)})

    return applied


def restart_bot():
    """通过给 daemon 发信号来重启 smart_bot"""
    try:
        pid_file = BOT_DIR / "daemon.pid"
        if not pid_file.exists():
            return "daemon 未运行，跳过重启"
        pid = int(pid_file.read_text().strip())
        # 给 daemon 进程发 SIGHUP，daemon 收到后重启 bot
        # 如果 daemon 没有 SIGHUP handler，直接 kill bot 子进程让 daemon 自动重启
        import signal as _signal

        # 读取 bot 进程 PID (daemon 的子进程)
        result = subprocess.run(
            ['pgrep', '-P', str(pid)],
            capture_output=True, text=True, timeout=5
        )
        bot_pids = result.stdout.strip().split('\n')
        for bp in bot_pids:
            bp = bp.strip()
            if bp:
                os.kill(int(bp), _signal.SIGTERM)
        # daemon 会检测到 bot 退出并自动重启
        return f"已终止 bot 进程 {bot_pids}，daemon 将自动重启"
    except Exception as e:
        return f"重启失败: {e}"


def save_reflection(reflection: dict, auto_fix_results: list):
    """保存反思记录"""
    today = date.today().isoformat()
    reflection["auto_fixes"] = auto_fix_results
    reflection["generated_at"] = datetime.now().isoformat()

    # 保存当日反思
    reflect_file = REFLECT_DIR / f"{today}.json"
    with open(reflect_file, 'w', encoding='utf-8') as f:
        json.dump(reflection, f, ensure_ascii=False, indent=2)

    # 更新反思索引
    index_file = REFLECT_DIR / "index.json"
    index = {}
    if index_file.exists():
        try:
            with open(index_file, 'r', encoding='utf-8') as f:
                index = json.load(f)
        except:
            pass

    index[today] = {
        "score": reflection.get("score", 0),
        "summary": reflection.get("summary", ""),
        "auto_fixes_count": len(auto_fix_results),
        "action_items_count": len(reflection.get("action_items", [])),
    }

    # 只保留最近90天
    cutoff = (date.today() - timedelta(days=90)).isoformat()
    index = {k: v for k, v in index.items() if k >= cutoff}

    with open(index_file, 'w', encoding='utf-8') as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    return str(reflect_file)


def main():
    print(f"[{datetime.now().isoformat()}] 开始每日自我反思...")

    # 1. 收集数据
    print("  收集今日数据...")
    today_data = collect_today_data()
    conv_count = sum(c["message_count"] for c in today_data["conversations"])
    print(f"  今日对话: {len(today_data['conversations'])} 用户, {conv_count} 条消息")
    print(f"  错误数: {len(today_data['errors'])}")

    # 如果今天没有任何活动，跳过
    if conv_count == 0 and not today_data["errors"] and not today_data["bot_log_snippets"]:
        print("  今日无活动，跳过反思")
        # 仍然记录一条空反思
        save_reflection({
            "date": date.today().isoformat(),
            "score": 0,
            "summary": "今日无活动",
        }, [])
        return

    # 2. 调用 Claude 反思
    print("  调用 Claude 进行反思...")
    reflection = run_reflection(today_data)
    score = reflection.get("score", 0)
    print(f"  反思完成，评分: {score}/10")
    print(f"  总结: {reflection.get('summary', 'N/A')}")

    # 3. 自动修复（仅当有 auto_fixable 项时）
    auto_fix_results = []
    if reflection.get("code_improvements"):
        auto_fixable = [imp for imp in reflection["code_improvements"]
                       if imp.get("auto_fixable") and imp.get("priority") in ("high", "medium")]
        if auto_fixable:
            print(f"  发现 {len(auto_fixable)} 项可自动修复，开始修复...")
            auto_fix_results = apply_auto_fixes(reflection)
            print(f"  自动修复完成")

    # 4. 保存反思记录
    reflect_file = save_reflection(reflection, auto_fix_results)
    print(f"  反思记录已保存: {reflect_file}")

    # 5. 如果有代码修改，重启 bot
    if auto_fix_results and any(not r.get("error") for r in auto_fix_results):
        print("  检测到代码修改，重启 smart_bot...")
        restart_result = restart_bot()
        print(f"  {restart_result}")
    else:
        print("  无代码修改，跳过重启")

    print(f"[{datetime.now().isoformat()}] 每日自我反思完成")


if __name__ == "__main__":
    main()
