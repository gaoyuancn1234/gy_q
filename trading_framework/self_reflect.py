#!/usr/bin/env python3
"""
每日综合反思系统
- 收集系统运行日志 (精简)
- 收集量化策略数据 (因子池/挖掘/shadow/持仓/模型配置)
- 检索市场情报 (宏观新闻 + arXiv 论文)
- 调用 Claude 进行综合反思 → 结构化 JSON
- 保存行动计划 (Markdown) + 反思记录 (JSON)
- 推送飞书 markdown 卡片简报
"""

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, date, timedelta
from pathlib import Path
from cli_paths import CLAUDE_BIN

BOT_DIR = Path(__file__).parent
WORK_DIR = BOT_DIR.parent

# 清除 Claude Code 注入的环境变量，避免嵌套会话错误
_CLEAN_ENV = {k: v for k, v in os.environ.items() if k not in ('CLAUDECODE', 'CLAUDE_CODE_ENTRYPOINT')}
MEMORY_DIR = BOT_DIR / "memory"
LOGS_DIR = BOT_DIR / "logs"
REFLECT_DIR = BOT_DIR / "reflections"
REFLECT_DIR.mkdir(exist_ok=True)


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


def collect_strategy_data() -> dict:
    """收集量化策略数据: 因子池、挖掘进展、shadow、持仓、模型配置"""
    data = {}

    # 1. 因子池
    pool_file = BOT_DIR / "factor_lab" / "mining_results" / "global_factor_pool.json"
    if pool_file.exists():
        try:
            pool = json.loads(pool_file.read_text(encoding='utf-8'))
            factors = pool.get("factors", [])
            # top10 by abs(ICIR)
            sorted_factors = sorted(factors, key=lambda f: abs(f.get("icir", 0)), reverse=True)
            top10 = [
                {"name": f["name"], "icir": round(f.get("icir", 0), 3),
                 "rank_ic": round(f.get("rank_ic", 0), 4), "direction": f.get("direction", "")}
                for f in sorted_factors[:10]
            ]
            # 方向分布
            direction_counts = {}
            for f in factors:
                d = f.get("direction", "unknown")
                direction_counts[d] = direction_counts.get(d, 0) + 1
            # 平均 IC/ICIR
            ics = [f.get("rank_ic", 0) for f in factors if f.get("rank_ic")]
            icirs = [f.get("icir", 0) for f in factors if f.get("icir")]
            data["factor_pool"] = {
                "pool_size": pool.get("global_pool_size", len(factors)),
                "top10_by_icir": top10,
                "direction_distribution": direction_counts,
                "avg_rank_ic": round(sum(ics) / max(len(ics), 1), 4),
                "avg_icir": round(sum(icirs) / max(len(icirs), 1), 3),
                "pool_stats": pool.get("pool_stats", {}),
            }
        except Exception as e:
            data["factor_pool"] = {"error": str(e)}
    else:
        data["factor_pool"] = {"error": "global_factor_pool.json 不存在"}

    # 2. 最近3个挖掘 run
    runs_dir = BOT_DIR / "factor_lab" / "mining_results" / "runs"
    if runs_dir.exists():
        try:
            run_files = sorted(runs_dir.glob("run_*.json"), reverse=True)[:3]
            recent_runs = []
            for rf in run_files:
                run = json.loads(rf.read_text(encoding='utf-8'))
                backtest = run.get("backtest", {})
                recent_runs.append({
                    "run_id": run.get("run_id", rf.stem),
                    "date": run.get("date", ""),
                    "mode": run.get("mode", ""),
                    "factors_tested": run.get("factors_tested", 0),
                    "factors_admitted": run.get("factors_admitted_this_session", 0),
                    "global_pool_size": run.get("global_pool_size", 0),
                    "promising_count": run.get("promising_count", 0),
                    "backtest_sharpe_delta": backtest.get("sharpe_delta"),
                    "total_time_min": round(run.get("total_time", 0) / 60, 1),
                })
            data["recent_runs"] = recent_runs
        except Exception as e:
            data["recent_runs"] = [{"error": str(e)}]
    else:
        data["recent_runs"] = []

    # 3. Shadow 验证
    shadow_reg = BOT_DIR / "shadow" / "registry.json"
    if shadow_reg.exists():
        try:
            shadows = json.loads(shadow_reg.read_text(encoding='utf-8'))
            data["shadows"] = {
                sid: {
                    "status": s.get("status", ""),
                    "source": s.get("source", ""),
                    "created_date": s.get("created_date", ""),
                    "duration_days": s.get("duration_days", 0),
                    "elapsed_days": s.get("elapsed_days", 0),
                    "reason": s.get("reason", ""),
                }
                for sid, s in shadows.items()
            }
        except Exception as e:
            data["shadows"] = {"error": str(e)}
    else:
        data["shadows"] = {}

    # 4. 实盘持仓
    holdings_file = BOT_DIR / "portfolio" / "live_holdings.json"
    if holdings_file.exists():
        try:
            h = json.loads(holdings_file.read_text(encoding='utf-8'))
            positions = h.get("positions", {})
            data["live_holdings"] = {
                "initial_capital": h.get("initial_capital", 0),
                "cash": h.get("cash", 0),
                "position_count": len(positions),
                "positions": [
                    {"code": code, "name": p.get("name", ""), "shares": p.get("shares", 0),
                     "cost_price": p.get("cost_price", 0), "entry_date": p.get("entry_date", "")}
                    for code, p in positions.items()
                ],
                "last_signal_date": h.get("last_signal_date", ""),
                "last_update": h.get("last_update", ""),
                "rebalance_day_count": h.get("rebalance_day_count", 0),
            }
        except Exception as e:
            data["live_holdings"] = {"error": str(e)}
    else:
        data["live_holdings"] = {"error": "live_holdings.json 不存在"}

    # 5. 模型配置
    config_file = BOT_DIR / "config" / "signal_config.yaml"
    if config_file.exists():
        try:
            import yaml
            with open(config_file, 'r', encoding='utf-8') as f:
                cfg = yaml.safe_load(f)
            data["model_config"] = {
                "model_id": cfg.get("model_id", ""),
                "model_tag": cfg.get("model_tag", ""),
                "model": cfg.get("model", ""),
                "preset": cfg.get("preset", ""),
                "rolling_config": cfg.get("rolling_config", ""),
                "topk": cfg.get("topk", 0),
                "rebalance_every": cfg.get("rebalance_every", 0),
                "stop_loss": cfg.get("stop_loss", 0),
                "last_retrain": cfg.get("last_retrain", ""),
                "test_start": cfg.get("test_start", ""),
                "test_end": cfg.get("test_end", ""),
            }
        except Exception as e:
            data["model_config"] = {"error": str(e)}
    else:
        data["model_config"] = {"error": "signal_config.yaml 不存在"}

    return data


def collect_market_intelligence() -> dict:
    """检索市场情报: 宏观新闻 + arXiv 论文"""
    data = {"macro_news": [], "arxiv_papers": []}

    # 1. 宏观新闻 (复用 news_sentinel)
    try:
        if str(BOT_DIR) not in sys.path:
            sys.path.insert(0, str(BOT_DIR))
        from news_sentinel import NewsSentinel
        sentinel = NewsSentinel()
        macro = sentinel.fetch_macro_news()
        # 合并 CCTV + 财联社, 取最新 20 条
        news_items = []
        for n in macro.get("cctv", []):
            news_items.append({"source": "CCTV", "title": n.get("title", "")})
        for n in macro.get("cls", []):
            news_items.append({"source": "财联社", "title": n.get("title", ""),
                               "content": n.get("content", "")[:100]})
        data["macro_news"] = news_items[:20]
        print(f"  [market] 宏观新闻: {len(data['macro_news'])} 条")
    except Exception as e:
        print(f"  [market] 宏观新闻获取失败: {e}")
        data["macro_news_error"] = str(e)

    # 2. arXiv 论文 (复用 paper_researcher.search_arxiv)
    try:
        from paper_researcher import search_arxiv
        queries = [
            "quantitative trading alpha factor",
            "stock prediction machine learning",
            "portfolio optimization deep learning",
        ]
        seen_ids = set()
        all_papers = []
        for q in queries:
            try:
                papers = search_arxiv(q, max_results=3)
                for p in papers:
                    aid = p.get("arxiv_id", "")
                    if aid not in seen_ids:
                        seen_ids.add(aid)
                        all_papers.append({
                            "title": p.get("title", ""),
                            "arxiv_id": aid,
                            "summary": p.get("summary", "")[:200],
                            "date": p.get("date", ""),
                        })
                time.sleep(1)  # arXiv API 礼貌等待
            except Exception as e:
                print(f"  [market] arXiv 搜索 '{q}' 失败: {e}")
        data["arxiv_papers"] = all_papers
        print(f"  [market] arXiv 论文: {len(all_papers)} 篇 (去重)")
    except Exception as e:
        print(f"  [market] arXiv 搜索失败: {e}")
        data["arxiv_papers_error"] = str(e)

    return data


def _build_reflection_prompt(system_data: dict, strategy_data: dict,
                             market_intel: dict, compact: bool = False) -> str:
    """构建综合反思 prompt"""
    today = system_data.get("date", date.today().isoformat())

    # 系统运行概况 (精简)
    errors_text = "\n".join(system_data.get("errors", [])[:10]) or "(无错误)"
    log_snippets = "\n".join(system_data.get("bot_log_snippets", [])[:15]) or "(无日志)"
    conv_count = sum(c["message_count"] for c in system_data.get("conversations", []))

    # 量化策略数据
    pool = strategy_data.get("factor_pool", {})
    pool_text = json.dumps(pool, ensure_ascii=False, indent=2)[:2000] if not pool.get("error") else pool["error"]

    runs = strategy_data.get("recent_runs", [])
    runs_text = json.dumps(runs, ensure_ascii=False, indent=2)[:1500] if runs else "(无挖掘记录)"

    shadows = strategy_data.get("shadows", {})
    shadows_text = json.dumps(shadows, ensure_ascii=False, indent=2)[:1000] if shadows else "(无 shadow)"

    holdings = strategy_data.get("live_holdings", {})
    holdings_text = json.dumps(holdings, ensure_ascii=False, indent=2)[:1500] if not holdings.get("error") else holdings["error"]

    config = strategy_data.get("model_config", {})
    config_text = json.dumps(config, ensure_ascii=False, indent=2)[:500] if not config.get("error") else config["error"]

    # 市场情报
    macro = market_intel.get("macro_news", [])
    macro_text = "\n".join(
        f"- [{n.get('source', '')}] {n.get('title', '')}" for n in macro
    ) or "(无新闻)"

    papers = market_intel.get("arxiv_papers", [])
    papers_text = "\n".join(
        f"- [{p.get('arxiv_id', '')}] {p.get('title', '')}\n  {p.get('summary', '')}"
        for p in papers
    ) or "(无论文)"

    if compact:
        pool_text = pool_text[:1000]
        runs_text = runs_text[:800]
        macro_text = "\n".join(macro_text.split("\n")[:10])
        papers_text = "\n".join(papers_text.split("\n")[:10])

    return f"""你是一位资深量化研究员兼系统架构师。请对以下量化交易系统进行每日综合反思。

## 一、系统运行概况 (简要)

### 日期: {today}
### 对话数: {conv_count} 条
### 错误汇总
{errors_text}

### 关键日志
{log_snippets}

## 二、量化策略现状

### 当前模型配置
{config_text}

### 因子池
{pool_text}

### 近期挖掘进展 (最近3个run)
{runs_text}

### 影子验证状态
{shadows_text}

### 实盘持仓
{holdings_text}

## 三、市场环境

### 宏观新闻
{macro_text}

### 最新研究论文
{papers_text}

## 反思任务

请从以下维度进行综合反思，输出 JSON 格式：

1. **系统健康** (1-2句概述)
2. **策略绩效诊断**: 因子池质量如何？挖掘效率？模型是否需要重训？
3. **因子分析**: 哪些方向表现强？哪些方向枯竭？有衰减迹象吗？
4. **市场研判**: 宏观环境对策略的影响，是否需要调整？
5. **研究方向**: 基于论文和市场环境，有什么值得探索的新方向？
6. **行动计划**: 方向性建议 (不修改代码)，按优先级排列
7. **优先级排序**: 最值得投入精力的1-2件事

请严格输出以下 JSON 格式（不要输出其他内容）：
```json
{{
    "date": "{today}",
    "score": 8,
    "summary": "整体表现概述 (1-2句话)",
    "system_health": {{
        "status": "healthy/warning/critical",
        "issues": ["问题描述"]
    }},
    "strategy_diagnosis": {{
        "factor_pool_quality": "因子池质量评估 (2-3句)",
        "mining_efficiency": "挖掘效率分析 (2-3句)",
        "risks": ["潜在风险1", "潜在风险2"]
    }},
    "factor_analysis": {{
        "strong_directions": ["表现好的方向"],
        "weak_directions": ["枯竭的方向"],
        "decay_signals": ["衰减迹象"],
        "recommendations": ["因子相关建议"]
    }},
    "market_outlook": {{
        "macro_assessment": "宏观环境评估 (2-3句)",
        "impact_on_strategy": "对策略的影响",
        "adjustments": ["建议调整"]
    }},
    "research_directions": [
        {{
            "title": "方向名",
            "rationale": "为什么值得探索 (2-3句)",
            "related_papers": ["相关论文标题"],
            "effort": "small/medium/large"
        }}
    ],
    "action_plan": [
        {{
            "title": "计划标题",
            "description": "详细描述 (3-5句, 具体可执行)",
            "priority": "P0/P1/P2",
            "category": "factor/model/data/infra/research"
        }}
    ],
    "top_priorities": ["最值得投入精力的1-2件事"]
}}
```"""


def _call_claude_reflection(prompt: str, timeout: int) -> dict:
    """单次调用 Claude 反思，返回解析后的 dict 或包含 error 的 dict"""
    cmd = [
        CLAUDE_BIN,
        '--print',
        '--dangerously-skip-permissions',
        '--output-format', 'text',
    ]
    # prompt 走 stdin: Windows 上 CLAUDE_BIN 解析到 claude.CMD (批处理)，
    # argv 里的换行会截断命令行，多行 prompt 只有第一行送达。
    result = subprocess.run(
        cmd,
        input=prompt,
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=timeout,
        cwd=str(WORK_DIR),
        env=_CLEAN_ENV
    )
    output = result.stdout.strip()
    stderr = result.stderr.strip() if result.stderr else ""

    # 检查 CLI 是否报错
    if result.returncode != 0:
        print(f"  [reflect] Claude CLI 退出码 {result.returncode}")
        if stderr:
            print(f"  [reflect] stderr: {stderr[:500]}")
        # OAuth token 过期时尝试自动刷新并重试
        combined = (output + " " + stderr).lower()
        if "auth" in combined or "token" in combined or "expired" in combined or "403" in combined or "401" in combined:
            print("  [reflect] 检测到认证错误，尝试刷新 OAuth token...")
            try:
                # 不能用 shell 管道 f"echo '/exit' | {CLAUDE_BIN}":
                # Windows 的 cmd.exe 不剥单引号，Claude 收到的是 "'/exit'" 而非
                # "/exit"，刷新等于没做；CLAUDE_BIN 含空格时还会被拆断。
                # 直接用 stdin 传，与本项目其他 CLI 调用一致。
                subprocess.run(
                    [CLAUDE_BIN], input="/exit\n",
                    capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30,
                    env=_CLEAN_ENV
                )
                print("  [reflect] token 刷新完成，重试反思...")
                retry = subprocess.run(
                    cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                    timeout=timeout, cwd=str(WORK_DIR), env=_CLEAN_ENV
                )
                if retry.returncode == 0 and retry.stdout.strip():
                    output = retry.stdout.strip()
                    stderr = retry.stderr.strip() if retry.stderr else ""
                    print("  [reflect] 重试成功")
                    # 跳过下面的 return，继续正常解析
                else:
                    print(f"  [reflect] 重试仍失败 (exit={retry.returncode})")
                    return {
                        "score": 0,
                        "summary": f"Claude CLI 认证错误，自动刷新后仍失败 (exit={retry.returncode})",
                        "raw_output": (retry.stdout or "")[:2000],
                        "stderr": (retry.stderr or "")[:2000],
                        "error": f"returncode={retry.returncode} after token refresh"
                    }
            except Exception as e:
                print(f"  [reflect] token 刷新异常: {e}")
                return {
                    "score": 0,
                    "summary": f"Claude CLI 错误 (exit={result.returncode})",
                    "raw_output": output[:2000],
                    "stderr": stderr[:2000],
                    "error": f"returncode={result.returncode}: {stderr[:200]}"
                }
        else:
            return {
                "score": 0,
                "summary": f"Claude CLI 错误 (exit={result.returncode})",
                "raw_output": output[:2000],
                "stderr": stderr[:2000],
                "error": f"returncode={result.returncode}: {stderr[:200]}"
            }

    if not output:
        print(f"  [reflect] Claude 返回空输出")
        if stderr:
            print(f"  [reflect] stderr: {stderr[:500]}")
        return {
            "score": 0,
            "summary": "Claude 返回空输出",
            "stderr": stderr[:2000],
            "error": f"空输出, stderr={stderr[:200]}"
        }

    json_match = re.search(r'\{[\s\S]*\}', output)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError as e:
            return {
                "score": 0,
                "summary": "反思输出 JSON 解析失败",
                "raw_output": output[:2000],
                "error": f"JSON 解析失败: {e}"
            }
    else:
        return {
            "score": 0,
            "summary": "反思输出解析失败",
            "raw_output": output[:2000],
            "error": "无法提取 JSON"
        }


def run_reflection(system_data: dict, strategy_data: dict,
                   market_intel: dict) -> dict:
    """调用 Claude 进行综合反思，超时后用精简内容重试一次"""
    max_attempts = 2
    timeouts = [10800, 3600]  # 第一次 3 小时，重试 1 小时

    for attempt in range(max_attempts):
        compact = (attempt > 0)
        prompt = _build_reflection_prompt(system_data, strategy_data,
                                          market_intel, compact=compact)
        timeout = timeouts[attempt]

        try:
            if attempt > 0:
                print(f"  重试反思 (精简模式, 超时 {timeout // 60} 分钟)...")
            reflection = _call_claude_reflection(prompt, timeout)
            reflection["date"] = system_data.get("date", date.today().isoformat())
            if "error" not in reflection or reflection.get("score", 0) > 0:
                return reflection
            # 解析失败但没超时，直接返回
            return reflection
        except subprocess.TimeoutExpired:
            print(f"  反思超时 (第 {attempt + 1} 次, {timeout // 60} 分钟)")
            if attempt < max_attempts - 1:
                continue
            return {
                "date": system_data.get("date", date.today().isoformat()),
                "score": 0,
                "summary": "反思超时 (已重试)",
                "error": f"Claude 调用超时 ({max_attempts} 次)"
            }
        except Exception as e:
            return {
                "date": system_data.get("date", date.today().isoformat()),
                "score": 0,
                "summary": f"反思失败: {e}",
                "error": str(e)
            }

    return {
        "date": system_data.get("date", date.today().isoformat()),
        "score": 0,
        "summary": "反思失败",
        "error": "未知错误"
    }


def save_action_plan(reflection: dict):
    """将行动计划保存为独立 Markdown 文件"""
    today = reflection.get("date", date.today().isoformat())
    plan_file = REFLECT_DIR / f"{today}_plan.md"

    lines = [
        f"# 每日反思行动计划 - {today}",
        "",
        f"**综合评分**: {reflection.get('score', 'N/A')}/10",
        f"**概述**: {reflection.get('summary', 'N/A')}",
        "",
    ]

    # 最高优先级
    top = reflection.get("top_priorities", [])
    if top:
        lines.append("## 最高优先级")
        for i, t in enumerate(top, 1):
            lines.append(f"{i}. {t}")
        lines.append("")

    # 系统健康
    health = reflection.get("system_health", {})
    if health:
        lines.append(f"## 系统健康: {health.get('status', 'N/A')}")
        for issue in health.get("issues", []):
            lines.append(f"- {issue}")
        lines.append("")

    # 策略诊断
    diag = reflection.get("strategy_diagnosis", {})
    if diag:
        lines.append("## 策略诊断")
        if diag.get("factor_pool_quality"):
            lines.append(f"**因子池质量**: {diag['factor_pool_quality']}")
        if diag.get("mining_efficiency"):
            lines.append(f"**挖掘效率**: {diag['mining_efficiency']}")
        for r in diag.get("risks", []):
            lines.append(f"- 风险: {r}")
        lines.append("")

    # 因子分析
    fa = reflection.get("factor_analysis", {})
    if fa:
        lines.append("## 因子分析")
        if fa.get("strong_directions"):
            lines.append(f"**强势方向**: {', '.join(fa['strong_directions'])}")
        if fa.get("weak_directions"):
            lines.append(f"**弱势方向**: {', '.join(fa['weak_directions'])}")
        if fa.get("decay_signals"):
            lines.append(f"**衰减信号**: {', '.join(fa['decay_signals'])}")
        for rec in fa.get("recommendations", []):
            lines.append(f"- {rec}")
        lines.append("")

    # 市场研判
    mo = reflection.get("market_outlook", {})
    if mo:
        lines.append("## 市场研判")
        if mo.get("macro_assessment"):
            lines.append(f"**宏观评估**: {mo['macro_assessment']}")
        if mo.get("impact_on_strategy"):
            lines.append(f"**策略影响**: {mo['impact_on_strategy']}")
        for adj in mo.get("adjustments", []):
            lines.append(f"- {adj}")
        lines.append("")

    # 研究方向
    rds = reflection.get("research_directions", [])
    if rds:
        lines.append("## 研究方向")
        for rd in rds:
            lines.append(f"### {rd.get('title', '')} [{rd.get('effort', '')}]")
            lines.append(f"{rd.get('rationale', '')}")
            related = rd.get("related_papers", [])
            if related:
                lines.append(f"相关论文: {', '.join(related)}")
            lines.append("")

    # 行动计划
    plans = reflection.get("action_plan", [])
    if plans:
        lines.append("## 行动计划")
        # 按优先级分组
        for priority in ["P0", "P1", "P2"]:
            group = [p for p in plans if p.get("priority") == priority]
            if group:
                lines.append(f"### {priority}")
                for p in group:
                    cat = p.get("category", "")
                    lines.append(f"- **[{cat}] {p.get('title', '')}**")
                    lines.append(f"  {p.get('description', '')}")
                lines.append("")

    plan_file.write_text("\n".join(lines), encoding='utf-8')
    return str(plan_file)


def save_reflection(reflection: dict):
    """保存反思记录"""
    # 使用 reflection 中的日期 (数据收集时确定), 避免跨午夜日期漂移
    today = reflection.get("date", date.today().isoformat())
    reflection["generated_at"] = datetime.now().isoformat()

    # 保存当日反思
    reflect_file = REFLECT_DIR / f"{today}.json"
    with open(reflect_file, 'w', encoding='utf-8') as f:
        json.dump(reflection, f, ensure_ascii=False, indent=2)

    # 写入通知
    try:
        from notifications import write_notification
        score = reflection.get("score", 0)
        summary = reflection.get("summary", "")
        write_notification("reflection", f"反思评分: {score}/10", summary)
    except Exception:
        pass

    # 更新反思索引
    index_file = REFLECT_DIR / "index.json"
    index = {}
    if index_file.exists():
        try:
            with open(index_file, 'r', encoding='utf-8') as f:
                index = json.load(f)
        except Exception:
            pass

    index[today] = {
        "score": reflection.get("score", 0),
        "summary": reflection.get("summary", ""),
        "system_health": reflection.get("system_health", {}).get("status", ""),
        "top_priorities": reflection.get("top_priorities", []),
        "action_plan_count": len(reflection.get("action_plan", [])),
    }

    # 只保留最近90天
    cutoff = (date.today() - timedelta(days=90)).isoformat()
    index = {k: v for k, v in index.items() if k >= cutoff}

    with open(index_file, 'w', encoding='utf-8') as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    return str(reflect_file)


def push_reflection_to_feishu(reflection: dict):
    """推送反思简报到飞书 (markdown 卡片)"""
    try:
        if str(BOT_DIR) not in sys.path:
            sys.path.insert(0, str(BOT_DIR))
        from send_signal import get_client, USER_OPEN_ID
        from lark_oapi.api.im.v1 import CreateMessageRequest, CreateMessageRequestBody
    except Exception as e:
        print(f"  [feishu] 导入失败: {e}")
        return False

    if not USER_OPEN_ID:
        print("  [feishu] 未配置 USER_OPEN_ID, 跳过推送")
        return False

    # 构建精简简报 markdown
    today = reflection.get("date", "")
    score = reflection.get("score", "N/A")
    summary = reflection.get("summary", "")

    health = reflection.get("system_health", {})
    health_status = health.get("status", "N/A")
    health_issues = health.get("issues", [])

    diag = reflection.get("strategy_diagnosis", {})
    mo = reflection.get("market_outlook", {})
    top = reflection.get("top_priorities", [])
    rds = reflection.get("research_directions", [])

    md_parts = [
        f"**评分**: {score}/10",
        f"**概述**: {summary}",
        "",
        f"**系统状态**: {health_status}",
    ]
    if health_issues:
        for issue in health_issues[:3]:
            md_parts.append(f"- {issue}")

    md_parts.append("")
    if diag.get("factor_pool_quality"):
        md_parts.append(f"**因子池**: {diag['factor_pool_quality'][:200]}")
    if diag.get("risks"):
        md_parts.append(f"**风险**: {'; '.join(diag['risks'][:3])}")

    if mo.get("macro_assessment"):
        md_parts.append(f"\n**市场**: {mo['macro_assessment'][:200]}")

    if top:
        md_parts.append("\n**最高优先级**:")
        for i, t in enumerate(top, 1):
            md_parts.append(f"{i}. {t}")

    if rds:
        md_parts.append("\n**新方向**:")
        for rd in rds[:3]:
            md_parts.append(f"- {rd.get('title', '')} [{rd.get('effort', '')}]")

    md_content = "\n".join(md_parts)
    # 飞书卡片内容限制
    if len(md_content) > 4500:
        md_content = md_content[:4500] + "\n...(已截断)"

    card = {
        "config": {"wide_screen_mode": True},
        "header": {
            "title": {"tag": "plain_text", "content": f"每日反思简报 - {today}"},
            "template": "purple"
        },
        "elements": [
            {"tag": "markdown", "content": md_content},
        ]
    }

    # 3 次重试
    client = get_client()

    for attempt in range(3):
        try:
            request = CreateMessageRequest.builder() \
                .receive_id_type("open_id") \
                .request_body(CreateMessageRequestBody.builder()
                    .receive_id(USER_OPEN_ID)
                    .msg_type("interactive")
                    .content(json.dumps(card, ensure_ascii=False))
                    .build()) \
                .build()
            response = client.im.v1.message.create(request)
            if response.success():
                print(f"  [feishu] 反思简报已推送")
                return True
            else:
                print(f"  [feishu] 推送失败 (attempt {attempt+1}): {response.code} - {response.msg}")
        except Exception as e:
            print(f"  [feishu] 推送异常 (attempt {attempt+1}): {e}")
        if attempt < 2:
            time.sleep(2)

    print("  [feishu] 反思简报推送失败 (3次重试)")
    return False


def main():
    print(f"[{datetime.now().isoformat()}] 开始每日综合反思...")

    # 1. 收集系统数据 (保留原有逻辑)
    print("  收集系统运行数据...")
    system_data = collect_today_data()
    conv_count = sum(c["message_count"] for c in system_data["conversations"])
    print(f"  今日对话: {len(system_data['conversations'])} 用户, {conv_count} 条消息")
    print(f"  错误数: {len(system_data['errors'])}")

    # 2. 收集策略数据
    print("  收集量化策略数据...")
    strategy_data = collect_strategy_data()
    pool_size = strategy_data.get("factor_pool", {}).get("pool_size", 0)
    run_count = len(strategy_data.get("recent_runs", []))
    shadow_count = len(strategy_data.get("shadows", {}))
    pos_count = strategy_data.get("live_holdings", {}).get("position_count", 0)
    print(f"  因子池: {pool_size}, 近期 run: {run_count}, shadow: {shadow_count}, 持仓: {pos_count}")

    # 3. 检索市场情报
    print("  检索市场情报...")
    market_intel = collect_market_intelligence()

    # 4. 跳过判断 (放宽: 有因子池或有对话即反思)
    has_strategy_data = pool_size > 0 or pos_count > 0
    has_activity = conv_count > 0 or system_data["errors"]
    if not has_strategy_data and not has_activity:
        is_weekend = date.today().weekday() >= 5
        summary = "周末无活动" if is_weekend else "今日无活动且无策略数据"
        print(f"  {summary}，跳过反思")
        save_reflection({
            "date": date.today().isoformat(),
            "score": -1,
            "summary": summary,
        })
        return

    # 5. 调用 Claude 综合反思
    print("  调用 Claude 进行综合反思...")
    reflection = run_reflection(system_data, strategy_data, market_intel)
    score = reflection.get("score", 0)
    print(f"  反思完成，评分: {score}/10")
    print(f"  总结: {reflection.get('summary', 'N/A')}")

    # 6. 保存行动计划 (Markdown)
    plan_file = save_action_plan(reflection)
    print(f"  行动计划已保存: {plan_file}")

    # 7. 保存反思记录 (JSON)
    reflect_file = save_reflection(reflection)
    print(f"  反思记录已保存: {reflect_file}")

    # 8. 推送飞书简报
    print("  推送飞书简报...")
    push_reflection_to_feishu(reflection)

    print(f"[{datetime.now().isoformat()}] 每日综合反思完成")


if __name__ == "__main__":
    main()
