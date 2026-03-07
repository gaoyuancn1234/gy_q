"""统一通知文件夹 — 各子系统写入, smart_bot 读取."""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path

NOTIFY_DIR = Path(__file__).parent

CATEGORIES = ("mining", "signal", "reflection", "monitor")


def write_notification(category: str, title: str, body: str, level: str = "info"):
    """写一条通知到 notifications/{category}/{date}.jsonl"""
    if category not in CATEGORIES:
        raise ValueError(f"Invalid category: {category}, must be one of {CATEGORIES}")
    if level not in ("info", "warn", "error"):
        level = "info"

    cat_dir = NOTIFY_DIR / category
    cat_dir.mkdir(exist_ok=True)

    entry = {
        "ts": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "level": level,
        "title": title[:100],
        "body": body[:500],
    }

    today = datetime.now().strftime("%Y-%m-%d")
    filepath = cat_dir / f"{today}.jsonl"

    try:
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass  # 通知写入失败不应影响主流程


def read_recent(categories=None, hours=24, max_items=20):
    """读取最近 N 小时的通知, 按时间倒序, 截断到 max_items 条.

    Returns:
        list[dict]: [{ts, level, category, title, body}, ...]
    """
    if categories is None:
        categories = CATEGORIES

    cutoff = datetime.now() - timedelta(hours=hours)
    cutoff_date = cutoff.strftime("%Y-%m-%d")
    results = []

    for cat in categories:
        cat_dir = NOTIFY_DIR / cat
        if not cat_dir.is_dir():
            continue

        for filepath in sorted(cat_dir.glob("*.jsonl"), reverse=True):
            # 文件名即日期, 跳过太旧的
            date_str = filepath.stem
            if date_str < cutoff_date:
                break

            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            entry = json.loads(line)
                            ts_str = entry.get("ts", "")
                            if ts_str >= cutoff.strftime("%Y-%m-%dT%H:%M:%S"):
                                entry["category"] = cat
                                results.append(entry)
                        except json.JSONDecodeError:
                            continue
            except Exception:
                continue

    # 按时间倒序
    results.sort(key=lambda x: x.get("ts", ""), reverse=True)

    # 顺便清理旧文件
    try:
        cleanup()
    except Exception:
        pass

    return results[:max_items]


def cleanup(days=7):
    """删除超过 days 天的 JSONL 文件."""
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    for cat in CATEGORIES:
        cat_dir = NOTIFY_DIR / cat
        if not cat_dir.is_dir():
            continue
        for filepath in cat_dir.glob("*.jsonl"):
            if filepath.stem < cutoff:
                try:
                    filepath.unlink()
                except Exception:
                    pass
