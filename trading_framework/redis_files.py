#!/usr/bin/env python3
"""
Redis 临时文件管理工具
供 Claude CLI 调用，查看/提取用户历史文件

用法:
  python redis_files.py list <user_open_id> [数量] [偏移]           # 列出历史文件（默认10条）
  python redis_files.py search <user_open_id> <关键词> [数量]       # 按文件名模糊搜索
  python redis_files.py get <user_open_id> <file_id>               # 提取文件到磁盘，输出路径
"""

import sys
import os
import json
import time
from pathlib import Path
from dotenv import load_dotenv

BOT_DIR = Path(__file__).parent
load_dotenv(BOT_DIR / ".env", override=True)

import redis

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
PREFIX = "tmpfile"

rds = redis.Redis.from_url(REDIS_URL, decode_responses=False)


def _print_files(files, total, offset):
    """格式化输出文件列表"""
    for meta in files:
        fid = meta["file_id"]
        ttl = meta.get("ttl_remaining", 0)
        days_left = ttl // 86400 if ttl > 0 else 0
        ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(meta.get("ts", 0)))
        size_kb = meta.get("size", 0) / 1024
        print(f"  {fid}  |  {meta.get('name', '?')}  |  {size_kb:.1f}KB  |  {ts}  |  剩余{days_left}天")

    shown = offset + len(files)
    if shown < total:
        print(f"\n还有 {total - shown} 个更多文件，用 offset={shown} 查看下一页")


def list_files(user_id: str, limit: int = 10, offset: int = 0):
    """列出用户临时文件"""
    idx_key = f"{PREFIX}:idx:{user_id}"
    all_ids = rds.lrange(idx_key, 0, -1)
    if not all_ids:
        print("没有历史文件")
        return

    files = []
    for fid_bytes in all_ids:
        fid = fid_bytes.decode()
        key = f"{PREFIX}:{user_id}:{fid}"
        meta_raw = rds.hget(key, "meta")
        if meta_raw:
            meta = json.loads(meta_raw)
            meta["file_id"] = fid
            meta["ttl_remaining"] = rds.ttl(key)
            files.append(meta)

    total = len(files)
    page = files[offset:offset + limit]
    print(f"共 {total} 个文件，显示第 {offset+1}-{offset+len(page)} 个:\n")
    _print_files(page, total, offset)


def search_files(user_id: str, keyword: str, limit: int = 10):
    """按文件名模糊搜索"""
    idx_key = f"{PREFIX}:idx:{user_id}"
    all_ids = rds.lrange(idx_key, 0, -1)
    if not all_ids:
        print("没有历史文件")
        return

    kw = keyword.lower()
    matched = []
    for fid_bytes in all_ids:
        fid = fid_bytes.decode()
        key = f"{PREFIX}:{user_id}:{fid}"
        meta_raw = rds.hget(key, "meta")
        if not meta_raw:
            continue
        meta = json.loads(meta_raw)
        name = meta.get("name", "")
        if kw in name.lower() or kw in fid.lower():
            meta["file_id"] = fid
            meta["ttl_remaining"] = rds.ttl(key)
            matched.append(meta)

    if not matched:
        print(f"没有匹配 '{keyword}' 的文件")
        return

    total = len(matched)
    page = matched[:limit]
    print(f"搜索 '{keyword}'，匹配 {total} 个:\n")
    _print_files(page, total, 0)


def get_file(user_id: str, file_id: str):
    """从 Redis 取出文件到磁盘"""
    key = f"{PREFIX}:{user_id}:{file_id}"
    data = rds.hget(key, "data")
    meta_raw = rds.hget(key, "meta")

    if not data:
        print(f"文件不存在或已过期: {file_id}", file=sys.stderr)
        sys.exit(1)

    meta = json.loads(meta_raw) if meta_raw else {}
    name = meta.get("name", f"tmp_{file_id}")
    dest = BOT_DIR / f"_tmp_{name}"
    with open(dest, 'wb') as f:
        f.write(data)
    print(str(dest))


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]
    user_id = sys.argv[2]

    if cmd == "list":
        limit = int(sys.argv[3]) if len(sys.argv) > 3 else 10
        offset = int(sys.argv[4]) if len(sys.argv) > 4 else 0
        list_files(user_id, limit, offset)
    elif cmd == "search" and len(sys.argv) >= 4:
        keyword = sys.argv[3]
        limit = int(sys.argv[4]) if len(sys.argv) > 4 else 10
        search_files(user_id, keyword, limit)
    elif cmd == "get" and len(sys.argv) >= 4:
        get_file(user_id, sys.argv[3])
    else:
        print(__doc__)
        sys.exit(1)
