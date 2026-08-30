#!/usr/bin/env python3
"""
智能飞书机器人 v4 (Claude CLI session)
- Claude CLI session resume 自动维护对话上下文
- 每用户独立消息队列/状态
- 消息队列处理（防止错位）
"""

import os
import sys

# Windows 控制台默认非 UTF-8 编码（如 GBK），print 中文/emoji 会 UnicodeEncodeError 崩溃
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import json
import time
import shutil
import threading
import queue
import re
import subprocess
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from collections import deque

from dotenv import load_dotenv
import redis

import lark_oapi as lark
from lark_oapi.api.im.v1 import *
from lark_oapi.event.callback.model.p2_card_action_trigger import P2CardActionTriggerResponse
# 使用 claude CLI 而不是 anthropic SDK
# 调用 claude 子进程时需清除 Claude Code 注入的环境变量，避免嵌套会话错误
_CLEAN_ENV = {k: v for k, v in os.environ.items() if k not in ('CLAUDECODE', 'CLAUDE_CODE_ENTRYPOINT')}

# ============================================================
# 配置
# ============================================================

WORK_DIR = Path(__file__).parent.parent
BOT_DIR = Path(__file__).parent
load_dotenv(BOT_DIR / ".env", override=True)

APPS = []
for i in range(1, 10):
    app_id = os.environ.get(f"FEISHU_APP_ID_{i}")
    app_secret = os.environ.get(f"FEISHU_APP_SECRET_{i}")
    if app_id and app_secret:
        APPS.append({"app_id": app_id, "app_secret": app_secret})
    else:
        break

# Claude CLI 可执行文件路径（跨平台自动探测，可用 CLAUDE_BIN 环境变量覆盖）
CLAUDE_BIN = os.environ.get("CLAUDE_BIN") or shutil.which("claude") or "claude"

# 授权用户白名单（逗号分隔的 open_id）。
# 机器人会以 --dangerously-skip-permissions 执行 Claude，等同于把本机的
# 任意命令执行权交给发消息的人，因此未列入白名单的用户一律拒绝。
# 留空 = 拒绝所有人（首次使用时给自己发条消息，控制台会打印你的 open_id）。
ALLOWED_OPEN_IDS = {
    x.strip() for x in os.environ.get("FEISHU_ALLOWED_OPEN_IDS", "").split(",") if x.strip()
}

# 消息处理配置
MESSAGE_BATCH_WAIT = 1.2  # 等待批量消息的时间(秒)
MAX_WAIT_FOR_REPLY = 30   # 等待用户回复的最大时间(秒)

# Redis 配置
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
TEMP_FILE_TTL = int(os.environ.get("TEMP_FILE_TTL", 86400))  # 默认 24 小时
MSG_DEDUP_TTL = 3600  # 消息去重 key 保留 1 小时

# ============================================================
# Redis 临时文件存储
# ============================================================

class TempFileStore:
    """将临时文件存入 Redis，用时取出，TTL 自动过期"""

    PREFIX = "tmpfile"
    DEDUP_KEY = "bot:processed_msgs"

    def __init__(self, redis_url: str = REDIS_URL):
        self.rds = redis.Redis.from_url(redis_url, decode_responses=False)
        try:
            self.rds.ping()
            self._available = True
            print(f"  • Redis 已连接")
        except redis.ConnectionError:
            self._available = False
            print(f"  ⚠ Redis 不可用，临时文件将保留在磁盘")

    @property
    def available(self) -> bool:
        return self._available

    def _key(self, user_id: str, file_id: str) -> str:
        return f"{self.PREFIX}:{user_id}:{file_id}"

    def _index_key(self, user_id: str) -> str:
        return f"{self.PREFIX}:idx:{user_id}"

    def store(self, file_path: str, user_id: str, ttl: int = TEMP_FILE_TTL) -> Optional[str]:
        """存入文件到 Redis，返回 file_id，并删除磁盘文件"""
        if not self._available:
            return None
        p = Path(file_path)
        if not p.exists():
            return None

        file_id = f"{int(time.time())}_{p.name}"
        key = self._key(user_id, file_id)
        meta = json.dumps({"name": p.name, "size": p.stat().st_size,
                           "ts": time.time()}).encode()
        try:
            with open(p, 'rb') as f:
                data = f.read()
            pipe = self.rds.pipeline()
            pipe.hset(key, mapping={b"data": data, b"meta": meta})
            pipe.expire(key, ttl)
            # 维护用户文件索引（最近 50 个）
            idx_key = self._index_key(user_id)
            pipe.lpush(idx_key, file_id.encode())
            pipe.ltrim(idx_key, 0, 49)
            pipe.expire(idx_key, ttl)
            pipe.execute()
            # 删除磁盘文件
            p.unlink(missing_ok=True)
            print(f"[Redis] 已存储 {p.name} ({len(data)} bytes, TTL={ttl}s)")
            return file_id
        except Exception as e:
            print(f"[Redis] 存储失败: {e}")
            return None

    def retrieve(self, user_id: str, file_id: str, dest_dir: Path = None) -> Optional[str]:
        """从 Redis 取出文件到磁盘临时路径，返回路径"""
        if not self._available:
            return None
        key = self._key(user_id, file_id)
        try:
            data = self.rds.hget(key, "data")
            meta_raw = self.rds.hget(key, "meta")
            if not data:
                return None
            meta = json.loads(meta_raw) if meta_raw else {}
            name = meta.get("name", f"tmp_{file_id}")
            dest = (dest_dir or BOT_DIR) / f"_tmp_{name}"
            with open(dest, 'wb') as f:
                f.write(data)
            print(f"[Redis] 已取出 {name} → {dest}")
            return str(dest)
        except Exception as e:
            print(f"[Redis] 取出失败: {e}")
            return None

    def list_files(self, user_id: str, limit: int = 10, offset: int = 0, keyword: str = "") -> tuple[List[dict], int]:
        """列出用户临时文件，支持按文件名模糊搜索，返回 (文件列表, 总数)"""
        if not self._available:
            return [], 0
        idx_key = self._index_key(user_id)
        try:
            all_ids = self.rds.lrange(idx_key, 0, -1)
            # 构建完整列表（带 meta），顺便过滤关键词
            kw = keyword.lower()
            matched = []
            for fid_bytes in all_ids:
                fid = fid_bytes.decode()
                meta_raw = self.rds.hget(self._key(user_id, fid), "meta")
                if not meta_raw:
                    continue
                meta = json.loads(meta_raw)
                if kw and kw not in meta.get("name", "").lower() and kw not in fid.lower():
                    continue
                meta["file_id"] = fid
                meta["ttl_remaining"] = self.rds.ttl(self._key(user_id, fid))
                matched.append(meta)
            total = len(matched)
            return matched[offset:offset + limit], total
        except Exception as e:
            print(f"[Redis] 列出失败: {e}")
            return [], 0

    # --- 消息去重 ---

    def is_msg_processed(self, msg_id: str) -> bool:
        if not self._available:
            return False
        return self.rds.sismember(self.DEDUP_KEY, msg_id)

    def mark_msg_processed(self, msg_id: str):
        if not self._available:
            return
        self.rds.sadd(self.DEDUP_KEY, msg_id)
        # 定期清理：集合超过 2000 时随机移除一半
        if self.rds.scard(self.DEDUP_KEY) > 2000:
            members = self.rds.srandmember(self.DEDUP_KEY, 1000)
            if members:
                self.rds.srem(self.DEDUP_KEY, *members)

    def cleanup_disk(self, pattern: str = "user_image_*.png"):
        """启动时清理磁盘上残留的临时文件"""
        count = 0
        for p in WORK_DIR.glob(pattern):
            p.unlink(missing_ok=True)
            count += 1
        for p in BOT_DIR.glob(pattern):
            p.unlink(missing_ok=True)
            count += 1
        if count:
            print(f"[清理] 删除 {count} 个残留临时文件")


# 动态超时配置（秒）
TIMEOUT_QUICK = 60        # 极简问答: 1分钟
TIMEOUT_SIMPLE = 120      # 简单问答: 2分钟
TIMEOUT_SCRIPT = 600      # 简单脚本: 10分钟
TIMEOUT_MEDIUM = 1800     # 中等任务: 30分钟
TIMEOUT_ANALYSIS = 10800  # 复杂分析: 3小时
TIMEOUT_COMPLEX = 21600   # 复杂代码任务: 6小时

# ============================================================
# 数据结构
# ============================================================

@dataclass
class Message:
    """消息"""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: float = field(default_factory=time.time)
    msg_id: str = ""
    image_path: str = ""  # 图片路径

# ============================================================
# 消息队列处理器
# ============================================================

class MessageQueue:
    """消息队列 - 处理消息批量和等待"""

    def __init__(self):
        self.pending: List[Message] = []
        self.lock = threading.Lock()
        self.waiting_event = threading.Event()
        self.is_waiting = False

    def add(self, msg: Message):
        """添加消息"""
        with self.lock:
            self.pending.append(msg)
            if self.is_waiting:
                self.waiting_event.set()

    def collect(self, wait_time: float = MESSAGE_BATCH_WAIT) -> List[Message]:
        """收集批量消息"""
        time.sleep(wait_time)
        with self.lock:
            batch = self.pending.copy()
            self.pending.clear()
            return batch

    def wait_for_reply(self, timeout: float = MAX_WAIT_FOR_REPLY) -> Optional[str]:
        """
        智能等待用户回复
        - 如果已有消息，立即返回
        - 否则等待直到超时或收到新消息
        """
        # 先检查已有消息
        with self.lock:
            if self.pending:
                combined = " ".join([m.content for m in self.pending])
                self.pending.clear()
                return combined

        # 等待新消息
        self.waiting_event.clear()
        self.is_waiting = True
        got_reply = self.waiting_event.wait(timeout=timeout)
        self.is_waiting = False

        if got_reply:
            time.sleep(0.5)  # 短暂等待可能的连续消息
            with self.lock:
                if self.pending:
                    combined = " ".join([m.content for m in self.pending])
                    self.pending.clear()
                    return combined
        return None

    def has_pending(self) -> bool:
        with self.lock:
            return len(self.pending) > 0

# ============================================================
# Claude CLI Session 管理 (单一共享 session)
# ============================================================

class ClaudeSession:
    """Claude CLI session 管理 — 所有用户共享同一个 session"""

    SESSION_FILE = BOT_DIR / "claude_session_id.txt"

    def __init__(self):
        self.session_id: Optional[str] = None
        self.lock = threading.Lock()
        self._load_session_id()

    def _load_session_id(self):
        if self.SESSION_FILE.exists():
            sid = self.SESSION_FILE.read_text().strip()
            if sid:
                self.session_id = sid
                print(f"  • Claude session: {sid[:20]}...")

    def _save_session_id(self):
        self.SESSION_FILE.write_text(self.session_id or "")

    def update(self, session_id: str):
        """更新 session_id 并持久化"""
        if session_id and session_id != self.session_id:
            self.session_id = session_id
            self._save_session_id()

    def reset(self):
        """重置 session（resume 失败时调用）"""
        self.session_id = None
        self.SESSION_FILE.unlink(missing_ok=True)
        print("[ClaudeSession] session 已重置，下次将创建新 session")


# ============================================================
# 用户会话 (每用户独立状态)
# ============================================================

class UserSession:
    """封装单个用户的所有状态"""

    def __init__(self, open_id: str):
        self.open_id = open_id
        self.lark_client: Optional[lark.Client] = None  # 由 on_message/on_card 注入
        self.msg_queue = MessageQueue()
        self.processing_lock = threading.Lock()
        self.is_processing = False
        self.current_process: Optional[subprocess.Popen] = None
        self.pending_messages: List[str] = []
        self.current_task: str = ""
        self.last_result: str = ""
        self.pending_actions: Dict[str, dict] = {}
        self._actions_lock = threading.Lock()
        self.paper_context: Optional[str] = None  # 当前论文对话上下文 (paper_id)
        self.last_images: List[tuple] = []  # 最近图片列表 [(path, timestamp), ...] 用于跨消息关联

    # ---- pending_actions 管理 ----

    def add_pending_action(self, action_id: str, action_type: str, data: dict):
        with self._actions_lock:
            self.pending_actions[action_id] = {
                "type": action_type,
                "data": data,
                "created": datetime.now().isoformat()
            }

    def pop_pending_action(self, action_id: str) -> Optional[dict]:
        with self._actions_lock:
            return self.pending_actions.pop(action_id, None)

    def get_pending_actions(self) -> dict:
        with self._actions_lock:
            return dict(self.pending_actions)

    def clear_pending_actions(self):
        with self._actions_lock:
            self.pending_actions.clear()


class SessionManager:
    """线程安全的会话注册表，按 open_id 懒创建"""

    def __init__(self):
        self._sessions: Dict[str, UserSession] = {}
        self._lock = threading.Lock()

    def get_or_create(self, open_id: str) -> UserSession:
        with self._lock:
            if open_id not in self._sessions:
                self._sessions[open_id] = UserSession(open_id)
                print(f"[会话] 创建新用户会话: {open_id[:20]}...")
            return self._sessions[open_id]

# ============================================================
# 智能机器人
# ============================================================

class SmartBot:
    """智能飞书机器人（多用户隔离）"""

    def __init__(self):
        self.session_mgr = SessionManager()
        self.claude_session = ClaudeSession()
        self.tmp_store = TempFileStore()
        self.tmp_store.cleanup_disk()  # 启动时清理残留
        self.processed_msg_ids: set = set()  # 内存降级用（Redis 不可用时）
        self._ws_disconnect_times: deque = deque()  # 断连时间戳滑动窗口
        self._ws_disconnect_alert_cd: float = 0  # 告警冷却截止时间戳
        self._ws_clients: list = []  # WebSocket 客户端引用 (心跳自适应用)
        self._default_ping_interval: int = 120  # SDK 默认 ping 间隔

    def send_text(self, text: str, session: 'UserSession' = None):
        """发送文本消息 (超长自动分段)"""
        client = session.lark_client if session else None
        open_id = session.open_id if session else None
        if not client or not open_id:
            print(f"[Bot] {text}")
            return

        # 飞书文本消息限制 ~4000 字符，超长自动分段
        MAX_TEXT = 4000
        chunks = self._split_text(text, MAX_TEXT) if len(text) > MAX_TEXT else [text]
        total = len(chunks)

        for i, chunk in enumerate(chunks):
            if total > 1:
                chunk = f"{chunk}\n\n— ({i+1}/{total})"
            self._send_lark_message(client, open_id, "text",
                                    json.dumps({"text": chunk}))

    def _send_lark_message(self, client, open_id: str, msg_type: str,
                           content: str, max_retries: int = 3):
        """发送飞书消息，失败时尝试重启 VPN 后重试"""
        vpn_restarted = False
        for attempt in range(max_retries):
            try:
                req = CreateMessageRequest.builder() \
                    .receive_id_type("open_id") \
                    .request_body(CreateMessageRequestBody.builder()
                        .receive_id(open_id)
                        .msg_type(msg_type)
                        .content(content)
                        .build()) \
                    .build()
                resp = client.im.v1.message.create(req)
                if resp.code == 0:
                    return True
                raise Exception(f"lark api code={resp.code} msg={resp.msg}")
            except Exception as e:
                print(f"[发送失败] 第{attempt+1}次: {e}")
                if attempt < max_retries - 1:
                    if not vpn_restarted:
                        print("[VPN] 发送失败，尝试重启 LetsVPN...")
                        self._restart_vpn()
                        vpn_restarted = True
                        time.sleep(10)
                    else:
                        time.sleep(3)
                else:
                    print(f"[发送失败] {max_retries}次全部失败，消息丢失: {content[:100]}")
        return False

    def send_markdown(self, content: str, title: str = "", session: 'UserSession' = None):
        """发送 Markdown 卡片消息 (超长自动分段)"""
        client = session.lark_client if session else None
        open_id = session.open_id if session else None
        if not client or not open_id:
            print(f"[Bot MD] {content[:100]}")
            return

        # 飞书卡片 JSON 限制 ~28KB, 预留 header/元数据 (~1KB)
        # 中文字符 UTF-8 占 3 bytes, 保守取 5000 字符 (~15KB content + overhead 安全)
        MAX_CONTENT = 5000
        chunks = self._split_text(content, MAX_CONTENT) if len(content) > MAX_CONTENT else [content]

        for i, chunk in enumerate(chunks):
            card = {
                "schema": "2.0",
                "config": {"wide_screen_mode": True},
                "body": {
                    "direction": "vertical",
                    "elements": [
                        {"tag": "markdown", "content": chunk}
                    ]
                }
            }
            chunk_title = title
            if title and len(chunks) > 1:
                chunk_title = f"{title} ({i+1}/{len(chunks)})"
            if chunk_title:
                card["header"] = {
                    "title": {"tag": "plain_text", "content": chunk_title},
                    "template": "blue"
                }

            ok = self._send_lark_message(client, open_id, "interactive",
                                         json.dumps(card))
            if not ok:
                # markdown 卡片发送失败，降级为纯文本
                self.send_text(chunk, session)

    def upload_image(self, image_path: str, session: 'UserSession' = None) -> Optional[str]:
        """上传图片到飞书，返回image_key"""
        client = session.lark_client if session else None
        if not client:
            print(f"[上传图片] 客户端未初始化")
            return None

        try:
            from lark_oapi.api.im.v1 import CreateImageRequest, CreateImageRequestBody

            with open(image_path, 'rb') as f:
                req = CreateImageRequest.builder() \
                    .request_body(CreateImageRequestBody.builder()
                        .image_type("message")
                        .image(f)
                        .build()) \
                    .build()
                resp = client.im.v1.image.create(req)

                if resp.success():
                    image_key = resp.data.image_key
                    print(f"[上传成功] image_key: {image_key}")
                    return image_key
                else:
                    print(f"[上传失败] {resp.code}: {resp.msg}")
                    return None
        except Exception as e:
            print(f"[上传图片错误] {e}")
            return None

    def download_image(self, msg_id: str, image_key: str, session: 'UserSession' = None) -> Optional[str]:
        """下载图片到本地"""
        client = session.lark_client if session else None
        if not client or not image_key:
            return None

        try:
            from lark_oapi.api.im.v1 import GetMessageResourceRequest

            req = GetMessageResourceRequest.builder() \
                .message_id(msg_id) \
                .file_key(image_key) \
                .type("image") \
                .build()
            resp = client.im.v1.message_resource.get(req)

            if resp.success():
                # 保存到临时文件
                save_path = WORK_DIR / f"user_image_{int(time.time() * 1000)}.png"
                with open(save_path, 'wb') as f:
                    f.write(resp.file.read())
                print(f"[下载图片] {save_path}")
                return str(save_path)
            else:
                print(f"[下载图片失败] {resp.code}: {resp.msg}")
                return None
        except Exception as e:
            print(f"[下载图片错误] {e}")
            return None

    def get_message_content(self, message_id: str, session: 'UserSession' = None) -> tuple[Optional[str], Optional[str]]:
        """获取消息内容（用于引用消息），返回 (文本, 图片路径)"""
        client = session.lark_client if session else None
        if not client or not message_id:
            return None, None

        try:
            from lark_oapi.api.im.v1 import GetMessageRequest

            req = GetMessageRequest.builder() \
                .message_id(message_id) \
                .build()
            resp = client.im.v1.message.get(req)

            if resp.success():
                msg = resp.data.items[0] if resp.data.items else None
                if msg:
                    content = json.loads(msg.body.content)
                    if msg.msg_type == "text":
                        return content.get("text", "")[:200], None
                    elif msg.msg_type == "image":
                        # 下载被引用的图片
                        image_key = content.get("image_key", "")
                        if image_key:
                            img_path = self.download_image(message_id, image_key, session)
                            return "[引用图片]", img_path
                        return "[图片]", None
                    else:
                        return f"[{msg.msg_type}]", None
            return None, None
        except Exception as e:
            print(f"[获取消息失败] {e}")
            return None, None

    def send_image(self, image_path: str, session: 'UserSession' = None) -> bool:
        """发送图片消息"""
        client = session.lark_client if session else None
        open_id = session.open_id if session else None
        if not client or not open_id:
            print(f"[发送图片] 客户端未初始化")
            return False

        # 检查文件是否存在
        if not Path(image_path).exists():
            self.send_text(f"❌ 图片不存在: {image_path}", session)
            return False

        # 上传图片
        image_key = self.upload_image(image_path, session)
        if not image_key:
            self.send_text("❌ 图片上传失败", session)
            return False

        # 发送图片消息
        try:
            req = CreateMessageRequest.builder() \
                .receive_id_type("open_id") \
                .request_body(CreateMessageRequestBody.builder()
                    .receive_id(open_id)
                    .msg_type("image")
                    .content(json.dumps({"image_key": image_key}))
                    .build()) \
                .build()
            resp = client.im.v1.message.create(req)
            if resp.success():
                print(f"[发送图片成功] {image_path}")
                return True
            else:
                print(f"[发送图片失败] {resp.code}: {resp.msg}")
                self.send_text(f"❌ 发送失败: {resp.msg}", session)
                return False
        except Exception as e:
            print(f"[发送图片错误] {e}")
            self.send_text(f"❌ 发送错误: {e}", session)
            return False

    def add_reaction(self, message_id: str, emoji: str = "THUMBSUP", session: 'UserSession' = None) -> bool:
        """给消息添加表情回应"""
        client = session.lark_client if session else None
        if not client or not message_id:
            return False

        try:
            from lark_oapi.api.im.v1 import CreateMessageReactionRequest, CreateMessageReactionRequestBody, Emoji

            req = CreateMessageReactionRequest.builder() \
                .message_id(message_id) \
                .request_body(CreateMessageReactionRequestBody.builder()
                    .reaction_type(Emoji.builder().emoji_type(emoji).build())
                    .build()) \
                .build()
            resp = client.im.v1.message_reaction.create(req)
            if resp.success():
                print(f"[添加表情] {emoji} -> {message_id[:20]}")
                return True
            else:
                print(f"[添加表情失败] {resp.code}: {resp.msg}")
                return False
        except Exception as e:
            print(f"[添加表情错误] {e}")
            return False

    def send_card(self, card: dict, session: 'UserSession' = None):
        """发送卡片消息"""
        client = session.lark_client if session else None
        open_id = session.open_id if session else None
        if not client or not open_id:
            print(f"[Card] {card.get('header', {}).get('title', {}).get('content', '')}")
            return

        try:
            req = CreateMessageRequest.builder() \
                .receive_id_type("open_id") \
                .request_body(CreateMessageRequestBody.builder()
                    .receive_id(open_id)
                    .msg_type("interactive")
                    .content(json.dumps(card, ensure_ascii=False))
                    .build()) \
                .build()
            client.im.v1.message.create(req)
        except Exception as e:
            print(f"[发送卡片失败] {e}")

    def send_confirm_card(self, title: str, content: str, action_id: str, session: 'UserSession' = None):
        """发送确认卡片"""
        card = {
            "config": {"wide_screen_mode": True},
            "header": {"title": {"tag": "plain_text", "content": title}, "template": "blue"},
            "elements": [
                {"tag": "markdown", "content": content},
                {"tag": "hr"},
                {
                    "tag": "action",
                    "actions": [
                        {"tag": "button", "text": {"tag": "plain_text", "content": "✅ 允许"},
                         "type": "primary", "value": {"action": "allow", "id": action_id}},
                        {"tag": "button", "text": {"tag": "plain_text", "content": "❌ 取消"},
                         "type": "danger", "value": {"action": "cancel", "id": action_id}}
                    ]
                }
            ]
        }
        self.send_card(card, session)

    def ask_and_wait(self, question: str, session: 'UserSession' = None, timeout: float = 30) -> Optional[str]:
        """
        发送问题并智能等待回复
        - 如果队列已有消息，直接使用
        - 否则发送问题并等待
        """
        msg_queue = session.msg_queue if session else MessageQueue()
        # 先检查是否已有消息
        if msg_queue.has_pending():
            batch = msg_queue.collect(wait_time=0.3)
            if batch:
                return " ".join([m.content for m in batch])

        # 发送问题并等待
        self.send_text(question, session)
        return msg_queue.wait_for_reply(timeout=timeout)

    def estimate_task_complexity(self, user_input: str, session: 'UserSession' = None) -> tuple:
        """
        调用 Claude CLI 智能体判断任务复杂度，返回 (超时秒数, 复杂度描述)
        传入最近对话上下文，让 AI 理解追问/延续性指令的真实复杂度
        如果智能体调用失败，使用简单的 fallback 逻辑
        """
        # 超时级别映射
        level_map = {
            1: (TIMEOUT_QUICK, "快速问答(1分钟)"),
            2: (TIMEOUT_SIMPLE, "简单任务(2分钟)"),
            3: (TIMEOUT_SCRIPT, "脚本任务(10分钟)"),
            4: (TIMEOUT_MEDIUM, "中等任务(30分钟)"),
            5: (TIMEOUT_ANALYSIS, "分析任务(3小时)"),
            6: (TIMEOUT_COMPLEX, "复杂任务(6小时)"),
        }

        # Claude CLI session 自动维护对话上下文，此处不再手动注入
        recent_context = ""

        try:
            context_block = ""
            if recent_context:
                context_block = f"""
【最近对话上下文】（用于理解当前指令是否是追问/延续）
{recent_context}

"""

            prompt = f"""你是一个任务复杂度评估智能体。分析用户发给AI助手的指令，判断AI完成该任务需要多少时间。
{context_block}【用户指令】
{user_input[:500]}

请从以下6个级别中选择最合适的：
1 = 极简问答（闲聊、打招呼、简单问答，如"你好"、"在吗"、"XX是什么"）
2 = 简单任务（查看状态、简单查询、一句话能答的问题）
3 = 脚本任务（画简单图、运行脚本、小修改、写个简单函数）
4 = 中等任务（添加功能、中等编码、调试问题）
5 = 复杂分析（爬取数据、数据分析+画图、多文件修改、系统调试、深度分析、写报告、房价/股票等预测分析）
6 = 大型任务（重构系统、架构改造、从零开发、全面改造、大规模爬虫+分析）

【重要提示】
- 涉及爬取/抓取/采集数据、分析+画图/可视化、预测/建模/训练、XX市/XX区/XX行业+分析 通常至少级别5
- 如果用户指令很短但上下文显示是对复杂任务的追问（如"你计划怎么修复"、"继续"、"执行吧"），要结合上下文判断真实复杂度，不要只看当前指令长度
- 追问/延续性指令的复杂度应≥上一轮任务的复杂度"""

            json_schema = json.dumps({
                "type": "object",
                "properties": {
                    "level": {"type": "integer", "minimum": 1, "maximum": 6, "description": "复杂度级别1-6"},
                    "reason": {"type": "string", "description": "一句话理由"}
                },
                "required": ["level", "reason"]
            })

            cmd = [
                CLAUDE_BIN, '--print', '--dangerously-skip-permissions',
                '--output-format', 'json',
                '--model', 'haiku',
                '--json-schema', json_schema,
                '-p', prompt
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, encoding='utf-8', errors='replace',
                timeout=15, cwd=str(WORK_DIR), env=_CLEAN_ENV
            )
            output = result.stdout.strip()

            if output:
                data = json.loads(output)
                # --json-schema 的结构化输出在 structured_output 字段
                if isinstance(data, dict) and "structured_output" in data:
                    result_data = data["structured_output"]
                elif isinstance(data, dict) and "result" in data:
                    result_text = data["result"]
                    if isinstance(result_text, str):
                        result_data = json.loads(result_text)
                    else:
                        result_data = result_text
                elif isinstance(data, dict) and "level" in data:
                    result_data = data
                else:
                    raise ValueError(f"未知的返回结构: {list(data.keys()) if isinstance(data, dict) else type(data)}")

                level = int(result_data["level"])
                reason = result_data.get("reason", "")
                level = max(1, min(6, level))
                timeout, level_desc = level_map[level]
                desc = f"{level_desc} (AI判断: {reason})"
                return (timeout, desc)

        except subprocess.TimeoutExpired:
            print("[复杂度评估] Claude 智能体超时，使用 fallback")
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            print(f"[复杂度评估] 解析失败: {e}，使用 fallback")
        except Exception as e:
            print(f"[复杂度评估] 异常: {e}，使用 fallback")

        # ── Fallback: 关键词规则兜底（不依赖输入长度） ──
        text_lower = user_input.lower()
        # 高复杂度关键词检测
        heavy_keywords = ['分析', '预测', '爬取', '抓取', '采集', '画图', '可视化',
                          '训练', '建模', '回测', '报告', '房价', '股票', '行情',
                          '排序', '排名', '对比', '趋势', '策略', '优化']
        # 极简问答关键词（纯闲聊，不需要执行任何操作）
        trivial_keywords = ['你好', '在吗', '谢谢', '感谢', '没事了', '好的',
                            '知道了', '明白了', '收到', 'hi', 'hello', '嗯']
        hit_count = sum(1 for kw in heavy_keywords if kw in text_lower)

        if hit_count >= 2:
            return (TIMEOUT_ANALYSIS, "分析任务(30分钟) (fallback: 多关键词命中)")
        elif hit_count == 1:
            return (TIMEOUT_MEDIUM, "中等任务(10分钟) (fallback: 关键词命中)")

        # 纯闲聊/简单回应 → 快速
        text_stripped = user_input.strip()
        if any(text_stripped == kw or text_stripped.startswith(kw) for kw in trivial_keywords):
            return (TIMEOUT_QUICK, "快速问答(1分钟) (fallback: 闲聊)")

        # 其他情况统一给脚本级超时（5分钟），因为无法从长度判断复杂度
        return (TIMEOUT_SCRIPT, "脚本任务(5分钟) (fallback: 默认)")

    def call_claude(self, user_input: str, session: 'UserSession' = None, image_path: str = None) -> str:
        """调用 Claude CLI (共享 session + resume)"""
        timeout = TIMEOUT_COMPLEX

        # 构建消息前缀 (动态上下文)
        user_id = session.open_id if session else "unknown"
        parts = [f"[用户ID: {user_id}]"]

        # 待确认操作
        if session:
            pending = session.get_pending_actions()
            if pending:
                parts.append("[待确认操作]")
                for aid, action in pending.items():
                    parts.append(f"- {action.get('type', '?')}: {json.dumps(action.get('data', {}), ensure_ascii=False)}")
                parts.append("用户可能在回应这些待确认操作。")

        # 近期系统通知
        try:
            from notifications import read_recent
            recent = read_recent(hours=24, max_items=10)
            if recent:
                parts.append("\n[近期系统通知]")
                for n in recent:
                    marker = {"info": "📋", "warn": "⚠️", "error": "❌"}.get(n["level"], "📋")
                    parts.append(f"{marker} [{n['category']}] {n['title']} ({n['ts'][-8:-3]})")
                    if n["level"] != "info":
                        parts.append(f"  {n['body'][:200]}")
        except Exception:
            pass

        # 图片
        if image_path and Path(image_path).exists():
            parts.append(f"\n用户发送了图片，路径: {image_path}")
            parts.append("请先用 Read 工具查看这张图片，然后回答用户问题。")
            print(f"[Claude] 需要读取图片: {image_path}")

        parts.append(f"\n[用户 {user_id[:8]}]: {user_input}")
        message = "\n".join(parts)

        api_error_keywords = ['403', 'forbidden', 'Failed to authenticate',
                              'API Error', '401', 'unauthorized']
        max_retries = 3
        last_output = ""
        vpn_restarted = False

        self.claude_session.lock.acquire()
        acquired = True
        try:
            for attempt in range(max_retries):
                # 构建命令
                cmd = [CLAUDE_BIN, '-p',
                       '--dangerously-skip-permissions',
                       '--output-format', 'json']
                if self.claude_session.session_id:
                    cmd.extend(['--resume', self.claude_session.session_id])
                cmd.append(message)

                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                    encoding='utf-8', errors='replace',
                    cwd=str(WORK_DIR), env=_CLEAN_ENV
                )
                if session:
                    session.current_process = proc
                try:
                    stdout, stderr = proc.communicate(timeout=timeout)
                    raw = stdout.strip() or stderr.strip() or ""
                except subprocess.TimeoutExpired:
                    print(f"[超时] {timeout}秒已到，任务转入后台继续执行")
                    if session:
                        session.current_process = None
                    # 释放锁再启动后台线程
                    self.claude_session.lock.release()
                    acquired = False
                    self._wait_bg_result(proc, session, "统一6小时")
                    return "⏳ 任务较复杂，已转入后台继续执行，完成后会自动发送结果。"
                finally:
                    if session:
                        session.current_process = None

                # 解析 JSON 输出
                result_text = raw
                try:
                    data = json.loads(raw)
                    result_text = data.get("result", raw)
                    new_sid = data.get("session_id", "")
                    if new_sid:
                        self.claude_session.update(new_sid)
                    if data.get("is_error"):
                        result_text = f"⚠️ Claude 错误: {result_text}"
                except json.JSONDecodeError:
                    pass  # 非 JSON 输出，原样使用

                last_output = result_text or "无响应"

                # resume 失败检测：首次尝试且有 session_id
                if (attempt == 0 and self.claude_session.session_id
                        and proc.returncode != 0 and not result_text):
                    print(f"[resume失败] 重置 session，重试...")
                    self.claude_session.reset()
                    continue

                # API 错误检测 + VPN 重启重试
                if any(kw.lower() in last_output.lower() for kw in api_error_keywords):
                    print(f"[API错误] 第{attempt+1}次: {last_output[:200]}")
                    if attempt < max_retries - 1:
                        if not vpn_restarted:
                            print("[VPN] 疑似 VPN 断连，尝试重启 LetsVPN...")
                            self._restart_vpn()
                            vpn_restarted = True
                            print("[VPN] 等待 10s 让 VPN 建立连接...")
                            time.sleep(10)
                        else:
                            time.sleep(5)
                        continue
                    return f"⚠️ Claude API 访问错误（已重试{max_retries}次），需要你处理：\n\n{last_output}"

                print(f"[Claude输出长度] {len(last_output)} 字符")
                return last_output

            return last_output
        except Exception as e:
            if session:
                session.current_process = None
            return f"调用失败: {e}"
        finally:
            if acquired:
                self.claude_session.lock.release()

    @staticmethod
    def _restart_vpn():
        """尝试重启 VPN 以恢复网络连接

        命令由 .env 的 VPN_RESTART_CMD 提供（原先硬编码 macOS 的 LetsVPN 路径，
        在 Windows 上会静默失败却仍打印成功）。未配置时跳过。
        """
        cmd = os.environ.get("VPN_RESTART_CMD", "").strip()
        if not cmd:
            print("[VPN] 未配置 VPN_RESTART_CMD，跳过重启")
            return
        try:
            subprocess.Popen(cmd, shell=True)
            print("[VPN] 重启命令已执行")
        except Exception as e:
            print(f"[VPN] 重启失败: {e}")

    def _wait_bg_result(self, proc: subprocess.Popen, session: 'UserSession', complexity_desc: str):
        """后台线程等待超时任务完成，完成后主动推送结果给用户"""
        def _is_api_error(text: str) -> bool:
            error_indicators = ['403', 'forbidden', 'Request not allowed',
                                'Failed to authenticate', 'API Error',
                                '401', 'unauthorized', 'rate_limit']
            text_lower = text.lower()
            return any(indicator.lower() in text_lower for indicator in error_indicators)

        def _wait():
            try:
                stdout, stderr = proc.communicate(timeout=1800)
                raw = stdout.strip() or stderr.strip()
                if not raw:
                    self.send_text("后台任务已完成，但没有产生输出。", session)
                    return

                # 解析 JSON 输出，提取 result 和 session_id
                output = raw
                try:
                    data = json.loads(raw)
                    output = data.get("result", raw)
                    new_sid = data.get("session_id", "")
                    if new_sid:
                        with self.claude_session.lock:
                            self.claude_session.update(new_sid)
                except json.JSONDecodeError:
                    pass

                print(f"[后台任务完成] {len(output)} 字符")
                if _is_api_error(output):
                    print(f"[后台任务API错误] {output[:200]}")
                    self._restart_vpn()
                    self.send_text(
                        f"⚠️ 后台任务遇到 API 访问错误（已尝试重启VPN），需要你处理：\n\n"
                        f"{output[:500]}\n\n"
                        f"（原始任务复杂度: {complexity_desc}）",
                        session
                    )
                    return
                session.last_result = output[:500]
                self.process_response(output, session)
            except subprocess.TimeoutExpired:
                proc.kill()
                self.send_text("后台任务执行超过30分钟，已终止。", session)
            except Exception as e:
                print(f"[后台任务异常] {e}")
                self.send_text(f"后台任务执行出错: {e}", session)

        t = threading.Thread(target=_wait, daemon=True)
        t.start()

    def execute_command(self, cmd: str, session: 'UserSession' = None):
        """执行命令 - 动态超时"""
        # 根据命令复杂度设置超时
        timeout = TIMEOUT_SCRIPT  # 默认5分钟
        if 'python' in cmd and any(kw in cmd for kw in ['train', 'backtest', 'test']):
            timeout = TIMEOUT_ANALYSIS  # 1小时
        try:
            result = subprocess.run(
                cmd, shell=True, cwd=str(WORK_DIR),
                capture_output=True, text=True, encoding='utf-8', errors='replace',
                timeout=timeout
            )
            output = result.stdout or result.stderr or "完成"

            # 智能处理错误
            if "请先设置" in output or "错误" in output or "Error" in output:
                self.send_text(f"⚠️ {output[:500]}", session)
                # 分析错误并给出建议
                response = self.call_claude(f"命令执行结果: {output[:200]}\n请告诉用户下一步该怎么做。", session)
                self.process_response(response, session)
            else:
                self.send_text(f"✅ {output[:1000]}", session)
        except Exception as e:
            self.send_text(f"❌ 执行失败: {e}", session)

    @staticmethod
    def _split_text(text: str, max_len: int = 4000) -> list:
        """在段落/标题边界处分割长文本，确保代码块不被截断"""
        if len(text) <= max_len:
            return [text]
        chunks = []
        while text:
            if len(text) <= max_len:
                chunks.append(text)
                break
            # 在 max_len 范围内找最佳分割点: ```代码块边界 > ## 标题 > 空行 > 换行
            cut = max_len
            # 优先在代码块结束符后分割 (避免截断代码块)
            code_end = text.rfind('\n```\n', 0, max_len)
            if code_end > max_len // 3:
                cut = code_end + 4  # 包含 ```\n
            else:
                for sep in ['\n## ', '\n\n', '\n']:
                    pos = text.rfind(sep, 0, max_len)
                    if pos > max_len // 3:
                        cut = pos + (len(sep) if sep == '\n' else 0)
                        break
            chunks.append(text[:cut].rstrip())
            text = text[cut:].lstrip('\n')
        return chunks

    def process_response(self, response: str, session: 'UserSession' = None) -> bool:
        """处理 Claude 响应 - 解析文本、图片、动作标签。返回 True 表示有后台线程在使用图片。"""
        print(f"[Claude响应] {response[:500]}...")

        # ── 解析动作标签 ──
        # [PAPER_SEARCH:url] → 创建论文调研确认卡片
        paper_matches = re.findall(r'\[PAPER_SEARCH:([^\]]+)\]', response)
        # [CANCEL_ACTION] → 取消待确认操作
        has_cancel = '[CANCEL_ACTION]' in response
        # [TRADE_UPDATE] → 成交截图触发持仓更新
        has_trade_update = '[TRADE_UPDATE]' in response

        # 清理动作标签 (不展示给用户)
        clean_text = re.sub(r'\[PAPER_SEARCH:[^\]]+\]', '', response)
        clean_text = clean_text.replace('[CANCEL_ACTION]', '')
        clean_text = clean_text.replace('[TRADE_UPDATE]', '')

        # 提取图片路径 [IMAGE:/path/to/image.png]
        image_pattern = r'\[IMAGE:([^\]]+)\]'
        image_matches = re.findall(image_pattern, clean_text)

        # 也检测常见路径格式
        path_pattern = r'[`\s](/[^\s`\n]+\.(?:png|jpg|jpeg|gif|webp))'
        path_matches = re.findall(path_pattern, clean_text, re.IGNORECASE)

        all_images = list(set(image_matches + path_matches))

        # 清理文本，移除 [IMAGE:] 标记
        clean_text = re.sub(image_pattern, '', clean_text).strip()

        # 发送文本（用 Markdown 卡片以支持格式化）
        if clean_text:
            # 超长文本分段发送 (飞书卡片限制 ~30KB)
            MAX_CHUNK = 4000
            chunks = self._split_text(clean_text, MAX_CHUNK) if len(clean_text) > MAX_CHUNK else [clean_text]
            for chunk in chunks:
                has_markdown = any(x in chunk for x in ['|', '```', '- ', '* ', '**', '##'])
                if has_markdown:
                    self.send_markdown(chunk, session=session)
                else:
                    self.send_text(chunk, session)

        # 发送图片
        for img_path in all_images:
            img_path = img_path.strip()
            if Path(img_path).exists():
                print(f"[发送图片] {img_path}")
                self.send_image(img_path, session)
            else:
                print(f"[图片不存在] {img_path}")

        # ── 执行动作 ──
        if has_cancel and session:
            session.clear_pending_actions()
            print("[动作] 已取消待确认操作")

        for url in paper_matches:
            url = url.strip()
            print(f"[动作] 论文调研: {url}")
            if not session:
                continue
            # 提取论文标识: arXiv ID 优先, 否则用 URL hash
            arxiv_m = re.search(r'(\d{4}\.\d{4,5})', url)
            if arxiv_m:
                paper_key = arxiv_m.group(1)
                full_url = f"https://arxiv.org/abs/{paper_key}" if 'arxiv.org' not in url else url
                desc = f"arXiv: **{paper_key}**"
            elif url.startswith('http'):
                # 非 arXiv 论文 (IJCAI/NeurIPS/其他)
                import hashlib
                paper_key = hashlib.md5(url.encode()).hexdigest()[:8]
                full_url = url
                desc = f"论文: **{url.split('/')[-1]}**"
            else:
                continue
            action_id = f"PAPER_{paper_key}_{int(time.time())}"
            session.add_pending_action(action_id, 'start_paper_research', {"url": full_url})
            self.send_confirm_card(
                "📄 论文调研",
                f"开始调研 {desc} ？\n\n将创建隔离分支，下载并阅读论文。",
                action_id, session
            )

        # [TRADE_UPDATE] → 解析成交截图并创建持仓更新确认卡片
        trade_in_flight = False
        if has_trade_update and session and session.last_images:
            valid_images = [p for p, t in session.last_images if Path(p).exists()]
            if valid_images:
                trade_in_flight = True
                session.last_images = []  # 防止同一批截图重复触发
                self.send_text("⏳ 正在解析成交截图...", session)
                threading.Thread(
                    target=self._handle_screenshot_trades,
                    args=("用户确认成交", valid_images, session),
                    daemon=True
                ).start()
            else:
                self.send_text("未找到最近的成交截图，请重新发送截图", session)
        elif has_trade_update and session:
            self.send_text("未找到最近的成交截图，请重新发送截图", session)

        return trade_in_flight

    def _handle_ml_signal(self, session: 'UserSession' = None):
        """生成 ML 信号并推送"""
        try:
            sys.path.insert(0, str(BOT_DIR))
            from factor_lab.signal_generator import SignalGenerator
            from portfolio.live_portfolio import (
                load_live_holdings, get_current_prices as get_live_prices,
                generate_live_instructions, get_portfolio_summary,
            )

            sg = SignalGenerator()
            signal = sg.get_signal()
            if 'error' in signal:
                self.send_text(f"❌ ML 信号错误: {signal['error']}", session)
                return

            holdings = load_live_holdings()
            all_instruments = list(set(
                list(holdings.get('positions', {}).keys()) + signal['target_stocks']
            ))
            prices, _from_cache = get_live_prices(all_instruments)
            if not prices:
                self.send_text("❌ 获取价格失败，请稍后重试", session)
                return

            message = generate_live_instructions(signal, holdings, prices)

            # 模型新鲜度
            freshness = sg.check_model_freshness()
            if freshness['is_stale']:
                message += f"\n\n⚠️ {freshness['message']}"

            self.send_markdown(message, title="ML调仓信号", session=session)
        except Exception as e:
            self.send_text(f"❌ 信号生成异常: {e}", session)
            import traceback
            traceback.print_exc()

    def _handle_screenshot_trades(self, text: str, image_paths, session: 'UserSession' = None):
        """截图解析持仓更新 (支持多张截图, 支持成交截图和持仓截图)

        成交截图: 直接提取 buy/sell trades
        持仓截图: 提取当前持仓 → 与 live_holdings 对比 → 生成 trades
        image_paths: 单张路径 (str) 或多张路径列表 (list)
        """
        # 兼容旧调用: str → list
        if isinstance(image_paths, str):
            image_paths = [image_paths]
        image_paths = [p for p in image_paths if Path(p).exists()]
        if not image_paths:
            self.send_text("❌ 截图文件不存在，请重新发送", session)
            return

        try:
            # 读取当前系统持仓，供对比
            from portfolio.live_portfolio import load_live_holdings
            live_h = load_live_holdings()
            current_positions = live_h.get('positions', {})
            pos_summary = json.dumps({code: {"shares": p["shares"], "cost": p.get("avg_cost", 0)}
                                      for code, p in current_positions.items()}, ensure_ascii=False) if current_positions else "{}"

            # 合并多张截图让 Claude 一次性解析
            read_cmds = "\n".join(f"请用 Read 工具查看图片 {p}" for p in image_paths)
            n_imgs = len(image_paths)

            prompt = f"""分析{'这' + str(n_imgs) + '张' if n_imgs > 1 else '这张'}股票截图，提取信息用于更新持仓系统。

用户说: {text}

截图可能是以下两种之一：
A) 成交记录/交割单 — 显示买入/卖出明细（成交价、成交量）
B) 持仓截图 — 显示当前持有的股票列表（持仓数量、成本价、现价）

【系统当前持仓】:
{pos_summary}

请按以下规则输出 JSON:

如果是成交记录 (A)，直接提取每笔成交:
{{"type": "trades", "trades": [
    {{"action": "buy", "code": "SH600036", "shares": 200, "price": 38.50, "name": "招商银行"}}
]}}

如果是持仓截图 (B)，提取截图中所有持仓，系统会自动与当前持仓对比:
{{"type": "holdings", "holdings": [
    {{"code": "SH600036", "shares": 200, "price": 38.50, "cost": 37.80, "name": "招商银行"}}
]}}

规则:
1. code 用 Qlib 格式: 沪市 SH + 6位数字, 深市 SZ + 6位数字
2. action: buy=买入, sell=卖出
3. shares: 持仓数量或成交数量 (股)
4. price: 现价或成交均价
5. cost: 成本价 (仅持仓截图需要)
6. 多张截图的内容请合并 (可能是同一个持仓列表分页截图)
7. 只输出一个 JSON，不要其他文字"""

            cmd = [
                CLAUDE_BIN, '--print', '--dangerously-skip-permissions',
                '-p', f"{read_cmds}\n\n{prompt}"
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, encoding='utf-8', errors='replace',
                timeout=120, cwd=str(WORK_DIR), env=_CLEAN_ENV
            )
            output = result.stdout.strip()

            json_match = re.search(r'\{.*\}', output, re.DOTALL)
            if not json_match:
                self.send_text("❌ 无法从截图中解析内容，请确认图片清晰", session)
                return

            data = json.loads(json_match.group())
            screenshot_type = data.get('type', 'trades')

            if screenshot_type == 'holdings':
                # 持仓截图 → 与当前持仓对比生成 trades
                all_trades = self._diff_holdings(current_positions, data.get('holdings', []))
            else:
                # 成交截图 → 直接使用
                all_trades = data.get('trades', [])

            if not all_trades:
                self.send_text("截图持仓与系统一致，无需更新", session)
                return

            # 构建确认内容
            lines = ["识别到以下变动:\n"]
            for t in all_trades:
                action_str = "买入" if t['action'] == 'buy' else "卖出"
                name = t.get('name', t['code'])
                lines.append(f"• {action_str} {name} {t['shares']}股 ×{t['price']}")

            action_id = f"TRADE_{int(time.time())}"
            session.add_pending_action(action_id, 'apply_trades', {'trades': all_trades})
            self.send_confirm_card(
                "📸 确认持仓更新",
                "\n".join(lines),
                action_id,
                session
            )
        except json.JSONDecodeError:
            self.send_text("❌ 解析截图失败，请重新发送", session)
        except subprocess.TimeoutExpired:
            self.send_text("⏳ 截图解析超时，请重试", session)
        except Exception as e:
            self.send_text(f"❌ 截图处理异常: {e}", session)
        finally:
            # 清理临时图片文件 → Redis
            for p in image_paths:
                if Path(p).exists():
                    self.tmp_store.store(p, session.open_id if session else "unknown")

    def _diff_holdings(self, current_positions: dict, screenshot_holdings: list) -> list:
        """对比系统持仓和截图持仓，生成 trades 列表"""
        trades = []
        screenshot_map = {}
        for h in screenshot_holdings:
            code = h['code']
            screenshot_map[code] = h

        # 截图中有但系统没有 → 买入
        # 截图中有且系统也有但数量不同 → 差额买入/卖出
        for code, h in screenshot_map.items():
            sys_shares = current_positions.get(code, {}).get('shares', 0)
            scr_shares = h['shares']
            price = h.get('cost', h.get('price', 0))  # 优先用成本价
            name = h.get('name', code)

            if scr_shares > sys_shares:
                trades.append({"action": "buy", "code": code, "shares": scr_shares - sys_shares, "price": price, "name": name})
            elif scr_shares < sys_shares:
                trades.append({"action": "sell", "code": code, "shares": sys_shares - scr_shares, "price": price, "name": name})

        # 系统有但截图中没有 → 卖出 (清仓)
        for code, pos in current_positions.items():
            if code not in screenshot_map:
                trades.append({"action": "sell", "code": code, "shares": pos['shares'], "price": pos.get('cost_price', pos.get('avg_cost', 0)), "name": pos.get('name', code)})

        return trades

    # ── 论文调研处理 ────────────────────────────────

    def _handle_paper_status(self, session):
        """查看论文调研状态"""
        try:
            sys.path.insert(0, str(BOT_DIR))
            from paper_researcher import PaperResearcher
            pr = PaperResearcher()
            status = pr.get_status()
            self.send_markdown(status, title="📄 论文调研", session=session)
        except Exception as e:
            self.send_text(f"❌ 论文状态查询失败: {e}", session)

    def _handle_paper_research(self, session, url: str):
        """启动论文调研: 下载 → 阅读 → 展示价值评估 → 用户决定是否继续"""
        try:
            sys.path.insert(0, str(BOT_DIR))
            from paper_researcher import PaperResearcher
            pr = PaperResearcher()
            paper_id = pr.add_paper(url)
            self.send_text(f"📄 论文 {paper_id} 已添加，开始阅读...", session)

            # Phase 1: 阅读 + 价值评估
            pr.run_phase(paper_id, "read")
            if pr.registry[paper_id].get("status") == "failed":
                self.send_text(f"❌ {paper_id} 阅读失败: {pr.registry[paper_id].get('progress_summary', '')}", session)
                return

            # 展示论文分析 (核心思想、方法、框架关联、价值评估)
            entry = pr.registry[paper_id]
            title = entry.get('title') or entry.get('arxiv_id', '?')
            summary = pr.get_read_summary(paper_id)
            self.send_markdown(summary, title=f"📄 {title}", session=session)

            # 用户决定: 继续规划+复现 or 仅对话探索
            action_id = f"PAPER_PLAN_{paper_id}_{int(time.time())}"
            session.add_pending_action(action_id, 'plan_paper', {"paper_id": paper_id})
            self.send_confirm_card(
                f"📄 是否继续复现?",
                f"**确认** → 生成复现计划并继续\n"
                f"**取消** → 仅保留分析，可随时发「论文 {paper_id}」进入对话深入讨论",
                action_id, session
            )
        except Exception as e:
            self.send_text(f"❌ 论文调研失败: {e}", session)

    def _handle_paper_plan_and_replicate(self, session, paper_id: str):
        """用户确认有价值后: 规划 → 确认复现"""
        try:
            sys.path.insert(0, str(BOT_DIR))
            from paper_researcher import PaperResearcher
            pr = PaperResearcher()
            self.send_text(f"📄 {paper_id} 开始生成复现计划...", session)

            pr.run_phase(paper_id, "plan")
            if pr.registry[paper_id].get("status") == "failed":
                self.send_text(f"❌ {paper_id} 规划失败: {pr.registry[paper_id].get('progress_summary', '')}", session)
                return
            self.send_markdown(pr.get_status(paper_id), title=f"📄 {paper_id} 计划就绪", session=session)

            # 复现需要二次确认
            action_id = f"PAPER_REPLICATE_{paper_id}_{int(time.time())}"
            session.add_pending_action(action_id, 'continue_paper', {"paper_id": paper_id})
            worktree = pr.registry[paper_id].get('worktree_path', '')
            self.send_confirm_card(
                f"📄 {paper_id} 开始复现?",
                f"复现计划已就绪，是否开始自动复现？\n\n"
                f"查看详情: `cat {worktree}/PROGRESS.md`",
                action_id, session
            )
        except Exception as e:
            self.send_text(f"❌ 论文规划失败: {e}", session)

    def _handle_paper_search(self, session, query: str):
        """搜索 arXiv 并展示结果"""
        try:
            sys.path.insert(0, str(BOT_DIR))
            from paper_researcher import search_arxiv
            # 清理搜索词: 去除引号和多余空格
            clean_query = query.strip().strip('"').strip("'").strip('"').strip('"').strip()
            self.send_text(f"📄 正在搜索 arXiv: {clean_query}...", session)
            results = search_arxiv(clean_query, max_results=3)
            if not results:
                self.send_text(f"📄 未在 arXiv 找到匹配论文: {query}", session)
                return

            lines = ["📄 arXiv 搜索结果", ""]
            for i, p in enumerate(results, 1):
                lines.append(f"**{i}. {p['title']}**")
                lines.append(f"   arXiv: {p['arxiv_id']} | {p['date']} | {p['authors']}")
                lines.append(f"   {p['summary'][:150]}...")
                lines.append("")

            # 用第一个结果创建确认卡片
            top = results[0]
            action_id = f"PAPER_{top['arxiv_id']}_{int(time.time())}"
            session.add_pending_action(action_id, 'start_paper_research', {"url": top['url']})

            lines.append(f"---\n点击确认将调研排名第一的论文: **{top['title']}**")
            self.send_confirm_card(
                "📄 论文调研",
                "\n".join(lines),
                action_id, session
            )
        except Exception as e:
            self.send_text(f"❌ arXiv 搜索失败: {e}", session)

    def _handle_paper_chat(self, session, paper_id: str, message: str):
        """与论文助手交互对话"""
        try:
            sys.path.insert(0, str(BOT_DIR))
            from paper_researcher import PaperResearcher
            pr = PaperResearcher()
            if paper_id not in pr.registry:
                self.send_text(f"❌ 未找到 {paper_id}，发「论文」查看列表", session)
                session.paper_context = None
                return
            result = pr.chat(paper_id, message)
            title = pr.registry[paper_id].get("title") or paper_id
            self.send_markdown(result, title=f"📄 {title}", session=session)
        except Exception as e:
            self.send_text(f"❌ 论文对话失败: {e}", session)

    def _handle_paper_replicate(self, session, paper_id: str):
        """执行论文复现 + 评估"""
        try:
            sys.path.insert(0, str(BOT_DIR))
            from paper_researcher import PaperResearcher
            pr = PaperResearcher()
            self.send_text(f"📄 {paper_id} 开始复现...", session)

            pr.run_phase(paper_id, "replicate")
            self.send_markdown(pr.get_status(paper_id), title=f"📄 {paper_id} 复现进度", session=session)

            pr.run_phase(paper_id, "evaluate")
            self.send_markdown(pr.get_status(paper_id), title=f"📄 {paper_id} 评估完成", session=session)
        except Exception as e:
            self.send_text(f"❌ 论文复现失败: {e}", session)

    def handle_quick_commands(self, text: str, session: 'UserSession' = None, image_path: str = None) -> bool:
        """处理快捷命令，返回是否已处理"""
        # 简单问候 → 主动功能引导（不走 Claude）
        greeting_keywords = {'你好', '在吗', 'hi', 'hello', '嗨', '早上好', '下午好', '晚上好', '早'}
        if text.strip().lower() in greeting_keywords:
            self.send_text(
                "你好！我可以帮你：\n"
                "• 「信号」查看今日调仓信号\n"
                "• 「持仓」查看实盘持仓\n"
                "• 「监控」盘中监控状态\n"
                "• 「挖掘」因子挖掘进展\n"
                "• 「影子」影子交易验证\n"
                "• 「帮助」查看全部命令\n"
                "或直接输入任何问题，我来帮你分析 😊",
                session
            )
            return True

        # 帮助
        if text in ['帮助', 'help', '?']:
            self.send_text(
                "📖 命令:\n"
                "• 信号/调仓 - ML调仓信号\n"
                "• 持仓 - 查看实盘持仓\n"
                "• 监控 - 盘中监控状态\n"
                "• 挖掘 - 因子挖掘状态\n"
                "• 影子 - 影子交易验证状态\n"
                "• 实验盘 - 实验盘状态\n"
                "• 重训 - 季度模型重训\n"
                "• 追加5万 - 追加资金(下次调仓分配)\n"
                "• 清仓 - 清空所有持仓\n"
                "• 资金10万 - 设置初始资金(需先清仓)\n"
                "• 历史文件 - 查看历史图片/文件\n"
                "• 反思 - 查看每日自我反思记录\n"
                "• 论文 - 论文调研状态\n"
                "• 论文 paper_001 - 进入论文对话\n"
                "• @paper_001 <问题> - 一次性提问\n"
                "• 退出论文 - 退出论文对话\n"
                "• 发截图+「已执行」- 更新持仓\n"
                "• 发 arXiv 链接 - 启动论文调研",
                session
            )
            return True

        # 历史文件列表
        if text in ['历史文件', '历史图片', '文件列表']:
            files, total = self.tmp_store.list_files(session.open_id, limit=10)
            if not files:
                self.send_text("📂 暂无历史文件", session)
            else:
                import datetime as _dt
                lines = [f"📂 最近文件（共{total}个，显示最新10个）:"]
                for f in files:
                    ts = _dt.datetime.fromtimestamp(f.get("ts", 0)).strftime("%m-%d %H:%M")
                    days = f.get("ttl_remaining", 0) // 86400
                    size_kb = f.get("size", 0) / 1024
                    lines.append(f"  {f['file_id']}  {size_kb:.0f}KB  {ts}  剩{days}天")
                if total > 10:
                    lines.append(f"\n还有 {total - 10} 个更早的文件，发「查看更多」可继续")
                self.send_text("\n".join(lines), session)
            return True

        # 盘中监控状态
        if text in ['监控', 'monitor']:
            try:
                state_file = BOT_DIR / "monitor" / "monitor_state.json"
                if not state_file.exists():
                    self.send_text("📡 盘中监控: 今日未启动\n(launchd 每天 9:25 自动启动)", session)
                else:
                    with open(state_file, 'r', encoding='utf-8') as f:
                        state = json.load(f)
                    from datetime import date as _date
                    today = _date.today().isoformat()
                    if state.get('date') != today:
                        self.send_text("📡 盘中监控: 今日未启动\n(launchd 每天 9:25 自动启动)", session)
                    else:
                        alerted = state.get('alerted_today', {})
                        lines = [
                            "📡 盘中监控状态",
                            "",
                            f"启动时间: {state.get('started_at', '?')}",
                            f"检查次数: {state.get('check_count', 0)}",
                            f"最近检查: {state.get('last_check', '?')}",
                            f"已触发告警: {len(alerted)} 条",
                        ]
                        if alerted:
                            lines.append("")
                            for key in alerted:
                                lines.append(f"  • {key}")
                        self.send_text("\n".join(lines), session)
            except Exception as e:
                self.send_text(f"❌ 监控状态查询失败: {e}", session)
            return True

        # 反思记录查看
        if text in ['反思', 'reflect', '反思记录']:
            try:
                reflect_dir = BOT_DIR / "reflections"
                index_file = reflect_dir / "index.json"
                if not index_file.exists():
                    self.send_text("📝 暂无反思记录\n(每晚 23:30 自动执行)", session)
                else:
                    with open(index_file, 'r', encoding='utf-8') as f:
                        index = json.load(f)
                    if not index:
                        self.send_text("📝 暂无反思记录", session)
                    else:
                        # 显示最近5条
                        recent = sorted(index.items(), reverse=True)[:5]
                        lines = ["📝 最近反思记录", ""]
                        for dt, info in recent:
                            score = info.get('score', 0)
                            summary = info.get('summary', '')
                            fixes = info.get('auto_fixes_count', 0)
                            score_bar = '★' * score + '☆' * (10 - score)
                            lines.append(f"**{dt}** {score_bar}")
                            lines.append(f"  {summary}")
                            if fixes > 0:
                                lines.append(f"  🔧 自动修复: {fixes} 项")
                            lines.append("")
                        avg_score = sum(v.get('score', 0) for v in index.values()) / len(index) if index else 0
                        lines.append(f"平均评分: {avg_score:.1f}/10 (共 {len(index)} 天)")
                        self.send_markdown("\n".join(lines), session=session)
            except Exception as e:
                self.send_text(f"❌ 反思记录查询失败: {e}", session)
            return True

        # 因子挖掘状态
        if text in ['挖掘', 'mining']:
            try:
                mining_index = BOT_DIR / "factor_lab" / "mining_results" / "index.json"
                if not mining_index.exists():
                    self.send_text("🔬 因子挖掘: 暂无运行记录\n(每周日 20:00 自动执行)", session)
                else:
                    with open(mining_index, 'r', encoding='utf-8') as f:
                        index = json.load(f)
                    if not index:
                        self.send_text("🔬 因子挖掘: 暂无运行记录", session)
                    else:
                        total_runs = len(index)
                        total_tested = sum(r.get("factors_tested", 0) for r in index.values())
                        total_promising = sum(r.get("promising_count", 0) for r in index.values())
                        total_beat = sum(1 for r in index.values() if r.get("beat_baseline"))
                        recent = sorted(index.items(), key=lambda x: x[1].get("date", ""), reverse=True)[:5]
                        lines = [
                            "🔬 因子挖掘状态",
                            "",
                            f"总运行: {total_runs} 次 | 测试因子: {total_tested} | 有效: {total_promising} | 超越M01: {total_beat}",
                            "",
                            "最近运行:",
                        ]
                        for rid, info in recent:
                            beat = "✅" if info.get("beat_baseline") else "  "
                            lines.append(f"  {beat} {rid} ({info.get('date', '?')}): "
                                         f"测试{info.get('factors_tested', 0)}, "
                                         f"有效{info.get('promising_count', 0)}")
                        self.send_text("\n".join(lines), session)
            except Exception as e:
                self.send_text(f"❌ 挖掘状态查询失败: {e}", session)
            return True

        # 影子交易状态
        if text in ['影子', 'shadow']:
            try:
                sys.path.insert(0, str(BOT_DIR))
                from shadow_manager import ShadowManager
                sm = ShadowManager()
                status = sm.get_status_text()
                self.send_text(f"🧪 {status}", session)
            except Exception as e:
                self.send_text(f"❌ 影子验证查询失败: {e}", session)
            return True

        # 实验盘状态
        if text in ['实验盘', '实验', 'experiment']:
            try:
                sys.path.insert(0, str(BOT_DIR))
                from experiment_manager import ExperimentManager
                em = ExperimentManager()
                status = em.get_status_text()
                self.send_text(f"🔬 {status}", session)
            except Exception as e:
                self.send_text(f"❌ 实验盘查询失败: {e}", session)
            return True

        # 持仓查询 — ML 实盘持仓
        if text in ['持仓', '仓位', 'positions']:
            try:
                from portfolio.live_portfolio import (
                    load_live_holdings, get_current_prices as get_live_prices,
                    get_portfolio_summary,
                )
                holdings = load_live_holdings()
                positions = holdings.get('positions', {})
                if positions:
                    prices, _ = get_live_prices(list(positions.keys()))
                else:
                    prices = {}
                summary = get_portfolio_summary(holdings, prices)
                self.send_markdown(summary, session=session)
            except Exception as e:
                self.send_text(f"❌ 持仓查询失败: {e}", session)
            return True

        # 生成 ML 信号
        if text in ['信号', '调仓', 'signal']:
            self.send_text("⏳ 正在生成 ML 信号...", session)
            threading.Thread(
                target=self._handle_ml_signal,
                args=(session,),
                daemon=True
            ).start()
            return True

        # 季度重训
        if text in ['重训', 'retrain']:
            action_id = f"RETRAIN_{int(time.time())}"
            session.add_pending_action(action_id, 'execute_command', {
                "command": "python trading_framework/retrain_pipeline.py"
            })
            self.send_confirm_card(
                "🔄 季度模型重训",
                "将执行完整重训 pipeline:\n1. 刷新 BaoStock 数据 (~3min)\n2. 扩展 rolling 预测 (~2min)\n3. 重算信号质量\n4. 验证\n\n预计耗时 5-8 分钟",
                action_id,
                session
            )
            return True

        # 截图 + 成交确认 (支持 session 最近 5 分钟内的图片)
        all_images = []
        if image_path:
            all_images = [image_path]
        if session and session.last_images:
            all_images = list(dict.fromkeys(  # 去重保序
                [image_path] + [p for p, _ in session.last_images] if image_path
                else [p for p, _ in session.last_images]
            ))
        if all_images and any(kw in text for kw in ['已执行', '成交', '已买入', '已卖出', '已买', '已卖']):
            self.send_text("⏳ 正在解析成交截图...", session)
            if session:
                session.last_images = []  # 防止重复触发
            threading.Thread(
                target=self._handle_screenshot_trades,
                args=(text, all_images, session),
                daemon=True
            ).start()
            return True

        # 清仓 / 清空持仓
        if text in ['清仓', '清空持仓', '清空']:
            from portfolio.live_portfolio import load_live_holdings
            live_h = load_live_holdings()
            n_pos = len(live_h.get('positions', {}))
            cash = live_h.get('initial_capital', 100000)

            action_id = f"CLR_{int(time.time())}"
            session.add_pending_action(action_id, 'clear_positions', {})
            if n_pos > 0:
                self.send_confirm_card(
                    "⚠️ 清空持仓",
                    f"将清空 **{n_pos}** 只持仓，现金重置为 **{cash:,.0f}元**\n确认清仓？",
                    action_id,
                    session
                )
            else:
                self.send_text("📋 当前已是空仓状态", session)
            return True

        # 追加资金
        add_capital_match = re.search(r'(?:追加|增加)\s*(?:资金)?\s*(\d+(?:\.\d+)?)\s*([万元])?', text)
        if add_capital_match:
            amount = float(add_capital_match.group(1))
            unit = add_capital_match.group(2)
            if unit == '万':
                amount *= 10000

            from portfolio.live_portfolio import load_live_holdings
            live_h = load_live_holdings()
            old_cash = live_h.get('cash', 0)
            old_capital = live_h.get('initial_capital', 0)

            action_id = f"ADD_{int(time.time())}"
            session.add_pending_action(action_id, 'add_capital', {'amount': amount})
            self.send_confirm_card(
                "💰 追加资金",
                f"追加 **{amount:,.0f}元**\n"
                f"现金: {old_cash:,.0f} → {old_cash + amount:,.0f}\n"
                f"总资金: {old_capital:,.0f} → {old_capital + amount:,.0f}\n"
                f"持仓不变，下次调仓日自动分配",
                action_id,
                session
            )
            return True

        # 设置资金 (有持仓时拒绝)
        capital_match = re.search(r'(?:资金|总资金|本金)[是为]?\s*(\d+(?:\.\d+)?)\s*([万元])?', text)
        if capital_match:
            amount = float(capital_match.group(1))
            unit = capital_match.group(2)
            if unit == '万':
                amount *= 10000

            from portfolio.live_portfolio import load_live_holdings
            live_h = load_live_holdings()

            # 有持仓时拒绝，引导用"追加"或"清仓"
            if live_h.get('positions'):
                n_pos = len(live_h['positions'])
                self.send_text(
                    f"⚠️ 当前有 {n_pos} 只持仓，不能直接设置资金\n\n"
                    f"请先选择:\n"
                    f"• 发送「追加X万」— 追加资金，持仓不变\n"
                    f"• 发送「清仓」— 清空持仓后再设置",
                    session
                )
                return True

            action_id = f"CAP_{int(time.time())}"
            session.add_pending_action(action_id, 'set_capital', {'amount': amount})
            self.send_confirm_card(
                "📋 设置初始资金",
                f"设置初始资金为 **{amount:,.0f}元**",
                action_id,
                session
            )
            return True

        # --- 论文调研 ---

        # 退出论文对话模式
        if text in ['退出论文', '返回', 'exit paper'] and session.paper_context:
            pid = session.paper_context
            session.paper_context = None
            self.send_text(f"已退出 {pid} 对话模式", session)
            return True

        # 论文状态总览
        if text in ['论文', '调研', 'papers']:
            if session.paper_context:
                self.send_text(f"当前在 {session.paper_context} 对话模式中\n发「退出论文」返回", session)
            threading.Thread(target=self._handle_paper_status, args=(session,), daemon=True).start()
            return True

        # 进入论文对话模式: "论文 paper_001" / "对话 paper_001"
        paper_ctx_match = re.match(r'(?:论文|对话|paper)\s+(paper_\d{3})$', text)
        if paper_ctx_match:
            paper_id = paper_ctx_match.group(1)
            session.paper_context = paper_id
            self.send_text(
                f"📄 已进入 **{paper_id}** 对话模式\n\n"
                f"直接发消息即可与论文助手对话，日常命令(持仓/信号等)仍可正常使用。\n"
                f"发「退出论文」返回。",
                session
            )
            return True

        # 一次性提问: "@paper_001 这个因子怎么算的?"
        oneshot_match = re.match(r'@(paper_\d{3})\s+(.+)', text, re.DOTALL)
        if oneshot_match:
            paper_id = oneshot_match.group(1)
            message = oneshot_match.group(2).strip()
            self.send_text(f"📄 正在向 {paper_id} 提问...", session)
            threading.Thread(
                target=self._handle_paper_chat, args=(session, paper_id, message),
                daemon=True
            ).start()
            return True

        # arXiv URL 自动检测
        arxiv_match = re.search(r'arxiv\.org/(?:abs|pdf|html)/(\d{4}\.\d{4,5})', text)
        if arxiv_match:
            arxiv_id = arxiv_match.group(1)
            action_id = f"PAPER_{arxiv_id}_{int(time.time())}"
            session.add_pending_action(action_id, 'start_paper_research', {"url": text.strip()})
            self.send_confirm_card(
                "📄 论文调研",
                f"检测到 arXiv 论文: **{arxiv_id}**\n\n将创建隔离 worktree 分支进行调研与复现。",
                action_id, session
            )
            return True

        # 文字确认/取消
        pending = session.get_pending_actions()
        if text in ['允许', 'y', 'yes', '确认'] and pending:
            action_id = list(pending.keys())[-1]
            action = session.pop_pending_action(action_id)
            if action:
                self.send_text("✅ 正在执行...", session)
                self.execute_action(action, session)
            return True

        if text in ['取消', 'n', 'no', '不'] and pending:
            session.clear_pending_actions()
            self.send_text("❌ 已取消", session)
            return True

        # --- 论文对话模式: 所有未匹配命令转发给论文助手 ---
        if session.paper_context:
            self.send_text(f"📄 [{session.paper_context}] 思考中...", session)
            threading.Thread(
                target=self._handle_paper_chat,
                args=(session, session.paper_context, text),
                daemon=True
            ).start()
            return True

        return False

    def execute_action(self, action: dict, session: 'UserSession' = None):
        """执行动作"""
        action_type = action.get('type', '')
        data = action.get('data', {})

        if action_type == 'execute_command':
            cmd = data.get('command', '')
            if cmd:
                self.execute_command(cmd, session)

        elif action_type == 'clear_positions':
            try:
                from portfolio.live_portfolio import load_live_holdings, clear_positions
                live_h = load_live_holdings()
                clear_positions(live_h)
                cash = live_h.get('cash', 0)
                self.send_text(f"✅ 已清空持仓\n现金: {cash:,.0f}元", session)
            except Exception as e:
                self.send_text(f"❌ 清仓失败: {e}", session)

        elif action_type == 'add_capital':
            try:
                from portfolio.live_portfolio import load_live_holdings, add_capital
                amount = data.get('amount', 0)
                live_h = load_live_holdings()
                add_capital(live_h, amount)
                self.send_text(
                    f"✅ 已追加 {amount:,.0f}元\n"
                    f"现金: {live_h['cash']:,.0f} | 总资金: {live_h['initial_capital']:,.0f}\n"
                    f"下次调仓日自动分配",
                    session
                )
            except Exception as e:
                self.send_text(f"❌ 追加资金失败: {e}", session)

        elif action_type == 'set_capital':
            try:
                from portfolio.live_portfolio import load_live_holdings, save_live_holdings
                amount = data.get('amount', 0)
                live_h = load_live_holdings()
                live_h['initial_capital'] = amount
                live_h['cash'] = amount
                live_h['positions'] = {}
                save_live_holdings(live_h)
                self.send_text(f"✅ 初始资金已设置为 {amount:,.0f}元", session)
            except Exception as e:
                self.send_text(f"❌ 设置失败: {e}", session)

        elif action_type == 'update_holdings':
            try:
                holdings_file = WORK_DIR / "portfolio" / "holdings.json"
                with open(holdings_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                self.send_text(f"✅ 持仓已更新\n总资产: {data.get('total_capital', 0):,.0f}元", session)
            except Exception as e:
                self.send_text(f"❌ 更新失败: {e}", session)

        elif action_type == 'write_file':
            path = data.get('path', '')
            content = data.get('content', '')
            try:
                # 处理相对路径
                if not path.startswith('/'):
                    path = str(WORK_DIR / path)
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(content)
                self.send_text(f"✅ 已写入: {path}", session)
            except Exception as e:
                self.send_text(f"❌ 写入失败: {e}", session)

        elif action_type == 'edit_file':
            path = data.get('path', '')
            old_text = data.get('old', '')
            new_text = data.get('new', '')
            try:
                if not path.startswith('/'):
                    path = str(WORK_DIR / path)
                with open(path, 'r', encoding='utf-8') as f:
                    file_content = f.read()
                if old_text not in file_content:
                    self.send_text(f"❌ 未找到要替换的文本", session)
                    return
                file_content = file_content.replace(old_text, new_text, 1)
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(file_content)
                self.send_text(f"✅ 已修改: {path}", session)
            except Exception as e:
                self.send_text(f"❌ 修改失败: {e}", session)

        elif action_type == 'apply_trades':
            try:
                from portfolio.live_portfolio import apply_trades
                trades = data.get('trades', [])
                result = apply_trades(trades)
                self.send_text(f"✅ 持仓已更新:\n{result}", session)
            except Exception as e:
                self.send_text(f"❌ 持仓更新失败: {e}", session)

        elif action_type == 'send_image':
            path = data.get('path', '')
            self.send_image(path, session)

        elif action_type == 'start_paper_research':
            url = data.get("url", "")
            threading.Thread(target=self._handle_paper_research, args=(session, url), daemon=True).start()

        elif action_type == 'plan_paper':
            paper_id = data["paper_id"]
            threading.Thread(target=self._handle_paper_plan_and_replicate, args=(session, paper_id), daemon=True).start()

        elif action_type == 'continue_paper':
            paper_id = data["paper_id"]
            threading.Thread(target=self._handle_paper_replicate, args=(session, paper_id), daemon=True).start()

        # search_paper 已由 Claude 通用对话中的上下文注入处理，无需单独 action type

    def _filter_pending(self, messages: List[str]) -> List[str]:
        """用 Claude 判断待处理消息中哪些该执行、哪些已被撤回"""
        if len(messages) <= 1:
            return messages

        numbered = "\n".join(f"{i+1}. {m}" for i, m in enumerate(messages))
        print(f"[过滤] 待处理 {len(messages)} 条消息，调用 Claude 判断...")

        prompt = f"""以下是用户在排队期间依次发送的消息：
{numbered}

用户可能在后面的消息中撤回/取消了前面的某些请求。
请判断哪些消息应该实际执行（跳过被撤回的任务和纯取消指令本身）。
execute 数组中填入应执行的消息编号（从1开始），如果全部不需要执行则返回空数组。"""

        schema = json.dumps({
            "type": "object",
            "properties": {
                "execute": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "应执行的消息编号列表，从1开始"
                },
                "reason": {"type": "string", "description": "一句话判断理由"}
            },
            "required": ["execute", "reason"]
        })

        try:
            result = subprocess.run(
                [CLAUDE_BIN, '--print', '--dangerously-skip-permissions',
                 '--output-format', 'json', '--json-schema', schema,
                 '-p', prompt],
                capture_output=True, text=True, encoding='utf-8', errors='replace',
                timeout=15, env=_CLEAN_ENV
            )
            output = result.stdout.strip()
            data = json.loads(output)
            # 解析 --output-format json 的嵌套结构
            if isinstance(data, dict) and "result" in data:
                r = data["result"]
                result_data = json.loads(r) if isinstance(r, str) else r
            elif isinstance(data, dict) and "execute" in data:
                result_data = data
            else:
                raise ValueError(f"未知结构: {data}")

            execute = result_data.get("execute", [])
            reason = result_data.get("reason", "")
            print(f"[过滤] Claude 返回: execute={execute}, reason={reason}")

            indices = [n - 1 for n in execute if 1 <= n <= len(messages)]
            filtered = [messages[i] for i in indices]
            skipped = [m for i, m in enumerate(messages) if i not in indices]
            for s in skipped:
                print(f"[跳过] 被撤回: {s[:40]}")
            print(f"[过滤] 保留 {len(filtered)}/{len(messages)} 条")
            return filtered
        except Exception as e:
            print(f"[过滤失败] {e}，全部执行")
            return messages

    def process_message(self, text: str, session: 'UserSession' = None, msg_id: str = None, image_path: str = None):
        """处理用户消息"""
        with session.processing_lock:
            if session.is_processing:
                return
            session.is_processing = True

        try:
            session.current_task = text[:30]

            # 存储图片到 session (跨消息关联, 5分钟窗口)
            now = time.time()
            if image_path:
                session.last_images.append((image_path, now))
            # 清理过期图片 (>5分钟)
            session.last_images = [(p, t) for p, t in session.last_images if now - t < 300]

            # 快捷命令
            if self.handle_quick_commands(text, session, image_path=image_path):
                return

            # fallback: 使用 session 中最近 5 分钟内的图片
            effective_image = image_path
            if not effective_image and session and session.last_images:
                effective_image = session.last_images[-1][0]  # 最新一张传给 Claude

            # 给消息加"收到"表情
            if msg_id:
                self.add_reaction(msg_id, "OK", session)
            response = self.call_claude(text, session, effective_image)
            trade_in_flight = self.process_response(response, session)

            # 处理完毕后将临时图片存入 Redis 并删除磁盘文件
            # 如果有 TRADE_UPDATE 后台线程在使用图片，跳过删除
            if image_path and Path(image_path).exists() and not trade_in_flight:
                self.tmp_store.store(image_path, session.open_id)

            # 保存结果供后续任务参考
            session.last_result = response[:500]

        finally:
            with session.processing_lock:
                session.is_processing = False
            session.current_task = ""

    def _is_authorized(self, open_id: str, source: str) -> bool:
        """白名单校验 — 未授权用户不得触发 Claude 执行

        白名单为空时拒绝所有人，并打印 open_id 便于首次配置。
        """
        if open_id in ALLOWED_OPEN_IDS:
            return True

        if not ALLOWED_OPEN_IDS:
            print(f"\n⚠ [{source}] 未配置白名单，已拒绝。若这是你本人，请在 .env 中设置:")
            print(f"    FEISHU_ALLOWED_OPEN_IDS={open_id}")
            print("  然后重启机器人。\n")
        else:
            print(f"⚠ [{source}] 拒绝未授权用户: {open_id}")
        return False

    def _handle_message(self, data, lark_client: lark.Client):
        """消息事件处理（由各 App 的 handler 闭包调用）"""
        try:
            msg = data.event.message
            msg_id = msg.message_id

            # 去重（Redis 优先，内存降级）
            if self.tmp_store.available:
                if self.tmp_store.is_msg_processed(msg_id):
                    return
                self.tmp_store.mark_msg_processed(msg_id)
            else:
                if msg_id in self.processed_msg_ids:
                    return
                self.processed_msg_ids.add(msg_id)
                if len(self.processed_msg_ids) > 1000:
                    self.processed_msg_ids = set(list(self.processed_msg_ids)[-500:])

            # 获取用户ID，创建/获取会话，注入 lark_client
            open_id = data.event.sender.sender_id.open_id
            if not self._is_authorized(open_id, "消息"):
                return
            session = self.session_mgr.get_or_create(open_id)
            session.lark_client = lark_client

            # 解析消息
            msg_type = msg.message_type
            content = json.loads(msg.content)
            text = ""
            image_path = None

            if msg_type == "text":
                text = content.get("text", "").strip()
            elif msg_type == "image":
                # 下载图片
                image_key = content.get("image_key", "")
                if image_key:
                    image_path = self.download_image(msg_id, image_key, session)
                    text = f"[用户发送了图片: {image_path}]" if image_path else "[图片下载失败]"
            elif msg_type == "post":
                # 富文本消息，提取文本内容
                post_content = content.get("content", [])
                texts = []
                for line in post_content:
                    for item in line:
                        if item.get("tag") == "text":
                            texts.append(item.get("text", ""))
                        elif item.get("tag") == "img":
                            image_key = item.get("image_key", "")
                            if image_key:
                                img_path = self.download_image(msg_id, image_key, session)
                                if img_path:
                                    image_path = img_path
                                    texts.append(f"[图片: {img_path}]")
                text = "".join(texts).strip()

            # 处理引用消息
            if hasattr(msg, 'parent_id') and msg.parent_id:
                quoted_text, quoted_image = self.get_message_content(msg.parent_id, session)
                if quoted_text:
                    text = f"[引用: {quoted_text}]\n{text}"
                if quoted_image and not image_path:
                    image_path = quoted_image
                    print(f"[引用图片] {quoted_image}")

            if not text:
                return

            print(f"\n[收到] [{open_id[:12]}] {msg_type}: {text[:100]}")

            # 检测取消命令
            if text in ['取消', '停止', 'cancel', 'stop'] and session.current_process:
                print("[取消] 终止当前任务")
                session.current_process.kill()
                session.current_process = None
                session.pending_messages.clear()
                self.add_reaction(msg_id, "OK", session)
                self.send_text("⏹️ 已取消当前任务", session)
                with session.processing_lock:
                    session.is_processing = False
                return

            # 如果正在处理，保存消息到待处理队列
            with session.processing_lock:
                if session.is_processing:
                    session.pending_messages.append(text)
                    self.send_text(f"⏳ 收到，稍后处理（当前任务: {session.current_task[:20]}...）", session)
                    print(f"[队列] 消息已暂存，待当前任务完成后处理: {text[:30]}")
                    return

            # 添加到队列
            user_msg = Message(role="user", content=text, msg_id=msg_id, image_path=image_path or "")
            session.msg_queue.add(user_msg)

            # 如果正在等待回复，不主动处理
            if session.msg_queue.is_waiting:
                print("[队列] 正在等待回复，消息已加入队列")
                return

            # 等待收集批量消息
            time.sleep(MESSAGE_BATCH_WAIT)

            # 收集所有消息
            batch = session.msg_queue.collect(wait_time=0.3)
            if not batch:
                return

            combined_text = " ".join([m.content for m in batch])
            last_msg_id = batch[-1].msg_id if batch else None
            # 收集 batch 中所有图片 → 注入 session (多截图支持)
            batch_images = [m.image_path for m in batch if m.image_path]
            if len(batch_images) > 1:
                now = time.time()
                for img in batch_images[:-1]:  # 最后一张由 process_message 存入
                    session.last_images.append((img, now))
            last_image = batch_images[-1] if batch_images else None
            print(f"[处理] [{open_id[:12]}] {combined_text[:50]}{'...' if len(combined_text) > 50 else ''}")

            # 异步处理（含 pending 调度）
            def _dispatch(text, sess, mid, img):
                self.process_message(text, sess, mid, img)
                # 任务完成后，处理排队期间积累的 pending 消息
                while sess.pending_messages:
                    pending = sess.pending_messages.copy()
                    sess.pending_messages.clear()
                    tasks = self._filter_pending(pending)
                    for t in tasks:
                        print(f"[继续处理] {t[:50]}")
                        self.process_message(t, sess)
            threading.Thread(target=_dispatch, args=(combined_text, session, last_msg_id, last_image), daemon=True).start()

        except Exception as e:
            print(f"[消息处理错误] {e}")
            import traceback
            traceback.print_exc()

    def _handle_card_action(self, data, lark_client: lark.Client):
        """卡片动作处理（由各 App 的 handler 闭包调用）"""
        try:
            value = data.event.action.value
            action = value.get('action', '')
            action_id = value.get('id', '')

            open_id = data.event.operator.open_id
            if not self._is_authorized(open_id, "按钮"):
                return
            session = self.session_mgr.get_or_create(open_id)
            session.lark_client = lark_client
            print(f"\n[按钮] [{open_id[:12]}] {action}, ID: {action_id}")

            if action == 'allow':
                pending_action = session.pop_pending_action(action_id)
                print(f"[查找action] ID={action_id}, 结果={pending_action is not None}")
                if pending_action:
                    print(f"[执行] 类型={pending_action.get('type')}")
                    # 避免 lambda 闭包问题，使用命名函数
                    def do_execute(act, sess):
                        print(f"[线程开始] 执行action")
                        self.send_text("✅ 正在执行...", sess)
                        self.execute_action(act, sess)
                        print(f"[线程结束] 执行完成")
                    threading.Thread(target=do_execute, args=(pending_action, session), daemon=True).start()
                    return P2CardActionTriggerResponse({
                        "toast": {"type": "success", "content": "已确认，执行中..."}
                    })
                else:
                    print(f"[警告] action未找到: {action_id}")
                    return P2CardActionTriggerResponse({
                        "toast": {"type": "warning", "content": "操作已过期"}
                    })

            elif action == 'cancel':
                session.pop_pending_action(action_id)
                def do_cancel(sess):
                    self.send_text("❌ 已取消", sess)
                threading.Thread(target=do_cancel, args=(session,), daemon=True).start()
                return P2CardActionTriggerResponse({
                    "toast": {"type": "info", "content": "已取消"}
                })

            return P2CardActionTriggerResponse({})

        except Exception as e:
            print(f"[卡片处理错误] {e}")
            return P2CardActionTriggerResponse({"toast": {"type": "error", "content": str(e)[:50]}})

    def _record_disconnect(self, app_id: str):
        """记录断连事件，10分钟内>=3次触发告警并缩短心跳间隔"""
        now = time.time()
        self._ws_disconnect_times.append(now)
        # 清理10分钟前的记录
        cutoff = now - 600
        while self._ws_disconnect_times and self._ws_disconnect_times[0] < cutoff:
            self._ws_disconnect_times.popleft()

        recent = len(self._ws_disconnect_times)
        print(f"[WS] App {app_id} 断连 (10min内 {recent} 次)")

        # 频繁断连 → 缩短心跳间隔 (120s → 30s)
        if recent >= 3:
            self._set_ping_interval(30)
            if now > self._ws_disconnect_alert_cd:
                self._ws_disconnect_alert_cd = now + 600  # 10分钟冷却
                self._send_disconnect_alert(recent)

    def _set_ping_interval(self, interval: int):
        """调整所有 WebSocket 客户端的 ping 间隔"""
        for ws_client in self._ws_clients:
            old = getattr(ws_client, '_ping_interval', self._default_ping_interval)
            if old != interval:
                ws_client._ping_interval = interval
                print(f"[WS] ping 间隔: {old}s → {interval}s")

    def _maybe_restore_ping_interval(self):
        """30分钟内无断连 → 恢复默认心跳间隔"""
        if not self._ws_disconnect_times:
            return
        now = time.time()
        latest = self._ws_disconnect_times[-1]
        if now - latest > 1800:  # 30分钟无断连
            self._set_ping_interval(self._default_ping_interval)
            self._ws_disconnect_times.clear()

    def _send_disconnect_alert(self, count: int):
        """WebSocket 断连频繁时，通过飞书告警通知"""
        try:
            app_id = APPS[0]["app_id"]
            app_secret = APPS[0]["app_secret"]
            user_id = os.environ.get("FEISHU_USER_OPEN_ID", "")
            if not user_id:
                print(f"[WS告警] 10min内断连 {count} 次，但无 FEISHU_USER_OPEN_ID 无法推送")
                return

            client = lark.Client.builder().app_id(app_id).app_secret(app_secret).build()
            msg = f"🔴 Smart Bot WebSocket 10分钟内断连 {count} 次\n\n请检查网络连接或飞书服务状态。\n时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            req = CreateMessageRequest.builder() \
                .receive_id_type("open_id") \
                .request_body(CreateMessageRequestBody.builder()
                    .receive_id(user_id)
                    .msg_type("text")
                    .content(json.dumps({"text": msg}))
                    .build()) \
                .build()
            resp = client.im.v1.message.create(req)
            if resp.success():
                print(f"[WS告警] 断连告警已发送")
            else:
                print(f"[WS告警] 发送失败: {resp.code}")
        except Exception as e:
            print(f"[WS告警] 告警推送异常: {e}")

    def run(self):
        """启动机器人（多 App 并行）"""
        print("=" * 50)
        print("智能飞书机器人 v4 (Claude CLI session)")
        print("=" * 50)
        print()
        print("✓ 会话管理器已就绪 (Claude CLI session resume)")
        print(f"  • Claude session: {self.claude_session.session_id or '(新建)'}")
        print(f"  • 临时文件: Redis (TTL={TEMP_FILE_TTL}s)" if self.tmp_store.available else "  • 临时文件: 磁盘（Redis 不可用）")
        print(f"  • App 数量: {len(APPS)}")
        print()

        ws_clients = []
        for app in APPS:
            app_id = app["app_id"]
            app_secret = app["app_secret"]

            lark_client = lark.Client.builder() \
                .app_id(app_id) \
                .app_secret(app_secret) \
                .log_level(lark.LogLevel.INFO) \
                .build()

            # 闭包捕获当前 app 的 lark_client
            def make_handlers(client):
                def on_msg(data):
                    # 立即派发到线程，避免 time.sleep 阻塞 asyncio event loop
                    threading.Thread(target=self._handle_message, args=(data, client), daemon=True).start()
                def on_card(data):
                    return self._handle_card_action(data, client)
                return on_msg, on_card

            on_msg, on_card = make_handlers(lark_client)

            event_handler = lark.EventDispatcherHandler.builder("", "") \
                .register_p2_im_message_receive_v1(on_msg) \
                .register_p2_im_message_message_read_v1(lambda data: None) \
                .register_p2_card_action_trigger(on_card) \
                .build()

            ws_client = lark.ws.Client(
                app_id, app_secret,
                event_handler=event_handler,
                log_level=lark.LogLevel.INFO
            )
            ws_clients.append(ws_client)
            self._ws_clients.append(ws_client)
            print(f"  ✓ App {app_id} 已注册")

        print()
        if ALLOWED_OPEN_IDS:
            print(f"✓ 白名单已启用 ({len(ALLOWED_OPEN_IDS)} 个授权用户)")
        else:
            print("⚠ 未配置白名单 (FEISHU_ALLOWED_OPEN_IDS)，将拒绝所有消息")
            print("  给机器人发条消息，控制台会打印你的 open_id")
        print()
        print("✓ 启动成功")
        print("  • 发送「信号」生成ML调仓信号")
        print("  • 发送「持仓」查看实盘持仓")
        print("  • 发送「重训」季度模型重训")
        print("  • 发截图+「已执行」更新持仓")
        print("  • 发送「帮助」查看全部命令")
        print("  • 操作需点击【允许】确认")
        print()
        print("等待消息...")
        print()

        # 使用 SDK 模块级缓存的 event loop（SDK 内部 _connect/_ping 都用这个 loop）
        from lark_oapi.ws.client import loop as sdk_loop

        async def _connect_with_retry(ws_client, app_id: str):
            """带断连统计的连接包装（滑动窗口告警）"""
            consecutive = 0
            while True:
                try:
                    await ws_client._connect()
                    if consecutive > 0:
                        print(f"[WS] App {app_id} 重连成功 (此前连续断连 {consecutive} 次)")
                    consecutive = 0
                    # 记录服务端配置的 ping 间隔作为默认值
                    server_interval = getattr(ws_client, '_ping_interval', 120)
                    if server_interval != self._default_ping_interval and server_interval > 30:
                        self._default_ping_interval = server_interval
                    sdk_loop.create_task(ws_client._ping_loop())
                    while True:
                        await asyncio.sleep(30)
                        self._maybe_restore_ping_interval()
                except Exception as e:
                    consecutive += 1
                    self._record_disconnect(app_id)

                    # 指数退避重连: 2s, 4s, 8s, 16s, 32s, 60s (封顶)
                    backoff = min(2 * (2 ** (consecutive - 1)), 60)
                    print(f"[WS] App {app_id} 断连第{consecutive}次, {backoff}s 后重连...")
                    await asyncio.sleep(backoff)

        async def _run_all():
            # 带重连逻辑并发启动所有 App
            tasks = []
            for ws_client, app in zip(ws_clients, APPS):
                tasks.append(_connect_with_retry(ws_client, app["app_id"]))
            await asyncio.gather(*tasks)

        try:
            sdk_loop.run_until_complete(_run_all())
        except KeyboardInterrupt:
            print("\n已停止")

# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    bot = SmartBot()
    bot.run()
