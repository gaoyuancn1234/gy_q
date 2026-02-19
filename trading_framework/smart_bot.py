#!/usr/bin/env python3
"""
智能飞书机器人 v3 (多用户隔离)
- 每用户独立记忆/消息队列/Claude进程
- 消息队列处理（防止错位）
- 智能等待（不盲目 sleep）
- 记忆系统（重要记忆持久化，向量召回）
"""

import os
import sys
import json
import time
import hashlib
import threading
import queue
import re
import shutil
import subprocess
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict, field
from collections import deque

from dotenv import load_dotenv
import redis
import numpy as np
import faiss
import onnxruntime as ort
from tokenizers import Tokenizer as HFTokenizer
from huggingface_hub import hf_hub_download
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
MEMORY_DIR = BOT_DIR / "memory"
MEMORY_DIR.mkdir(exist_ok=True)

load_dotenv(BOT_DIR / ".env", override=True)

APPS = []
for i in range(1, 10):
    app_id = os.environ.get(f"FEISHU_APP_ID_{i}")
    app_secret = os.environ.get(f"FEISHU_APP_SECRET_{i}")
    if app_id and app_secret:
        APPS.append({"app_id": app_id, "app_secret": app_secret})
    else:
        break

# 消息处理配置
MESSAGE_BATCH_WAIT = 1.2  # 等待批量消息的时间(秒)
MAX_WAIT_FOR_REPLY = 30   # 等待用户回复的最大时间(秒)

# Embedding 模型配置 (多语言 Transformer, 支持中文)
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIM = 384  # 输出维度
EMBEDDING_MAX_LEN = 128  # 最大 token 长度

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
TIMEOUT_SCRIPT = 300      # 简单脚本: 5分钟
TIMEOUT_MEDIUM = 600      # 中等任务: 10分钟
TIMEOUT_ANALYSIS = 3600   # 复杂分析: 1小时
TIMEOUT_COMPLEX = 3600    # 复杂代码任务: 1小时

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

@dataclass
class Memory:
    """记忆单元"""
    id: str
    content: str
    summary: str
    importance: float  # 0-1
    embedding: List[float] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 1

# ============================================================
# Transformer Embedding (ONNX Runtime)
# ============================================================

class TransformerEmbedder:
    """基于 ONNX Runtime 的 Transformer 文本向量化"""

    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        print("[Embedding] 加载 Transformer 模型...")
        try:
            tok_path = hf_hub_download(EMBEDDING_MODEL, 'tokenizer.json')
            onnx_path = hf_hub_download(EMBEDDING_MODEL, 'onnx/model.onnx')

            self.tokenizer = HFTokenizer.from_file(tok_path)
            # 自动检测 pad token (不同模型不同)
            vocab = self.tokenizer.get_vocab()
            if '<pad>' in vocab:
                pad_id, pad_token = vocab['<pad>'], '<pad>'
            elif '[PAD]' in vocab:
                pad_id, pad_token = vocab['[PAD]'], '[PAD]'
            else:
                pad_id, pad_token = 0, '[PAD]'
            self.tokenizer.enable_padding(pad_id=pad_id, pad_token=pad_token, length=EMBEDDING_MAX_LEN)
            self.tokenizer.enable_truncation(max_length=EMBEDDING_MAX_LEN)

            self.session = ort.InferenceSession(onnx_path)
            self.dim = EMBEDDING_DIM
            self.ready = True
            print(f"[Embedding] 模型加载成功, dim={self.dim}")
        except Exception as e:
            print(f"[Embedding] 模型加载失败: {e}, 降级为 hash 模式")
            self.ready = False
            self.dim = EMBEDDING_DIM

    def encode(self, text: str) -> np.ndarray:
        """单条文本编码"""
        return self.encode_batch([text])[0]

    def encode_batch(self, texts: list) -> np.ndarray:
        """批量编码，返回 (N, dim) 的归一化向量"""
        if not self.ready:
            return self._hash_fallback(texts)

        try:
            encodings = self.tokenizer.encode_batch(texts)
            input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
            attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
            token_type_ids = np.zeros_like(input_ids, dtype=np.int64)

            outputs = self.session.run(None, {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids
            })

            # Mean pooling
            token_emb = outputs[0]
            mask_exp = attention_mask[:, :, np.newaxis].astype(np.float32)
            embeddings = np.sum(token_emb * mask_exp, axis=1) / (np.sum(mask_exp, axis=1) + 1e-9)

            # L2 归一化
            faiss.normalize_L2(embeddings)
            return embeddings
        except Exception as e:
            print(f"[Embedding] 编码失败: {e}, 降级为 hash")
            return self._hash_fallback(texts)

    def _hash_fallback(self, texts: list) -> np.ndarray:
        """降级：基于字符 hash 的简单编码"""
        results = []
        for text in texts:
            vec = np.zeros(self.dim, dtype=np.float32)
            for i, char in enumerate(text):
                idx = hash(char) % self.dim
                vec[idx] += 1.0 / (1 + i * 0.1)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            results.append(vec)
        return np.array(results, dtype=np.float32)


# ============================================================
# 记忆系统 (Faiss 向量检索)
# ============================================================

class MemorySystem:
    """记忆系统"""

    def __init__(self, memory_dir: Path):
        self.memory_dir = memory_dir
        self.short_term_file = memory_dir / "short_term.json"
        self.long_term_file = memory_dir / "long_term.json"
        self.embeddings_file = memory_dir / "embeddings.npy"
        self.index_file = memory_dir / "index.json"
        self.faiss_index_file = memory_dir / "faiss.index"

        self.short_term: deque = deque(maxlen=50)
        self.long_term: Dict[str, Memory] = {}
        self.embeddings: Optional[np.ndarray] = None
        self.embedding_ids: List[str] = []

        # Faiss 索引 (Inner Product = 余弦相似度，因为向量已归一化)
        self.faiss_index: Optional[faiss.IndexFlatIP] = None

        # Transformer embedder
        self.embedder = TransformerEmbedder.get_instance()

        self._load()

    def _load(self):
        """加载记忆"""
        if self.short_term_file.exists():
            try:
                data = json.loads(self.short_term_file.read_text())
                for item in data:
                    self.short_term.append(Message(**item))
            except Exception as e:
                print(f"[记忆] 加载短期记忆失败: {e}")

        if self.long_term_file.exists():
            try:
                data = json.loads(self.long_term_file.read_text())
                for k, v in data.items():
                    self.long_term[k] = Memory(**v)
            except Exception as e:
                print(f"[记忆] 加载长期记忆失败: {e}")

        # 加载 Faiss 索引
        if self.faiss_index_file.exists() and self.index_file.exists():
            try:
                self.faiss_index = faiss.read_index(str(self.faiss_index_file))
                self.embedding_ids = json.loads(self.index_file.read_text())
                # 同步加载 embeddings 备份
                if self.embeddings_file.exists():
                    self.embeddings = np.load(str(self.embeddings_file))
                print(f"[记忆] Faiss 索引已加载: {self.faiss_index.ntotal} 条向量")
            except Exception as e:
                print(f"[记忆] 加载 Faiss 索引失败: {e}")
                self.faiss_index = None

        # 旧数据迁移：从 numpy 迁移到 Faiss
        if self.faiss_index is None and self.embeddings_file.exists() and self.index_file.exists():
            try:
                old_embeddings = np.load(str(self.embeddings_file))
                self.embedding_ids = json.loads(self.index_file.read_text())

                if old_embeddings.shape[1] != EMBEDDING_DIM:
                    # 维度不匹配(旧的128维)，需要用 Transformer 重新编码
                    print(f"[记忆] 旧向量维度 {old_embeddings.shape[1]} != {EMBEDDING_DIM}，重新编码...")
                    self._rebuild_embeddings()
                else:
                    # 维度匹配，直接建 Faiss 索引
                    self.embeddings = old_embeddings.astype(np.float32)
                    faiss.normalize_L2(self.embeddings)
                    self.faiss_index = faiss.IndexFlatIP(EMBEDDING_DIM)
                    self.faiss_index.add(self.embeddings)
                    self._save_faiss()
                    print(f"[记忆] 从 numpy 迁移到 Faiss: {self.faiss_index.ntotal} 条向量")
            except Exception as e:
                print(f"[记忆] 迁移旧数据失败: {e}")

        # 确保有空的 Faiss 索引
        if self.faiss_index is None:
            self.faiss_index = faiss.IndexFlatIP(EMBEDDING_DIM)
            print("[记忆] 创建空 Faiss 索引")

    def _rebuild_embeddings(self):
        """用 Transformer 重新编码所有长期记忆"""
        if not self.long_term:
            return

        texts = []
        ids = []
        for mem_id, mem in self.long_term.items():
            texts.append(mem.content)
            ids.append(mem_id)

        if not texts:
            return

        embeddings = self.embedder.encode_batch(texts)
        self.embeddings = embeddings
        self.embedding_ids = ids
        self.faiss_index = faiss.IndexFlatIP(EMBEDDING_DIM)
        self.faiss_index.add(embeddings)

        # 更新 Memory 对象中的 embedding
        for i, mem_id in enumerate(ids):
            self.long_term[mem_id].embedding = embeddings[i].tolist()

        self._save_faiss()
        print(f"[记忆] 重建完成: {len(ids)} 条记忆已重新编码")

    def _save(self):
        """保存记忆"""
        try:
            self.short_term_file.write_text(
                json.dumps([asdict(m) for m in self.short_term], ensure_ascii=False, indent=2)
            )
            self.long_term_file.write_text(
                json.dumps({k: asdict(v) for k, v in self.long_term.items()}, ensure_ascii=False, indent=2)
            )
            self._save_faiss()
        except Exception as e:
            print(f"[记忆] 保存失败: {e}")

    def _save_faiss(self):
        """保存 Faiss 索引和 embeddings"""
        try:
            if self.faiss_index is not None and self.faiss_index.ntotal > 0:
                faiss.write_index(self.faiss_index, str(self.faiss_index_file))
                self.index_file.write_text(json.dumps(self.embedding_ids))
                # 同时保存 numpy 备份
                if self.embeddings is not None:
                    np.save(str(self.embeddings_file), self.embeddings)
        except Exception as e:
            print(f"[记忆] 保存 Faiss 索引失败: {e}")

    def add_message(self, msg: Message):
        """添加消息到短期记忆"""
        self.short_term.append(msg)
        self._save()

    def get_recent_context(self, n: int = 10) -> List[Message]:
        """获取最近n轮对话"""
        return list(self.short_term)[-n*2:]

    def add_long_term(self, content: str, summary: str, importance: float, embedding: List[float] = None):
        """添加长期记忆（自动 Transformer 编码 + Faiss 索引）"""
        mem_id = hashlib.md5(content.encode()).hexdigest()[:12]
        now = time.time()

        # 如果没有提供 embedding，用 Transformer 自动生成
        if embedding is None or len(embedding) != EMBEDDING_DIM:
            vec = self.embedder.encode(content)
            embedding = vec.tolist()

        memory = Memory(
            id=mem_id,
            content=content,
            summary=summary,
            importance=importance,
            embedding=embedding,
            created_at=now,
            last_accessed=now,
            access_count=1
        )
        self.long_term[mem_id] = memory

        # 添加到 Faiss 索引
        vec = np.array(embedding, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(vec)
        self.faiss_index.add(vec)
        self.embedding_ids.append(mem_id)

        # 同步 numpy 备份
        if self.embeddings is None:
            self.embeddings = vec
        else:
            self.embeddings = np.vstack([self.embeddings, vec])

        self._save()
        print(f"[记忆] 保存长期记忆: {summary[:30]}... (Faiss 索引: {self.faiss_index.ntotal})")
        return mem_id

    def recall(self, query_embedding: List[float], top_k: int = 5) -> List[Memory]:
        """Faiss 向量召回"""
        if self.faiss_index is None or self.faiss_index.ntotal == 0:
            return []

        query_vec = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(query_vec)

        # Faiss 搜索
        k = min(top_k, self.faiss_index.ntotal)
        scores, indices = self.faiss_index.search(query_vec, k)

        results = []
        for i in range(k):
            idx = indices[0][i]
            score = scores[0][i]
            if idx < 0 or score < 0.1:
                continue
            if idx < len(self.embedding_ids):
                mem_id = self.embedding_ids[idx]
                if mem_id in self.long_term:
                    mem = self.long_term[mem_id]
                    mem.last_accessed = time.time()
                    mem.access_count += 1
                    results.append(mem)

        if results:
            self._save()
        return results

    def get_important_memories(self, threshold: float = 0.7) -> List[Memory]:
        """获取重要记忆"""
        return [m for m in self.long_term.values() if m.importance >= threshold]

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
# 用户会话 (每用户独立状态)
# ============================================================

# 已知用户 open_id（用于旧记忆迁移）
KNOWN_USER_OPEN_ID = os.environ.get("FEISHU_USER_OPEN_ID", "")

class UserSession:
    """封装单个用户的所有状态"""

    def __init__(self, open_id: str, memory_dir: Path):
        self.open_id = open_id
        self.lark_client: Optional[lark.Client] = None  # 由 on_message/on_card 注入
        self.memory = MemorySystem(memory_dir)
        self.msg_queue = MessageQueue()
        self.processing_lock = threading.Lock()
        self.is_processing = False
        self.current_process: Optional[subprocess.Popen] = None
        self.pending_messages: List[str] = []
        self.current_task: str = ""
        self.last_result: str = ""
        self.pending_actions: Dict[str, dict] = {}
        self._actions_lock = threading.Lock()

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

    def __init__(self, base_memory_dir: Path):
        self.base_memory_dir = base_memory_dir
        self._sessions: Dict[str, UserSession] = {}
        self._lock = threading.Lock()
        self._migrate_old_memory()

    def _migrate_old_memory(self):
        """将旧的扁平 memory/ 文件迁移到已知用户子目录"""
        flat_files = [
            "short_term.json", "long_term.json",
            "embeddings.npy", "faiss.index", "index.json"
        ]
        # 检查是否存在需要迁移的扁平文件
        has_flat = any((self.base_memory_dir / f).exists() for f in flat_files)
        if not has_flat:
            return

        target_dir = self.base_memory_dir / KNOWN_USER_OPEN_ID
        if target_dir.exists() and any(target_dir.iterdir()):
            # 目标目录已有文件，跳过迁移
            return

        target_dir.mkdir(parents=True, exist_ok=True)
        migrated = []
        for fname in flat_files:
            src = self.base_memory_dir / fname
            dst = target_dir / fname
            if src.exists():
                shutil.move(str(src), str(dst))
                migrated.append(fname)

        if migrated:
            print(f"[迁移] 旧记忆已迁移到 {target_dir.name}/: {', '.join(migrated)}")

    def get_or_create(self, open_id: str) -> UserSession:
        with self._lock:
            if open_id not in self._sessions:
                mem_dir = self.base_memory_dir / open_id
                mem_dir.mkdir(parents=True, exist_ok=True)
                self._sessions[open_id] = UserSession(open_id, mem_dir)
                print(f"[会话] 创建新用户会话: {open_id[:20]}...")
            return self._sessions[open_id]

# ============================================================
# 智能机器人
# ============================================================

class SmartBot:
    """智能飞书机器人（多用户隔离）"""

    def __init__(self):
        self.session_mgr = SessionManager(MEMORY_DIR)
        self.tmp_store = TempFileStore()
        self.tmp_store.cleanup_disk()  # 启动时清理残留
        self.processed_msg_ids: set = set()  # 内存降级用（Redis 不可用时）
        self._ws_disconnect_count = 0  # WebSocket 断连统计
        self._ws_disconnect_alerted = False  # 是否已发送断连告警

    def send_text(self, text: str, session: 'UserSession' = None):
        """发送文本消息"""
        client = session.lark_client if session else None
        open_id = session.open_id if session else None
        if not client or not open_id:
            print(f"[Bot] {text}")
            return

        try:
            req = CreateMessageRequest.builder() \
                .receive_id_type("open_id") \
                .request_body(CreateMessageRequestBody.builder()
                    .receive_id(open_id)
                    .msg_type("text")
                    .content(json.dumps({"text": text}))
                    .build()) \
                .build()
            client.im.v1.message.create(req)
        except Exception as e:
            print(f"[发送失败] {e}")

    def send_markdown(self, content: str, title: str = "", session: 'UserSession' = None):
        """发送 Markdown 卡片消息"""
        client = session.lark_client if session else None
        open_id = session.open_id if session else None
        if not client or not open_id:
            print(f"[Bot MD] {content[:100]}")
            return

        try:
            card = {
                "schema": "2.0",
                "config": {"wide_screen_mode": True},
                "body": {
                    "direction": "vertical",
                    "elements": [
                        {"tag": "markdown", "content": content}
                    ]
                }
            }
            if title:
                card["header"] = {
                    "title": {"tag": "plain_text", "content": title},
                    "template": "blue"
                }

            req = CreateMessageRequest.builder() \
                .receive_id_type("open_id") \
                .request_body(CreateMessageRequestBody.builder()
                    .receive_id(open_id)
                    .msg_type("interactive")
                    .content(json.dumps(card))
                    .build()) \
                .build()
            client.im.v1.message.create(req)
        except Exception as e:
            print(f"[发送MD失败] {e}")
            # 降级为普通文本
            self.send_text(content, session)

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
                save_path = WORK_DIR / f"user_image_{int(time.time())}.png"
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

    def get_embedding(self, text: str) -> List[float]:
        """Transformer 文本向量化 (all-MiniLM-L6-v2, 384维)"""
        embedder = TransformerEmbedder.get_instance()
        return embedder.encode(text).tolist()

    def build_context(self, user_input: str, session: 'UserSession' = None) -> str:
        """构建上下文"""
        memory = session.memory if session else MemorySystem(MEMORY_DIR)
        parts = []

        # 重要记忆
        important = memory.get_important_memories(threshold=0.8)
        if important:
            parts.append("【重要记忆】")
            for mem in important[:3]:
                parts.append(f"- {mem.summary}")

        # 相关记忆
        query_embedding = self.get_embedding(user_input)
        related = memory.recall(query_embedding, top_k=3)
        if related:
            parts.append("\n【相关记忆】")
            for mem in related:
                parts.append(f"- {mem.summary}")

        # 最近对话
        recent = memory.get_recent_context(n=5)
        if recent:
            parts.append("\n【最近对话】")
            for msg in recent[-6:]:
                role = "用户" if msg.role == "user" else "助手"
                parts.append(f"{role}: {msg.content[:80]}")

        return "\n".join(parts)

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
            3: (TIMEOUT_SCRIPT, "脚本任务(5分钟)"),
            4: (TIMEOUT_MEDIUM, "中等任务(10分钟)"),
            5: (TIMEOUT_ANALYSIS, "分析任务(1小时)"),
            6: (TIMEOUT_COMPLEX, "复杂任务(1小时)"),
        }

        # 构建最近对话上下文（帮助理解追问类消息）
        recent_context = ""
        if session and session.memory:
            recent = session.memory.get_recent_context(n=4)
            if recent:
                lines = []
                for msg in recent[-8:]:
                    role = "用户" if msg.role == "user" else "助手"
                    lines.append(f"{role}: {msg.content[:300]}")
                recent_context = "\n".join(lines)

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
                '/usr/local/bin/claude', '--print', '--dangerously-skip-permissions',
                '--output-format', 'json',
                '--model', 'haiku',
                '--json-schema', json_schema,
                '-p', prompt
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=15, cwd=str(WORK_DIR), env=_CLEAN_ENV
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
        """调用 Claude - 让 Claude 直接执行操作，动态超时"""
        context = self.build_context(user_input, session)

        # 估算任务复杂度和超时时间（传入 session 以获取上下文）
        timeout, complexity_desc = self.estimate_task_complexity(user_input, session)
        print(f"[超时设置] {complexity_desc} -> {timeout}秒")

        # 如果有上次结果，加入上下文
        if session and session.last_result:
            context += f"\n\n上一个任务的结果:\n{session.last_result[:500]}"

        user_id = session.open_id if session else "unknown"
        system_prompt = f"""你是一个主动的量化交易助手，直接帮用户完成任务。

{context}

工作目录: {WORK_DIR}
当前用户ID: {user_id}

历史文件查看：用户之前发送的图片等临时文件存储在 Redis 中（保留90天）。
  列出最近10个: python {BOT_DIR}/redis_files.py list {user_id}
  翻页(第11-20个): python {BOT_DIR}/redis_files.py list {user_id} 10 10
  按文件名搜索: python {BOT_DIR}/redis_files.py search {user_id} <关键词>
  提取某个文件: python {BOT_DIR}/redis_files.py get {user_id} <file_id>
  提取后会输出磁盘路径，你可以用 Read 工具查看该文件。
  当用户提到"之前的图片"、"上次的截图"等，先用 list 或 search 查找。

规则：
1. 直接执行任务，不要问"是否需要"，不要等确认
2. 用户说"画图"就直接写代码、执行、生成图片
3. 生成图片后，在回复中写明图片路径，格式：[IMAGE:/path/to/image.png]
4. 如果用户发送了图片，仔细分析图片内容
5. 回复简洁，中文，告诉用户做了什么"""

        try:
            # 如果有图片，在 prompt 中加入图片路径让 Claude 读取
            if image_path and Path(image_path).exists():
                prompt = f"{system_prompt}\n\n用户发送了图片，路径: {image_path}\n请先用 Read 工具查看这张图片，然后回答用户问题。\n\n用户: {user_input}"
                print(f"[Claude] 需要读取图片: {image_path}")
            else:
                prompt = f"{system_prompt}\n\n用户: {user_input}"

            cmd = ['/usr/local/bin/claude', '--print', '--dangerously-skip-permissions', '-p', prompt]

            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                cwd=str(WORK_DIR), env=_CLEAN_ENV
            )
            if session:
                session.current_process = proc
            try:
                stdout, stderr = proc.communicate(timeout=timeout)
                output = stdout.strip() or stderr.strip() or "无响应"
            except subprocess.TimeoutExpired:
                # 不杀进程，启动后台线程继续等待结果
                print(f"[超时] {timeout}秒已到，任务转入后台继续执行")
                if session:
                    session.current_process = None
                self._wait_bg_result(proc, session, complexity_desc)
                return f"⏳ 任务较复杂（{complexity_desc}），已转入后台继续执行，完成后会自动发送结果。"
            finally:
                if session:
                    session.current_process = None
            print(f"[Claude输出长度] {len(output)} 字符")
            # 前台也检测 API 错误，加醒目标记
            api_error_keywords = ['403', 'forbidden', 'Failed to authenticate',
                                  'API Error', '401', 'unauthorized']
            if any(kw.lower() in output.lower() for kw in api_error_keywords):
                print(f"[前台API错误] {output[:200]}")
                return f"⚠️ Claude API 访问错误，需要你处理：\n\n{output}"
            return output
        except Exception as e:
            if session:
                session.current_process = None
            return f"调用失败: {e}"

    def _wait_bg_result(self, proc: subprocess.Popen, session: 'UserSession', complexity_desc: str):
        """后台线程等待超时任务完成，完成后主动推送结果给用户"""
        def _is_api_error(text: str) -> bool:
            """检测是否为 API 认证/访问错误"""
            error_indicators = ['403', 'forbidden', 'Request not allowed',
                                'Failed to authenticate', 'API Error',
                                '401', 'unauthorized', 'rate_limit']
            text_lower = text.lower()
            return any(indicator.lower() in text_lower for indicator in error_indicators)

        def _wait():
            try:
                # 最多再等30分钟
                stdout, stderr = proc.communicate(timeout=1800)
                output = stdout.strip() or stderr.strip()
                if output:
                    print(f"[后台任务完成] {len(output)} 字符")
                    # 检测 API 403 等错误，立即通知用户
                    if _is_api_error(output):
                        print(f"[后台任务API错误] {output[:200]}")
                        self.send_text(
                            f"⚠️ 后台任务遇到 API 访问错误，需要你处理：\n\n"
                            f"{output[:500]}\n\n"
                            f"（原始任务复杂度: {complexity_desc}）",
                            session
                        )
                        return
                    session.last_result = output[:500]
                    self.process_response(output, session)
                else:
                    self.send_text("后台任务已完成，但没有产生输出。", session)
            except subprocess.TimeoutExpired:
                proc.kill()
                self.send_text("后台任务执行超过30分钟，已终止。", session)
            except Exception as e:
                print(f"[后台任务异常] {e}")
                self.send_text(f"后台任务执行出错: {e}", session)

        t = threading.Thread(target=_wait, daemon=True)
        t.start()

    def evaluate_and_save_memory(self, user_text: str, assistant_response: str, session: 'UserSession' = None):
        """异步调用 Claude CLI agent 评估对话是否值得记忆，并浓缩存入长期记忆"""
        memory = session.memory if session else None
        if not memory:
            return
        try:
            # 取最近的对话上下文
            recent = memory.get_recent_context(n=3)
            recent_ctx = ""
            if recent:
                for msg in recent[-4:]:
                    role = "用户" if msg.role == "user" else "助手"
                    recent_ctx += f"{role}: {msg.content[:100]}\n"

            # 获取已有长期记忆的摘要，用于去重
            existing_summaries = ""
            if memory.long_term:
                summaries = [m.summary for m in list(memory.long_term.values())[-20:]]
                existing_summaries = "\n".join(f"- {s}" for s in summaries)

            prompt = f"""你是一个通用记忆评估 agent。分析下面的对话，判断是否包含值得长期记住的信息。

【最近上下文】
{recent_ctx}
【本轮对话】
用户: {user_text[:300]}
助手: {assistant_response[:300]}

{"【已有记忆（避免重复）】" + chr(10) + existing_summaries if existing_summaries else ""}

请判断这段对话是否包含以下任何一类值得记忆的信息：
1. 用户偏好与习惯（工作方式、技术偏好、沟通风格、个人喜好等）
2. 用户个人信息（身份、账号、配置、环境等关键事实）
3. 重要决策与结论（用户做出的决定、达成的共识、分析结论）
4. 关键知识（学到的经验教训、问题的解决方案、有价值的发现）
5. 项目/任务上下文（项目目标、架构决策、待办事项、进度里程碑）
6. 用户明确要求记住的任何内容

去重规则：如果【已有记忆】中已包含相同或高度相似的信息，不要重复记忆。只记录新增或更新的信息。

如果有值得记忆的信息，请严格按以下 JSON 格式输出（可以输出多条）：
{{"memories": [{{"content": "具体内容", "summary": "一句话摘要", "importance": 0.8}}]}}

importance 评分标准：0.6=一般有用, 0.7=比较重要, 0.8=重要, 0.9=非常重要, 0.95=关键信息

如果没有值得记忆的信息（闲聊、简单问答、重复内容），输出：
{{"memories": []}}

只输出 JSON，不要其他文字。"""

            cmd = ['/usr/local/bin/claude', '--print', '--dangerously-skip-permissions', '-p', prompt]
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30, cwd=str(WORK_DIR), env=_CLEAN_ENV
            )
            output = result.stdout.strip()
            if not output:
                return

            # 提取 JSON（处理可能的 markdown 包裹）
            json_match = re.search(r'\{.*\}', output, re.DOTALL)
            if not json_match:
                print(f"[记忆评估] 未找到有效 JSON")
                return

            data = json.loads(json_match.group())
            memories = data.get("memories", [])

            if not memories:
                print(f"[记忆评估] 本轮对话无需记忆")
                return

            for mem in memories:
                content = mem.get("content", "")
                summary = mem.get("summary", content[:50])
                importance = float(mem.get("importance", 0.7))
                if content and importance >= 0.6:
                    memory.add_long_term(
                        content=content,
                        summary=summary,
                        importance=importance
                    )
                    print(f"[记忆评估] 已存入: {summary[:40]} (重要度: {importance})")

        except json.JSONDecodeError as e:
            print(f"[记忆评估] JSON 解析失败: {e}")
        except subprocess.TimeoutExpired:
            print(f"[记忆评估] 超时，跳过")
        except Exception as e:
            print(f"[记忆评估] 异常: {e}")

    def execute_command(self, cmd: str, session: 'UserSession' = None):
        """执行命令 - 动态超时"""
        # 根据命令复杂度设置超时
        timeout = TIMEOUT_SCRIPT  # 默认5分钟
        if 'python' in cmd and any(kw in cmd for kw in ['train', 'backtest', 'test']):
            timeout = TIMEOUT_ANALYSIS  # 1小时
        try:
            result = subprocess.run(
                cmd, shell=True, cwd=str(WORK_DIR),
                capture_output=True, text=True, timeout=timeout
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

    def process_response(self, response: str, session: 'UserSession' = None):
        """处理 Claude 响应 - 简化版，Claude 已直接执行操作"""
        print(f"[Claude响应] {response[:500]}...")

        # 提取图片路径 [IMAGE:/path/to/image.png]
        image_pattern = r'\[IMAGE:([^\]]+)\]'
        image_matches = re.findall(image_pattern, response)

        # 也检测常见路径格式
        path_pattern = r'[`\s](/[^\s`\n]+\.(?:png|jpg|jpeg|gif|webp))'
        path_matches = re.findall(path_pattern, response, re.IGNORECASE)

        all_images = list(set(image_matches + path_matches))

        # 清理文本，移除 [IMAGE:] 标记
        clean_text = re.sub(image_pattern, '', response).strip()

        # 发送文本（用 Markdown 卡片以支持格式化）
        if clean_text:
            # 截断过长的文本
            if len(clean_text) > 3000:
                clean_text = clean_text[:3000] + "..."
            # 检测是否包含 Markdown 格式（表格、代码块、列表等）
            has_markdown = any(x in clean_text for x in ['|', '```', '- ', '* ', '**', '##'])
            if has_markdown:
                self.send_markdown(clean_text, session=session)
            else:
                self.send_text(clean_text, session)

        # 发送图片
        for img_path in all_images:
            img_path = img_path.strip()
            if Path(img_path).exists():
                print(f"[发送图片] {img_path}")
                self.send_image(img_path, session)
            else:
                print(f"[图片不存在] {img_path}")

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

    def _handle_screenshot_trades(self, text: str, image_path: str, session: 'UserSession' = None):
        """截图解析持仓更新

        用户发送成交截图 + "已执行/成交/已买入/已卖出" → Claude 解析 → 确认卡片 → apply_trades
        """
        try:
            # 调用 Claude CLI 解析截图
            prompt = f"""分析这张股票交易成交截图，提取所有成交记录。

用户说: {text}

请仔细识别截图中的每笔成交，输出 JSON 格式:
{{"trades": [
    {{"action": "buy", "code": "SH600036", "shares": 200, "price": 38.50, "name": "招商银行"}},
    {{"action": "sell", "code": "SH601166", "shares": 300, "price": 16.80, "name": "兴业银行"}}
]}}

规则:
1. code 用 Qlib 格式: 沪市 SH + 6位数字, 深市 SZ + 6位数字
2. action: buy=买入, sell=卖出
3. shares: 成交数量 (股)
4. price: 成交均价
5. 只输出 JSON，不要其他文字"""

            cmd = [
                '/usr/local/bin/claude', '--print', '--dangerously-skip-permissions',
                '-p', f"请先用 Read 工具查看图片 {image_path}，然后:\n\n{prompt}"
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=60, cwd=str(WORK_DIR), env=_CLEAN_ENV
            )
            output = result.stdout.strip()

            # 提取 JSON
            json_match = re.search(r'\{.*\}', output, re.DOTALL)
            if not json_match:
                self.send_text("❌ 无法从截图中解析成交记录，请确认图片清晰", session)
                return

            data = json.loads(json_match.group())
            trades = data.get('trades', [])
            if not trades:
                self.send_text("未识别到成交记录", session)
                return

            # 构建确认内容
            lines = ["识别到以下成交记录:\n"]
            for t in trades:
                action_str = "买入" if t['action'] == 'buy' else "卖出"
                name = t.get('name', t['code'])
                lines.append(f"• {action_str} {name} {t['shares']}股 ×{t['price']}")

            action_id = f"TRADE_{int(time.time())}"
            session.add_pending_action(action_id, 'apply_trades', {'trades': trades})
            self.send_confirm_card(
                "📸 确认成交记录",
                "\n".join(lines),
                action_id,
                session
            )
        except json.JSONDecodeError:
            self.send_text("❌ 解析成交记录失败，请重新发送截图", session)
        except subprocess.TimeoutExpired:
            self.send_text("⏳ 截图解析超时，请重试", session)
        except Exception as e:
            self.send_text(f"❌ 截图处理异常: {e}", session)

    def handle_quick_commands(self, text: str, session: 'UserSession' = None, image_path: str = None) -> bool:
        """处理快捷命令，返回是否已处理"""
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
                "• 发截图+「已执行」- 更新持仓",
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

        # 截图 + 成交确认
        if image_path and any(kw in text for kw in ['已执行', '成交', '已买入', '已卖出', '已买', '已卖']):
            self.send_text("⏳ 正在解析成交截图...", session)
            threading.Thread(
                target=self._handle_screenshot_trades,
                args=(text, image_path, session),
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
                ['/usr/local/bin/claude', '--print', '--dangerously-skip-permissions',
                 '--output-format', 'json', '--json-schema', schema,
                 '-p', prompt],
                capture_output=True, text=True, timeout=15, env=_CLEAN_ENV
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
            # 记录用户消息
            session.memory.add_message(Message(role="user", content=text))
            session.current_task = text[:30]

            # 快捷命令
            if self.handle_quick_commands(text, session, image_path=image_path):
                return

            # 给消息加"收到"表情
            if msg_id:
                self.add_reaction(msg_id, "OK", session)
            response = self.call_claude(text, session, image_path)
            self.process_response(response, session)

            # 处理完毕后将临时图片存入 Redis 并删除磁盘文件
            if image_path and Path(image_path).exists():
                self.tmp_store.store(image_path, session.open_id)

            # 保存结果供后续任务参考
            session.last_result = response[:500]

            # 记录助手回复
            clean_response = re.sub(r'\[IMAGE:[^\]]*\]', '', response).strip()
            if clean_response:
                session.memory.add_message(Message(role="assistant", content=clean_response[:500]))

            # 异步评估记忆（不阻塞主流程）
            threading.Thread(
                target=self.evaluate_and_save_memory,
                args=(text, clean_response[:300], session),
                daemon=True
            ).start()

        finally:
            with session.processing_lock:
                session.is_processing = False
            session.current_task = ""

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
            # 获取最后一张图片
            last_image = next((m.image_path for m in reversed(batch) if m.image_path), None)
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

    def _send_disconnect_alert(self, count: int):
        """WebSocket 连续断连超阈值时，通过飞书告警通知"""
        try:
            app_id = APPS[0]["app_id"]
            app_secret = APPS[0]["app_secret"]
            user_id = os.environ.get("FEISHU_USER_OPEN_ID", "")
            if not user_id:
                print(f"[WS告警] 连续断连 {count} 次，但无 FEISHU_USER_OPEN_ID 无法推送")
                return

            client = lark.Client.builder().app_id(app_id).app_secret(app_secret).build()
            msg = f"🔴 Smart Bot WebSocket 连续断连 {count} 次\n\n请检查网络连接或飞书服务状态。\n时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
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
        print("智能飞书机器人 v3 (多用户隔离 + 多App)")
        print("=" * 50)
        print()
        print("✓ 会话管理器已就绪 (按用户隔离记忆/队列/进程)")
        print(f"  • 记忆目录: {MEMORY_DIR}")
        print(f"  • Embedding 模型: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
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
            print(f"  ✓ App {app_id} 已注册")

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
            """带断连统计的连接包装"""
            while True:
                try:
                    await ws_client._connect()
                    # 连接成功，重置计数
                    if self._ws_disconnect_count > 0:
                        print(f"[WS] App {app_id} 重连成功 (此前断连 {self._ws_disconnect_count} 次)")
                    self._ws_disconnect_count = 0
                    self._ws_disconnect_alerted = False
                    sdk_loop.create_task(ws_client._ping_loop())
                    # 连接建立后等待，直到连接断开
                    while True:
                        await asyncio.sleep(30)
                except Exception as e:
                    self._ws_disconnect_count += 1
                    print(f"[WS] App {app_id} 断连 #{self._ws_disconnect_count}: {e}")

                    # 连续断连超过5次，发送告警
                    if self._ws_disconnect_count >= 5 and not self._ws_disconnect_alerted:
                        self._ws_disconnect_alerted = True
                        self._send_disconnect_alert(self._ws_disconnect_count)

                    # 指数退避重连: 5s, 10s, 20s, ... 最大 300s
                    backoff = min(5 * (2 ** (self._ws_disconnect_count - 1)), 300)
                    print(f"[WS] {backoff}s 后重连...")
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
