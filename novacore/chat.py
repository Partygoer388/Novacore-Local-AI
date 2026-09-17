"""对话会话管理:多会话历史持久化、上下文自动截断、流式生成线程(带取消与速度统计)。"""
from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Callable, Optional

from PyQt6.QtCore import QThread, pyqtSignal

from .config import Config
from .engines.base import EngineError, GenParams, GenerationCancelled
from .paths import HISTORY_FILE, SESSIONS_FILE

DEFAULT_MAX_HISTORY = 200  # 最多保留的消息条数(不含 system)


def save_json_atomic(path, obj) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp_", suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
        return True
    except OSError:
        return False


def _new_session_id() -> str:
    return uuid.uuid4().hex[:12]


class ConversationStore:
    """多个对话会话的持久化存储(全部会话保存在一个 JSON 文件里)。

    会话结构:{"id", "title", "created_at", "updated_at", "messages"}。
    """

    def __init__(self, store_file=SESSIONS_FILE):
        self.store_file = Path(store_file)
        self.sessions: list[dict] = []
        self._seq = 0
        self.load()

    def load(self) -> None:
        try:
            if not self.store_file.exists():
                self.sessions = []
                self._seq = 0
                return
            data = json.loads(self.store_file.read_text(encoding="utf-8"))
            arr = data.get("sessions") if isinstance(data, dict) else data
            if not isinstance(arr, list):
                self.sessions = []
                self._seq = 0
                return
            self.sessions = [s for s in arr
                             if isinstance(s, dict) and s.get("id")]
            # 恢复后让序号大于已有最大值,保证后续单调递增
            self._seq = max([int(s.get("seq", 0)) for s in self.sessions] or [0])
        except Exception:
            self.sessions = []
            self._seq = 0

    def save(self) -> bool:
        return save_json_atomic(
            self.store_file, {"version": 1, "sessions": self.sessions})

    def list_summaries(self) -> list[dict]:
        """返回轻量摘要(不含完整 messages),按最近活动倒序。"""
        out = [{
            "id": s["id"],
            "title": s.get("title") or "新对话",
            "created_at": s.get("created_at", 0),
            "updated_at": s.get("updated_at", 0),
            "seq": s.get("seq", 0),
            "count": len(s.get("messages", [])),
        } for s in self.sessions]
        # 先按 updated_at,再按单调序号兜底(避免极快连续操作时间戳相同)
        out.sort(key=lambda x: (x["updated_at"], x["seq"]), reverse=True)
        return out

    def get(self, sid: str) -> Optional[dict]:
        for s in self.sessions:
            if s.get("id") == sid:
                return s
        return None

    def create(self) -> dict:
        now = time.time()
        self._seq += 1
        s = {"id": _new_session_id(), "title": "新对话",
             "created_at": now, "updated_at": now, "seq": self._seq,
             "messages": []}
        self.sessions.append(s)
        self.save()
        return s

    def update_session(self, sid: str, messages: list[dict],
                       title: Optional[str] = None) -> None:
        s = self.get(sid)
        if s is None:
            return
        s["messages"] = [dict(m) for m in messages]
        # 仅当用户未手动命名时才用自动标题覆盖
        if title and not s.get("user_renamed"):
            s["title"] = title
        s["updated_at"] = time.time()
        self._seq += 1
        s["seq"] = self._seq
        self.save()

    def rename(self, sid: str, title: str) -> None:
        s = self.get(sid)
        if s is not None:
            s["title"] = (title or "新对话").strip() or "新对话"
            s["user_renamed"] = True  # 标记:用户已手动命名,自动命名不再覆盖
            s["updated_at"] = time.time()
            self.save()

    def delete(self, sid: str) -> bool:
        before = len(self.sessions)
        self.sessions = [s for s in self.sessions if s.get("id") != sid]
        if len(self.sessions) != before:
            self.save()
            return True
        return False

    def import_messages(self, messages: list[dict],
                        title: Optional[str] = None) -> dict:
        """把一批历史消息导入为一个新会话(用于旧版单文件历史迁移)。"""
        s = self.create()
        s["messages"] = [dict(m) for m in messages]
        s["title"] = title or "历史对话"
        self.save()
        return s


class ChatSession:
    """当前对话状态:消息历史 + 多会话持久化 + 上下文截断。"""

    def __init__(self, cfg: Config, token_counter: Optional[Callable[[str], int]] = None,
                 history_file=HISTORY_FILE, sessions_file=SESSIONS_FILE):
        self.cfg = cfg
        self.messages: list[dict] = []  # [{"role","content"}],不含 system
        self.history_file = history_file  # 旧版单文件(仅用于迁移)
        self._token_counter = token_counter
        self.store = ConversationStore(sessions_file)
        self.session_id: Optional[str] = None
        self._migrate_legacy_history()
        # 打开最近一次会话;没有则新建
        summaries = self.store.list_summaries()
        if summaries:
            self.load_session(summaries[0]["id"])
        else:
            self.new_session()

    # ---------- 多会话 ----------
    def _migrate_legacy_history(self) -> None:
        """旧版本只有单个 chat_history.json;若已有会话则忽略,否则迁移为一个会话。"""
        if self.store.sessions:
            return
        try:
            p = Path(self.history_file)
            if not p.exists():
                return
            data = json.loads(p.read_text(encoding="utf-8"))
            msgs = data.get("messages") if isinstance(data, dict) else data
            if not isinstance(msgs, list) or not msgs:
                return
            clean = [m for m in msgs
                     if isinstance(m, dict)
                     and m.get("role") in ("user", "assistant")
                     and isinstance(m.get("content"), str)]
            if clean:
                self.store.import_messages(clean, title="历史对话")
        except Exception:
            pass

    def new_session(self) -> str:
        s = self.store.create()
        self.session_id = s["id"]
        self.messages = []
        return self.session_id

    def load_session(self, sid: str) -> bool:
        s = self.store.get(sid)
        if s is None:
            return False
        self.session_id = sid
        self.messages = [dict(m) for m in s.get("messages", [])]
        return True

    def delete_session(self, sid: str) -> bool:
        return self.store.delete(sid)

    def rename_session(self, sid: str, title: str) -> None:
        """用户手动重命名;之后自动命名不再覆盖。"""
        self.store.rename(sid, title)

    def list_sessions(self) -> list[dict]:
        return self.store.list_summaries()

    def auto_title(self) -> str:
        for m in self.messages:
            if m.get("role") == "user":
                t = str(m.get("content", "")).strip().replace("\n", " ")
                if t:
                    return t[:24] + ("…" if len(t) > 24 else "")
        return "新对话"

    # ---------- 历史 ----------
    def add(self, role: str, content: str) -> None:
        content = content.strip()
        if content:
            self.messages.append({"role": role, "content": content})
        max_hist = int(self.cfg.get("max_history", DEFAULT_MAX_HISTORY)) \
            if self.cfg else DEFAULT_MAX_HISTORY
        if len(self.messages) > max_hist:
            self.messages = self.messages[-max_hist:]

    def clear(self) -> None:
        self.messages = []

    def load_history(self) -> bool:
        # 兼容旧接口:加载当前会话消息
        s = self.store.get(self.session_id) if self.session_id else None
        if s is not None:
            self.messages = [dict(m) for m in s.get("messages", [])]
            return True
        self.messages = []
        return False

    def save_history(self) -> bool:
        if self.session_id is None:
            self.new_session()
        self.store.update_session(self.session_id, self.messages,
                                 title=self.auto_title())
        return True

    # ---------- 上下文 ----------
    def _estimate_tokens(self, text: str) -> int:
        if self._token_counter is not None:
            try:
                return max(1, self._token_counter(text))
            except Exception:
                pass
        return max(1, len(text) // 2)

    def trim_to_budget(self, messages: list[dict], max_tokens: int) -> list[dict]:
        """保留 system 与最近的对话:最新一条必留,再往前加直到超出预算。"""
        if not messages:
            return []
        system_msgs = [m for m in messages if m.get("role") == "system"]
        rest = [m for m in messages if m.get("role") != "system"]
        if not rest:
            return system_msgs
        kept = [rest[-1]]  # 最新一条必留(即使超出预算)
        used = sum(self._estimate_tokens(m.get("content", "")) for m in kept)
        used += sum(self._estimate_tokens(m.get("content", "")) for m in system_msgs)
        for m in reversed(rest[:-1]):
            cost = self._estimate_tokens(m.get("content", ""))
            if used + cost > max_tokens:
                break
            kept.append(m)
            used += cost
        kept.reverse()
        return system_msgs + kept

    def build_messages(self, user_text: str, params: GenParams) -> list[dict]:
        """组装发送给引擎的消息:system + 历史 + 新消息,并做上下文截断。"""
        system = params.system_prompt
        all_msgs: list[dict] = []
        if system:
            all_msgs.append({"role": "system", "content": system})
        all_msgs.extend(self.messages)
        all_msgs.append({"role": "user", "content": user_text.strip()})
        budget = max(256, int(params.ctx_len) - int(params.max_tokens) - 256)
        return self.trim_to_budget(all_msgs, budget)


class GenerationWorker(QThread):
    """在后台线程执行引擎流式生成,通过信号把 token 实时送回 UI。"""
    token = pyqtSignal(str)
    finished = pyqtSignal(str, bool)   # (完整文本, 是否被取消)
    error = pyqtSignal(str)
    stat = pyqtSignal(dict)            # {"tokens": n, "elapsed": s, "tok_per_s": x}

    def __init__(self, engine_manager, messages: list[dict], params: GenParams,
                 parent=None):
        super().__init__(parent)
        self._mgr = engine_manager
        self._messages = messages
        self._params = params
        self._cancel_evt = threading.Event()

    def cancel(self) -> None:
        self._cancel_evt.set()
        try:
            self._mgr.cancel()
        except Exception:
            pass

    def run(self) -> None:
        chunks: list[str] = []
        t0 = time.time()
        cancelled = False
        try:
            for tok in self._mgr.generate(self._messages, self._params):
                if self._cancel_evt.is_set():
                    raise GenerationCancelled()
                chunks.append(tok)
                self.token.emit(tok)
        except GenerationCancelled:
            cancelled = True
        except Exception as e:
            if isinstance(e, EngineError):
                self.error.emit(str(e))
            else:
                self.error.emit(f"生成失败: {e}")
            return
        elapsed = max(time.time() - t0, 0.001)
        full = "".join(chunks)
        self.stat.emit({
            "tokens": len(chunks),
            "elapsed": round(elapsed, 2),
            "tok_per_s": round(len(chunks) / elapsed, 1),
        })
        self.finished.emit(full, cancelled)
