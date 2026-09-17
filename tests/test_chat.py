# -*- coding: utf-8 -*-
"""novacore.chat 自测:直接运行 `python tests/test_chat.py`。
覆盖:历史持久化、损坏容错、上下文截断、生成线程的流式/取消/错误/统计。"""
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PyQt6.QtWidgets import QApplication

_app = QApplication.instance() or QApplication([])

from novacore.chat import ChatSession, GenerationWorker, save_json_atomic
from novacore.config import Config
from novacore.engines.base import EngineError, GenParams, GenerationCancelled


def process_for(seconds: float):
    end = time.time() + seconds
    while time.time() < end:
        _app.processEvents()
        time.sleep(0.005)


class FakeEngine:
    """可控假引擎:可配置 token 流与异常。"""
    def __init__(self, tokens=None, error=None, load_ok=True, delay=0.0):
        self.tokens = tokens or ["你", "好"]
        self.error = error
        self.load_ok = load_ok
        self.delay = delay
        self.loaded = True
        self.cancelled = False

    def load_model(self, path, device=None, progress=None):
        if not self.load_ok:
            raise EngineError("load fail")
        self.loaded = True

    def unload(self):
        self.loaded = False

    def generate(self, messages, params):
        if self.error:
            raise self.error
        for t in self.tokens:
            if self.delay:
                time.sleep(self.delay)
            yield t
        if self.cancelled:
            raise GenerationCancelled()

    def cancel(self):
        self.cancelled = True

    def count_tokens(self, text):
        return max(1, len(text) // 2)


class FakeManager:
    def __init__(self, engine):
        self.engine = engine

    @property
    def current(self):
        return self.engine

    def generate(self, messages, params):
        yield from self.engine.generate(messages, params)

    def cancel(self):
        self.engine.cancel()

    def count_tokens(self, text):
        return self.engine.count_tokens(text)


class TestChatSession(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.hist = Path(self._tmp.name) / "hist.json"
        self.sess = Path(self._tmp.name) / "sessions.json"
        self.cfg = Config({"gen_ctx_len": 2048, "gen_max_tokens": 512,
                           "max_history": 10})

    def test_add_clear(self):
        s = ChatSession(self.cfg, history_file=self.hist, sessions_file=self.sess)
        s.add("user", "  你好  ")
        s.add("assistant", "你好!")
        s.add("user", "   ")  # 空白不入
        self.assertEqual(len(s.messages), 2)
        self.assertEqual(s.messages[0], {"role": "user", "content": "你好"})
        s.clear()
        self.assertEqual(s.messages, [])

    def test_save_load_roundtrip(self):
        s = ChatSession(self.cfg, history_file=self.hist, sessions_file=self.sess)
        s.add("user", "问题一")
        s.add("assistant", "回答一")
        self.assertTrue(s.save_history())
        s2 = ChatSession(self.cfg, history_file=self.hist, sessions_file=self.sess)
        self.assertEqual(s2.messages, s.messages)

    def test_load_corrupted_history(self):
        self.hist.write_text("{ broken json", encoding="utf-8")
        s = ChatSession(self.cfg, history_file=self.hist, sessions_file=self.sess)
        self.assertEqual(s.messages, [])

    def test_load_invalid_entries_filtered(self):
        self.hist.write_text(json.dumps({"messages": [
            {"role": "user", "content": "ok"},
            {"role": "hacker", "content": "x"},       # 非法角色
            {"role": "user", "content": 123},          # 非法内容
        ]}), encoding="utf-8")
        s = ChatSession(self.cfg, history_file=self.hist, sessions_file=self.sess)
        self.assertEqual(len(s.messages), 1)

    def test_max_history_enforced(self):
        s = ChatSession(self.cfg, history_file=self.hist, sessions_file=self.sess)
        for i in range(25):
            s.add("user", f"消息{i}")
        self.assertLessEqual(len(s.messages), 10)
        self.assertEqual(s.messages[-1]["content"], "消息24")

    def test_build_messages_with_system(self):
        s = ChatSession(self.cfg, history_file=self.hist, sessions_file=self.sess)
        s.add("user", "历史问题")
        msgs = s.build_messages("当前问题", GenParams(system_prompt="SYSPROMPT"))
        self.assertEqual(msgs[0], {"role": "system", "content": "SYSPROMPT"})
        self.assertEqual(msgs[-1], {"role": "user", "content": "当前问题"})

    def test_trim_keeps_system_and_recent(self):
        s = ChatSession(self.cfg, history_file=self.hist, sessions_file=self.sess)
        msgs = [{"role": "system", "content": "S" * 100}] + [
            {"role": "user", "content": "x" * 200} for _ in range(20)]
        trimmed = s.trim_to_budget(msgs, max_tokens=100)
        self.assertEqual(trimmed[0]["role"], "system")
        self.assertEqual(trimmed[-1]["content"], "x" * 200, "最新一条必留")
        self.assertLess(len(trimmed), 21, "应截断最旧的对话")

    def test_trim_fits_under_budget(self):
        s = ChatSession(self.cfg, history_file=self.hist, sessions_file=self.sess)
        msgs = [{"role": "user", "content": "x" * 100} for _ in range(10)]
        # 每条 50 tokens,预算 160 → 应保留最近 3 条
        trimmed = s.trim_to_budget(msgs, max_tokens=160)
        self.assertEqual(len(trimmed), 3)
        self.assertEqual(trimmed[-1]["content"], msgs[-1]["content"])

    def test_save_json_atomic(self):
        self.assertTrue(save_json_atomic(self.hist, {"a": 1}))
        with open(self.hist, encoding="utf-8") as f:
            self.assertEqual(json.load(f), {"a": 1})

    # ---------- 多会话 ----------
    def test_multi_session_create_switch_delete(self):
        s = ChatSession(self.cfg, history_file=self.hist, sessions_file=self.sess)
        s.add("user", "第一个会话的问题")
        s.add("assistant", "第一个会话的回答")
        s.save_history()
        id1 = s.session_id
        self.assertEqual(s.auto_title(), "第一个会话的问题")

        s.new_session()
        self.assertNotEqual(s.session_id, id1)
        self.assertEqual(s.messages, [])
        s.add("user", "第二个会话")
        s.save_history()
        id2 = s.session_id

        # 列表按更新时间倒序,应有两条
        summaries = s.list_sessions()
        self.assertEqual(len(summaries), 2)
        self.assertEqual(summaries[0]["id"], id2)

        # 切回第一个会话,消息正确恢复
        self.assertTrue(s.load_session(id1))
        self.assertEqual(s.messages[0]["content"], "第一个会话的问题")

        # 删除第二个会话
        self.assertTrue(s.delete_session(id2))
        self.assertEqual(len(s.list_sessions()), 1)
        self.assertFalse(s.delete_session("not-exist"))

    def test_legacy_history_migrated_to_session(self):
        # 旧版单文件历史存在、且无会话文件时,应自动迁移为一个会话
        self.hist.write_text(json.dumps({"messages": [
            {"role": "user", "content": "旧问题"},
            {"role": "assistant", "content": "旧回答"},
        ]}), encoding="utf-8")
        s = ChatSession(self.cfg, history_file=self.hist, sessions_file=self.sess)
        self.assertEqual(len(s.list_sessions()), 1)
        self.assertEqual(len(s.messages), 2)
        self.assertEqual(s.messages[0]["content"], "旧问题")

    def test_reopen_restores_latest_session(self):
        s = ChatSession(self.cfg, history_file=self.hist, sessions_file=self.sess)
        s.add("user", "会话A")
        s.save_history()
        s.new_session()
        s.add("user", "会话B")
        s.save_history()
        # 重新打开:应自动加载最近更新的会话B
        s2 = ChatSession(self.cfg, history_file=self.hist, sessions_file=self.sess)
        self.assertEqual(s2.messages[0]["content"], "会话B")


class TestGenerationWorker(unittest.TestCase):

    def test_streams_and_finishes(self):
        mgr = FakeManager(FakeEngine(tokens=["世", "界"]))
        toks, done, stats = [], [], []
        w = GenerationWorker(mgr, [{"role": "user", "content": "x"}], GenParams())
        w.token.connect(toks.append)
        w.finished.connect(lambda full, cancelled: done.append((full, cancelled)))
        w.stat.connect(stats.append)
        w.start()
        process_for(1.0)
        self.assertEqual("".join(toks), "世界")
        self.assertEqual(done, [("世界", False)])
        self.assertEqual(len(stats), 1)
        self.assertIn("tok_per_s", stats[0])

    def test_cancel_partial_result(self):
        eng = FakeEngine(tokens=[str(i) for i in range(1000)], delay=0.002)
        mgr = FakeManager(eng)
        done = []
        w = GenerationWorker(mgr, [], GenParams())
        w.finished.connect(lambda full, cancelled: done.append((full, cancelled)))
        w.start()
        time.sleep(0.05)
        w.cancel()
        process_for(1.5)
        self.assertEqual(len(done), 1)
        full, cancelled = done[0]
        self.assertTrue(cancelled)
        self.assertLess(len(full), 1000)

    def test_engine_error_emitted(self):
        mgr = FakeManager(FakeEngine(error=EngineError("显存不足")))
        errs = []
        w = GenerationWorker(mgr, [], GenParams())
        w.error.connect(errs.append)
        w.start()
        process_for(1.0)
        self.assertEqual(errs, ["显存不足"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
