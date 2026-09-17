# -*- coding: utf-8 -*-
"""对话页与模型加载链路集成自测(离屏):直接运行 `python tests/test_ui_chat.py`。
用假引擎注入 EngineManager,验证:模型加载→发送→流式显示→历史保存→停止→退出全链路。"""
import os
import sys
import struct
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication

_app = QApplication.instance() or QApplication([])

import novacore_main as m  # noqa: E402
from novacore.engines.base import BaseEngine, GenParams  # noqa: E402
from novacore import paths  # noqa: E402


class FakeLlmEngine(BaseEngine):
    engine_id = "llamacpp"
    display_name = "Fake"

    def __init__(self, tokens=None, delay=0.01):
        super().__init__()
        self.tokens = tokens or ["你", "好", "世", "界"]
        self.delay = delay

    def check_available(self):
        return True, "测试引擎"

    def load_model(self, path, device=None, progress=None):
        self._loaded_path = str(path)
        self._model_name = Path(path).name
        self._load_detail = device or "CPU 测试"

    def unload(self):
        self._loaded_path = None
        self._model_name = None

    def generate(self, messages, params):
        for t in self.tokens:
            time.sleep(self.delay)
            yield t

    def count_tokens(self, text):
        return max(1, len(text) // 2)


class FakeOvEngine(FakeLlmEngine):
    """记录实际收到 device 的假 OpenVINO 引擎。"""
    engine_id = "openvino"
    display_name = "FakeOV"

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.loaded_devices: list = []

    def load_model(self, path, device=None, progress=None):
        self.loaded_devices.append(device)
        super().load_model(path, device=device, progress=progress)


def build_fake_gguf(path: Path):
    """构造结构合法的 GGUF 文件(键长必须与实际字节数一致)。"""
    def s(b: bytes) -> bytes:
        return struct.pack("<Q", len(b)) + b

    buf = bytearray(b"GGUF")
    buf += struct.pack("<I", 3)            # version
    buf += struct.pack("<Q", 1)            # tensor count
    buf += struct.pack("<Q", 2)            # kv count
    buf += s(b"general.name") + struct.pack("<I", 8) + s(b"fake")
    buf += s(b"general.architecture") + struct.pack("<I", 8) + s(b"llama")
    buf += b"\x00" * (2 * 1024 * 1024)     # 张量数据区
    path.write_bytes(bytes(buf))


def pump(seconds: float):
    end = time.time() + seconds
    while time.time() < end:
        _app.processEvents()
        time.sleep(0.005)


def wait_until(cond, timeout=15.0):
    end = time.time() + timeout
    while time.time() < end:
        _app.processEvents()
        if cond():
            return True
        time.sleep(0.01)
    return False


class MsgBoxRecorder:
    """拦截模态对话框:离屏测试下 QMessageBox.exec 会永久阻塞,改为记录调用。"""
    calls = []

    @classmethod
    def patch(cls):
        cls.calls = []
        from PyQt6.QtWidgets import QMessageBox
        cls._orig = {}
        for name in ("warning", "information", "critical", "question"):
            cls._orig[name] = getattr(QMessageBox, name)

            def make(name):
                def fake(parent, title, text, *a, **k):
                    cls.calls.append((name, title, text))
                    from PyQt6.QtWidgets import QMessageBox
                    return QMessageBox.StandardButton.Ok
                return fake
            setattr(QMessageBox, name, make(name))
        return cls

    @classmethod
    def restore(cls):
        from PyQt6.QtWidgets import QMessageBox
        for name, orig in cls._orig.items():
            setattr(QMessageBox, name, orig)

    @classmethod
    def any(cls, name):
        return any(c[0] == name for c in cls.calls)


class TestChatIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        MsgBoxRecorder.patch()
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tmpdir = Path(cls._tmp.name)
        model = cls.tmpdir / "fake.gguf"
        build_fake_gguf(model)
        cls.model_path = str(model)
        # 注入假引擎(必须在 MainWindow 创建前,探测缓存才会记录可用)
        cls.fake = FakeLlmEngine()
        m.engine_mgr._engines["llamacpp"] = cls.fake
        # 确保测试使用 auto 引擎选择(避免用户真实配置的 default_engine 干扰)
        m.cfg.set("default_engine", "auto")
        # 防止启动钩子去拉起真实 Ollama 服务
        m.cfg.set("ollama_auto_start", False)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()
        MsgBoxRecorder.restore()
        # 还原被本测试修改的全局配置,避免污染真实配置
        m.cfg.set("default_engine", "auto")
        m.cfg.set("prefer_device", "CPU")
        m.cfg.set("current_loaded_model", "")
        m.save_config(m.cfg)

    def setUp(self):
        self.w = m.MainWindow()
        # 每个测试使用独立、空的会话存储,避免污染/读取真实 chat_sessions.json
        from novacore.chat import ConversationStore
        sess_file = self.tmpdir / f"sess_{id(self)}.json"
        self.w.chat_session.store = ConversationStore(sess_file)
        self.w.chat_session.new_session()
        self.w.refresh_history_list()

    def tearDown(self):
        self.w.close()
        pump(0.3)

    def test_load_model_then_chat(self):
        w = self.w
        self.assertFalse(m.engine_mgr.is_loaded())
        w.request_load_model(self.model_path, "fake.gguf")
        # 等 UI 侧完成加载回调(_loading 复位),而不是只等引擎侧 is_loaded
        self.assertTrue(wait_until(lambda: not w._loading, timeout=20),
                        "模型应在后台加载完成")
        self.assertTrue(m.engine_mgr.is_loaded())
        self.assertIn("✅", w.chat_box.toPlainText())

        # 发送消息并流式接收
        w.msg_input.setText("你好")
        w.on_send()
        self.assertTrue(wait_until(lambda: not w._generating, timeout=20),
                        "生成应正常结束")
        full = w.chat_box.toPlainText()
        self.assertIn("用户", full)
        self.assertIn("你好世界", full, "流式 token 应全部出现在聊天框")
        self.assertEqual(len(w.chat_session.messages), 2)
        self.assertEqual(w.chat_session.messages[0]["role"], "user")
        self.assertEqual(w.chat_session.messages[1]["content"], "你好世界")
        self.assertTrue(w.chat_session.store.store_file.exists(),
                        "会话应已持久化到多会话存储")
        self.assertIn("tok/s", w.speed_lab.text())

    def test_stop_generation_keeps_partial(self):
        w = self.w
        slow = FakeLlmEngine(tokens=[str(i) for i in range(200)], delay=0.02)
        m.engine_mgr._engines["llamacpp"] = slow
        slow.load_model(self.model_path)
        w.msg_input.setText("测试停止")
        w.on_send()
        QTimer.singleShot(120, w.on_stop_generation)
        self.assertTrue(wait_until(lambda: not w._generating, timeout=20))
        self.assertIn("手动停止", w.speed_lab.text())
        self.assertEqual(len(w.chat_session.messages), 2)
        self.assertLess(len(w.chat_session.messages[1]["content"]), 200)

    def test_new_chat_clears(self):
        w = self.w
        m.engine_mgr._engines["llamacpp"].load_model(self.model_path)
        w.msg_input.setText("第一条")
        w.on_send()
        self.assertTrue(wait_until(lambda: not w._generating))
        w.on_new_chat()
        self.assertEqual(w.chat_session.messages, [])
        self.assertIn("新会话", w.chat_box.toPlainText())

    def test_send_without_model_warns(self):
        w = self.w
        m.engine_mgr.unload_all()
        self.assertFalse(m.engine_mgr.is_loaded())
        MsgBoxRecorder.calls.clear()
        w.msg_input.setText("无模型")
        w.on_send()  # 应弹提示框而非崩溃
        self.assertFalse(w._generating)
        self.assertTrue(MsgBoxRecorder.any("warning"), "未加载模型时应提示用户")

    def test_loaded_model_shown_in_combo_and_config(self):
        w = self.w
        w.request_load_model(self.model_path, "fake.gguf")
        self.assertTrue(wait_until(lambda: not w._loading, timeout=20))
        self.assertIn("fake.gguf", w.model_sel.currentText())
        self.assertIn("已加载", w.model_sel.currentText())
        self.assertEqual(m.cfg.get("current_loaded_model"), "fake.gguf")

    def test_topbar_npu_selection_reaches_engine(self):
        """回归:顶部「推理设备」选 NPU-OpenVINO 时,即使配置 prefer_device
        仍是 CPU,也必须把 NPU 真正传给引擎(旧版只读配置,导致选了 NPU
        实际仍跑在 CPU 上)。"""
        w = self.w
        ovdir = self.tmpdir / "ov_model"
        ovdir.mkdir()
        (ovdir / "openvino_model.xml").write_text("<xml/>")
        fake_ov = FakeOvEngine()
        m.engine_mgr._engines["openvino"] = fake_ov
        m.engine_mgr._cap_cache.pop("openvino", None)
        m.cfg.set("prefer_device", "CPU")  # 配置仍是 CPU
        w.device_sel.setCurrentText("CPU")          # 先确保回到 CPU
        w.device_sel.setCurrentText("NPU-OpenVINO")  # 顶部切到 NPU(触发信号)
        w.request_load_model(str(ovdir), "ov_model")
        self.assertTrue(wait_until(lambda: not w._loading, timeout=20))
        self.assertEqual(fake_ov.loaded_devices, ["NPU"],
                         "顶部选择的 NPU 必须真正传给引擎")
        self.assertEqual(m.cfg.get("prefer_device"), "NPU-OpenVINO",
                         "顶部选择应同步保存到配置")

    def test_topbar_cpu_selection_reaches_engine(self):
        """顶部选 CPU 时,即使配置是 NPU-OpenVINO,也按 CPU 加载(即时选择优先)。"""
        w = self.w
        ovdir = self.tmpdir / "ov_model2"
        ovdir.mkdir()
        (ovdir / "openvino_model.xml").write_text("<xml/>")
        fake_ov = FakeOvEngine()
        m.engine_mgr._engines["openvino"] = fake_ov
        m.engine_mgr._cap_cache.pop("openvino", None)
        m.cfg.set("prefer_device", "NPU-OpenVINO")
        w.device_sel.setCurrentText("NPU-OpenVINO")  # 先切到 NPU
        w.device_sel.setCurrentText("CPU")           # 再切回 CPU(触发信号)
        w.request_load_model(str(ovdir), "ov_model2")
        self.assertTrue(wait_until(lambda: not w._loading, timeout=20))
        self.assertEqual(fake_ov.loaded_devices, ["CPU"],
                         "顶部选择的 CPU 必须真正传给引擎")

    def test_attach_text_file_into_input(self):
        w = self.w
        txt = self.tmpdir / "note.txt"
        txt.write_text("文件内容 ABC", encoding="utf-8")
        import unittest.mock as mock
        with mock.patch.object(m.QFileDialog, "getOpenFileNames",
                               return_value=([str(txt)], "")):
            w.on_attach_file()
        self.assertIn("[文件: note.txt]", w.msg_input.text())
        self.assertIn("文件内容 ABC", w.msg_input.text())

    def test_attach_image_friendly_note(self):
        w = self.w
        img = self.tmpdir / "pic.png"
        img.write_bytes(b"\x89PNG")
        import unittest.mock as mock
        with mock.patch.object(m.QFileDialog, "getOpenFileName",
                               return_value=(str(img), "")):
            w.on_attach_image()
        self.assertEqual(w._pending_image, str(img))
        self.assertIn("图片", w.chat_box.toPlainText())
        self.assertIn("不支持", w.chat_box.toPlainText())


if __name__ == "__main__":
    unittest.main(verbosity=2)
