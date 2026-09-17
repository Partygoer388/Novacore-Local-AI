# -*- coding: utf-8 -*-
"""更新功能端到端测试:本地 HTTP server + PyQt GUI"""
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import unittest
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PyQt6.QtWidgets import QApplication

_app = QApplication.instance() or QApplication(sys.argv)


class UpdateTest(unittest.TestCase):
    PORT = 18765

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.mkdtemp()
        (Path(cls._tmp) / "update.json").write_text(
            json.dumps({"version": "9.9.9-beta", "notes": "本地测试", "url": "https://x.com"}))
        cls._orig_cwd = os.getcwd()

        # 把 novacore 数据目录整体重定向到临时目录——
        # 旧实现直接 rmtree 工作区真实的 novacore_data(模型/对话/配置全部丢失),
        # 这是严重的数据破坏,测试必须完全隔离。
        import novacore.paths as npaths
        cls._saved_paths = {k: getattr(npaths, k) for k in (
            "WORK_DIR", "LOCAL_MODEL_DIR", "LORAS_DIR", "DATASETS_DIR",
            "LOGS_DIR", "CONFIG_FILE", "HISTORY_FILE", "SESSIONS_FILE",
            "MODELS_MANIFEST_FILE", "ALL_DIRS")}
        cls._data = Path(cls._tmp) / "novacore_data"
        npaths.WORK_DIR = cls._data
        npaths.LOCAL_MODEL_DIR = cls._data / "local_models"
        npaths.LORAS_DIR = cls._data / "loras"
        npaths.DATASETS_DIR = cls._data / "datasets"
        npaths.LOGS_DIR = cls._data / "logs"
        npaths.CONFIG_FILE = cls._data / "config.json"
        npaths.HISTORY_FILE = cls._data / "chat_history.json"
        npaths.SESSIONS_FILE = cls._data / "chat_sessions.json"
        npaths.MODELS_MANIFEST_FILE = cls._data / "manifest.json"
        npaths.ALL_DIRS = [npaths.WORK_DIR, npaths.LOCAL_MODEL_DIR,
                           npaths.LORAS_DIR, npaths.DATASETS_DIR, npaths.LOGS_DIR]
        npaths.ensure_dirs()

        class H(SimpleHTTPRequestHandler):
            def log_message(self, *a, **k): pass
            def translate_path(self, path):
                # 强制用 cls._tmp 作为根目录
                import os as _os
                _p = _os.path.join(cls._tmp, path.lstrip("/\\"))
                return _p

        cls.httpd = ThreadingHTTPServer(("127.0.0.1", cls.PORT), H)
        cls._t = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls._t.start()
        time.sleep(0.3)

        import novacore_main
        cls.w = novacore_main.MainWindow()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        import novacore.paths as npaths
        for k, v in cls._saved_paths.items():
            setattr(npaths, k, v)
        shutil.rmtree(cls._tmp, ignore_errors=True)

    def _pump(self, secs=2.0):
        end = time.time() + secs
        while time.time() < end:
            _app.processEvents()
            time.sleep(0.05)

    def test_custom_update_valid(self):
        """自定义更新源有效地址 → 发现新版本"""
        self.w.update_log.clear()
        self.w.custom_update.setChecked(True)
        self.w.update_url.setText(f"http://127.0.0.1:{self.PORT}/update.json")
        self.w.check_update()
        self._pump(3.0)
        log = self.w.update_log.toPlainText()
        self.assertIn("9.9.9-beta", log, f"应发现新版本,实际:\n{log}")
        self.assertIn("发现新版本", log)

    def test_custom_update_empty_rejected(self):
        """自定义更新源但空地址 → 拒绝"""
        self.w.update_log.clear()
        self.w.custom_update.setChecked(True)
        self.w.update_url.setText("")
        self.w.check_update()
        self._pump(0.5)
        log = self.w.update_log.toPlainText()
        self.assertIn("未填写", log)

    def test_not_custom_uses_default(self):
        """非自定义源 → 使用内置默认更新地址"""
        self.w.update_log.clear()
        self.w.custom_update.setChecked(False)
        self.w.update_url.setText("")
        self.w.check_update()
        self._pump(0.5)
        log = self.w.update_log.toPlainText()
        self.assertIn("默认更新源", log)

    def test_same_version_shows_up_to_date(self):
        """版本号相同时显示已是最新版本"""
        self.w.update_log.clear()
        self.w.custom_update.setChecked(True)
        self.w.update_url.setText(f"http://127.0.0.1:{self.PORT}/update.json")
        # 把 mock 改成当前版本
        (Path(self._tmp) / "update.json").write_text(
            json.dumps({"version": "0.0.1-alpha", "notes": "", "url": ""}))
        self.w.check_update()
        self._pump(3.0)
        log = self.w.update_log.toPlainText()
        self.assertIn("已是最新版本", log, f"实际:\n{log}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
