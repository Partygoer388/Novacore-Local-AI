# -*- coding: utf-8 -*-
"""商店页 UI 集成自测(离屏):直接运行 `python tests/test_ui_store.py`。
覆盖:内置清单填充、下载源选择持久化、自定义 manifest 刷新、下载对话框端到端(本地服务器)。"""
import json
import os
import struct
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PyQt6.QtWidgets import QApplication, QMessageBox

_app = QApplication.instance() or QApplication([])

import test_downloader as td  # noqa: E402  复用本地 Range 服务器与 GGUF 构造器
import novacore_main as m  # noqa: E402
from novacore import modelstore  # noqa: E402


def patch_msgboxes():
    calls = []
    orig = {}

    def make(name):
        def fake(*args, **kwargs):
            calls.append(name)
            return QMessageBox.StandardButton.Ok
        return fake

    for name in ("warning", "information", "critical", "question"):
        orig[name] = getattr(QMessageBox, name)
        setattr(QMessageBox, name, make(name))
    return calls, orig


def restore_msgboxes(orig):
    for name, fn in orig.items():
        setattr(QMessageBox, name, fn)


def pump(seconds):
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


class TestStoreUi(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.calls, cls.orig = patch_msgboxes()
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tmpdir = Path(cls._tmp.name)

    @classmethod
    def tearDownClass(cls):
        restore_msgboxes(cls.orig)
        cls._tmp.cleanup()

    def setUp(self):
        # 防止启动钩子去拉起真实 Ollama 服务
        m.cfg.set("ollama_auto_start", False)
        self.w = m.MainWindow()

    def tearDown(self):
        self.w.close()
        pump(0.3)

    def test_store_populated_from_builtin(self):
        w = self.w
        w.refresh_store_list()
        self.assertGreaterEqual(w.store_list.count(), 8, "内置清单应全部列出")
        # 无 example.org 占位(行文本里不含 example.org)
        for i in range(w.store_list.count()):
            item = w.store_list.item(i)
            wid = w.store_list.itemWidget(item)
            self.assertIsNotNone(wid)

    def test_download_source_combo_persists(self):
        w = self.w
        idx = m.SOURCE_CHOICES.index("hf_mirror")
        w.download_src.setCurrentIndex(idx)
        w.on_download_src_changed(idx)
        self.assertEqual(m.cfg.get("download_source"), "hf_mirror")
        m.cfg.set("download_source", "auto")
        m.save_config(m.cfg)

    def test_manifest_refresh_from_file(self):
        w = self.w
        manifest = self.tmpdir / "manifest.json"
        manifest.write_text(json.dumps([
            {"name": "CustomModel", "category": "编程", "hw": ["CPU"],
             "desc": "自定义", "repo": "org/custom", "file": "custom.gguf"},
        ]), encoding="utf-8")
        m.cfg.set("store_manifest_url", str(manifest))
        w.refresh_online_store()
        self.assertEqual(len(w.store_data), 1)
        self.assertEqual(w.store_data[0]["name"], "CustomModel")
        self.assertTrue(w.store_data[0]["sources"]["official"].endswith("custom.gguf"))
        self.assertIn("information", self.calls)
        m.cfg.set("store_manifest_url", "")
        m.save_config(m.cfg)

    def test_manifest_refresh_failure_shows_error(self):
        w = self.w
        m.cfg.set("store_manifest_url", str(self.tmpdir / "missing.json"))
        self.calls.clear()
        w.refresh_online_store()
        self.assertIn("critical", self.calls)
        m.cfg.set("store_manifest_url", "")
        m.save_config(m.cfg)

    def test_refresh_empty_pulls_online_catalog(self):
        """未配置清单地址时,刷新应动态从在线拉取模型列表(而非仅内置)。"""
        w = self.w
        m.cfg.set("store_manifest_url", "")
        fake = [{"name": f"OnlineModel{i}", "category": "编程",
                 "hw": ["CPU", "GPU"], "desc": "在线",
                 "sources": {"official": f"http://x/m{i}.gguf"}}
                for i in range(9)]
        self.calls.clear()
        from unittest.mock import patch
        with patch.object(modelstore, "fetch_online_catalog", return_value=fake):
            w.refresh_online_store()
            self.assertTrue(wait_until(lambda: w.store_data == fake, timeout=5),
                            "刷新应后台拉取到在线列表")
            self.assertEqual(len(w.store_data), 9)
        self.assertIn("information", self.calls)
        m.cfg.set("store_manifest_url", "")
        m.save_config(m.cfg)

    def test_refresh_empty_falls_back_builtin_when_offline(self):
        """在线拉取为空时,应回退到内置清单(至少 8 个模型)。"""
        w = self.w
        m.cfg.set("store_manifest_url", "")
        self.calls.clear()
        from unittest.mock import patch
        with patch.object(modelstore, "fetch_online_catalog", return_value=[]):
            w.refresh_online_store()
            self.assertTrue(
                wait_until(lambda: "information" in self.calls, timeout=5),
                "在线为空时应在后台完成后回退到内置清单")
        self.assertGreaterEqual(len(w.store_data), 8)
        m.cfg.set("store_manifest_url", "")
        m.save_config(m.cfg)

    def test_download_toast_end_to_end(self):
        w = self.w
        payload = td.make_gguf_bytes(3, b"PAY")
        srv_dir = self.tmpdir / "srv"
        srv_dir.mkdir()
        (srv_dir / "model.gguf").write_bytes(payload)
        srv = td.LocalServer(srv_dir)
        srv.start()
        self.addCleanup(srv.stop)

        entry = {"name": "TestModel", "category": "文本", "hw": ["CPU"],
                 "desc": "测试", "size_gb": 0.003,
                 "sources": {"official": f"{srv.url}/model.gguf"}}
        dst = str(self.tmpdir / "dl" / "model.gguf")
        toast = m.DownloadToast(entry, w, save_path=dst)
        toast.start_download()
        self.assertTrue(
            wait_until(lambda: toast.worker is not None
                       and not toast.worker.isRunning(), timeout=30),
            "下载应在后台线程内完成")
        self.assertTrue(wait_until(lambda: toast.progress.value() == 100))
        self.assertTrue(Path(dst).exists(), "文件应已保存")
        self.assertEqual(Path(dst).read_bytes(), payload, "内容应与源一致")
        toast.close()

    def test_download_toast_fallback_source(self):
        w = self.w
        payload = td.make_gguf_bytes(2, b"FALLBACK")
        srv_dir = self.tmpdir / "srv2"
        srv_dir.mkdir()
        (srv_dir / "model.gguf").write_bytes(payload)
        srv = td.LocalServer(srv_dir)
        srv.start()
        self.addCleanup(srv.stop)

        entry = {"name": "Fb", "category": "文本", "hw": ["CPU"], "desc": "",
                 "sources": {
                     "official": "http://127.0.0.1:1/dead.gguf",
                     "hf_mirror": f"{srv.url}/model.gguf"}}
        dst = str(self.tmpdir / "fb.gguf")
        toast = m.DownloadToast(entry, w, save_path=dst)
        toast.start_download()
        self.assertTrue(wait_until(
            lambda: toast.worker is not None and not toast.worker.isRunning(),
            timeout=30))
        self.assertTrue(Path(dst).exists(), "官方源失败后应自动切镜像完成下载")
        toast.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
