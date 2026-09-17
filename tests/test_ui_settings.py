# -*- coding: utf-8 -*-
"""系统设置页自测(离屏):回归「切换主题保存设置」时 QComboBox 被 GC 删除的崩溃。
直接运行 `python tests/test_ui_settings.py`。"""
import os
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PyQt6.QtWidgets import QApplication, QMessageBox, QComboBox

_app = QApplication.instance() or QApplication([])

import novacore_main as m  # noqa: E402


def pump(seconds: float):
    end = time.time() + seconds
    while time.time() < end:
        _app.processEvents()
        time.sleep(0.005)


class TestSettingsPage(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # 拦截模态对话框(离屏下 exec 会永久阻塞)
        cls._orig = {}
        for name in ("warning", "information", "critical", "question"):
            cls._orig[name] = getattr(QMessageBox, name)
            setattr(QMessageBox, name,
                    staticmethod(lambda *a, **k: QMessageBox.StandardButton.Ok))

    @classmethod
    def tearDownClass(cls):
        for name, orig in cls._orig.items():
            setattr(QMessageBox, name, orig)

    def setUp(self):
        # 防止启动钩子去拉起真实 Ollama 服务(钩子在 MainWindow 创建时注册)
        m.cfg.set("ollama_auto_start", False)
        self.w = m.MainWindow()
        self.w.show()
        pump(0.3)
        self.w.switch_page(6)
        pump(0.2)

    def tearDown(self):
        self.w.close()
        pump(0.2)

    def test_pip_mirror_combo_alive_and_parented(self):
        """pip_mirror_sel 必须仍在布局树中(旧版 gb_pip 未加入页面被 GC,
        导致保存设置时 RuntimeError: QComboBox has been deleted)。"""
        combo = self.w.pip_mirror_sel
        self.assertIsInstance(combo, QComboBox)
        data = combo.currentData()  # 旧版这里直接抛 RuntimeError
        self.assertIn(data, ["auto", "official", "tsinghua", "aliyun",
                             "douban", "ustc", "huawei", "tencent",
                             "netease", "sjtu"])
        # 必须挂在可见的祖先链上(MainWindow)
        self.assertIsNotNone(combo.parentWidget())
        ancestor = combo.parentWidget()
        while ancestor.parentWidget() is not None:
            ancestor = ancestor.parentWidget()
        self.assertIs(ancestor, self.w)

    def test_save_settings_with_theme_switch(self):
        """切换主题 → 保存设置:不得崩溃,配置应更新并即时生效。"""
        for theme in ("light", "deep_black", "soft_dark"):
            idx = m.THEME_CHOICES.index(theme)
            self.w.theme_sel.setCurrentIndex(idx)
            pump(0.1)
            self.w.save_settings()  # 旧版在此抛 RuntimeError
            pump(0.1)
            self.assertEqual(m.cfg.get("theme"), theme)
        # 恢复默认主题,避免影响其它测试
        self.w.theme_sel.setCurrentIndex(m.THEME_CHOICES.index("soft_dark"))
        self.w.save_settings()
        # 确保启动钩子不会去拉起真实 Ollama 服务
        m.cfg.set("ollama_auto_start", False)
        m.save_config(m.cfg)

    def test_startup_auto_start_ollama_hook(self):
        """回归:勾选「自动启动 Ollama 服务」后,程序打开时应后台自动拉起服务。"""
        m.cfg.set("ollama_auto_start", True)
        m.save_config(m.cfg)
        w2 = m.MainWindow()
        calls = []
        w2._ensure_ollama_auto = lambda: calls.append(1) or (True, "在线")
        pump(1.8)
        self.assertEqual(calls, [1], "启动钩子应在打开后自动调用服务确保逻辑")
        self.assertIn("✅", w2.ollama_status_lab.text())
        w2.close()
        pump(0.3)
        m.cfg.set("ollama_auto_start", False)
        m.save_config(m.cfg)

    def test_manual_ollama_start_button(self):
        """设置页应有「启动服务」按钮,点击后走后台启动并更新状态。"""
        w = self.w
        calls = []
        w._ollama_start_worker = lambda: calls.append(1) or (True, "服务在线")
        w.start_ollama_manually()
        pump(0.6)
        self.assertEqual(calls, [1])
        self.assertIn("✅", w.ollama_status_lab.text())

    def test_save_settings_roundtrip_all_combos(self):
        """所有设置控件在保存后仍可正常访问(无被删除的 C/C++ 对象)。"""
        self.w.save_settings()
        pump(0.1)
        combos = [
            self.w.engine_sel, self.w.prefer_device_sel, self.w.theme_sel,
            self.w.ui_scale_sel, self.w.pip_mirror_sel,
            self.w.download_source_sel,
        ]
        for c in combos:
            self.assertIsInstance(c, QComboBox)
            _ = c.currentText()  # 不抛 RuntimeError 即通过
        # 恢复默认主题
        self.w.theme_sel.setCurrentIndex(m.THEME_CHOICES.index("soft_dark"))
        self.w.save_settings()


if __name__ == "__main__":
    unittest.main(verbosity=2)
