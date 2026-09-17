# -*- coding: utf-8 -*-
"""updater 自测:版本比较、Release 解析、下载、更新脚本生成(网络全部 mock)。
直接运行 `python tests/test_updater.py`。"""
import io
import os
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from novacore import updater


class FakeResp:
    def __init__(self, data=None, status=200, content=b"", headers=None):
        self._data = data
        self.status_code = status
        self._content = content
        self.headers = headers or {}

    def json(self):
        if self._data is None:
            raise ValueError("no json")
        return self._data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def iter_content(self, size=1 << 20):
        for i in range(0, len(self._content), size):
            yield self._content[i:i + size]

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class TestVersion(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(updater._parse_version("v0.0.2-alpha"), (0, 0, 2))
        self.assertEqual(updater._parse_version("1.2.3"), (1, 2, 3))
        self.assertEqual(updater._parse_version(""), (0,))

    def test_is_newer(self):
        self.assertTrue(updater.is_newer("v0.0.2", "0.0.1-alpha"))
        self.assertTrue(updater.is_newer("0.1.0", "0.0.9"))
        self.assertFalse(updater.is_newer("v0.0.1-alpha", "0.0.1-alpha"))
        self.assertFalse(updater.is_newer("0.0.0", "0.0.1-alpha"))
        # 数字相同但 tag 不同 -> 视为新构建
        self.assertTrue(updater.is_newer("0.0.1-beta", "0.0.1-alpha"))


class TestParseRelease(unittest.TestCase):
    def test_normalize_picks_zip_asset(self):
        data = {
            "tag_name": "v0.0.2-alpha",
            "body": "修复若干问题",
            "html_url": "https://github.com/x/y/releases/tag/v0.0.2-alpha",
            "assets": [
                {"name": "checksums.txt",
                 "browser_download_url": "https://x/checksums.txt"},
                {"name": "NovaCore-Local-v0.0.2-alpha-win64.zip",
                 "browser_download_url": "https://x/app.zip"},
            ],
        }
        rel = updater._normalize_release(data)
        self.assertEqual(rel["version"], "0.0.2-alpha")
        self.assertEqual(rel["notes"], "修复若干问题")
        self.assertEqual(rel["asset_name"], "NovaCore-Local-v0.0.2-alpha-win64.zip")
        self.assertEqual(rel["asset_url"], "https://x/app.zip")

    def test_latest_release_mocked(self):
        with mock.patch("novacore.updater.requests.get",
                        return_value=FakeResp({"tag_name": "v9.9.9",
                                               "body": "n",
                                               "html_url": "h",
                                               "assets": []})):
            rel = updater.latest_release()
        self.assertEqual(rel["version"], "9.9.9")
        self.assertEqual(rel["source"], "github")

    def test_from_custom_json(self):
        with mock.patch("novacore.updater.requests.get",
                        return_value=FakeResp({"version": "2.0.0",
                                               "url": "https://dl/2.0.0",
                                               "notes": "x"})):
            rel = updater.from_custom_json("https://example.com/u.json")
        self.assertEqual(rel["version"], "2.0.0")
        self.assertEqual(rel["url"], "https://dl/2.0.0")
        self.assertEqual(rel["source"], "custom")


class TestDownload(unittest.TestCase):
    def test_download_asset_writes_file(self):
        payload = b"PK\x03\x04" + b"0" * (3 * 1024 * 1024)
        with tempfile.TemporaryDirectory() as d:
            dest = os.path.join(d, "app.zip")
            got = []
            with mock.patch("novacore.updater.requests.get",
                            return_value=FakeResp(content=payload,
                                                  headers={"Content-Length": str(len(payload))})):
                updater.download_asset("https://x/app.zip", dest,
                                       progress=lambda dn, t: got.append((dn, t)))
            self.assertTrue(os.path.exists(dest))
            self.assertEqual(os.path.getsize(dest), len(payload))
            self.assertFalse(os.path.exists(dest + ".part"))
            self.assertTrue(got and got[-1][0] == len(payload))

    def test_download_cancel(self):
        payload = b"x" * (2 * 1024 * 1024)
        with tempfile.TemporaryDirectory() as d:
            dest = os.path.join(d, "app.zip")
            with mock.patch("novacore.updater.requests.get",
                            return_value=FakeResp(content=payload)):
                with self.assertRaises(InterruptedError):
                    updater.download_asset("https://x/app.zip", dest,
                                           cancel_check=lambda: True)


class TestUpdateScript(unittest.TestCase):
    def test_script_is_ascii_and_param_based(self):
        script = updater.build_update_script()
        # 脚本必须纯 ASCII(避免 PowerShell 5.1 按 ANSI 读中文脚本导致乱码)
        script.encode("ascii")
        self.assertIn("param(", script)
        self.assertIn("$Zip", script)
        self.assertIn("$App", script)
        self.assertIn("$Exe", script)
        self.assertIn("$AppPid", script)
        self.assertIn("Expand-Archive", script)
        self.assertIn("Copy-Item", script)
        self.assertIn("Start-Process", script)
        # 路径不再硬编码进脚本
        self.assertNotIn("NovaCore-Local.exe", script)

    def test_source_mode_no_self_update(self):
        # 单元测试在源码模式下运行,应报告不支持自更新
        self.assertFalse(updater.supports_self_update())
        self.assertIsNone(updater.current_app_dir())


if __name__ == "__main__":
    unittest.main(verbosity=2)
