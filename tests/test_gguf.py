# -*- coding: utf-8 -*-
"""novacore.gguf 自测:直接运行 `python tests/test_gguf.py`。"""
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from novacore import gguf


def build_fake_gguf(path: Path, n_tensors: int = 1, pad_mb: int = 2):
    """构造一个结构合法的 GGUF 文件(含元数据 + 填充)。"""
    def s(b: bytes) -> bytes:
        return struct.pack("<Q", len(b)) + b

    buf = bytearray()
    buf += gguf.GGUF_MAGIC
    buf += struct.pack("<I", 3)          # version
    buf += struct.pack("<Q", n_tensors)  # tensor count
    kv = [
        ("general.name", 8, s("TestModel".encode())),
        ("general.architecture", 8, s("llama".encode())),
        ("llama.context_length", 4, struct.pack("<I", 2048)),
    ]
    buf += struct.pack("<Q", len(kv))
    for key, vtype, val in kv:
        buf += s(key.encode())
        buf += struct.pack("<I", vtype)
        buf += val
    buf += b"\x00" * (pad_mb * 1024 * 1024)  # 模拟张量数据区
    path.write_bytes(bytes(buf))


class TestGguf(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def test_parse_valid_gguf(self):
        p = self.dir / "model.gguf"
        build_fake_gguf(p)
        parsed = gguf.parse_gguf_header(p)
        self.assertEqual(parsed["version"], 3)
        self.assertEqual(parsed["tensor_count"], 1)
        self.assertEqual(parsed["metadata"]["general.name"], "TestModel")
        self.assertEqual(parsed["metadata"]["general.architecture"], "llama")

    def test_parse_not_gguf_raises(self):
        p = self.dir / "junk.bin"
        p.write_bytes(b"PK\x03\x04 not a gguf " * 100)
        with self.assertRaises(ValueError):
            gguf.parse_gguf_header(p)

    def test_validate_valid_gguf(self):
        p = self.dir / "model.gguf"
        build_fake_gguf(p)
        ok, reason, info = gguf.validate_model_file(p)
        self.assertTrue(ok, reason)
        self.assertEqual(info["model_name"], "TestModel")
        self.assertEqual(info["architecture"], "llama")
        self.assertEqual(info["context_length"], 2048)

    def test_validate_html_fake_model(self):
        """关键场景:把网页当模型下载下来的假文件必须被识别。"""
        p = self.dir / "fake.gguf"
        p.write_bytes(b"<!doctype html><html><title>Example Domain</title></html>")
        ok, reason, info = gguf.validate_model_file(p)
        self.assertFalse(ok)
        self.assertIn("网页", reason)

    def test_validate_too_small(self):
        p = self.dir / "tiny.gguf"
        p.write_bytes(b"GGUF" + b"\x00" * 100)
        ok, reason, _ = gguf.validate_model_file(p)
        self.assertFalse(ok)
        self.assertIn("过小", reason)

    def test_validate_unknown_format(self):
        p = self.dir / "notes.txt"
        p.write_bytes(b"hello world " * 200000)  # >1MB 但非 GGUF
        ok, reason, _ = gguf.validate_model_file(p)
        self.assertFalse(ok)
        self.assertIn("无法识别", reason)

    def test_validate_empty_tensor_gguf(self):
        p = self.dir / "empty.gguf"
        build_fake_gguf(p, n_tensors=0)
        ok, reason, _ = gguf.validate_model_file(p)
        self.assertFalse(ok)
        self.assertIn("没有任何张量", reason)

    def test_scan_mixed_directory(self):
        good = self.dir / "good.gguf"
        build_fake_gguf(good)
        bad = self.dir / "fake.gguf"
        bad.write_bytes(b"<html>placeholder page</html>")
        (self.dir / "subdir").mkdir()
        items = gguf.scan_local_models(self.dir)
        by_name = {i["name"]: i for i in items}
        self.assertIs(by_name["good.gguf"]["valid"], True)
        self.assertIs(by_name["fake.gguf"]["valid"], False)
        self.assertIn("网页", by_name["fake.gguf"]["reason"])
        self.assertIs(by_name["subdir"]["valid"], False)
        self.assertIn("文件夹", by_name["subdir"]["reason"])

    def test_real_dummy_model_detected(self):
        """针对本仓库真实存在的 559 字节假模型 Phi3‑mini‑NPU 的检测。"""
        import novacore.paths as paths
        dummy = paths.LOCAL_MODEL_DIR / "Phi3\u2011mini\u2011NPU"
        if dummy.exists():
            ok, reason, _ = gguf.validate_model_file(dummy)
            self.assertFalse(ok)
            self.assertIn("网页", reason)


if __name__ == "__main__":
    unittest.main(verbosity=2)
