# -*- coding: utf-8 -*-
"""novacore.config 自测:直接运行 `python tests/test_config.py`。"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import novacore.config as cfg_mod


class TestConfig(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._orig = cfg_mod.CONFIG_FILE
        cfg_mod.CONFIG_FILE = Path(self._tmp.name) / "config.json"
        self.addCleanup(self._restore)

    def _restore(self):
        cfg_mod.CONFIG_FILE = self._orig

    def test_first_load_creates_defaults(self):
        cfg = cfg_mod.load_config()
        self.assertEqual(cfg.get("api_port"), 8000)
        self.assertEqual(cfg.get("default_engine"), "auto")
        self.assertTrue(cfg_mod.CONFIG_FILE.exists())
        with open(cfg_mod.CONFIG_FILE, encoding="utf-8") as f:
            on_disk = json.load(f)
        self.assertEqual(on_disk["api_port"], 8000)

    def test_missing_keys_merged(self):
        cfg_mod.CONFIG_FILE.write_text(json.dumps({"api_port": 9000}), encoding="utf-8")
        cfg = cfg_mod.load_config()
        self.assertEqual(cfg.get("api_port"), 9000)
        self.assertEqual(cfg.get("api_enable"), False)      # 补默认
        self.assertEqual(cfg.get("gen_temperature"), 0.7)   # 补默认

    def test_type_coercion(self):
        cfg_mod.CONFIG_FILE.write_text(json.dumps({
            "api_port": "9999",          # str -> int
            "api_enable": "true",        # str -> bool
            "gen_temperature": "0.5",    # str -> float
        }), encoding="utf-8")
        cfg = cfg_mod.load_config()
        self.assertEqual(cfg.get("api_port"), 9999)
        self.assertIs(cfg.get("api_enable"), True)
        self.assertEqual(cfg.get("gen_temperature"), 0.5)

    def test_invalid_values_fallback(self):
        cfg_mod.CONFIG_FILE.write_text(json.dumps({
            "api_port": 80,                    # 越界 -> 8000
            "default_engine": "bogus",         # 非法 -> auto
            "gen_temperature": 9.9,            # 越界 -> 0.7
            "prefer_device": 12345,            # 非法类型 -> CPU
        }), encoding="utf-8")
        cfg = cfg_mod.load_config()
        self.assertEqual(cfg.get("api_port"), 8000)
        self.assertEqual(cfg.get("default_engine"), "auto")
        self.assertEqual(cfg.get("gen_temperature"), 0.7)
        self.assertEqual(cfg.get("prefer_device"), "CPU")

    def test_corrupted_json_backup_and_rebuild(self):
        cfg_mod.CONFIG_FILE.write_text("{ not valid json !!!", encoding="utf-8")
        cfg = cfg_mod.load_config()
        self.assertEqual(cfg.get("api_port"), 8000)
        bak = cfg_mod.CONFIG_FILE.with_suffix(".json.bak")
        self.assertTrue(bak.exists(), "损坏配置应被备份")
        with open(bak, encoding="utf-8") as f:
            self.assertIn("not valid", f.read())

    def test_unknown_keys_dropped_extra_preserved(self):
        cfg_mod.CONFIG_FILE.write_text(json.dumps({
            "api_port": 8123, "hacker_key": "drop-me", "token_stat_enable": True,
        }), encoding="utf-8")
        cfg = cfg_mod.load_config()
        self.assertEqual(cfg.get("api_port"), 8123)
        with open(cfg_mod.CONFIG_FILE, encoding="utf-8") as f:
            on_disk = json.load(f)
        self.assertNotIn("hacker_key", on_disk)
        self.assertIn("token_stat_enable", on_disk)

    def test_save_roundtrip(self):
        cfg = cfg_mod.load_config()
        cfg.set("api_port", 8100)
        cfg.set("prefer_device", "GPU")
        self.assertTrue(cfg_mod.save_config(cfg))
        cfg2 = cfg_mod.load_config()
        self.assertEqual(cfg2.get("api_port"), 8100)
        self.assertEqual(cfg2.get("prefer_device"), "GPU")

    def test_sanitize_non_dict(self):
        out = cfg_mod.sanitize(["not", "a", "dict"])
        self.assertEqual(out["api_port"], 8000)
        self.assertEqual(out["default_engine"], "auto")


if __name__ == "__main__":
    unittest.main(verbosity=2)
