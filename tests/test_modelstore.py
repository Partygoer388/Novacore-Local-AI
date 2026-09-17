# -*- coding: utf-8 -*-
"""novacore.modelstore 自测:直接运行 `python tests/test_modelstore.py`。
重点:内置清单真实可用、三种下载源 URL 构造正确、modelscope 缺失自动回退、manifest 加载容错。"""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from novacore import modelstore as ms


class TestBuiltinStore(unittest.TestCase):

    def test_non_empty(self):
        store = ms.get_builtin_store()
        self.assertGreaterEqual(len(store), 6)
        for m in store:
            self.assertTrue(m["name"])
            self.assertTrue(m["desc"])
            self.assertIn(m["category"], ("文本", "编程", "多模态", "配音"))

    def test_all_have_official_and_mirror_urls(self):
        # 仅 GGUF 模型有统一 sources;openvino_dir 按文件逐个解析
        for m in ms.get_builtin_store():
            if m.get("format") == "openvino_dir":
                continue  # NPU 专区目录模型无统一 sources
            self.assertTrue(m["sources"]["official"].startswith(
                "https://huggingface.co/"), m["name"])
            self.assertTrue(m["sources"]["hf_mirror"].startswith(
                "https://hf-mirror.com/"), m["name"])
            self.assertIn("/resolve/main/", m["sources"]["official"])
            self.assertTrue(m["sources"]["official"].endswith(".gguf"))

    def test_no_dummy_urls(self):
        """关键修复:内置清单绝不能有 example.org 占位地址。"""
        for m in ms.get_builtin_store():
            srcs = m.get("sources") or {}
            for k, v in srcs.items():
                if v:
                    self.assertNotIn("example.org", v, f"{m['name']} {k}")

    def test_qwen_models_have_modelscope(self):
        # 仅检查 GGUF(CPU/GPU)模型;NPU 专区(openvino_dir)按文件逐个解析,无统一 sources
        qwen = [m for m in ms.get_builtin_store()
                if m["name"].startswith("Qwen")
                and m.get("format") != "openvino_dir"]
        self.assertGreater(len(qwen), 0)
        for m in qwen:
            self.assertTrue(m["sources"]["modelscope"],
                            f"{m['name']} 应有魔搭源")
            self.assertTrue(m["sources"]["modelscope"].startswith(
                "https://modelscope.cn/"))

    def test_modelscope_fallback_to_hf_mirror(self):
        """没有魔搭源的模型选 modelscope 时应回退 hf-mirror。"""
        llama = [m for m in ms.get_builtin_store()
                 if m["name"].startswith("Llama")
                 and m.get("format") != "openvino_dir"][0]
        self.assertIsNone(llama["sources"]["modelscope"])
        url = ms.pick_source_url(llama, "modelscope")
        self.assertTrue(url.startswith("https://hf-mirror.com/"))

    def test_resolve_source_auto_prefers_official(self):
        m = [x for x in ms.get_builtin_store()
             if x.get("format") != "openvino_dir"][0]
        self.assertEqual(ms.resolve_source("auto", m), m["sources"]["official"])
        self.assertEqual(ms.resolve_source("hf_mirror", m),
                         m["sources"]["hf_mirror"])
        self.assertEqual(ms.resolve_source("official", m),
                         m["sources"]["official"])

    def test_source_fallback_list_order_and_dedup(self):
        """故障转移顺序:auto/official → [official, hf_mirror, modelscope]。"""
        m = [x for x in ms.get_builtin_store()
             if x.get("format") != "openvino_dir"][0]  # Qwen,三源齐全
        lst = ms.source_fallback_list(m, "auto")
        ids = [sid for sid, _ in lst]
        self.assertEqual(ids, ["official", "hf_mirror", "modelscope"])
        # hf_mirror 偏好:镜像优先,官方兜底
        ids = [sid for sid, _ in ms.source_fallback_list(m, "hf_mirror")]
        self.assertEqual(ids, ["hf_mirror", "official", "modelscope"])

    def test_source_fallback_list_skips_missing(self):
        """缺失的源(如无魔搭)自动跳过,不产生 None 条目。"""
        llama = [m for m in ms.get_builtin_store()
                 if m["name"].startswith("Llama")][0]
        lst = ms.source_fallback_list(llama, "auto")
        self.assertEqual([sid for sid, _ in lst], ["official", "hf_mirror"])
        for _, url in lst:
            self.assertTrue(url and url.startswith("http"))

    def test_source_fallback_list_dedup_same_url(self):
        entry = {"sources": {"official": "https://x/a.gguf",
                             "hf_mirror": "https://x/a.gguf"}}
        lst = ms.source_fallback_list(entry, "auto")
        self.assertEqual(len(lst), 1)


class TestManifest(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def test_normalize_valid_entries(self):
        data = [
            {"name": "A", "repo": "org/a", "file": "a.gguf"},
            {"name": "B", "sources": {"official": "https://huggingface.co/org/b/resolve/main/b.gguf"}},
            {"name": "", "repo": "org/c", "file": "c.gguf"},        # 空名 → 丢弃
            {"name": "D"},                                           # 无源 → 丢弃
            "not-a-dict",                                            # 非 dict → 丢弃
        ]
        out = ms.normalize_manifest(data)
        self.assertEqual([m["name"] for m in out], ["A", "B"])
        self.assertTrue(out[0]["sources"]["hf_mirror"])
        self.assertTrue(out[1]["sources"]["hf_mirror"] is None)

    def test_manifest_wrapped_in_models_key(self):
        out = ms.normalize_manifest({"models": [
            {"name": "A", "repo": "org/a", "file": "a.gguf"}]})
        self.assertEqual(len(out), 1)

    def test_load_manifest_from_file(self):
        p = self.dir / "manifest.json"
        p.write_text(json.dumps([
            {"name": "Custom", "repo": "org/custom", "file": "custom.gguf",
             "ms_repo": "org/custom-ms", "ms_file": "custom.gguf"},
        ]), encoding="utf-8")
        out = ms.load_manifest(str(p))
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["name"], "Custom")
        self.assertTrue(out[0]["sources"]["modelscope"])

    def test_load_manifest_missing_file(self):
        self.assertEqual(ms.load_manifest(str(self.dir / "nope.json")), [])

    def test_load_manifest_bad_json(self):
        p = self.dir / "bad.json"
        p.write_text("{oops", encoding="utf-8")
        self.assertEqual(ms.load_manifest(str(p)), [])

    def test_load_manifest_from_url(self):
        with mock.patch("novacore.modelstore.requests.get") as get:
            resp = mock.Mock()
            resp.raise_for_status.return_value = None
            resp.text = json.dumps([
                {"name": "Remote", "repo": "org/r", "file": "r.gguf"}])
            get.return_value = resp
            out = ms.load_manifest("https://example.org/manifest.json")
        self.assertEqual([m["name"] for m in out], ["Remote"])

    def test_load_manifest_url_error(self):
        with mock.patch("novacore.modelstore.requests.get",
                        side_effect=ConnectionError("no net")):
            self.assertEqual(ms.load_manifest("https://example.org/x.json"), [])

    def test_save_manifest_roundtrip(self):
        p = self.dir / "out.json"
        models = [{"name": "A", "repo": "org/a", "file": "a.gguf",
                   "category": "文本", "hw": ["CPU"], "desc": "", "size_gb": 1.0,
                   "sources": ms.build_sources("org/a", "a.gguf")}]
        self.assertTrue(ms.save_manifest(p, models))
        out = ms.load_manifest(str(p))
        self.assertEqual([m["name"] for m in out], ["A"])


class TestOnlineCatalog(unittest.TestCase):
    """动态在线列表:搜索优先 + 仓库兜底 + 文件量化选择,均不依赖网络。"""

    def setUp(self):
        # 隔离磁盘缓存,避免上个用例写入的缓存污染后续断言
        p1 = mock.patch.object(ms, "_load_catalog_cache", return_value=None)
        p2 = mock.patch.object(ms, "_save_catalog_cache", return_value=None)
        self.addCleanup(p1.stop)
        self.addCleanup(p2.stop)
        p1.start()
        p2.start()

    def test_fetch_online_catalog_uses_search_and_builds_sources(self):
        repos = ["org/A-GGUF", "org/B-GGUF", "org/C-GGUF"]

        def fake_entry(repo):
            return {"name": repo.split("/")[-1], "category": "文本",
                    "hw": ["CPU", "GPU"], "desc": repo, "size_gb": 1.0,
                    "sources": ms.build_sources(repo, "m.gguf")}

        with mock.patch.object(ms, "_search_gguf_repos", return_value=repos), \
             mock.patch.object(ms, "_ms_search_gguf_repos", return_value=[]), \
             mock.patch.object(ms, "_repo_to_entry", side_effect=fake_entry):
            out = ms.fetch_online_catalog(limit=3)
        self.assertEqual([m["name"] for m in out],
                         ["A-GGUF", "B-GGUF", "C-GGUF"])
        self.assertEqual(
            out[0]["sources"]["official"],
            "https://huggingface.co/org/A-GGUF/resolve/main/m.gguf")
        self.assertTrue(out[0]["sources"]["hf_mirror"].startswith(
            "https://hf-mirror.com/"))

    def test_fetch_online_catalog_falls_back_to_popular(self):
        with mock.patch.object(ms, "_search_gguf_repos", return_value=[]), \
             mock.patch.object(ms, "_ms_search_gguf_repos", return_value=[]), \
             mock.patch.object(ms, "_repo_to_entry",
                               side_effect=lambda r: {
                                   "name": r.split("/")[-1], "category": "文本",
                                   "hw": ["CPU"], "desc": r, "size_gb": 1.0,
                                   "sources": ms.build_sources(r, "m.gguf")}):
            out = ms.fetch_online_catalog(limit=3)
        # 搜索为空 → 应返回来自热门仓库清单的 3 个
        self.assertGreaterEqual(len(out), 3)
        self.assertTrue(any(m["name"].startswith("Qwen") for m in out))

    def test_repo_to_entry_picks_q4_k_m(self):
        files = [
            {"path": "m-f16.gguf", "size": 9000},
            {"path": "m-q4_k_m.gguf", "size": 3000},
            {"path": "m-q8_0.gguf", "size": 5000},
        ]
        with mock.patch.object(ms, "_list_gguf_files", return_value=files):
            e = ms._repo_to_entry("org/Model-GGUF")
        self.assertIsNotNone(e)
        self.assertEqual(e["name"], "Model-GGUF")
        self.assertTrue(e["sources"]["official"].endswith("m-q4_k_m.gguf"))

    def test_repo_to_entry_none_when_no_gguf(self):
        with mock.patch.object(ms, "_list_gguf_files", return_value=[]):
            self.assertIsNone(ms._repo_to_entry("org/Empty-GGUF"))

    def test_ms_repo_to_entry_builds_modelscope_source(self):
        files = [
            {"path": "m-f16.gguf", "size": 9000},
            {"path": "m-q4_k_m.gguf", "size": 3000},
            {"path": "config.json", "size": 50},
        ]
        with mock.patch.object(ms, "_ms_list_gguf_files", return_value=files):
            e = ms._ms_repo_to_entry("unsloth/DeepSeek-R1-GGUF")
        self.assertIsNotNone(e)
        self.assertEqual(e["name"], "DeepSeek-R1-GGUF")
        self.assertTrue(
            e["sources"]["modelscope"].endswith(
                "/unsloth/DeepSeek-R1-GGUF/resolve/master/m-q4_k_m.gguf"))

    def test_ms_repo_to_entry_none_when_no_gguf(self):
        with mock.patch.object(ms, "_ms_list_gguf_files", return_value=[]):
            self.assertIsNone(ms._ms_repo_to_entry("org/Empty-GGUF"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
