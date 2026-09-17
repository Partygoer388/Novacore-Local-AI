# -*- coding: utf-8 -*-
"""EngineManager 自动探测与选择自测:直接运行 `python tests/test_engine_manager.py`。
使用可控的假引擎,不依赖真实模型/服务/网络。"""
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from novacore.config import Config
from novacore.engines import EngineManager
from novacore.engines.base import (
    BaseEngine, GenParams, ModelLoadError, ModelNotLoadedError,
)


class FakeEngine(BaseEngine):
    engine_id = "fake"

    def __init__(self, available=True, detail="ok"):
        super().__init__()
        self._available = available
        self._detail = detail
        self.load_calls = []
        self.gen_calls = 0
        self.cancel_calls = 0

    def check_available(self):
        return self._available, self._detail

    def load_model(self, path, device=None, progress=None):
        if not self._available:
            raise ModelLoadError("unavailable")
        self.load_calls.append((path, device))
        self._loaded_path = path
        self._model_name = path

    def unload(self):
        self._loaded_path = None
        self._model_name = None

    def generate(self, messages, params):
        self.gen_calls += 1
        for tok in ["a", "b"]:
            self._check_cancel()
            yield tok

    def cancel(self):
        self.cancel_calls += 1
        super().cancel()


def make_manager(**avail):
    mgr = EngineManager()
    mgr._engines = {}
    mgr._order = []
    for eid, (av, detail) in avail.items():
        fe = FakeEngine(available=av, detail=detail)
        fe.engine_id = eid
        mgr._engines[eid] = fe
        mgr._order.append(eid)
    return mgr


class TestProbe(unittest.TestCase):

    def test_probe_all_flags(self):
        mgr = make_manager(llamacpp=(True, "v1"), ollama=(False, "down"),
                           openvino=(True, "npu"))
        info = {i["id"]: i for i in mgr.probe_all()}
        self.assertIs(info["llamacpp"]["available"], True)
        self.assertIs(info["ollama"]["available"], False)
        self.assertIn("down", info["ollama"]["detail"])

    def test_probe_cached(self):
        mgr = make_manager(llamacpp=(True, "v1"))
        self.assertEqual(mgr.probe("llamacpp"), (True, "v1"))
        mgr._engines["llamacpp"]._available = False
        mgr._engines["llamacpp"]._detail = "down"
        # 缓存期内不重新探测
        self.assertEqual(mgr.probe("llamacpp"), (True, "v1"))
        # refresh=True 强制重探测
        self.assertEqual(mgr.probe("llamacpp", refresh=True), (False, "down"))

    def test_probe_exception_caught(self):
        mgr = make_manager(llamacpp=(True, "v1"))
        mgr._engines["llamacpp"].check_available = lambda: (_ for _ in ()).throw(
            RuntimeError("boom"))
        ok, detail = mgr.probe("llamacpp", refresh=True)
        self.assertFalse(ok)
        self.assertIn("探测异常", detail)


class TestPick(unittest.TestCase):

    def test_explicit_engine_available(self):
        mgr = make_manager(llamacpp=(True, ""), ollama=(True, ""), openvino=(False, ""))
        cfg = Config({"default_engine": "ollama", "prefer_device": "CPU"})
        self.assertEqual(mgr.pick_engine(cfg), "ollama")

    def test_explicit_engine_unavailable_falls_back(self):
        mgr = make_manager(llamacpp=(True, ""), ollama=(False, ""))
        cfg = Config({"default_engine": "ollama", "prefer_device": "CPU"})
        self.assertEqual(mgr.pick_engine(cfg), "llamacpp")

    def test_npu_preferred_with_npu_hardware(self):
        mgr = make_manager(llamacpp=(True, ""), openvino=(True, ""))
        cfg = Config({"default_engine": "auto", "prefer_device": "NPU-OpenVINO"})
        with mock.patch("novacore.engines.hardware.has_npu", return_value=True):
            self.assertEqual(mgr.pick_engine(cfg), "openvino")

    def test_gpu_preferred_with_cuda(self):
        mgr = make_manager(llamacpp=(True, ""), ollama=(True, ""))
        cfg = Config({"default_engine": "auto", "prefer_device": "GPU"})
        with mock.patch("novacore.engines.hardware.has_cuda_gpu", return_value=True), \
                mock.patch("novacore.engines.hardware.has_npu", return_value=False):
            self.assertEqual(mgr.pick_engine(cfg), "llamacpp")

    def test_none_available_returns_llamacpp_for_guidance(self):
        mgr = make_manager(llamacpp=(False, "not installed"),
                           ollama=(False, "no server"), openvino=(False, "no"))
        cfg = Config({"default_engine": "auto", "prefer_device": "CPU"})
        # 全部不可用也要返回引擎 id,便于界面提示安装引导
        self.assertEqual(mgr.pick_engine(cfg), "llamacpp")

    def test_cfg_none_uses_defaults(self):
        mgr = make_manager(llamacpp=(False, ""), ollama=(True, ""))
        self.assertEqual(mgr.pick_engine(None), "ollama")


class TestOps(unittest.TestCase):

    def test_load_sets_current(self):
        mgr = make_manager(llamacpp=(True, ""), ollama=(True, ""))
        engine = mgr.load("model.gguf", engine_id="llamacpp", device="GPU")
        self.assertEqual(mgr.current_id, "llamacpp")
        self.assertTrue(engine.loaded)
        self.assertEqual(engine.load_calls, [("model.gguf", "GPU")])

    def test_generate_delegates_and_cancel(self):
        mgr = make_manager(llamacpp=(True, ""))
        mgr.load("m.gguf", engine_id="llamacpp")
        out = "".join(mgr.generate([{"role": "user", "content": "x"}], GenParams()))
        self.assertEqual(out, "ab")
        mgr.cancel()
        self.assertEqual(mgr._engines["llamacpp"].cancel_calls, 1)

    def test_generate_without_engine_raises(self):
        mgr = make_manager(llamacpp=(True, ""))
        with self.assertRaises(ModelNotLoadedError):
            list(mgr.generate([], GenParams()))

    def test_unload_all(self):
        mgr = make_manager(llamacpp=(True, ""), ollama=(True, ""))
        mgr.load("m.gguf", engine_id="llamacpp")
        mgr.load("qwen", engine_id="ollama")
        mgr.unload_all()
        self.assertFalse(mgr.is_loaded())
        for e in mgr._engines.values():
            self.assertFalse(e.loaded)

    def test_unknown_engine_raises(self):
        mgr = make_manager(llamacpp=(True, ""))
        with self.assertRaises(KeyError):
            mgr.get("nope")


if __name__ == "__main__":
    unittest.main(verbosity=2)
