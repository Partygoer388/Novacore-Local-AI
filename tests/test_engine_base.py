# -*- coding: utf-8 -*-
"""novacore.engines.base 自测:直接运行 `python tests/test_engine_base.py`。
用一个 DummyEngine 验证接口契约:加载/卸载/流式生成/取消/参数来自配置。"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from novacore.config import Config
from novacore.engines.base import (
    BaseEngine, EngineCapabilities, EngineError, EngineUnavailableError,
    ModelLoadError, ModelNotLoadedError, GenerationCancelled, GenParams,
)


class DummyEngine(BaseEngine):
    engine_id = "dummy"
    display_name = "Dummy"

    def check_available(self):
        return True, "测试引擎始终可用"

    def load_model(self, path, device=None, progress=None):
        self._loaded_path = path
        self._model_name = path

    def unload(self):
        self._loaded_path = None
        self._model_name = None

    def generate(self, messages, params):
        self._check_cancel()
        if not self.loaded:
            raise ModelNotLoadedError("未加载模型")
        for tok in ["你", "好", "!", "世", "界"]:
            self._check_cancel()
            yield tok


class TestGenParams(unittest.TestCase):

    def test_defaults(self):
        p = GenParams()
        self.assertEqual(p.temperature, 0.7)
        self.assertGreater(p.max_tokens, 0)

    def test_from_config(self):
        cfg = Config({"gen_temperature": 0.3, "gen_max_tokens": 512,
                      "gen_ctx_len": 8192, "system_prompt": "test"})
        p = GenParams.from_config(cfg)
        self.assertEqual(p.temperature, 0.3)
        self.assertEqual(p.max_tokens, 512)
        self.assertEqual(p.ctx_len, 8192)
        self.assertEqual(p.system_prompt, "test")

    def test_from_config_bad_values_falls_back(self):
        cfg = Config({"gen_temperature": "not-a-number"})
        p = GenParams.from_config(cfg)
        self.assertEqual(p.temperature, 0.7)


class TestErrors(unittest.TestCase):

    def test_hierarchy(self):
        self.assertTrue(issubclass(ModelLoadError, EngineError))
        self.assertTrue(issubclass(EngineUnavailableError, EngineError))
        self.assertTrue(issubclass(ModelNotLoadedError, EngineError))
        self.assertTrue(issubclass(GenerationCancelled, Exception))
        self.assertFalse(issubclass(GenerationCancelled, EngineError))


class TestDummyEngine(unittest.TestCase):

    def test_lifecycle(self):
        e = DummyEngine()
        self.assertFalse(e.loaded)
        e.load_model("/tmp/x.gguf")
        self.assertTrue(e.loaded)
        self.assertEqual(e.model_name, "/tmp/x.gguf")
        e.unload()
        self.assertFalse(e.loaded)
        e.unload()  # 幂等

    def test_generate_streams(self):
        e = DummyEngine()
        e.load_model("/tmp/x.gguf")
        out = "".join(e.generate([{"role": "user", "content": "hi"}], GenParams()))
        self.assertEqual(out, "你好!世界")

    def test_generate_without_model_raises(self):
        e = DummyEngine()
        with self.assertRaises(ModelNotLoadedError):
            list(e.generate([], GenParams()))

    def test_cancel_raises(self):
        e = DummyEngine()
        e.load_model("/tmp/x.gguf")
        tokens = []
        try:
            for i, tok in enumerate(e.generate([], GenParams())):
                if i == 1:
                    e.cancel()
                tokens.append(tok)
        except GenerationCancelled:
            pass
        self.assertEqual(tokens, ["你", "好"], "取消后应立即中断生成")

    def test_reset_cancel(self):
        e = DummyEngine()
        e.load_model("/tmp/x.gguf")
        e.cancel()
        e.reset_cancel()
        out = "".join(e.generate([], GenParams()))
        self.assertEqual(out, "你好!世界")

    def test_describe(self):
        e = DummyEngine()
        d = e.describe()
        self.assertEqual(d["engine_id"], "dummy")
        self.assertFalse(d["loaded"])

    def test_capabilities_dataclass(self):
        c = EngineCapabilities(engine_id="x", display_name="X", available=False,
                               detail="缺依赖", supported_devices=["CPU"])
        self.assertFalse(c.available)
        self.assertEqual(c.supported_devices, ["CPU"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
