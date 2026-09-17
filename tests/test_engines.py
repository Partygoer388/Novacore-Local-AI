# -*- coding: utf-8 -*-
"""三个推理引擎的契约自测(全部使用 mock,不依赖真实模型/服务):
直接运行 `python tests/test_engines.py`。"""
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from novacore.engines.base import (
    EngineError, GenerationCancelled, GenParams, ModelLoadError,
    ModelNotLoadedError,
)
from novacore.engines.llamacpp import LlamaCppEngine
from novacore.engines.ollama import OllamaEngine
from novacore.engines.openvino import OpenVinoEngine, build_prompt


# ==================== llama.cpp ====================

class FakeLlama:
    def __init__(self, **kw):
        self.kw = kw

    def create_chat_completion(self, messages=None, **kw):
        def stream():
            for tok in ["你", "好", "!"]:
                yield {"choices": [{"delta": {"content": tok}}]}
        return stream()

    def tokenize(self, data):
        return list(range(len(data) // 2))


def make_fake_llama_cpp_module():
    m = types.ModuleType("llama_cpp")
    m.Llama = FakeLlama
    m.__version__ = "0.2.99"
    return m


class TestLlamaCppEngine(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.model = self.dir / "model.gguf"
        self.model.write_bytes(b"GGUF" + b"\x00" * (2 * 1024 * 1024))

    def test_check_available_when_missing(self):
        e = LlamaCppEngine()
        with mock.patch.dict(sys.modules, {"llama_cpp": None}):
            ok, msg = e.check_available()
        self.assertFalse(ok)
        self.assertIn("llama-cpp-python", msg)

    def test_check_available_when_present(self):
        e = LlamaCppEngine()
        with mock.patch.dict(sys.modules, {"llama_cpp": make_fake_llama_cpp_module()}):
            ok, msg = e.check_available()
        self.assertTrue(ok)
        self.assertIn("0.2.99", msg)

    def test_load_success_and_generate(self):
        e = LlamaCppEngine()
        with mock.patch.dict(sys.modules, {"llama_cpp": make_fake_llama_cpp_module()}):
            e.load_model(str(self.model), device="CPU")
            self.assertTrue(e.loaded)
            out = "".join(e.generate([{"role": "user", "content": "hi"}], GenParams()))
            self.assertEqual(out, "你好!")
            self.assertEqual(e.count_tokens("abcdefgh"), 4)  # FakeLlama.tokenize
            e.unload()
            self.assertFalse(e.loaded)

    def test_load_missing_file_raises(self):
        e = LlamaCppEngine()
        with mock.patch.dict(sys.modules, {"llama_cpp": make_fake_llama_cpp_module()}):
            with self.assertRaises(ModelLoadError):
                e.load_model(str(self.dir / "nope.gguf"))

    def test_load_tiny_file_raises(self):
        e = LlamaCppEngine()
        tiny = self.dir / "tiny.gguf"
        tiny.write_bytes(b"GGUF" + b"\x00" * 100)
        with mock.patch.dict(sys.modules, {"llama_cpp": make_fake_llama_cpp_module()}):
            with self.assertRaises(ModelLoadError):
                e.load_model(str(tiny))

    def test_generate_without_model_raises(self):
        e = LlamaCppEngine()
        with mock.patch.dict(sys.modules, {"llama_cpp": make_fake_llama_cpp_module()}):
            with self.assertRaises(ModelNotLoadedError):
                list(e.generate([], GenParams()))

    def test_cancel_during_generate(self):
        e = LlamaCppEngine()
        with mock.patch.dict(sys.modules, {"llama_cpp": make_fake_llama_cpp_module()}):
            e.load_model(str(self.model))

            class CancelAfterOne:
                def __init__(self, eng):
                    self.eng = eng
                    self.n = 0

                def create_chat_completion(self, **kw):
                    def stream():
                        for tok in ["a", "b", "c"]:
                            self.n += 1
                            if self.n > 1:
                                self.eng.cancel()
                            yield {"choices": [{"delta": {"content": tok}}]}
                    return stream()

            e._llm = CancelAfterOne(e)
            tokens = []
            with self.assertRaises(GenerationCancelled):
                for tok in e.generate([], GenParams()):
                    tokens.append(tok)
            self.assertEqual(tokens, ["a"])


# ==================== Ollama ====================

class FakeResp:
    def __init__(self, status=200, data=None, lines=None):
        self.status_code = status
        self._data = data if data is not None else {}
        self._lines = lines or []

    def json(self):
        return self._data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def iter_lines(self, decode_unicode=True):
        return iter(self._lines)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class TestOllamaEngine(unittest.TestCase):

    def setUp(self):
        self.e = OllamaEngine("http://fake:11434")

    def test_check_available_ok(self):
        with mock.patch("novacore.engines.ollama.requests.get",
                        return_value=FakeResp(200, {"models": []})):
            ok, msg = self.e.check_available()
        self.assertTrue(ok)
        self.assertIn("在线", msg)

    def test_check_available_down(self):
        def boom(*a, **k):
            raise ConnectionError("refused")
        with mock.patch("novacore.engines.ollama.requests.get", side_effect=boom):
            ok, msg = self.e.check_available()
        self.assertFalse(ok)
        self.assertIn("不可用", msg)

    def test_load_existing_model(self):
        with mock.patch.object(self.e, "list_models", return_value=["qwen2.5:0.5b"]):
            self.e.load_model("qwen2.5:0.5b")
        self.assertTrue(self.e.loaded)
        self.assertEqual(self.e.model_name, "qwen2.5:0.5b")

    def test_load_missing_model_auto_pulls(self):
        pulled = {}
        with mock.patch.object(self.e, "list_models", side_effect=[[], ["qwen2.5:0.5b"]]):
            with mock.patch.object(self.e, "pull_model") as pm:
                pm.side_effect = lambda name, progress=None: pulled.update(name=name)
                self.e.load_model("qwen2.5:0.5b")
        self.assertEqual(pulled.get("name"), "qwen2.5:0.5b")
        self.assertTrue(self.e.loaded)

    def test_generate_streams_and_done(self):
        self.e._loaded_path = "qwen2.5:0.5b"
        self.e._model_name = "qwen2.5:0.5b"
        lines = [
            json.dumps({"message": {"content": "你"}, "done": False}),
            json.dumps({"message": {"content": "好"}, "done": False}),
            json.dumps({"message": {"content": ""}, "done": True}),
        ]
        with mock.patch("novacore.engines.ollama.requests.post",
                        return_value=FakeResp(200, lines=lines)):
            out = "".join(self.e.generate([], GenParams()))
        self.assertEqual(out, "你好")

    def test_generate_cancel(self):
        self.e._loaded_path = "m"
        self.e._model_name = "m"
        lines = [json.dumps({"message": {"content": "x"}, "done": False})] * 100

        def slow_iter():
            for line in lines:
                yield line
        resp = FakeResp(200, lines=[])
        resp.iter_lines = lambda decode_unicode=True: slow_iter()
        with mock.patch("novacore.engines.ollama.requests.post", return_value=resp):
            with self.assertRaises(GenerationCancelled):
                for _tok in self.e.generate([], GenParams()):
                    self.e.cancel()

    def test_generate_error_from_server(self):
        self.e._loaded_path = "m"
        self.e._model_name = "m"
        lines = [json.dumps({"error": "model not found", "done": True})]
        with mock.patch("novacore.engines.ollama.requests.post",
                        return_value=FakeResp(200, lines=lines)):
            with self.assertRaises(EngineError):
                list(self.e.generate([], GenParams()))


# ==================== OpenVINO ====================

class FakeGenConfig:
    max_new_tokens = 100
    temperature = 0.7
    top_p = 0.9
    top_k = 40


class FakeStreamer:
    def __init__(self, tok_or_cb, cb=None):
        self.cb = cb if cb is not None else tok_or_cb


class FakeStatus:
    """genai StreamingStatus 的替身:RUNNING=继续, STOP=停止。"""
    RUNNING = 0
    STOP = 1
    CANCEL = 2


class FakeTokenizer:
    def apply_chat_template(self, messages, add_generation_prompt=True):
        return "tmpl:" + "|".join(str(m.get("content", "")) for m in messages)

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


class FakePipeline:
    def __init__(self, path, device, **kwargs):
        self.path = path
        self.device = device
        self.kwargs = kwargs

    def get_tokenizer(self):
        return FakeTokenizer()

    def start_chat(self):
        pass

    def finish_chat(self):
        pass

    def generate(self, prompt, cfg, streamer):
        for tok in ["世", "界"]:
            streamer.cb(tok)


class FakeCore:
    def __init__(self, devices=None):
        self._devices = devices or ["CPU", "GPU"]

    @property
    def available_devices(self):
        return list(self._devices)


def make_fake_ovg_module():
    m = types.ModuleType("openvino_genai")
    m.LLMPipeline = FakePipeline
    m.GenerationConfig = FakeGenConfig
    m.Streamer = FakeStreamer
    m.TextStreamer = FakeStreamer
    m.StreamingStatus = FakeStatus
    return m


def make_fake_ov_module(with_npu=False):
    m = types.ModuleType("openvino")
    devices = ["CPU", "GPU", "NPU"] if with_npu else ["CPU", "GPU"]
    m.Core = lambda: FakeCore(devices)  # noqa: E731
    m.__version__ = "2026.2"
    return m


class TestOpenVinoEngine(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        (self.dir / "openvino_model.xml").write_text("<xml/>")

    def test_check_available_with_modules(self):
        e = OpenVinoEngine()
        with mock.patch.dict(sys.modules, {"openvino": make_fake_ov_module(),
                                           "openvino_genai": make_fake_ovg_module()}):
            ok, msg = e.check_available()
        self.assertTrue(ok)
        self.assertIn("CPU", msg)

    def test_check_available_missing_genai(self):
        e = OpenVinoEngine()
        with mock.patch.dict(sys.modules, {"openvino": make_fake_ov_module(),
                                           "openvino_genai": None}):
            ok, msg = e.check_available()
        self.assertFalse(ok)
        self.assertIn("openvino-genai", msg)

    def test_load_requires_directory(self):
        e = OpenVinoEngine()
        with mock.patch.dict(sys.modules, {"openvino": make_fake_ov_module(),
                                           "openvino_genai": make_fake_ovg_module()}):
            with self.assertRaises(ModelLoadError):
                e.load_model(str(self.dir / "openvino_model.xml"))  # 文件而非目录

    def test_load_npu_success_with_blob(self):
        """NPU 可用且编译成功:引擎应真正跑在 NPU 上。"""
        e = OpenVinoEngine()
        ovg = make_fake_ovg_module()
        with mock.patch.dict(sys.modules, {"openvino": make_fake_ov_module(with_npu=True),
                                           "openvino_genai": ovg}):
            with mock.patch.object(e, "_npu_compile_blob", return_value=None):
                e.load_model(str(self.dir), device="NPU")
        self.assertTrue(e.loaded)
        self.assertEqual(e._device, "NPU")
        self.assertEqual(e._load_detail, "NPU")

    def test_load_gguf_npu_not_supported_falls_back_cpu(self):
        """GGUF 无法保证 NPU 兼容(编译可能杀死进程),必须回退 CPU 并说明。"""
        gguf = self.dir / "model.gguf"
        gguf.write_bytes(b"fake gguf")
        e = OpenVinoEngine()
        with mock.patch.dict(sys.modules, {"openvino": make_fake_ov_module(with_npu=True),
                                           "openvino_genai": make_fake_ovg_module()}):
            e.load_model(str(gguf), device="NPU")
        self.assertTrue(e.loaded)
        self.assertEqual(e._device, "CPU", "GGUF 不应尝试 NPU,应回退 CPU")
        self.assertIn("回退", e._load_detail)
        self.assertEqual(e._model_name, "model.gguf")

    def test_load_npu_fails_falls_back_to_cpu(self):
        """NPU 加载失败时必须自动回退 CPU,并把回退原因写进 _load_detail。"""
        ovg = make_fake_ovg_module()

        class NpuOnlyFailPipeline:
            def __init__(self, path, device, **kw):
                if device == "NPU":
                    raise RuntimeError("NPU not available")

        ovg.LLMPipeline = NpuOnlyFailPipeline
        e = OpenVinoEngine()
        with mock.patch.dict(sys.modules, {"openvino": make_fake_ov_module(with_npu=True),
                                           "openvino_genai": ovg}):
            with mock.patch.object(e, "_npu_compile_blob", return_value=None):
                e.load_model(str(self.dir), device="NPU")
        self.assertTrue(e.loaded)
        self.assertEqual(e._device, "CPU", "应回退到 CPU")
        self.assertIn("回退", e._load_detail)

    def test_load_no_npu_device_falls_back_cpu(self):
        """机器无 NPU:预检直接跳过 NPU,回退 CPU。"""
        e = OpenVinoEngine()
        with mock.patch.dict(sys.modules, {"openvino": make_fake_ov_module(),
                                           "openvino_genai": make_fake_ovg_module()}):
            e.load_model(str(self.dir), device="NPU")
        self.assertTrue(e.loaded)
        self.assertEqual(e._device, "CPU")

    def test_load_all_devices_fail_raises(self):
        ovg = make_fake_ovg_module()

        class AlwaysFailPipeline:
            def __init__(self, path, device):
                raise RuntimeError("boom")

        ovg.LLMPipeline = AlwaysFailPipeline
        e = OpenVinoEngine()
        with mock.patch.dict(sys.modules, {"openvino": make_fake_ov_module(),
                                           "openvino_genai": ovg}):
            with self.assertRaises(ModelLoadError):
                e.load_model(str(self.dir), device="NPU")
        self.assertFalse(e.loaded)

    def test_generate_streams(self):
        e = OpenVinoEngine()
        with mock.patch.dict(sys.modules, {"openvino": make_fake_ov_module(),
                                           "openvino_genai": make_fake_ovg_module()}):
            e.load_model(str(self.dir), device="CPU")
            out = "".join(e.generate([{"role": "user", "content": "hi"}], GenParams()))
        self.assertEqual(out, "世界")

    def test_streamer_callback_continues_generation(self):
        """回归:回调返回 False/RUNNING 才继续生成(旧代码返回 True 导致
        1-2 个 token 后生成中断,表现为 AI 只吐零零散散的字)。"""
        e = OpenVinoEngine()
        with mock.patch.dict(sys.modules, {"openvino": make_fake_ov_module(),
                                           "openvino_genai": make_fake_ovg_module()}):
            e.load_model(str(self.dir), device="CPU")
            captured = {}

            class CapPipe(FakePipeline):
                def generate(self, prompt, cfg, streamer):
                    captured["ret"] = streamer.cb("你")

            e._pipe = CapPipe(str(self.dir), "CPU")
            out = "".join(e.generate([{"role": "user", "content": "hi"}], GenParams()))
        self.assertEqual(out, "你")
        self.assertIn(captured.get("ret"), (False, 0), "回调必须返回 RUNNING/False 表示继续")

    def test_generate_cancel(self):
        e = OpenVinoEngine()
        with mock.patch.dict(sys.modules, {"openvino": make_fake_ov_module(),
                                           "openvino_genai": make_fake_ovg_module()}):
            e.load_model(str(self.dir), device="CPU")

            def slow_cb_pipe(path, device):
                class SlowPipe(FakePipeline):
                    def generate(self, prompt, cfg, streamer):
                        import time
                        for i in range(50):
                            streamer.cb(str(i))
                            time.sleep(0.02)
                return SlowPipe(path, device)

            e._pipe = slow_cb_pipe(str(self.dir), "CPU")
            tokens = []
            with self.assertRaises(GenerationCancelled):
                for tok in e.generate([], GenParams()):
                    tokens.append(tok)
                    e.cancel()
            self.assertLess(len(tokens), 50)

    def test_build_prompt(self):
        prompt = build_prompt(
            [{"role": "user", "content": "hi"},
             {"role": "assistant", "content": "hello"}],
            system_prompt="sys")
        self.assertIn("<|system|>\nsys", prompt)
        self.assertIn("<|user|>\nhi", prompt)
        self.assertIn("<|assistant|>\nhello", prompt)


if __name__ == "__main__":
    unittest.main(verbosity=2)
