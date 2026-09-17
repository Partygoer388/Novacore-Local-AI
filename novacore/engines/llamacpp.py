"""llama.cpp 引擎:llama-cpp-python 本地 GGUF 推理,CPU/GPU(CUDA)自适应。"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

from .base import (
    BaseEngine, GenerationCancelled, GenParams, ModelLoadError,
    ModelNotLoadedError,
)
from .. import hardware


def _import_llama_cpp():
    """惰性导入;未安装时给出可操作的提示。"""
    try:
        import llama_cpp  # noqa: F401
        return llama_cpp
    except ImportError as e:
        raise ModelLoadError(
            "未安装 llama-cpp-python。请在「依赖管理」页安装 llama-cpp-python,"
            "或使用 CPU 版:pip install llama-cpp-python") from e


class LlamaCppEngine(BaseEngine):
    engine_id = "llamacpp"
    display_name = "llama.cpp (GGUF 本地推理)"

    def __init__(self):
        super().__init__()
        self._llm = None
        self._device = "CPU"
        self._load_detail = ""

    # ---------- 可用性 ----------
    def check_available(self) -> tuple[bool, str]:
        try:
            import llama_cpp
            ver = getattr(llama_cpp, "__version__", "?")
            gpu = "CUDA GPU 可用" if hardware.has_cuda_gpu() else "仅 CPU(未检测到 NVIDIA GPU)"
            return True, f"llama-cpp-python {ver} | {gpu}"
        except ImportError:
            return False, "未安装 llama-cpp-python(可在依赖管理页安装)"
        except Exception as e:
            return False, f"llama.cpp 初始化异常: {e}"

    # ---------- 加载 ----------
    def load_model(self, path: str, device: Optional[str] = None,
                   progress=None) -> None:
        llama_cpp = _import_llama_cpp()
        p = Path(path)
        if not p.exists():
            raise ModelLoadError(f"模型文件不存在: {path}")
        if p.stat().st_size < 1_000_000:
            raise ModelLoadError("文件过小,不是有效的 GGUF 模型")
        self.unload()
        self._device = device or "CPU"
        n_ctx = int(getattr(self, "_pending_ctx", 4096))
        try:
            if self._device == "GPU" and hardware.has_cuda_gpu():
                self._llm = llama_cpp.Llama(
                    model_path=str(p), n_ctx=n_ctx, n_gpu_layers=-1,
                    verbose=False)
                self._load_detail = f"GPU(CUDA) 加速,上下文 {n_ctx}"
            else:
                self._llm = llama_cpp.Llama(
                    model_path=str(p), n_ctx=n_ctx, n_gpu_layers=0,
                    verbose=False)
                self._load_detail = f"CPU 推理,上下文 {n_ctx}"
        except Exception as e:
            self._llm = None
            self._loaded_path = None
            raise ModelLoadError(f"模型加载失败({self._device}): {e}") from e
        self._loaded_path = str(p)
        self._model_name = p.name

    def set_context_size(self, n_ctx: int) -> None:
        self._pending_ctx = int(n_ctx)

    def unload(self) -> None:
        self._llm = None
        self._loaded_path = None
        self._model_name = None
        self._load_detail = ""

    # ---------- 生成 ----------
    def generate(self, messages: list[dict], params: GenParams) -> Iterator[str]:
        if not self.loaded or self._llm is None:
            raise ModelNotLoadedError("尚未加载模型,请先在「本地模型」页加载")
        self._check_cancel()
        try:
            stream = self._llm.create_chat_completion(
                messages=messages,
                max_tokens=params.max_tokens,
                temperature=params.temperature,
                top_p=params.top_p,
                top_k=params.top_k,
                stream=True,
            )
            for chunk in stream:
                self._check_cancel()
                delta = (chunk.get("choices") or [{}])[0].get("delta") or {}
                content = delta.get("content") or ""
                if content:
                    yield content
        except GenerationCancelled:
            raise
        except Exception as e:
            from .base import EngineError
            raise EngineError(f"推理失败: {e}") from e

    def count_tokens(self, text: str) -> int:
        if self._llm is not None:
            try:
                return len(self._llm.tokenize(text.encode("utf-8")))
            except Exception:
                pass
        return super().count_tokens(text)
