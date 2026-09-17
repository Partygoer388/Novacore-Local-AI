"""统一推理引擎接口:所有引擎(llamacpp/ollama/openvino)实现同一套契约,
上层(对话、API 服务)只依赖本接口,可任意切换。"""
from __future__ import annotations

import abc
import threading
from dataclasses import dataclass, field
from typing import Callable, Iterator, Optional


class EngineError(RuntimeError):
    """引擎运行时错误基类。"""


class EngineUnavailableError(EngineError):
    """引擎依赖缺失或后端服务不可用。"""


class ModelLoadError(EngineError):
    """模型加载失败。"""


class ModelNotLoadedError(EngineError):
    """未加载模型时尝试推理。"""


class GenerationCancelled(Exception):
    """生成被用户取消(非错误)。"""


@dataclass
class GenParams:
    """生成参数;from_config() 从应用配置读取。"""
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    max_tokens: int = 2048
    ctx_len: int = 4096
    system_prompt: str = "你是一个乐于助人的中文AI助手,回答简洁、准确。"

    @classmethod
    def from_config(cls, cfg) -> "GenParams":
        try:
            return cls(
                temperature=float(cfg.get("gen_temperature", 0.7)),
                top_p=float(cfg.get("gen_top_p", 0.9)),
                top_k=int(cfg.get("gen_top_k", 40)),
                max_tokens=int(cfg.get("gen_max_tokens", 2048)),
                ctx_len=int(cfg.get("gen_ctx_len", 4096)),
                system_prompt=str(cfg.get("system_prompt", cls.system_prompt)),
            )
        except (TypeError, ValueError):
            return cls()


@dataclass
class EngineCapabilities:
    """引擎能力与可用性描述(供自动探测与设置页展示)。"""
    engine_id: str
    display_name: str
    available: bool
    detail: str = ""
    supported_devices: list[str] = field(default_factory=list)
    streaming: bool = True
    gguf_support: bool = False
    remote_only: bool = False  # True 表示依赖外部服务(如 Ollama)


class BaseEngine(abc.ABC):
    """推理引擎抽象基类。"""

    engine_id: str = "base"
    display_name: str = "Base"

    def __init__(self):
        self._lock = threading.RLock()
        self._cancel_evt = threading.Event()
        self._loaded_path: Optional[str] = None
        self._model_name: Optional[str] = None

    # ---------- 生命周期 ----------
    @abc.abstractmethod
    def check_available(self) -> tuple[bool, str]:
        """返回 (是否可用, 说明)。不抛异常。"""

    @abc.abstractmethod
    def load_model(self, path: str, device: Optional[str] = None,
                   progress: Optional[Callable[[str], None]] = None) -> None:
        """加载模型;失败抛 ModelLoadError。加载后 loaded=True。
        progress(text) 可选:加载耗时较长时回报进度文案。"""

    @abc.abstractmethod
    def unload(self) -> None:
        """卸载模型并释放资源;幂等。"""

    @abc.abstractmethod
    def generate(self, messages: list[dict], params: GenParams) -> Iterator[str]:
        """流式生成。messages 为 [{"role","content"}, ...]。
        用户取消时抛 GenerationCancelled。未加载模型抛 ModelNotLoadedError。"""

    # ---------- 取消 ----------
    def cancel(self) -> None:
        self._cancel_evt.set()

    def reset_cancel(self) -> None:
        self._cancel_evt.clear()

    def _check_cancel(self) -> None:
        if self._cancel_evt.is_set():
            self._cancel_evt.clear()
            raise GenerationCancelled()

    # ---------- 状态 ----------
    @property
    def loaded(self) -> bool:
        with self._lock:
            return self._loaded_path is not None

    @property
    def model_path(self) -> Optional[str]:
        with self._lock:
            return self._loaded_path

    @property
    def model_name(self) -> Optional[str]:
        with self._lock:
            return self._model_name

    # ---------- 工具 ----------
    def count_tokens(self, text: str) -> int:
        """粗略 token 估算(默认按字符数/2);各引擎可覆写为精确实现。"""
        return max(1, len(text) // 2)

    def describe(self) -> dict:
        return {
            "engine_id": self.engine_id,
            "display_name": self.display_name,
            "loaded": self.loaded,
            "model_path": self.model_path,
        }
