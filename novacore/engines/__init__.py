"""引擎注册表与自动探测:
- 集中管理三种引擎,统一加载/生成/取消入口
- 启动时自动探测可用性并缓存(避免 Ollama 等每次做网络探测)
- 依据用户配置(default_engine / prefer_device)与硬件能力自动选择最优引擎
"""
from __future__ import annotations

import threading
import time
from typing import Iterator, Optional

from .base import (
    BaseEngine, GenParams, ModelNotLoadedError,
)
from .. import hardware
from .llamacpp import LlamaCppEngine
from .ollama import DEFAULT_HOST, OllamaEngine
from .openvino import OpenVinoEngine

# 引擎探测结果的缓存时长(秒)
CAPABILITY_TTL = 60.0


class EngineManager:
    def __init__(self, ollama_host: Optional[str] = None):
        self._engines: dict[str, BaseEngine] = {
            "llamacpp": LlamaCppEngine(),
            "ollama": OllamaEngine(ollama_host or DEFAULT_HOST),
            "openvino": OpenVinoEngine(),
        }
        self._order = ["llamacpp", "ollama", "openvino"]
        self._current_id: Optional[str] = None
        self._cap_cache: dict[str, tuple[float, tuple[bool, str]]] = {}
        self._lock = threading.RLock()

    # ---------- 探测 ----------
    def probe(self, engine_id: str, refresh: bool = False) -> tuple[bool, str]:
        """返回 (是否可用, 说明);结果缓存 60 秒。"""
        with self._lock:
            cached = self._cap_cache.get(engine_id)
            if not refresh and cached and time.time() - cached[0] < CAPABILITY_TTL:
                return cached[1]
        engine = self._engines[engine_id]
        try:
            result = engine.check_available()
        except Exception as e:
            result = (False, f"探测异常: {e}")
        with self._lock:
            self._cap_cache[engine_id] = (time.time(), result)
        return result

    def probe_all(self, refresh: bool = False) -> list[dict]:
        out = []
        for eid in self._order:
            ok, detail = self.probe(eid, refresh=refresh)
            engine = self._engines[eid]
            out.append({
                "id": eid,
                "display_name": engine.display_name,
                "available": ok,
                "detail": detail,
            })
        return out

    # ---------- 选择 ----------
    def pick_engine(self, cfg, refresh: bool = False) -> str:
        """按配置与硬件能力选择引擎;永远返回一个引擎 id(即使不可用,便于给出安装引导)。"""
        default_engine = cfg.get("default_engine", "auto") if cfg else "auto"
        prefer_device = str(cfg.get("prefer_device", "CPU")) if cfg else "CPU"
        results = {eid: self.probe(eid, refresh=refresh) for eid in self._order}
        ok_of = lambda eid: results.get(eid, (False, ""))[0]  # noqa: E731

        # 用户显式指定且可用 → 直接采用
        if default_engine in self._engines and default_engine != "auto":
            if ok_of(default_engine):
                return default_engine

        # 按偏好设备选:NPU → OpenVINO;GPU → llama.cpp
        if "NPU" in prefer_device and ok_of("openvino") and hardware.has_npu():
            return "openvino"
        if prefer_device == "GPU" and ok_of("llamacpp") and hardware.has_cuda_gpu():
            return "llamacpp"

        # 按可用性兜底:llama.cpp 优先(自包含),其次 Ollama,再 OpenVINO
        for eid in self._order:
            if ok_of(eid):
                return eid
        # 全部不可用:返回注册表第一个以便界面提示安装依赖
        return self._order[0] if self._order else "llamacpp"

    # ---------- 访问 ----------
    def get(self, engine_id: str) -> BaseEngine:
        if engine_id not in self._engines:
            raise KeyError(f"未知引擎: {engine_id}")
        return self._engines[engine_id]

    @property
    def current_id(self) -> Optional[str]:
        return self._current_id

    @property
    def current(self) -> Optional[BaseEngine]:
        if self._current_id is None:
            return None
        return self._engines[self._current_id]

    def is_loaded(self) -> bool:
        cur = self.current
        return bool(cur and cur.loaded)

    # ---------- 统一操作 ----------
    def load(self, model_ref: str, engine_id: Optional[str] = None,
             device: Optional[str] = None, progress=None) -> BaseEngine:
        """加载模型到指定引擎;成功后将引擎设为当前引擎。"""
        engine = self.get(engine_id or self._current_id or "llamacpp")
        engine.load_model(model_ref, device=device, progress=progress)
        self._current_id = engine.engine_id
        return engine

    def unload_all(self) -> None:
        for engine in self._engines.values():
            try:
                engine.unload()
            except Exception:
                pass

    def generate(self, messages: list[dict], params: GenParams) -> Iterator[str]:
        engine = self.current
        if engine is None:
            raise ModelNotLoadedError("尚未选择引擎")
        yield from engine.generate(messages, params)

    def cancel(self) -> None:
        engine = self.current
        if engine is not None:
            engine.cancel()

    def count_tokens(self, text: str) -> int:
        engine = self.current
        if engine is None:
            return max(1, len(text) // 2)
        return engine.count_tokens(text)


# 模块级单例(整个应用共用一个管理器)
_manager: Optional[EngineManager] = None
_manager_lock = threading.Lock()


def get_manager(ollama_host: Optional[str] = None) -> EngineManager:
    global _manager
    with _manager_lock:
        if _manager is None:
            _manager = EngineManager(ollama_host=ollama_host)
        return _manager
