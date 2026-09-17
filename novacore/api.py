"""NovaCore 本地 API 服务(OpenAI 兼容):
- POST /v1/chat/completions(支持 stream SSE 与一次性返回两种模式)
- GET  /v1/models、GET /health
FastAPI/uvicorn 为可选依赖:未安装时 start() 返回 (False, 提示),绝不影响主程序。
服务运行于守护线程,stop() 优雅关闭。
注意:本模块禁止启用 `from __future__ import annotations`,否则 FastAPI 无法
解析局部定义的 Pydantic 模型注解(会退化成 query 参数导致 422)。
"""

import json
import threading
import time
import uuid
from typing import TYPE_CHECKING, Optional

from . import APP_VERSION
from .config import Config
from .engines import EngineManager
from .engines.base import EngineError, GenParams, GenerationCancelled

if TYPE_CHECKING:  # 仅供类型检查,不产生运行时依赖
    from fastapi import FastAPI


def create_app(mgr: EngineManager, cfg: Config) -> "FastAPI":
    """构建 FastAPI 应用(需要 fastapi 已安装,由调用方保证)。"""
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel, Field

    app = FastAPI(title="NovaCore Local API", version=APP_VERSION)

    class ChatMessage(BaseModel):
        role: str
        content: str

    class ChatRequest(BaseModel):
        model: Optional[str] = None
        messages: list[ChatMessage]
        temperature: Optional[float] = Field(default=None, ge=0, le=2)
        top_p: Optional[float] = Field(default=None, gt=0, le=1)
        max_tokens: Optional[int] = Field(default=None, ge=1, le=32768)
        stream: bool = False

    def _require_loaded() -> None:
        if not mgr.is_loaded():
            raise HTTPException(
                status_code=503,
                detail="尚未加载模型,请先在 NovaCore 界面加载模型后再调用 API")

    def _build_params(req: ChatRequest) -> GenParams:
        params = GenParams.from_config(cfg)
        if req.temperature is not None:
            params.temperature = float(req.temperature)
        if req.top_p is not None:
            params.top_p = float(req.top_p)
        if req.max_tokens is not None:
            params.max_tokens = int(req.max_tokens)
        return params

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "app": "NovaCore",
            "version": APP_VERSION,
            "loaded": mgr.is_loaded(),
            "engine": mgr.current_id,
            "model": mgr.current.model_name if mgr.current else None,
        }

    @app.get("/v1/models")
    def list_models():
        _require_loaded()
        model = mgr.current.model_name or "unknown"
        return {"object": "list", "data": [{"id": model, "object": "model",
                                            "owned_by": "novacore"}]}

    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatRequest):
        _require_loaded()
        params = _build_params(req)
        messages = [{"role": m.role, "content": m.content} for m in req.messages]
        model_name = mgr.current.model_name or "novacore"
        created = int(time.time())

        if not req.stream:
            chunks: list[str] = []
            try:
                for tok in mgr.generate(messages, params):
                    chunks.append(tok)
            except GenerationCancelled:
                pass  # 用户在界面点了停止:返回已生成部分
            except EngineError as e:
                raise HTTPException(status_code=500, detail=str(e)) from e
            return {
                "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                "object": "chat.completion",
                "created": created,
                "model": model_name,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "".join(chunks)},
                    "finish_reason": "stop",
                }],
            }

        req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

        def sse():
            def fmt(delta_content: Optional[str], finish: Optional[str] = None) -> str:
                payload = {
                    "id": req_id, "object": "chat.completion.chunk",
                    "created": created, "model": model_name,
                    "choices": [{"index": 0,
                                 "delta": ({} if delta_content is None
                                           else {"content": delta_content}),
                                 "finish_reason": finish}],
                }
                return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

            try:
                for tok in mgr.generate(messages, params):
                    yield fmt(tok)
            except GenerationCancelled:
                pass
            except EngineError as e:
                yield f"data: {json.dumps({'error': {'message': str(e)}})}\n\n"
            yield fmt(None, finish="stop")
            yield "data: [DONE]\n\n"

        return StreamingResponse(sse(), media_type="text/event-stream")

    return app


class ApiServer:
    """uvicorn 服务线程封装:start/stop 幂等,可重复调用。"""

    def __init__(self, mgr: EngineManager, cfg: Config):
        self._mgr = mgr
        self._cfg = cfg
        self._server = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()  # 可重入:running() 会在持锁时被调用

    @property
    def running(self) -> bool:
        with self._lock:
            return bool(self._thread and self._thread.is_alive()
                        and self._server is not None
                        and not self._server.should_exit)

    def start(self) -> tuple[bool, str]:
        """启动 API 服务;返回 (是否成功, 说明)。"""
        with self._lock:
            if self.running:
                return True, "API 服务已在运行"
            try:
                import uvicorn  # noqa: F401
            except ImportError:
                return False, "未安装 fastapi/uvicorn,可在「依赖管理」页安装后启用 API"
            try:
                app = create_app(self._mgr, self._cfg)
                port = int(self._cfg.get("api_port", 8000))
                server = uvicorn.Config(
                    app, host="127.0.0.1", port=port,
                    log_level="warning", lifespan="off")
                self._server = uvicorn.Server(server)
                self._thread = threading.Thread(
                    target=self._server.run, daemon=True,
                    name=f"novacore-api-{port}")
                self._thread.start()
                deadline = time.time() + 6.0
                while time.time() < deadline:
                    if self._server.started:
                        return True, f"API 服务已启动: http://127.0.0.1:{port}"
                    if not self._thread.is_alive():
                        msg = f"API 服务启动失败(端口 {port} 可能被占用)"
                        self._server = None
                        self._thread = None
                        return False, msg
                    time.sleep(0.05)
                self.stop()
                return False, "API 服务启动超时"
            except Exception as e:
                self._server = None
                self._thread = None
                return False, f"API 服务启动异常: {e}"

    def stop(self) -> None:
        """请求优雅关闭并等待线程退出(幂等)。"""
        with self._lock:
            server, thread = self._server, self._thread
        if server is not None:
            try:
                server.should_exit = True
            except Exception:
                pass
        if thread is not None and thread.is_alive():
            thread.join(timeout=5.0)
        with self._lock:
            self._server = None
            self._thread = None
