"""Ollama 引擎:通过 HTTP 接入本机/远程 Ollama 服务(支持流式与模型拉取)。"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable, Iterator, Optional

import requests

from .base import (
    BaseEngine, GenParams, ModelLoadError, ModelNotLoadedError,
)

DEFAULT_HOST = "http://127.0.0.1:11434"


def _safe_tag(name: str) -> str:
    """把文件名转换为合法的 Ollama 模型标签(仅保留字母数字 . _ -)。"""
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "-", name).strip("-")
    return cleaned or "model"


def _looks_like_local_gguf(name: str) -> bool:
    """判断模型引用是否为本地 GGUF 文件(用于自动导入 Ollama)。"""
    try:
        return name.lower().endswith(".gguf") and Path(name).is_file()
    except Exception:
        return False


def find_ollama_exe() -> Optional[str]:
    """定位 ollama 可执行文件(PATH 优先,再找常见安装路径)。

    覆盖 Windows 常见布局:用户级安装(%LOCALAPPDATA%\\Programs\\Ollama)、
    系统级安装(%PROGRAMFILES%\\Ollama)、home 兜底;Linux/macOS 查 PATH 与
    /usr/local/bin、/usr/bin。找不到返回 None。
    """
    exe = shutil.which("ollama") or shutil.which("ollama.exe")
    if exe:
        return exe
    candidates: list[Path] = []
    local_appdata = os.environ.get("LOCALAPPDATA")
    if local_appdata:
        candidates.append(Path(local_appdata) / "Programs/Ollama/ollama.exe")
    program_files = os.environ.get("PROGRAMFILES")
    if program_files:
        candidates.append(Path(program_files) / "Ollama/ollama.exe")
    program_files_x86 = os.environ.get("PROGRAMFILES(X86)")
    if program_files_x86:
        candidates.append(Path(program_files_x86) / "Ollama/ollama.exe")
    candidates += [
        Path.home() / "AppData/Local/Programs/Ollama/ollama.exe",
        Path.home() / ".ollama/ollama.exe",
        Path("/usr/local/bin/ollama"),
        Path("/usr/bin/ollama"),
    ]
    for p in candidates:
        try:
            if p.is_file():
                return str(p)
        except OSError:
            continue
    return None


class OllamaEngine(BaseEngine):
    engine_id = "ollama"
    display_name = "Ollama 本地服务"

    def __init__(self, host: str = DEFAULT_HOST):
        super().__init__()
        self.host = (host or DEFAULT_HOST).rstrip("/")
        self._active_response = None  # 当前流式响应用于主动中断

    # ---------- 可用性 ----------
    def check_available(self) -> tuple[bool, str]:
        try:
            r = requests.get(f"{self.host}/api/tags", timeout=5)
            if r.status_code == 200:
                return True, f"Ollama 服务在线: {self.host}"
            return False, f"Ollama 返回异常状态码 {r.status_code}"
        except requests.ConnectionError:
            return False, (f"无法连接 Ollama 服务({self.host})。"
                           "请确认已安装并启动 Ollama,或到「系统设置」勾选自动启动。")
        except Exception as e:
            return False, f"Ollama 服务不可用({self.host}): {e}"

    def list_models(self) -> list[str]:
        try:
            r = requests.get(f"{self.host}/api/tags", timeout=5)
            r.raise_for_status()
            return [m.get("name", "") for m in r.json().get("models", [])]
        except Exception:
            return []

    def pull_model(self, name: str,
                   progress: Optional[Callable[[str, str], None]] = None) -> None:
        """拉取模型到 Ollama。progress(status, detail)。"""
        try:
            with requests.post(
                f"{self.host}/api/pull", json={"name": name},
                stream=True, timeout=(5, 3600),
            ) as r:
                r.raise_for_status()
                for line in r.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if progress:
                        status = data.get("status", "")
                        detail = ""
                        if "completed" in data and "total" in data:
                            total = data["total"] or 0
                            done = data["completed"] or 0
                            if total:
                                detail = f"{done / total * 100:.1f}%"
                        progress(status, detail)
                    if data.get("error"):
                        raise ModelLoadError(f"Ollama 拉取失败: {data['error']}")
        except ModelLoadError:
            raise
        except Exception as e:
            raise ModelLoadError(f"Ollama 拉取失败: {e}") from e

    # ---------- 加载 ----------
    def load_model(self, name: str, device: Optional[str] = None,
                   progress=None) -> None:
        """加载 Ollama 模型。name 可为模型 tag(如 qwen2.5:0.5b)或本地 GGUF 路径。

        - tag:不存在则自动拉取;
        - 本地 .gguf 文件:先用 Ollama 原生方式注册(create)再加载,
          使 NovaCore 下载的 GGUF 无需 llama.cpp 也能经 Ollama 运行。
        """
        self.unload()
        if not name:
            raise ModelLoadError("请指定 Ollama 模型名(如 qwen2.5:0.5b)")
        if _looks_like_local_gguf(name):
            # create 导入成功后模型即已注册,无需再走 pull;
            # 否则会把 novacore/xxx 当成远程仓库误拉,报 "pull manifest" 错误。
            name = self.create_from_gguf(name)
        else:
            available = self.list_models()
            if name not in available:
                # 自动拉取(阻塞;UI 层可先调用 pull_model 显示进度)
                self.pull_model(name)
                if name not in self.list_models():
                    raise ModelLoadError(f"模型 {name} 拉取后仍不可用")
        self._loaded_path = name
        self._model_name = name

    def create_from_gguf(self, gguf_path: str) -> str:
        """把本地 GGUF 文件注册为 Ollama 模型,返回其 tag(如 novacore/qwen2.5:latest)。

        新版 Ollama(0.5.5+)的 /api/create 不再解析 modelfile 里的 FROM 指令,
        故改用官方 CLI `ollama create -f` 导入,它对本地 GGUF 支持最完整。
        """
        p = Path(gguf_path)
        if not p.is_file():
            raise ModelLoadError(f"GGUF 文件不存在: {gguf_path}")
        # Ollama 对无显式 tag 的模型会补 :latest;/api/tags 也返回 :latest 后缀,
        # 故这里用带 :latest 的规范名,保证与 list_models() 结果一致。
        tag = f"novacore/{_safe_tag(p.stem)}:latest"
        if tag in self.list_models():
            return tag

        exe = find_ollama_exe()
        if not exe:
            raise ModelLoadError(
                "未找到 ollama 可执行文件,无法导入 GGUF")
        # 写 Modelfile:Windows 下用正斜杠绝对路径并加引号,兼容空格/中文
        abs_path = str(p.resolve()).replace("\\", "/")
        fd, mf_path = tempfile.mkstemp(suffix=".Modelfile")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(f'FROM "{abs_path}"\n')
            try:
                proc = subprocess.run(
                    [exe, "create", tag, "-f", mf_path],
                    capture_output=True, text=True,
                    encoding="utf-8", errors="replace",
                    timeout=1800,
                    creationflags=(subprocess.CREATE_NO_WINDOW
                                   if sys.platform == "win32" else 0),
                )
            finally:
                try:
                    os.remove(mf_path)
                except OSError:
                    pass
        except subprocess.TimeoutExpired:
            raise ModelLoadError("Ollama 导入 GGUF 超时(>30 分钟)")
        except ModelLoadError:
            raise
        except Exception as e:
            raise ModelLoadError(f"调用 ollama create 失败: {e}") from e

        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "").strip()
            raise ModelLoadError(
                f"Ollama 导入 GGUF 失败: "
                f"{err[:500] or ('返回码 ' + str(proc.returncode))}")
        return tag

    def unload(self) -> None:
        name = self._model_name
        self._loaded_path = None
        self._model_name = None
        if name:
            # 让 Ollama 立即卸载模型释放内存(keep_alive=0),非阻塞尽力而为
            try:
                requests.post(
                    f"{self.host}/api/chat",
                    json={"model": name, "messages": [],
                          "keep_alive": 0},
                    timeout=10,
                )
            except Exception:
                pass

    def cancel(self) -> None:
        """覆写:设置取消事件并主动关闭流式连接,以立即中断阻塞中的生成。"""
        super().cancel()
        resp = self._active_response
        if resp is not None:
            try:
                resp.close()
            except Exception:
                pass

    # ---------- 生成 ----------
    def generate(self, messages: list[dict], params: GenParams) -> Iterator[str]:
        if not self.loaded:
            raise ModelNotLoadedError("尚未加载 Ollama 模型")
        self._check_cancel()
        payload = {
            "model": self._model_name,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": params.temperature,
                "top_p": params.top_p,
                "top_k": params.top_k,
                "num_predict": params.max_tokens,
                "num_ctx": params.ctx_len,
            },
        }
        self._active_response = None
        try:
            with requests.post(
                f"{self.host}/api/chat", json=payload,
                stream=True, timeout=(5, 1800),
            ) as r:
                self._active_response = r
                try:
                    r.raise_for_status()
                    for line in r.iter_lines(decode_unicode=True):
                        self._check_cancel()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if data.get("error"):
                            from .base import EngineError
                            raise EngineError(f"Ollama 推理错误: {data['error']}")
                        content = (data.get("message") or {}).get("content") or ""
                        if content:
                            yield content
                        if data.get("done"):
                            break
                finally:
                    self._active_response = None
        except GeneratorExit:
            raise
        except Exception as e:
            from .base import EngineError, GenerationCancelled
            if isinstance(e, GenerationCancelled):
                raise
            if self._cancel_evt.is_set():
                # 主动关闭连接打断了阻塞读,按用户取消处理并清理事件
                self._cancel_evt.clear()
                raise GenerationCancelled()
            if isinstance(e, EngineError):
                raise
            raise EngineError(f"Ollama 请求失败: {e}") from e
