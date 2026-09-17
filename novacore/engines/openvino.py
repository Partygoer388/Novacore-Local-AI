"""OpenVINO 引擎:Intel CPU/GPU/NPU 推理(依赖 openvino + openvino-genai)。
生成在守护线程中执行,主线程通过队列流式取 token 并可即时取消。

NPU 安全策略(重要):
- Intel NPU 仅支持「对称 INT4 + 有界/静态形状」的 IR 模型(如 OpenVINO 官方
  `-int4-gq-ov` / `-int4-cw-ov` 系列);其它模型(旧版 `-int4-ov`、非对称量化、
  动态形状)会在 vpux 编译器里触发 LLVM ERROR 直接终止整个进程,Python 无法捕获。
- 因此首次 NPU 加载在独立子进程中编译并导出 blob(openvino_npu.blob),
  主进程只做 blob 导入(快、无编译、零崩溃风险);子进程失败只影响自身。
- 预检不通过 / 编译失败 → 明确回退 CPU,并把原因写进 `_load_detail`,
  由 UI 如实展示「当前实际运行设备」,绝不静默假装 NPU 在跑。
"""
from __future__ import annotations

import json
import logging
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Iterator, Optional

from .base import (
    BaseEngine, EngineError, GenParams,
    ModelLoadError, ModelNotLoadedError,
)

logger = logging.getLogger("novacore")

# NPU 预编译 blob 缓存文件名(与模型目录同级)
NPU_BLOB_NAME = "openvino_npu.blob"
NPU_META_NAME = "openvino_npu.meta"
# 首次 NPU 编译超时(秒):1.5B 模型首次编译可能数分钟
NPU_COMPILE_TIMEOUT = 1800.0


def build_prompt(messages: list[dict], system_prompt: str) -> str:
    """把消息列表拼成文本提示(旧版 openvino-genai 无 chat template 时兜底)。"""
    parts: list[str] = []
    if system_prompt:
        parts.append(f"<|system|>\n{system_prompt}")
    for m in messages:
        role = m.get("role", "user")
        content = str(m.get("content", ""))
        if role == "system":
            parts.append(f"<|system|>\n{content}")
        elif role == "assistant":
            parts.append(f"<|assistant|>\n{content}")
        else:
            parts.append(f"<|user|>\n{content}")
    parts.append("<|assistant|>\n")
    return "\n".join(parts)


def _npu_compile_subprocess_script() -> str:
    """子进程编译脚本:在 NPU 上编译模型并导出 blob。

    在子进程中执行的原因:vpux 编译器对不兼容模型会触发 LLVM ERROR 直接
    中止进程,Python 的 try/except 无法捕获;子进程崩溃只影响它自身。
    """
    return r'''
import os
import sys
import openvino_genai as ovg

model_dir, blob_path = sys.argv[1], sys.argv[2]
pipe = ovg.LLMPipeline(
    model_dir, "NPU",
    EXPORT_BLOB="YES", BLOB_PATH=blob_path, CACHE_MODE="OPTIMIZE_SPEED")
del pipe
sys.stdout.write("OK\n")
sys.stdout.flush()
# 直接退出:OpenVINO 插件卸载在部分机器上会挂起,os._exit 干净利落
os._exit(0)
'''


class OpenVinoEngine(BaseEngine):
    engine_id = "openvino"
    display_name = "OpenVINO NPU/CPU"

    def __init__(self):
        super().__init__()
        self._pipe = None
        self._device = "NPU"
        self._load_detail = ""  # 实际设备/回退原因,供 UI 如实展示

    # ---------- 可用性 ----------
    def check_available(self) -> tuple[bool, str]:
        try:
            import openvino  # noqa: F401
        except ImportError as e:
            return False, f"未安装 openvino(需 pip install openvino): {e.name}"
        try:
            import openvino_genai  # noqa: F401
        except ImportError as e:
            return False, f"未安装 openvino-genai(需 pip install openvino-genai): {e.name}"
        try:
            import openvino as ov
            core = ov.Core()
            devices = sorted(core.available_devices or [])
            npu = "NPU" in devices
            return True, (f"OpenVINO 可用设备: {devices}" +
                          ("(含 NPU)" if npu else "(无 NPU,可回退 CPU)"))
        except Exception as e:
            return False, f"OpenVINO 初始化异常: {e}"

    # ---------- NPU 预检 ----------
    def _npu_precheck(self, p: Path) -> tuple[bool, str]:
        """加载前 NPU 基础预检(毫秒级,不触发编译):
        1) 机器是否有 NPU 设备;
        2) 目标是否为 IR 目录(GGUF 无可靠预检手段,NPU 路径整体禁用)。
        模型是否真正能被 vpux 编译器编译,统一由子进程编译阶段把关——
        vpux 编译器对不兼容模型会触发 LLVM ERROR 直接杀死进程,
        必须把编译隔离在子进程里,而不是靠进程内的 try/except。
        """
        try:
            import openvino as ov
        except ImportError as e:
            return False, f"未安装 openvino: {e.name}"
        try:
            core = ov.Core()
            if "NPU" not in (core.available_devices or []):
                return False, "当前机器没有可用的 Intel NPU 设备"
        except Exception as e:
            return False, f"NPU 设备探测失败: {e}"
        if not p.is_dir():
            return False, ("GGUF 模型的量化格式无法在进程内安全预检;"
                           "Intel NPU 仅支持对称 INT4 的 IR 目录模型,"
                           "请下载 NPU 专区的 -int4-gq-ov/-int4-cw-ov 模型")
        if not (p / "openvino_model.xml").is_file():
            return False, "模型目录缺少 openvino_model.xml"
        return True, "OK"

    # ---------- NPU blob 编译缓存 ----------
    def _npu_blob_paths(self, p: Path) -> tuple[Path, Path]:
        return p / NPU_BLOB_NAME, p / NPU_META_NAME

    def _npu_blob_valid(self, p: Path, blob: Path, meta: Path) -> bool:
        """blob 是否存在且与当前模型文件/库版本一致。"""
        if not blob.is_file() or blob.stat().st_size <= 0:
            return False
        try:
            import openvino_genai as ovg
            lib_version = getattr(ovg, "__version__", "unknown")
        except Exception:
            lib_version = "unknown"
        try:
            data = json.loads(meta.read_text(encoding="utf-8"))
        except Exception:
            return False
        try:
            xml_mtime = (p / "openvino_model.xml").stat().st_mtime
            bin_mtime = (p / "openvino_model.bin").stat().st_mtime
        except OSError:
            return False
        return (abs(data.get("xml_mtime", 0) - xml_mtime) < 0.5
                and abs(data.get("bin_mtime", 0) - bin_mtime) < 0.5
                and data.get("lib_version") == lib_version)

    def _npu_compile_blob(self, p: Path, progress=None) -> None:
        """在子进程里完成 NPU 编译并导出 blob(进程崩溃不影响主程序)。

        源码运行:spawn `python -c <脚本>` 编译;
        PyInstaller 打包运行:spawn 自己带 `--npu-compile` 参数编译
        (windowed exe 无法用 -c,由 novacore_main 的入口分支处理)。
        """
        blob, meta = self._npu_blob_paths(p)
        try:
            import openvino_genai as ovg
            lib_version = getattr(ovg, "__version__", "unknown")
        except Exception:
            lib_version = "unknown"
        blob.unlink(missing_ok=True)
        meta.unlink(missing_ok=True)
        t0 = time.time()
        if progress:
            progress("首次 NPU 编译中(约 1-5 分钟,之后秒开)...")
        if getattr(sys, "frozen", False):
            cmd = [sys.executable, "--npu-compile", str(p), str(blob)]
        else:
            cmd = [sys.executable, "-c", _npu_compile_subprocess_script(),
                   str(p), str(blob)]
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
            creationflags=(subprocess.CREATE_NO_WINDOW
                           if sys.platform == "win32" else 0),
        )
        assert proc.stdout is not None
        lines: list[str] = []

        def _drain():
            try:
                for line in proc.stdout:
                    lines.append(line.strip())
                    if len(lines) > 60:
                        del lines[:-30]
            except Exception:
                pass

        reader = threading.Thread(target=_drain, daemon=True)
        reader.start()
        last_note = -1
        while proc.poll() is None:
            elapsed = int(time.time() - t0)
            if progress and elapsed != last_note and elapsed % 10 == 0:
                progress(f"NPU 编译中... 已用 {elapsed}s")
                last_note = elapsed
            time.sleep(1.0)
            if time.time() - t0 > NPU_COMPILE_TIMEOUT:
                proc.kill()
                raise ModelLoadError("NPU 编译超时(>30 分钟),已放弃 NPU 并回退 CPU")
        reader.join(timeout=5)
        tail = lines[-30:]
        ok = proc.returncode == 0 and blob.is_file() and blob.stat().st_size > 0
        if not ok:
            detail = "\n".join(tail[-12:]) or "(无输出)"
            raise ModelLoadError(
                f"NPU 编译失败(退出码 {proc.returncode});"
                "该模型与 NPU 不兼容,已回退 CPU。"
                f"编译日志: {detail}")
        try:
            meta.write_text(json.dumps({
                "xml_mtime": (p / "openvino_model.xml").stat().st_mtime,
                "bin_mtime": (p / "openvino_model.bin").stat().st_mtime,
                "lib_version": lib_version,
                "created_at": time.time(),
            }, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError:
            pass

    # ---------- 加载 ----------
    def load_model(self, path: str, device: Optional[str] = None,
                   progress=None) -> None:
        p = Path(path)
        # 新版 openvino-genai(2024.x+)支持直接加载 GGUF 文件;
        # 旧版只支持已转换的模型目录(含 .xml/.bin)。两种都尝试。
        is_dir = p.is_dir()
        is_gguf = p.is_file() and p.suffix.lower() == ".gguf"
        if not (is_dir or is_gguf):
            raise ModelLoadError(
                "OpenVINO 模型必须是已转换的模型目录(含 .xml/.bin)或 GGUF 文件;"
                "当前文件不被 OpenVINO 支持")
        try:
            import openvino_genai as ovg  # noqa: F401
        except ImportError as e:
            raise ModelLoadError(f"未安装 openvino-genai: {e.name}") from e
        self.unload()
        target = device or "NPU"
        self._load_detail = ""
        errors: list[str] = []

        # ---- 优先 NPU ----
        if "NPU" in str(target).upper():
            ok, why = self._npu_precheck(p)
            if not ok:
                errors.append(f"NPU: 预检未通过({why})")
                logger.info("OpenVINO NPU 预检未通过(%s): %s", p.name, why)
            else:
                try:
                    self._load_npu(p, progress)
                    self._loaded_path = str(p)
                    self._model_name = p.name
                    logger.info("OpenVINO 模型已在 NPU 上加载: %s", p.name)
                    return
                except ModelLoadError as e:
                    errors.append(str(e))
                    logger.warning("OpenVINO NPU 加载失败,将回退 CPU(%s): %s",
                                   p.name, str(e)[:200])
                except Exception as e:  # 兜底:任何 NPU 异常都走 CPU 回退
                    errors.append(f"NPU: {e}")
                    logger.warning("OpenVINO NPU 加载异常,将回退 CPU(%s): %s",
                                   p.name, e)
                    self._pipe = None

        # ---- CPU 回退 ----
        try:
            if progress and errors:
                progress("NPU 不可用,回退 CPU 加载...")
            self._pipe = ovg.LLMPipeline(str(p), "CPU")
            self._device = "CPU"
            self._loaded_path = str(p)
            self._model_name = p.name
            if errors:
                self._load_detail = f"已从 NPU 回退到 CPU({errors[-1][:200]})"
            else:
                self._load_detail = "CPU"
            logger.info("OpenVINO 模型已在 CPU 上加载: %s%s",
                        p.name, f"({self._load_detail})" if errors else "")
            return
        except Exception as e:
            errors.append(f"CPU: {e}")
            self._pipe = None
        raise ModelLoadError(
            f"模型加载失败: {'; '.join(errors)}"
            + ("\n提示:GGUF 模型需要 openvino-genai 2024.x+ 才支持;"
               "可升级 pip install -U openvino openvino-genai" if is_gguf else ""))

    def _load_npu(self, p: Path, progress=None) -> None:
        """NPU 加载:优先复用预编译 blob,否则子进程编译一次再导入。"""
        import openvino_genai as ovg
        if p.is_dir():
            blob, meta = self._npu_blob_paths(p)
            if not self._npu_blob_valid(p, blob, meta):
                self._npu_compile_blob(p, progress=progress)
            if progress:
                progress("正在导入 NPU 编译产物...")
            # blob 导入:无编译,进程安全;失败(如 blob 损坏)再删缓存重试一次
            try:
                self._pipe = ovg.LLMPipeline(str(p), "NPU", BLOB_PATH=str(blob))
            except Exception:
                blob.unlink(missing_ok=True)
                meta.unlink(missing_ok=True)
                self._npu_compile_blob(p, progress=progress)
                self._pipe = ovg.LLMPipeline(str(p), "NPU", BLOB_PATH=str(blob))
        else:
            # GGUF:量化格式无法在进程内安全预检,而 vpux 编译器对不兼容
            # 模型会直接杀死进程,因此 GGUF 一律不尝试 NPU,交由 CPU 兜底。
            raise ModelLoadError(
                "GGUF 模型暂不支持 NPU 推理(其量化格式无法保证 NPU 兼容,"
                "强行编译可能导致程序崩溃);请下载 NPU 专区的 -int4-gq-ov 模型")
        self._device = "NPU"
        self._load_detail = "NPU"

    def unload(self) -> None:
        self._pipe = None
        self._device = "NPU"
        self._load_detail = ""
        self._loaded_path = None
        self._model_name = None

    # ---------- 提示词 ----------
    def _build_prompt(self, messages: list[dict], params: GenParams) -> str:
        """优先使用模型自带 chat template(OpenAI 消息格式);
        不支持时回退手工拼装。"""
        pipe = self._pipe
        if pipe is not None:
            try:
                tok = pipe.get_tokenizer()
                if hasattr(tok, "apply_chat_template"):
                    return tok.apply_chat_template(
                        list(messages), add_generation_prompt=True)
            except Exception:
                pass
        return build_prompt(messages, params.system_prompt)

    # ---------- 生成 ----------
    def generate(self, messages: list[dict], params: GenParams) -> Iterator[str]:
        if not self.loaded or self._pipe is None:
            raise ModelNotLoadedError("尚未加载 OpenVINO 模型")
        self._check_cancel()
        import openvino_genai as ovg
        cfg = ovg.GenerationConfig()
        cfg.max_new_tokens = params.max_tokens
        try:
            cfg.temperature = params.temperature
            cfg.top_p = params.top_p
            cfg.top_k = params.top_k
        except Exception:
            pass  # 某些版本不支持这些参数,忽略即可
        prompt = self._build_prompt(messages, params)

        token_q: queue.Queue = queue.Queue()

        def _make_streamer():
            """兼容新版(≥2025.1) TextStreamer 与旧版 Streamer。

            回调返回值语义(新旧版一致):False/RUNNING = 继续;
            True/STOP = 停止。旧代码返回 True 表示继续导致 1-2 个 token
            后生成即中断(表现为"AI 只吐零零散散的字")。
            """
            if hasattr(ovg, "TextStreamer"):
                # 新版:TextStreamer(tokenizer, callback),回调收到解码后的文本
                tok = self._pipe.get_tokenizer()
                _status = getattr(ovg, "StreamingStatus", None)
                RUNNING = _status.RUNNING if _status is not None else False
                STOP = _status.STOP if _status is not None else True

                def _cb(token: str):
                    token_q.put(token)
                    if self._cancel_evt.is_set():
                        return STOP
                    return RUNNING

                return ovg.TextStreamer(tok, _cb)

            # 旧版(2024.x):Streamer 回调收到的是 token id(int),
            # 需要自行解码;返回 True 同样表示停止。
            tok = self._pipe.get_tokenizer()

            def _legacy_cb(token_id):
                try:
                    text = tok.decode([int(token_id)])
                except Exception:
                    text = ""
                token_q.put(text)
                if self._cancel_evt.is_set():
                    return True
                return False

            return ovg.Streamer(_legacy_cb)

        def worker():
            try:
                try:
                    self._pipe.start_chat()
                except Exception:
                    pass  # 新版已废弃 chat 模式,单次生成不需要
                try:
                    self._pipe.generate(prompt, cfg, _make_streamer())
                finally:
                    try:
                        self._pipe.finish_chat()
                    except Exception:
                        pass
            except Exception as e:
                token_q.put(EngineError(f"OpenVINO 推理失败: {e}"))
            finally:
                token_q.put(None)  # 结束哨兵

        threading.Thread(target=worker, daemon=True).start()
        while True:
            item = token_q.get()
            if item is None:
                break
            if isinstance(item, BaseException):
                if isinstance(item, EngineError):
                    raise item
                raise EngineError(str(item)) from item
            self._check_cancel()
            if item:
                yield item

    def describe(self) -> dict:
        info = super().describe()
        info["device"] = self._device if self.loaded else None
        info["load_detail"] = self._load_detail
        return info
