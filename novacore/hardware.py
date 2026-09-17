"""硬件能力探测(CPU / NVIDIA GPU / Intel NPU)与 GPU 实时监控。
所有探测均惰性导入 + 异常兜底:缺少 pynvml / openvino / nvidia-smi 时返回空结果,绝不抛异常。"""
import platform
import os
import subprocess
import sys
from functools import lru_cache

import psutil


def _run_hidden(cmd: list[str], timeout: float = 15.0) -> str:
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            encoding="utf-8", errors="replace",
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        return proc.stdout or ""
    except Exception:
        return ""


@lru_cache(maxsize=1)
def detect_cpu() -> dict:
    return {
        "model": platform.processor() or "Unknown CPU",
        "cores_physical": psutil.cpu_count(logical=False) or 0,
        "cores_logical": psutil.cpu_count(logical=True) or 0,
        "ram_total_gb": round(psutil.virtual_memory().total / (1024 ** 3), 1),
    }


@lru_cache(maxsize=1)
def detect_gpu() -> list[dict]:
    """返回 NVIDIA GPU 列表;无 GPU 或无检测手段时返回空列表。"""
    gpus: list[dict] = []
    # 方式一:pynvml
    try:
        import pynvml
        pynvml.nvmlInit()
        try:
            count = pynvml.nvmlDeviceGetCount()
            for i in range(count):
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(h)
                if not isinstance(name, str):
                    name = name.decode(errors="replace")
                mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                gpus.append({
                    "index": i, "name": name,
                    "vram_total_gb": round(mem.total / (1024 ** 3), 1),
                })
        finally:
            pynvml.nvmlShutdown()
        if gpus:
            return gpus
    except Exception:
        pass
    # 方式二:nvidia-smi 命令兜底
    out = _run_hidden(["nvidia-smi",
                       "--query-gpu=name,memory.total",
                       "--format=csv,noheader,nounits"])
    for i, line in enumerate(out.strip().splitlines()):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            try:
                gpus.append({"index": i, "name": parts[0],
                             "vram_total_gb": round(float(parts[1]) / 1024, 1)})
            except ValueError:
                continue
    return gpus


@lru_cache(maxsize=1)
def detect_npu() -> dict:
    """探测 Intel NPU(OpenVINO 设备列表中的 NPU)。"""
    info = {"available": False, "detail": "未检测到 OpenVINO 或 NPU 设备"}
    try:
        import openvino as ov
        core = ov.Core()
        devices = set(core.available_devices or [])
        if "NPU" in devices:
            info = {"available": True, "detail": "Intel NPU 可用(OpenVINO)"}
        else:
            info = {"available": False, "detail": f"OpenVINO 可用设备: {sorted(devices) or '无'}"}
    except Exception as e:
        info = {"available": False, "detail": f"OpenVINO 不可用: {e}"}
    return info


@lru_cache(maxsize=1)
def get_system_info() -> dict:
    return {
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "cpu": detect_cpu(),
        "gpus": detect_gpu(),
        "npu": detect_npu(),
    }


def gpu_stats() -> dict:
    """GPU 实时占用(供监控线程);无 GPU/驱动时返回空 dict。"""
    try:
        import pynvml
        pynvml.nvmlInit()
        try:
            if pynvml.nvmlDeviceGetCount() < 1:
                return {}
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            util = pynvml.nvmlDeviceGetUtilizationRates(h)
            return {
                "gpu_vram_used": round(mem.used / (1024 ** 3), 2),
                "gpu_vram_total": round(mem.total / (1024 ** 3), 2),
                "gpu_util": util.gpu,
            }
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        return {}


def has_cuda_gpu() -> bool:
    return len(detect_gpu()) > 0


def has_npu() -> bool:
    return detect_npu()["available"]


def ensure_telemetry_optout() -> None:
    """若用户从未做过 OpenVINO 遥测选择,则写入拒绝(0)。

    OpenVINO 默认(无 consent 文件时)会向 Google Analytics 上报使用数据,
    其后台发送线程在无外网/受限网络环境下会阻塞解释器退出(实测卡死);
    写入官方 consent 文件(%LOCALAPPDATA%\\Intel Corporation\\openvino_telemetry)
    内容 '0' 可彻底关闭遥测。已存在任何内容(用户明确选择过)则不动。
    """
    if sys.platform != "win32":
        return
    try:
        base = os.environ.get("LOCALAPPDATA")
        if not base:
            return
        consent = os.path.join(base, "Intel Corporation", "openvino_telemetry")
        if os.path.exists(consent):
            return  # 用户已做过选择(1=同意 / 0=拒绝),尊重之
        os.makedirs(os.path.dirname(consent), exist_ok=True)
        with open(consent, "w", encoding="utf-8") as f:
            f.write("0")
    except OSError:
        pass  # 写失败不影响主流程;退出挂起由主程序 os._exit 兜底
