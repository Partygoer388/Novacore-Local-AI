"""依赖检测与安装:
- 修正原代码的模块名错误(PyQt6 大小写、llama-cpp-python 实为 llama_cpp)
- 安装使用 subprocess + 流式读取,避免 os.system 的编码乱码与不可取消问题
- 支持国内镜像:官方 PyPI 失败后自动切换清华/阿里云镜像重试
"""
import importlib
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import requests

# key = pip 包名;module = import 模块名;optional = 缺少不影响核心功能
DEPENDENCY_LIST = [
    {"key": "PyQt6",              "module": "PyQt6",              "name": "PyQt6 界面框架",    "desc": "图形界面运行基础",            "optional": False},
    {"key": "psutil",             "module": "psutil",             "name": "Psutil 系统监控",    "desc": "硬件资源占用检测",              "optional": False},
    {"key": "requests",           "module": "requests",           "name": "Requests 网络库",     "desc": "模型下载、网络请求",            "optional": False},
    {"key": "llama-cpp-python",   "module": "llama_cpp",          "name": "llama.cpp 推理引擎", "desc": "CPU/GPU 通用 GGUF 推理",        "optional": True},
    {"key": "torch",              "module": "torch",              "name": "PyTorch",             "desc": "训练与部分引擎依赖",            "optional": True},
    {"key": "transformers",       "module": "transformers",       "name": "Transformers",        "desc": "LoRA 微调基础库",               "optional": True},
    {"key": "peft",               "module": "peft",               "name": "PEFT",                "desc": "LoRA 微调参数高效库",           "optional": True},
    {"key": "datasets",           "module": "datasets",           "name": "Datasets",            "desc": "训练数据集加载",                "optional": True},
    {"key": "openvino",           "module": "openvino",           "name": "OpenVINO NPU 引擎",   "desc": "Intel NPU 硬件推理加速",        "optional": True},
    {"key": "fastapi",            "module": "fastapi",            "name": "FastAPI 接口框架",    "desc": "API 服务后端框架",              "optional": True},
    {"key": "uvicorn",            "module": "uvicorn",            "name": "Uvicorn 服务器",      "desc": "API 服务运行容器",              "optional": True},
    {"key": "nvidia-ml-py",       "module": "pynvml",             "name": "NVIDIA 显卡监控",     "desc": "GPU 显存/利用率监控(可选)",    "optional": True},
]

# 国内镜像(按优先级):官方 + 多个国内镜像,覆盖不同网络环境
PIP_MIRRORS = [
    ("https://pypi.org/simple", "官方 PyPI"),
    ("https://pypi.tuna.tsinghua.edu.cn/simple", "清华镜像"),
    ("https://mirrors.aliyun.com/pypi/simple", "阿里云镜像"),
    ("https://pypi.doubanio.com/simple", "豆瓣镜像"),
    ("https://mirrors.ustc.edu.cn/pypi/web/simple", "中科大镜像"),
    ("https://repo.huaweicloud.com/repository/pypi/simple", "华为云镜像"),
    ("https://mirrors.cloud.tencent.com/pypi/simple", "腾讯云镜像"),
    ("https://mirrors.163.com/pypi/simple", "网易镜像"),
    ("https://mirror.sjtu.edu.cn/pypi/web/simple", "上海交大镜像"),
]

# 与 PIP_MIRRORS 一一对应的稳定 id(config 用)
PIP_MIRROR_IDS = [
    "official", "tsinghua", "aliyun", "douban", "ustc",
    "huawei", "tencent", "netease", "sjtu",
]

# 设置页选择项(id → 显示名)
PIP_MIRROR_CHOICES = ["auto"] + PIP_MIRROR_IDS


def mirror_by_id(mirror_id: str):
    """按 id 返回 (url, label);未知返回 None。"""
    if mirror_id in PIP_MIRROR_IDS:
        return PIP_MIRRORS[PIP_MIRROR_IDS.index(mirror_id)]
    return None


def mirror_order(preference: str = "auto") -> list[tuple[str, str]]:
    """按偏好生成有序镜像列表:指定镜像优先,其余兜底;auto=默认顺序。"""
    mirror_id = str(preference or "auto").strip()
    target = mirror_by_id(mirror_id)
    if target is None:
        return list(PIP_MIRRORS)
    rest = [m for m in PIP_MIRRORS if m != target]
    return [target] + rest


def probe_mirror(url: str, timeout: float = 3.0):
    """探测单个镜像可用性与延迟。返回 (ok, latency_ms);失败 latency 为 None。"""
    try:
        t0 = time.time()
        with requests.get(url, stream=True, timeout=timeout) as r:
            latency = (time.time() - t0) * 1000
            return r.status_code < 500, round(latency, 1)
    except Exception:
        return False, None


def probe_pip_mirrors(timeout: float = 3.0) -> list[dict]:
    """并行探测全部镜像,返回 [{id, label, url, ok, latency_ms}] 供 UI 展示。
    并行化把总等待从 ~30s 降到单次请求耗时(~3s)。"""
    items = list(zip(PIP_MIRROR_IDS, PIP_MIRRORS))

    def _probe(item):
        mid, (url, label) = item
        ok, lat = probe_mirror(url, timeout=timeout)
        return {"id": mid, "label": label, "url": url,
                "ok": ok, "latency_ms": lat}

    with ThreadPoolExecutor(max_workers=len(items)) as pool:
        return list(pool.map(_probe, items))


def module_available(module: str) -> bool:
    try:
        importlib.import_module(module)
        return True
    except Exception:
        return False


def check_deps() -> list[dict]:
    """返回每项依赖的检测结果(附带 installed 状态)。"""
    result = []
    for dep in DEPENDENCY_LIST:
        item = dict(dep)
        item["installed"] = module_available(dep["module"])
        result.append(item)
    return result


def missing_required() -> list[dict]:
    return [d for d in check_deps() if not d["installed"] and not d["optional"]]


def missing_all() -> list[dict]:
    return [d for d in check_deps() if not d["installed"]]


# llama-cpp-python 预编译 CPU wheel 源:安装时避免从源码编译(极慢)
LLAMA_CPP_WHEEL_INDEX = "https://abetlen.github.io/llama-cpp-python/whl/cpu"

# GitHub Pages 在校内网往往缓慢/不可达;探测结果缓存避免反复试探
_wheel_reachable_cache: tuple[float, bool] = (0.0, True)


def llama_wheel_reachable(timeout: float = 3.0) -> bool:
    """探测 llama-cpp-python 预编译 wheel 源可达性(带 60s 缓存)。
    不可达时安装会跳过该源,避免 pip 长时间卡在 GitHub Pages。"""
    global _wheel_reachable_cache
    now = time.time()
    if now - _wheel_reachable_cache[0] < 60:
        return _wheel_reachable_cache[1]
    ok = False
    try:
        with requests.get(LLAMA_CPP_WHEEL_INDEX, stream=True, timeout=timeout) as r:
            ok = r.status_code < 500
    except Exception:
        ok = False
    _wheel_reachable_cache = (now, ok)
    return ok


def build_pip_cmd(package: str, mirror_index: str) -> list[str]:
    cmd = [
        sys.executable, "-m", "pip", "install", package,
        "-i", mirror_index,
        "--disable-pip-version-check",
    ]
    if str(package).lower() == "llama-cpp-python":
        # 优先用官方预编译 wheel,学校网络下避免 C++ 源码编译超时;
        # 仅当 wheel 源可达时才追加,否则跳过以免 pip 长时间卡住。
        cmd += ["--prefer-binary"]
        if llama_wheel_reachable():
            cmd += ["--extra-index-url", LLAMA_CPP_WHEEL_INDEX]
    return cmd


class PipInstallError(RuntimeError):
    pass


# 自动模式下探测出的最佳镜像顺序缓存(URL, label);TTL 内复用,避免重复自检
_best_mirror_cache: tuple[float, list[tuple[str, str]]] = (0.0, [])


def _best_mirror_order(progress=None, ttl: float = 120.0) -> list[tuple[str, str]]:
    """自动模式获取按延迟升序的可用镜像列表(带 TTL 缓存)。
    同时并发探测 wheel 源可达性(结果写入缓存),避免安装时二次阻塞。"""
    global _best_mirror_cache
    now = time.time()
    if now - _best_mirror_cache[0] < ttl and _best_mirror_cache[1]:
        return _best_mirror_cache[1]

    if progress:
        progress("🔎 正在并行自检 pip 镜像与 wheel 节点...")
    # 并发执行镜像探测 + wheel 源探测(结果各自缓存)
    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_mirrors = pool.submit(probe_pip_mirrors, 3.0)
        pool.submit(llama_wheel_reachable, 3.0)
        results = fut_mirrors.result()

    results.sort(key=lambda r: (not r["ok"],
                                r["latency_ms"] if r["latency_ms"] else 999999))
    order = [(r["url"], r["label"]) for r in results if r["ok"]]
    if not order:
        order = mirror_order("auto")  # 全部不可达则退回默认顺序(逐个失败)
    _best_mirror_cache = (now, order)
    if progress:
        parts = []
        for r in results:
            lat = r.get("latency_ms")
            lat_txt = f"{lat:.0f}ms" if lat else "不可达"
            parts.append(f"{r['label']}={lat_txt}")
        best = order[0][1] if order else "无"
        progress("节点自检: " + ", ".join(parts) + f" → 选用 {best}")
    return order


def install_package(package: str, progress=None, cancel_check=None,
                    mirror: str = "auto") -> bool:
    """同步安装单个包;镜像故障转移。progress(line), cancel_check()->bool。
    mirror: 为 PIP_MIRROR_IDS 之一时该镜像优先,auto 自检节点按延迟排序(带缓存)。"""
    if str(mirror or "auto") == "auto":
        order = _best_mirror_order(progress)
    else:
        order = mirror_order(mirror)
    last_error: Optional[str] = None
    for index, label in order:
        if cancel_check is not None and cancel_check():
            raise InterruptedError("安装已取消")
        if progress:
            progress(f"[{label}] 正在安装 {package} ...")
        cmd = build_pip_cmd(package, index)
        proc = None
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            )
            assert proc.stdout is not None
            try:
                for line in proc.stdout:
                    line = line.rstrip()
                    if line and progress:
                        progress(line)
                    if cancel_check is not None and cancel_check():
                        proc.terminate()
                        try:
                            proc.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                        raise InterruptedError("安装已取消")
                ret = proc.wait()
            finally:
                proc.stdout.close()
            if ret == 0:
                if progress:
                    progress(f"✅ {package} 安装成功")
                return True
            last_error = f"{package} 安装返回码 {ret}"
        except InterruptedError:
            raise
        except Exception as e:
            last_error = f"{package} 安装异常: {e}"
        finally:
            if proc is not None and proc.poll() is None:
                try:
                    proc.terminate()
                except OSError:
                    pass
        if progress:
            progress(f"❌ [{label}] {last_error},尝试下一个镜像...")
    raise PipInstallError(last_error or "未知错误")


def check_pip_available() -> bool:
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pip", "--version"],
            capture_output=True, timeout=30,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        return proc.returncode == 0
    except Exception:
        return False
