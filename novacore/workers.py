"""通用后台线程(QThread 封装),统一支持取消与优雅退出。"""
import threading

from PyQt6.QtCore import QThread, pyqtSignal

import psutil

from . import deps


class DepInstallWorker(QThread):
    """依赖安装线程:subprocess 流式输出 + 可取消 + 镜像故障转移。"""
    log = pyqtSignal(str)
    done = pyqtSignal(bool, str)

    def __init__(self, packages, parent=None, mirror: str = "auto"):
        super().__init__(parent)
        self.packages = list(packages)
        self.mirror = mirror or "auto"
        self._cancel = threading.Event()

    def cancel(self):
        self._cancel.set()

    def run(self):
        all_ok = True
        for pkg in self.packages:
            if self._cancel.is_set():
                self.done.emit(False, "安装已取消")
                return
            try:
                deps.install_package(
                    pkg,
                    progress=lambda line: self.log.emit(line),
                    cancel_check=self._cancel.is_set,
                    mirror=self.mirror,
                )
            except InterruptedError:
                self.log.emit("⚠️ 安装已取消")
                self.done.emit(False, "安装已取消")
                return
            except deps.PipInstallError as e:
                self.log.emit(f"❌ {pkg} 安装失败: {e}")
                all_ok = False
        self.done.emit(all_ok, "全部依赖安装完成" if all_ok else "部分依赖安装失败")


class HardwareMonitorThread(QThread):
    """硬件监控线程:可停止,退出前不再有无限循环导致的进程挂起。"""
    stat = pyqtSignal(dict)

    def __init__(self, interval: float = 1.0, parent=None):
        super().__init__(parent)
        self._interval = max(0.2, float(interval))
        self._stop_evt = threading.Event()

    def stop(self, timeout_s: float = 5.0) -> None:
        """请求停止并等待线程结束(幂等,可重复调用)。"""
        self._stop_evt.set()
        if self.isRunning():
            self.wait(int(timeout_s * 1000))

    def run(self):
        while not self._stop_evt.is_set():
            try:
                mem = psutil.virtual_memory()
                stat = {
                    "ram_used": round(mem.used / (1024 ** 3), 2),
                    "ram_total": round(mem.total / (1024 ** 3), 2),
                    "cpu": psutil.cpu_percent(interval=None),
                }
                try:
                    from . import hardware
                    stat.update(hardware.gpu_stats())
                except Exception:
                    pass  # GPU 监控失败不影响 CPU/RAM 数据
                self.stat.emit(stat)
            except Exception:
                pass  # 监控失败不影响主流程
            self._stop_evt.wait(self._interval)


class ModelLoadWorker(QThread):
    """在后台线程加载模型(加载可能耗时数秒,避免卡死界面)。
    首次 NPU 编译可能需要数分钟,progress 信号持续回报进度文案。"""
    done = pyqtSignal(bool, str, dict)
    progress = pyqtSignal(str)

    def __init__(self, engine_mgr, model_ref: str, engine_id: str,
                 device=None, parent=None):
        super().__init__(parent)
        self._mgr = engine_mgr
        self._model_ref = model_ref
        self._engine_id = engine_id
        self._device = device

    def run(self):
        try:
            engine = self._mgr.load(self._model_ref,
                                    engine_id=self._engine_id,
                                    device=self._device,
                                    progress=lambda s: self.progress.emit(str(s)))
            desc = engine.describe() if hasattr(engine, "describe") else {}
            self.done.emit(True, f"模型已加载: {engine.model_name}", {
                "engine_id": engine.engine_id,
                "detail": getattr(engine, "_load_detail", ""),
                "device": desc.get("device"),
            })
        except Exception as e:
            self.done.emit(False, str(e), {})


class EngineProbeWorker(QThread):
    """后台探测各引擎可用性(避免启动时阻塞界面)。"""
    result = pyqtSignal(list)

    def __init__(self, engine_mgr, parent=None):
        super().__init__(parent)
        self._mgr = engine_mgr

    def run(self):
        self.result.emit(self._mgr.probe_all(refresh=True))


class WorkerRegistry:
    """统一登记后台线程:finished 自动移除,shutdown 时全部取消/停止并等待。
    修复原代码线程对象被覆盖/销毁时仍在运行导致的崩溃。"""

    def __init__(self):
        self._workers: list[QThread] = []
        self._lock = threading.Lock()

    def register(self, worker: QThread) -> QThread:
        with self._lock:
            self._workers.append(worker)
        worker.finished.connect(lambda w=worker: self._discard(w))
        return worker

    def _discard(self, worker: QThread) -> None:
        with self._lock:
            if worker in self._workers:
                self._workers.remove(worker)

    def active_count(self) -> int:
        with self._lock:
            return len(self._workers)

    def shutdown(self, wait_ms: int = 8000) -> None:
        """取消/停止所有已登记线程并等待其结束。"""
        with self._lock:
            workers = list(self._workers)
        for w in workers:
            if not w.isRunning():
                continue
            try:
                stop = getattr(w, "stop", None)
                cancel = getattr(w, "cancel", None)
                if stop is not None:
                    stop(1.0)          # 优雅停止(如硬件监控)
                elif cancel is not None:
                    cancel()           # 协作取消(如安装/下载)
            except Exception:
                pass
        for w in workers:
            if w.isRunning():
                w.wait(wait_ms)
