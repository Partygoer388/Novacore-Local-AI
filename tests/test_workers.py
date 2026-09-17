# -*- coding: utf-8 -*-
"""novacore.workers 线程生命周期自测:直接运行 `python tests/test_workers.py`。
重点验证:监控线程可停止、登记表 shutdown 不再导致退出崩溃/挂起。"""
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PyQt6.QtCore import QThread
from PyQt6.QtWidgets import QApplication

_app = QApplication.instance() or QApplication([])

from novacore.workers import HardwareMonitorThread, WorkerRegistry, DepInstallWorker


def process_for(seconds: float):
    end = time.time() + seconds
    while time.time() < end:
        _app.processEvents()
        time.sleep(0.01)


class SleepWorker(QThread):
    """模拟长任务线程:仅在 stop() 后才会结束。"""
    def __init__(self):
        super().__init__()
        import threading
        self._evt = threading.Event()
        self._stopped = False

    def stop(self, timeout_s=5.0):
        self._evt.set()
        self._stopped = True
        if self.isRunning():
            self.wait(int(timeout_s * 1000))

    def run(self):
        self._evt.wait(30.0)


class TestHardwareMonitor(unittest.TestCase):

    def test_emits_stats_and_stops_cleanly(self):
        stats = []
        t = HardwareMonitorThread(interval=0.2)
        t.stat.connect(stats.append)
        t.start()
        process_for(0.9)
        self.assertTrue(t.isRunning())
        self.assertGreaterEqual(len(stats), 1, "应至少收到一次硬件统计")
        s = stats[0]
        for k in ("ram_used", "ram_total", "cpu"):
            self.assertIn(k, s)
        self.assertGreater(s["ram_total"], 0)
        t.stop()
        self.assertFalse(t.isRunning())
        t.stop()  # 幂等,重复 stop 不报错

    def test_stop_before_start_is_safe(self):
        t = HardwareMonitorThread(interval=0.2)
        t.stop()
        self.assertFalse(t.isRunning())


class TestWorkerRegistry(unittest.TestCase):

    def test_shutdown_stops_all_workers(self):
        reg = WorkerRegistry()
        hw = reg.register(HardwareMonitorThread(interval=0.2))
        sw = reg.register(SleepWorker())
        hw.start()
        sw.start()
        process_for(0.3)
        self.assertTrue(hw.isRunning())
        self.assertTrue(sw.isRunning())
        reg.shutdown(wait_ms=5000)
        process_for(0.2)  # 处理 finished 信号完成自动移除
        self.assertFalse(hw.isRunning())
        self.assertFalse(sw.isRunning())
        self.assertEqual(reg.active_count(), 0)

    def test_finished_workers_auto_discarded(self):
        reg = WorkerRegistry()

        class QuickWorker(QThread):
            def run(self):
                time.sleep(0.05)

        qw = reg.register(QuickWorker())
        self.assertEqual(reg.active_count(), 1)
        qw.start()
        process_for(0.5)
        self.assertEqual(reg.active_count(), 0, "finished 后应自动从登记表移除")


class TestDepInstallWorker(unittest.TestCase):

    def test_cancel_before_start_no_network(self):
        results = {}
        w = DepInstallWorker(["peft"])
        w.done.connect(lambda ok, msg: results.update(ok=ok, msg=msg))
        w.cancel()  # 启动前取消:应立即结束且不发起任何网络请求
        w.start()
        process_for(0.5)
        self.assertIn("ok", results)
        self.assertIs(results["ok"], False)
        self.assertIn("取消", results["msg"])
        self.assertFalse(w.isRunning())


if __name__ == "__main__":
    unittest.main(verbosity=2)
