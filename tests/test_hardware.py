# -*- coding: utf-8 -*-
"""novacore.hardware 自测:直接运行 `python tests/test_hardware.py`。
本机没有 pynvml/nvidia-smi/独立显卡,重点验证所有探测函数不抛异常并返回结构正确的数据。"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from novacore import hardware


class TestHardware(unittest.TestCase):

    def test_detect_cpu(self):
        cpu = hardware.detect_cpu()
        self.assertGreater(cpu["cores_logical"], 0)
        self.assertGreater(cpu["ram_total_gb"], 0)
        self.assertIn("model", cpu)

    def test_get_system_info_structure(self):
        info = hardware.get_system_info()
        for key in ("os", "python", "cpu", "gpus", "npu"):
            self.assertIn(key, info)
        self.assertIsInstance(info["gpus"], list)
        self.assertIsInstance(info["npu"], dict)
        self.assertIn("available", info["npu"])

    def test_detect_gpu_never_raises(self):
        gpus = hardware.detect_gpu()  # 本机预期空列表,但绝不能抛异常
        self.assertIsInstance(gpus, list)
        for g in gpus:
            self.assertIn("name", g)
            self.assertIn("vram_total_gb", g)

    def test_detect_npu_never_raises(self):
        npu = hardware.detect_npu()
        self.assertIsInstance(npu["available"], bool)
        self.assertIn("detail", npu)

    def test_gpu_stats_never_raises(self):
        stats = hardware.gpu_stats()  # 无 GPU 时返回空 dict
        self.assertIsInstance(stats, dict)

    def test_bool_helpers(self):
        self.assertIsInstance(hardware.has_cuda_gpu(), bool)
        self.assertIsInstance(hardware.has_npu(), bool)

    def test_caching_consistent(self):
        self.assertEqual(hardware.detect_cpu(), hardware.detect_cpu())
        self.assertEqual(hardware.get_system_info(), hardware.get_system_info())


if __name__ == "__main__":
    unittest.main(verbosity=2)
