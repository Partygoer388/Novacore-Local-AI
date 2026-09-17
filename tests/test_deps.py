# -*- coding: utf-8 -*-
"""novacore.deps 自测:直接运行 `python tests/test_deps.py`。"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from novacore import deps


class TestDeps(unittest.TestCase):

    def test_module_map_llama_cpp(self):
        """关键修复:llama-cpp-python 的导入模块是 llama_cpp 而非 llama_cpp_python。"""
        entry = [d for d in deps.DEPENDENCY_LIST if d["key"] == "llama-cpp-python"][0]
        self.assertEqual(entry["module"], "llama_cpp")
        self.assertTrue(deps.module_available("sys"))

    def test_pyqt6_module_case(self):
        """关键修复:PyQt6 模块名大小写正确,检测不会误报缺失。"""
        entry = [d for d in deps.DEPENDENCY_LIST if d["key"] == "PyQt6"][0]
        self.assertEqual(entry["module"], "PyQt6")
        self.assertTrue(deps.module_available("PyQt6"))

    def test_check_deps_env_snapshot(self):
        status = {d["key"]: d["installed"] for d in deps.check_deps()}
        self.assertEqual(len(status), 12)
        self.assertIs(status["PyQt6"], True)            # 核心依赖本机必装
        self.assertIs(status["psutil"], True)
        self.assertIs(status["requests"], True)
        # 可选依赖的安装状态必须与实际模块可导入性一致(不写死,兼容已安装环境)
        for key, module in [("llama-cpp-python", "llama_cpp"),
                            ("peft", "peft"),
                            ("nvidia-ml-py", "pynvml"),
                            ("torch", "torch"),
                            ("openvino", "openvino")]:
            self.assertIs(status[key], deps.module_available(module),
                          f"{key} 的检测状态应与 module_available({module}) 一致")

    def test_required_all_present(self):
        """核心依赖缺失应为空(本机环境已具备)。"""
        self.assertEqual(deps.missing_required(), [])

    def test_build_pip_cmd(self):
        cmd = deps.build_pip_cmd("peft", deps.PIP_MIRRORS[0][0])
        self.assertEqual(cmd[0:3], [sys.executable, "-m", "pip"])
        self.assertIn("install", cmd)
        self.assertIn("peft", cmd)
        self.assertIn("https://pypi.org/simple", cmd)

    def test_mirrors_have_china_options(self):
        urls = [m[0] for m in deps.PIP_MIRRORS]
        self.assertIn("https://pypi.tuna.tsinghua.edu.cn/simple", urls)
        self.assertIn("https://mirrors.aliyun.com/pypi/simple", urls)

    def test_install_missing_pkg_fails_gracefully(self):
        """安装不存在的包应抛 PipInstallError 而不是崩溃(走完 3 个镜像后)。"""
        with self.assertRaises(deps.PipInstallError):
            deps.install_package("this-package-does-not-exist-xyz-12345",
                                 progress=lambda _: None)

    def test_cancel_cooperates(self):
        """取消检查生效时立即中断。"""
        with self.assertRaises(InterruptedError):
            deps.install_package("peft", progress=lambda _: None,
                                 cancel_check=lambda: True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
