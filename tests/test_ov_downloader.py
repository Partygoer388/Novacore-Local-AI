# -*- coding: utf-8 -*-
"""OVDirectoryDownloadWorker 自测(本地 HTTP,不依赖外网):
覆盖多文件目录下载+大小校验、连接掐断自动断点续传、404 自动换源、
仓库不存在(RepoNotFoundError)干净退出不留空目录、取消保留 .part 可续传。"""
import os
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PyQt6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication([])

from novacore import modelstore  # noqa: E402
from novacore.downloader import OVDirectoryDownloadWorker  # noqa: E402


# 测试"仓库"文件清单
FILES = {
    "config.json": b'{"ok": true}',
    "openvino_model.xml": b"<xml>model</xml>" * 100,
    "openvino_model.bin": os.urandom(3 * 1024 * 1024 + 777),  # 3MB+
    "tokenizer.json": b"{}" * 50,
}


def _manifest():
    return [{"path": n, "size": len(b)} for n, b in FILES.items()]


class OVHandler(BaseHTTPRequestHandler):
    """支持 Range 的 OV 文件服务;可模拟每连接截断(cap)与 404。"""
    serve_dir = None
    cap = None            # 每连接最多发 cap 字节就关(模拟网络掐断)
    not_found = False     # 整个服务对所有文件返回 404

    def do_GET(self):
        name = self.path.lstrip("/").split("?")[0]
        if self.not_found or name not in FILES:
            self.send_response(404)
            self.end_headers()
            return
        payload = FILES[name]
        total = len(payload)
        rng = self.headers.get("Range", "")
        start = int(rng.split("=")[1].split("-")[0]) if "=" in rng else 0
        end = total - 1
        if self.cap:
            end = min(end, start + self.cap - 1)
        self.send_response(206 if start else 200)
        if start:
            self.send_header("Content-Range",
                             f"bytes {start}-{end}/{total}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        try:
            self.wfile.write(payload[start:end + 1])
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, *a):
        pass


class OVServer:
    def __init__(self, cap=None, not_found=False):
        handler = type("H", (OVHandler,),
                       {"cap": cap, "not_found": not_found})
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(
            target=self.httpd.serve_forever, daemon=True)

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}"

    def start(self):
        self.thread.start()

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def make_entry():
    return {"name": "T", "repo": "org/model", "ms_repo": "org/model",
            "format": "openvino_dir", "hw": ["NPU"]}


def run(w, timeout=60.0):
    results = {}
    sources = []
    w.done.connect(lambda ok, msg, p: results.update(ok=ok, msg=msg, path=p))
    w.source.connect(lambda sid, url: sources.append(sid))
    w.start()
    end = time.time() + timeout
    while time.time() < end and "ok" not in results:
        _app.processEvents()
        time.sleep(0.01)
    w.wait(3000)
    return results, sources


def patched_sources(url_map):
    """构造 build_ov_file_sources 的替身:三源指向给定 URL。"""
    def _fake(repo, file_name, ms_repo=None):
        return {sid: f"{base}/{file_name}" for sid, base in url_map.items()}
    return _fake


class TestOVDownloader(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dst = os.path.join(self._tmp.name, "Model-OV")

    def _patch_manifest(self):
        return mock.patch.object(
            modelstore, "list_ov_repo_files",
            lambda repo, ms_repo=None, preference="auto", timeout=15:
                _manifest())

    def test_full_directory_download_with_size_check(self):
        srv = OVServer()
        srv.start()
        self.addCleanup(srv.stop)
        url_map = {"official": srv.url, "hf_mirror": srv.url,
                   "modelscope": srv.url}
        with self._patch_manifest(), \
             mock.patch.object(modelstore, "build_ov_file_sources",
                               patched_sources(url_map)):
            w = OVDirectoryDownloadWorker(make_entry(), self.dst,
                                          preference="modelscope")
            results, _ = run(w)
        self.assertTrue(results["ok"], results.get("msg"))
        for name, payload in FILES.items():
            p = os.path.join(self.dst, name)
            self.assertTrue(os.path.exists(p), f"{name} 应存在")
            self.assertEqual(Path(p).read_bytes(), payload)
            self.assertFalse(os.path.exists(p + ".part"))

    def test_resume_through_repeated_connection_drops(self):
        """关键:服务器每连接只发 256KB 就掐断,worker 必须自动断点续传完成。"""
        srv = OVServer(cap=256 * 1024)
        srv.start()
        self.addCleanup(srv.stop)
        url_map = {"modelscope": srv.url, "hf_mirror": srv.url,
                   "official": srv.url}
        with self._patch_manifest(), \
             mock.patch.object(modelstore, "build_ov_file_sources",
                               patched_sources(url_map)):
            w = OVDirectoryDownloadWorker(make_entry(), self.dst,
                                          preference="auto")
            w.BACKOFF = (0, 0, 0, 0, 0, 0)  # 测试不等待
            results, _ = run(w, timeout=90)
        self.assertTrue(results["ok"], results.get("msg"))
        binp = os.path.join(self.dst, "openvino_model.bin")
        self.assertEqual(
            os.path.getsize(binp), len(FILES["openvino_model.bin"]),
            "掐断续传后 bin 字节数必须与清单完全一致")

    def test_404_fatally_skips_to_next_source(self):
        bad = OVServer(not_found=True)
        bad.start()
        self.addCleanup(bad.stop)
        good = OVServer()
        good.start()
        self.addCleanup(good.stop)
        url_map = {"official": bad.url, "modelscope": bad.url,
                   "hf_mirror": good.url}
        with self._patch_manifest(), \
             mock.patch.object(modelstore, "build_ov_file_sources",
                               patched_sources(url_map)):
            w = OVDirectoryDownloadWorker(make_entry(), self.dst,
                                          preference="official")
            results, sources = run(w)
        self.assertTrue(results["ok"], results.get("msg"))
        # official 排第一(404)→ modelscope(404)→ hf_mirror 成功
        self.assertEqual(sources[0], "official")
        self.assertIn("hf_mirror", sources)

    def test_repo_missing_does_not_create_empty_dir(self):
        with mock.patch.object(
                modelstore, "list_ov_repo_files",
                side_effect=modelstore.RepoNotFoundError("不存在")):
            w = OVDirectoryDownloadWorker(make_entry(), self.dst)
            results, _ = run(w)
        self.assertFalse(results["ok"])
        self.assertIn("不存在", results["msg"])
        self.assertFalse(
            os.path.exists(self.dst),
            "仓库不存在时必须在创建目录之前退出,不留 0KB 空目录")

    def test_cancel_keeps_partial_for_resume(self):
        srv = OVServer(cap=64 * 1024)   # 慢速:每连接 64KB
        srv.start()
        self.addCleanup(srv.stop)
        url_map = {"modelscope": srv.url, "hf_mirror": srv.url,
                   "official": srv.url}
        with self._patch_manifest(), \
             mock.patch.object(modelstore, "build_ov_file_sources",
                               patched_sources(url_map)):
            w = OVDirectoryDownloadWorker(make_entry(), self.dst,
                                          preference="auto")
            w.BACKOFF = (0, 0, 0, 0, 0, 0)
            w.start()
            part = os.path.join(self.dst, "openvino_model.bin.part")
            end = time.time() + 20
            while time.time() < end and not (
                    os.path.exists(part)
                    and os.path.getsize(part) > 128 * 1024):
                _app.processEvents()
                time.sleep(0.01)
            self.assertTrue(os.path.exists(part), "应已产生 .part 半成品")
            w.cancel()
            w.wait(5000)
            # .part 保留,供下次续传
            self.assertTrue(os.path.exists(part),
                            "取消后应保留 .part 以便断点续传")


if __name__ == "__main__":
    unittest.main(verbosity=2)
