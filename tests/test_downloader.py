# -*- coding: utf-8 -*-
"""novacore.downloader 自测(本地 HTTP 服务器,不依赖外网):
直接运行 `python tests/test_downloader.py`。
覆盖:正常下载+GGUF 校验、多源故障转移、断点续传、取消清理、假文件(HTML)拦截、SHA256 校验。"""
import hashlib
import os
import struct
import sys
import tempfile
import threading
import time
import unittest
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PyQt6.QtWidgets import QApplication

_app = QApplication.instance() or QApplication([])

from novacore.downloader import DownloadWorker  # noqa: E402


class RangeHandler(SimpleHTTPRequestHandler):
    """支持 Range 断点续传的静态文件服务,可限速。"""
    DELAY = 0.0

    def do_GET(self):
        path = self.translate_path(self.path)
        if not os.path.isfile(path):
            self.send_error(404)
            return
        size = os.path.getsize(path)
        rng = self.headers.get("Range")
        start = 0
        if rng and rng.startswith("bytes="):
            try:
                start = int(rng[6:].split("-")[0])
                self.send_response(206)
                self.send_header("Content-Range",
                                 f"bytes {start}-{size - 1}/{size}")
                self.send_header("Content-Length", str(size - start))
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()
                with open(path, "rb") as f:
                    f.seek(start)
                    self._stream(f)
                return
            except (ValueError, OSError):
                pass
        self.send_response(200)
        self.send_header("Content-Length", str(size))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        with open(path, "rb") as f:
            self._stream(f)

    def _stream(self, f):
        try:
            while True:
                chunk = f.read(65536)
                if not chunk:
                    break
                self.wfile.write(chunk)
                if self.DELAY:
                    time.sleep(self.DELAY)
        except (ConnectionAbortedError, BrokenPipeError, ConnectionResetError):
            pass  # 客户端取消/断开是正常情况

    def log_message(self, *args):
        pass

    def handle_error(self, request, client_address):
        pass  # 客户端取消下载导致的连接中止属正常情况,不打印


class DirRangeHandler(RangeHandler):
    """按构造参数绑定目录的 Range 文件服务(可限速)。"""
    serve_dir = None
    delay = 0.0

    def __init__(self, *args, **kwargs):
        kwargs["directory"] = self.serve_dir
        super().__init__(*args, **kwargs)
        if self.delay:
            self.DELAY = self.delay

    def _stream(self, f):
        try:
            while True:
                chunk = f.read(65536)
                if not chunk:
                    break
                self.wfile.write(chunk)
                if self.DELAY:
                    time.sleep(self.DELAY)
        except (ConnectionAbortedError, BrokenPipeError, ConnectionResetError):
            pass  # 客户端取消/断开是正常情况


class LocalServer:
    def __init__(self, directory, delay=0.0):
        # delay 必须作为类属性在请求处理前生效(BaseHTTPRequestHandler 在
        # __init__ 内即完成请求处理,放到 __init__ 里设置会来不及)。
        handler = type("H", (DirRangeHandler,), {
            "serve_dir": str(directory),
            "delay": delay,
            "DELAY": delay,
        })
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}"

    def start(self):
        self.thread.start()

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def make_gguf_bytes(size_mb: int, content_marker: bytes) -> bytes:
    buf = bytearray(b"GGUF")
    buf += struct.pack("<I", 3)
    buf += struct.pack("<Q", 1)
    buf += struct.pack("<Q", 0)  # 无元数据
    while len(buf) < size_mb * 1024 * 1024:
        buf += content_marker
    return bytes(buf[:size_mb * 1024 * 1024])


def pump(seconds: float):
    end = time.time() + seconds
    while time.time() < end:
        _app.processEvents()
        time.sleep(0.005)


def run_worker(w, timeout=30.0):
    results = {}
    w.done.connect(lambda ok, msg, path: results.update(ok=ok, msg=msg, path=path))
    sources = []
    w.source.connect(lambda sid, url: sources.append(sid))
    w.start()
    end = time.time() + timeout
    while time.time() < end and "ok" not in results:
        _app.processEvents()
        time.sleep(0.01)
    w.wait(3000)
    return results, sources


class TestDownloader(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.payload = make_gguf_bytes(3, b"PAYLOAD")
        self.src = self.dir / "model.gguf"
        self.src.write_bytes(self.payload)

    def test_download_and_verify(self):
        srv = LocalServer(self.dir)
        srv.start()
        self.addCleanup(srv.stop)
        entry = {"name": "m", "sources": {"official": f"{srv.url}/model.gguf"}}
        dst = str(self.dir / "out" / "model.gguf")
        results, sources = run_worker(DownloadWorker(entry, dst))
        self.assertTrue(results["ok"], results.get("msg"))
        self.assertEqual(Path(dst).read_bytes(), self.payload)
        self.assertFalse(Path(dst + ".part").exists(), "半成品应清理/替换")
        self.assertEqual(sources[0], "official")

    def test_fallback_when_first_source_dead(self):
        srv = LocalServer(self.dir)
        srv.start()
        self.addCleanup(srv.stop)
        entry = {"name": "m", "sources": {
            "official": "http://127.0.0.1:1/dead.gguf",   # 端口 1 不可达
            "hf_mirror": f"{srv.url}/model.gguf",
        }}
        results, sources = run_worker(
            DownloadWorker(entry, str(self.dir / "m.gguf"), preference="auto"))
        self.assertTrue(results["ok"], results.get("msg"))
        # 自动模式下载前自检节点:不可达的官方源被跳过,直接用可达镜像
        self.assertEqual(sources, ["hf_mirror"],
                         "自检后应跳过不可达源,直接用可达镜像")

    def test_resume_after_cancel(self):
        """限速下载 → 中途取消 → 断点续传完成 → 内容与源一致。"""
        srv = LocalServer(self.dir, delay=0.004)
        srv.start()
        self.addCleanup(srv.stop)
        big = make_gguf_bytes(6, b"BIG")
        bigfile = self.dir / "big.gguf"
        bigfile.write_bytes(big)
        entry = {"name": "big", "sources": {"official": f"{srv.url}/big.gguf"}}
        dst = str(self.dir / "big_out.gguf")

        w1 = DownloadWorker(entry, dst)
        w1.start()
        part = dst + ".part"
        end = time.time() + 20
        while time.time() < end and (not os.path.exists(part)
                                     or os.path.getsize(part) < 512 * 1024):
            _app.processEvents()
            time.sleep(0.01)
        w1.cancel()
        w1.wait(5000)
        self.assertFalse(os.path.exists(part), "取消后半成品应被清理")

        # 重新下载(带 Range 续传能力的服务器 + 已完成部分)……part 已清理,即从头下载
        results, _ = run_worker(DownloadWorker(entry, dst))
        self.assertTrue(results["ok"], results.get("msg"))
        self.assertEqual(Path(dst).read_bytes(), big)

    def test_cancel_emits_cancelled(self):
        # 较大限速保证下载持续足够久,取消时稳定处于"进行中"状态,消除竞态
        srv = LocalServer(self.dir, delay=0.05)
        srv.start()
        self.addCleanup(srv.stop)
        entry = {"name": "m", "sources": {"official": f"{srv.url}/model.gguf"}}
        dst = str(self.dir / "m.gguf")
        total = os.path.getsize(str(self.src))
        part = dst + ".part"
        w = DownloadWorker(entry, dst)
        results = {}
        w.done.connect(lambda ok, msg, path: results.update(ok=ok, msg=msg, path=path))
        w.start()
        # 等到确实下载了一部分但尚未完成(稳定处于中途),再取消
        end = time.time() + 10
        mid_flight = False
        while time.time() < end:
            _app.processEvents()
            if os.path.exists(part):
                got = os.path.getsize(part)
                if 65536 < got < total - 65536:
                    mid_flight = True
                    break
            time.sleep(0.02)
        self.assertTrue(mid_flight, "取消前应已下载部分数据但未完成")
        w.cancel()
        w.wait(8000)
        pump(0.5)
        self.assertIs(results.get("ok"), False)
        self.assertIn("取消", results.get("msg", ""))
        self.assertFalse(os.path.exists(dst + ".part"))

    def test_html_fake_file_blocked_and_fallback(self):
        """关键场景:源返回网页而非 GGUF → 校验拦截 → 自动切下一源。"""
        bad_dir = Path(self._tmp.name) / "bad"
        bad_dir.mkdir()
        (bad_dir / "model.gguf").write_bytes(b"<!doctype html><html>Example</html>")
        srv_bad = LocalServer(bad_dir)
        srv_bad.start()
        self.addCleanup(srv_bad.stop)
        srv_good = LocalServer(self.dir)
        srv_good.start()
        self.addCleanup(srv_good.stop)
        entry = {"name": "m", "sources": {
            "official": f"{srv_bad.url}/model.gguf",
            "hf_mirror": f"{srv_good.url}/model.gguf",
        }}
        # 用固定官方源(非 auto)以稳定复现"假文件 → 校验拦截 → 切下一源"
        results, sources = run_worker(
            DownloadWorker(entry, str(self.dir / "m.gguf"), preference="official"))
        self.assertTrue(results["ok"], results.get("msg"))
        self.assertEqual(sources, ["official", "hf_mirror"])
        self.assertEqual(Path(self.dir / "m.gguf").read_bytes(), self.payload)

    def test_sha256_mismatch_fails(self):
        srv = LocalServer(self.dir)
        srv.start()
        self.addCleanup(srv.stop)
        entry = {"name": "m", "sources": {"official": f"{srv.url}/model.gguf"}}
        results, _ = run_worker(
            DownloadWorker(entry, str(self.dir / "m.gguf"),
                           sha256="0" * 64))
        self.assertFalse(results["ok"])
        self.assertIn("失败", results["msg"])

    def test_sha256_match_ok(self):
        srv = LocalServer(self.dir)
        srv.start()
        self.addCleanup(srv.stop)
        digest = hashlib.sha256(self.payload).hexdigest()
        entry = {"name": "m", "sources": {"official": f"{srv.url}/model.gguf"}}
        results, _ = run_worker(
            DownloadWorker(entry, str(self.dir / "m.gguf"), sha256=digest))
        self.assertTrue(results["ok"], results.get("msg"))

    def test_no_sources_fails_immediately(self):
        entry = {"name": "m", "sources": {}}
        results, _ = run_worker(DownloadWorker(entry, str(self.dir / "m.gguf")))
        self.assertFalse(results["ok"])
        self.assertIn("没有任何", results["msg"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
