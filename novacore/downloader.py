"""模型下载器(生产级):
- 断点续传(Range 请求,服务器不支持时自动重头下载)
- 多源自动故障转移:官方 → hf-mirror → 魔搭(按用户偏好排序)
- 实时进度/速度/剩余时间信号
- 取消协作:半成品立即清理
- 下载完成校验:SHA256(提供时)+ GGUF 魔数,校验失败自动尝试下一源
"""
from __future__ import annotations

import hashlib
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import requests
from PyQt6.QtCore import QThread, pyqtSignal

from . import gguf, modelstore

CHUNK_SIZE = 1024 * 1024


class DownloadWorker(QThread):
    progress = pyqtSignal(int, int, int)   # (百分比, 已下载字节, 总字节)
    speed = pyqtSignal(float)              # 字节/秒
    log = pyqtSignal(str)
    source = pyqtSignal(str, str)          # 当前 (源id, URL),故障转移时更新
    done = pyqtSignal(bool, str, str)      # (成功, 消息, 最终文件路径)

    def __init__(self, entry: dict, save_path: str, preference: str = "auto",
                 sha256: Optional[str] = None, parent=None):
        super().__init__(parent)
        self.entry = entry
        self.save_path = str(save_path)
        self.preference = preference
        self.sha256 = (sha256 or "").strip().lower() or None
        self._cancel_evt = threading.Event()
        self._pause_evt = threading.Event()

    def cancel(self) -> None:
        self._cancel_evt.set()

    def pause(self) -> None:
        self._pause_evt.set()

    def resume(self) -> None:
        self._pause_evt.clear()

    @property
    def is_paused(self) -> bool:
        return self._pause_evt.is_set()

    def _check_cancel(self) -> None:
        if self._cancel_evt.is_set():
            raise InterruptedError("下载已取消")

    def _wait_if_paused(self) -> None:
        while self._pause_evt.is_set():
            if self._cancel_evt.is_set():
                raise InterruptedError("下载已取消")
            time.sleep(0.1)

    def _tmp_path(self) -> str:
        return self.save_path + ".part"

    def run(self) -> None:
        tmp = self._tmp_path()
        self._cleanup(tmp)
        try:
            Path(self.save_path).parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        fallbacks = modelstore.source_fallback_list(self.entry, self.preference)
        if not fallbacks:
            self.done.emit(False, "该模型没有任何可用的下载地址", "")
            return
        # 自动模式:下载前自检各源节点,按延迟升序,优先用最快可达源,
        # 避免在校内网下先卡在官方源超时才切镜像。
        if self.preference == "auto" and len(fallbacks) > 1:
            self.log.emit("🔎 正在自检下载节点...")

            def _probe(item):
                src_id, url = item
                return src_id, url, modelstore.probe_url(url)

            with ThreadPoolExecutor(max_workers=len(fallbacks)) as pool:
                scored = list(pool.map(_probe, fallbacks))
            summary = ", ".join(
                f"{modelstore.SOURCE_NAMES.get(s, s)}="
                f"{f'{lat:.0f}ms' if lat is not None else '不可达'}"
                for s, _u, lat in scored)
            self.log.emit(f"节点自检: {summary}")
            scored.sort(key=lambda x: (x[2] is None,
                                       x[2] if x[2] is not None else float("inf")))
            fallbacks = [(s, u) for s, u, _ in scored]
        errors: list[str] = []
        for src_id, url in fallbacks:
            if self._cancel_evt.is_set():
                break
            self.source.emit(src_id, url)
            self.log.emit(f"[{modelstore.SOURCE_NAMES.get(src_id, src_id)}] 开始下载")
            try:
                self._download_one(url, tmp)
                if self._verify(tmp):
                    final = self.save_path
                    if os.path.exists(final):
                        os.remove(final)
                    os.replace(tmp, final)
                    self.done.emit(True, f"下载完成并通过校验: {final}", final)
                    return
                self.log.emit("❌ 文件校验失败(GGUF 魔数/校验和不匹配),尝试下一源...")
                errors.append("校验失败")
                self._cleanup(tmp)
            except InterruptedError:
                self._cleanup(tmp)
                self.done.emit(False, "下载已取消", "")
                return
            except Exception as e:
                self.log.emit(f"❌ 该源失败: {e}")
                errors.append(str(e))
                self._cleanup(tmp)
        self.done.emit(False, f"所有下载源均失败: {'; '.join(errors[:3])}", "")

    @staticmethod
    def _cleanup(tmp: str) -> None:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass

    def _download_one(self, url: str, tmp: str) -> None:
        resume = os.path.exists(tmp)
        start_pos = os.path.getsize(tmp) if resume else 0
        headers = {"Range": f"bytes={start_pos}-"} if resume else {}
        with requests.get(url, stream=True, headers=headers,
                          timeout=(10, 120)) as resp:
            mode = "ab" if resume else "wb"
            if resp.status_code == 200 and resume:
                # 服务器不支持 Range:重新从头下载
                resume = False
                start_pos = 0
                mode = "wb"
            elif resp.status_code == 206:
                pass
            else:
                resp.raise_for_status()
            total = start_pos
            clen = resp.headers.get("content-length")
            if clen and clen.isdigit():
                total = start_pos + int(clen)
            downloaded = start_pos
            last_emit = 0.0
            last_bytes = 0
            with open(tmp, mode) as f:
                for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                    self._check_cancel()
                    self._wait_if_paused()
                    if not chunk:
                        continue
                    f.write(chunk)
                    downloaded += len(chunk)
                    now = time.time()
                    if now - last_emit >= 0.2:
                        pct = int(downloaded / total * 100) if total else 0
                        speed = (downloaded - last_bytes) / max(now - last_emit, 0.001)
                        self.progress.emit(pct, downloaded, total)
                        self.speed.emit(speed)
                        last_emit = now
                        last_bytes = downloaded
            self._check_cancel()
            if total and downloaded < total:
                raise RuntimeError(f"文件不完整({downloaded}/{total} 字节)")

    def _verify(self, path: str) -> bool:
        if not os.path.exists(path):
            return False
        if os.path.getsize(path) == 0:
            return False
        if self.sha256:
            h = hashlib.sha256()
            with open(path, "rb") as f:
                for block in iter(lambda: f.read(CHUNK_SIZE), b""):
                    h.update(block)
            if h.hexdigest().lower() != self.sha256:
                return False
        # GGUF 魔数校验:防止把网页/错误页面当模型保存
        return gguf.is_gguf_file(path)


class OVDirectoryDownloadWorker(QThread):
    """OpenVINO IR 目录模型下载器(NPU 专区,INT4)。

    与 DownloadWorker 信号兼容,但下载物是一个模型目录:
    openvino_model.xml/.bin + config/tokenizer 等多文件。

    抗网络抖动设计(校园网/弱网关键):
    - 源顺序:国内环境 ModelScope CDN 优先,其次 hf-mirror,最后官方 HF;
    - 同源断点重试:大文件连接中途被重置/超时时,不判死,用 Range 从
      .part 断点继续(每源最多 MAX_RESUME 次,指数退避);
    - 跨源续传:某源累计重试仍失败才换源,.part 保留,新源 Range 续传;
    - 完成大小校验:以 tree API 的字节数为准,不一致继续补齐而非误判成功;
    - 仓库不存在(401/404)在创建目录之前直接报错,杜绝 0KB 空目录残件。
    """
    progress = pyqtSignal(int, int, int)   # (整体百分比, 已下载字节, 总字节)
    speed = pyqtSignal(float)
    log = pyqtSignal(str)
    source = pyqtSignal(str, str)
    done = pyqtSignal(bool, str, str)      # (成功, 消息, 最终目录路径)

    MAX_RESUME = 6          # 每个源单文件最多断点重试次数
    BACKOFF = (1, 2, 4, 6, 6, 6)  # 重试退避秒数

    def __init__(self, entry: dict, save_dir: str, preference: str = "auto",
                 parent=None):
        super().__init__(parent)
        self.entry = entry
        self.save_dir = str(save_dir)
        self.preference = preference
        self._cancel_evt = threading.Event()

    def cancel(self) -> None:
        self._cancel_evt.set()

    @property
    def is_paused(self) -> bool:
        return False

    def _check_cancel(self) -> None:
        if self._cancel_evt.is_set():
            raise InterruptedError("下载已取消")

    def _sleep(self, sec: float) -> None:
        """可被取消打断的 sleep。"""
        end = time.time() + sec
        while time.time() < end:
            self._check_cancel()
            time.sleep(0.1)

    def run(self) -> None:
        repo = str(self.entry.get("repo", "")).strip()
        if not repo:
            self.done.emit(False, "该 NPU 模型条目缺少仓库 ID", "")
            return
        ms_repo = str(self.entry.get("ms_repo") or repo).strip()

        # 1) 先解析文件清单(此时还不创建目录,仓库不存在可干净退出,不留空目录)
        self.log.emit("🔎 正在解析 OpenVINO 模型文件清单(ModelScope/HF 镜像)...")
        try:
            files = modelstore.list_ov_repo_files(
                repo, ms_repo=ms_repo, preference=self.preference)
        except modelstore.RepoNotFoundError as e:
            self.done.emit(False,
                           f"❌ {e}。该条目可能已下架,请更新模型清单或换其他模型。",
                           "")
            return
        except Exception as e:
            self.done.emit(False,
                           f"❌ 解析模型文件清单失败(网络不可达?): {e}。"
                           "可在「系统设置 → 模型下载源」自检节点后重试。",
                           "")
            return
        required = set(modelstore.OV_REQUIRED_FILES)
        if not all(any(f["path"] == n for f in files) for n in required):
            self.done.emit(False, "模型仓库缺少 openvino_model.xml/.bin", "")
            return

        # 2) 清单有效,才创建目录
        try:
            Path(self.save_dir).mkdir(parents=True, exist_ok=True)
        except OSError as e:
            self.done.emit(False, f"无法创建模型目录: {e}", "")
            return

        # 3) 源顺序(国内网络:ModelScope CDN 实测最稳,206+Accept-Ranges)
        if self.preference == "official":
            src_order = ["official", "modelscope", "hf_mirror"]
        elif self.preference == "hf_mirror":
            src_order = ["hf_mirror", "modelscope", "official"]
        else:  # auto / modelscope
            src_order = ["modelscope", "hf_mirror", "official"]

        total_bytes = sum(int(f.get("size", 0)) for f in files) or None
        # 已完成文件的字节数(断点重开时跳过的部分)
        base_done = 0
        for f in files:
            dst = os.path.join(self.save_dir, f["path"])
            if self._dst_complete(dst, int(f.get("size", 0))):
                base_done += os.path.getsize(dst)

        failed: list[str] = []
        for idx, f in enumerate(files, 1):
            self._check_cancel()
            fname = f["path"]
            expected = int(f.get("size", 0)) or None
            dst = os.path.join(self.save_dir, fname)
            if self._dst_complete(dst, expected or 0):
                self.log.emit(f"[{idx}/{len(files)}] 已存在,跳过 {fname}")
                continue

            ok = False
            last_err = ""
            for src_id in src_order:
                self._check_cancel()
                urls = modelstore.build_ov_file_sources(repo, fname, ms_repo)
                url = urls.get(src_id)
                if not url:
                    continue
                self.source.emit(src_id, url)
                try:
                    self._download_file(url, dst, fname, idx, len(files),
                                        expected, base_done, total_bytes)
                    ok = True
                    break
                except InterruptedError:
                    # 用户取消:保留 .part,下次点同一模型可继续
                    self.log.emit("已取消,半成品已保留,重新下载可断点续传")
                    self._cleanup_empty_dir()
                    self.done.emit(False, "下载已取消(进度已保留)", "")
                    return
                except _FatalSourceError as e:
                    # 401/403/404:该源无此文件,直接换源,不浪费重试
                    last_err = str(e)
                    self.log.emit(f"❌ {fname} 源 {src_id} 不可用({e}),换源...")
                except Exception as e:
                    last_err = str(e)
                    self.log.emit(
                        f"❌ {fname} 源「{modelstore.SOURCE_NAMES.get(src_id, src_id)}」"
                        f"多次断点重试仍失败: {e},换源...")
            if not ok:
                failed.append(f"{fname}({last_err[:60]})")
                continue
            size = os.path.getsize(dst)
            base_done += size
            if total_bytes:
                self.progress.emit(min(99, int(base_done / total_bytes * 100)),
                                   base_done, total_bytes)

        # 4) 必需文件校验
        missing = [n for n in required
                   if not self._dst_complete(
                       os.path.join(self.save_dir, n), 0)]
        if missing:
            self._cleanup_empty_dir()
            tip = "。可重新点击下载进行断点续传" if self._has_part() else ""
            self.done.emit(
                False,
                f"关键文件下载失败: {', '.join(missing)}"
                f"{'; '.join(failed[:2])}{tip}。"
                "若网络环境特殊,可在设置中把下载源切到 ModelScope 后重试。",
                "")
            return
        self.progress.emit(100, base_done, total_bytes or base_done)
        self.done.emit(True, f"OpenVINO NPU 模型下载完成: {self.save_dir}",
                       self.save_dir)

    @staticmethod
    def _dst_complete(dst: str, expected: int) -> bool:
        """目标文件是否已完整(非空;有清单大小时必须大小一致)。"""
        if not os.path.exists(dst) or os.path.getsize(dst) <= 0:
            return False
        return expected == 0 or os.path.getsize(dst) == expected

    def _has_part(self) -> bool:
        try:
            return any(str(n).endswith(".part")
                       for n in os.listdir(self.save_dir))
        except OSError:
            return False

    def _cleanup_empty_dir(self) -> None:
        """保守清理:只在目录完全空的时候 rmdir。

        任何带非 .part 文件的目录(包括 0 字节占位)一律不碰;
        只有 .part 文件的目录也保留(用户下次下载可断点续传)。
        这个函数的唯一目的是:防止仓库不存在时残留 0KB 纯空目录,
        绝不删除任何用户可能想保留的东西。
        """
        try:
            names = os.listdir(self.save_dir)
        except OSError:
            return
        # 目录里有任何东西(文件或子目录)都不动——哪怕只有 .part
        if not names:
            try:
                os.rmdir(self.save_dir)
            except OSError:
                pass

    def _download_file(self, url: str, dst: str, fname: str,
                       idx: int, total_files: int,
                       expected: Optional[int],
                       base_bytes: int, total_bytes: Optional[int]) -> None:
        """下载单文件,带同源 Range 断点重试 + 完成大小校验。

        成功返回;.part 保留在 dst+'.part',校验通过后 os.replace 到 dst。
        401/403/404 抛 _FatalSourceError(应直接换源);
        其他网络错误在重试耗尽后抛出(调用方换源)。
        """
        tmp = dst + ".part"
        last_t = time.time()
        last_log_t = 0.0
        last_bytes = 0

        for attempt in range(self.MAX_RESUME):
            self._check_cancel()
            # 脏数据保护:.part 比清单大小还大 → 删除重来
            if expected and os.path.exists(tmp) \
                    and os.path.getsize(tmp) > expected:
                self._safe_remove(tmp)
            start_pos = os.path.getsize(tmp) if os.path.exists(tmp) else 0
            if start_pos and attempt == 0:
                self.log.emit(
                    f"[{idx}/{total_files}] {fname} 从 "
                    f"{start_pos/1024/1024:.1f}MB 处断点续传")
            headers = {"Range": f"bytes={start_pos}-"} if start_pos else {}
            try:
                with requests.get(url, stream=True, timeout=(10, 60),
                                  headers=headers) as resp:
                    if resp.status_code in (401, 403, 404):
                        raise _FatalSourceError(f"HTTP {resp.status_code}")
                    if resp.status_code not in (200, 206):
                        raise RuntimeError(f"HTTP {resp.status_code}")
                    mode = "ab"
                    if resp.status_code == 200 and start_pos:
                        # 服务器忽略 Range:从头重下
                        start_pos = 0
                        mode = "wb"
                    with open(tmp, mode) as f:
                        last_bytes = start_pos
                        last_t = time.time()
                        for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                            self._check_cancel()
                            if not chunk:
                                continue
                            f.write(chunk)
                            now = time.time()
                            if now - last_t >= 0.2:
                                file_pos = os.path.getsize(tmp)
                                overall = base_bytes + (file_pos - start_pos)
                                if total_bytes:
                                    self.progress.emit(
                                        min(99, int(overall / total_bytes * 100)),
                                        overall, total_bytes)
                                self.speed.emit(max(
                                    0.0,
                                    (file_pos - last_bytes)
                                    / max(now - last_t, 0.001)))
                                if now - last_log_t >= 1.5:
                                    if total_bytes:
                                        self.log.emit(
                                            f"[{idx}/{total_files}] {fname} "
                                            f"{file_pos/1024/1024:.1f}MB · "
                                            f"总进度 {overall/1024/1024:.1f}/"
                                            f"{total_bytes/1024/1024:.1f}MB")
                                    else:
                                        self.log.emit(
                                            f"[{idx}/{total_files}] {fname} "
                                            f"{file_pos/1024/1024:.1f}MB")
                                    last_log_t = now
                                last_bytes, last_t = file_pos, now
                # 流正常结束:按清单大小验收
                got = os.path.getsize(tmp)
                if expected and got < expected:
                    raise RuntimeError(
                        f"连接提前结束({got/1024/1024:.1f}/"
                        f"{expected/1024/1024:.1f}MB)")
                if got <= 0:
                    raise RuntimeError("空文件(可能是错误页面)")
                os.replace(tmp, dst)
                return
            except InterruptedError:
                raise
            except _FatalSourceError:
                raise
            except Exception as e:
                if attempt == self.MAX_RESUME - 1:
                    raise RuntimeError(
                        f"重试 {self.MAX_RESUME} 次仍失败: {e}") from e
                wait = self.BACKOFF[min(attempt, len(self.BACKOFF) - 1)]
                self.log.emit(
                    f"⚠️ {fname} 连接中断({str(e)[:50]}),"
                    f"{wait}s 后断点重试 {attempt + 2}/{self.MAX_RESUME}")
                self._sleep(wait)

    @staticmethod
    def _safe_remove(path: str) -> None:
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError:
            pass


class _FatalSourceError(Exception):
    """该源对此文件确定性不可用(401/403/404),重试无意义,应直接换源。"""
