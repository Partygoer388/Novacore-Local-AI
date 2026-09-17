"""NovaCore-Local 主程序(PyQt6 GUI)。
完整接入 novacore 引擎包:
- 对话:流式生成/停止/新会话/历史持久化/文件与图片附件/速度统计
- 模型商店:真实 HuggingFace GGUF 清单 + 多源下载(断点续传/故障转移/校验)
- 本地模型:GGUF 校验扫描/导入/加载(llama.cpp)/OpenVINO 目录加载/Ollama 拉取
- 模型训练:LoRA 微调(transformers + peft)
- 系统更新/依赖管理/API 服务/主题设置
"""
from __future__ import annotations

import logging
import os
import sys
import time
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor, QFont, QIcon, QMouseEvent
from PyQt6.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QComboBox,
    QDoubleSpinBox, QFileDialog, QFormLayout, QFrame, QGroupBox,
    QHBoxLayout, QHeaderView, QInputDialog, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QMainWindow, QMessageBox, QProgressBar, QPushButton,
    QScrollArea, QSlider, QSpinBox, QSplitter, QStackedWidget, QTableWidget,
    QTableWidgetItem, QTextEdit, QVBoxLayout, QWidget,
)
from PyQt6.QtCore import QThread, pyqtSignal

from novacore import APP_NAME, APP_VERSION, gguf, hardware, modelstore, paths
from novacore import deps as deps_core
from novacore.api import ApiServer
from novacore.chat import ChatSession, ConversationStore, GenerationWorker
from novacore.config import (
    DOWNLOAD_SOURCE_CHOICES, ENGINE_CHOICES, THEME_CHOICES,
    UI_SCALE_CHOICES, load_config, save_config,
)
from novacore.downloader import DownloadWorker, OVDirectoryDownloadWorker
from novacore.engines import get_manager
from novacore.engines.ollama import find_ollama_exe
from novacore.engines.base import GenParams
from novacore.logging_setup import (
    install_excepthook, setup_logging, startup_selfcheck,
)
from novacore.trainer import TrainWorker
from novacore.workers import (
    DepInstallWorker, EngineProbeWorker, HardwareMonitorThread,
    ModelLoadWorker, WorkerRegistry,
)

SOURCE_CHOICES = DOWNLOAD_SOURCE_CHOICES

# 界面整体缩放系数(由 main() 依据配置在创建主窗口前设置)
UI_SCALE = 1.0


def _s(px: float) -> int:
    """按当前界面缩放系数换算像素尺寸,避免文字/控件被裁切。"""
    return max(1, int(round(px * UI_SCALE)))


# ==================== 通用后台线程 ====================
class SimpleWorker(QThread):
    """在线程中执行一个函数:ok 返回值,err 返回异常文本。"""
    ok = pyqtSignal(object)
    err = pyqtSignal(str)

    def __init__(self, fn: Callable, parent=None):
        super().__init__(parent)
        self._fn = fn

    def run(self) -> None:
        try:
            self.ok.emit(self._fn())
        except Exception as e:  # noqa: BLE001
            self.err.emit(str(e))


# ==================== 右下角下载弹窗 ====================
def _entry_save_path(entry: dict) -> str:
    """由条目推导默认保存路径(local_models 目录)。

    GGUF → 单个 .gguf 文件;OpenVINO IR → 同名模型目录。"""
    if str(entry.get("format", "gguf")) == "openvino_dir":
        safe = str(entry.get("name", "openvino_model")).strip() or "openvino_model"
        return str(paths.LOCAL_MODEL_DIR / safe)
    url = (entry.get("sources") or {}).get("official") or ""
    fname = Path(url).name if url else ""
    if not fname:
        fname = f"{entry.get('name', 'model')}.gguf"
    return str(paths.LOCAL_MODEL_DIR / fname)


# ==================== 悬浮球(应用内子组件) ====================
class FloatingBall(QWidget):
    """应用内右下角悬浮球:点开显示下载任务 + 硬件占用摘要。
    替代旧的外部 DownloadToast 弹窗,不会被任务栏截断。"""

    BALL_SIZE = 44
    PANEL_W = 320
    PANEL_H = 260

    def __init__(self, parent_win):
        super().__init__(parent_win)
        self.win = parent_win
        self._download_rows: list[QWidget] = []
        self._expanded = False

        # 悬浮球本体
        self.setFixedSize(_s(self.BALL_SIZE), _s(self.BALL_SIZE))
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._ball_style = (
            "QWidget{background:#2563eb; border-radius:22px;}"
            "QWidget:hover{background:#3b82f6;}")
        self.setStyleSheet(self._ball_style)

        self._badge_lab = QLabel("⬇", self)
        self._badge_lab.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._badge_lab.setStyleSheet(
            "color:white; font-size:16px; font-weight:700; background:transparent;")
        self._badge_lab.setGeometry(0, 0, _s(self.BALL_SIZE), _s(self.BALL_SIZE))

        self._count_lab = QLabel("", self)
        self._count_lab.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._count_lab.setStyleSheet(
            "color:white; font-size:10px; background:#ef4444; border-radius:8px;")
        self._count_lab.setFixedSize(16, 16)
        self._count_lab.move(_s(self.BALL_SIZE) - 16, 0)
        self._count_lab.hide()

        # 展开面板(初始隐藏)
        self._panel = QFrame(parent_win)
        self._panel.setObjectName("floatPanel")
        self._panel.setFixedWidth(_s(self.PANEL_W))
        self._panel.setStyleSheet(
            "QFrame#floatPanel{background:#1e2028; border:1px solid #3a3f47;"
            "border-radius:10px;}")
        pl = QVBoxLayout(self._panel)
        pl.setContentsMargins(12, 10, 12, 10)
        pl.setSpacing(6)

        head = QHBoxLayout()
        head.addWidget(QLabel("📥 下载任务"))
        head.addStretch()
        self._btn_panel_close = QPushButton("✕")
        self._btn_panel_close.setFixedSize(22, 22)
        self._btn_panel_close.setStyleSheet(
            "QPushButton{border:none; background:transparent; color:#889; font-size:13px;}"
            "QPushButton:hover{color:#ef4444;}")
        self._btn_panel_close.clicked.connect(self._toggle)
        head.addWidget(self._btn_panel_close)
        pl.addLayout(head)

        self._dl_list = QVBoxLayout()
        self._dl_list.setSpacing(4)
        pl.addLayout(self._dl_list, stretch=1)

        self._dl_empty = QLabel("暂无下载任务")
        self._dl_empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._dl_empty.setStyleSheet("color:#667; font-size:12px;")
        self._dl_empty.setFixedHeight(40)
        self._dl_list.addWidget(self._dl_empty)

        pl.addSpacing(4)
        pl.addWidget(QLabel("📊 硬件占用"))
        self._hw_lab = QLabel("💻 RAM -- GB  ·  CPU --%\n🎮 GPU --%  VRAM -- GB")
        self._hw_lab.setStyleSheet("color:#889; font-size:11px;")
        self._hw_lab.setWordWrap(True)
        pl.addWidget(self._hw_lab)

        self._panel.hide()

    # ---------- 基础交互 ----------
    def mousePressEvent(self, e):
        self._toggle()
        super().mousePressEvent(e)

    def _toggle(self) -> None:
        if self._expanded:
            self._panel.hide()
            self._expanded = False
        else:
            self._reposition_panel()
            self._panel.show()
            self._panel.raise_()
            self._expanded = True

    def _reposition(self) -> None:
        """父窗口 resize 时重新定位到右下角。"""
        geo = self.win.rect()
        x = geo.right() - _s(self.BALL_SIZE) - 12
        y = geo.bottom() - _s(self.BALL_SIZE) - 12
        self.move(x, y)
        if self._expanded:
            self._reposition_panel()

    def _reposition_panel(self) -> None:
        """面板定位到悬浮球上方。"""
        bx, by = self.x(), self.y()
        pw, ph = _s(self.PANEL_W), _s(self.PANEL_H)
        px = bx - pw + _s(self.BALL_SIZE)
        py = by - ph - 6
        # 防止超出左边界
        px = max(6, px)
        self._panel.setGeometry(px, py, pw, ph)

    # ---------- 下载任务管理 ----------
    def add_download(self, entry: dict, worker) -> QWidget:
        """新增一条下载任务行,返回行控件供外部更新。
        worker 可为 DownloadWorker(GGUF)或 OVDirectoryDownloadWorker(IR 目录),
        两者信号签名一致。"""
        row = QFrame()
        row.setStyleSheet("QFrame{background:#2b2e36; border-radius:6px;}")
        rl = QVBoxLayout(row)
        rl.setContentsMargins(10, 6, 10, 6)
        rl.setSpacing(3)

        fmt_tag = "📦 OpenVINO IR·NPU" if str(
            entry.get("format", "gguf")) == "openvino_dir" else "📦 GGUF"
        name_lab = QLabel(f"📥 {entry.get('name', '下载中')}")
        name_lab.setStyleSheet("font-size:12px; font-weight:500;")
        name_lab.setToolTip(fmt_tag)
        rl.addWidget(name_lab)
        fmt_lab = QLabel(fmt_tag)
        fmt_lab.setStyleSheet("color:#7aa; font-size:10px;")
        rl.addWidget(fmt_lab)

        prog = QProgressBar()
        prog.setRange(0, 100)
        prog.setFixedHeight(8)
        prog.setTextVisible(False)
        prog.setStyleSheet(
            "QProgressBar{background:#23262d; border:none; border-radius:4px;}"
            "QProgressBar::chunk{background:#2563eb; border-radius:4px;}")
        rl.addWidget(prog)

        status = QLabel("⏳ 连接下载源...")
        status.setStyleSheet("color:#889; font-size:11px;")
        status.setWordWrap(True)
        rl.addWidget(status)

        self._dl_list.insertWidget(0, row)
        self._download_rows.append(row)
        self._dl_empty.hide()
        self._update_count()

        # 绑定信号(两种 worker 信号签名一致)
        def _on_progress(p, d, t):
            prog.setValue(p)
            if t > 0:
                status.setText(
                    f"{p}%  ·  {d/1024/1024:.1f}/{t/1024/1024:.1f} MB")

        worker.progress.connect(_on_progress)
        worker.speed.connect(
            lambda s: status.setText(
                f"{prog.value()}%  ·  {s/1024/1024:.2f} MB/s"))
        worker.source.connect(
            lambda sid, url: status.setText(
                f"下载源:{modelstore.SOURCE_NAMES.get(sid, sid)}"))
        # OV 目录下载器的逐文件日志(截断防止撑爆面板)
        if hasattr(worker, "log"):
            worker.log.connect(
                lambda m: status.setText(m[:60]))
        worker.done.connect(
            lambda ok, msg, path: self._on_download_done(
                row, prog, status, ok, msg))

        return row

    def _on_download_done(self, row, prog, status, ok, msg) -> None:
        prog.setValue(100 if ok else prog.value())
        status.setText("✅ 完成,正在刷新本地模型列表..." if ok else f"❌ {msg[:60]}")
        # 下载成功:通知主窗口刷新本地模型列表
        if ok and hasattr(self.win, "scan_local_models_gui"):
            try:
                self.win.scan_local_models_gui()
            except Exception:
                pass
        # 5 秒后自动移除(失败也给足阅读时间)
        QTimer.singleShot(5000, lambda: self._remove_row(row))

    def _remove_row(self, row) -> None:
        if row in self._download_rows:
            self._download_rows.remove(row)
        row.setParent(None)
        row.deleteLater()
        self._update_count()
        if not self._download_rows:
            self._dl_empty.show()

    def _update_count(self) -> None:
        n = len(self._download_rows)
        if n > 0:
            self._count_lab.setText(str(n))
            self._count_lab.show()
        else:
            self._count_lab.hide()

    def update_hw(self, stat: dict) -> None:
        """从 MainWindow 的硬件监控线程更新摘要显示。"""
        txt = f"💻 RAM {stat.get('ram_used', 0):.1f}/{stat.get('ram_total', 0):.0f}GB  ·  CPU {stat.get('cpu', 0):.0f}%"
        if "gpu_util" in stat:
            txt += f"\n🎮 GPU {stat.get('gpu_util', 0)}%  VRAM {stat.get('gpu_vram_used', 0):.1f}/{stat.get('gpu_vram_total', 0):.1f}GB"
        self._hw_lab.setText(txt)

    def set_enabled(self, enabled: bool) -> None:
        """开关显示。"""
        self.setVisible(enabled)
        if not enabled:
            self._panel.hide()
            self._expanded = False


class DownloadToast(QFrame):
    """右下角非阻塞下载弹窗:实时进度 + 暂停/继续 + 取消。
    替代旧的大窗口模态对话框,不遮挡主界面。"""

    WIDTH = 360

    def __init__(self, entry: dict, parent_win, save_path: Optional[str] = None):
        super().__init__(None)  # 顶层无父窗口
        self.entry = entry
        self.win = parent_win
        self.save_path = save_path or _entry_save_path(entry)
        self.worker: Optional[DownloadWorker] = None
        self._finished = False

        self.setWindowFlags(
            Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint)
        self.setFixedWidth(_s(self.WIDTH))

        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 12, 14, 12)
        lay.setSpacing(8)

        head = QHBoxLayout()
        title = QLabel(f"📥 {entry.get('name', '下载模型')}")
        title.setStyleSheet("font-weight:600;")
        self.btn_pause = QPushButton("⏸ 暂停")
        self.btn_pause.setFixedWidth(_s(72))
        self.btn_cancel = QPushButton("✕")
        self.btn_cancel.setFixedWidth(_s(30))
        head.addWidget(title, stretch=1)
        head.addWidget(self.btn_pause)
        head.addWidget(self.btn_cancel)
        lay.addLayout(head)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        lay.addWidget(self.progress)

        self.status_lab = QLabel("⏳ 连接下载源...")
        self.status_lab.setStyleSheet("color:#99a;")
        self.status_lab.setWordWrap(True)
        lay.addWidget(self.status_lab)

        self.setStyleSheet("""
QFrame {{ background:#23262d; border:1px solid #3a3f47; border-radius:10px; }}
QPushButton {{ background:#363942; border:none; border-radius:6px;
    padding:5px 10px; color:#eee; }}
QPushButton:hover {{ background:#444854; }}
QProgressBar {{ background:#2b2e36; border:none; border-radius:6px;
    text-align:center; color:#eee; }}
QProgressBar::chunk {{ background:#2563eb; border-radius:5px; }}
        """)

        self.btn_pause.clicked.connect(self._toggle_pause)
        self.btn_cancel.clicked.connect(self._cancel)

    # ---------- 下载控制 ----------
    def start_download(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            return
        worker = DownloadWorker(self.entry, self.save_path,
                                preference=str(cfg.get("download_source", "auto")))
        worker.progress.connect(self._on_progress)
        worker.speed.connect(self._on_speed)
        worker.source.connect(self._on_source)
        worker.log.connect(self._on_log)
        worker.done.connect(self._on_done)
        self.worker = worker
        registry.register(worker)
        worker.start()

    def _toggle_pause(self) -> None:
        if self.worker is None or not self.worker.isRunning():
            return
        if self.worker.is_paused:
            self.worker.resume()
            self.btn_pause.setText("⏸ 暂停")
            self.status_lab.setText(f"{self.progress.value()}% 已继续")
        else:
            self.worker.pause()
            self.btn_pause.setText("▶ 继续")
            self.status_lab.setText(f"{self.progress.value()}% 已暂停")

    def _cancel(self) -> None:
        if self._finished:
            self.close()
            return
        if self.worker is not None:
            self.worker.cancel()
        self.status_lab.setText("正在取消...")

    # ---------- 信号槽 ----------
    def _on_progress(self, pct: int, downloaded: int, total: int) -> None:
        self.progress.setValue(pct)
        self.status_lab.setText(
            f"{pct}%  ({downloaded / 1024 / 1024:.1f} / "
            f"{max(total / 1024 / 1024, 0.1):.1f} MB)")

    def _on_speed(self, speed: float) -> None:
        if self.worker is not None and self.worker.is_paused:
            return
        self.status_lab.setText(
            f"{self.progress.value()}%  ·  {speed / 1024 / 1024:.2f} MB/s")

    def _on_source(self, src_id: str, url: str) -> None:
        name = modelstore.SOURCE_NAMES.get(src_id, src_id)
        self.status_lab.setText(f"下载源:{name}")

    def _on_log(self, text: str) -> None:
        self.status_lab.setText(text)

    def _on_done(self, ok: bool, msg: str, path: str) -> None:
        self._finished = True
        self.btn_pause.hide()
        self.btn_cancel.setText("关闭")
        if ok:
            self.progress.setValue(100)
            self.status_lab.setText("✅ 下载完成")
            if hasattr(self.win, "scan_local_models_gui"):
                self.win.scan_local_models_gui()
        else:
            self.status_lab.setText(f"❌ {msg}")

    def closeEvent(self, event) -> None:  # noqa: N802
        if self.worker is not None and self.worker.isRunning():
            self.worker.cancel()
        if hasattr(self.win, "_unregister_toast"):
            self.win._unregister_toast(self)
        super().closeEvent(event)


# ==================== 商店行控件 ====================
class StoreRow(QWidget):
    def __init__(self, model: dict, parent_win, hw_filter: str = "全部"):
        super().__init__()
        self.model = model
        self.win = parent_win
        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 8)

        # 根据当前硬件筛选上下文决定显示哪些标签
        mhw = model.get("hw") or []
        if hw_filter == "NPU专区":
            hw_display = ["NPU"]  # NPU专区只标 NPU,不混 CPU/GPU
        elif hw_filter == "CPU/GPU":
            hw_display = [h for h in mhw if h in ("CPU", "GPU")] or ["CPU"]
        else:
            hw_display = mhw if mhw else ["CPU"]
        hw_str = "/".join(hw_display)

        # GGUF 格式的格式提示
        is_npu_model = "NPU" in mhw
        fmt_hint = ""
        if is_npu_model and hw_filter == "NPU专区":
            fmt_hint = "  [GGUF·OpenVINO加载]"
        elif is_npu_model and hw_filter == "CPU/GPU":
            fmt_hint = "  [GGUF·llama.cpp加载]"

        name_lab = QLabel(f"{model['name']}")
        name_lab.setStyleSheet("font-weight:600;")
        info_lab = QLabel(
            f"  |  {hw_str}  |  {model.get('category', '')}  |  {model.get('size_gb', 0)} GB"
            f"{fmt_hint}\n  {model.get('desc', '')}")
        info_lab.setStyleSheet("color:#889;")
        info_lab.setWordWrap(True)

        text_col = QVBoxLayout()
        text_col.setSpacing(2)
        text_col.addWidget(name_lab)
        text_col.addWidget(info_lab)
        text_box = QWidget()
        text_box.setLayout(text_col)

        btn = QPushButton("下载")
        btn.setFixedWidth(_s(90))
        btn.clicked.connect(self.open_download)
        lay.addWidget(text_box, stretch=8)
        lay.addWidget(btn, stretch=1)

    def open_download(self) -> None:
        self.win.start_via_ball(self.model)


# ==================== 本地模型行控件 ====================
class HistoryRow(QWidget):
    """对话历史列表中的一行:标题(自动省略)+ 副信息(时间+条数) + 悬停显示的删除按钮。"""

    def __init__(self, summary: dict, parent_win, active: bool = False):
        super().__init__()
        self.sid = summary["id"]
        self.win = parent_win

        outer = QHBoxLayout(self)
        outer.setContentsMargins(10, 6, 8, 6)
        outer.setSpacing(6)

        # 左侧:标题+副信息(纵向堆叠)
        text_lay = QVBoxLayout()
        text_lay.setSpacing(1)
        text_lay.setContentsMargins(0, 0, 0, 0)

        title = str(summary.get("title") or "新对话")
        self.lab_title = QLabel(title)
        self.lab_title.setStyleSheet("font-size:13px; font-weight:500;")
        self.lab_title.setWordWrap(False)
        self.lab_title.setTextInteractionFlags(
            Qt.TextInteractionFlag.NoTextInteraction)
        text_lay.addWidget(self.lab_title)

        ts = summary.get("updated_at", 0)
        try:
            when = datetime.fromtimestamp(float(ts)).strftime("%m-%d %H:%M")
        except Exception:
            when = ""
        cnt = summary.get("count", 0)
        self.lab_sub = QLabel(f"🕐 {when or '--'}  ·  💬 {cnt} 条")
        self.lab_sub.setStyleSheet("color:#889; font-size:11px;")
        text_lay.addWidget(self.lab_sub)

        title_box = QWidget()
        title_box.setLayout(text_lay)
        outer.addWidget(title_box, stretch=1)

        # 右侧按钮组:改名 + 删除(默认隐藏,悬停时显示)
        self.btn_rename = QPushButton("✎")
        self.btn_rename.setFixedSize(22, 22)
        self.btn_rename.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_rename.setToolTip("重命名这条对话")
        self.btn_rename.setStyleSheet(
            "QPushButton{border:none; background:transparent; color:#889; font-size:13px;}"
            "QPushButton:hover{background:#2563eb; color:#fff; border-radius:4px;}")
        self.btn_rename.clicked.connect(self._rename)
        self.btn_rename.hide()
        outer.addWidget(self.btn_rename)

        self.btn_del = QPushButton("✕")
        self.btn_del.setFixedSize(22, 22)
        self.btn_del.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_del.setToolTip("删除这条对话记录")
        self.btn_del.setStyleSheet(
            "QPushButton{border:none; background:transparent; color:#889; font-size:13px;}"
            "QPushButton:hover{background:#ef4444; color:#fff; border-radius:4px;}")
        self.btn_del.clicked.connect(self._delete)
        self.btn_del.hide()
        outer.addWidget(self.btn_del)

        # 点击整行切换对话(用 mousePressEvent)
        self._title_box = title_box
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        if active:
            self.setObjectName("historyRowActive")

    # 鼠标悬停时显示/隐藏按钮组
    def enterEvent(self, e):
        self.btn_rename.show()
        self.btn_del.show()
        super().enterEvent(e)

    def leaveEvent(self, e):
        self.btn_rename.hide()
        self.btn_del.hide()
        super().leaveEvent(e)

    def mousePressEvent(self, e):
        # 双击触发改名
        if e.type() == QMouseEvent.Type.MouseButtonDblClick:
            self._rename()
            return
        self._open()
        super().mousePressEvent(e)

    def mouseDoubleClickEvent(self, e):
        self._rename()
        super().mouseDoubleClickEvent(e)

    def _open(self) -> None:
        self.win.on_load_session(self.sid)

    def _delete(self) -> None:
        self.win.on_delete_session(self.sid)

    def _rename(self) -> None:
        cur = self.lab_title.text()
        text, ok = QInputDialog.getText(
            self, "重命名对话", "新名称:", text=cur)
        if ok and text.strip():
            self.win.on_rename_session(self.sid, text.strip())


class LocalModelRow(QWidget):
    def __init__(self, item: dict, parent_win):
        super().__init__()
        self.item = item
        self.win = parent_win
        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 8)

        name = item["name"]
        size_kb = item.get("size_kb", 0)
        if size_kb >= 1024 * 1024:
            size_txt = f"{size_kb / 1024 / 1024:.2f} GB"
        elif size_kb >= 1024:
            size_txt = f"{size_kb / 1024:.1f} MB"
        else:
            size_txt = f"{size_kb:.0f} KB"
        info = item.get("info") or {}
        arch = info.get("architecture", "")
        ctx = info.get("context_length")

        if item.get("valid"):
            kind, kind_color = "✅ 有效 GGUF", "#10b981"
            extra = f"  |  {arch}" + (f" / ctx {ctx}" if ctx else "")
        elif item.get("reason") == gguf.DIR_HINT and self._is_openvino_dir():
            kind, kind_color = "✅ OpenVINO 模型目录", "#10b981"
            extra = ""
        else:
            kind, kind_color = f"❌ {item.get('reason', '无效文件')}", "#ef4444"
            extra = ""

        lab = QLabel(f"{name}  |  {size_txt}  |  {kind}{extra}")
        lab.setWordWrap(True)
        lab.setStyleSheet(f"color:{kind_color};")
        lay.addWidget(lab, stretch=7)

        # 当前运行状态徽章(启动中/运行中/已停止)
        self.badge = QLabel("💤 已停止")
        self.badge.setStyleSheet("color:#889;")
        self.badge.setFixedWidth(_s(96))
        self.badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self.badge)

        loadable = item.get("valid") or (kind.startswith("✅ OpenVINO"))
        self.btn_load = None
        self.btn_close = None
        if loadable:
            self.btn_load = QPushButton("加载模型")
            self.btn_load.setFixedWidth(_s(100))
            self.btn_load.clicked.connect(self.load_model)
            lay.addWidget(self.btn_load)
            self.btn_close = QPushButton("关闭模型")
            self.btn_close.setFixedWidth(_s(100))
            self.btn_close.clicked.connect(self.close_model)
            self.btn_close.setVisible(False)
            lay.addWidget(self.btn_close)
        btn_del = QPushButton("删除")
        btn_del.setFixedWidth(_s(70))
        btn_del.clicked.connect(self.delete_file)
        lay.addWidget(btn_del)

    def _is_openvino_dir(self) -> bool:
        p = Path(self.item.get("path", ""))
        if not p.is_dir():
            return False
        try:
            return any(f.suffix == ".xml" for f in p.iterdir())
        except OSError:
            return False

    def set_status(self, text: str, color: str, running: bool) -> None:
        """更新运行状态徽章与 加载/关闭 按钮显隐。"""
        self.badge.setText(text)
        self.badge.setStyleSheet(f"color:{color};")
        if self.btn_load is not None:
            self.btn_load.setVisible(not running)
        if self.btn_close is not None:
            self.btn_close.setVisible(running)

    def load_model(self) -> None:
        self.win.request_load_model(self.item["path"], self.item["name"])

    def close_model(self) -> None:
        self.win.on_unload_model()

    def delete_file(self) -> None:
        name = self.item["name"]
        reply = QMessageBox.question(
            self.win, "确认删除",
            f"确定删除模型文件(夹) {name} 吗?此操作不可恢复。")
        if reply != QMessageBox.StandardButton.Yes:
            return
        try:
            p = paths.resolve_model_path(name)
        except ValueError as e:
            QMessageBox.warning(self.win, "错误", str(e))
            return
        import shutil
        try:
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
            elif p.exists():
                p.unlink()
        except OSError as e:
            QMessageBox.warning(self.win, "删除失败", str(e))
            return
        self.win.scan_local_models_gui()


# ==================== 主窗口 ====================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} {APP_VERSION}")
        self.resize(_s(1320), _s(820))
        self.setMinimumSize(_s(1100), _s(680))

        # 运行状态
        self._loading = False
        self._generating = False
        self._pending_image: Optional[str] = None
        self._closing = False
        self._local_models: list[dict] = []
        self._model_rows: dict[str, LocalModelRow] = {}
        self._loaded_model_name: str = ""

        # 悬浮球
        self.floating_ball = FloatingBall(self)

        # 引擎与会话
        self.chat_session = ChatSession(cfg, history_file=paths.HISTORY_FILE)
        self.store_data = modelstore.get_builtin_store()
        self.api_server = ApiServer(engine_mgr, cfg)

        self.stack = QStackedWidget()
        self.nav_btns: list[QPushButton] = []
        self.build_nav()
        self.build_pages()
        self.bind_all_signals()
        self.apply_theme()

        # 恢复上次会话:填充历史列表,若上次会话有消息则回显
        self.refresh_history_list()
        if self.chat_session.messages:
            self._render_chat_messages()

        central = QWidget()
        main_lay = QHBoxLayout(central)
        main_lay.setContentsMargins(0, 0, 0, 0)
        main_lay.setSpacing(0)

        nav_box = QWidget()
        nav_box.setFixedWidth(_s(220))
        nav_box.setStyleSheet("background:#1e1f24;")
        nav_lay = QVBoxLayout(nav_box)
        nav_lay.setContentsMargins(0, 20, 0, 20)
        nav_lay.setSpacing(4)
        logo = QLabel(f"✨ {APP_NAME}")
        logo.setStyleSheet(
            "font-size:20px; font-weight:700; color:#e8ecff; padding:0 24px 20px;")
        nav_lay.addWidget(logo)
        for b in self.nav_btns:
            nav_lay.addWidget(b)
        nav_lay.addStretch()
        ver_lab = QLabel(APP_VERSION)
        ver_lab.setStyleSheet("color:#667; padding-left:28px;")
        nav_lay.addWidget(ver_lab)
        main_lay.addWidget(nav_box)
        main_lay.addWidget(self.stack)
        self.setCentralWidget(central)

        # 后台线程
        self._npu_info = hardware.detect_npu()  # 探测一次,供左下角硬件监控显示
        self.hw_thread = HardwareMonitorThread()
        self.hw_thread.stat.connect(self.on_hw_stat)
        registry.register(self.hw_thread)
        if cfg.get("hardware_monitor", True):
            self.hw_thread.start()

        self.probe_worker = EngineProbeWorker(engine_mgr)
        self.probe_worker.result.connect(self.on_probe_done)
        registry.register(self.probe_worker)
        self.probe_worker.start()

        # 启动任务
        QTimer.singleShot(200, self.scan_local_models_gui)
        QTimer.singleShot(300, self.refresh_store_list)
        QTimer.singleShot(400, self.update_loaded_model)
        if cfg.get("api_enable", False):
            QTimer.singleShot(600, self._auto_start_api)
        # 勾选了「自动启动 Ollama 服务」:程序打开即后台拉起(不阻塞界面)
        if cfg.get("ollama_auto_start", False):
            QTimer.singleShot(1200, self._startup_ollama_auto)

        # 悬浮球:立即初始化(不依赖 showEvent,确保可见)
        QTimer.singleShot(10, self._setup_floating_ball)

    # ==================== 导航 ====================
    def build_nav(self) -> None:
        items = [
            ("💬  对话面板", 0),
            ("📦  模型商店", 1),
            ("💾  本地模型", 2),
            ("🎯  模型训练", 3),
            ("🔄  系统更新", 4),
            ("🔧  依赖管理", 5),
            ("⚙️  系统设置", 6),
        ]
        for txt, idx in items:
            b = QPushButton(txt)
            b.setCheckable(True)
            b.setFixedHeight(_s(46))
            b.setStyleSheet(f"""
                QPushButton {{
                    text-align:left; padding-left:{_s(28)}px; border:none; border-radius:0;
                    margin:2px {_s(14)}px; background:transparent; color:#aab; font-size:{_s(14)}px;
                }}
                QPushButton:hover {{ background:#292b33; color:#dde; }}
                QPushButton:checked {{ background:#2563eb; color:#fff; }}
            """)
            b.clicked.connect(lambda _, i=idx: self.switch_page(i))
            self.nav_btns.append(b)

    def switch_page(self, index: int) -> None:
        for i, btn in enumerate(self.nav_btns):
            btn.setChecked(i == index)
        self.stack.setCurrentIndex(index)

    # ==================== 页面构建 ====================
    def build_pages(self) -> None:
        self._build_chat_page()
        self._build_store_page()
        self._build_local_page()
        self._build_train_page()
        self._build_update_page()
        self._build_deps_page()
        self._build_settings_page()

    # ----- 0 对话页 -----
    def _build_chat_page(self) -> None:
        p0 = QWidget()
        outer = QHBoxLayout(p0)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ===== 左侧:对话历史面板 =====
        hist = QFrame()
        hist.setObjectName("chatHistory")
        hist.setMinimumWidth(_s(200))
        hist.setMaximumWidth(_s(400))
        hl = QVBoxLayout(hist)
        hl.setContentsMargins(10, 12, 10, 12)
        hl.setSpacing(8)
        hist_title = QLabel("📜 对话历史")
        hist_title.setStyleSheet("font-size:14px; font-weight:600;")
        hl.addWidget(hist_title)
        btn_new_chat = QPushButton("🆕 新建对话")
        btn_new_chat.clicked.connect(self.on_new_chat)
        hl.addWidget(btn_new_chat)
        self.hist_list = QListWidget()
        self.hist_list.setFrameShape(QFrame.Shape.NoFrame)
        self.hist_list.setStyleSheet(
            "QListWidget::item{border-radius:6px; margin:2px 4px;}"
            "QListWidget::item:hover{background:rgba(255,255,255,0.06);}"
            "QListWidget::item:selected{background:rgba(37,99,235,0.2);}")
        hl.addWidget(self.hist_list, stretch=1)
        btn_clear_all = QPushButton("🗑 清空全部历史")
        btn_clear_all.setStyleSheet("color:#c66;")
        btn_clear_all.clicked.connect(self.on_clear_all_sessions)
        hl.addWidget(btn_clear_all)
        # 左下角:硬件监控(独立一行,避免与按钮挤在一起)
        hl.addSpacing(8)
        self.stat_lab = QLabel("💻 RAM: -- GB  |  CPU: --%")
        self.stat_lab.setStyleSheet("color:#99a; font-size:11px;")
        self.stat_lab.setWordWrap(True)
        hl.addWidget(self.stat_lab)

        # ===== 右侧:对话区 =====
        right = QWidget()
        l0 = QVBoxLayout(right)
        l0.setContentsMargins(24, 20, 24, 20)
        l0.setSpacing(14)

        top_bar = QFrame()
        top_bar.setObjectName("chatTop")
        top_lay = QHBoxLayout(top_bar)
        top_lay.setContentsMargins(14, 10, 14, 10)
        top_lay.addWidget(QLabel("推理设备:"))
        self.device_sel = QComboBox()
        self.device_sel.addItems(["CPU", "GPU", "NPU-OpenVINO"])
        self.device_sel.setCurrentText(str(cfg.get("prefer_device", "CPU")))
        top_lay.addWidget(self.device_sel)
        top_lay.addSpacing(12)
        top_lay.addWidget(QLabel("选择模型:"))
        self.model_sel = QComboBox()
        self.model_sel.setMinimumWidth(240)
        top_lay.addWidget(self.model_sel)
        btn_load = QPushButton("加载")
        btn_load.setFixedWidth(_s(70))
        btn_load.clicked.connect(self.on_load_selected_model)
        self.btn_chat_load = btn_load
        top_lay.addWidget(btn_load)
        btn_unload = QPushButton("停止")
        btn_unload.setFixedWidth(_s(70))
        btn_unload.setToolTip("停止当前模型并释放显存/内存(不会删除模型文件)")
        btn_unload.clicked.connect(self.on_unload_model)
        top_lay.addWidget(btn_unload)
        top_lay.addStretch()
        l0.addWidget(top_bar)

        self.chat_box = QTextEdit()
        self.chat_box.setReadOnly(True)
        self.chat_box.append(f"=== {APP_NAME} {APP_VERSION} 本地AI引擎 就绪 ===")
        self.chat_box.append("请在「本地模型」页或上方下拉框加载模型后开始对话;"
                             "API 服务可在「系统设置」页开启。\n")
        l0.addWidget(self.chat_box)

        # 底部状态栏:引擎状态(左) + 推理速度(右)
        bottom_bar = QFrame()
        bottom_bar.setObjectName("chatBottom")
        bot_lay = QHBoxLayout(bottom_bar)
        bot_lay.setContentsMargins(6, 2, 6, 2)
        self.engine_lab = QLabel("🔍 引擎探测中...")
        self.engine_lab.setStyleSheet("color:#889; font-size:11px;")
        bot_lay.addWidget(self.engine_lab)
        bot_lay.addStretch()
        self.speed_lab = QLabel("--")
        self.speed_lab.setStyleSheet("color:#7aa; font-size:11px;")
        bot_lay.addWidget(self.speed_lab)
        l0.addWidget(bottom_bar)

        input_frame = QFrame()
        input_frame.setObjectName("chatInput")
        input_lay = QVBoxLayout(input_frame)
        input_lay.setContentsMargins(14, 12, 14, 12)
        self.msg_input = QLineEdit()
        self.msg_input.setPlaceholderText("输入消息,按回车发送...")
        input_lay.addWidget(self.msg_input)
        btn_row = QHBoxLayout()
        btn_send = QPushButton("发送")
        btn_send.setFixedWidth(_s(90))
        btn_stop = QPushButton("⏹ 停止")
        btn_file = QPushButton("📎 文件")
        btn_img = QPushButton("🖼️ 图片")
        btn_new = QPushButton("🆕 新会话")
        btn_row.addWidget(btn_send)
        btn_row.addWidget(btn_stop)
        btn_row.addWidget(btn_file)
        btn_row.addWidget(btn_img)
        btn_row.addWidget(btn_new)
        btn_row.addStretch()
        input_lay.addLayout(btn_row)
        l0.addWidget(input_frame)

        # ===== 用 QSplitter 可拖拽分割 =====
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(hist)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)  # 左侧历史栏不自动伸缩
        splitter.setStretchFactor(1, 1)  # 右侧对话区占满剩余空间
        splitter.setHandleWidth(4)
        # 默认比例:左侧 260px,右侧自适应
        splitter.setSizes([_s(260), 800])
        splitter.setChildrenCollapsible(False)
        outer.addWidget(splitter)
        self.stack.addWidget(p0)

        self.btn_send = btn_send
        self.btn_stop = btn_stop
        self.btn_stop.setEnabled(False)
        btn_send.clicked.connect(self.on_send)
        btn_stop.clicked.connect(self.on_stop_generation)
        btn_file.clicked.connect(self.on_attach_file)
        btn_img.clicked.connect(self.on_attach_image)
        btn_new.clicked.connect(self.on_new_chat)
        self.msg_input.returnPressed.connect(self.on_send)

    # ----- 1 模型商店 -----
    def _build_store_page(self) -> None:
        p1 = QWidget()
        l1 = QVBoxLayout(p1)
        l1.setContentsMargins(24, 20, 24, 20)
        l1.setSpacing(14)

        title1 = QLabel("📦 模型商店")
        title1.setStyleSheet("font-size:18px; font-weight:600;")
        l1.addWidget(title1)

        filter_bar = QFrame()
        filter_lay = QHBoxLayout(filter_bar)
        filter_lay.setContentsMargins(12, 10, 12, 10)
        filter_lay.addWidget(QLabel("下载源:"))
        self.download_src = QComboBox()
        self.download_src.addItems(SOURCE_CHOICES)
        cur_src = str(cfg.get("download_source", "auto"))
        if cur_src in SOURCE_CHOICES:
            self.download_src.setCurrentIndex(SOURCE_CHOICES.index(cur_src))
        self.download_src.currentIndexChanged.connect(self.on_download_src_changed)
        filter_lay.addWidget(self.download_src)
        self.btn_probe_nodes = QPushButton("🔍 自检节点")
        self.btn_probe_nodes.clicked.connect(self.on_probe_download_nodes)
        filter_lay.addWidget(self.btn_probe_nodes)
        filter_lay.addSpacing(14)
        filter_lay.addWidget(QLabel("硬件筛选:"))
        self.filter_hw = QComboBox()
        self.filter_hw.addItems(["全部", "CPU/GPU", "NPU专区"])
        filter_lay.addWidget(self.filter_hw)
        filter_lay.addSpacing(14)
        filter_lay.addWidget(QLabel("类型筛选:"))
        self.filter_type = QComboBox()
        self.filter_type.addItems(["全部", "文本", "编程", "多模态", "配音"])
        filter_lay.addWidget(self.filter_type)
        filter_lay.addStretch()
        btn_refresh_online = QPushButton("🔄 刷新在线列表")
        filter_lay.addWidget(btn_refresh_online)
        self.btn_refresh_online = btn_refresh_online
        l1.addWidget(filter_bar)

        search_bar = QFrame()
        search_lay = QHBoxLayout(search_bar)
        search_lay.setContentsMargins(12, 4, 12, 4)
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText(
            "在线拉取任意模型:输入关键词或 HF 仓库 ID,如 Qwen2.5 或 "
            "Qwen/Qwen2.5-0.5B-Instruct-GGUF")
        self.search_edit.returnPressed.connect(self.search_online)
        self.btn_search = QPushButton("🔍 在线搜索")
        self.btn_restore = QPushButton("恢复内置清单")
        search_lay.addWidget(self.search_edit, stretch=1)
        search_lay.addWidget(self.btn_search)
        search_lay.addWidget(self.btn_restore)
        l1.addWidget(search_bar)

        hint = QLabel("点击「刷新在线列表」从 HuggingFace 动态拉取热门 GGUF 模型;"
                      "上方筛选对在线/内置列表均生效。"
                      "也可用搜索框直接拉取任意仓库,或到「系统设置」指定自定义清单。")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#889;")
        l1.addWidget(hint)

        self.store_list = QListWidget()
        self.store_list.verticalScrollBar().valueChanged.connect(
            self._on_store_scroll)
        l1.addWidget(self.store_list)
        self.stack.addWidget(p1)
        # 商店列表分批无感渲染状态
        self._store_filtered: list[dict] = []
        self._store_rendered = 0
        self._STORE_BATCH = 40

    # ----- 2 本地模型 -----
    def _build_local_page(self) -> None:
        p2 = QWidget()
        l2 = QVBoxLayout(p2)
        l2.setContentsMargins(24, 20, 24, 20)
        l2.setSpacing(14)

        title2 = QLabel("💾 本地模型")
        title2.setStyleSheet("font-size:18px; font-weight:600;")
        l2.addWidget(title2)

        btn_bar2 = QHBoxLayout()
        btn_scan = QPushButton("🔄 刷新列表")
        btn_import = QPushButton("📂 导入模型文件")
        btn_import_ollama = QPushButton("🦙 导入 Ollama 模型")
        btn_open_dir = QPushButton("📁 打开模型目录")
        btn_bar2.addWidget(btn_scan)
        btn_bar2.addWidget(btn_import)
        btn_bar2.addWidget(btn_import_ollama)
        btn_bar2.addWidget(btn_open_dir)
        btn_bar2.addStretch()
        l2.addLayout(btn_bar2)
        self.btn_scan_local = btn_scan
        self.btn_import_local = btn_import

        self.local_list = QListWidget()
        l2.addWidget(self.local_list)
        self.stack.addWidget(p2)

        btn_scan.clicked.connect(self.scan_local_models_gui)
        btn_import.clicked.connect(self.import_model)
        btn_import_ollama.clicked.connect(self.import_ollama_model)
        btn_open_dir.clicked.connect(self._open_model_dir)

    # ----- 3 模型训练 -----
    def _build_train_page(self) -> None:
        p3 = QWidget()
        l3 = QVBoxLayout(p3)
        l3.setContentsMargins(24, 20, 24, 20)
        l3.setSpacing(14)

        title3 = QLabel("🎯 LoRA 微调训练")
        title3.setStyleSheet("font-size:18px; font-weight:600;")
        l3.addWidget(title3)

        form_zone = QFrame()
        fz = QHBoxLayout(form_zone)
        fz.setContentsMargins(0, 0, 0, 0)

        gb_data = QGroupBox("模型与数据")
        form_data = QFormLayout(gb_data)
        form_data.setSpacing(10)
        self.train_base = QComboBox()
        self.train_base.setEditable(True)
        self.train_base.setPlaceholderText(
            "HF 模型目录或仓库名(如 Qwen/Qwen2.5-0.5B-Instruct)")
        form_data.addRow("基座模型", self.train_base)
        ds_row = QHBoxLayout()
        self.train_dataset = QLineEdit()
        self.train_dataset.setPlaceholderText("JSONL 文件路径(prompt/completion 或 text)")
        btn_pick_ds = QPushButton("选择")
        btn_pick_ds.setFixedWidth(70)
        btn_pick_ds.clicked.connect(self._pick_dataset)
        ds_row.addWidget(self.train_dataset)
        ds_row.addWidget(btn_pick_ds)
        form_data.addRow("数据集", ds_row)
        self.train_outname = QLineEdit("my-lora")
        form_data.addRow("输出名称", self.train_outname)
        fz.addWidget(gb_data, stretch=1)

        gb_param = QGroupBox("训练参数")
        form_p = QFormLayout(gb_param)
        form_p.setSpacing(10)
        self.epoch_spin = QSpinBox()
        self.epoch_spin.setRange(1, 100)
        self.epoch_spin.setValue(3)
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setDecimals(6)
        self.lr_spin.setRange(0.000001, 1.0)
        self.lr_spin.setSingleStep(0.0001)
        self.lr_spin.setValue(0.0001)
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 64)
        self.batch_spin.setValue(1)
        self.lora_r_spin = QSpinBox()
        self.lora_r_spin.setRange(1, 128)
        self.lora_r_spin.setValue(8)
        self.maxlen_spin = QSpinBox()
        self.maxlen_spin.setRange(64, 8192)
        self.maxlen_spin.setValue(512)
        form_p.addRow("训练轮数", self.epoch_spin)
        form_p.addRow("学习率", self.lr_spin)
        form_p.addRow("批次大小", self.batch_spin)
        form_p.addRow("LoRA rank", self.lora_r_spin)
        form_p.addRow("最大长度", self.maxlen_spin)
        fz.addWidget(gb_param, stretch=1)
        l3.addWidget(form_zone)

        self.train_progress = QProgressBar()
        self.train_progress.setRange(0, 100)
        l3.addWidget(self.train_progress)

        self.train_log = QTextEdit()
        self.train_log.setReadOnly(True)
        self.train_log.append("训练日志:等待开始训练...\n")
        l3.addWidget(self.train_log)

        btn_row = QHBoxLayout()
        btn_start_train = QPushButton("开始训练")
        btn_start_train.setFixedHeight(38)
        btn_stop_train = QPushButton("停止训练")
        btn_stop_train.setEnabled(False)
        btn_row.addWidget(btn_start_train)
        btn_row.addWidget(btn_stop_train)
        btn_row.addStretch()
        l3.addLayout(btn_row)
        self.stack.addWidget(p3)

        self.btn_start_train = btn_start_train
        self.btn_stop_train = btn_stop_train
        self._train_worker: Optional[TrainWorker] = None
        btn_start_train.clicked.connect(self.start_train)
        btn_stop_train.clicked.connect(self.stop_train)

    # ----- 4 系统更新 -----
    def _build_update_page(self) -> None:
        p4 = QWidget()
        l4 = QVBoxLayout(p4)
        l4.setContentsMargins(24, 20, 24, 20)
        l4.setSpacing(14)

        title4 = QLabel("🔄 系统更新")
        title4.setStyleSheet("font-size:18px; font-weight:600;")
        l4.addWidget(title4)

        gb4 = QGroupBox("更新源配置")
        lay4 = QVBoxLayout(gb4)
        lay4.setSpacing(10)
        lay4.addWidget(QLabel("更新源地址(返回 JSON:{\"version\",\"url\",\"notes\"}):"))
        self.update_url = QLineEdit(str(cfg.get("update_source_url", "")))
        self.update_url.setPlaceholderText("https://example.com/novacore-update.json")
        lay4.addWidget(self.update_url)
        self.custom_update = QCheckBox("启用自定义更新源")
        self.custom_update.setChecked(bool(cfg.get("custom_update_enable", False)))
        lay4.addWidget(self.custom_update)
        btn_check_update = QPushButton("检查更新")
        lay4.addWidget(btn_check_update)
        l4.addWidget(gb4)
        self.btn_check_update = btn_check_update

        self.update_log = QTextEdit()
        self.update_log.setReadOnly(True)
        self.update_log.append(f"当前版本:{APP_VERSION}")
        self.update_log.append("更新日志:等待检查更新...\n")
        l4.addWidget(self.update_log)
        self.stack.addWidget(p4)

    # ----- 5 依赖管理 -----
    def _build_deps_page(self) -> None:
        p5 = QWidget()
        l5 = QVBoxLayout(p5)
        l5.setContentsMargins(24, 20, 24, 20)
        l5.setSpacing(14)

        title5 = QLabel("🔧 依赖管理")
        title5.setStyleSheet("font-size:18px; font-weight:600;")
        l5.addWidget(title5)

        self.dep_table = QTableWidget()
        self.dep_table.setColumnCount(4)
        self.dep_table.setHorizontalHeaderLabels(
            ["状态", "依赖名称", "功能说明", "操作"])
        self.dep_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents)
        self.dep_table.horizontalHeader().setStretchLastSection(True)
        self.dep_table.verticalHeader().setVisible(False)
        self.dep_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        l5.addWidget(self.dep_table)

        btn_bar5 = QHBoxLayout()
        btn_check_dep = QPushButton("🔍 检测全部依赖")
        btn_install_all = QPushButton("📦 一键补全缺失依赖")
        btn_bar5.addWidget(btn_check_dep)
        btn_bar5.addWidget(btn_install_all)
        btn_bar5.addStretch()
        l5.addLayout(btn_bar5)
        self.btn_check_dep = btn_check_dep
        self.btn_install_all = btn_install_all

        self.dep_log = QTextEdit()
        self.dep_log.setReadOnly(True)
        self.dep_log.setMaximumHeight(160)
        self.dep_log.append("依赖日志:点击「检测全部依赖」开始扫描\n")
        l5.addWidget(self.dep_log)
        self.stack.addWidget(p5)

    # ----- 6 系统设置 -----
    def _build_settings_page(self) -> None:
        p6 = QWidget()
        scroll6 = QScrollArea()
        scroll6.setWidgetResizable(True)
        cont6 = QWidget()
        lay6 = QVBoxLayout(cont6)
        lay6.setContentsMargins(24, 20, 24, 20)
        lay6.setSpacing(14)

        title6 = QLabel("⚙️ 系统设置")
        title6.setStyleSheet("font-size:18px; font-weight:600;")
        lay6.addWidget(title6)

        # 引擎设置(独立控件,修复旧版 QComboBox.clone 崩溃)
        gb_engine = QGroupBox("推理引擎设置")
        form_engine = QFormLayout(gb_engine)
        form_engine.setSpacing(12)
        self.engine_sel = QComboBox()
        self.engine_sel.addItems(ENGINE_CHOICES)
        cur_engine = str(cfg.get("default_engine", "auto"))
        if cur_engine in ENGINE_CHOICES:
            self.engine_sel.setCurrentIndex(ENGINE_CHOICES.index(cur_engine))
        form_engine.addRow("默认推理引擎", self.engine_sel)
        self.prefer_device_sel = QComboBox()
        self.prefer_device_sel.addItems(["CPU", "GPU", "NPU-OpenVINO"])
        cur_dev = str(cfg.get("prefer_device", "CPU"))
        if [self.prefer_device_sel.itemText(i)
                for i in range(self.prefer_device_sel.count())].count(cur_dev):
            self.prefer_device_sel.setCurrentText(cur_dev)
        form_engine.addRow("偏好推理设备", self.prefer_device_sel)
        self.ollama_host_edit = QLineEdit(str(cfg.get("ollama_host", "")))
        form_engine.addRow("Ollama 地址", self.ollama_host_edit)
        ollama_ctl = QHBoxLayout()
        self.ollama_auto_check = QCheckBox("自动启动 Ollama 服务(程序打开时自动拉起)")
        self.ollama_auto_check.setChecked(bool(cfg.get("ollama_auto_start", False)))
        btn_ollama_start = QPushButton("🚀 启动服务")
        btn_ollama_start.setToolTip("立即手动启动本机 Ollama 服务(仅本地地址)")
        btn_ollama_start.clicked.connect(self.start_ollama_manually)
        btn_ollama_test = QPushButton("测试连接")
        btn_ollama_test.clicked.connect(self.test_ollama_connection)
        ollama_ctl.addWidget(self.ollama_auto_check, stretch=1)
        ollama_ctl.addWidget(btn_ollama_start)
        ollama_ctl.addWidget(btn_ollama_test)
        form_engine.addRow("Ollama", ollama_ctl)
        self.ollama_status_lab = QLabel("Ollama 状态: 未检测")
        self.ollama_status_lab.setWordWrap(True)
        self.ollama_status_lab.setStyleSheet("color:#889;")
        form_engine.addRow("", self.ollama_status_lab)
        # NPU 探测状态(让用户一眼看到自己的机器是否有 Intel NPU)
        npu_info = hardware.detect_npu()
        npu_mark = "✅ 可用" if npu_info.get("available") else "❌ 不可用"
        self.npu_lab = QLabel(f"{npu_mark} | {npu_info.get('detail', '')}")
        self.npu_lab.setStyleSheet(
            "color:#10b981;" if npu_info.get("available") else "color:#ef4444;")
        form_engine.addRow("Intel NPU 状态", self.npu_lab)
        lay6.addWidget(gb_engine)

        # 生成参数
        gb_gen = QGroupBox("生成参数")
        form_gen = QFormLayout(gb_gen)
        form_gen.setSpacing(12)
        self.temp_slider = QSlider(Qt.Orientation.Horizontal)
        self.temp_slider.setRange(0, 200)
        self.temp_slider.setValue(int(float(cfg.get("gen_temperature", 0.7)) * 100))
        self.temp_lab = QLabel(f"{self.temp_slider.value() / 100:.2f}")
        row_t = QHBoxLayout()
        row_t.addWidget(self.temp_slider, stretch=9)
        row_t.addWidget(self.temp_lab, stretch=1)
        form_gen.addRow("温度 temperature", row_t)
        self.topp_spin = QDoubleSpinBox()
        self.topp_spin.setRange(0.01, 1.0)
        self.topp_spin.setSingleStep(0.05)
        self.topp_spin.setValue(float(cfg.get("gen_top_p", 0.9)))
        form_gen.addRow("Top-P", self.topp_spin)
        self.topk_spin = QSpinBox()
        self.topk_spin.setRange(0, 1000)
        self.topk_spin.setValue(int(cfg.get("gen_top_k", 40)))
        form_gen.addRow("Top-K", self.topk_spin)
        self.maxtok_spin = QSpinBox()
        self.maxtok_spin.setRange(1, 32768)
        self.maxtok_spin.setValue(int(cfg.get("gen_max_tokens", 2048)))
        form_gen.addRow("最大生成 tokens", self.maxtok_spin)
        self.ctx_spin = QSpinBox()
        self.ctx_spin.setRange(256, 131072)
        self.ctx_spin.setValue(int(cfg.get("gen_ctx_len", 4096)))
        form_gen.addRow("上下文长度", self.ctx_spin)
        self.sysopt_edit = QLineEdit(str(cfg.get("system_prompt", "")))
        form_gen.addRow("系统提示词", self.sysopt_edit)
        lay6.addWidget(gb_gen)

        # API 服务
        gb_api = QGroupBox("API 服务设置(OpenAI 兼容接口)")
        form_api = QFormLayout(gb_api)
        form_api.setSpacing(12)
        self.port_spin = QSpinBox()
        self.port_spin.setRange(1024, 65535)
        self.port_spin.setValue(int(cfg.get("api_port", 8000)))
        form_api.addRow("API 监听端口", self.port_spin)
        self.api_check = QCheckBox("启用 API 服务(可供外部前端调用)")
        self.api_check.setChecked(bool(cfg.get("api_enable", False)))
        form_api.addRow("", self.api_check)
        api_status_row = QHBoxLayout()
        self.api_status_lab = QLabel("状态:未启动")
        btn_api_toggle = QPushButton("立即启动/停止")
        btn_api_toggle.clicked.connect(self.on_toggle_api)
        api_status_row.addWidget(self.api_status_lab, stretch=1)
        api_status_row.addWidget(btn_api_toggle)
        form_api.addRow("当前状态", api_status_row)
        lay6.addWidget(gb_api)

        # 界面与杂项
        gb_ui = QGroupBox("界面与其它")
        form_ui = QFormLayout(gb_ui)
        form_ui.setSpacing(12)
        self.theme_sel = QComboBox()
        self.theme_sel.addItems(THEME_CHOICES)
        cur_theme = str(cfg.get("theme", "soft_dark"))
        if cur_theme in THEME_CHOICES:
            self.theme_sel.setCurrentIndex(THEME_CHOICES.index(cur_theme))
        form_ui.addRow("主题", self.theme_sel)
        self.ui_scale_sel = QComboBox()
        for label, factor in UI_SCALE_CHOICES.items():
            self.ui_scale_sel.addItem(label, float(factor))
        cur_scale = float(cfg.get("ui_scale", 1.0) or 1.0)
        scale_idx = self.ui_scale_sel.findData(cur_scale)
        self.ui_scale_sel.setCurrentIndex(scale_idx if scale_idx >= 0 else 0)
        form_ui.addRow("界面缩放", self.ui_scale_sel)
        form_ui.addRow("", QLabel("缩放越大文字/控件越大(部分布局重启后完全生效)"))
        self.manifest_edit = QLineEdit(str(cfg.get("store_manifest_url", "")))
        self.manifest_edit.setPlaceholderText("留空则使用内置模型清单")
        form_ui.addRow("自定义清单地址(可选)", self.manifest_edit)
        self.hwmon_check = QCheckBox("启用硬件实时监控")
        self.hwmon_check.setChecked(bool(cfg.get("hardware_monitor", True)))
        form_ui.addRow("", self.hwmon_check)
        self.fb_check = QCheckBox("显示下载悬浮球(右下角,可开关)")
        self.fb_check.setChecked(bool(cfg.get("floating_ball_enable", True)))
        self.fb_check.stateChanged.connect(self._on_fb_toggle)
        form_ui.addRow("", self.fb_check)
        lay6.addWidget(gb_ui)

        # 模型下载源(与商店页同步)
        gb_dl = QGroupBox("模型下载源")
        form_dl = QFormLayout(gb_dl)
        form_dl.setSpacing(12)
        self.download_source_sel = QComboBox()
        self.download_source_sel.addItems(
            ["auto", "official", "hf_mirror", "modelscope"])
        self.download_source_sel.setItemText(0,
            "自动(三源故障转移,国内镜像优先)")
        self.download_source_sel.setItemText(1, "官方 HuggingFace")
        self.download_source_sel.setItemText(2, "hf-mirror.com 国内镜像")
        self.download_source_sel.setItemText(3, "ModelScope 魔搭(国内最稳)")
        cur_dl_src = str(cfg.get("download_source", "auto"))
        dl_idx = self.download_source_sel.findText(cur_dl_src)
        self.download_source_sel.setCurrentIndex(
            dl_idx if dl_idx >= 0 else 0)
        self.download_source_sel.currentTextChanged.connect(
            self._on_dl_source_changed)
        form_dl.addRow("模型下载源", self.download_source_sel)
        btn_probe = QPushButton("🔎 自检各下载节点可达性")
        btn_probe.clicked.connect(self._probe_dl_nodes)
        form_dl.addRow("", btn_probe)
        lay6.addWidget(gb_dl)

        # pip 依赖下载镜像
        gb_pip = QGroupBox("依赖下载镜像(pip)")
        form_pip = QFormLayout(gb_pip)
        form_pip.setSpacing(12)
        self.pip_mirror_sel = QComboBox()
        self.pip_mirror_sel.addItem("自动(依次尝试全部镜像)", "auto")
        for mid, (_url, label) in zip(
                deps_core.PIP_MIRROR_IDS, deps_core.PIP_MIRRORS):
            self.pip_mirror_sel.addItem(f"{label}", mid)
        cur_mirror = str(cfg.get("pip_mirror", "auto"))
        mirror_idx = self.pip_mirror_sel.findData(cur_mirror)
        self.pip_mirror_sel.setCurrentIndex(mirror_idx if mirror_idx >= 0 else 0)
        form_pip.addRow("依赖下载镜像", self.pip_mirror_sel)
        mirror_row = QHBoxLayout()
        btn_mirror_test = QPushButton("测试各镜像")
        btn_mirror_test.clicked.connect(self.test_pip_mirrors)
        mirror_row.addWidget(btn_mirror_test)
        mirror_row.addStretch()
        # 注意:镜像测试按钮与结果标签必须加到 form_pip(gb_pip 组),
        # 且 gb_pip 必须加入页面布局——否则整个组无父被 GC 销毁,
        # 其中的 QComboBox 变成"已删除的 C/C++ 对象",保存设置时报
        # RuntimeError: wrapped C/C++ object of type QComboBox has been deleted。
        form_pip.addRow("", mirror_row)
        self.mirror_result_lab = QLabel(
            "提示:镜像用于加速 pip 依赖下载(如 llama-cpp-python、torch)")
        self.mirror_result_lab.setWordWrap(True)
        self.mirror_result_lab.setStyleSheet("color:#99a;")
        form_pip.addRow("", self.mirror_result_lab)
        lay6.addWidget(gb_pip)

        # 系统重置(危险操作区,带强确认防误触)
        gb_reset = QGroupBox("系统重置")
        reset_lay = QVBoxLayout(gb_reset)
        reset_lay.setSpacing(10)
        reset_lay.addWidget(QLabel(
            "「恢复默认设置」:仅把所有选项还原为默认值,保留已下载的模型与对话记录。\n"
            "「恢复出厂设置」:清空全部数据(配置、本地模型、对话历史、LoRA、数据集、日志),"
            "不可恢复!"))
        reset_btn_row = QHBoxLayout()
        self.btn_reset_default = QPushButton("↺ 恢复默认设置")
        self.btn_reset_default.setStyleSheet(
            "QPushButton{background:#b45309;} QPushButton:hover{background:#d97706;}")
        self.btn_reset_default.clicked.connect(self.on_reset_defaults)
        self.btn_reset_factory = QPushButton("☠ 恢复出厂设置")
        self.btn_reset_factory.setStyleSheet(
            "QPushButton{background:#b91c1c;} QPushButton:hover{background:#dc2626;}")
        self.btn_reset_factory.clicked.connect(self.on_factory_reset)
        reset_btn_row.addWidget(self.btn_reset_default)
        reset_btn_row.addWidget(self.btn_reset_factory)
        reset_btn_row.addStretch()
        reset_lay.addLayout(reset_btn_row)
        lay6.addWidget(gb_reset)

        btn_save = QPushButton("💾 保存全部设置")
        btn_save.setFixedHeight(_s(42))
        lay6.addWidget(btn_save)
        self.btn_save_set = btn_save

        lay6.addStretch()
        scroll6.setWidget(cont6)
        p6_lay = QVBoxLayout(p6)
        p6_lay.addWidget(scroll6)
        self.stack.addWidget(p6)

    def bind_all_signals(self) -> None:
        self.filter_hw.currentTextChanged.connect(self.refresh_store_list)
        self.filter_type.currentTextChanged.connect(self.refresh_store_list)
        self.device_sel.currentTextChanged.connect(self.on_device_sel_changed)
        self.btn_refresh_online.clicked.connect(self.refresh_online_store)
        self.btn_search.clicked.connect(self.search_online)
        self.btn_restore.clicked.connect(self._restore_builtin_store)
        self.btn_check_dep.clicked.connect(self.check_deps)
        self.btn_install_all.clicked.connect(self.install_all_missing_deps)
        self.btn_save_set.clicked.connect(self.save_settings)
        self.btn_check_update.clicked.connect(self.check_update)
        self.temp_slider.valueChanged.connect(
            lambda v: self.temp_lab.setText(f"{v / 100:.2f}"))

    # ==================== 对话功能 ====================
    def _chat_insert(self, text: str) -> None:
        """在聊天框末尾插入文本(流式 token 不换行)。"""
        cursor = self.chat_box.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(text)
        self.chat_box.setTextCursor(cursor)
        sb = self.chat_box.verticalScrollBar()
        sb.setValue(sb.maximum())

    def on_send(self) -> None:
        text = self.msg_input.text().strip()
        if not text or self._generating:
            return
        if not engine_mgr.is_loaded():
            QMessageBox.warning(
                self, "未加载模型",
                "请先加载模型:\n「本地模型」页选择模型点击「加载模型」,"
                "或上方下拉框选择后点「加载」。")
            return
        if self._pending_image:
            self._chat_insert(
                f"(图片已忽略:{Path(self._pending_image).name},"
                "当前文本引擎不支持图片输入)\n")
            self._pending_image = None

        params = GenParams.from_config(cfg)
        messages = self.chat_session.build_messages(text, params)
        self.chat_session.add("user", text)

        self._chat_insert(f"用户: {text}\n助手: ")
        self.msg_input.clear()
        self._generating = True
        self.btn_send.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.speed_lab.setText("生成中...")

        worker = GenerationWorker(engine_mgr, messages, params)
        worker.token.connect(self._chat_insert)
        worker.stat.connect(self._on_gen_stat)
        worker.finished.connect(self._on_gen_finished)
        worker.error.connect(self._on_gen_error)
        self._gen_worker = worker
        registry.register(worker)
        worker.start()

    def _on_gen_stat(self, stat: dict) -> None:
        self.speed_lab.setText(f"{stat.get('tok_per_s', 0)} tok/s")

    def _on_gen_error(self, msg: str) -> None:
        self._generating = False
        self.btn_send.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self._chat_insert("\n")
        self.chat_box.append(f"❌ {msg}")
        self.speed_lab.setText("生成失败")

    def _on_gen_finished(self, full: str, cancelled: bool) -> None:
        self._generating = False
        self.btn_send.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self._chat_insert("\n")
        if cancelled:
            self.speed_lab.setText("已手动停止")
            self.chat_box.append("(已手动停止,保留已生成部分)\n")
        else:
            self.chat_box.append("")
        if full.strip():
            self.chat_session.add("assistant", full)
        self.chat_session.save_history()
        self.refresh_history_list()

    def on_stop_generation(self) -> None:
        if not self._generating:
            return
        worker = getattr(self, "_gen_worker", None)
        if worker is not None:
            worker.cancel()
        engine_mgr.cancel()
        self.speed_lab.setText("正在停止...")

    def on_new_chat(self) -> None:
        if self._generating:
            self.on_stop_generation()
        self.chat_session.new_session()
        self.chat_box.clear()
        self.chat_box.append("=== 新会话已开始 ===\n")
        self.speed_lab.setText("--")
        self.refresh_history_list()

    # ---------- 对话历史(多会话) ----------
    def refresh_history_list(self) -> None:
        """根据持久化的会话列表刷新左侧历史面板,高亮当前会话。"""
        if not hasattr(self, "hist_list"):
            return
        cur = self.chat_session.session_id
        self.hist_list.clear()
        for s in self.chat_session.list_sessions():
            item = QListWidgetItem()
            row = HistoryRow(s, self, active=(s["id"] == cur))
            item.setSizeHint(row.sizeHint())
            self.hist_list.addItem(item)
            self.hist_list.setItemWidget(item, row)

    def _render_chat_messages(self) -> None:
        """把当前会话的消息重新渲染到聊天框(切换历史会话时使用)。"""
        self.chat_box.clear()
        msgs = self.chat_session.messages
        if not msgs:
            self.chat_box.append("=== 新会话已开始 ===\n")
            return
        for m in msgs:
            role = m.get("role")
            content = str(m.get("content", ""))
            if role == "user":
                self.chat_box.append(f"用户: {content}")
            elif role == "assistant":
                self.chat_box.append(f"助手: {content}\n")
        self.chat_box.append("")
        sb = self.chat_box.verticalScrollBar()
        sb.setValue(sb.maximum())

    def on_load_session(self, sid: str) -> None:
        """切换到指定历史会话。"""
        if sid == self.chat_session.session_id:
            return
        if self._generating:
            self.on_stop_generation()
        if not self.chat_session.load_session(sid):
            QMessageBox.warning(self, "无法打开", "该对话记录可能已被删除。")
            self.refresh_history_list()
            return
        self._render_chat_messages()
        self.speed_lab.setText("--")
        self.refresh_history_list()

    def on_delete_session(self, sid: str) -> None:
        """删除单条对话记录(隐私保护)。"""
        reply = QMessageBox.question(
            self, "删除对话",
            "确定删除这条对话记录吗?此操作不可恢复。")
        if reply != QMessageBox.StandardButton.Yes:
            return
        deleting_current = (sid == self.chat_session.session_id)
        self.chat_session.delete_session(sid)
        if deleting_current:
            # 删的是当前会话:切到最近一条,没有则新建
            summaries = self.chat_session.list_sessions()
            if summaries:
                self.chat_session.load_session(summaries[0]["id"])
            else:
                self.chat_session.new_session()
            self._render_chat_messages()
        self.refresh_history_list()

    def on_rename_session(self, sid: str, title: str) -> None:
        """重命名对话记录(手动命名后,自动命名不再覆盖)。"""
        self.chat_session.rename_session(sid, title)
        self.refresh_history_list()

    def on_clear_all_sessions(self) -> None:
        """清空全部对话历史(隐私保护)。"""
        reply = QMessageBox.question(
            self, "清空全部历史",
            "确定清空所有对话记录吗?此操作不可恢复,建议在需要保护隐私时使用。")
        if reply != QMessageBox.StandardButton.Yes:
            return
        if self._generating:
            self.on_stop_generation()
        for s in self.chat_session.list_sessions():
            self.chat_session.delete_session(s["id"])
        self.chat_session.new_session()
        self._render_chat_messages()
        self.refresh_history_list()

    # ---------- 附件 ----------
    def on_attach_file(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(self, "选择要附加的文件")
        if not files:
            return
        parts: list[str] = []
        for fp in files:
            p = Path(fp)
            try:
                content = p.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                parts.append(f"[文件: {p.name}](二进制文件,内容已跳过)")
                continue
            if len(content) > 200_000:
                content = content[:200_000] + "\n...(内容过长已截断)"
            parts.append(f"[文件: {p.name}]\n{content}\n")
        cur = self.msg_input.text()
        self.msg_input.setText((cur + "\n" if cur else "") + "\n".join(parts))
        self.msg_input.setFocus()

    def on_attach_image(self) -> None:
        fp, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "图片文件 (*.png *.jpg *.jpeg *.bmp *.webp *.gif);;所有文件 (*)")
        if not fp:
            return
        self._pending_image = fp
        self.chat_box.append(
            f"🖼️ 已选择图片 {Path(fp).name}(提示:当前文本引擎不支持图片输入,"
            "发送消息时将忽略该图片)\n")

    # ==================== 模型加载 ====================
    def _resolve_engine_id(self, model_ref: str) -> Optional[str]:
        """根据用户设置的 default_engine 选择引擎;auto 时按模型类型智能选。

        - 显式选择引擎时尊重用户意愿,只做可用性+兼容性校验;
        - auto:目录(OpenVINO IR)→openvino;GGUF→llamacpp 优先,否则 Ollama;
        - 顶部「推理设备」为 NPU-OpenVINO 时,强制 openvino 引擎。
        """
        p = Path(model_ref)
        is_dir = p.is_dir()
        is_gguf = p.is_file() and p.suffix.lower() == ".gguf"
        is_ov_dir = is_dir and (p / "openvino_model.xml").is_file()
        choice = str(cfg.get("default_engine", "auto")).strip()
        # 顶部推理设备选择(NPU-OpenVINO 强制走 openvino)
        device_choice = ""
        if hasattr(self, "device_sel"):
            try:
                device_choice = self.device_sel.currentText().strip()
            except Exception:
                device_choice = ""

        # 1) 用户在顶部选了 NPU-OpenVINO → 强制 openvino 引擎
        if "NPU" in device_choice:
            if is_ov_dir:
                ok, detail = engine_mgr.probe("openvino", refresh=True)
                if not ok:
                    QMessageBox.warning(
                        self, "OpenVINO 不可用",
                        f"NPU 推理需要 OpenVINO 引擎,但它当前不可用:\n{detail}\n\n"
                        f"请到「依赖管理」安装 openvino + openvino-genai。")
                    return None
                return "openvino"
            QMessageBox.warning(
                self, "模型与 NPU 不兼容",
                "NPU-OpenVINO 设备仅支持 NPU 专区的 OpenVINO IR 目录模型\n"
                "(商店「NPU专区」中标为 -int4-gq-ov / -int4-cw-ov 的对称 INT4 模型)。\n"
                "当前选择的不是有效的 NPU 模型;GGUF 可改用 CPU/GPU 设备加载。")
            return None

        # 2) auto:按模型类型+设备智能选
        if choice == "auto":
            if is_dir:
                return "openvino" if engine_mgr.probe("openvino")[0] else None
            # GGUF / 其它文件:llamacpp 优先(快+稳,genai 2026 对 GGUF 有已知崩溃)
            if engine_mgr.probe("llamacpp")[0]:
                return "llamacpp"
            if engine_mgr.probe("ollama")[0]:
                return "ollama"
            return None

        # 3) 显式选择引擎 → 尊重,但加安全校验
        ok, detail = engine_mgr.probe(choice, refresh=True)
        if not ok:
            QMessageBox.warning(
                self, "引擎不可用",
                f"你在设置中选择了「{choice}」引擎,但它当前不可用:\n{detail}\n\n"
                f"请先到「依赖管理」安装对应依赖,或把默认引擎改回 auto。")
            return None
        if choice == "openvino" and not (is_dir or is_gguf):
            QMessageBox.warning(
                self, "模型与引擎不兼容",
                "OpenVINO 引擎需要模型目录或 GGUF 文件;当前选择的不是有效模型。")
            return None
        if choice == "llamacpp" and not is_gguf:
            QMessageBox.warning(
                self, "模型与引擎不兼容",
                "llama.cpp 引擎只能加载 GGUF 文件;请选择 GGUF 模型。")
            return None
        return choice

    def _set_loading(self, loading: bool) -> None:
        self._loading = loading
        self.btn_chat_load.setEnabled(not loading)

    def _mark_model_status(self, name: str, text: str, color: str,
                           running: bool) -> None:
        """更新本地模型列表中对应模型行的运行状态徽章与按钮。"""
        row = self._model_rows.get(name)
        if row is not None:
            row.set_status(text, color, running)

    def request_load_model(self, model_ref: str,
                           display_name: Optional[str] = None,
                           engine_id: Optional[str] = None) -> None:
        if self._loading:
            QMessageBox.information(self, "提示", "正在加载模型,请稍候")
            return
        name = display_name or Path(model_ref).name
        eid = engine_id or self._resolve_engine_id(model_ref)
        if eid is None:
            msg = (f"无法加载 {name}:OpenVINO 模型目录需要安装 "
                   "openvino + openvino-genai。")
            self.chat_box.append(f"❌ {msg}\n")
            QMessageBox.warning(self, "无法加载", msg)
            return
        if eid == "ollama":
            # 全自动 Ollama 链路:启动服务 → 导入 GGUF(如需) → 加载
            self._load_ollama_chain(model_ref, name)
            return

        self._set_loading(True)
        self._loaded_model_name = name
        self._mark_model_status(name, "⏳ 启动中", "#f59e0b", True)
        # 设备选择:顶部「推理设备」下拉框的即时选择优先(用户在界面上的
        # 最新意图),配置 prefer_device 仅作兜底。旧实现只读配置,导致
        # 顶部选了 NPU-OpenVINO 而配置仍是 CPU 时模型实际跑在 CPU 上。
        device = ""
        if hasattr(self, "device_sel"):
            try:
                device = self.device_sel.currentText().strip()
            except Exception:
                device = ""
        if not device:
            device = str(cfg.get("prefer_device", "CPU"))
        if eid == "llamacpp":
            device = "GPU" if "GPU" in device and hardware.has_cuda_gpu() else "CPU"
        elif eid == "openvino":
            device = "NPU" if "NPU" in device else "CPU"
        self.chat_box.append(f"⏳ 正在加载模型 {name} ({eid}/{device})...")
        if eid == "openvino" and device == "NPU":
            self.chat_box.append(
                "提示:NPU 首次加载需编译模型(约 1-5 分钟),之后走编译缓存秒开\n")
        else:
            self.chat_box.append("")
        worker = ModelLoadWorker(engine_mgr, model_ref, eid, device=device)
        worker.progress.connect(
            lambda s: self._mark_model_status(name, f"⏳ {s}", "#f59e0b", True))
        worker.done.connect(
            lambda ok, msg, extra, n=name: self.on_load_done(ok, msg, extra, n))
        self._load_worker = worker
        registry.register(worker)
        worker.start()

    def _load_ollama_chain(self, model_ref: str, name: str) -> None:
        """全自动 Ollama 加载:确保服务在线后导入/加载,全程 UI 内完成。"""
        self._set_loading(True)
        self._loaded_model_name = name
        self._mark_model_status(name, "⏳ 启动中", "#f59e0b", True)
        self.chat_box.append(f"⏳ 正在启动 Ollama 服务并加载 {name} ...\n")
        worker = SimpleWorker(self._ensure_ollama_auto)
        worker.ok.connect(
            lambda res: self._ollama_ready_then_load(res, model_ref, name))
        worker.err.connect(lambda e: self._ollama_load_failed(name, str(e)))
        self._load_worker = worker
        registry.register(worker)
        worker.start()

    def _ollama_ready_then_load(self, result: object, model_ref: str,
                                name: str) -> None:
        ok, detail = result if isinstance(result, tuple) else (False, str(result))
        if not ok:
            self._ollama_load_failed(name, detail)
            return
        # Ollama 已就绪;GGUF 会在 ollama.load_model 内自动 create 导入
        worker = ModelLoadWorker(engine_mgr, model_ref, "ollama", device=None)
        worker.done.connect(
            lambda ok2, msg, extra, n=name: self.on_load_done(ok2, msg, extra, n))
        self._load_worker = worker
        registry.register(worker)
        worker.start()

    def _ollama_load_failed(self, name: str, detail: str) -> None:
        self._set_loading(False)
        self._loaded_model_name = ""
        self._mark_model_status(name, "❌ 启动失败", "#ef4444", False)
        self.chat_box.append(f"❌ Ollama 加载失败: {detail}\n")
        QMessageBox.warning(
            self, "Ollama 加载失败",
            detail + "\n\n请确认已安装 Ollama(https://ollama.com)。")

    # ==================== Ollama 服务管理 ====================
    def _ollama_sync_host(self) -> str:
        """把配置中的 Ollama 地址同步到引擎实例,返回当前地址。"""
        eng = engine_mgr.get("ollama")
        configured = str(cfg.get("ollama_host", "")).strip().rstrip("/")
        if configured:
            eng.host = configured
        return eng.host

    def _set_ollama_status(self, text: str) -> None:
        if hasattr(self, "ollama_status_lab"):
            self.ollama_status_lab.setText(text)

    def _spawn_and_wait_ollama(self, exe: str) -> tuple[bool, str]:
        """拉起 ollama serve 并等待就绪(最长约 60s)。返回 (是否在线, 说明)。"""
        try:
            proc = subprocess.Popen(
                [exe, "serve"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=(subprocess.CREATE_NO_WINDOW
                               if sys.platform == "win32" else 0))
        except OSError as e:
            return False, f"启动 ollama serve 失败: {e}"
        # 先等 2 秒:若进程立刻退出,说明启动被拒绝(端口占用/权限等),直接报错
        time.sleep(2.0)
        if proc.poll() is not None:
            return False, ("ollama serve 启动后立即退出(返回码 "
                           f"{proc.returncode});可能是 11434 端口被占用或"
                           "另一个 Ollama 实例异常。请关闭占用程序后重试。")
        eng = engine_mgr.get("ollama")
        # 等待就绪:前 20s 每 1s 探测,之后每 2s,最长约 60s
        start = time.time()
        deadline = start + 60.0
        while time.time() < deadline:
            ok, detail = eng.check_available()
            if ok:
                return True, detail
            elapsed = time.time() - start
            wait = 1.0 if elapsed < 20 else 2.0
            time.sleep(min(wait, max(deadline - time.time(), 0.2)))
        ok, detail = eng.check_available()
        if ok:
            return True, detail
        return False, ("已发起启动 Ollama 服务,但 60 秒内仍未就绪。"
                       "服务可能仍在后台启动中(首次启动较慢),请稍候重试;"
                       "若持续失败,请手动运行 ollama serve 查看错误输出。")

    def _ensure_ollama_auto(self) -> tuple[bool, str]:
        """确保 Ollama 服务在线。

        行为由「自动启动 Ollama 服务」checkbox 决定:
        - 勾选:检测到服务没跑就自动拉起本地 ollama serve 并等待就绪;
        - 未勾选:只做健康检查,不主动启动,由用户自己启动。
        仅当配置的 host 是本地地址时才自动启动;远程地址只做健康检查。
        """
        host = self._ollama_sync_host()
        eng = engine_mgr.get("ollama")
        ok, detail = eng.check_available()
        if ok:
            return True, detail

        is_local = any(mark in host for mark in ("127.0.0.1", "localhost",
                                                 "[::1]", "0.0.0.0"))
        auto = bool(cfg.get("ollama_auto_start", False))
        if not auto:
            exe = find_ollama_exe()
            location = f"(已找到:{exe})" if exe else \
                "(未在 PATH 或常见安装目录中找到 ollama.exe)"
            return False, (f"Ollama 服务未运行。{location}\n\n"
                           f"请先安装 Ollama,在系统托盘/开始菜单启动,点"
                           f"「启动服务」手动拉起,或在「系统设置」勾选"
                           f"「自动启动 Ollama 服务」让本程序代劳。")

        if not is_local:
            return False, (f"Ollama 服务未运行({host})。该地址不是本地地址,"
                           f"「自动启动」仅对本机服务生效;"
                           f"请确认远程 Ollama 已启动或改回本地地址。")

        exe = find_ollama_exe()
        if not exe:
            return False, "未找到 Ollama 可执行文件,请先安装 Ollama(https://ollama.com)"
        return self._spawn_and_wait_ollama(exe)

    def _startup_ollama_auto(self) -> None:
        """程序打开时:勾选了「自动启动」则后台拉起 Ollama 服务(不阻塞界面)。"""
        self._set_ollama_status("⏳ 正在自动启动 Ollama 服务...")
        worker = SimpleWorker(self._ensure_ollama_auto)
        worker.ok.connect(self._on_startup_ollama_done)
        worker.err.connect(lambda e: self._on_startup_ollama_done((False, str(e))))
        registry.register(worker)
        worker.start()

    def _on_startup_ollama_done(self, result: object) -> None:
        ok, detail = result if isinstance(result, tuple) else (False, str(result))
        if ok:
            self._set_ollama_status(f"✅ {detail}")
            logging.getLogger("novacore").info("Ollama 自动启动成功: %s", detail)
        else:
            self._set_ollama_status(f"⚠️ {detail}")
            logging.getLogger("novacore").warning("Ollama 自动启动失败: %s", detail)

    def start_ollama_manually(self) -> None:
        """手动启动 Ollama 服务(设置页「启动服务」按钮),后台执行。"""
        self._set_ollama_status("⏳ 正在启动 Ollama 服务...")
        worker = SimpleWorker(self._ollama_start_worker)
        worker.ok.connect(self._on_ollama_start_done)
        worker.err.connect(lambda e: self._on_ollama_start_done((False, str(e))))
        registry.register(worker)
        worker.start()

    def _ollama_start_worker(self) -> tuple[bool, str]:
        """手动启动的工作体:在线即返回;否则拉起本地 serve(仅本地地址)。"""
        host = self._ollama_sync_host()
        eng = engine_mgr.get("ollama")
        ok, detail = eng.check_available()
        if ok:
            return True, detail
        is_local = any(mark in host for mark in ("127.0.0.1", "localhost",
                                                 "[::1]", "0.0.0.0"))
        if not is_local:
            return False, (f"Ollama 服务未运行({host})。当前配置的是远程地址,"
                           f"本程序只能启动本机服务;请确认远程服务已启动,"
                           f"或把地址改回本地(127.0.0.1)后再启动。")
        exe = find_ollama_exe()
        if not exe:
            return False, "未找到 Ollama 可执行文件,请先安装 Ollama(https://ollama.com)"
        return self._spawn_and_wait_ollama(exe)

    def _on_ollama_start_done(self, result: object) -> None:
        ok, detail = result if isinstance(result, tuple) else (False, str(result))
        if ok:
            self._set_ollama_status(f"✅ {detail}")
            QMessageBox.information(self, "Ollama 已启动", detail)
        else:
            self._set_ollama_status(f"❌ {detail}")
            QMessageBox.warning(
                self, "Ollama 启动失败",
                detail + "\n\n请确认已安装 Ollama(https://ollama.com),"
                "或手动运行 ollama serve 查看错误。")

    def on_load_done(self, ok: bool, msg: str, extra: dict,
                     name: str) -> None:
        self._set_loading(False)
        if ok:
            real_name = (engine_mgr.current.model_name if engine_mgr.current
                         else name) or name
            cfg.set("current_loaded_model", real_name)
            save_config(cfg)
            self._loaded_model_name = name
            detail = extra.get("detail", "") if isinstance(extra, dict) else ""
            device = extra.get("device") if isinstance(extra, dict) else None
            # 状态徽章如实标注实际运行设备(NPU 未成功时绝不假装在跑 NPU)
            if device:
                self._mark_model_status(name, f"✅ 运行中({device})", "#10b981", True)
            else:
                self._mark_model_status(name, "✅ 运行中", "#10b981", True)
            self.chat_box.append(f"✅ {msg}" + (f"({detail})" if detail else "")
                                 + "\n")
            self.engine_lab.setText(f"当前引擎: {engine_mgr.current_id}")
            self.update_loaded_model()
            # NPU 回退 CPU:明确告知用户原因,而不是静默降级
            if detail and "回退" in str(detail):
                QMessageBox.warning(
                    self, "已回退到 CPU 运行",
                    f"模型已在 CPU 上运行,NPU 未启用。\n\n{detail}\n\n"
                    "如希望在 NPU 上运行,请在「模型商店 → NPU专区」下载 "
                    "标记为 -int4-gq-ov / -int4-cw-ov 的 NPU 专用模型。")
        else:
            self._loaded_model_name = ""
            self._mark_model_status(name, "❌ 启动失败", "#ef4444", False)
            self.chat_box.append(f"❌ 模型加载失败: {msg}\n")
            QMessageBox.critical(self, "模型加载失败", msg)

    def on_load_selected_model(self) -> None:
        display = self.model_sel.currentText().strip()
        if not display:
            QMessageBox.information(self, "提示", "暂无本地模型,请先下载或导入")
            return
        name = display.replace("(已加载)", "").strip()
        model_ref = self._model_ref_for(name)
        if model_ref is None:
            QMessageBox.warning(self, "错误", f"找不到模型文件: {name}")
            return
        self.request_load_model(model_ref, name)

    def _model_ref_for(self, name: str) -> Optional[str]:
        for item in self._local_models:
            if item["name"] == name:
                return item["path"]
        p = paths.LOCAL_MODEL_DIR / name
        return str(p) if p.exists() else None

    def import_ollama_model(self) -> None:
        """本地模型页:连接 Ollama(自动启动)并选择其已有模型导入加载。"""
        self.chat_box.append("⏳ 正在连接 Ollama 服务...\n")
        worker = SimpleWorker(self._ensure_ollama_auto)
        worker.ok.connect(self._on_ollama_ready)
        worker.err.connect(lambda e: QMessageBox.warning(
            self, "Ollama", f"连接异常: {e}"))
        registry.register(worker)
        worker.start()

    def on_load_ollama(self) -> None:
        self.chat_box.append("⏳ 正在检测 Ollama 服务...\n")
        worker = SimpleWorker(self._ensure_ollama_auto)
        worker.ok.connect(self._on_ollama_ready)
        worker.err.connect(lambda e: QMessageBox.warning(
            self, "Ollama", f"检测异常: {e}"))
        registry.register(worker)
        worker.start()

    def _on_ollama_ready(self, result: object) -> None:
        ok, detail = result if isinstance(result, tuple) else (False, str(result))
        if not ok:
            QMessageBox.warning(
                self, "Ollama 不可用",
                detail + "\n\n请确认已安装 Ollama,或到「系统设置」勾选自动启动。")
            return
        eng = engine_mgr.get("ollama")
        models = eng.list_models()
        if models:
            name, okdlg = QInputDialog.getItem(
                self, "加载 Ollama 模型", "选择模型:", models, 0, False)
            if okdlg and name:
                self.request_load_model(name, name, engine_id="ollama")
            return
        name, okdlg = QInputDialog.getText(
            self, "拉取 Ollama 模型",
            "Ollama 尚无模型,输入模型名拉取(如 qwen2.5:0.5b):",
            text="qwen2.5:0.5b")
        if okdlg and name.strip():
            self._pull_ollama_model(name.strip())

    def _pull_ollama_model(self, name: str) -> None:
        self.chat_box.append(f"⏳ 正在拉取 Ollama 模型 {name} ...\n")
        eng = engine_mgr.get("ollama")

        def do_pull():
            eng.pull_model(name, progress=lambda _s, _d: None)
            return name

        worker = SimpleWorker(do_pull)
        worker.ok.connect(lambda n: self.request_load_model(n, n, engine_id="ollama"))
        worker.err.connect(lambda e: self._pull_failed(str(e)))
        registry.register(worker)
        worker.start()

    def _pull_failed(self, msg: str) -> None:
        self.chat_box.append(f"❌ Ollama 拉取失败: {msg}\n")
        QMessageBox.critical(self, "拉取失败", msg)

    def on_unload_model(self) -> None:
        if self._generating:
            self.on_stop_generation()
        engine_mgr.unload_all()
        cfg.set("current_loaded_model", "")
        save_config(cfg)
        self._loaded_model_name = ""
        # 所有模型行状态复位为已停止
        for row in self._model_rows.values():
            row.set_status("💤 已停止", "#889", False)
        self.engine_lab.setText("当前引擎: 未加载")
        self.chat_box.append("⏹️ 已停止模型并释放显存/内存(模型文件保留,可随时重新加载)\n")
        self.update_loaded_model()

    def update_loaded_model(self) -> None:
        self.model_sel.blockSignals(True)
        self.model_sel.clear()
        loaded = str(cfg.get("current_loaded_model", ""))
        added = set()
        if loaded:
            suffix = " (已加载)" if engine_mgr.is_loaded() else ""
            self.model_sel.addItem(f"{loaded}{suffix}")
            added.add(loaded)
        for item in self._local_models:
            if item["name"] not in added and item.get("valid"):
                self.model_sel.addItem(item["name"])
                added.add(item["name"])
        if loaded:
            idx = self.model_sel.findText(
                loaded, Qt.MatchFlag.MatchStartsWith)
            if idx >= 0:
                self.model_sel.setCurrentIndex(idx)
        self.model_sel.blockSignals(False)
        # 训练页基座下拉同步本地 HF 目录(非 GGUF)
        self._sync_train_base_choices(loaded)

    def _sync_train_base_choices(self, loaded: str) -> None:
        cur = self.train_base.currentText()
        self.train_base.clear()
        self.train_base.addItem("Qwen/Qwen2.5-0.5B-Instruct")
        for item in self._local_models:
            p = Path(item["path"])
            if p.is_dir() and (p / "config.json").exists():
                self.train_base.addItem(item["name"])
        if cur:
            self.train_base.setEditText(cur)

    # ==================== 商店 ====================
    def refresh_store_list(self) -> None:
        """按筛选条件分批渲染商店列表:先渲染一批,滚动到底部时无感追加,
        避免一次性构造所有 StoreRow 造成卡顿。

        硬件筛选规则:
        - 全部:不做硬件过滤
        - CPU/GPU:匹配 hw 含 CPU 或 GPU 的模型(但不含 NPU-only 逻辑,
          有 NPU 的模型也会出现在这里,因为它们确实能跑 CPU/GPU)
        - NPU专区:只显示 hw 含 NPU 的模型(GGUF 格式,通过 OpenVINO 加载)
        """
        hw = self.filter_hw.currentText()
        typ = self.filter_type.currentText()
        filtered: list[dict] = []
        for m in self.store_data:
            mhw = m.get("hw") or []
            if hw == "全部":
                ok1 = True
            elif hw == "CPU/GPU":
                ok1 = ("CPU" in mhw) or ("GPU" in mhw)
            elif hw == "NPU专区":
                ok1 = "NPU" in mhw
            else:
                ok1 = True
            ok2 = (typ == "全部") or (typ == m.get("category"))
            if ok1 and ok2:
                filtered.append(m)
        self._store_filtered = filtered
        self._store_rendered = 0
        self.store_list.clear()
        self._append_store_batch(self._STORE_BATCH)

    def _append_store_batch(self, n: int) -> bool:
        """追加渲染下一批模型行(纯本地操作,瞬时无感);返回是否仍有未渲染项。"""
        items = self._store_filtered
        start = self._store_rendered
        end = min(start + n, len(items))
        for m in items[start:end]:
            item = QListWidgetItem()
            row = StoreRow(m, self, hw_filter=self.filter_hw.currentText())
            item.setSizeHint(row.sizeHint())
            self.store_list.addItem(item)
            self.store_list.setItemWidget(item, row)
        self._store_rendered = end
        return end < len(items)

    def _on_store_scroll(self, value: int) -> None:
        """滚动接近底部时无感追加下一批(不弹出任何加载提示)。"""
        sb = self.store_list.verticalScrollBar()
        if sb.maximum() - value <= sb.pageStep():
            self._append_store_batch(self._STORE_BATCH)

    def on_download_src_changed(self, index: int) -> None:
        if 0 <= index < len(SOURCE_CHOICES):
            cfg.set("download_source", SOURCE_CHOICES[index])
            save_config(cfg)
            # 同步设置页(双向同步)
            if hasattr(self, "download_source_sel"):
                self.download_source_sel.blockSignals(True)
                self.download_source_sel.setCurrentIndex(index)
                self.download_source_sel.blockSignals(False)

    def on_probe_download_nodes(self) -> None:
        """后台自检三个下载源节点,展示可达性与延迟,供用户选择最快源。"""
        def _probe():
            return modelstore.probe_download_sources()

        def _done(results: object) -> None:
            if not isinstance(results, list) or not results:
                QMessageBox.warning(self, "自检失败", "未获取到节点探测结果")
                return
            lines = []
            best = None
            for r in results:
                if r.get("ok"):
                    lines.append(f"✅ {r['label']}  {r.get('latency_ms')}ms")
                    if best is None or r.get("latency_ms", 1e9) < best.get("latency_ms", 1e9):
                        best = r
                else:
                    lines.append(f"❌ {r['label']}  不可达")
            text = "下载节点自检结果:\n" + "\n".join(lines)
            if best:
                text += (f"\n\n建议选择:「{best['label']}」" 
                         f"(可在上方「下载源」下拉切换到 {best['id']})")
            QMessageBox.information(self, "节点自检", text)

        self.btn_probe_nodes.setText("⏳ 自检中...")
        self.btn_probe_nodes.setEnabled(False)
        worker = SimpleWorker(_probe)
        worker.ok.connect(_done)
        worker.err.connect(lambda e: QMessageBox.warning(self, "自检失败", str(e)))
        worker.finished.connect(
            lambda: (self.btn_probe_nodes.setText("🔍 自检节点"),
                     self.btn_probe_nodes.setEnabled(True)))
        registry.register(worker)
        worker.start()

    # ==================== 下载(通过悬浮球) ====================
    def start_via_ball(self, entry: dict) -> None:
        """启动下载,进度行渲染到悬浮球面板内(应用内,不会被任务栏截断)。

        GGUF → 单文件下载;OpenVINO IR(NPU 专区)→ 目录多文件下载。"""
        save_path = _entry_save_path(entry)
        pref = str(cfg.get("download_source", "auto"))
        if str(entry.get("format", "gguf")) == "openvino_dir":
            # 仅当 xml + bin 均完整时才视为已存在(半成品 .part 可续传)
            _xml = Path(save_path) / "openvino_model.xml"
            _bin = Path(save_path) / "openvino_model.bin"
            if _xml.exists() and _bin.exists() and _bin.stat().st_size > 0:
                QMessageBox.information(
                    self, "模型已存在",
                    f"该 NPU 模型已下载:\n{save_path}\n可直接在「本地模型」页加载。")
                return
            worker = OVDirectoryDownloadWorker(entry, save_path, preference=pref)
        else:
            worker = DownloadWorker(entry, save_path, preference=pref)
        self.floating_ball.add_download(entry, worker)
        registry.register(worker)
        worker.start()

    # ==================== 在线搜索 ====================
    def search_online(self) -> None:
        q = self.search_edit.text().strip()
        if not q:
            QMessageBox.information(
                self, "提示",
                "请输入关键词或 HuggingFace 仓库 ID\n"
                "(如 Qwen2.5 或 Qwen/Qwen2.5-0.5B-Instruct-GGUF)")
            return
        self.btn_search.setEnabled(False)
        self.btn_restore.setEnabled(False)
        worker = SimpleWorker(lambda: modelstore.search_hub(q))
        worker.ok.connect(self._on_search_result)
        worker.err.connect(self._on_search_error)
        self._search_worker = worker
        registry.register(worker)
        worker.start()

    def _restore_builtin_store(self) -> None:
        self.store_data = modelstore.get_builtin_store()
        self.refresh_store_list()

    def _on_search_result(self, entries: object) -> None:
        self.btn_search.setEnabled(True)
        self.btn_restore.setEnabled(True)
        if not isinstance(entries, list) or not entries:
            QMessageBox.information(
                self, "未找到",
                "未在线找到匹配的 GGUF 模型。\n"
                "可尝试更短的关键词,或直接输入仓库 ID(org/repo)。")
            return
        self.store_data = entries
        self.refresh_store_list()
        QMessageBox.information(self, "搜索完成",
                                f"在线找到 {len(entries)} 个 GGUF 模型")

    def _on_search_error(self, msg: str) -> None:
        self.btn_search.setEnabled(True)
        self.btn_restore.setEnabled(True)
        QMessageBox.warning(self, "搜索失败",
                            f"在线搜索失败:{msg}\n请检查网络连接后重试。")

    def refresh_online_store(self) -> None:
        url = str(cfg.get("store_manifest_url", "")).strip()
        if url:
            # 配置了自定义清单地址 → 优先加载清单
            entries = modelstore.load_manifest(url)
            if not entries:
                QMessageBox.critical(
                    self, "加载失败",
                    f"无法从 {url} 加载模型清单\n"
                    "(检查文件/地址是否有效,JSON 是否为模型数组)")
                return
            self.store_data = entries
            self.refresh_store_list()
            QMessageBox.information(self, "成功",
                                    f"清单加载完成,共 {len(entries)} 个模型")
            return
        # 未配置清单地址 → 动态从 HuggingFace 拉取在线热门模型列表
        self.btn_refresh_online.setEnabled(False)
        self.btn_refresh_online.setText("⏳ 正在加载在线列表...")
        worker = SimpleWorker(lambda: modelstore.fetch_online_catalog())
        worker.ok.connect(self._on_online_catalog_result)
        worker.err.connect(self._on_online_catalog_error)
        self._catalog_worker = worker
        registry.register(worker)
        worker.start()

    def _on_online_catalog_result(self, entries: object) -> None:
        self._catalog_done()
        entries = entries if isinstance(entries, list) else []
        if entries:
            self.store_data = entries
            self.refresh_store_list()
            QMessageBox.information(
                self, "已刷新在线列表",
                f"从 HuggingFace 动态拉取到 {len(entries)} 个 GGUF 模型,\n"
                "上方筛选可直接过滤在线列表。")
        else:
            self.store_data = modelstore.get_builtin_store()
            self.refresh_store_list()
            QMessageBox.information(
                self, "已回退内置清单",
                "在线拉取暂无结果(网络不可达或未找到 GGUF),\n"
                "已回退到内置清单。")

    def _on_online_catalog_error(self, msg: str) -> None:
        self._catalog_done()
        self.store_data = modelstore.get_builtin_store()
        self.refresh_store_list()
        QMessageBox.warning(
            self, "在线拉取失败",
            f"{msg}\n\n请检查网络后重试,已回退到内置清单。")

    def _catalog_done(self) -> None:
        self.btn_refresh_online.setEnabled(True)
        self.btn_refresh_online.setText("🔄 刷新在线列表")

    # ==================== 本地模型 ====================
    def scan_local_models_gui(self) -> None:
        def do_scan():
            return gguf.scan_local_models(paths.LOCAL_MODEL_DIR)

        self._scan_worker = SimpleWorker(do_scan)
        self._scan_worker.ok.connect(self._on_scan_done)
        registry.register(self._scan_worker)
        self._scan_worker.start()

    def _on_scan_done(self, items: object) -> None:
        if not isinstance(items, list):
            return
        self._local_models = items
        self._model_rows.clear()
        self.local_list.clear()
        if not items:
            it = QListWidgetItem("  暂无本地模型,可在商店下载或手动导入")
            it.setForeground(QColor("#889"))
            self.local_list.addItem(it)
        for m in items:
            item = QListWidgetItem()
            row = LocalModelRow(m, self)
            item.setSizeHint(row.sizeHint())
            self.local_list.addItem(item)
            self.local_list.setItemWidget(item, row)
            self._model_rows[m["name"]] = row
        # 若已有模型在运行,重新标注其状态
        if self._loaded_model_name:
            self._mark_model_status(self._loaded_model_name,
                                    "✅ 运行中", "#10b981", True)
        self.update_loaded_model()

    def import_model(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择模型文件", str(paths.LOCAL_MODEL_DIR),
            "模型文件 (*.gguf *.bin *.safetensors);;所有文件 (*)")
        if not files:
            return
        import shutil
        copied = 0
        for fp in files:
            src = Path(fp)
            dst = paths.LOCAL_MODEL_DIR / src.name
            if src.resolve() == dst.resolve():
                continue
            try:
                shutil.copy2(src, dst)
                copied += 1
            except OSError as e:
                QMessageBox.warning(self, "导入失败", f"{src.name}: {e}")
        self.scan_local_models_gui()
        if copied:
            QMessageBox.information(self, "完成", f"成功导入 {copied} 个模型文件")

    def _open_model_dir(self) -> None:
        paths.LOCAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        if sys.platform == "win32":
            import os
            os.startfile(str(paths.LOCAL_MODEL_DIR))  # noqa: S606

    # ==================== 训练 ====================
    def _pick_dataset(self) -> None:
        fp, _ = QFileDialog.getOpenFileName(
            self, "选择训练数据集", "", "JSONL 数据集 (*.jsonl *.json);;所有文件 (*)")
        if fp:
            self.train_dataset.setText(fp)

    def start_train(self) -> None:
        if self._train_worker is not None and self._train_worker.isRunning():
            QMessageBox.information(self, "提示", "已有训练任务在进行中")
            return
        base = self.train_base.currentText().strip()
        ds = self.train_dataset.text().strip()
        out_name = self.train_outname.text().strip() or "my-lora"
        if not base:
            QMessageBox.warning(self, "错误", "请填写基座模型")
            return
        if not ds:
            QMessageBox.warning(self, "错误", "请选择训练数据集(JSONL)")
            return
        out_dir = paths.LORAS_DIR / Path(out_name).name
        worker = TrainWorker(
            base_model=base, dataset_path=ds, output_dir=str(out_dir),
            epochs=self.epoch_spin.value(), lr=self.lr_spin.value(),
            batch_size=self.batch_spin.value(), lora_r=self.lora_r_spin.value(),
            max_len=self.maxlen_spin.value())
        worker.log.connect(self.train_log.append)
        worker.progress.connect(self.train_progress.setValue)
        worker.done.connect(self._on_train_done)
        self._train_worker = worker
        registry.register(worker)
        self.btn_start_train.setEnabled(False)
        self.btn_stop_train.setEnabled(True)
        self.train_log.append(f"===== 提交训练任务:{base} =====")
        worker.start()

    def _on_train_done(self, ok: bool, msg: str) -> None:
        self.btn_start_train.setEnabled(True)
        self.btn_stop_train.setEnabled(False)
        self.train_log.append(f"===== 训练结束:{msg} =====\n")

    def stop_train(self) -> None:
        if self._train_worker is not None and self._train_worker.isRunning():
            self._train_worker.stop()
            self.train_log.append("⏹ 正在停止训练(完成当前步后安全中止)...")

    # ==================== 系统更新 ====================
    DEFAULT_UPDATE_URL = ("https://raw.githubusercontent.com/NovaCore-Local/"
                          "NovaCore/main/update.json")

    def check_update(self) -> None:
        url = self.update_url.text().strip()
        use_custom = self.custom_update.isChecked()
        cfg.set("update_source_url", url)
        cfg.set("custom_update_enable", use_custom)
        save_config(cfg)

        if use_custom and not url:
            self.update_log.append("❌ 已勾选「启用自定义更新源」但未填写地址")
            return
        if not use_custom:
            # 不启用自定义源 → 使用内置默认更新地址
            url = self.DEFAULT_UPDATE_URL
            self.update_log.append(f"使用默认更新源: {url}")
        else:
            self.update_log.append(f"使用自定义更新源: {url}")

        def do_check():
            import requests
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            if not isinstance(data, dict):
                raise ValueError("更新信息不是 JSON 对象")
            return data

        self._update_worker = SimpleWorker(do_check)
        self._update_worker.ok.connect(self._on_update_info)
        self._update_worker.err.connect(
            lambda e: self.update_log.append(f"❌ 更新检查失败: {e}"))
        registry.register(self._update_worker)
        self._update_worker.start()

    def _on_update_info(self, data: dict) -> None:
        latest = str(data.get("version", "")).strip()
        notes = str(data.get("notes", ""))
        dl_url = str(data.get("url", ""))
        self.update_log.append(f"当前版本: {APP_VERSION}")
        self.update_log.append(f"最新版本: {latest or '(未提供)'}")
        if notes:
            self.update_log.append(f"更新说明: {notes}")
        if latest and latest != APP_VERSION:
            self.update_log.append("🆕 发现新版本!" +
                                   (f" 下载地址: {dl_url}" if dl_url else ""))
        elif latest:
            self.update_log.append("✅ 已是最新版本")

    # ==================== 依赖管理 ====================
    def check_deps(self) -> None:
        results = deps_core.check_deps()
        self.dep_table.setRowCount(len(results))
        for row_idx, dep in enumerate(results):
            ok = dep["installed"]
            status = "✅" if ok else ("⚠️ 可选" if dep["optional"] else "❌ 缺失")
            item_stat = QTableWidgetItem(status)
            item_stat.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            item_stat.setForeground(QColor("#10b981" if ok else "#ef4444"))
            self.dep_table.setItem(row_idx, 0, item_stat)
            self.dep_table.setItem(row_idx, 1, QTableWidgetItem(dep["name"]))
            self.dep_table.setItem(row_idx, 2, QTableWidgetItem(dep["desc"]))
            btn = QPushButton("重新安装" if ok else "安装")
            btn.setFixedWidth(_s(90))
            btn.clicked.connect(
                lambda _, pkg=dep["key"]: self.install_single_dep(pkg))
            self.dep_table.setCellWidget(row_idx, 3, btn)
        missing = [d["key"] for d in results if not d["installed"]]
        self.dep_log.append(f"检测完成:{len(results) - len(missing)} 已安装,"
                            f" {len(missing)} 缺失"
                            + (f"({', '.join(missing)})" if missing else ""))

    def install_single_dep(self, pkg: str) -> None:
        self.dep_log.append(f"正在安装 {pkg} ...")
        worker = DepInstallWorker([pkg],
                                  mirror=str(cfg.get("pip_mirror", "auto")))
        worker.log.connect(self.dep_log.append)
        worker.done.connect(lambda ok, msg: self.check_deps())
        registry.register(worker)
        worker.start()

    def install_all_missing_deps(self) -> None:
        missing = [d["key"] for d in deps_core.check_deps() if not d["installed"]]
        if not missing:
            QMessageBox.information(self, "提示", "所有依赖均已安装")
            return
        reply = QMessageBox.question(
            self, "确认",
            f"将安装以下缺失依赖:\n{', '.join(missing)}\n是否继续?")
        if reply != QMessageBox.StandardButton.Yes:
            return
        self.dep_log.append(f"开始批量安装:{', '.join(missing)}")
        worker = DepInstallWorker(missing,
                                  mirror=str(cfg.get("pip_mirror", "auto")))
        worker.log.connect(self.dep_log.append)
        worker.done.connect(lambda ok, msg: self.check_deps())
        registry.register(worker)
        worker.start()

    # ==================== 镜像探测 / Ollama 连接 ====================
    def test_pip_mirrors(self) -> None:
        """后台探测各 pip 镜像可用性与延迟,把结果写入设置页。"""
        self.mirror_result_lab.setText("正在测试各镜像,请稍候...")
        worker = SimpleWorker(deps_core.probe_pip_mirrors)
        worker.ok.connect(self._on_mirror_probe_done)
        worker.err.connect(lambda e: self.mirror_result_lab.setText(
            f"探测失败: {e}"))
        registry.register(worker)
        worker.start()

    def _on_mirror_probe_done(self, results: object) -> None:
        if not isinstance(results, list) or not results:
            self.mirror_result_lab.setText("未获取到镜像探测结果")
            return
        lines = []
        for r in results:
            if r.get("ok"):
                lines.append(f"✅ {r['label']}  {r.get('latency_ms')}ms")
            else:
                lines.append(f"❌ {r['label']} 不可达")
        self.mirror_result_lab.setText("镜像探测结果:\n" + "\n".join(lines))

    def test_ollama_connection(self) -> None:
        host = self.ollama_host_edit.text().strip().rstrip("/")
        # 先查本地 exe 位置(不区分是否勾选 checkbox,只查可用性)
        exe = find_ollama_exe() or ""
        exe_msg = f"ollama.exe 位于: {exe}" if exe else \
            "未在 PATH 或常见安装目录中找到 ollama.exe(可能未安装)"
        # 再测 HTTP 连通
        eng = engine_mgr.get("ollama")
        eng.host = host or "http://127.0.0.1:11434"
        ok, detail = eng.check_available()
        auto = bool(cfg.get("ollama_auto_start", False))
        auto_msg = "自动启动:✅ 已勾选(程序会自动拉起)" if auto else "自动启动:❌ 未勾选(需手动启动)"
        body = f"📡 目标地址: {eng.host}\n{auto_msg}\n🗂️ {exe_msg}\n\n"
        if ok:
            self._set_ollama_status(f"✅ {detail}")
            QMessageBox.information(self, "Ollama 连接正常", body + f"✅ {detail}")
        else:
            self._set_ollama_status(f"⚠️ {detail}")
            QMessageBox.warning(self, "Ollama 连接失败", body +
                                f"❌ {detail}\n\n"
                                f"如已安装 Ollama,请点「启动服务」手动拉起;"
                                f"或勾选「自动启动 Ollama 服务」让本程序代劳。")

    # ==================== API 服务 ====================
    def _auto_start_api(self) -> None:
        ok, msg = self.api_server.start()
        self._update_api_status(ok, msg)

    def on_toggle_api(self) -> None:
        if self.api_server.running:
            self.api_server.stop()
            self._update_api_status(False, "API 服务已停止")
        else:
            ok, msg = self.api_server.start()
            self._update_api_status(ok, msg)
            if not ok:
                QMessageBox.warning(self, "API 启动失败", msg)

    def _update_api_status(self, running: bool, msg: str) -> None:
        self.api_status_lab.setText(
            f"状态:{'运行中' if running else '未运行'}({msg})")

    # ==================== 设置保存 ====================
    def save_settings(self) -> None:
        cfg.set("default_engine", self.engine_sel.currentText())
        cfg.set("prefer_device", self.prefer_device_sel.currentText())
        cfg.set("ollama_host", self.ollama_host_edit.text().strip())
        cfg.set("ollama_auto_start", self.ollama_auto_check.isChecked())
        cfg.set("pip_mirror", self.pip_mirror_sel.currentData() or "auto")
        cfg.set("ui_scale", float(self.ui_scale_sel.currentData() or 1.0))
        cfg.set("gen_temperature", self.temp_slider.value() / 100)
        cfg.set("gen_top_p", self.topp_spin.value())
        cfg.set("gen_top_k", self.topk_spin.value())
        cfg.set("gen_max_tokens", self.maxtok_spin.value())
        cfg.set("gen_ctx_len", self.ctx_spin.value())
        cfg.set("system_prompt", self.sysopt_edit.text())
        cfg.set("api_port", self.port_spin.value())
        cfg.set("api_enable", self.api_check.isChecked())
        cfg.set("theme", self.theme_sel.currentText())
        cfg.set("store_manifest_url", self.manifest_edit.text().strip())
        cfg.set("hardware_monitor", self.hwmon_check.isChecked())
        cfg.set("floating_ball_enable", self.fb_check.isChecked())
        cfg.set("download_source", self.download_source_sel.currentText())
        cfg.set("update_source_url", self.update_url.text().strip())
        cfg.set("custom_update_enable", self.custom_update.isChecked())
        save_config(cfg)

        # 应用到运行中的组件(保存后即时生效)
        ollama = engine_mgr.get("ollama")
        ollama.host = str(cfg.get("ollama_host", "")).rstrip("/") or \
            "http://127.0.0.1:11434"
        self.device_sel.setCurrentText(str(cfg.get("prefer_device", "CPU")))
        if cfg.get("hardware_monitor", True):
            if not self.hw_thread.isRunning():
                self.hw_thread.start()
        else:
            self.hw_thread.stop()
        if self.api_check.isChecked() and not self.api_server.running:
            ok, msg = self.api_server.start()
            self._update_api_status(ok, msg)
        elif not self.api_check.isChecked() and self.api_server.running:
            self.api_server.stop()
            self._update_api_status(False, "API 服务已停止")
        self.apply_theme()
        global UI_SCALE
        try:
            UI_SCALE = float(cfg.get("ui_scale", 1.0) or 1.0)
        except (TypeError, ValueError):
            UI_SCALE = 1.0
        app = QApplication.instance()
        if app is not None:
            app.setFont(QFont("Microsoft YaHei UI",
                              max(8, round(9 * UI_SCALE))))
        QMessageBox.information(self, "保存成功",
                                "✅ 配置已保存并即时生效\n"
                                "界面缩放的像素级布局将在重启后完全生效")

    def _on_dl_source_changed(self, text) -> None:
        """设置页下载源变更:实时保存 + 同步商店页。"""
        cfg.set("download_source", text)
        save_config(cfg)
        if hasattr(self, "download_src"):
            idx = self.download_src.findText(text)
            if idx >= 0:
                self.download_src.blockSignals(True)
                self.download_src.setCurrentIndex(idx)
                self.download_src.blockSignals(False)

    def on_device_sel_changed(self, text) -> None:
        """顶部「推理设备」下拉框变更:实时保存 + 同步设置页,
        保证用户的选择真正生效(加载时以此为准)。"""
        if not text:
            return
        cfg.set("prefer_device", text)
        save_config(cfg)
        if hasattr(self, "prefer_device_sel"):
            idx = self.prefer_device_sel.findText(text)
            if idx >= 0:
                self.prefer_device_sel.blockSignals(True)
                self.prefer_device_sel.setCurrentIndex(idx)
                self.prefer_device_sel.blockSignals(False)

    def _probe_dl_nodes(self) -> None:
        """自检各下载节点可达性,显示 QMessageBox 结果。"""
        self.chat_box.append("⏳ 正在自检下载节点可达性...\n")
        worker = SimpleWorker(modelstore.probe_download_sources)
        worker.ok.connect(self._on_dl_probe_done)
        registry.register(worker)
        worker.start()

    def _on_dl_probe_done(self, results: object) -> None:
        lines = ["📊 下载节点自检结果:"]
        for r in results if isinstance(results, list) else []:
            mark = "✅" if r.get("ok") else "❌"
            lat = f"{r.get('latency_ms', '超时')}ms" if r.get("ok") else "不可达"
            lines.append(f"  {mark} {r['label']}: {lat}")
        self.chat_box.append("\n".join(lines) + "\n")

    def _on_fb_toggle(self, state) -> None:
        """悬浮球开关实时生效。"""
        enabled = bool(state)
        cfg.set("floating_ball_enable", enabled)
        save_config(cfg)
        self.floating_ball.set_enabled(enabled)

    # ==================== 系统重置 ====================
    def on_reset_defaults(self) -> None:
        """恢复默认设置:仅重置配置项,保留模型与对话。需两次确认防误触。"""
        reply = QMessageBox.question(
            self, "恢复默认设置",
            "将把所有设置项(引擎、设备、生成参数、主题、镜像等)还原为默认值。\n"
            "已下载的模型和对话记录会保留。\n\n确定继续吗?")
        if reply != QMessageBox.StandardButton.Yes:
            return
        # 停止正在进行的生成,避免重置中途状态错乱
        if self._generating:
            self.on_stop_generation()
        cfg.reset_defaults()
        save_config(cfg)
        self._reload_settings_ui()
        # 应用默认主题/缩放/设备
        self.apply_theme()
        self.device_sel.setCurrentText(str(cfg.get("prefer_device", "CPU")))
        QMessageBox.information(
            self, "已恢复默认设置",
            "✅ 所有设置已还原为默认值(模型与对话保留)。\n"
            "界面缩放等部分改动重启后完全生效。")

    def on_factory_reset(self) -> None:
        """恢复出厂设置:清空全部数据。要求输入确认词,彻底防止误触。"""
        confirm_word = "FACTORY"
        text, ok = QInputDialog.getText(
            self, "恢复出厂设置(高危)",
            "此操作将永久删除以下全部数据,且不可恢复:\n"
            "  • 所有设置配置\n"
            "  • 已下载的本地模型\n"
            "  • 全部对话历史记录\n"
            "  • LoRA 微调产物与数据集\n"
            "  • 日志与缓存\n\n"
            f"如确认清空,请输入 {confirm_word} 后回车:")
        if not ok or text.strip() != confirm_word:
            if ok:
                QMessageBox.information(self, "已取消",
                                        "确认词不匹配,已取消出厂重置。")
            return
        # 二次确认
        reply = QMessageBox.question(
            self, "最终确认",
            "真的要清空全部数据吗?此操作不可恢复!")
        if reply != QMessageBox.StandardButton.Yes:
            return
        # 先停止一切占用文件的组件
        if self._generating:
            self.on_stop_generation()
        try:
            if self.api_server.running:
                self.api_server.stop()
        except Exception:
            pass
        try:
            engine_mgr.unload_all()
        except Exception:
            pass
        ok_wipe, errors = paths.wipe_all_data()
        # 重新初始化配置与会话(数据目录已被清空重建)
        cfg.reset_defaults()
        save_config(cfg)
        try:
            self.chat_session.store = ConversationStore(paths.SESSIONS_FILE)
            self.chat_session.new_session()
        except Exception:
            pass
        self.refresh_history_list()
        self._render_chat_messages()
        self.scan_local_models_gui()
        if errors:
            QMessageBox.warning(
                self, "出厂重置完成(有少量文件未能删除)",
                "已清空数据,但以下文件可能被占用而未删除(重启后可手动删除 "
                f"novacore_data 目录):\n" + "\n".join(errors[:10]))
        else:
            QMessageBox.information(
                self, "出厂重置完成",
                "✅ 已清空全部数据并恢复出厂状态。\n建议重启应用以使所有设置完全生效。")

    def _reload_settings_ui(self) -> None:
        """把当前 cfg 的值回填到设置页所有控件(重置后调用)。"""
        def set_combo(combo, value):
            idx = combo.findText(str(value))
            if idx >= 0:
                combo.setCurrentIndex(idx)

        def set_combo_data(combo, value):
            idx = combo.findData(value)
            if idx >= 0:
                combo.setCurrentIndex(idx)

        set_combo(self.engine_sel, cfg.get("default_engine", "auto"))
        set_combo(self.prefer_device_sel, cfg.get("prefer_device", "CPU"))
        self.ollama_host_edit.setText(str(cfg.get("ollama_host", "")))
        self.ollama_auto_check.setChecked(bool(cfg.get("ollama_auto_start", False)))
        set_combo_data(self.pip_mirror_sel, cfg.get("pip_mirror", "auto"))
        set_combo_data(self.ui_scale_sel, float(cfg.get("ui_scale", 1.0) or 1.0))
        self.temp_slider.setValue(int(float(cfg.get("gen_temperature", 0.7)) * 100))
        self.topp_spin.setValue(float(cfg.get("gen_top_p", 0.9)))
        self.topk_spin.setValue(int(cfg.get("gen_top_k", 40)))
        self.maxtok_spin.setValue(int(cfg.get("gen_max_tokens", 2048)))
        self.ctx_spin.setValue(int(cfg.get("gen_ctx_len", 4096)))
        self.sysopt_edit.setText(str(cfg.get("system_prompt", "")))
        self.port_spin.setValue(int(cfg.get("api_port", 8000)))
        self.api_check.setChecked(bool(cfg.get("api_enable", False)))
        set_combo(self.theme_sel, cfg.get("theme", "soft_dark"))
        self.manifest_edit.setText(str(cfg.get("store_manifest_url", "")))
        self.hwmon_check.setChecked(bool(cfg.get("hardware_monitor", True)))
        self.update_url.setText(str(cfg.get("update_source_url", "")))
        self.custom_update.setChecked(bool(cfg.get("custom_update_enable", False)))

    # ==================== 硬件/探测/主题 ====================
    def on_hw_stat(self, stat: dict) -> None:
        # 左下角硬件监控:紧凑格式,适配左侧窄面板
        ram_pct = (stat.get('ram_used', 0) / max(stat.get('ram_total', 1), 1)) * 100
        txt = f"💻 RAM {stat.get('ram_used', 0):.1f}/{stat.get('ram_total', 0):.0f}GB ({ram_pct:.0f}%)"
        txt += f"  · CPU {stat.get('cpu', 0):.0f}%"
        if "gpu_util" in stat:
            txt += (f"\n🎮 GPU {stat.get('gpu_util', 0)}%"
                    f"  VRAM {stat.get('gpu_vram_used', 0):.1f}/"
                    f"{stat.get('gpu_vram_total', 0):.1f}GB")
        # NPU 为静态探测结果,仅首次探测后显示
        npu_info = getattr(self, '_npu_info', None)
        if npu_info:
            mark = "✅" if npu_info.get('available') else "❌"
            txt += f"\n🧠 NPU {mark}"
        self.stat_lab.setText(txt)
        # 同步更新悬浮球面板内的硬件摘要
        if hasattr(self, "floating_ball"):
            self.floating_ball.update_hw(stat)

    def on_probe_done(self, results: list) -> None:
        parts = []
        for r in results:
            mark = "✅" if r.get("available") else "❌"
            parts.append(f"{r.get('display_name', r.get('id', '?'))}: {mark}")
        self.engine_lab.setText("引擎状态 | " + "  ".join(parts))

    def apply_theme(self) -> None:
        theme = str(cfg.get("theme", "soft_dark"))
        palettes = {
            "soft_dark": dict(win="#24262b", card="#1e2028", border="#333640",
                              text="#e8e8ed", sub="#aab"),
            "light": dict(win="#f3f4f8", card="#ffffff", border="#d6d9e0",
                          text="#22242a", sub="#667"),
            "deep_black": dict(win="#0b0c10", card="#13151a", border="#24262e",
                               text="#dfe2ea", sub="#8a90a0"),
        }
        c = palettes.get(theme, palettes["soft_dark"])
        self.setStyleSheet(f"""
QMainWindow, QWidget {{ background:{c['win']}; color:{c['text']};
    font-family:"Microsoft YaHei UI"; }}
#chatTop, #chatInput {{ background:{c['card']}; border-radius:10px; }}
QGroupBox {{ border:1px solid {c['border']}; border-radius:10px;
    margin-top:12px; color:{c['sub']}; font-weight:600; }}
QGroupBox::title {{ subcontrol-origin:margin; left:16px; padding:0 8px; }}
QPushButton {{ background:#363942; border:none; border-radius:8px;
    padding:8px 16px; color:#f0f0f5; }}
QPushButton:hover {{ background:#444854; }}
QPushButton:disabled {{ background:#26282f; color:#667; }}
QLineEdit, QTextEdit, QListWidget, QComboBox, QSpinBox, QDoubleSpinBox {{
    background:{c['card']}; border:1px solid {c['border']}; border-radius:8px;
    padding:6px 10px; color:{c['text']}; }}
QTableWidget {{ background:{c['card']}; border:1px solid {c['border']};
    border-radius:8px; gridline-color:{c['border']}; }}
QHeaderView::section {{ background:{c['card']}; border:none; padding:6px;
    color:{c['text']}; }}
QScrollBar:vertical {{ width:8px; background:transparent; }}
QScrollBar::handle:vertical {{ background:#444854; border-radius:4px; }}
QCheckBox {{ spacing:8px; }}
QProgressBar {{ background:{c['card']}; border:1px solid {c['border']};
    border-radius:6px; text-align:center; color:{c['text']}; }}
QProgressBar::chunk {{ background:#2563eb; border-radius:5px; }}
        """)
        for i, btn in enumerate(self.nav_btns):
            btn.setChecked(i == self.stack.currentIndex())

    # ==================== 启动自检 ====================
    def run_startup_selfcheck(self) -> None:
        """非阻塞启动自检:后台记录硬件/依赖后,缺失关键依赖给出提示。"""
        worker = SimpleWorker(startup_selfcheck)
        worker.ok.connect(self._on_selfcheck_done)
        worker.err.connect(lambda e: None)  # 错误已写入日志,不打扰用户
        registry.register(worker)
        worker.start()

    def _on_selfcheck_done(self, result: object) -> None:
        missing = (result.get("missing_required", [])
                   if isinstance(result, dict) else [])
        if missing:
            QMessageBox.warning(
                self, "依赖缺失",
                "检测到缺失关键依赖:\n" + ", ".join(missing)
                + "\n请到「依赖管理」页一键补全。")

    # ==================== 退出清理 ====================
    def _setup_floating_ball(self) -> None:
        """初始化悬浮球:显示、置顶、定位到右下角。"""
        enabled = bool(cfg.get("floating_ball_enable", True))
        if enabled:
            self.floating_ball.show()
            self.floating_ball.raise_()
            self.floating_ball._reposition()
            # 确保展开面板的 parent 也是 MainWindow(否则会被 centralWidget 遮挡)
            self.floating_ball._panel.setParent(self)
            self.floating_ball._panel.raise_()
        else:
            self.floating_ball.hide()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if hasattr(self, "floating_ball"):
            self.floating_ball._reposition()
            self.floating_ball.raise_()

    def closeEvent(self, event) -> None:  # noqa: N802
        self._closing = True
        try:
            self.chat_session.save_history()
        except Exception:
            pass
        if self._generating:
            self.on_stop_generation()
        if self._loading:
            worker = getattr(self, "_load_worker", None)
            if worker is not None:
                worker.wait(5000)
        try:
            registry.shutdown()
        except Exception:
            pass
        try:
            self.api_server.stop()
        except Exception:
            pass
        save_config(cfg)
        super().closeEvent(event)
        event.accept()


# ==================== 模块级单例 ====================
cfg = load_config()
engine_mgr = get_manager(ollama_host=cfg.get("ollama_host"))
registry = WorkerRegistry()


def _npu_compile_cli(model_dir: str, blob_path: str) -> None:
    """打包版子进程入口:在 NPU 上编译模型并导出 blob(由引擎 spawn 自己调用)。

    成功退出码 0,失败非 0;windowed exe 无控制台,不写任何 stdout,
    错误写入日志文件供排查。
    """
    import logging
    logger = logging.getLogger("novacore")
    try:
        logger.info("NPU 编译子进程开始: %s", model_dir)
        import openvino_genai as ovg
        pipe = ovg.LLMPipeline(
            model_dir, "NPU",
            EXPORT_BLOB="YES", BLOB_PATH=blob_path,
            CACHE_MODE="OPTIMIZE_SPEED")
        del pipe
        logger.info("NPU 编译完成: %s", blob_path)
    except Exception:
        logger.exception("NPU 编译子进程失败")
        os._exit(1)
    os._exit(0)


def main() -> None:
    setup_logging()
    install_excepthook()
    # 关闭 OpenVINO 遥测(默认会向 Google Analytics 上报,且其发送线程在
    # 受限网络下会卡死解释器退出;见 hardware.ensure_telemetry_optout)
    hardware.ensure_telemetry_optout()
    # PyInstaller 打包版的 NPU 编译子进程入口(无 GUI,编译完即退出)
    if len(sys.argv) >= 4 and sys.argv[1] == "--npu-compile":
        _npu_compile_cli(sys.argv[2], sys.argv[3])
        return
    global UI_SCALE
    try:
        UI_SCALE = float(cfg.get("ui_scale", 1.0) or 1.0)
    except (TypeError, ValueError):
        UI_SCALE = 1.0
    UI_SCALE = min(max(UI_SCALE, 0.5), 3.0)
    app = QApplication(sys.argv)
    app.setFont(QFont("Microsoft YaHei UI", max(8, round(9 * UI_SCALE))))
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)
    # 窗口/任务栏图标(与打包 exe 图标一致;缺失时静默降级)
    try:
        if getattr(sys, "frozen", False):
            base = getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
        else:
            base = os.path.dirname(os.path.abspath(__file__))
        icon_path = os.path.join(base, "novacore.ico")
        if os.path.exists(icon_path):
            app.setWindowIcon(QIcon(icon_path))
    except Exception:
        pass
    w = MainWindow()
    w.show()
    w.run_startup_selfcheck()
    sys.exit(app.exec())
    # app.exec() 返回后进程可能被 OpenVINO 插件的卸载流程挂起(实测在
    # 创建过 ov.Core() 的机器上,解释器退出阶段会卡住数十秒甚至永久),
    # 主窗口 closeEvent 已完成全部清理(会话保存/线程停止/配置落盘),
    # 这里直接硬退出,避免用户关闭窗口后进程仍在后台占资源。
    os._exit(0)


if __name__ == "__main__":
    main()
