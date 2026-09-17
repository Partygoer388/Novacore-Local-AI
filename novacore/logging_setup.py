"""全局日志与异常钩子:
- setup_logging():根 logger 落到 logs 目录(滚动),同时输出到控制台
- install_excepthook():捕获未处理异常(主线程/子线程/Qt 消息),写入日志避免静默崩溃
- startup_selfcheck():启动自检,记录系统信息与依赖缺失,依赖不阻塞主流程
"""
from __future__ import annotations

import logging
import sys
import threading
from logging.handlers import RotatingFileHandler
from typing import Optional

from . import APP_NAME, APP_VERSION
from .paths import LOGS_DIR

LOGGER_NAME = "novacore"

# 持有 Qt 消息处理器引用,防止被垃圾回收
_qt_handler_ref = None


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """幂等:已配置则直接返回。日志写入 logs/novacore.log。"""
    try:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    logger = logging.getLogger(LOGGER_NAME)
    if logger.handlers:
        return logger
    logger.setLevel(level)

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)s [%(name)s] %(message)s")
    try:
        fh = RotatingFileHandler(
            LOGS_DIR / "novacore.log", maxBytes=2_000_000, backupCount=3,
            encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    except OSError:
        pass  # 只读介质等:降级为仅控制台
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.propagate = False
    logger.info("%s %s 启动", APP_NAME, APP_VERSION)
    return logger


def install_excepthook(logger: Optional[logging.Logger] = None) -> None:
    """安装全局异常钩子:主线程、子线程、Qt 消息统一写入日志。不改变异常语义。"""
    logger = logger or logging.getLogger(LOGGER_NAME)

    def _hook(exc_type, exc_value, exc_tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return
        logger.critical(
            "未处理异常(主线程):",
            exc_info=(exc_type, exc_value, exc_tb))
    sys.excepthook = _hook

    def _thread_hook(args):
        logger.critical(
            "未处理异常(子线程):",
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback))
    threading.excepthook = _thread_hook

    try:
        from PyQt6.QtCore import QtMsgType, qInstallMessageHandler
    except ImportError:
        return

    _levels = {
        QtMsgType.QtDebugMsg: logging.DEBUG,
        QtMsgType.QtInfoMsg: logging.INFO,
        QtMsgType.QtWarningMsg: logging.WARNING,
        QtMsgType.QtCriticalMsg: logging.ERROR,
        QtMsgType.QtFatalMsg: logging.CRITICAL,
    }

    def _qt_hook(mode, context, message):
        logger.log(_levels.get(mode, logging.INFO), f"Qt: {message}")

    global _qt_handler_ref
    _qt_handler_ref = _qt_hook
    qInstallMessageHandler(_qt_hook)


def startup_selfcheck(logger: Optional[logging.Logger] = None) -> dict:
    """非阻塞启动自检:记录硬件信息与依赖缺失。返回缺失的关键依赖列表(供 UI 提示)。"""
    logger = logger or logging.getLogger(LOGGER_NAME)
    result = {"missing_required": [], "missing_optional": []}
    try:
        from . import hardware
        info = hardware.get_system_info()
        logger.info("系统信息: %s", info.get("os"))
        gpus = info.get("gpus") or []
        for g in gpus:
            logger.info("GPU: %s (显存 %s GB)", g.get("name"), g.get("vram_total_gb"))
        npu = info.get("npu") or {}
        logger.info("NPU: %s", npu.get("detail"))
    except Exception as e:  # noqa: BLE001
        logger.warning("硬件探测失败: %s", e)

    try:
        from . import deps
        for dep in deps.check_deps():
            if dep["installed"]:
                continue
            if dep["optional"]:
                result["missing_optional"].append(dep["key"])
                logger.info("可选依赖缺失: %s (%s)", dep["key"], dep["name"])
            else:
                result["missing_required"].append(dep["key"])
                logger.warning("关键依赖缺失: %s (%s)", dep["key"], dep["name"])
    except Exception as e:  # noqa: BLE001
        logger.warning("依赖检测失败: %s", e)

    if result["missing_required"]:
        logger.warning("存在缺失关键依赖: %s", ", ".join(result["missing_required"]))
    else:
        logger.info("启动自检完成:关键依赖齐备")
    return result