"""统一路径管理:所有数据目录锚定到可写目录。

- 源码运行:锚定到 novacore_main.py 所在目录(novacore/ 的上一级)
- PyInstaller 打包:锚定到 exe 所在目录(通过 sys.frozen / sys._MEIPASS 检测)
绝不依赖当前工作目录。
"""
import sys
from pathlib import Path


def _resolve_app_root() -> Path:
    """解析应用根目录(数据目录 novacore_data 将放在此目录下)。"""
    if getattr(sys, "frozen", False):
        # PyInstaller 打包:sys.executable 是 exe 完整路径
        return Path(sys.executable).resolve().parent
    # 源码运行:novacore/paths.py → novacore → novacore_main.py 所在目录
    return Path(__file__).resolve().parent.parent


APP_ROOT = _resolve_app_root()

WORK_DIR = APP_ROOT / "novacore_data"
LOCAL_MODEL_DIR = WORK_DIR / "local_models"
LORAS_DIR = WORK_DIR / "loras"
DATASETS_DIR = WORK_DIR / "datasets"
LOGS_DIR = WORK_DIR / "logs"
CONFIG_FILE = WORK_DIR / "config.json"
HISTORY_FILE = WORK_DIR / "chat_history.json"
SESSIONS_FILE = WORK_DIR / "chat_sessions.json"
MODELS_MANIFEST_FILE = WORK_DIR / "manifest.json"

ALL_DIRS = [WORK_DIR, LOCAL_MODEL_DIR, LORAS_DIR, DATASETS_DIR, LOGS_DIR]


def ensure_dirs() -> None:
    for d in ALL_DIRS:
        try:
            d.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass  # 只读介质等极端情况不阻塞启动


def resolve_model_path(name: str) -> Path:
    """把模型名解析为本地模型目录下的绝对路径,防止路径穿越。"""
    p = (LOCAL_MODEL_DIR / name).resolve()
    root = LOCAL_MODEL_DIR.resolve()
    if root not in p.parents and p != root:
        raise ValueError(f"非法模型路径: {name}")
    return p


def wipe_all_data() -> tuple[bool, list[str]]:
    """出厂重置:清空全部用户数据(配置/模型/对话/LoRA/数据集/日志/缓存)。

    返回 (是否完全成功, 失败项说明列表)。目录会被清空后重建,保留 novacore_data
    目录本身。调用方应先停止模型/API 等占用文件的组件。
    """
    import shutil
    errors: list[str] = []
    for d in ALL_DIRS:
        if not d.exists():
            continue
        try:
            for child in d.iterdir():
                try:
                    if child.is_dir():
                        shutil.rmtree(child, ignore_errors=False)
                    else:
                        child.unlink()
                except OSError as e:
                    errors.append(f"{child.name}: {e}")
        except OSError as e:
            errors.append(f"{d}: {e}")
    ensure_dirs()
    return (len(errors) == 0), errors


ensure_dirs()
