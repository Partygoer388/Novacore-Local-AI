"""配置管理:
- 缺失键自动用默认值补齐,旧配置平滑升级
- 类型强校验 + 非法值回退默认,损坏 JSON 自动备份并重建
- 原子写入(临时文件 + os.replace),多线程安全
"""
import json
import os
import shutil
import tempfile
import threading
from typing import Any, Callable, Optional

from .paths import CONFIG_FILE

# 可用的推理引擎与下载源
ENGINE_CHOICES = ["auto", "llamacpp", "ollama", "openvino"]
DOWNLOAD_SOURCE_CHOICES = ["auto", "official", "hf_mirror", "modelscope"]
THEME_CHOICES = ["soft_dark", "light", "deep_black"]
# 界面缩放档位(数值为缩放系数,存到配置为 float)
UI_SCALE_CHOICES = {"默认 100%": 1.0, "紧凑 90%": 0.9, "放大 110%": 1.1,
                    "放大 125%": 1.25, "放大 150%": 1.5}

# (类型, 默认值, 校验函数(返回 True/False))
CONFIG_SPEC: dict[str, tuple[type, Any, Optional[Callable[[Any], bool]]]] = {
    "api_port": (int, 8000, lambda v: 1024 <= v <= 65535),
    "api_enable": (bool, False, None),
    "default_engine": (str, "auto", lambda v: v in ENGINE_CHOICES),
    "prefer_device": (str, "CPU", lambda v: v in ("CPU", "GPU", "NPU-OpenVINO", "NPU", "CUDA")),
    "update_source_url": (str, "", None),
    "store_manifest_url": (str, "", None),
    "custom_update_enable": (bool, False, None),
    "current_loaded_model": (str, "", None),
    "download_source": (str, "auto", lambda v: v in DOWNLOAD_SOURCE_CHOICES),
    "theme": (str, "soft_dark", lambda v: v in THEME_CHOICES),
    "ollama_host": (str, "http://127.0.0.1:11434", None),
    "ollama_auto_start": (bool, False, None),
    "hardware_monitor": (bool, True, None),
    "pip_mirror": (str, "auto", lambda v: v in ["auto", "official", "tsinghua",
        "aliyun", "douban", "ustc", "huawei", "tencent", "netease", "sjtu"]),
    "ui_scale": (float, 1.0, lambda v: 0.5 <= v <= 3.0),
    # 生成参数
    "gen_temperature": (float, 0.7, lambda v: 0.0 <= v <= 2.0),
    "gen_top_p": (float, 0.9, lambda v: 0.0 < v <= 1.0),
    "gen_top_k": (int, 40, lambda v: 0 <= v <= 1000),
    "gen_max_tokens": (int, 2048, lambda v: 1 <= v <= 32768),
    "gen_ctx_len": (int, 4096, lambda v: 256 <= v <= 131072),
    "max_history": (int, 200, lambda v: 1 <= v <= 2000),
    "system_prompt": (str, "你是一个乐于助人的中文AI助手,回答简洁、准确。", None),
    "floating_ball_enable": (bool, True, None),
}

_EXTRA_PRESERVED_KEYS = {"theme", "glow_ui", "token_stat_enable"}  # 旧版遗留键,保留不丢弃


class Config:
    """线程安全的配置容器。"""

    def __init__(self, data: dict[str, Any]):
        self._lock = threading.RLock()
        self._data: dict[str, Any] = {}
        self.update(data)

    # ---------- 读写 ----------
    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = value

    def as_dict(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._data)

    def update(self, data: dict[str, Any]) -> None:
        with self._lock:
            self._data.update({k: v for k, v in data.items() if k in CONFIG_SPEC
                               or k in _EXTRA_PRESERVED_KEYS})

    def replace(self, data: dict[str, Any]) -> None:
        """用给定(已清洗)数据整体替换内部配置,对象引用不变。"""
        with self._lock:
            self._data = dict(sanitize(data))

    def reset_defaults(self) -> None:
        """原地恢复为全部默认值(对象引用不变,保证持有方同步生效)。"""
        self.replace({})


def _coerce(spec_type: type, value: Any) -> Any:
    """尽力把 value 转换为目标类型,失败返回 None。"""
    if isinstance(value, spec_type):
        return value
    if spec_type is bool:
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        if isinstance(value, (int, float)):
            return bool(value)
        return None
    if spec_type is int:
        try:
            if isinstance(value, float) and not value.is_integer():
                return None
            return int(value)
        except (TypeError, ValueError):
            return None
    if spec_type is float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    if spec_type is str:
        if value is None:
            return None
        return str(value)
    return None


def sanitize(raw: dict[str, Any]) -> dict[str, Any]:
    """逐键校验:类型强制转换,非法值回退默认,缺失键补默认。"""
    out: dict[str, Any] = {}
    raw = raw if isinstance(raw, dict) else {}
    for key, (typ, default, validator) in CONFIG_SPEC.items():
        value = _coerce(typ, raw.get(key))
        if value is None:
            out[key] = default
            continue
        if validator is not None and not validator(value):
            out[key] = default
            continue
        out[key] = value
    # 保留对运行无害的遗留键(如旧版 token_stat_enable)
    for key in _EXTRA_PRESERVED_KEYS:
        if key in raw and key not in out:
            out[key] = raw[key]
    return out


def _load_raw() -> dict[str, Any]:
    if not CONFIG_FILE.exists():
        return {}
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("config root is not an object")
        return data
    except Exception:
        # 损坏配置:备份后重建
        try:
            shutil.copy2(CONFIG_FILE, CONFIG_FILE.with_suffix(".json.bak"))
        except OSError:
            pass
        return {}


def load_config() -> Config:
    """读取并清洗配置;第一次运行会落盘一份默认配置。"""
    cfg = Config(sanitize(_load_raw()))
    save_config(cfg)
    return cfg


def save_config(cfg: Config) -> bool:
    """原子写入:先写临时文件再替换,避免写一半损坏。"""
    data = cfg.as_dict()
    tmp_path = None
    try:
        CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            dir=str(CONFIG_FILE.parent), prefix=".config_", suffix=".tmp")
        tmp_path = tmp_name
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, CONFIG_FILE)
        return True
    except OSError:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        return False


def default_config() -> Config:
    """返回一份全新的默认配置(全部键取默认值)。"""
    return Config(sanitize({}))


def reset_to_defaults() -> Config:
    """把配置文件重置为出厂默认值(不影响模型/对话等其它数据)并落盘。"""
    cfg = default_config()
    save_config(cfg)
    return cfg


def ensure_default_config() -> Config:
    return load_config()
