"""GGUF 模型文件校验与元数据解析:
- 识别假模型(如把网页/HTML 当模型下载下来的文件)并给出原因
- 解析 GGUF 头与元数据(模型名/架构/上下文长度),供引擎加载与界面展示
"""
import struct
from pathlib import Path
from typing import Any, Optional

from .paths import LOCAL_MODEL_DIR

GGUF_MAGIC = b"GGUF"
MIN_MODEL_BYTES = 1_000_000  # 低于 1MB 的“模型”一律视为无效(真实 GGUF 至少几十 MB)

HTML_MARKERS = (b"<!doctype html", b"<html", b"<!DOCTYPE HTML")
HTML_HINT = "文件内容为网页(HTML),这是占位/错误下载链接的结果,不是模型文件,请删除后从真实模型源重新下载"
DIR_HINT = "文件夹,不是模型文件"
UNKNOWN_HINT = "无法识别的格式,当前仅支持 GGUF 模型"


def _read_string(f) -> str:
    (n,) = struct.unpack("<Q", f.read(8))
    return f.read(n).decode("utf-8", errors="replace")


def _read_value(f, vtype: int) -> Any:
    if vtype == 0:   return struct.unpack("<B", f.read(1))[0]
    if vtype == 1:   return struct.unpack("<b", f.read(1))[0]
    if vtype == 2:   return struct.unpack("<H", f.read(2))[0]
    if vtype == 3:   return struct.unpack("<h", f.read(2))[0]
    if vtype == 4:   return struct.unpack("<I", f.read(4))[0]
    if vtype == 5:   return struct.unpack("<i", f.read(4))[0]
    if vtype == 6:   return struct.unpack("<f", f.read(4))[0]
    if vtype == 7:   return struct.unpack("<?", f.read(1))[0]
    if vtype == 8:   return _read_string(f)
    if vtype == 9:
        (elem_type,) = struct.unpack("<I", f.read(4))
        (n,) = struct.unpack("<Q", f.read(8))
        return [_read_value(f, elem_type) for _ in range(n)]
    if vtype == 10:  return struct.unpack("<Q", f.read(8))[0]
    if vtype == 11:  return struct.unpack("<q", f.read(8))[0]
    if vtype == 12:  return struct.unpack("<d", f.read(8))[0]
    raise ValueError(f"未知 GGUF 值类型: {vtype}")


def parse_gguf_header(path) -> dict:
    """解析 GGUF 文件头,返回 {version, tensor_count, metadata}。非 GGUF 抛 ValueError。"""
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != GGUF_MAGIC:
            raise ValueError("不是 GGUF 文件(魔数不匹配)")
        (version,) = struct.unpack("<I", f.read(4))
        if version not in (2, 3):
            raise ValueError(f"不支持的 GGUF 版本: {version}")
        n_tensors, n_kv = struct.unpack("<QQ", f.read(16))
        metadata: dict[str, Any] = {}
        for _ in range(min(n_kv, 10000)):
            key = _read_string(f)
            (vtype,) = struct.unpack("<I", f.read(4))
            metadata[key] = _read_value(f, vtype)
    return {"version": version, "tensor_count": n_tensors, "metadata": metadata}


def _meta_first(meta: dict[str, Any], *keys: str) -> Optional[Any]:
    for k in keys:
        if k in meta:
            return meta[k]
    return None


def is_gguf_file(path) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(4) == GGUF_MAGIC
    except OSError:
        return False


def validate_model_file(path) -> tuple[bool, str, dict]:
    """校验本地模型文件。返回 (是否有效, 原因, 附加信息)。"""
    p = Path(path)
    info: dict = {}
    if not p.exists():
        return False, "文件不存在", info
    if p.is_dir():
        return False, DIR_HINT, info
    size = p.stat().st_size
    info["size_bytes"] = size
    if size == 0:
        return False, "文件为空", info
    try:
        with open(p, "rb") as f:
            head = f.read(64)
    except OSError as e:
        return False, f"无法读取文件: {e}", info
    # 先识别网页等明显假文件(即使体积很小也能给出准确原因)
    low = head.lower()
    if any(m in low for m in HTML_MARKERS):
        return False, HTML_HINT, info
    if size < MIN_MODEL_BYTES:
        return False, f"文件过小({size} 字节),疑似无效文件;真实 GGUF 模型至少数十 MB", info
    if head[:4] != GGUF_MAGIC:
        return False, UNKNOWN_HINT, info
    try:
        parsed = parse_gguf_header(p)
    except Exception as e:
        return False, f"GGUF 头解析失败(文件可能损坏): {e}", info
    meta = parsed["metadata"]
    info["gguf_version"] = parsed["version"]
    info["tensor_count"] = parsed["tensor_count"]
    info["model_name"] = str(_meta_first(meta, "general.name") or p.stem)
    info["architecture"] = str(_meta_first(meta, "general.architecture") or "unknown")
    ctx = _meta_first(meta, "llama.context_length", "general.context_length")
    if isinstance(ctx, list) and ctx:
        ctx = ctx[0]
    info["context_length"] = int(ctx) if isinstance(ctx, int) else None
    if parsed["tensor_count"] == 0:
        return False, "GGUF 内没有任何张量(空模型)", info
    return True, "", info


def is_openvino_dir(path) -> bool:
    """判断目录是否为 OpenVINO IR 模型目录(含 openvino_model.xml)。"""
    p = Path(path)
    return p.is_dir() and (p / "openvino_model.xml").is_file()


def scan_local_models(directory: Optional[Path] = None) -> list[dict]:
    """扫描本地模型目录,返回 GGUF 文件与 OpenVINO IR 目录及校验结果。"""
    directory = Path(directory) if directory else LOCAL_MODEL_DIR
    items: list[dict] = []
    try:
        entries = sorted(directory.iterdir(), key=lambda e: e.name)
    except OSError:
        return items
    for entry in entries:
        if entry.is_file():
            ok, reason, info = validate_model_file(entry)
            items.append({
                "name": entry.name,
                "path": str(entry),
                "format": "gguf",
                "size_kb": round(entry.stat().st_size / 1024, 2),
                "valid": ok,
                "reason": reason,
                "info": info,
            })
        elif entry.is_dir():
            if is_openvino_dir(entry):
                # 统计目录体积(xml/bin/tokenizer 全部文件,含 .part 半成品)
                all_files = [f for f in entry.rglob("*") if f.is_file()]
                total = sum(f.stat().st_size for f in all_files)
                bin_path = entry / "openvino_model.bin"
                bin_ok = bin_path.is_file() and bin_path.stat().st_size > 0
                # openvino-genai 2026.x+ 强制需要这四个文件
                missing_extra = []
                for extra in ("openvino_tokenizer.xml",
                              "openvino_tokenizer.bin",
                              "openvino_detokenizer.xml",
                              "openvino_detokenizer.bin"):
                    if not (entry / extra).is_file():
                        missing_extra.append(extra)
                reason = ""
                if not bin_ok:
                    reason = "缺少 openvino_model.bin"
                elif missing_extra:
                    reason = (f"⚠️ 缺少 {', '.join(missing_extra)}"
                              f"(openvino-genai 2026 必需,请在商店页重新下载该模型)")
                items.append({
                    "name": entry.name,
                    "path": str(entry),
                    "format": "openvino_dir",
                    "size_kb": round(total / 1024, 2),
                    "valid": bin_ok,
                    "reason": reason,
                    "info": {"model_name": entry.name,
                             "architecture": "openvino-ir",
                             "missing_tokenizer": bool(missing_extra)},
                })
            else:
                # 区分「下载未完成的半成品目录」与「无关空文件夹」,
                # 避免把下载残件误报成笼统的"文件夹,不是模型文件"
                part_files = list(entry.rglob("*.part"))
                if part_files:
                    part_size = sum(f.stat().st_size for f in part_files)
                    has_xml = (entry / "openvino_model.xml").is_file()
                    items.append({
                        "name": entry.name,
                        "path": str(entry),
                        "format": "openvino_dir",
                        "size_kb": round(part_size / 1024, 2),
                        "valid": False,
                        "reason": (
                            "⏳ 下载未完成"
                            f"({part_size/1024/1024:.1f}MB),"
                            "重新点击该模型可断点续传,或删除后重下"),
                        "info": {"model_name": entry.name,
                                 "architecture": "openvino-ir"
                                 if has_xml else "partial"},
                    })
                else:
                    items.append({
                        "name": entry.name,
                        "path": str(entry), "format": "dir",
                        "size_kb": 0,
                        "valid": False, "reason": DIR_HINT, "info": {},
                    })
    return items
