"""模型商店:内置真实 HuggingFace GGUF 清单 + 自定义 manifest 加载 + 国内镜像源映射。

清单条目结构:
{
    "name": "Qwen2.5-0.5B-Instruct",        # 显示名
    "category": "文本",                      # 文本/编程/多模态/配音
    "hw": ["CPU", "GPU"],                    # 支持的硬件
    "desc": "...",
    "size_gb": 0.4,
    "sources": {                             # 三种下载源(URL 或 None)
        "official":   "https://huggingface.co/.../resolve/main/xxx.gguf",
        "hf_mirror":  "https://hf-mirror.com/.../resolve/main/xxx.gguf",
        "modelscope": "https://modelscope.cn/models/.../resolve/master/xxx.gguf",
    },
}

manifest(自定义清单):JSON 数组(或 {"models": [...]})。每条目可写 sources 三源,
也可写 repo/file/ms_repo/ms_file 由程序生成 URL。缺 sources 的条目自动补生成。
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

import requests

from . import paths

HF_RESOLVE = "https://huggingface.co/{repo}/resolve/main/{file}"
HF_MIRROR_RESOLVE = "https://hf-mirror.com/{repo}/resolve/main/{file}"
MS_RESOLVE = "https://modelscope.cn/models/{repo}/resolve/master/{file}"

SOURCE_NAMES = {
    "official": "官方 HuggingFace",
    "hf_mirror": "hf-mirror.com 国内镜像",
    "modelscope": "ModelScope 魔搭",
}
SOURCE_ORDER = ["official", "hf_mirror", "modelscope"]


def build_sources(repo: str, file: str,
                  ms_repo: Optional[str] = None,
                  ms_file: Optional[str] = None) -> dict[str, Optional[str]]:
    """由仓库/文件名生成三种下载源 URL(modelscope 未提供时回退 hf_mirror)。"""
    return {
        "official": HF_RESOLVE.format(repo=repo, file=file),
        "hf_mirror": HF_MIRROR_RESOLVE.format(repo=repo, file=file),
        "modelscope": (MS_RESOLVE.format(repo=ms_repo, file=ms_file)
                       if ms_repo and ms_file else None),
    }


def _mk(name, category, hw, desc, size_gb, repo, file,
        ms_repo=None, ms_file=None):
    return {
        "name": name,
        "category": category,
        "hw": hw,
        "format": "gguf",
        "desc": desc,
        "size_gb": size_gb,
        "sources": build_sources(repo, file, ms_repo, ms_file),
    }


def _mk_ov(name, category, desc, size_gb, repo, ms_repo=None):
    """构造 OpenVINO IR 目录模型条目(NPU 专区,INT4 量化)。

    下载物为模型目录(openvino_model.xml/.bin + tokenizer 等多文件),
    由 OVDirectoryDownloadWorker 通过 resolve URL 逐文件拉取。
    """
    return {
        "name": name,
        "category": category,
        "hw": ["NPU"],
        "format": "openvino_dir",
        "desc": desc,
        "size_gb": size_gb,
        "repo": repo,
        "ms_repo": ms_repo,
    }


# 内置商店:真实 HuggingFace GGUF 直链(全部为官方仓库发布的量化模型)
# CPU/GPU 与 NPU 严格分区:GGUF 只标 CPU/GPU;
# NPU 专区使用 OpenVINO 官方 INT4 量化的 IR 目录模型(见 BUILTIN_NPU_STORE),
# 避免「选 NPU 却下载 GGUF 并被 llama.cpp 启动」的混乱。
BUILTIN_STORE: list[dict] = [
    _mk("Qwen2.5-0.5B-Instruct", "文本", ["CPU", "GPU"],
        "通义千问 0.5B 中文对话,超轻量,低配 CPU 可流畅运行",
        0.4, "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        "qwen2.5-0.5b-instruct-q4_k_m.gguf",
        "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        "qwen2.5-0.5b-instruct-q4_k_m.gguf"),
    _mk("Qwen2.5-1.5B-Instruct", "文本", ["CPU", "GPU"],
        "通义千问 1.5B 中文对话,体积与质量平衡之选",
        1.0, "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        "qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        "qwen2.5-1.5b-instruct-q4_k_m.gguf"),
    _mk("Qwen2.5-3B-Instruct", "文本", ["CPU", "GPU"],
        "通义千问 3B 中文对话,质量更佳",
        2.1, "Qwen/Qwen2.5-3B-Instruct-GGUF",
        "qwen2.5-3b-instruct-q4_k_m.gguf",
        "Qwen/Qwen2.5-3B-Instruct-GGUF",
        "qwen2.5-3b-instruct-q4_k_m.gguf"),
    _mk("Qwen2.5-Coder-0.5B-Instruct", "编程", ["CPU", "GPU"],
        "通义千问代码模型 0.5B,写代码/排错,轻量可跑",
        0.4, "Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF",
        "qwen2.5-coder-0.5b-instruct-q4_k_m.gguf",
        "Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF",
        "qwen2.5-coder-0.5b-instruct-q4_k_m.gguf"),
    _mk("Llama-3.2-1B-Instruct", "文本", ["CPU", "GPU"],
        "Meta Llama 3.2 1B 指令模型,英文为主,轻量友好",
        0.8, "bartowski/Llama-3.2-1B-Instruct-GGUF",
        "Llama-3.2-1B-Instruct-Q4_K_M.gguf"),
    _mk("Llama-3.2-3B-Instruct", "文本", ["CPU", "GPU"],
        "Meta Llama 3.2 3B 指令模型,多语言",
        2.0, "bartowski/Llama-3.2-3B-Instruct-GGUF",
        "Llama-3.2-3B-Instruct-Q4_K_M.gguf"),
    _mk("Phi-3-mini-4k-instruct", "文本", ["CPU", "GPU"],
        "微软 Phi-3-mini 3.8B,轻量高效",
        2.3, "microsoft/Phi-3-mini-4k-instruct-gguf",
        "Phi-3-mini-4k-instruct-q4.gguf"),
    _mk("Gemma-2-2B-it", "文本", ["CPU", "GPU"],
        "Google Gemma 2 2B 指令模型,轻量高效",
        1.7, "bartowski/gemma-2-2b-it-GGUF",
        "gemma-2-2b-it-Q4_K_M.gguf"),
    _mk("Qwen2.5-7B-Instruct", "文本", ["CPU", "GPU"],
        "通义千问 7B 中文对话,高质量(需较大显存/内存)",
        4.7, "Qwen/Qwen2.5-7B-Instruct-GGUF",
        "qwen2.5-7b-instruct-q4_k_m.gguf",
        "Qwen/Qwen2.5-7B-Instruct-GGUF",
        "qwen2.5-7b-instruct-q4_k_m.gguf"),
    _mk("DeepSeek-R1-Distill-Qwen-1.5B", "文本", ["CPU", "GPU"],
        "DeepSeek-R1 蒸馏版 1.5B,推理能力强",
        1.0, "unsloth/DeepSeek-R1-Distill-Qwen-1.5B-GGUF",
        "DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf"),
    _mk("SmolLM2-1.7B-Instruct", "文本", ["CPU", "GPU"],
        "HuggingFace SmolLM2 1.7B,超小体积,轻量可跑",
        1.1, "bartowski/SmolLM2-1.7B-Instruct-GGUF",
        "SmolLM2-1.7B-Instruct-Q4_K_M.gguf"),
]


# NPU 专区:OpenVINO 官方 NPU 优化 IR 模型(目录格式,原生 NPU 推理)。
# 仅收录 HuggingFace「LLMs optimized for NPU」合集中 -int4-gq-ov / -int4-cw-ov
# 仓库(对称 INT4 + 有界形状)——这是 openvino-genai 在 Intel NPU 上能真正
# 编译运行的格式。旧版 -int4-ov 仓库为非对称量化 + 动态形状,vpux 编译器
# 对其触发 LLVM ERROR 会直接杀死整个程序,严禁再放进 NPU 专区。
# 以下仓库均已在 hf-mirror 与 ModelScope 双源 tree API 实测存在(HTTP 200)。
BUILTIN_NPU_STORE: list[dict] = [
    _mk_ov("DeepSeek-R1-Distill-Qwen-1.5B-OV-NPU", "文本",
           "DeepSeek-R1 蒸馏 1.5B 对称 INT4(GQ128),NPU 原生推理,约 1.1GB",
           1.1, "OpenVINO/DeepSeek-R1-Distill-Qwen-1.5B-int4-gq-ov",
           ms_repo="OpenVINO/DeepSeek-R1-Distill-Qwen-1.5B-int4-gq-ov"),
    _mk_ov("Phi-3-mini-4k-instruct-OV-NPU", "文本",
           "微软 Phi-3-mini 3.8B 对称 INT4(GQ128),NPU 原生推理,约 2.0GB",
           2.0, "OpenVINO/Phi-3-mini-4k-instruct-int4-gq-ov",
           ms_repo="OpenVINO/Phi-3-mini-4k-instruct-int4-gq-ov"),
    _mk_ov("Phi-3.5-mini-instruct-OV-NPU", "文本",
           "微软 Phi-3.5-mini 对称 INT4(GQ128),NPU 原生推理,约 2.0GB",
           2.0, "OpenVINO/Phi-3.5-mini-instruct-int4-gq-ov",
           ms_repo="OpenVINO/Phi-3.5-mini-instruct-int4-gq-ov"),
    _mk_ov("Mistral-7B-Instruct-v0.3-OV-NPU", "文本",
           "Mistral 7B 对称 INT4(CW 通道级),高质量 NPU 模型,约 3.6GB(需较大内存)",
           3.6, "OpenVINO/Mistral-7B-Instruct-v0.3-int4-cw-ov",
           ms_repo="OpenVINO/Mistral-7B-Instruct-v0.3-int4-cw-ov"),
    _mk_ov("Qwen3-8B-OV-NPU", "文本",
           "通义千问 Qwen3 8B 对称 INT4(CW 通道级),高质量 NPU 模型,约 4.5GB(需较大内存)",
           4.5, "OpenVINO/Qwen3-8B-int4-cw-ov",
           ms_repo="OpenVINO/Qwen3-8B-int4-cw-ov"),
]


def get_builtin_store() -> list[dict]:
    """内置清单 = CPU/GPU 的 GGUF + NPU 专区的 OpenVINO IR 目录模型。"""
    out = [dict(m, sources=dict(m["sources"])) for m in BUILTIN_STORE]
    out.extend(dict(m) for m in BUILTIN_NPU_STORE)
    return out


# ==================== OpenVINO IR 目录模型文件解析 ====================
# OpenVINO GenAI 标准模型目录必需/常见文件(LLMPipeline 加载所需)
# openvino_tokenizer.xml/.bin 和 openvino_detokenizer.xml/.bin 是 OpenVINO
# 官方导出的标配,openvino-genai 2026.x+ 强制要求,漏了会在 generate 时报错。
OV_REQUIRED_FILES = ("openvino_model.xml", "openvino_model.bin")
OV_TOKENIZER_CANDIDATES = (
    # OpenVINO genai 专用 tokenizer/detokenizer(必须有)
    "openvino_tokenizer.xml", "openvino_tokenizer.bin",
    "openvino_detokenizer.xml", "openvino_detokenizer.bin",
    # HuggingFace 通用 tokenizer(某些模型导出会保留)
    "tokenizer.json", "tokenizer.model", "tokenizer_config.json",
    "special_tokens_map.json", "added_tokens.json",
    "config.json", "openvino_config.json", "configuration.json",
    "generation_config.json", "merges.txt", "vocab.json",
    "chat_template.jinja", "generation_config.json",
)
# tree API 不可达时的兜底文件列表(标准 optimum-intel 导出布局)
OV_FALLBACK_FILES = OV_REQUIRED_FILES + (
    "openvino_tokenizer.xml", "openvino_tokenizer.bin",
    "openvino_detokenizer.xml", "openvino_detokenizer.bin",
    "config.json", "openvino_config.json", "configuration.json",
    "generation_config.json", "tokenizer.json",
    "tokenizer_config.json", "special_tokens_map.json",
)


def list_ov_repo_files(repo: str, ms_repo: Optional[str] = None,
                       preference: str = "auto",
                       timeout: int = 15) -> list[dict]:
    """列出 OpenVINO 模型仓库根目录中需要下载的文件 [{path, size}]。

    依次查询 ModelScope / hf-mirror / 官方 HF 的 tree API(兼容不同仓库
    tokenizer 布局);auto/modelscope 偏好时国内 ModelScope 优先。
    - 所有镜像均明确返回 401/404 → 抛 RepoNotFoundError(仓库不存在/下架),
      避免回退固定清单后去硬下不存在的文件,产生 0KB 空目录;
    - 全部为网络层故障 → 回退标准文件清单(size=0)。
    """
    ms_repo = ms_repo or repo
    if preference == "official":
        plan = [("hf", repo), ("ms", ms_repo)]
    else:  # auto / hf_mirror / modelscope:国内网络下 ModelScope 优先最快最稳
        plan = [("ms", ms_repo), ("hf", repo)]

    network_failed = False
    for platform, rid in plan:
        if platform == "hf":
            items, missing, err = _ov_tree_hf(rid, timeout)
        else:
            items, missing, err = _ov_tree_ms(rid, timeout)
        if items is not None:
            picked = _ov_pick_ov_files(items)
            if all(any(f["path"] == n for f in picked)
                   for n in OV_REQUIRED_FILES):
                return picked
            # 200 但布局异常:尝试下一平台
            continue
        if missing:
            continue  # 该镜像明确无此仓库,试下一镜像
        network_failed = True

    if not network_failed:
        raise RepoNotFoundError(
            f"模型仓库 {repo} 在 HuggingFace 与 ModelScope 均不存在或已下架")
    # 全部为网络层故障:兜底标准清单(无 size,下载时按响应头取总大小)
    return [{"path": n, "size": 0} for n in OV_FALLBACK_FILES]


class RepoNotFoundError(Exception):
    """tree API 在所有镜像均明确返回 401/404:仓库不存在或不可见。"""


def _ov_pick_ov_files(items: list) -> list[dict]:
    """从 tree API 原始条目(HF 或 MS 格式)挑出 OV 所需的根目录文件。"""
    want = set(OV_REQUIRED_FILES) | set(OV_TOKENIZER_CANDIDATES)
    picked: list[dict] = []
    for f in items:
        if not isinstance(f, dict):
            continue
        # HF: type=file;ModelScope: Type=blob
        ftype = str(f.get("type") or f.get("Type") or "").lower()
        if ftype not in ("file", "blob"):
            continue
        path = str(f.get("path") or f.get("Path") or "")
        if "/" in path or path not in want:
            continue
        try:
            size = int(f.get("size", f.get("Size", 0)) or 0)
        except (TypeError, ValueError):
            size = 0
        picked.append({"path": path, "size": size})
    return picked


def _ov_tree_hf(repo: str, timeout: int):
    """HF 系 tree API(hf-mirror 优先,官方兜底)。

    返回 (items|None, missing: bool, err|None):
    200 → (json列表, False, None);401/404 → (None, True, None);
    全部网络/临时错误 → (None, False, err)。
    """
    quoted = _quote(repo)
    last_err: Optional[Exception] = None
    for base in HF_API_BASES:
        try:
            resp = requests.get(
                f"{base}/models/{quoted}/tree/main", timeout=timeout)
        except Exception as e:  # 网络层故障,试下一基点
            last_err = e
            continue
        if resp.status_code == 200:
            try:
                return resp.json(), False, None
            except Exception as e:
                return None, False, e
        if resp.status_code in (401, 404):
            return None, True, None
        last_err = Exception(f"HTTP {resp.status_code}")
    return None, False, last_err or Exception("HF API 不可达")


def _ov_tree_ms(ms_repo: str, timeout: int):
    """ModelScope files API。返回值语义同 _ov_tree_hf。"""
    url = (f"https://modelscope.cn/api/v1/models/{_quote(ms_repo)}/repo/files"
           f"?Revision=master&Root=")
    try:
        resp = requests.get(url, timeout=timeout)
    except Exception as e:
        return None, False, e
    if resp.status_code == 200:
        try:
            files = (resp.json().get("Data") or {}).get("Files") or []
            return files, False, None
        except Exception as e:
            return None, False, e
    if resp.status_code in (401, 404):
        return None, True, None
    return None, False, Exception(f"HTTP {resp.status_code}")


def build_ov_file_sources(repo: str, file_name: str,
                          ms_repo: Optional[str] = None) -> dict[str, Optional[str]]:
    """目录模型中单个文件的三源 URL。"""
    return {
        "official": HF_RESOLVE.format(repo=repo, file=file_name),
        "hf_mirror": HF_MIRROR_RESOLVE.format(repo=repo, file=file_name),
        "modelscope": (MS_RESOLVE.format(repo=ms_repo, file=file_name)
                       if ms_repo else None),
    }


def pick_source_url(entry: dict, source: str) -> Optional[str]:
    """按下载源取 URL;modelscope 缺失时自动回退 hf_mirror;再回退 official。"""
    sources = entry.get("sources") or {}
    url = sources.get(source)
    if not url and source == "modelscope":
        url = sources.get("hf_mirror")
    if not url:
        url = sources.get("official")
    return url


def resolve_source(preference: str, entry: dict) -> Optional[str]:
    """把配置偏好(auto/official/hf_mirror/modelscope)解析为实际 URL。"""
    if preference in SOURCE_ORDER and preference != "auto":
        return pick_source_url(entry, preference)
    # auto:官方优先(下载器会做故障转移)
    return pick_source_url(entry, "official")


def source_fallback_list(entry: dict, preference: str = "auto") -> list[tuple[str, str]]:
    """按偏好生成有序的 (源id, URL) 故障转移列表,去重并跳过缺失源。
    官方不可用自动切镜像;镜像失败回退官方。"""
    sources = entry.get("sources") or {}
    order = {
        "official": ["official", "hf_mirror", "modelscope"],
        "hf_mirror": ["hf_mirror", "official", "modelscope"],
        "modelscope": ["modelscope", "hf_mirror", "official"],
    }.get(preference, ["official", "hf_mirror", "modelscope"])
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for sid in order:
        url = sources.get(sid)
        if url and url not in seen:
            seen.add(url)
            out.append((sid, url))
    return out


def probe_url(url: str, timeout: float = 3.0) -> Optional[float]:
    """下载前自检节点:仅建立连接并读取 1 字节验证可达性,不下载完整文件。
    返回延迟(毫秒),失败/状态异常返回 None。"""
    try:
        t0 = time.time()
        with requests.get(url, stream=True, timeout=(3, timeout)) as r:
            if r.status_code not in (200, 206):
                return None
            for _ in r.iter_content(chunk_size=1):
                break
            return round((time.time() - t0) * 1000, 1)
    except Exception:
        return None


# 三个下载源的主站(用于"自检节点"探活显示)
DOWNLOAD_BASES = {
    "official": "https://huggingface.co",
    "hf_mirror": "https://hf-mirror.com",
    "modelscope": "https://modelscope.cn",
}


def probe_download_sources(timeout: float = 3.0) -> list[dict]:
    """并行自检三个下载源主站可达性与延迟,返回 [{id, label, ok, latency_ms}]。"""
    def _probe(sid):
        lat = probe_url(DOWNLOAD_BASES[sid], timeout)
        return {"id": sid, "label": SOURCE_NAMES[sid],
                "ok": lat is not None, "latency_ms": lat}

    with ThreadPoolExecutor(max_workers=len(SOURCE_ORDER)) as pool:
        return list(pool.map(_probe, SOURCE_ORDER))


def _normalize_entry(raw: dict) -> Optional[dict]:
    if not isinstance(raw, dict):
        return None
    name = str(raw.get("name", "")).strip()
    if not name:
        return None
    fmt = str(raw.get("format", "gguf")).strip().lower()
    fmt = fmt if fmt in ("gguf", "openvino_dir") else "gguf"
    sources = raw.get("sources")
    is_ov_dir = (fmt == "openvino_dir")
    # OpenVINO 目录条目只要有 repo 即可(文件列表运行时解析);
    # GGUF 条目必须有完整 sources 或 repo+file。
    if is_ov_dir:
        if not raw.get("repo"):
            return None
        clean_sources = {k: None for k in SOURCE_ORDER}
    elif not isinstance(sources, dict) or not any(
            sources.get(k) for k in SOURCE_ORDER):
        repo, file = raw.get("repo"), raw.get("file")
        if not repo or not file:
            return None  # 既无 sources 又无 repo/file → 丢弃
        sources = build_sources(str(repo), str(file),
                                raw.get("ms_repo"), raw.get("ms_file"))
        clean_sources = {k: (str(sources.get(k)) if sources.get(k) else None)
                         for k in SOURCE_ORDER}
    else:
        clean_sources = {k: (str(sources.get(k)) if sources.get(k) else None)
                         for k in SOURCE_ORDER}
    entry = {
        "name": name,
        "category": str(raw.get("category", "文本")),
        "hw": raw.get("hw") if isinstance(raw.get("hw"), list) else ["CPU"],
        "format": fmt,
        "desc": str(raw.get("desc", "")),
        "size_gb": float(raw.get("size_gb", 0)),
        "sources": clean_sources,
    }
    # OpenVINO 目录条目:保留仓库 ID(逐文件下载时解析)
    if raw.get("repo"):
        entry["repo"] = str(raw["repo"])
    if raw.get("ms_repo"):
        entry["ms_repo"] = str(raw["ms_repo"])
    return entry


def normalize_manifest(data: Any) -> list[dict]:
    """清洗 manifest(列表或 {"models": [...]}):丢弃非法条目。"""
    if isinstance(data, dict):
        data = data.get("models")
    if not isinstance(data, list):
        return []
    out = []
    for raw in data:
        entry = _normalize_entry(raw)
        if entry:
            out.append(entry)
    return out


def load_manifest(source) -> list[dict]:
    """从本地文件路径或 URL 加载自定义 manifest。失败返回空列表(不抛异常)。"""
    text: Optional[str] = None
    src = str(source).strip()
    if not src:
        return []
    if src.startswith(("http://", "https://")):
        try:
            resp = requests.get(src, timeout=15)
            resp.raise_for_status()
            text = resp.text
        except Exception:
            return []
    else:
        p = Path(src)
        if not p.exists():
            return []
        try:
            text = p.read_text(encoding="utf-8")
        except OSError:
            return []
    try:
        return normalize_manifest(json.loads(text))
    except json.JSONDecodeError:
        return []


def save_manifest(path, models: list[dict]) -> bool:
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"models": models}, ensure_ascii=False, indent=2),
                     encoding="utf-8")
        return True
    except OSError:
        return False


# ==================== 在线模型仓库搜索 ====================
# 下载器不依赖离线缓存的下载地址,而是直接查询 HuggingFace 相关 API,
# 动态发现任意 GGUF 模型并生成三源直链。官方 API 不可达时自动切换国内镜像。

# 国内网络优先 hf-mirror(镜像),官方 HF 兜底;官方访问慢会拖垮整个
# 在线目录/仓库解析流程,故镜像排第一。
HF_API_BASES = ["https://hf-mirror.com/api", "https://huggingface.co/api"]

# 已探测出的可用 API 基点(进程内缓存),避免每个仓库都反复慢速试探官方源
_active_api_base: Optional[str] = None


def _preferred_api_base(timeout: float = 6.0) -> str:
    """探测并缓存当前可用的 HF API 基点(优先国内镜像,大幅提速)。"""
    global _active_api_base
    if _active_api_base:
        return _active_api_base
    for base in HF_API_BASES:
        try:
            with requests.get(f"{base}/models?limit=1", timeout=timeout,
                              stream=True) as r:
                if r.status_code == 200:
                    _active_api_base = base
                    return base
        except Exception:
            continue
    _active_api_base = HF_API_BASES[0]
    return _active_api_base


def _api_get_json(path: str, timeout: int = 12):
    """返回首个 200 的 JSON;优先使用已探测的可用基点,全部失败返回 None。"""
    preferred = _preferred_api_base()
    candidates = [preferred] + [b for b in HF_API_BASES if b != preferred]
    for base in candidates:
        url = f"{base}{path}"
        try:
            resp = requests.get(url, timeout=timeout)
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            continue
    return None


def _quote(s: str) -> str:
    try:
        from urllib.parse import quote
        return quote(s)
    except ImportError:
        return s  # 极老环境降级,极少触发


def _classify(repo: str, name: str) -> str:
    s = f"{repo} {name}".lower()
    if any(k in s for k in ("coder", "code", "deepseek-coder")):
        return "编程"
    if any(k in s for k in ("vision", "vl-", "multimodal", "llava",
                            "qwen2-vl", "image", "moondream")):
        return "多模态"
    if any(k in s for k in ("tts", "voice", "audio", "speech", "whisper")):
        return "配音"
    return "文本"


def _list_gguf_files(repo: str, timeout: int = 20) -> list[dict]:
    """列出仓库根目录下的 GGUF 文件 [{path, size}],失败返回空列表。"""
    data = _api_get_json(f"/models/{_quote(repo)}/tree/main", timeout)
    files: list[dict] = []
    if not isinstance(data, list):
        return files
    for f in data:
        if not isinstance(f, dict) or f.get("type") != "file":
            continue
        path = str(f.get("path", ""))
        if path.lower().endswith(".gguf"):
            files.append({"path": path, "size": int(f.get("size", 0))})
    return files


def _pick_gguf_file(files: list[dict]) -> dict:
    """优先 *_q4_k_m.gguf,否则取体积最小的文件(通常是最通用的量化)。"""
    if not files:
        return {}
    for f in files:
        if f["path"].lower().endswith("q4_k_m.gguf"):
            return f
    return min(files, key=lambda f: f.get("size", 0))


def fetch_repo(repo: str) -> list[dict]:
    """直接按 HuggingFace 仓库 ID 拉取模型(如 Qwen/Qwen2.5-0.5B-Instruct-GGUF)。
    返回标准商店条目;失败/无 GGUF 返回空列表。"""
    repo = str(repo).strip().strip("/")
    if not repo or "/" not in repo:
        return []
    files = _list_gguf_files(repo)
    file = _pick_gguf_file(files)
    if not file:
        return []
    name = repo.split("/")[-1]
    return [{
        "name": name,
        "category": _classify(repo, name),
        "hw": ["CPU", "GPU"],
        "desc": f"仓库:{repo}",
        "size_gb": round(file.get("size", 0) / (1024 ** 3), 2),
        "sources": build_sources(repo, file["path"]),
    }]


def search_hub(query: str, limit: int = 15, sort: str = "") -> list[dict]:
    """在线搜索 HuggingFace GGUF 模型。query 可为关键词或仓库 ID。
    返回标准商店条目;失败返回空列表(不抛异常)。
    sort 可传 "downloads" 按下载量排序。"""
    q = str(query).strip()
    if not q:
        return []
    limit = max(1, min(int(limit), 50))

    # 输入形如 org/repo → 直接拉取该仓库
    if "/" in q:
        entries = fetch_repo(q)
        if entries:
            return entries

    params = f"search={_quote(q)}"
    if sort:
        params += f"&sort={sort}&direction=-1"
    params += f"&limit={limit}"
    data = _api_get_json(f"/models?{params}")
    if not isinstance(data, list):
        return []
    models = data

    entries: list[dict] = []
    for m in models:
        if not isinstance(m, dict):
            continue
        repo = str(m.get("id") or m.get("modelId") or "").strip()
        if not repo:
            continue
        files = _list_gguf_files(repo, timeout=15)
        file = _pick_gguf_file(files)
        if not file:
            continue
        name = repo.split("/")[-1]
        entries.append({
            "name": name,
            "category": _classify(repo, name),
            "hw": ["CPU", "GPU"],
            "desc": f"在线结果:{repo}",
            "size_gb": round(file.get("size", 0) / (1024 ** 3), 2),
            "sources": build_sources(repo, file["path"]),
        })
        if len(entries) >= limit:
            break
    return entries


# 热门 GGUF 仓库清单(仅仓库 ID,不缓存下载地址;URL 由 fetch_repo 实时生成)。
# 作为在线列表的兜底:当 HF 搜索返回不足时逐仓库实时解析文件与量化。
POPULAR_GGUF_REPOS = [
    "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
    "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
    "Qwen/Qwen2.5-3B-Instruct-GGUF",
    "Qwen/Qwen2.5-7B-Instruct-GGUF",
    "Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF",
    "Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF",
    "Qwen/Qwen2.5-Coder-3B-Instruct-GGUF",
    "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
    "bartowski/Llama-3.2-1B-Instruct-GGUF",
    "bartowski/Llama-3.2-3B-Instruct-GGUF",
    "bartowski/Llama-3.1-8B-Instruct-GGUF",
    "bartowski/Mistral-7B-Instruct-v0.3-GGUF",
    "bartowski/gemma-2-2b-it-GGUF",
    "bartowski/gemma-2-9b-it-GGUF",
    "bartowski/Phi-3.5-mini-instruct-GGUF",
    "microsoft/Phi-3-mini-4k-instruct-gguf",
    "microsoft/Phi-3.5-mini-instruct-gguf",
    "unsloth/DeepSeek-R1-Distill-Qwen-1.5B-GGUF",
    "unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF",
    "unsloth/Llama-3.2-1B-Instruct-GGUF",
    "lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF",
    "lmstudio-community/gemma-2-2b-it-GGUF",
]


def fetch_online_catalog(limit: int = 100,
                         progress=None) -> list[dict]:
    """动态拉取在线热门 GGUF 模型列表(融合 HuggingFace + ModelScope)。

    - 优先用磁盘缓存(30 分钟内直接返回,避免校园网下反复慢速拉取);
    - 否则并行从 HuggingFace 与 ModelScope 搜索 GGUF 仓库并解析量化文件;
    - ModelScope 为国内源、网络更稳定,优先采纳;HF 作为补充兜底;
    - 任何网络异常都会被吞掉,返回已成功解析的条目(可能为空)。
    progress(已解析数, 总数) 可选,用于 UI 展示加载进度。
    """
    limit = max(1, min(int(limit), 200))

    # 磁盘缓存命中 → 秒出
    cached = _load_catalog_cache()
    if cached is not None:
        if progress:
            progress(len(cached), len(cached))
        return cached[:limit]

    # 收集候选仓库:ModelScope(国内优先)+ HF 搜索 + 热门清单兜底
    hf_repos: list[str] = []
    for repo in _search_gguf_repos(limit):
        if repo not in hf_repos:
            hf_repos.append(repo)
    for repo in POPULAR_GGUF_REPOS:
        if repo not in hf_repos:
            hf_repos.append(repo)
    ms_repos = _ms_search_gguf_repos(limit)

    # ModelScope 来源放在前面,同名仓库(如 unsloth/DeepSeek-R1-GGUF)自动去重
    jobs = [(r, True) for r in ms_repos] + [(r, False) for r in hf_repos]

    def _resolve(job):
        repo, is_ms = job
        return _ms_repo_to_entry(repo) if is_ms else _repo_to_entry(repo)

    entries: list[dict] = []
    seen: set[str] = set()
    # 并行解析(网络 IO 密集,多线程显著提速)
    with ThreadPoolExecutor(max_workers=8) as pool:
        for e in pool.map(_resolve, jobs):
            if not e:
                continue
            if e["name"] in seen:
                continue
            seen.add(e["name"])
            entries.append(e)
            if progress:
                progress(len(entries), len(jobs))
            if len(entries) >= limit:
                break

    _save_catalog_cache(entries)
    return entries


# 在线目录缓存:文件名 + TTL(秒)
_CATALOG_CACHE_FILE = paths.WORK_DIR / "online_catalog_cache.json"
_CATALOG_CACHE_TTL = 30 * 60  # 30 分钟


def _load_catalog_cache() -> Optional[list[dict]]:
    try:
        if not _CATALOG_CACHE_FILE.exists():
            return None
        data = json.loads(_CATALOG_CACHE_FILE.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return None
        ts = data.get("timestamp", 0)
        if time.time() - float(ts) > _CATALOG_CACHE_TTL:
            return None
        models = data.get("models")
        return models if isinstance(models, list) else None
    except Exception:
        return None


def _save_catalog_cache(entries: list[dict]) -> None:
    try:
        _CATALOG_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _CATALOG_CACHE_FILE.write_text(
            json.dumps({"timestamp": time.time(), "models": entries},
                       ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass


def _search_gguf_repos(limit: int) -> list[str]:
    """通过 HF 搜索 API 返回热门 GGUF 仓库 ID 列表(按下载量)。失败返回空。"""
    data = _api_get_json(
        f"/models?search=GGUF&sort=downloads&direction=-1&limit={max(1, min(limit, 100))}")
    if not isinstance(data, list):
        return []
    repos: list[str] = []
    for m in data:
        if not isinstance(m, dict):
            continue
        repo = str(m.get("id") or m.get("modelId") or "").strip()
        if repo:
            repos.append(repo)
    return repos


def _repo_to_entry(repo: str) -> Optional[dict]:
    """把单个仓库解析为标准条目;无 GGUF 文件返回 None。"""
    files = _list_gguf_files(repo, timeout=15)
    file = _pick_gguf_file(files)
    if not file:
        return None
    name = repo.split("/")[-1]
    return {
        "name": name,
        "category": _classify(repo, name),
        "hw": ["CPU", "GPU"],
        "desc": f"在线模型:{repo}",
        "size_gb": round(file.get("size", 0) / (1024 ** 3), 2),
        "sources": build_sources(repo, file["path"]),
    }


# ==================== ModelScope(魔搭)在线模型来源 ====================
# 国内网络下 HuggingFace 常不稳定,ModelScope 作为更稳定的国内源并行拉取。
# 搜索走新 OpenAPI,文件列表走仓库 files API。URL 结构与 HF 侧完全不同。

MS_SEARCH_API = "https://modelscope.cn/openapi/v1/models"
MS_FILES_API = ("https://modelscope.cn/api/v1/models/{repo}/repo/files"
                "?Revision=master&Root=")


def _ms_search_gguf_repos(limit: int) -> list[str]:
    """搜索 ModelScope 热门 GGUF 仓库 ID(按下载量,自动翻页最多 3 页)。"""
    want = max(1, min(int(limit), 150))
    repos: list[str] = []
    for page in range(1, 4):
        try:
            resp = requests.get(
                MS_SEARCH_API,
                params={"search": "GGUF",
                        "page_size": 50,
                        "page_number": page,
                        "sort": "downloads"},
                timeout=15,
            )
            if resp.status_code != 200:
                break
            data = resp.json()
            models = ((data or {}).get("data") or {}).get("models")
            if not isinstance(models, list) or not models:
                break
            for m in models:
                if not isinstance(m, dict):
                    continue
                rid = str(m.get("id") or "").strip()
                if rid and "/" in rid and rid not in repos:
                    repos.append(rid)
                    if len(repos) >= want:
                        return repos
        except Exception:
            break
    return repos


def _ms_list_gguf_files(repo: str, timeout: int = 20) -> list[dict]:
    """列出 ModelScope 仓库根目录 GGUF 文件 [{path, size}],失败返回空列表。"""
    try:
        resp = requests.get(
            MS_FILES_API.format(repo=_quote(repo)),
            timeout=timeout,
        )
        if resp.status_code != 200:
            return []
        data = resp.json()
        files_raw = ((data or {}).get("Data") or {}).get("Files")
        if not isinstance(files_raw, list):
            return []
        files: list[dict] = []
        for f in files_raw:
            if not isinstance(f, dict):
                continue
            path = str(f.get("Path") or "")
            if path.lower().endswith(".gguf"):
                files.append({"path": path, "size": int(f.get("Size", 0))})
        return files
    except Exception:
        return []


def _ms_repo_to_entry(repo: str) -> Optional[dict]:
    """把 ModelScope 仓库解析为标准条目(modelscope 源优先,HF 同名仓库兜底)。"""
    files = _ms_list_gguf_files(repo)
    file = _pick_gguf_file(files)
    if not file:
        return None
    name = repo.split("/")[-1]
    return {
        "name": name,
        "category": _classify(repo, name),
        "hw": ["CPU", "GPU"],
        "desc": f"在线模型(ModelScope):{repo}",
        "size_gb": round(file.get("size", 0) / (1024 ** 3), 2),
        # 传 ms_repo/ms_file,令 modelscope 直链可用;HF 同名仓库 URL 作为兜底
        "sources": build_sources(repo, file["path"],
                                 ms_repo=repo, ms_file=file["path"]),
    }
