# NovaCore 本地 AI 引擎核心包
import os

APP_NAME = "NovaCore-Local"
APP_VERSION = "0.0.1-alpha"
APP_VERSION_TAG = "v0.0.1-alpha"


def _ensure_localhost_no_proxy() -> None:
    """确保本地回环地址不被系统代理劫持。

    关键:不少机器(尤其校园网/VPN 环境)设置了 127.0.0.1 端口代理,而 Windows
    的 requests/urllib 会把 127.0.0.1 的本地请求(如 Ollama 11434)也送进代理,
    导致 Ollama 连不上、下载 502/超时。故强制把回环地址加入 no_proxy。
    """
    loopback = ["127.0.0.1", "localhost", "::1"]
    existing = os.environ.get("no_proxy") or os.environ.get("NO_PROXY") or ""
    parts = [p.strip() for p in existing.replace(",", " ").replace(";", " ").split()
             if p.strip()]
    for h in loopback:
        if h not in parts:
            parts.append(h)
    merged = ",".join(parts)
    os.environ["no_proxy"] = merged
    os.environ["NO_PROXY"] = merged


_ensure_localhost_no_proxy()
