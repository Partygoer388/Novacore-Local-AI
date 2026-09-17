"""应用自动更新:
- 检查:读取 GitHub Releases 最新版本(或自定义 JSON 源),与当前版本比较
- 下载:流式下载新版本 zip,带进度与取消
- 应用:生成一个 PowerShell 脚本,等本程序退出后解压覆盖安装目录并重启
  (Windows 下运行中的 exe/DLL 无法自替换,必须由外部脚本在退出后完成)

仅打包版(frozen)支持"立即更新";源码运行模式只能检查并跳转下载页。
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable, Optional

import requests

# 项目 GitHub 仓库(发布 Release 与附件)
GITHUB_REPO = "Partygoer388/Novacore-Local-AI"
GITHUB_API_LATEST = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
GITHUB_RELEASES_PAGE = f"https://github.com/{GITHUB_REPO}/releases/latest"


def _parse_version(text: str) -> tuple:
    """把版本字符串解析为数字元组,便于比较(如 v0.0.2-alpha -> (0,0,2))。"""
    nums = re.findall(r"\d+", str(text or ""))
    return tuple(int(n) for n in nums[:3]) if nums else (0,)


def is_newer(latest: str, current: str) -> bool:
    """latest 是否比 current 新(数字更大;数字相同但标记不同视为新构建)。"""
    def _norm(s: str) -> str:
        return str(s or "").strip().lower().lstrip("v")

    lp, cp = _parse_version(latest), _parse_version(current)
    if lp > cp:
        return True
    if lp == cp and _norm(latest) != _norm(current):
        return True
    return False


def _normalize_release(data: dict) -> dict:
    """统一成 {version, notes, url, asset_url, asset_name, source}。"""
    assets = data.get("assets") or []
    asset_url, asset_name = "", ""
    for a in assets:
        name = str(a.get("name", ""))
        if name.lower().endswith(".zip"):
            asset_url = str(a.get("browser_download_url", ""))
            asset_name = name
            if "win64" in name.lower() or "windows" in name.lower():
                break
    tag = str(data.get("tag_name") or data.get("version") or "").strip()
    return {
        "version": tag.lstrip("vV"),
        "notes": str(data.get("body") or data.get("notes") or ""),
        "url": str(data.get("html_url") or data.get("url") or GITHUB_RELEASES_PAGE),
        "asset_url": asset_url,
        "asset_name": asset_name,
        "source": data.get("_source", "github"),
    }


def latest_release(timeout: int = 20) -> dict:
    """读取 GitHub 最新 Release(公开仓库,无需 token)。失败抛异常。"""
    resp = requests.get(
        GITHUB_API_LATEST, timeout=timeout,
        headers={"Accept": "application/vnd.github+json",
                 "User-Agent": "NovaCore-Local-Updater"})
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, dict) or not data.get("tag_name"):
        raise ValueError("GitHub 未返回有效的 Release 信息")
    data["_source"] = "github"
    return _normalize_release(data)


def from_custom_json(url: str, timeout: int = 15) -> dict:
    """读取自定义更新源 JSON:{"version","url","notes"}(兼容旧格式)。"""
    resp = requests.get(url, timeout=timeout,
                        headers={"User-Agent": "NovaCore-Local-Updater"})
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, dict):
        raise ValueError("更新信息不是 JSON 对象")
    data["_source"] = "custom"
    rel = _normalize_release(data)
    if not rel["url"]:
        rel["url"] = url
    return rel


def download_asset(url: str, dest: str,
                   progress: Optional[Callable[[int, int], None]] = None,
                   cancel_check: Optional[Callable[[], bool]] = None) -> None:
    """流式下载到 dest(先写 .part 再改名);progress(done, total)。"""
    tmp = dest + ".part"
    done = 0
    with requests.get(url, stream=True, timeout=(15, 300),
                      headers={"User-Agent": "NovaCore-Local-Updater"}) as r:
        r.raise_for_status()
        try:
            total = int(r.headers.get("Content-Length") or 0)
        except (TypeError, ValueError):
            total = 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                if cancel_check is not None and cancel_check():
                    raise InterruptedError("更新下载已取消")
                if chunk:
                    f.write(chunk)
                    done += len(chunk)
                    if progress:
                        progress(done, total)
    os.replace(tmp, dest)


def current_app_dir() -> Optional[Path]:
    """打包运行时的应用目录(exe 所在目录);源码运行返回 None。"""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return None


def supports_self_update() -> bool:
    return getattr(sys, "frozen", False)


def _ps_quote(path: str) -> str:
    """PowerShell 单引号字符串转义。"""
    return "'" + str(path).replace("'", "''") + "'"


def build_update_script(zip_path: str, app_dir: str, exe_path: str,
                        pid: int) -> str:
    """生成更新用的 PowerShell 脚本内容(等待退出 -> 解压覆盖 -> 重启)。"""
    return f"""$ErrorActionPreference = 'SilentlyContinue'
$zip = {_ps_quote(zip_path)}
$app = {_ps_quote(app_dir)}
$exe = {_ps_quote(exe_path)}
$pid_to_wait = {int(pid)}
# 1) 等待当前程序退出(最多 5 分钟)
for ($i = 0; $i -lt 600; $i++) {{
    if (-not (Get-Process -Id $pid_to_wait -ErrorAction SilentlyContinue)) {{ break }}
    Start-Sleep -Milliseconds 500
}}
Start-Sleep -Seconds 2
# 2) 解压到临时目录
$tmp = Join-Path $env:TEMP ('novacore_upd_' + [System.Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Force -Path $tmp | Out-Null
Expand-Archive -LiteralPath $zip -DestinationPath $tmp -Force
# 3) 定位解压后的根目录(zip 内通常有一层 NovaCore-Local/)
$src = $tmp
$dirs = @(Get-ChildItem -Path $tmp -Directory)
if ($dirs.Count -eq 1) {{ $src = $dirs[0].FullName }}
# 4) 覆盖安装目录
Copy-Item -Path (Join-Path $src '*') -Destination $app -Recurse -Force
# 5) 清理与重启
Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
Remove-Item -Force $zip -ErrorAction SilentlyContinue
if (Test-Path $exe) {{ Start-Process -FilePath $exe }}
Remove-Item -Force $MyInvocation.MyCommand.Path -ErrorAction SilentlyContinue
"""


def launch_update(zip_path: str, app_dir: Path, exe_path: Path, pid: int) -> Path:
    """把更新脚本写入临时文件并分离启动,随后本进程应尽快退出。"""
    script = build_update_script(str(zip_path), str(app_dir), str(exe_path), pid)
    fd, script_path = tempfile.mkstemp(prefix="novacore_update_", suffix=".ps1")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(script)
    flags = 0
    if sys.platform == "win32":
        flags = (getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
                 | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200))
    subprocess.Popen(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
         "-WindowStyle", "Hidden", "-File", script_path],
        close_fds=True, creationflags=flags)
    return Path(script_path)
