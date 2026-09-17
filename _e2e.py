# -*- coding: utf-8 -*-
"""端到端验证:下载 DeepSeek R1 NPU 模型 → NPU 加载(失败自动回退 CPU)→ 生成。"""
import os, sys, time, subprocess, json
from pathlib import Path
from PyQt6.QtCore import QCoreApplication

app = QCoreApplication(sys.argv)

# ----- 1. 下载(ModelScope 源) -----
from novacore import modelstore
from novacore.downloader import OVDirectoryDownloadWorker

entry = {
    "name": "DeepSeek-R1-Distill-Qwen-1.5B-OV-INT4-NPU",
    "repo": "OpenVINO/DeepSeek-R1-Distill-Qwen-1.5B-int4-gq-ov",
    "ms_repo": "OpenVINO/DeepSeek-R1-Distill-Qwen-1.5B-int4-gq-ov",
    "format": "openvino_dir",
    "hw": ["NPU"],
}

dst = str(Path("novacore_data") / "local_models" / "DeepSeek-R1-Distill-Qwen-1.5B-OV-INT4-NPU")

if os.path.exists(dst):
    print(f"[跳过] 模型已存在于 {dst}")
else:
    print(f"[下载] 目标目录: {dst}")
    results = {}
    logs = []
    w = OVDirectoryDownloadWorker(entry, dst, preference="modelscope")
    w.log.connect(lambda s: logs.append(s))
    w.done.connect(lambda ok, msg, p: results.update(ok=ok, msg=msg, path=p))
    w.progress.connect(lambda pct, done, total: None)
    w.start()
    end = time.time() + 600
    while time.time() < end and "ok" not in results:
        app.processEvents()
        time.sleep(0.05)
    w.wait(5000)

    if not results.get("ok"):
        print("下载失败:", results.get("msg"))
        for s in logs[-15:]:
            print("  ", s)
        sys.exit(1)
    print("[下载完成]", results.get("path"))
    for f in os.listdir(dst):
        sz = os.path.getsize(os.path.join(dst, f))
        tag = "⚠️" if f.endswith(".part") else ""
        print(f"  {sz/1024/1024:.2f}MB  {f} {tag}")

# 检查必需文件
required = ["openvino_model.xml", "openvino_model.bin",
            "openvino_tokenizer.xml", "openvino_tokenizer.bin"]
for r in required:
    p = os.path.join(dst, r)
    assert os.path.exists(p) and os.path.getsize(p) > 0, f"缺 {r}"
print("✅ 必需文件全部齐全")

# ----- 2. 加载 + 生成(子进程里跑,防 NPU 编译崩溃影响主进程) -----
print("\n[加载 + 生成] 引擎自动选择设备(优先 NPU,失败回退 CPU)...")
load_script = f'''
import sys, os
sys.path.insert(0, ".")

from novacore.engines.openvino import OpenVinoEngine, GenParams

e = OpenVinoEngine()
e.load_model({dst!r}, device="NPU",
             progress=lambda s: print("  [progress]", s, flush=True))
print("✅ 加载成功, device:", e._device, "| detail:", e._load_detail)

params = GenParams(max_tokens=64, temperature=0.7)
msgs = [{{"role": "user", "content": "用一句话自我介绍"}}]
output = []
for token in e.generate(msgs, params):
    output.append(token)
text = "".join(output)
print("=== 生成输出 ===")
print(text)
print("\\n=== 生成完成,共", len(output), "个token, 设备:", e._device, "===")
'''

proc = subprocess.run(
    [sys.executable, "-c", load_script],
    capture_output=True, text=True, timeout=1800)

print(proc.stdout[-3000:] if proc.stdout else "(no stdout)")
if proc.stderr:
    print("STDERR:", proc.stderr[-800:])
print("\nreturncode:", proc.returncode)
sys.exit(0 if proc.returncode == 0 else 1)
