# NovaCore-Local

本地 AI 快速部署工具 —— 一键下载模型、加载推理、开箱即用。支持 **llama.cpp（CPU/GPU）**、**Ollama** 与 **Intel NPU（OpenVINO）** 三种推理引擎，并内置模型商店、对话历史、LoRA 微调、OpenAI 兼容 API 服务与依赖管理。界面支持**中文 / English 切换**。

A local AI quick-deploy tool — download models, load, and run inference out of the box. Supports three engines — **llama.cpp (CPU/GPU)**, **Ollama**, and **Intel NPU (OpenVINO)** — plus a model store, chat history, LoRA fine-tuning, an OpenAI-compatible API, and dependency management. UI language: **中文 / English**.

---

## 特性 / Features

- **三种推理引擎，自动探测选择 / Three engines, auto-detected**：llama.cpp（GGUF，CPU/GPU）、Ollama（本地服务，全自动拉起与模型导入）、OpenVINO（Intel NPU/CPU 原生推理）
- **Intel NPU 原生加速 / Intel NPU acceleration**：NPU 专区提供对称 INT4 的 OpenVINO 优化模型；首次加载在独立子进程中编译并缓存，之后秒级加载；不兼容时安全回退 CPU 并如实提示
- **模型商店 / Model store**：内置精选清单 + HuggingFace 在线搜索；多下载源（官方 / hf-mirror / ModelScope）故障转移与断点续传
- **对话面板 / Chat**：多会话历史、流式输出、随时停止、速度统计
- **模型训练 / Training**：LoRA 微调（transformers + peft）
- **OpenAI 兼容 API / OpenAI-compatible API**：`/v1/chat/completions`（stream）、`/v1/models`、`/health`
- **依赖管理 / Dependency manager**：一键检测与安装，支持国内 pip 镜像自动测速
- **系统设置 / Settings**：主题切换、界面缩放、**中英双语切换**、生成参数、Ollama 服务自动/手动启动

---

## 环境要求 / Requirements

- Windows 10/11（x64）
- Python 3.10+（源码运行 / for running from source）
- 依赖：PyQt6、psutil、requests
- 可选 / Optional：`llama-cpp-python`、`openvino` + `openvino-genai`、Ollama、`torch` + `transformers` + `peft`

> 缺失的可选依赖可在程序内「依赖管理」页一键安装。
> Missing optional dependencies can be installed in the "Dependencies" page.

## 运行（源码）/ Run (source)

```bat
pip install PyQt6 psutil requests
python novacore_main.py
```

首次运行会在 `novacore_data\` 下自动创建配置、日志与模型目录。
First run creates `novacore_data\` (config / logs / models).

## 打包 / Build exe

```bat
build_exe.bat
```

生成 `dist\NovaCore-Local\NovaCore-Local.exe`，可整个文件夹分发。
Produces `dist\NovaCore-Local\NovaCore-Local.exe`; distribute the whole folder.

## 数据目录 / Data directory

所有用户数据位于程序同级的 `novacore_data\`。All user data lives in `novacore_data\` next to the program.

```
novacore_data\
├── config.json          # 配置 / config
├── chat_sessions.json   # 对话历史 / chat history
├── local_models\        # 本地模型 / models (GGUF / OpenVINO IR)
├── logs\                # 运行日志 / logs
├── datasets\  loras\    # 训练数据与 LoRA 产物 / training artifacts
```

## NPU 使用提示 / NPU tips

1. 「模型商店 → NPU专区」下载标记 `-int4-gq-ov` / `-int4-cw-ov` 的对称 INT4 模型；
2. 顶部「推理设备」选择 **NPU-OpenVINO**，加载模型；
3. 本地模型列表的徽章会显示实际运行设备（`运行中(NPU)` / `运行中(CPU)`）；
4. 首次加载需编译（约 1~5 分钟），之后走缓存秒开。

> 1. Download `-int4-gq-ov` / `-int4-cw-ov` models from "Store → NPU";
> 2. Choose **NPU-OpenVINO** as the device and load;
> 3. The badge shows the real device (`Running(NPU)` / `Running(CPU)`);
> 4. First load compiles (~1–5 min), then cached for instant loads.

## 目录结构 / Structure

```
novacore/            # 核心包 / core package
novacore_main.py     # PyQt6 图形界面主程序 / GUI entry
tests/               # 自测 / tests
_e2e.py              # 端到端验证 / end-to-end check
build_exe.bat        # 打包脚本 / build script
NovaCore-Local.spec  # PyInstaller 配置 / spec
novacore.ico/.png    # 应用图标 / app icon
```

## 版本 / Version

`0.0.1-alpha`
