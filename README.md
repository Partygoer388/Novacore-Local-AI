# NovaCore-Local

本地 AI 快速部署工具 —— 一键下载模型、加载推理、开箱即用。支持 **llama.cpp（CPU/GPU）**、**Ollama** 与 **Intel NPU（OpenVINO）** 三种推理引擎，并内置模型商店、对话历史、LoRA 微调、OpenAI 兼容 API 服务与依赖管理。

## 特性

- **三种推理引擎，自动探测选择**：llama.cpp（GGUF，CPU/GPU 通用）、Ollama（本地服务，全自动拉起与模型导入）、OpenVINO（Intel NPU/CPU 原生推理）
- **Intel NPU 原生加速**：NPU 专区提供对称 INT4 的 OpenVINO 优化模型；首次加载在独立子进程中编译并缓存（`openvino_npu.blob`），之后秒级加载；模型不兼容时安全回退 CPU 并如实提示，绝不静默假装在 NPU 上运行
- **模型商店**：内置精选清单 + HuggingFace 在线搜索；多下载源（官方 / hf-mirror / ModelScope）故障转移与断点续传
- **对话面板**：多会话历史、流式输出、随时停止、速度统计
- **模型训练**：LoRA 微调（transformers + peft）
- **OpenAI 兼容 API**：`/v1/chat/completions`（支持 stream）、`/v1/models`、`/health`
- **依赖管理**：一键检测与安装缺失依赖，支持国内 pip 镜像自动测速
- **系统设置**：主题切换、界面缩放、生成参数、Ollama 服务自动/手动启动

## 环境要求

- Windows 10/11（x64）
- Python 3.10+（开发运行）
- 依赖：PyQt6、psutil、requests
- 可选：`llama-cpp-python`（CPU/GPU GGUF 推理）、`openvino` + `openvino-genai`（NPU/CPU 推理）、Ollama（本地服务）、torch + transformers + peft（微调）

> 缺失的可选依赖可在程序内「依赖管理」页一键安装。

## 运行（源码方式）

```bat
pip install PyQt6 psutil requests
python novacore_main.py
```

首次运行会在 `novacore_data\` 下自动创建配置、日志与模型目录。

## 打包为独立 exe

```bat
build_exe.bat
```

生成 `dist\NovaCore-Local\NovaCore-Local.exe`，可整个文件夹分发，开箱即用。

## 数据目录

所有用户数据位于程序同级的 `novacore_data\`：

```
novacore_data\
├── config.json          # 配置
├── chat_sessions.json   # 对话历史
├── local_models\        # 本地模型(GGUF / OpenVINO IR 目录)
├── logs\                # 运行日志(排查设备/回退问题时看这里)
├── datasets\  loras\    # 训练数据与 LoRA 产物
```

## NPU 使用提示

1. 「模型商店 → NPU专区」下载标记 `-int4-gq-ov` / `-int4-cw-ov` 的对称 INT4 模型；
2. 顶部「推理设备」选择 **NPU-OpenVINO**，加载模型；
3. 本地模型列表的徽章会显示实际运行设备（`运行中(NPU)` / `运行中(CPU)`）；
4. 首次加载需编译（约 1~5 分钟），之后走缓存秒开。

## 目录结构

```
novacore/            # 核心包(引擎/配置/下载/商店/训练/API)
novacore_main.py     # PyQt6 图形界面主程序
tests/               # 自测(离屏运行,不依赖真实模型)
_e2e.py              # 端到端验证脚本
build_exe.bat        # 打包脚本
NovaCore-Local.spec  # PyInstaller 配置(由 build_exe.bat 生成)
novacore.ico/.png    # 应用图标
```

## 版本

`0.0.1-alpha`
