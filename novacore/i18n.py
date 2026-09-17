"""轻量级中英文界面翻译层。

默认中文;当语言为英文时,tr() 查表返回英文,未收录的字符串原样返回(中文兜底)。
语言由配置项 ui_lang 控制("zh"/"en"),在 main() 启动时设置,重启后生效。
"""
from __future__ import annotations

import threading

ZH = "zh"
EN = "en"

# 设置页语言下拉框:显示名 -> 值
LANG_CHOICES = [("中文 (Chinese)", ZH), ("English", EN)]

_T: dict[str, str] = {
    # ---------- 导航 ----------
    "💬  对话面板": "💬  Chat",
    "📦  模型商店": "📦  Model Store",
    "💾  本地模型": "💾  Local Models",
    "🎯  模型训练": "🎯  Training",
    "🔄  系统更新": "🔄  Updates",
    "🔧  依赖管理": "🔧  Dependencies",
    "⚙️  系统设置": "⚙️  Settings",

    # ---------- 页面标题 ----------
    "📦 模型商店": "📦 Model Store",
    "💾 本地模型": "💾 Local Models",
    "🎯 LoRA 微调训练": "🎯 LoRA Fine-tuning",
    "🔄 系统更新": "🔄 System Updates",
    "🔧 依赖管理": "🔧 Dependency Management",
    "⚙️ 系统设置": "⚙️ Settings",

    # ---------- 对话页 ----------
    "📜 对话历史": "📜 History",
    "🆕 新建对话": "🆕 New Chat",
    "🗑 清空全部历史": "🗑 Clear All",
    "推理设备:": "Device:",
    "选择模型:": "Model:",
    "加载": "Load",
    "停止": "Stop",
    "发送": "Send",
    "⏹ 停止": "⏹ Stop",
    "📎 文件": "📎 File",
    "🖼️ 图片": "🖼️ Image",
    "输入消息,按回车发送...": "Type a message, press Enter to send...",
    "选择要附加的文件": "Select files to attach",
    "选择图片": "Select an image",
    "图片文件 (*.png *.jpg *.jpeg *.bmp *.webp *.gif);;所有文件 (*)":
        "Image files (*.png *.jpg *.jpeg *.bmp *.webp *.gif);;All files (*)",
    "正在停止...": "Stopping...",
    "生成中...": "Generating...",
    "生成失败": "Generation failed",
    "已手动停止": "Stopped manually",
    "未加载模型": "No model loaded",
    "请先加载模型:": "Please load a model first:",
    "当前引擎: 未加载": "Engine: none",
    "引擎探测中...": "Detecting engines...",
    "=== 新会话已开始 ===": "=== New chat started ===",
    "(已手动停止,保留已生成部分)": "(Stopped manually, partial output kept)",
    "用户": "User",
    "助手": "Assistant",
    "新对话": "New Chat",

    # ---------- 商店页 ----------
    "下载源:": "Source:",
    "硬件筛选:": "Hardware:",
    "类型筛选:": "Type:",
    "全部": "All",
    "CPU/GPU": "CPU/GPU",
    "NPU专区": "NPU",
    "文本": "Text",
    "编程": "Coding",
    "多模态": "Multimodal",
    "配音": "Voice",
    "🔍 自检节点": "🔍 Check Sources",
    "🔄 刷新在线列表": "🔄 Refresh Online",
    "🔍 在线搜索": "🔍 Search",
    "恢复内置清单": "Restore Built-in",
    "在线拉取任意模型:输入关键词或 HF 仓库 ID,如 Qwen2.5 或 ":
        "Search any model online: keyword or HF repo ID, e.g. Qwen2.5 or ",
    "下载": "Download",

    # ---------- 本地模型页 ----------
    "🔄 刷新列表": "🔄 Refresh",
    "📂 导入模型文件": "📂 Import Model",
    "🦙 导入 Ollama 模型": "🦙 Import Ollama",
    "📁 打开模型目录": "📁 Open Model Folder",
    "加载模型": "Load",
    "关闭模型": "Unload",
    "删除": "Delete",
    "💤 已停止": "💤 Stopped",
    "✅ 运行中": "✅ Running",
    "⏳ 启动中": "⏳ Starting",
    "❌ 启动失败": "❌ Failed",
    "模型文件 (*.gguf *.bin *.safetensors);;所有文件 (*)":
        "Model files (*.gguf *.bin *.safetensors);;All files (*)",
    "暂无本地模型,可在商店下载或手动导入": "No local models yet — download from the store or import manually",
    "确定删除这条对话记录吗?此操作不可恢复。": "Delete this conversation? This cannot be undone.",
    "删除对话": "Delete Chat",

    # ---------- 训练页 ----------
    "模型与数据": "Model & Data",
    "基座模型": "Base Model",
    "数据集": "Dataset",
    "输出名称": "Output Name",
    "训练参数": "Training Params",
    "训练轮数": "Epochs",
    "学习率": "Learning Rate",
    "批次大小": "Batch Size",
    "LoRA rank": "LoRA rank",
    "最大长度": "Max Length",
    "开始训练": "Start Training",
    "停止训练": "Stop Training",
    "选择": "Browse",
    "HF 模型目录或仓库名(如 Qwen/Qwen2.5-0.5B-Instruct)":
        "HF model dir or repo (e.g. Qwen/Qwen2.5-0.5B-Instruct)",
    "JSONL 文件路径(prompt/completion 或 text)":
        "JSONL file path (prompt/completion or text)",

    # ---------- 更新页 ----------
    "更新源地址(返回 JSON:{\"version\",\"url\",\"notes\"}):":
        "Update source URL (returns JSON {\"version\",\"url\",\"notes\"}):",
    "启用自定义更新源": "Enable custom update source",
    "检查更新": "Check for Updates",

    # ---------- 依赖页 ----------
    "🔍 检测全部依赖": "🔍 Detect All",
    "📦 一键补全缺失依赖": "📦 Install Missing",
    "状态": "Status",
    "依赖名称": "Dependency",
    "功能说明": "Description",
    "操作": "Action",
    "安装": "Install",
    "重新安装": "Reinstall",

    # ---------- 设置页 ----------
    "推理引擎设置": "Inference Engine",
    "默认推理引擎": "Default Engine",
    "偏好推理设备": "Preferred Device",
    "Ollama 地址": "Ollama URL",
    "自动启动 Ollama 服务(程序打开时自动拉起)": "Auto-start Ollama (on app launch)",
    "🚀 启动服务": "🚀 Start Service",
    "测试连接": "Test Connection",
    "Intel NPU 状态": "Intel NPU Status",
    "生成参数": "Generation",
    "温度 temperature": "Temperature",
    "Top-P": "Top-P",
    "Top-K": "Top-K",
    "最大生成 tokens": "Max Tokens",
    "上下文长度": "Context Length",
    "系统提示词": "System Prompt",
    "API 服务设置(OpenAI 兼容接口)": "API Server (OpenAI-compatible)",
    "API 监听端口": "API Port",
    "启用 API 服务(可供外部前端调用)": "Enable API server (for external frontends)",
    "立即启动/停止": "Start / Stop",
    "界面与其它": "UI & Misc",
    "主题": "Theme",
    "界面缩放": "UI Scale",
    "界面语言": "Language",
    "启用硬件实时监控": "Enable hardware monitoring",
    "显示下载悬浮球(右下角,可开关)": "Show download ball (bottom-right)",
    "模型下载源": "Model Download Source",
    "模型下载源": "Model Download Source",
    "依赖下载镜像(pip)": "pip Mirror",
    "依赖下载镜像": "pip Mirror",
    "测试各镜像": "Test Mirrors",
    "系统重置": "System Reset",
    "↺ 恢复默认设置": "↺ Reset Settings",
    "☠ 恢复出厂设置": "☠ Factory Reset",
    "💾 保存全部设置": "💾 Save All Settings",
    "自动(三源故障转移,国内镜像优先)": "Auto (3 sources failover, CN mirror first)",
    "官方 HuggingFace": "Official HuggingFace",
    "hf-mirror.com 国内镜像": "hf-mirror.com (CN mirror)",
    "ModelScope 魔搭(国内最稳)": "ModelScope (fastest in CN)",
    "自动(依次尝试全部镜像)": "Auto (try all mirrors)",
    "保存成功": "Saved",
    "语言设置将在重启后生效": "Language takes effect after restart",
    "提示:镜像用于加速 pip 依赖下载(如 llama-cpp-python、torch)":
        "Mirror speeds up pip downloads (e.g. llama-cpp-python, torch)",
    "自定义清单地址(可选)": "Custom manifest URL (optional)",
    "留空则使用内置模型清单": "Leave empty to use the built-in catalog",
    "缩放越大文字/控件越大(部分布局重启后完全生效)":
        "Larger scale = larger UI (some layouts apply after restart)",

    # ---------- 通用 / 消息框 ----------
    "本地AI引擎 就绪": "Local AI engine ready",
    "✅ 有效 GGUF": "✅ Valid GGUF",
    "✅ OpenVINO 模型目录": "✅ OpenVINO model dir",
    "无效文件": "Invalid file",
    "更新源配置": "Update Source",
    "默认从 GitHub Releases 检查最新版本;勾选下方可改用自定义更新源。":
        "Checks the latest version from GitHub Releases; tick below to use a custom source.",
    "自定义更新源地址(JSON: version/url/notes):":
        "Custom update source URL (JSON: version/url/notes):",
    "⬆️ 立即更新": "⬆️ Update Now",
    "🌐 打开下载页": "🌐 Open Download Page",
    "自动下载并安装新版本(仅打包版可用)":
        "Download & install the new version (packaged build only)",
    "确认更新": "Confirm Update",
    "即将自动更新": "Auto-update starting",
    "训练日志:等待开始训练...": "Training log: waiting to start...",
    "依赖日志:点击「检测全部依赖」开始扫描": "Dependency log: click Detect All to scan",
    "提示": "Info",
    "错误": "Error",
    "取消": "Cancel",
    "确定": "OK",
    "成功": "Success",
    "失败": "Failed",
    "完成": "Done",
    "是": "Yes",
    "否": "No",
    "正在加载模型": "Loading model",
    "模型已加载": "Model loaded",
    "模型加载失败": "Model load failed",
    "无法加载": "Cannot load",
    "模型与 NPU 不兼容": "Model incompatible with NPU",
    "已回退到 CPU 运行": "Fell back to CPU",
    "OpenVINO 不可用": "OpenVINO unavailable",
    "引擎不可用": "Engine unavailable",
    "恢复默认设置": "Reset Settings",
    "已恢复默认设置": "Settings reset",
    "恢复出厂设置(高危)": "Factory reset (dangerous)",
    "恢复出厂设置": "Factory Reset",
    "出厂重置完成": "Factory reset done",
    "依赖缺失": "Missing dependencies",
    "Ollama 连接正常": "Ollama connected",
    "Ollama 连接失败": "Ollama connection failed",
    "Ollama 已启动": "Ollama started",
    "Ollama 启动失败": "Ollama start failed",
    "Ollama 加载失败": "Ollama load failed",
    "拉取失败": "Pull failed",
    "未找到": "Not found",
    "搜索失败": "Search failed",
    "搜索完成": "Search complete",
    "自检失败": "Probe failed",
    "节点自检": "Source probe",
    "导入失败": "Import failed",
    "保存失败": "Save failed",
    "无法打开": "Cannot open",
    "删除失败": "Delete failed",
    "拉取 Ollama 模型": "Pull Ollama model",
    "加载 Ollama 模型": "Load Ollama model",
    "Ollama 尚无模型,输入模型名拉取(如 qwen2.5:0.5b):":
        "No Ollama models yet. Enter a model name to pull (e.g. qwen2.5:0.5b):",
}

_lock = threading.Lock()
_lang = ZH


def set_language(lang: str) -> None:
    """设置界面语言("zh"/"en");非法值回退中文。"""
    global _lang
    with _lock:
        _lang = EN if lang == EN else ZH


def language() -> str:
    with _lock:
        return _lang


def tr(text: str) -> str:
    """返回当前语言下的文本;英文模式下未收录则原样返回(中文兜底)。"""
    if _lang == EN:
        return _T.get(text, text)
    return text
