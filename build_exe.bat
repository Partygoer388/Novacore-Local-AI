@echo off
chcp 65001 >nul
REM =============================================================
REM  NovaCore-Local v0.0.1-alpha 打包脚本
REM  生成 dist\NovaCore-Local\NovaCore-Local.exe (开箱即用)
REM  不包含 novacore_data\ 下的测试数据/聊天记录/配置
REM =============================================================

cd /d "%~dp0"

echo [1/3] 清理旧构建产物...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist NovaCore-Local.spec del /q NovaCore-Local.spec

echo [2/3] 检查 PyInstaller...
python -c "import PyInstaller" 2>nul
if errorlevel 1 (
    echo 未检测到 PyInstaller, 正在安装...
    python -m pip install pyinstaller --quiet
)

echo [3/3] 开始打包 (这可能需要几分钟)...
REM 定位 OpenVINO 与 Tokenizers 的 DLL 目录(PyInstaller 不会自动收集动态加载的插件)
for /f "delims=" %%i in ('python -c "import openvino,os;print(os.path.join(os.path.dirname(openvino.__file__),'libs'))"') do set OV_LIBS=%%i
for /f "delims=" %%i in ('python -c "import openvino_tokenizers,os;print(os.path.join(os.path.dirname(openvino_tokenizers.__file__),'lib'))"') do set OVTOK_LIB=%%i
pyinstaller ^
  --noconfirm ^
  --clean ^
  --windowed ^
  --name "NovaCore-Local" ^
  --icon "novacore.ico" ^
  --hidden-import=PyQt6.QtCore ^
  --hidden-import=PyQt6.QtGui ^
  --hidden-import=PyQt6.QtWidgets ^
  --hidden-import=psutil ^
  --hidden-import=requests ^
  --hidden-import=fastapi ^
  --hidden-import=uvicorn ^
  --hidden-import=uvicorn.logging ^
  --hidden-import=uvicorn.loops ^
  --hidden-import=uvicorn.loops.auto ^
  --hidden-import=uvicorn.protocols ^
  --hidden-import=uvicorn.protocols.http ^
  --hidden-import=uvicorn.protocols.http.auto ^
  --hidden-import=uvicorn.protocols.websockets ^
  --hidden-import=uvicorn.protocols.websockets.auto ^
  --hidden-import=uvicorn.lifespan ^
  --hidden-import=uvicorn.lifespan.on ^
  --hidden-import=pynvml ^
  --hidden-import=openvino ^
  --hidden-import=openvino_genai ^
  --hidden-import=openvino_tokenizers ^
  --collect-all=llama_cpp ^
  --hidden-import=gguf ^
  --hidden-import=numpy ^
  --add-data="novacore.ico;." ^
  --add-data="%OV_LIBS%;openvino\libs" ^
  --add-data="%OVTOK_LIB%\openvino_tokenizers.dll;openvino_tokenizers\lib" ^
  --collect-submodules=novacore ^
  --exclude-module=torch ^
  --exclude-module=torchvision ^
  --exclude-module=transformers ^
  --exclude-module=tensorflow ^
  --exclude-module=matplotlib ^
  --exclude-module=scipy ^
  --exclude-module=pandas ^
  novacore_main.py

if errorlevel 1 (
    echo.
    echo ❌ 打包失败! 请查看上方错误信息。
    pause
    exit /b 1
)

echo.
echo ========================================================
echo  ✅ 打包完成!
echo  输出目录: %cd%\dist\NovaCore-Local\
echo  可执行文件: %cd%\dist\NovaCore-Local\NovaCore-Local.exe
echo.
echo  首次运行会在 exe 同级目录自动创建 novacore_data\
echo  (配置/模型/对话等), 完全开箱即用, 不含测试数据。
echo ========================================================
pause
