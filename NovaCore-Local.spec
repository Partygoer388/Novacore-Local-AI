# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_submodules

hiddenimports = ['PyQt6.QtCore', 'PyQt6.QtGui', 'PyQt6.QtWidgets', 'psutil', 'requests', 'fastapi', 'uvicorn', 'uvicorn.logging', 'uvicorn.loops', 'uvicorn.loops.auto', 'uvicorn.protocols', 'uvicorn.protocols.http', 'uvicorn.protocols.http.auto', 'uvicorn.protocols.websockets', 'uvicorn.protocols.websockets.auto', 'uvicorn.lifespan', 'uvicorn.lifespan.on', 'pynvml', 'openvino', 'openvino_genai', 'openvino_tokenizers', 'gguf', 'numpy']
hiddenimports += collect_submodules('novacore')


a = Analysis(
    ['novacore_main.py'],
    pathex=[],
    binaries=[],
    datas=[('novacore.ico', '.'), ('C:\\Users\\Dylan\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\openvino\\libs', 'openvino\\libs'), ('C:\\Users\\Dylan\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\openvino_tokenizers\\lib\\openvino_tokenizers.dll', 'openvino_tokenizers\\lib')],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['torch', 'torchvision', 'transformers', 'tensorflow', 'matplotlib', 'scipy', 'pandas'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='NovaCore-Local',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['novacore.ico'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='NovaCore-Local',
)
