# -*- mode: python ; coding: utf-8 -*-

block_cipher = None


a = Analysis(
    ["main.py"],
    binaries=[],
    datas=[("data", "data")],  # Include all files in the 'data' directory.
    hiddenimports=[],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="start",  # Name of the executable.
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,  # Set to False to disable the Windows terminal.
    icon="icon.ico",  # Windows icon file.
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="hello_world",  # Name of the distribution directory.
)
