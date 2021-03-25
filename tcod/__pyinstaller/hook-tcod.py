"""PyInstaller hook for tcod.

Added here after tcod 12.0.0.

If this hook is modified then the contributed hook needs to be removed from:
https://github.com/pyinstaller/pyinstaller-hooks-contrib
"""
from PyInstaller.utils.hooks import collect_dynamic_libs  # type: ignore

hiddenimports = ["_cffi_backend"]

# Install shared libraries to the working directory.
binaries = collect_dynamic_libs("tcod")
