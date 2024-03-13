"""PyInstaller hook for tcod.

There were added since tcod 12.0.0.

If this hook is ever modified then the contributed hook needs to be removed from:
https://github.com/pyinstaller/pyinstaller-hooks-contrib
"""

from PyInstaller.utils.hooks import collect_dynamic_libs  # type: ignore

hiddenimports = ["_cffi_backend"]

# Install shared libraries to the working directory.
binaries = collect_dynamic_libs("tcod")
