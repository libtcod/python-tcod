"""
    Hook for https://github.com/HexDecimal/python-tdl
"""
from PyInstaller.utils.hooks import collect_dynamic_libs

hiddenimports = ['_cffi_backend']

# Install shared libraries to the working directory.
binaries = collect_dynamic_libs('tcod', destdir='.')
