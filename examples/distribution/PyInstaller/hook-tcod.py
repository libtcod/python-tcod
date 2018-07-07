"""
    Hook for https://github.com/libtcod/python-tcod
"""
from PyInstaller.utils.hooks import collect_dynamic_libs

hiddenimports = ['_cffi_backend']

# Install shared libraries to the working directory.
binaries = collect_dynamic_libs('tcod', destdir='.')
