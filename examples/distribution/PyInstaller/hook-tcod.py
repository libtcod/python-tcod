
from PyInstaller.utils.hooks import collect_dynamic_libs

hiddenimports = ['_cffi_backend']

# Install the SDL2 shared library to the working directory.
binaries = [(lib, '') for lib, _ in collect_dynamic_libs('tcod')]
