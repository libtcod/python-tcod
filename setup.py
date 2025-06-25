#!/usr/bin/env python
"""Python-tcod setup script."""

from __future__ import annotations

import platform
import sys
from pathlib import Path

from setuptools import setup

# ruff: noqa: T201

SDL_VERSION_NEEDED = (3, 2, 0)

SETUP_DIR = Path(__file__).parent  # setup.py current directory


def get_package_data() -> list[str]:
    """Get data files which will be included in the main tcod/ directory."""
    bit_size, _ = platform.architecture()
    files = [
        "py.typed",
        "lib/LIBTCOD-CREDITS.txt",
        "lib/LIBTCOD-LICENSE.txt",
        "lib/README-SDL.txt",
    ]
    if "win32" in sys.platform:
        if bit_size == "32bit":
            files += ["x86/SDL3.dll"]
        else:
            files += ["x64/SDL3.dll"]
    if sys.platform == "darwin":
        files += ["SDL3.framework/Versions/A/SDL3"]
    return files


if not (SETUP_DIR / "libtcod/src").exists():
    print("Libtcod submodule is uninitialized.")
    print("Did you forget to run 'git submodule update --init'?")
    sys.exit(1)


setup(
    py_modules=["libtcodpy"],
    packages=["tcod", "tcod.sdl", "tcod.__pyinstaller"],
    package_data={"tcod": get_package_data()},
    cffi_modules=["build_libtcod.py:ffi"],
    platforms=["Windows", "MacOS", "Linux"],
)
