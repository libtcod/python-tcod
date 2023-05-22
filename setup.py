#!/usr/bin/env python3
"""Python-tcod setup script."""
from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path

from setuptools import setup

SDL_VERSION_NEEDED = (2, 0, 5)

SETUP_DIR = Path(__file__).parent  # setup.py current directory


def get_package_data() -> list[str]:
    """Get data files which will be included in the main tcod/ directory."""
    BIT_SIZE, _ = platform.architecture()
    files = [
        "py.typed",
        "lib/LIBTCOD-CREDITS.txt",
        "lib/LIBTCOD-LICENSE.txt",
        "lib/README-SDL.txt",
    ]
    if "win32" in sys.platform:
        if BIT_SIZE == "32bit":
            files += ["x86/SDL2.dll"]
        else:
            files += ["x64/SDL2.dll"]
    if sys.platform == "darwin":
        files += ["SDL2.framework/Versions/A/SDL2"]
    return files


def check_sdl_version() -> None:
    """Check the local SDL version on Linux distributions."""
    if not sys.platform.startswith("linux"):
        return
    needed_version = "{}.{}.{}".format(*SDL_VERSION_NEEDED)
    try:
        sdl_version_str = subprocess.check_output(["sdl2-config", "--version"], universal_newlines=True).strip()
    except FileNotFoundError as exc:
        msg = (
            f"libsdl2-dev or equivalent must be installed on your system and must be at least version {needed_version}."
            "\nsdl2-config must be on PATH."
        )
        raise RuntimeError(msg) from exc
    print(f"Found SDL {sdl_version_str}.")
    sdl_version = tuple(int(s) for s in sdl_version_str.split("."))
    if sdl_version < SDL_VERSION_NEEDED:
        msg = f"SDL version must be at least {needed_version}, (found {sdl_version_str})"
        raise RuntimeError(msg)


if not (SETUP_DIR / "libtcod/src").exists():
    print("Libtcod submodule is uninitialized.")
    print("Did you forget to run 'git submodule update --init'?")
    sys.exit(1)

check_sdl_version()

setup(
    py_modules=["libtcodpy"],
    packages=["tcod", "tcod.sdl", "tcod.__pyinstaller"],
    package_data={"tcod": get_package_data()},
    cffi_modules=["build_libtcod.py:ffi"],
    platforms=["Windows", "MacOS", "Linux"],
)
