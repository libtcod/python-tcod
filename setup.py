#!/usr/bin/env python
"""Python-tcod setup script."""

from __future__ import annotations

import os
import platform
import subprocess
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


def check_sdl_version() -> None:
    """Check the local SDL version on Linux distributions."""
    if not sys.platform.startswith("linux"):
        return
    needed_version = "{}.{}.{}".format(*SDL_VERSION_NEEDED)
    try:
        sdl_version_str = subprocess.check_output(
            ["pkg-config", "sdl3", "--modversion"],  # noqa: S607
            universal_newlines=True,
        ).strip()
    except FileNotFoundError:
        try:
            sdl_version_str = subprocess.check_output(["sdl3-config", "--version"], universal_newlines=True).strip()  # noqa: S607
        except FileNotFoundError as exc:
            msg = (
                f"libsdl3-dev or equivalent must be installed on your system and must be at least version {needed_version}."
                "\nsdl3-config must be on PATH."
            )
            raise RuntimeError(msg) from exc
    except subprocess.CalledProcessError as exc:
        if sys.version_info >= (3, 11):
            exc.add_note(f"Note: {os.environ.get('PKG_CONFIG_PATH')=}")
        raise
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
