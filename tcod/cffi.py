"""This module handles loading of the libtcod cffi API."""

from __future__ import annotations

import logging
import os
import platform
import sys
from pathlib import Path
from typing import Any

import cffi

logger = logging.getLogger("tcod")

__sdl_version__ = ""

REQUIRED_SDL_VERSION = (3, 2, 0)

ffi_check = cffi.FFI()
ffi_check.cdef(
    """
int SDL_GetVersion(void);
"""
)


def verify_dependencies() -> None:
    """Try to make sure dependencies exist on this system."""
    if sys.platform == "win32":
        lib_test: Any = ffi_check.dlopen("SDL3.dll")  # Make sure SDL3.dll is here.
        int_version = lib_test.SDL_GetVersion()  # Need to check this version.
        major = int_version // 1000000
        minor = (int_version // 1000) % 1000
        patch = int_version % 1000
        version_tuple = major, minor, patch
        if version_tuple < REQUIRED_SDL_VERSION:
            msg = f"Tried to load an old version of SDL {version_tuple!r}"
            raise RuntimeError(msg)


def get_architecture() -> str:
    """Return the Windows architecture, one of "x86" or "x64"."""
    return "x86" if platform.architecture()[0] == "32bit" else "x64"


def get_sdl_version() -> str:
    int_version = lib.SDL_GetVersion()
    return f"{int_version // 1000000}.{(int_version // 1000) % 1000}.{int_version % 1000}"


if sys.platform == "win32":
    # add Windows dll's to PATH
    os.environ["PATH"] = f"""{Path(__file__).parent / get_architecture()}{os.pathsep}{os.environ["PATH"]}"""


verify_dependencies()
from tcod._libtcod import ffi, lib  # noqa: E402

__sdl_version__ = get_sdl_version()


@ffi.def_extern()  # type: ignore[misc]
def _libtcod_log_watcher(message: Any, _userdata: None) -> None:  # noqa: ANN401
    text = str(ffi.string(message.message), encoding="utf-8")
    source = str(ffi.string(message.source), encoding="utf-8")
    level = int(message.level)
    lineno = int(message.lineno)
    logger.log(level, "%s:%d:%s", source, lineno, text)


lib.TCOD_set_log_callback(lib._libtcod_log_watcher, ffi.NULL)
lib.TCOD_set_log_level(0)

__all__ = ["__sdl_version__", "ffi", "lib"]
