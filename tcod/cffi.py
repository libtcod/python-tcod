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

ffi_check = cffi.FFI()
ffi_check.cdef(
    """
typedef struct SDL_version
{
    uint8_t major;
    uint8_t minor;
    uint8_t patch;
} SDL_version;

void SDL_GetVersion(SDL_version * ver);
"""
)


def verify_dependencies() -> None:
    """Try to make sure dependencies exist on this system."""
    if sys.platform == "win32":
        lib_test: Any = ffi_check.dlopen("SDL2.dll")  # Make sure SDL2.dll is here.
        version: Any = ffi_check.new("struct SDL_version*")
        lib_test.SDL_GetVersion(version)  # Need to check this version.
        version_tuple = version.major, version.minor, version.patch
        if version_tuple < (2, 0, 5):
            msg = f"Tried to load an old version of SDL {version_tuple!r}"
            raise RuntimeError(msg)


def get_architecture() -> str:
    """Return the Windows architecture, one of "x86" or "x64"."""
    return "x86" if platform.architecture()[0] == "32bit" else "x64"


def get_sdl_version() -> str:
    sdl_version = ffi.new("SDL_version*")
    lib.SDL_GetVersion(sdl_version)
    return f"{sdl_version.major}.{sdl_version.minor}.{sdl_version.patch}"


if sys.platform == "win32":
    # add Windows dll's to PATH
    os.environ["PATH"] = f"""{Path(__file__).parent / get_architecture()}{os.pathsep}{os.environ["PATH"]}"""


verify_dependencies()
from tcod._libtcod import ffi, lib  # noqa

__sdl_version__ = get_sdl_version()


@ffi.def_extern()  # type: ignore
def _libtcod_log_watcher(message: Any, userdata: None) -> None:  # noqa: ANN401
    text = str(ffi.string(message.message), encoding="utf-8")
    source = str(ffi.string(message.source), encoding="utf-8")
    level = int(message.level)
    lineno = int(message.lineno)
    logger.log(level, "%s:%d:%s", source, lineno, text)


lib.TCOD_set_log_callback(lib._libtcod_log_watcher, ffi.NULL)
lib.TCOD_set_log_level(0)

__all__ = ["ffi", "lib", "__sdl_version__"]
