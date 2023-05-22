"""This module handles loading of the libtcod cffi API."""
from __future__ import annotations

import os
import platform
import sys
from pathlib import Path
from typing import Any  # noqa: F401

import cffi

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


class _Mock(object):
    """Mock object needed for ReadTheDocs."""

    @staticmethod
    def def_extern() -> Any:
        """Pass def_extern call silently."""
        return lambda func: func

    def __getattr__(self, attr: str) -> None:
        """Return None on any attribute."""
        return None

    def __bool__(self) -> bool:
        """Allow checking for this mock object at import time."""
        return False


lib: Any = None
ffi: Any = None

if os.environ.get("READTHEDOCS"):
    # Mock the lib and ffi objects needed to compile docs for readthedocs.io
    # Allows an import without building the cffi module first.
    lib = ffi = _Mock()
else:
    verify_dependencies()
    from tcod._libtcod import ffi, lib  # type: ignore # noqa: F401

    __sdl_version__ = get_sdl_version()

__all__ = ["ffi", "lib"]
