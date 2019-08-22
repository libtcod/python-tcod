"""This module handles loading of the libtcod cffi API.
"""
import sys
import os

import cffi  # type: ignore
import platform
from typing import Any  # noqa: F401

from tcod import __path__  # type: ignore

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
        lib_test = ffi_check.dlopen(  # Make sure SDL2.dll is here.
            os.path.join(__path__[0], get_architecture(), "SDL2.dll")
        )
        version = ffi_check.new("struct SDL_version*")
        lib_test.SDL_GetVersion(version)  # Need to check this version.
        version = version.major, version.minor, version.patch
        if version < (2, 0, 5):
            raise RuntimeError(
                "Tried to load an old version of SDL %r" % (version,)
            )
        try:
            ffi_check.dlopen("vcruntime140.dll")  # Make sure VC++ 2015 exists.
        except OSError:
            print(
                "You will need to install 'vc_redist.{arch}.exe'"
                " from Microsoft at:\n"
                "https://support.microsoft.com/en-us/help/2977003/"
                "the-latest-supported-visual-c-downloads\n".format(
                    arch=get_architecture()
                ),
                file=sys.stderr,
            )
            raise


def get_architecture() -> str:
    """Return the Windows architecture, one of "x86" or "x64"."""
    return "x86" if platform.architecture()[0] == "32bit" else "x64"


def get_sdl_version() -> str:
    sdl_version = ffi.new("SDL_version*")
    lib.SDL_GetVersion(sdl_version)
    return "%s.%s.%s" % (
        sdl_version.major,
        sdl_version.minor,
        sdl_version.patch,
    )


if sys.platform == "win32":
    # add Windows dll's to PATH
    _bits, _linkage = platform.architecture()
    os.environ["PATH"] = "%s;%s" % (
        os.path.join(__path__[0], get_architecture()),
        os.environ["PATH"],
    )


class _Mock(object):
    """Mock object needed for ReadTheDocs."""

    CData = ()  # This gets passed to an isinstance call.

    @staticmethod
    def def_extern() -> Any:
        """Pass def_extern call silently."""
        return lambda func: func

    def __getattr__(self, attr: Any) -> Any:
        """This object pretends to have everything."""
        return self

    def __call__(self, *args: Any, **kargs: Any) -> Any:
        """Suppress any other calls"""
        return self

    def __str__(self) -> Any:
        """Just have ? in case anything leaks as a parameter default."""
        return "?"


lib = None  # type: Any
ffi = None  # type: Any

if os.environ.get("READTHEDOCS"):
    # Mock the lib and ffi objects needed to compile docs for readthedocs.io
    # Allows an import without building the cffi module first.
    lib = ffi = _Mock()
else:
    verify_dependencies()
    from tcod._libtcod import lib, ffi  # type: ignore # noqa: F401

    __sdl_version__ = get_sdl_version()

__all__ = ["ffi", "lib"]
