"""The fast Python port of libtcod.

This module can be used as a drop in replacement for the official libtcodpy module.

Bring any issues or feature requests to GitHub: https://github.com/libtcod/python-tcod

Read the documentation online: https://python-tcod.readthedocs.io/en/latest/
"""

from __future__ import annotations

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

from tcod import bsp, color, console, context, event, image, los, map, noise, path, random, tileset  # noqa: A004
from tcod.cffi import __sdl_version__, ffi, lib
from tcod.tcod import __getattr__  # noqa: F401
from tcod.version import __version__

__all__ = [
    "Console",
    "__sdl_version__",
    "__version__",
    "bsp",
    "color",
    "console",
    "context",
    "event",
    "ffi",
    "image",
    "lib",
    "los",
    "map",
    "noise",
    "path",
    "random",
    "tileset",
    "tileset",
]
