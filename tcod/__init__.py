"""
    This module provides a simple CFFI API to libtcod.

    This port has large partial support for libtcod's C functions.
    Use tcod/libtcod_cdef.h in the source distribution to see specially what
    functions were exported and what new functions have been added by TDL.

    The ffi and lib variables should be familiar to anyone that has used CFFI
    before, otherwise it's time to read up on how they work:
    https://cffi.readthedocs.org/en/latest/using.html

    Otherwise this module can be used as a drop in replacement for the official
    libtcod.py module.

    Bring any issues or requests to GitHub:
    https://github.com/HexDecimal/libtcod-cffi
"""
from tcod.libtcod import lib, ffi, __sdl_version__  # noqa: F4
from tcod.libtcodpy import *  # noqa: F4

try:
    from tcod.version import __version__
except ImportError:  # Gets imported without version.py by ReadTheDocs
    __version__ = ""
