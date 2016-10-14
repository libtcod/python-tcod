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
from __future__ import absolute_import as _

import os as _os

import re as _re

from tcod.libtcodpy import *
from tcod.tcod import *

with open(_os.path.join(__path__[0], 'version.txt'), 'r') as _f:
    # exclude the git commit number (PEP 396)
    __version__ = _re.match(r'([0-9]+)\.([0-9]+).*?', _f.read()).groups()
    assert __version__, 'version.txt parse error'

__all__ = [name for name in list(globals()) if name[0] != '_']
