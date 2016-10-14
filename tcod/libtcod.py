"""This module handles loading of the libtcod cffi API.
"""
from __future__ import absolute_import as _

import sys as _sys
import os as _os

import platform as _platform

from tcod import __path__

# add Windows dll's to PATH
if _sys.platform == 'win32':
    _bits, _linkage = _platform.architecture()
    _os.environ['PATH'] += (';' +
        _os.path.join(__path__[0], 'x86/' if _bits == '32bit' else 'x64'))

from tcod._libtcod import lib, ffi

def _import_library_functions(lib):
    # imports libtcod namespace into thie module
    # does not override existing names
    g = globals()
    for name in dir(lib):
        if name[:5] == 'TCOD_':
            if (isinstance(getattr(lib, name), ffi.CData) and
                ffi.typeof(getattr(lib, name)) == ffi.typeof('TCOD_color_t')):
                g[name[5:]] = _FrozenColor.from_cdata(getattr(lib, name))
            elif name.isupper():
                g[name[5:]] = getattr(lib, name) # const names
            #else:
            #    g[name[5:]] = getattr(lib, name) # function names
        elif name[:6] == 'TCODK_': # key name
            g['KEY_' + name[6:]] = getattr(lib, name)

NOISE_DEFAULT_HURST = 0.5
NOISE_DEFAULT_LACUNARITY = 2.0

def FOV_PERMISSIVE(p) :
    return FOV_PERMISSIVE_0+p

def BKGND_ALPHA(a):
    return BKGND_ALPH | (int(a * 255) << 8)

def BKGND_ADDALPHA(a):
    return BKGND_ADDA | (int(a * 255) << 8)

from tcod.tcod import FrozenColor as _FrozenColor

_import_library_functions(lib)

__all__ = [_name for _name in list(globals()) if _name[0] != '_']
