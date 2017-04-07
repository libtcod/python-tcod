"""This module handles loading of the libtcod cffi API.
"""
from __future__ import absolute_import as _

import sys as _sys
import os as _os

import platform as _platform


from tcod import __path__

if _sys.platform == 'win32':
    # add Windows dll's to PATH
    _bits, _linkage = _platform.architecture()
    _os.environ['PATH'] += (';' +
        _os.path.join(__path__[0], 'x86/' if _bits == '32bit' else 'x64'))

def _import_library_functions(lib):
    # imports libtcod namespace into thie module
    # does not override existing names
    g = globals()
    for name in dir(lib):
        if name[:5] == 'TCOD_':
            if name.isupper():
                g[name[5:]] = getattr(lib, name) # const names
        elif name.startswith('FOV'):
            g[name] = getattr(lib, name) # fov const names
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

class _Mock(object):
    """Mock object needed for ReadTheDocs."""

    TCOD_RENDERER_SDL = 2
    TCOD_FONT_LAYOUT_ASCII_INCOL = 1
    TCOD_BKGND_SET = 1
    TCOD_BKGND_DEFAULT = 13
    TCOD_KEY_RELEASED = 2
    TCOD_NOISE_DEFAULT = 0
    TCOD_NOISE_SIMPLEX = 2
    TCOD_NOISE_WAVELET = 4
    FOV_RESTRICTIVE = 12

    TCOD_RNG_MT = 0
    TCOD_RNG_CMWC = 1

    CData = () # This gets passed to an isinstance call.

    def def_extern(self):
        """Pass def_extern call silently."""
        return lambda func:func

    def __getattr__(self, attr):
        """This object pretends to have everything."""
        return self

    def __call__(self, *args, **kargs):
        """Suppress any other calls"""
        return self

    def __str__(self):
        """Just have ? in case anything leaks as a parameter default."""
        return '?'


if _os.environ.get('READTHEDOCS'):
    # Mock the lib and ffi objects needed to compile docs for readthedocs.io
    # Allows an import without building the cffi module first.
    lib = ffi = _Mock()
else:
    from tcod._libtcod import lib, ffi

_import_library_functions(lib)
