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


def _import_library_functions(lib):
    # imports libtcod namespace into thie module
    # does not override existing names
    g = globals()
    for name in dir(lib):
        if name[:5] == 'TCOD_':
            if (isinstance(getattr(lib, name), ffi.CData) and
                ffi.typeof(getattr(lib, name)) == ffi.typeof('TCOD_color_t')):
                g[name[5:]] = Color.from_cdata(getattr(lib, name))
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

class _MockFFI(object):
    def def_extern(self):
        return lambda func:func

if _os.environ.get('READTHEDOCS'):
    # Mock the lib and ffi objects needed to compile docs for readthedocs.io
    # Allows an import without building the cffi module first.
    lib = object()
    ffi = _MockFFI()
    RENDERER_SDL = 2
    FONT_LAYOUT_ASCII_INCOL = 1
    BKGND_SET = 1
    BKGND_DEFAULT = 13
    KEY_RELEASED = 2
    NOISE_DEFAULT = 0
else:
    from tcod._libtcod import lib, ffi

class Color(list):
    """

    Args:
        r (int): Red value, from 0 to 255.
        g (int): Green value, from 0 to 255.
        b (int): Blue value, from 0 to 255.
    """

    def __init__(self, r=0, g=0, b=0):
        self[:] = (r & 0xff, g & 0xff, b & 0xff)

    def _get_r(self):
        """int: Red value, always normalised to 0-255."""
        return self[0]
    def _set_r(self, value):
        self[0] = value & 0xff

    def _get_g(self):
        """int: Green value, always normalised to 0-255."""
        return self[1]
    def _set_g(self, value):
        self[1] = value & 0xff

    def _get_b(self):
        """int: Blue value, always normalised to 0-255."""
        return self[2]
    def _set_b(self, value):
        self[2] = value & 0xff

    r = property(_get_r, _set_r)
    g = property(_get_g, _set_g)
    b = property(_get_b, _set_b)

    @classmethod
    def from_cdata(cls, tcod_color_t):
        """new in libtcod-cffi"""
        return cls(tcod_color_t.r, tcod_color_t.g, tcod_color_t.b)


    @classmethod
    def from_int(cls, integer):
        """a TDL int color: 0xRRGGBB

        new in libtcod-cffi"""
        return cls(lib.TDL_color_from_int(integer))

    def __eq__(self, other):
        return (isinstance(other, (Color)) and
                lib.TCOD_color_equals(self, other))

    def __mul__(self, other):
        if isinstance(other, (Color, list, tuple)):
            return Color.from_cdata(lib.TCOD_color_multiply(self,
                                                 other))
        else:
            return Color.from_cdata(lib.TCOD_color_multiply_scalar(self,
                                                        other))

    def __add__(self, other):
        return Color.from_cdata(lib.TCOD_color_add(self, other))

    def __sub__(self, other):
        return Color.from_cdata(lib.TCOD_color_subtract(self, other))

    def __repr__(self):
        return "%s(%i,%i,%i)" % (self.__class__.__name__,
                                 self.r, self.g, self.b)

    def __iter__(self):
        return iter((self.r, self.g, self.b))

    def __int__(self):
        # new in libtcod-cffi
        return lib.TDL_color_RGB(*self)


_import_library_functions(lib)

__all__ = [_name for _name in list(globals()) if _name[0] != '_']
