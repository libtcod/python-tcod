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
                g[name[5:]] = Color._new_from_cdata(getattr(lib, name))
            elif name.isupper():
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
    FOV_RESTRICTIVE = 12

    CData = () # This gets passed to an isinstance call.
    
    def def_extern(self):
        """Pass def_extern call silently."""
        return lambda func:func

    def __getattr__(self, attr):
        """This object pretends to have everything."""
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

class Color(list):
    """

    Args:
        r (int): Red value, from 0 to 255.
        g (int): Green value, from 0 to 255.
        b (int): Blue value, from 0 to 255.
    """

    def __init__(self, r=0, g=0, b=0):
        self[:] = (r & 0xff, g & 0xff, b & 0xff)

    @property
    def r(self):
        """int: Red value, always normalised to 0-255."""
        return self[0]

    @r.setter
    def r(self, value):
        self[0] = value & 0xff

    @property
    def g(self):
        """int: Green value, always normalised to 0-255."""
        return self[1]
    @g.setter
    def g(self, value):
        self[1] = value & 0xff

    @property
    def b(self):
        """int: Blue value, always normalised to 0-255."""
        return self[2]
    @b.setter
    def b(self, value):
        self[2] = value & 0xff

    @classmethod
    def _new_from_cdata(cls, cdata):
        """new in libtcod-cffi"""
        return cls(cdata.r, cdata.g, cdata.b)


    @classmethod
    def _new_from_int(cls, integer):
        """a TDL int color: 0xRRGGBB

        new in libtcod-cffi"""
        return cls(lib.TDL_color_from_int(integer))

    def __eq__(self, other):
        """Compare equality between colors."""
        return (isinstance(other, (Color)) and
                lib.TCOD_color_equals(self, other))

    def __add__(self, other):
        """Add two colors together."""
        return Color._new_from_cdata(lib.TCOD_color_add(self, other))

    def __sub__(self, other):
        """Subtract one color from another."""
        return Color._new_from_cdata(lib.TCOD_color_subtract(self, other))

    def __mul__(self, other):
        """Multiply with a scaler or another color."""
        if isinstance(other, (Color, list, tuple)):
            return Color._new_from_cdata(lib.TCOD_color_multiply(self, other))
        else:
            return Color._new_from_cdata(
                lib.TCOD_color_multiply_scalar(self, other))

    def __bytes__(self):
        """Return this color in a format suited for color control."""
        return b'%c%c%c' % tuple(self)

    def __str__(self):
        """Return this color in a format suited for color control."""
        return '%c%c%c' % tuple(self)

    def __repr__(self):
        """Return a printable representation of the current color."""
        return "%s(%i,%i,%i)" % (self.__class__.__name__,
                                 self.r, self.g, self.b)

    def __int__(self):
        """Return this color as an integer in 0xRRGGBB format."""
        return lib.TDL_color_RGB(*self)


_import_library_functions(lib)

__all__ = [_name for _name in list(globals()) if _name[0] != '_']
