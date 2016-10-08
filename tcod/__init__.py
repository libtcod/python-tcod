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
import os as _os
import sys as _sys

import re as _re

from .libtcod import lib, ffi, _lib, _ffi, _unpack_char_p

def _import_library_functions(lib, no_functions=False):
    # imports libtcod namespace into thie module
    # does not override existing names
    g = globals()
    for name in dir(lib):
        if name[:5] == 'TCOD_':
            if name[5:] in g:
                continue
            if (isinstance(getattr(lib, name), ffi.CData) and
                ffi.typeof(getattr(lib, name)) == ffi.typeof('TCOD_color_t')):
                g[name[5:]] = FrozenColor.from_cdata(getattr(lib, name))
            elif name.isupper():
                g[name[5:]] = getattr(lib, name) # const names
            else:
                if not no_functions:
                    g[name[5:]] = getattr(lib, name) # function names
        elif name[:6] == 'TCODK_': # key name
            if 'KEY_' + name[6:] in g:
                continue
            g['KEY_' + name[6:]] = getattr(lib, name)
        elif name[:4] == 'TCOD': # short constant names
            if name[4:] in g:
                continue
            g[name[4:]] = getattr(lib, name)

def _import_module_functions(module):
    g = globals()
    mod_name = module.__name__.rsplit('.')[-1]
    if mod_name[-1] == '_': # remove underscore from sys_.py
        mod_name = mod_name[:-1]
    for name in module.__all__:
        g['%s_%s' % (mod_name, name)] = getattr(module, name)

# ----------------------------------------------------------------------------
# the next functions are used to mimic the rest of the libtcodpy functionality

class Color(list):
    '''list-like behaviour could change in the future'''

    def __init__(self, *rgb):
        if len(rgb) < 3:
            rgb += (0,) * (3 - len(rgb))
        self[:] = rgb

    @classmethod
    def from_cdata(cls, tcod_color):
        '''new in libtcod-cffi'''
        return cls(tcod_color.r, tcod_color.g, tcod_color.b)

    @classmethod
    def from_int(cls, integer):
        '''a TDL int color: 0xRRGGBB

        new in libtcod-cffi'''
        return cls.from_cdata(lib.TDL_color_from_int(integer))

    def __eq__(self, other):
        return (isinstance(other, (Color, FrozenColor)) and
                _lib.TCOD_color_equals(self, other))

    def __mul__(self, other):
        if isinstance(other,(list, tuple)):
            return Color.from_cdata(_lib.TCOD_color_multiply(self, other))
        else:
            return Color.from_cdata(_lib.TCOD_color_multiply_scalar(self, other))

    def __add__(self, other):
        return Color.from_cdata(_lib.TCOD_color_add(self, other))

    def __sub__(self, other):
        return Color.from_cdata(_lib.TCOD_color_subtract(self, other))

    def __repr__(self):
        return "<%s%s>" % (self.__class__.__name__, list.__repr__(self))

    def _get_r(self):
        return self[0]

    def _set_r(self, val):
        self[0] = val & 0xff

    def _get_g(self):
        return self[1]

    def _set_g(self, val):
        self[1] = val & 0xff

    def _get_b(self):
        return self[2]

    def _set_b(self, val):
        self[2] = val & 0xff

    r = property(_get_r, _set_r)
    g = property(_get_g, _set_g)
    b = property(_get_b, _set_b)

    def __int__(self):
        # new in libtcod-cffi
        return lib.TDL_color_RGB(*self)


class FrozenColor(tuple):
    '''new in libtcod-cffi'''
    def __new__(cls, *rgb):
        if len(rgb) < 3:
            rgb += (0,) * (3 - len(rgb))
        return tuple.__new__(cls, (rgb))

    @classmethod
    def from_cdata(cls, tcod_color):
        return cls(tcod_color.r, tcod_color.g, tcod_color.b)

    @classmethod
    def from_int(cls, integer):
        return cls.from_cdata(lib.TDL_color_from_int(integer))

    __mul__ = Color.__mul__
    __add__ = Color.__add__
    __sub__ = Color.__sub__

    def __repr__(self):
        return "<%s%s>" % (self.__class__.__name__, tuple.__repr__(self))

    _get_r = Color._get_r
    _get_g = Color._get_g
    _get_b = Color._get_b
    def _set_rgb(self):
        raise TypleError('can not assign colors directally to %s' %
                         (self.__class__))

    r = property(_get_r, _set_rgb)
    g = property(_get_g, _set_rgb)
    b = property(_get_b, _set_rgb)

    __int__ = Color.__int__

class Key(object):
    def __init__(self):
        self._struct = ffi.new('TCOD_key_t *')

    def __getattr__(self, name):
        if name == 'c':
            return ord(getattr(self._struct, name))
        return getattr(self._struct, name)

class Mouse(object):
    def __init__(self):
        self._struct = ffi.new('TCOD_mouse_t *')

    def __getattr__(self, name):
        return getattr(self._struct, name)

class ConsoleBuffer:
    # simple console that allows direct (fast) access to cells. simplifies
    # use of the "fill" functions.
    def __init__(self, width, height, back_r=0, back_g=0, back_b=0, fore_r=0, fore_g=0, fore_b=0, char=' '):
        # initialize with given width and height. values to fill the buffer
        # are optional, defaults to black with no characters.
        n = width * height
        self.width = width
        self.height = height
        self.clear(back_r, back_g, back_b, fore_r, fore_g, fore_b, char)

    def clear(self, back_r=0, back_g=0, back_b=0, fore_r=0, fore_g=0, fore_b=0, char=' '):
        # clears the console. values to fill it with are optional, defaults
        # to black with no characters.
        n = self.width * self.height
        self.back_r = [back_r] * n
        self.back_g = [back_g] * n
        self.back_b = [back_b] * n
        self.fore_r = [fore_r] * n
        self.fore_g = [fore_g] * n
        self.fore_b = [fore_b] * n
        self.char = [ord(char)] * n

    def copy(self):
        # returns a copy of this ConsoleBuffer.
        other = ConsoleBuffer(0, 0)
        other.width = self.width
        other.height = self.height
        other.back_r = list(self.back_r)  # make explicit copies of all lists
        other.back_g = list(self.back_g)
        other.back_b = list(self.back_b)
        other.fore_r = list(self.fore_r)
        other.fore_g = list(self.fore_g)
        other.fore_b = list(self.fore_b)
        other.char = list(self.char)
        return other

    def set_fore(self, x, y, r, g, b, char):
        # set the character and foreground color of one cell.
        i = self.width * y + x
        self.fore_r[i] = r
        self.fore_g[i] = g
        self.fore_b[i] = b
        self.char[i] = ord(char)

    def set_back(self, x, y, r, g, b):
        # set the background color of one cell.
        i = self.width * y + x
        self.back_r[i] = r
        self.back_g[i] = g
        self.back_b[i] = b

    def set(self, x, y, back_r, back_g, back_b, fore_r, fore_g, fore_b, char):
        # set the background color, foreground color and character of one cell.
        i = self.width * y + x
        self.back_r[i] = back_r
        self.back_g[i] = back_g
        self.back_b[i] = back_b
        self.fore_r[i] = fore_r
        self.fore_g[i] = fore_g
        self.fore_b[i] = fore_b
        self.char[i] = ord(char)

    def blit(self, dest, fill_fore=True, fill_back=True):
        # use libtcod's "fill" functions to write the buffer to a console.
        if (console_get_width(dest) != self.width or
            console_get_height(dest) != self.height):
            raise ValueError('ConsoleBuffer.blit: Destination console has an incorrect size.')

        if fill_back:
            _lib.TCOD_console_fill_background(dest,
                                              _ffi.new('int[]', self.back_r),
                                              _ffi.new('int[]', self.back_g),
                                              _ffi.new('int[]', self.back_b))
        if fill_fore:
            _lib.TCOD_console_fill_foreground(dest,
                                              _ffi.new('int[]', self.fore_r),
                                              _ffi.new('int[]', self.fore_g),
                                              _ffi.new('int[]', self.fore_b))
            _lib.TCOD_console_fill_char(dest, _ffi.new('int[]', self.char))

# python class encapsulating the _CBsp pointer
class Bsp(object):
    def __init__(self, cnode):
        pcbsp = cnode
        self.p = pcbsp

    def getx(self):
        return self.p.x
    def setx(self, value):
        self.p.x = value
    x = property(getx, setx)

    def gety(self):
        return self.p.y
    def sety(self, value):
        self.p.y = value
    y = property(gety, sety)

    def getw(self):
        return self.p.w
    def setw(self, value):
        self.p.w = value
    w = property(getw, setw)

    def geth(self):
        return self.p.h
    def seth(self, value):
        self.p.h = value
    h = property(geth, seth)

    def getpos(self):
        return self.p.position
    def setpos(self, value):
        self.p.position = value
    position = property(getpos, setpos)

    def gethor(self):
        return self.p.horizontal
    def sethor(self,value):
        self.p.horizontal = value
    horizontal = property(gethor, sethor)

    def getlev(self):
        return self.p.level
    def setlev(self,value):
        self.p.level = value
    level = property(getlev, setlev)

class Dice(list):

    def __init__(self, nb_dices, nb_faces, multiplier, addsub):
        self[:] = (int(nb_dices), int(nb_faces), multiplier, addsub)

    @classmethod
    def from_cdata(cls, dice):
        return cls(dice.nb_rolls, dice.nb_faces, dice.multiplier, dice.addsub)

    def _get_nb_dices(self):
        return self[0]

    def _set_nb_dices(self, value):
        self[0] = value

    def _get_nb_faces(self):
        return self[1]

    def _set_nb_faces(self, value):
        self[1] = value

    def _get_multiplier(self):
        return self[2]

    def _set_multiplier(self, value):
        self[2] = value

    def _get_addsub(self):
        return self[3]

    def _set_addsub(self, value):
        self[3] = value

    nb_dices = property(_get_nb_dices, _set_nb_dices)
    nb_faces = property(_get_nb_faces, _set_nb_faces)
    multiplier = property(_get_multiplier, _set_multiplier)
    addsub = property(_get_addsub, _set_addsub)

    def __repr__(self):
        return "<Dice(%id%ix%s+(%s))>" % (self.nb_dices, self.nb_faces,
                                      self.multiplier, self.addsub)

class HeightMap(object):
    def __init__(self, chm):
        pchm = cast(chm, _CHeightMap)
        self.p = pchm

    def getw(self):
        return self.p.w
    def setw(self, value):
        self.p.w = value
    w = property(getw, setw)

    def geth(self):
        return self.p.h
    def seth(self, value):
        self.p.h = value
    h = property(geth, seth)


NOISE_DEFAULT_HURST = 0.5
NOISE_DEFAULT_LACUNARITY = 2.0

NOISE_DEFAULT = 0
NOISE_PERLIN = 1
NOISE_SIMPLEX = 2
NOISE_WAVELET = 4

def FOV_PERMISSIVE(p) :
    return FOV_PERMISSIVE_0+p

def BKGND_ALPHA(a):
    return BKGND_ALPH | (int(a * 255) << 8)

def BKGND_ADDALPHA(a):
    return BKGND_ADDA | (int(a * 255) << 8)

def struct_add_flag(struct, name):
    _lib.TCOD_struct_add_flag(struct, name)

def struct_add_property(struct, name, typ, mandatory):
    _lib.TCOD_struct_add_property(struct, name, typ, mandatory)

def struct_add_value_list(struct, name, value_list, mandatory):
    CARRAY = c_char_p * (len(value_list) + 1)
    cvalue_list = CARRAY()
    for i in range(len(value_list)):
        cvalue_list[i] = cast(value_list[i], c_char_p)
    cvalue_list[len(value_list)] = 0
    _lib.TCOD_struct_add_value_list(struct, name, cvalue_list, mandatory)

def struct_add_list_property(struct, name, typ, mandatory):
    _lib.TCOD_struct_add_list_property(struct, name, typ, mandatory)

def struct_add_structure(struct, sub_struct):
    _lib.TCOD_struct_add_structure(struct, sub_struct)

def struct_get_name(struct):
    return _unpack_char_p(_lib.TCOD_struct_get_name(struct))

def struct_is_mandatory(struct, name):
    return _lib.TCOD_struct_is_mandatory(struct, name)

def struct_get_type(struct, name):
    return _lib.TCOD_struct_get_type(struct, name)

_import_library_functions(lib, False)

from ._bsp import *
from ._color import *
from ._console import *
from ._dijkstra import *
from ._heightmap import *
from ._image import *
from ._line import *
from ._map import *
from ._mouse import *
from ._namegen import *
from ._noise import *
from ._parser import *
from ._path import *
from ._random import *
from ._sys import *

with open(_os.path.join(__path__[0], 'version.txt'), 'r') as _f:
    # exclude the git commit number (PEP 396)
    __version__ = _re.match(r'([0-9]+)\.([0-9]+).*?', _f.read()).groups()
    if not __version__:
        raise RuntimeError('version.txt parse error')

__all__ = [name for name in list(globals()) if name[0] != '_']
