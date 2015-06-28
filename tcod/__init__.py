"""
    This module provides a simple CFFI API to libtcod.
    
    This port has large partial support for libtcod's C functions.
    Use tcod/libtcod_cdef.h in the source distribution to see specially what
    functions were exported and what new functions have been added by TDL.
    
    The ffi and lib variables should be familiar to anyone that has used CFFI
    before, otherwise it's time to read up on how they work:
    https://cffi.readthedocs.org/en/latest/using.html
    
    Bring any issues or requests to GitHub:
    https://github.com/HexDecimal/libtcod-cffi
"""
import sys as _sys

from .libtcod import lib, ffi, _lib, _ffi

def _import_library_functions(lib, no_functions=False):
    g = globals()
    for name in dir(lib):
        if name[:5] == 'TCOD_':
            if (isinstance(getattr(lib, name), ffi.CData) and 
                ffi.typeof(getattr(lib, name)) == ffi.typeof('TCOD_color_t')):
                g[name[5:]] = Color.from_tcod(getattr(lib, name))
            elif name.isupper():
                g[name[5:]] = getattr(lib, name) # const names
            else:
                if not no_functions:
                    g[name[5:]] = getattr(lib, name) # function names
        elif name[:6] == 'TCODK_': # key name
            g['KEY_' + name[6:]] = getattr(lib, name)
        elif name[:4] == 'TCOD': # short constant names
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
    
    def __init__(self, r=0, g=0, b=0):
        self[:] = [r,g,b]
        #print(self)
        
    @classmethod
    def from_tcod(cls, tcod_color):
        # new in libtcod-cffi
        self = cls.__new__(cls)
        self.__init__(tcod_color.r, tcod_color.g, tcod_color.b)
        return self
    
    @classmethod
    def from_int(cls, integer): # 0xRRGGBB
        # new in libtcod-cffi
        return cls.from_tcod(lib.TDL_color_from_int(integer))

    def __eq__(self, other):
        return lib.TCOD_color_equals(self, other)

    def __mul__(self, other):
        if isinstance(other,(Color,list,tuple)):
            return lib.TCOD_color_multiply(self, other)
        else:
            return lib.TCOD_color_multiply_scalar(self, other)

    def __add__(self, other):
        return lib.TCOD_color_add(self, other)

    def __sub__(self, other):
        return lib.TCOD_color_subtract(self, other)

    def __repr__(self):
        return "%s%r" % (self.__class__.__name__, list.__repr__(self))
        
    def __getattr__(self, name):
        if name == 'r':
            return self[0]
            #return self._color >> 16 & 0xff
        if name == 'g':
            return self[1]
            #return self._color >> 8 & 0xff
        if name == 'b':
            return self[2]
            #return self._color & 0xff
        raise AttributeError('%r object has no attribute %r' %
                             (self.__class__.__name__, name))
        
        
    def __setattr__(self, name, value):
        if name == 'r':
            self[0] = value & 0xff
            #self._color = (self._color & 0x00ffff) | ((value & 0xff) << 16)
        elif name == 'g':
            self[1] = value & 0xff
            #self._color = (self._color & 0xff00ff) | ((value & 0xff) << 8)
        elif name == 'b':
            self[2] = value & 0xff
            #self._color = (self._color & 0xffff00) | (value & 0xff)
        else:
            object.__setattr__(self, name, value)

    #def __setitem__(self, i, c):
    #    if type(i) == str:
    #        setattr(self, i, c)
    #    else:
    #        setattr(self, "rgb"[i], c)

    def __int__(self):
        # new in libtcod-cffi
        return lib.TDL_color_RGB(*self)
 
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

def Dice(*args):
     return ffi.new('TCOD_dice_t *', args)
    
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
    return _lib.TCOD_struct_get_name(struct)

def struct_is_mandatory(struct, name):
    return _lib.TCOD_struct_is_mandatory(struct, name)

def struct_get_type(struct, name):
    return _lib.TCOD_struct_get_type(struct, name)

    
_import_library_functions(lib, True) # depends on Color
    
from . import bsp
from . import color
from . import console
from . import dijkstra
from . import heightmap
from . import image
from . import line
from . import map
from . import mouse
from . import namegen
from . import noise
from . import parser
from . import path
from . import random
from . import sys_ as sys

_import_module_functions(bsp)
_import_module_functions(color)
_import_module_functions(console)
_import_module_functions(dijkstra)
_import_module_functions(heightmap)
_import_module_functions(image)
_import_module_functions(line)
_import_module_functions(map)
_import_module_functions(mouse)
_import_module_functions(namegen)
_import_module_functions(noise)
_import_module_functions(parser)
_import_module_functions(path)
_import_module_functions(random)
_import_module_functions(sys)

# allow "import tcod.sys"
_sys.modules['tcod.sys'] = _sys.modules['tcod.sys_'] 

# tcod.line became both a module and a function due to the naming scheme
class _ModuleProxy():
    def __init__(self, module):
        for name in dir(module):
            setattr(self, name, getattr(module, name))
line = _ModuleProxy(line)
line.__call__ = line.line

__all__ = [name for name in list(globals()) if name[0] != '_']
