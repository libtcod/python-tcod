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
import os as _os

from math import floor as _floor

import platform as _platform

def _get_lib_path_crossplatform():
    '''Locate the right DLL path for this OS'''
    bits, linkage = _platform.architecture()
    if 'win32' in _sys.platform:
        return 'lib/win32/'
    elif 'linux' in _sys.platform:
        if bits == '32bit':
            return 'lib/linux32/'
        elif bits == '64bit':
            return 'lib/linux64/'
    elif 'darwin' in _sys.platform:
        return 'lib/darwin/'
    raise ImportError('Operating system "%s" has no supported dynamic link libarary. (%s, %s)' % (_sys.platform, bits, linkage))

def _import_library_functions(lib):
    g = globals()
    for name in dir(lib):
        if name[:5] == 'TCOD_':
            if (isinstance(getattr(lib, name), ffi.CData) and 
                ffi.typeof(getattr(lib, name)) == ffi.typeof('TCOD_color_t')):
                g[name[5:]] = Color.from_tcod(getattr(lib, name))
            else:
                g[name[5:]] = getattr(lib, name) # function names
        elif name[:6] == 'TCODK_': # key name
            g['KEY_' + name[6:]] = getattr(lib, name)
        elif name[:4] == 'TCOD': # short constant names
            g[name[4:]] = getattr(lib, name)
    
# add dll's to PATH
_os.environ['PATH'] += ';' + _os.path.join(__path__[0],
                                           _get_lib_path_crossplatform())

# import the right .pyd file for this Python implementation
try:
    import _libtcod # PyPy
except ImportError:
    # get implementation specific version of _libtcod.pyd
    import importlib as _importlib
    _module_name = '._libtcod'
    if _platform.python_implementation() == 'CPython':
        _module_name += '_cp%i%i' % _sys.version_info[:2]
        if _platform.architecture()[0] == '64bit':
            _module_name += '_x64'

    _libtcod = _importlib.import_module(_module_name, 'tcod')

ffi = _libtcod.ffi
lib = _libtcod.lib


# ----------------------------------------------------------------------------
# the next functions are used to mimic the rest of the libtcodpy functionality

def Key(*args):
    return ffi.new('TCOD_key_t *', args)
    
def Mouse(*args):
    return ffi.new('TCOD_mouse_t *', args)

class Color(list):
    
    #def __new__(cls, r=0, g=0, b=0):
    #    self = list.__new__(cls)
    #    return self
        
    def __init__(self, r=0, g=0, b=0):
        self[:] = [r,g,b]
        #print(self)
        
    #@classmethod
    #def _convert(cls, color):
    #    if isinstance(color, int)
        
    @classmethod
    def from_tcod(cls, tcod_color):
        # new in libtcod-cffi
        self = cls.__new__(cls)
        self.__init__(tcod_color.r, tcod_color.g, tcod_color.b)
        #self[:] = [tcod_color.r, tcod_color.g, tcod_color.b]
        #print(self)
        #self.color = lib.TDL_color_to_int(tcod_color)
        return self
    
    @classmethod
    def from_int(cls, integer): # 0xRRGGBB
        # new in libtcod-cffi
        return cls.from_tcod(lib.TDL_color_from_int(integer))
        #self = cls.__new__(cls)
        #self[:] = lib.TDL
        #return self

    def __eq__(self, other):
        return lib.TCOD_color_equals(self, other)

    def __mul__(self, other):
        if isinstance(c,Color):
            return lib.TCOD_color_multiply(self, other)
        else:
            return lib.TCOD_color_multiply_scalar(self, other)

    def __add__(self, other):
        return lib.TCOD_color_add(self, c)

    def __sub__(self, c):
        return lib.TCOD_color_subtract(self, c)

    def __repr__(self):
        return "%s%r" % (self.__class__.__name__, list.__repr__(self))
        #return "%s(%d,%d,%d)" % (self.__class__.__name__,
        #                         self.r, self.g, self.b)

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
        
    #def __getitem__(self, i):
    #    if type(i) == str:
    #        return getattr(self, i)
    #    else:
    #        return getattr(self, "rgb"[i])

    #def __setitem__(self, i, c):
    #    if type(i) == str:
    #        setattr(self, i, c)
    #    else:
    #        setattr(self, "rgb"[i], c)

    def __int__(self):
        # new in libtcod-cffi
        return lib.TDL_color_RGB(*self)
            
    #def __iter__(self):
    #    return iter((self.tocd_color.r, self.tocd_color.g, self.tocd_color.b))
 
_import_library_functions(lib) # depends on Color
 
def color_lerp(c1, c2, a):
    return Color.from_tcod(lib.TCOD_color_lerp(c1, c2, a))

def color_set_hsv(c, h, s, v):
    lib.TCOD_color_set_HSV(byref(c), c_float(h), c_float(s), c_float(v))

def color_get_hsv(c):
    h = c_float()
    s = c_float()
    v = c_float()
    lib.TCOD_color_get_HSV(c, byref(h), byref(s), byref(v))
    return h.value, s.value, v.value

def color_scale_HSV(c, scoef, vcoef) :
    lib.TCOD_color_scale_HSV(byref(c),c_float(scoef),c_float(vcoef))

def color_gen_map(colors, indexes):
    ccolors = (Color * len(colors))(*colors)
    cindexes = (c_int * len(indexes))(*indexes)
    cres = (Color * (max(indexes) + 1))()
    lib.TCOD_color_gen_map(cres, len(colors), ccolors, cindexes)
    return cres
       
        
        
# console
        
def console_init_root(w, h, title, fullscreen=False, renderer=RENDERER_SDL):
    lib.TCOD_console_init_root(w, h, title, fullscreen, renderer)

def console_set_custom_font(fontFile, flags=FONT_LAYOUT_ASCII_INCOL,
                            nb_char_horiz=0, nb_char_vertic=0):
    lib.TCOD_console_set_custom_font(fontFile, flags,
                                     nb_char_horiz, nb_char_vertic)

def console_set_default_background(con, col):
    if not con:
        con = ffi.NULL
    lib.TCOD_console_set_default_background(con, col)

def console_set_default_foreground(con, col):
    if not con:
        con = ffi.NULL
    lib.TCOD_console_set_default_foreground(con, col)

                                     
def _encode_fmt(fmt):
    return (s.encode() if isinstance(s, str) else s for s in fmt)

def console_print(con, x, y, *fmt):
    if not con:
        con = ffi.NULL
    return lib.TCOD_console_print(con, x, y, *_encode_fmt(fmt))
    
def console_print_ex(con, x, y, flag, alignment, *fmt):
    if not con:
        con = ffi.NULL
    return lib.TCOD_console_print_ex(con, x, y, flag, alignment, *_encode_fmt(fmt))

def console_print_rect(con, x, y, w, h, *fmt):
    if not con:
        con = ffi.NULL
    return lib.TCOD_console_print_rect
    
def console_print_rect_ex(con, x, y, w, h, flag, alignment, *fmt):
    if not con:
        con = ffi.NULL
    return lib.TCOD_console_print_rect_ex(con, x, y, w, h,
                                          flag, alignment, *_encode_fmt(fmt))

                                          
def console_blit(src, x, y, w, h, dst, xdst, ydst, ffade=1.0,bfade=1.0):
    if not src:
        src = ffi.NULL
    if not dst:
        dst = ffi.NULL
    
    lib.TCOD_console_blit(src, x, y, w, h, dst, xdst, ydst, ffade, bfade)

    
# parser
                                          
def parser_run(parser, filename, listener=0):
    if listener != 0:
        clistener=_CParserListener()
        def value_converter(name, typ, value):
            if typ == TYPE_BOOL:
                return listener.new_property(name, typ, value.c == 1)
            elif typ == TYPE_CHAR:
                return listener.new_property(name, typ, '%c' % (value.c & 0xFF))
            elif typ == TYPE_INT:
                return listener.new_property(name, typ, value.i)
            elif typ == TYPE_FLOAT:
                return listener.new_property(name, typ, value.f)
            elif typ == TYPE_STRING or \
                 TYPE_VALUELIST15 >= typ >= TYPE_VALUELIST00:
                 return listener.new_property(name, typ, value.s)
            elif typ == TYPE_COLOR:
                col = cast(value.col, POINTER(Color)).contents
                return listener.new_property(name, typ, col)
            elif typ == TYPE_DICE:
                dice = cast(value.dice, POINTER(Dice)).contents
                return listener.new_property(name, typ, dice)
            elif typ & TYPE_LIST:
                return listener.new_property(name, typ,
                                        _convert_TCODList(value.custom, typ & 0xFF))
            return True
        clistener.new_struct = _CFUNC_NEW_STRUCT(listener.new_struct)
        clistener.new_flag = _CFUNC_NEW_FLAG(listener.new_flag)
        clistener.new_property = _CFUNC_NEW_PROPERTY(value_converter)
        clistener.end_struct = _CFUNC_NEW_STRUCT(listener.end_struct)
        clistener.error = _CFUNC_NEW_FLAG(listener.error)
        lib.TCOD_parser_run(parser, filename, byref(clistener))
    else:
        lib.TCOD_parser_run(parser, filename, ffi.NULL)

def Dice(*args):
    return ffi.new('TCOD_dice_t *', args)

# parser
    
def _CParserListener(*args):
    return ffi.new('TCOD_parser_listener_t *', args)
        
def _CValue(*args):
    return ffi.new('TCOD_value_t *', args)
    
def _CFUNC_NEW_STRUCT(func):
    return ffi.callback('bool(TCOD_parser_struct_t, char*)')(func)

def _CFUNC_NEW_FLAG(func):
    return ffi.callback('bool(char*)')(func)
    
def _CFUNC_NEW_PROPERTY(func):
    return ffi.callback('bool(char*, TCOD_value_type_t, TCOD_value_t)')(func)
 
# random

def random_set_distribution(rnd, dist):
    if rnd is None:
        rnd = ffi.NULL
    lib.TCOD_random_set_distribution(rnd, dist)

def random_get_int(rnd, mi, ma):
    if rnd is None:
        rnd = ffi.NULL
    return lib.TCOD_random_get_int(rnd, mi, ma)

def random_get_float(rnd, mi, ma):
    if rnd is None:
        rnd = ffi.NULL
    return lib.TCOD_random_get_float(rnd, mi, ma)

def random_get_double(rnd, mi, ma):
    if rnd is None:
        rnd = ffi.NULL
    return lib.TCOD_random_get_double(rnd, mi, ma)

def random_get_int_mean(rnd, mi, ma, mean):
    if rnd is None:
        rnd = ffi.NULL
    return lib.TCOD_random_get_int_mean(rnd, mi, ma, mean)

def random_get_float_mean(rnd, mi, ma, mean):
    if rnd is None:
        rnd = ffi.NULL
    return lib.TCOD_random_get_float_mean(rnd, mi, ma, mean)

# image

def image_rotate90(image, num=1) :
    lib.TCOD_image_rotate90(image, num)
    
def image_load(filename):
    return lib.TCOD_image_load(filename)

def image_from_console(console):
    if not console:
        console = ffi.NULL
    return lib.TCOD_image_from_console(console)

def image_refresh_console(image, console):
    if not console:
        console = ffi.NULL
    lib.TCOD_image_refresh_console(image, console)

def image_get_size(image):
    w=ffi.new('int *')
    h=ffi.new('int *')
    lib.TCOD_image_get_size(image, w, h)
    return w[0], h[0]

def image_blit_rect(image, console, x, y, w, h, bkgnd_flag):
    if not console:
        console = ffi.NULL
    lib.TCOD_image_blit_rect(image, console, x, y, w, h, bkgnd_flag)

def image_blit_2x(image, console, dx, dy, sx=0, sy=0, w=-1, h=-1):
    if not console:
        console = ffi.NULL
    lib.TCOD_image_blit_2x(image, console, dx,dy,sx,sy,w,h)

def image_save(image, filename):
    _lib.TCOD_image_save(image, filename)


# noise

NOISE_DEFAULT_HURST = 0.5
NOISE_DEFAULT_LACUNARITY = 2.0

def noise_new(dim, h=NOISE_DEFAULT_HURST, l=NOISE_DEFAULT_LACUNARITY, random=ffi.NULL):
    return lib.TCOD_noise_new(dim, h, l, random)

def noise_get(n, f, typ=NOISE_DEFAULT):
    
    return lib.TCOD_noise_get_ex(n, ffi.new('float[]', f), typ)

def noise_get_fbm(n, f, oc, typ=NOISE_DEFAULT):
    return lib.TCOD_noise_get_fbm_ex(n, ffi.new('float[]', f), oc, typ)

def noise_get_turbulence(n, f, oc, typ=NOISE_DEFAULT):
    return lib.TCOD_noise_get_turbulence_ex(n, ffi.new('float[]', f), oc, typ)

# line

def line_init(xo, yo, xd, yd):
    lib.TCOD_line_init(xo, yo, xd, yd)

def line_step():
    x = ffi.new('int *')
    y = ffi.new('int *')
    if not lib.TCOD_line_step(byref(x), byref(y)):
        return x[0], y[0]
    return None,None

def line(xo,yo,xd,yd,py_callback) :
    #LINE_CBK_FUNC=CFUNCTYPE(c_bool,c_int,c_int)
    #c_callback=LINE_CBK_FUNC(py_callback)
    c_callback = ffi.callback('TCOD_line_listener_t')(py_callback)
    return lib.TCOD_line(xo,yo,xd,yd,c_callback)

def line_iter(xo, yo, xd, yd):
    data = ffi.new('TCOD_bresenham_data_t *')
    lib.TCOD_line_init_mt(xo, yo, xd, yd, data)
    x = ffi.new('int *')
    y = ffi.new('int *')
    done = False
    while not done:
        yield x[0], y[0]
        done = lib.TCOD_line_step_mt(x, y, data)
              
__all__ = [name for name in list(globals()) if name[0] != '_']
