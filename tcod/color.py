
from . import Color as _Color
from .libtcod import _lib, _ffi


def lerp(c1, c2, a):
    return _Color.from_tcod(_lib.TCOD_color_lerp(c1, c2, a))

def set_hsv(c, h, s, v):
    _lib.TCOD_color_set_HSV(c._struct, h, s, v)

def get_hsv(c):
    h = _ffi.new('float *')
    s = _ffi.new('float *')
    v = _ffi.new('float *')
    _lib.TCOD_color_get_HSV(c, h, s, v)
    return h[0], s[0], v[0]

def scale_HSV(c, scoef, vcoef) :
    _lib.TCOD_color_scale_HSV(c._struct, scoef, vcoef)

def gen_map(colors, indexes):
    ccolors = (_Color * len(colors))(*colors)
    cindexes = (c_int * len(indexes))(*indexes)
    cres = (_Color * (max(indexes) + 1))()
    _lib.TCOD_color_gen_map(cres, len(colors), ccolors, cindexes)
    return cres


__all__ = [name for name in list(globals()) if name[0] != '_']