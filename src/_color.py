
from . import Color as _Color
from .libtcod import _lib, _ffi


def color_lerp(c1, c2, a):
    return _Color.from_cdata(_lib.TCOD_color_lerp(c1, c2, a))

def color_set_hsv(c, h, s, v):
    tcod_color = _ffi.new('TCOD_color_t *', c)
    _lib.TCOD_color_set_HSV(tcod_color, h, s, v)
    c[0:3] = tcod_color.r, tcod_color.g, tcod_color.b

def color_get_hsv(c):
    h = _ffi.new('float *')
    s = _ffi.new('float *')
    v = _ffi.new('float *')
    _lib.TCOD_color_get_HSV(c, h, s, v)
    return h[0], s[0], v[0]

def color_scale_HSV(c, scoef, vcoef) :
    tcod_color = _ffi.new('TCOD_color_t *', c)
    _lib.TCOD_color_scale_HSV(tcod_color, scoef, vcoef)
    c[0:3] = tcod_color.r, tcod_color.g, tcod_color.b

def color_gen_map(colors, indexes):
    ccolors = (_Color * len(colors))(*colors)
    cindexes = (c_int * len(indexes))(*indexes)
    cres = (_Color * (max(indexes) + 1))()
    _lib.TCOD_color_gen_map(cres, len(colors), ccolors, cindexes)
    return cres


__all__ = [_name for _name in list(globals()) if _name[0] != '_']
