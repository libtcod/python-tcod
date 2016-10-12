
from . import Color
from .libtcod import _lib, _ffi


def color_lerp(c1, c2, a):
    return Color.from_cdata(_lib.TCOD_color_lerp(c1, c2, a))

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
    ccolors = _ffi.new('TCOD_color_t[]', colors)
    cindexes = _ffi.new('int[]', indexes)
    cres = _ffi.new('TCOD_color_t[]', max(indexes) + 1)
    _lib.TCOD_color_gen_map(cres, len(colors), ccolors, cindexes)
    return [Color.from_cdata(cdata) for cdata in cres]


__all__ = [_name for _name in list(globals()) if _name[0] != '_']
