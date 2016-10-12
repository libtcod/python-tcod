
import tcod as _tcod
from .libtcod import _lib, _ffi

class HeightMap(object):
    def __new__(cls, width, height):
        self = object.__new__(cls)
        self.cdata = _ffi.gc(_lib.TCOD_heightmap_new(width, height),
                             _lib.TCOD_heightmap_delete)
        return self

    def getw(self):
        return self.cdata.w
    def setw(self, value):
        self.cdata.w = value
    w = property(getw, setw)

    def geth(self):
        return self.cdata.h
    def seth(self, value):
        self.cdata.h = value
    h = property(geth, seth)


def heightmap_new(w, h):
    return HeightMap(w, h)

def heightmap_set_value(hm, x, y, value):
    _lib.TCOD_heightmap_set_value(hm.cdata, x, y, value)

def heightmap_add(hm, value):
    _lib.TCOD_heightmap_add(hm.cdata, value)

def heightmap_scale(hm, value):
    _lib.TCOD_heightmap_scale(hm.cdata, value)

def heightmap_clear(hm):
    _lib.TCOD_heightmap_clear(hm.cdata)

def heightmap_clamp(hm, mi, ma):
    _lib.TCOD_heightmap_clamp(hm.cdata, mi, ma)

def heightmap_copy(hm1, hm2):
    _lib.TCOD_heightmap_copy(hm1.cdata, hm2.cdata)

def heightmap_normalize(hm,  mi=0.0, ma=1.0):
    _lib.TCOD_heightmap_normalize(hm.cdata, mi, ma)

def heightmap_lerp_hm(hm1, hm2, hm3, coef):
    _lib.TCOD_heightmap_lerp_hm(hm1.cdata, hm2.cdata, hm3.cdata, coef)

def heightmap_add_hm(hm1, hm2, hm3):
    _lib.TCOD_heightmap_add_hm(hm1.cdata, hm2.cdata, hm3.cdata)

def heightmap_multiply_hm(hm1, hm2, hm3):
    _lib.TCOD_heightmap_multiply_hm(hm1.cdata, hm2.cdata, hm3.cdata)

def heightmap_add_hill(hm, x, y, radius, height):
    _lib.TCOD_heightmap_add_hill(hm.cdata, x, y, radius, height)

def heightmap_dig_hill(hm, x, y, radius, height):
    _lib.TCOD_heightmap_dig_hill(hm.cdata, x, y, radius, height)

def heightmap_rain_erosion(hm, nbDrops, erosionCoef, sedimentationCoef, rnd=None):
    _lib.TCOD_heightmap_rain_erosion(hm.cdata, nbDrops, erosionCoef,
                                     sedimentationCoef, rnd or _ffi.NULL)

def heightmap_kernel_transform(hm, kernelsize, dx, dy, weight, minLevel,
                               maxLevel):
    cdx = _ffi.new('int[]', dx)
    cdy = _ffi.new('int[]', dy)
    cweight = _ffi.new('float[]', weight)
    _lib.TCOD_heightmap_kernel_transform(hm.cdata, kernelsize, cdx, cdy, cweight,
                                         minLevel, maxLevel)

def heightmap_add_voronoi(hm, nbPoints, nbCoef, coef, rnd=None):
    ccoef = _ffi.new('float[]', coef)
    _lib.TCOD_heightmap_add_voronoi(hm.cdata, nbPoints, nbCoef, ccoef, rnd or _ffi.NULL)

def heightmap_add_fbm(hm, noise, mulx, muly, addx, addy, octaves, delta, scale):
    _lib.TCOD_heightmap_add_fbm(hm.cdata, noise, mulx, muly, addx, addy,
                                octaves, delta, scale)
def heightmap_scale_fbm(hm, noise, mulx, muly, addx, addy, octaves, delta,
                        scale):
    _lib.TCOD_heightmap_scale_fbm(hm.cdata, noise, mulx, muly, addx, addy,
                                  octaves, delta, scale)

def heightmap_dig_bezier(hm, px, py, startRadius, startDepth, endRadius,
                         endDepth):
    #IARRAY = c_int * 4
    cpx = _ffi.new('int[4]', px)
    cpy = _ffi.new('int[4]', py)
    _lib.TCOD_heightmap_dig_bezier(hm.cdata, cpx, cpy, startRadius,
                                   startDepth, endRadius,
                                   endDepth)

def heightmap_get_value(hm, x, y):
    return _lib.TCOD_heightmap_get_value(hm.cdata, x, y)

def heightmap_get_interpolated_value(hm, x, y):
    return _lib.TCOD_heightmap_get_interpolated_value(hm.cdata, x, y)

def heightmap_get_slope(hm, x, y):
    return _lib.TCOD_heightmap_get_slope(hm.cdata, x, y)

def heightmap_get_normal(hm, x, y, waterLevel):
    #FARRAY = c_float * 3
    cn = _ffi.new('float[3]')
    _lib.TCOD_heightmap_get_normal(hm.cdata, x, y, cn, waterLevel)
    return tuple(cn)

def heightmap_count_cells(hm, mi, ma):
    return _lib.TCOD_heightmap_count_cells(hm.cdata, mi, ma)

def heightmap_has_land_on_border(hm, waterlevel):
    return _lib.TCOD_heightmap_has_land_on_border(hm.cdata, waterlevel)

def heightmap_get_minmax(hm):
    mi = _ffi.new('float *')
    ma = _ffi.new('float *')
    _lib.TCOD_heightmap_get_minmax(hm.cdata, mi, ma)
    return mi[0], ma[0]

def heightmap_delete(hm):
    pass

__all__ = [_name for _name in list(globals()) if _name[0] != '_']
