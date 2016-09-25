
import tcod as _tcod
from .libtcod import _lib, _ffi

def heightmap_new(w, h):
    phm = _lib.TCOD_heightmap_new(w, h)
    return _tcod.HeightMap(phm)

def heightmap_set_value(hm, x, y, value):
    _lib.TCOD_heightmap_set_value(hm.p, x, y, value)

def heightmap_add(hm, value):
    _lib.TCOD_heightmap_add(hm.p, value)

def heightmap_scale(hm, value):
    _lib.TCOD_heightmap_scale(hm.p, value)

def heightmap_clear(hm):
    _lib.TCOD_heightmap_clear(hm.p)

def heightmap_clamp(hm, mi, ma):
    _lib.TCOD_heightmap_clamp(hm.p, mi, ma)

def heightmap_copy(hm1, hm2):
    _lib.TCOD_heightmap_copy(hm1.p, hm2.p)

def heightmap_normalize(hm,  mi=0.0, ma=1.0):
    _lib.TCOD_heightmap_normalize(hm.p, mi, ma)

def heightmap_lerp_hm(hm1, hm2, hm3, coef):
    _lib.TCOD_heightmap_lerp_hm(hm1.p, hm2.p, hm3.p, coef)

def heightmap_add_hm(hm1, hm2, hm3):
    _lib.TCOD_heightmap_add_hm(hm1.p, hm2.p, hm3.p)

def heightmap_multiply_hm(hm1, hm2, hm3):
    _lib.TCOD_heightmap_multiply_hm(hm1.p, hm2.p, hm3.p)

def heightmap_add_hill(hm, x, y, radius, height):
    _lib.TCOD_heightmap_add_hill(hm.p, x, y, radius, height)

def heightmap_dig_hill(hm, x, y, radius, height):
    _lib.TCOD_heightmap_dig_hill(hm.p, x, y, radius, height)

def heightmap_rain_erosion(hm, nbDrops, erosionCoef, sedimentationCoef, rnd=None):
    _lib.TCOD_heightmap_rain_erosion(hm.p, nbDrops, erosionCoef,
                                     sedimentationCoef, rnd or _ffi.NULL)

def heightmap_kernel_transform(hm, kernelsize, dx, dy, weight, minLevel,
                               maxLevel):
    #FARRAY = c_float * kernelsize
    #IARRAY = c_int * kernelsize
    cdx = _ffi.new('float[]', dx)
    cdy = _ffi.new('int[]', dy)
    cweight = _ffi.new('float[]', weight)
    _lib.TCOD_heightmap_kernel_transform(hm.p, kernelsize, cdx, cdy, cweight,
                                         minLevel, maxLevel)

def heightmap_add_voronoi(hm, nbPoints, nbCoef, coef, rnd=None):
    #FARRAY = c_float * nbCoef
    ccoef = _ffi.new('float[]', coef)
    _lib.TCOD_heightmap_add_voronoi(hm.p, nbPoints, nbCoef, ccoef, rnd or _ffi.NULL)

def heightmap_add_fbm(hm, noise, mulx, muly, addx, addy, octaves, delta, scale):
    _lib.TCOD_heightmap_add_fbm(hm.p, noise, mulx, muly, addx, addy,
                                octaves, delta, scale)
def heightmap_scale_fbm(hm, noise, mulx, muly, addx, addy, octaves, delta,
                        scale):
    _lib.TCOD_heightmap_scale_fbm(hm.p, noise, mulx, muly, addx, addy,
                                  octaves, delta, scale)

def heightmap_dig_bezier(hm, px, py, startRadius, startDepth, endRadius,
                         endDepth):
    #IARRAY = c_int * 4
    cpx = _ffi.new('int[4]', px)
    cpy = _ffi.new('int[4]', py)
    _lib.TCOD_heightmap_dig_bezier(hm.p, cpx, cpy, startRadius,
                                   startDepth, endRadius,
                                   endDepth)

def heightmap_get_value(hm, x, y):
    return _lib.TCOD_heightmap_get_value(hm.p, x, y)

def heightmap_get_interpolated_value(hm, x, y):
    return _lib.TCOD_heightmap_get_interpolated_value(hm.p, x, y)

def heightmap_get_slope(hm, x, y):
    return _lib.TCOD_heightmap_get_slope(hm.p, x, y)

def heightmap_get_normal(hm, x, y, waterLevel):
    #FARRAY = c_float * 3
    cn = _ffi.new('float[3]')
    _lib.TCOD_heightmap_get_normal(hm.p, x, y, cn, waterLevel)
    return tuple(cn)

def heightmap_count_cells(hm, mi, ma):
    return _lib.TCOD_heightmap_count_cells(hm.p, mi, ma)

def heightmap_has_land_on_border(hm, waterlevel):
    return _lib.TCOD_heightmap_has_land_on_border(hm.p, waterlevel)

def heightmap_get_minmax(hm):
    mi = _ffi.new('float *')
    ma = _ffi.new('float *')
    _lib.TCOD_heightmap_get_minmax(hm.p, mi, ma)
    return mi[0], ma[0]

def heightmap_delete(hm):
    _lib.TCOD_heightmap_delete(hm.p)

__all__ = [_name for _name in list(globals()) if _name[0] != '_']
