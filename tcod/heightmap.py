
from .libtcod import _lib, _ffi

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

def heightmap_new(w, h):
    phm = _lib.TCOD_heightmap_new(w, h)
    return HeightMap(phm)

def heightmap_set_value(hm, x, y, value):
    _lib.TCOD_heightmap_set_value(hm.p, x, y, c_float(value))

def heightmap_add(hm, value):
    _lib.TCOD_heightmap_add(hm.p, c_float(value))

def heightmap_scale(hm, value):
    _lib.TCOD_heightmap_scale(hm.p, c_float(value))

def heightmap_clear(hm):
    _lib.TCOD_heightmap_clear(hm.p)

def heightmap_clamp(hm, mi, ma):
    _lib.TCOD_heightmap_clamp(hm.p, c_float(mi),c_float(ma))

def heightmap_copy(hm1, hm2):
    _lib.TCOD_heightmap_copy(hm1.p, hm2.p)

def heightmap_normalize(hm,  mi=0.0, ma=1.0):
    _lib.TCOD_heightmap_normalize(hm.p, c_float(mi), c_float(ma))

def heightmap_lerp_hm(hm1, hm2, hm3, coef):
    _lib.TCOD_heightmap_lerp_hm(hm1.p, hm2.p, hm3.p, c_float(coef))

def heightmap_add_hm(hm1, hm2, hm3):
    _lib.TCOD_heightmap_add_hm(hm1.p, hm2.p, hm3.p)

def heightmap_multiply_hm(hm1, hm2, hm3):
    _lib.TCOD_heightmap_multiply_hm(hm1.p, hm2.p, hm3.p)

def heightmap_add_hill(hm, x, y, radius, height):
    _lib.TCOD_heightmap_add_hill(hm.p, c_float( x), c_float( y),
                                 c_float( radius), c_float( height))

def heightmap_dig_hill(hm, x, y, radius, height):
    _lib.TCOD_heightmap_dig_hill(hm.p, c_float( x), c_float( y),
                                 c_float( radius), c_float( height))

def heightmap_rain_erosion(hm, nbDrops, erosionCoef, sedimentationCoef, rnd=0):
    _lib.TCOD_heightmap_rain_erosion(hm.p, nbDrops, c_float( erosionCoef),
                                     c_float( sedimentationCoef), rnd)

def heightmap_kernel_transform(hm, kernelsize, dx, dy, weight, minLevel,
                               maxLevel):
    FARRAY = c_float * kernelsize
    IARRAY = c_int * kernelsize
    cdx = IARRAY(*dx)
    cdy = IARRAY(*dy)
    cweight = FARRAY(*weight)
    _lib.TCOD_heightmap_kernel_transform(hm.p, kernelsize, cdx, cdy, cweight,
                                         c_float(minLevel), c_float(maxLevel))

def heightmap_add_voronoi(hm, nbPoints, nbCoef, coef, rnd=0):
    FARRAY = c_float * nbCoef
    ccoef = FARRAY(*coef)
    _lib.TCOD_heightmap_add_voronoi(hm.p, nbPoints, nbCoef, ccoef, rnd)

def heightmap_add_fbm(hm, noise, mulx, muly, addx, addy, octaves, delta, scale):
    _lib.TCOD_heightmap_add_fbm(hm.p, noise, c_float(mulx), c_float(muly),
                                c_float(addx), c_float(addy),
                                c_float(octaves), c_float(delta),
                                c_float(scale))
def heightmap_scale_fbm(hm, noise, mulx, muly, addx, addy, octaves, delta,
                        scale):
    _lib.TCOD_heightmap_scale_fbm(hm.p, noise, c_float(mulx), c_float(muly),
                                  c_float(addx), c_float(addy),
                                  c_float(octaves), c_float(delta),
                                  c_float(scale))
def heightmap_dig_bezier(hm, px, py, startRadius, startDepth, endRadius,
                         endDepth):
    IARRAY = c_int * 4
    cpx = IARRAY(*px)
    cpy = IARRAY(*py)
    _lib.TCOD_heightmap_dig_bezier(hm.p, cpx, cpy, c_float(startRadius),
                                   c_float(startDepth), c_float(endRadius),
                                   c_float(endDepth))

def heightmap_get_value(hm, x, y):
    return _lib.TCOD_heightmap_get_value(hm.p, x, y)

def heightmap_get_interpolated_value(hm, x, y):
    return _lib.TCOD_heightmap_get_interpolated_value(hm.p, c_float(x),
                                                     c_float(y))

def heightmap_get_slope(hm, x, y):
    return _lib.TCOD_heightmap_get_slope(hm.p, x, y)

def heightmap_get_normal(hm, x, y, waterLevel):
    FARRAY = c_float * 3
    cn = FARRAY()
    _lib.TCOD_heightmap_get_normal(hm.p, c_float(x), c_float(y), cn,
                                   c_float(waterLevel))
    return cn[0], cn[1], cn[2]

def heightmap_count_cells(hm, mi, ma):
    return _lib.TCOD_heightmap_count_cells(hm.p, c_float(mi), c_float(ma))

def heightmap_has_land_on_border(hm, waterlevel):
    return _lib.TCOD_heightmap_has_land_on_border(hm.p, c_float(waterlevel))

def heightmap_get_minmax(hm):
    mi = c_float()
    ma = c_float()
    _lib.TCOD_heightmap_get_minmax(hm.p, byref(mi), byref(ma))
    return mi.value, ma.value

def heightmap_delete(hm):
    _lib.TCOD_heightmap_delete(hm.p)


__all__ = [name for name in list(globals()) if name[0] != '_']