
import tcod as _tcod
from .libtcod import _lib, _ffi

def noise_new(dim, h=_tcod.NOISE_DEFAULT_HURST, l=_tcod.NOISE_DEFAULT_LACUNARITY,
        random=None):
    return _lib.TCOD_noise_new(dim, h, l, random or _ffi.NULL)

def noise_set_type(n, typ) :
    _lib.TCOD_noise_set_type(n,typ)

def noise_get(n, f, typ=_tcod.NOISE_DEFAULT):
    return _lib.TCOD_noise_get_ex(n, _ffi.new('float[]', f), typ)

def noise_get_fbm(n, f, oc, typ=_tcod.NOISE_DEFAULT):
    return _lib.TCOD_noise_get_fbm_ex(n, _ffi.new('float[]', f), oc, typ)

def noise_get_turbulence(n, f, oc, typ=_tcod.NOISE_DEFAULT):
    return _lib.TCOD_noise_get_turbulence_ex(n, _ffi.new('float[]', f), oc, typ)

def noise_delete(n):
    _lib.TCOD_noise_delete(n)

__all__ = [_name for _name in list(globals()) if _name[0] != '_']
