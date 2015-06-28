
from .libtcod import _lib, _ffi

RNG_MT = 0
RNG_CMWC = 1

DISTRIBUTION_LINEAR = 0
DISTRIBUTION_GAUSSIAN = 1
DISTRIBUTION_GAUSSIAN_RANGE = 2
DISTRIBUTION_GAUSSIAN_INVERSE = 3
DISTRIBUTION_GAUSSIAN_RANGE_INVERSE = 4

def get_instance():
    return _lib.TCOD_random_get_instance()

def new(algo=RNG_CMWC):
    return _lib.TCOD_random_new(algo)

def new_from_seed(seed, algo=RNG_CMWC):
    return _lib.TCOD_random_new_from_seed(algo, seed)

def set_distribution(rnd, dist) :
	_lib.TCOD_random_set_distribution(rnd or _ffi.NULL, dist)

def get_int(rnd, mi, ma):
    return _lib.TCOD_random_get_int(rnd or _ffi.NULL, mi, ma)

def get_float(rnd, mi, ma):
    return _lib.TCOD_random_get_float(rnd or _ffi.NULL, mi, ma)

def get_double(rnd, mi, ma):
    return _lib.TCOD_random_get_double(rnd or _ffi.NULL, mi, ma)

def get_int_mean(rnd, mi, ma, mean):
    return _lib.TCOD_random_get_int_mean(rnd or _ffi.NULL, mi, ma, mean)

def get_float_mean(rnd, mi, ma, mean):
    return _lib.TCOD_random_get_float_mean(rnd or _ffi.NULL, mi, ma, mean)

def get_double_mean(rnd, mi, ma, mean):
    return _lib.TCOD_random_get_double_mean(rnd or _ffi.NULL, mi, ma, mean)

def save(rnd):
    return _lib.TCOD_random_save(rnd or _ffi.NULL)

def restore(rnd, backup):
    _lib.TCOD_random_restore(rnd or _ffi.NULL, backup)

def delete(rnd):
    _lib.TCOD_random_delete(rnd)

__all__ = [_name for _name in list(globals()) if name[0] != '_']
