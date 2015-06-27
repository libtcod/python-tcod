
from .libtcod import _lib, _ffi

FOV_BASIC = 0
FOV_DIAMOND = 1
FOV_SHADOW = 2
FOV_PERMISSIVE_0 = 3
FOV_PERMISSIVE_1 = 4
FOV_PERMISSIVE_2 = 5
FOV_PERMISSIVE_3 = 6
FOV_PERMISSIVE_4 = 7
FOV_PERMISSIVE_5 = 8
FOV_PERMISSIVE_6 = 9
FOV_PERMISSIVE_7 = 10
FOV_PERMISSIVE_8 = 11
FOV_RESTRICTIVE = 12
NB_FOV_ALGORITHMS = 13

def new(w, h):
    return _lib.TCOD_map_new(w, h)

def copy(source, dest):
    return _lib.TCOD_map_copy(source, dest)

def set_properties(m, x, y, isTrans, isWalk):
    _lib.TCOD_map_set_properties(m, x, y, isTrans, isWalk)

def clear(m,walkable=False,transparent=False):
    _lib.TCOD_map_clear(m, walkable, transparent)

def compute_fov(m, x, y, radius=0, light_walls=True, algo=FOV_RESTRICTIVE ):
    _lib.TCOD_map_compute_fov(m, x, y, radius, light_walls, algo)

def is_in_fov(m, x, y):
    return _lib.TCOD_map_is_in_fov(m, x, y)

def is_transparent(m, x, y):
    return _lib.TCOD_map_is_transparent(m, x, y)

def is_walkable(m, x, y):
    return _lib.TCOD_map_is_walkable(m, x, y)

def delete(m):
    return _lib.TCOD_map_delete(m)

def get_width(map):
    return _lib.TCOD_map_get_width(map)

def get_height(map):
    return _lib.TCOD_map_get_height(map)


__all__ = [name for name in list(globals()) if name[0] != '_']