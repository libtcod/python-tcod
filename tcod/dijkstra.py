
from .libtcod import _lib, _ffi
from .path import PATH_CBK_FUNC

def new(m, dcost=1.41):
    return (_lib.TCOD_dijkstra_new(m, dcost), None)

def new_using_function(w, h, func, userdata=0, dcost=1.41):
    cbk_func = PATH_CBK_FUNC(func)
    return (_lib.TCOD_path_dijkstra_using_function(w, h, cbk_func,
            py_object(userdata), c_float(dcost)), cbk_func)

def compute(p, ox, oy):
    _lib.TCOD_dijkstra_compute(p[0], ox, oy)

def path_set(p, x, y):
    return _lib.TCOD_dijkstra_path_set(p[0], x, y)

def get_distance(p, x, y):
    return _lib.TCOD_dijkstra_get_distance(p[0], x, y)

def size(p):
    return _lib.TCOD_dijkstra_size(p[0])

def reverse(p):
    _lib.TCOD_dijkstra_reverse(p[0])

def get(p, idx):
    x = _ffi.new('int *')
    y = _ffi.new('int *')
    _lib.TCOD_dijkstra_get(p[0], idx, x, y)
    return x[0], y[0]

def is_empty(p):
    return _lib.TCOD_dijkstra_is_empty(p[0])

def path_walk(p):
    x = _ffi.new('int *')
    y = _ffi.new('int *')
    if _lib.TCOD_dijkstra_path_walk(p[0], x, y):
        return x[0], y[0]
    return None,None

def delete(p):
    _lib.TCOD_dijkstra_delete(p[0])

__all__ = [name for name in list(globals()) if name[0] != '_']
