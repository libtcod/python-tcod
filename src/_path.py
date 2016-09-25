
from .libtcod import _lib, _ffi

@_ffi.def_extern()
def _pycall_path_func(x1, y1, x2, y2, func_handle):
    '''static float _pycall_path_func( int xFrom, int yFrom, int xTo, int yTo, void *user_data );
    '''
    return _ffi.from_handle(func_handle)(x1, y1, x2, y2)

def path_new_using_map(m, dcost=1.41):
    return (_lib.TCOD_path_new_using_map(m, dcost), None)

def path_new_using_function(w, h, func, dcost=1.41):
    python_handle = _ffi.new_handle(func)
    return (_lib.TCOD_path_new_using_function(w, h, _lib._pycall_path_func,
            python_handle, dcost), python_handle)

def path_compute(p, ox, oy, dx, dy):
    return _lib.TCOD_path_compute(p[0], ox, oy, dx, dy)

def path_get_origin(p):
    x = _ffi.new('int *')
    y = _ffi.new('int *')
    _lib.TCOD_path_get_origin(p[0], x, y)
    return x[0], y[0]

def path_get_destination(p):
    x = _ffi.new('int *')
    y = _ffi.new('int *')
    _lib.TCOD_path_get_destination(p[0], x, y)
    return x[0], y[0]

def path_size(p):
    return _lib.TCOD_path_size(p[0])

def path_reverse(p):
    _lib.TCOD_path_reverse(p[0])

def path_get(p, idx):
    x = _ffi.new('int *')
    y = _ffi.new('int *')
    _lib.TCOD_path_get(p[0], idx, x, y)
    return x[0], y[0]

def path_is_empty(p):
    return _lib.TCOD_path_is_empty(p[0])

def path_walk(p, recompute):
    x = _ffi.new('int *')
    y = _ffi.new('int *')
    if _lib.TCOD_path_walk(p[0], x, y, recompute):
        return x[0], y[0]
    return None,None

def path_delete(p):
    _lib.TCOD_path_delete(p[0])

__all__ = [_name for _name in list(globals()) if _name[0] != '_']
