
from .libtcod import _lib, _ffi

#PATH_CBK_FUNC = CFUNCTYPE(c_float, c_int, c_int, c_int, c_int, py_object)
def PATH_CBK_FUNC(func):
    def path_cb(x1, y1, x2, y2, pyobj):
        return func(x1, y1, x2, y2, _ffi.from_handle(pyobj))
    return _ffi.calback('float(int,int,int,int,void*)')(path_cb)

def new_using_map(m, dcost=1.41):
    return (_lib.TCOD_path_new_using_map(m, dcost), None)

def new_using_function(w, h, func, userdata=None, dcost=1.41):
    cbk_func = PATH_CBK_FUNC(func)
    userdata = _ffi.new_handle(userdata)
    return (_lib.TCOD_path_new_using_function(w, h, cbk_func,
            userdata, dcost), cbk_func)

def compute(p, ox, oy, dx, dy):
    return _lib.TCOD_path_compute(p[0], ox, oy, dx, dy)

def get_origin(p):
    x = _ffi.new('int *')
    y = _ffi.new('int *')
    _lib.TCOD_path_get_origin(p[0], x, y)
    return x[0], y[0]

def get_destination(p):
    x = _ffi.new('int *')
    y = _ffi.new('int *')
    _lib.TCOD_path_get_destination(p[0], x, y)
    return x[0], y[0]

def size(p):
    return _lib.TCOD_path_size(p[0])

def reverse(p):
    _lib.TCOD_path_reverse(p[0])  

def get(p, idx):
    x = _ffi.new('int *')
    y = _ffi.new('int *')
    _lib.TCOD_path_get(p[0], idx, x, y)
    return x[0], y[0]

def is_empty(p):
    return _lib.TCOD_path_is_empty(p[0])

def walk(p, recompute):
    x = _ffi.new('int *')
    y = _ffi.new('int *')
    if _lib.TCOD_path_walk(p[0], x, y, recompute):
        return x[0], y[0]
    return None,None

def delete(p):
    _lib.TCOD_path_delete(p[0])

__all__ = [_name for _name in list(globals()) if name[0] != '_']
