
from .libtcod import _lib, _ffi

def init(xo, yo, xd, yd):
    _lib.TCOD_line_init(xo, yo, xd, yd)

def step():
    x = _ffi.new('int *')
    y = _ffi.new('int *')
    ret = _lib.TCOD_line_step(x, y)
    if not ret:
        return x[0], y[0]
    return None,None

def line(xo, yo, xd, yd, py_callback) :
    'callback: bool(int, int)'
    c_callback = _ffi.callback('TCOD_line_listener_t')(py_callback)
    return _lib.TCOD_line(xo, yo, xd, yd, c_callback)

def iter(xo, yo, xd, yd):
    data = _ffi.new('TCOD_bresenham_data_t *')
    _lib.TCOD_line_init_mt(xo, yo, xd, yd, data)
    x = _ffi.new('int *')
    y = _ffi.new('int *')
    done = False
    while not _lib.TCOD_line_step_mt(x, y, data):
        yield (x[0], y[0])

__all__ = [_name for _name in list(globals()) if _name[0] != '_']
