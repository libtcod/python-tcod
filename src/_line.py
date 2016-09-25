
import threading as _threading

from .libtcod import _lib, _ffi

def line_init(xo, yo, xd, yd):
    _lib.TCOD_line_init(xo, yo, xd, yd)

def line_step():
    x = _ffi.new('int *')
    y = _ffi.new('int *')
    ret = _lib.TCOD_line_step(x, y)
    if not ret:
        return x[0], y[0]
    return None,None

_line_listener_lock = _threading.Lock()

def line_line(xo, yo, xd, yd, py_callback) :
    'callback: bool(int, int)'
    with _line_listener_lock:
        @_ffi.def_extern()
        def _pycall_line_listener(x, y):
            return py_callback(x, y)
        return _lib.TCOD_line(xo, yo, xd, yd, _lib._pycall_line_listener)

def line_iter(xo, yo, xd, yd):
    data = _ffi.new('TCOD_bresenham_data_t *')
    _lib.TCOD_line_init_mt(xo, yo, xd, yd, data)
    x = _ffi.new('int *')
    y = _ffi.new('int *')
    done = False
    while not _lib.TCOD_line_step_mt(x, y, data):
        yield (x[0], y[0])

__all__ = [_name for _name in list(globals()) if _name[0] != '_']
