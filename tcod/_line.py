
import threading as _threading

from .libtcod import _lib, _ffi, _PropagateException

def line_init(xo, yo, xd, yd):
    """Initilize a line whose points will be returned by `line_step`.

    This function does not return anything on its own.

    Does not include the origin point.

    :param int xo: x origin
    :param int yo: y origin
    :param int xd: x destination
    :param int yd: y destination

    .. deprecated:: 2.0
       Use `line_iter` instead.
    """
    _lib.TCOD_line_init(xo, yo, xd, yd)

def line_step():
    """After calling `line_init` returns (x, y) points of the line.

    Once all points are exhausted this function will return (None, None)

    :return: next (x, y) point of the line setup by `line_init`,
             or (None, None) if there are no more points.
    :rtype: tuple(x, y)

    .. deprecated:: 2.0
       Use `line_iter` instead
    """
    x = _ffi.new('int *')
    y = _ffi.new('int *')
    ret = _lib.TCOD_line_step(x, y)
    if not ret:
        return x[0], y[0]
    return None,None

_line_listener_lock = _threading.Lock()

def line(xo, yo, xd, yd, py_callback) :
    """ Iterate over a line using a callback function.

    Your callback function will take x and y parameters and return True to
    continue iteration or False to stop iteration and return.

    This function includes both the start and end points.

    :param int xo: x origin
    :param int yo: y origin
    :param int xd: x destination
    :param int yd: y destination
    :param function py_callback: Callback that takes x and y parameters and
                                 returns bool.
    :return: Returns False if the callback cancels the line interation by
             returning False or None, otherwise True.
    :rtype: bool

    .. deprecated:: 2.0
       Use `line_iter` instead.
    """
    with _PropagateException() as propagate:
        with _line_listener_lock:
            @_ffi.def_extern(onerror=propagate)
            def _pycall_line_listener(x, y):
                return py_callback(x, y)
            return bool(_lib.TCOD_line(xo, yo, xd, yd,
                                       _lib._pycall_line_listener))

def line_iter(xo, yo, xd, yd):
    """ returns an iterator

    This iterator does not include the origin point.

    :param int xo: x origin
    :param int yo: y origin
    :param int xd: x destination
    :param int yd: y destination

    :return: iterator of (x,y) points
    """
    data = _ffi.new('TCOD_bresenham_data_t *')
    _lib.TCOD_line_init_mt(xo, yo, xd, yd, data)
    x = _ffi.new('int *')
    y = _ffi.new('int *')
    done = False
    while not _lib.TCOD_line_step_mt(x, y, data):
        yield (x[0], y[0])

__all__ = [_name for _name in list(globals()) if _name[0] != '_']
