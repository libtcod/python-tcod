
from . import Mouse as _Mouse
from .libtcod import _lib, _ffi

def mouse_show_cursor(visible):
    _lib.TCOD_mouse_show_cursor(visible)

def mouse_is_cursor_visible():
    return _lib.TCOD_mouse_is_cursor_visible()

def mouse_move(x, y):
    _lib.TCOD_mouse_move(x, y)

def mouse_get_status():
    mouse=_Mouse()
    mouse._struct = _lib.TCOD_mouse_get_status()
    return mouse

__all__ = [_name for _name in list(globals()) if _name[0] != '_']
