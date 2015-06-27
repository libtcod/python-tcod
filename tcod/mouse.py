
from . import Mouse as _Mouse
from .libtcod import _lib, _ffi

def show_cursor(visible):
    _lib.TCOD_mouse_show_cursor(visible)

def is_cursor_visible():
    return _lib.TCOD_mouse_is_cursor_visible()

def move(x, y):
    _lib.TCOD_mouse_move(x, y)

def get_status():
    mouse=_Mouse()
    mouse._struct = _lib.TCOD_mouse_get_status()
    return mouse

__all__ = [name for name in list(globals()) if name[0] != '_']