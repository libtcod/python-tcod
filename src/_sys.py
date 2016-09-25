
from .libtcod import _lib, _ffi, _int

# high precision time functions
def sys_set_fps(fps):
    _lib.TCOD_sys_set_fps(fps)

def sys_get_fps():
    return _lib.TCOD_sys_get_fps()

def sys_get_last_frame_length():
    return _lib.TCOD_sys_get_last_frame_length()

def sys_sleep_milli(val):
    _lib.TCOD_sys_sleep_milli(val)

def sys_elapsed_milli():
    return _lib.TCOD_sys_elapsed_milli()

def sys_elapsed_seconds():
    return _lib.TCOD_sys_elapsed_seconds()

def sys_set_renderer(renderer):
    _lib.TCOD_sys_set_renderer(renderer)

def sys_get_renderer():
    return _lib.TCOD_sys_get_renderer()

# easy screenshots
def sys_save_screenshot(name=_ffi.NULL):
    _lib.TCOD_sys_save_screenshot(nam)

# custom fullscreen resolution
def sys_force_fullscreen_resolution(width, height):
    _lib.TCOD_sys_force_fullscreen_resolution(width, height)

def sys_get_current_resolution():
    w = _ffi.new('int *')
    h = _ffi.new('int *')
    _lib.TCOD_sys_get_current_resolution(w, h)
    return w[0], h[0]

def sys_get_char_size():
    w = _ffi.new('int *')
    h = _ffi.new('int *')
    _lib.TCOD_sys_get_char_size(w, h)
    return w[0], h[0]

# update font bitmap
def sys_update_char(asciiCode, fontx, fonty, img, x, y) :
    _lib.TCOD_sys_update_char(_int(asciiCode), fontx, fonty, img, x, y)

def sys_register_SDL_renderer(callback):
    @_ffi.def_extern()
    def _pycall_sdl_hook(sdl_surface):
        callback(sdl_surface)
    _lib.TCOD_sys_register_SDL_renderer(_lib._pycall_sdl_hook)

def sys_check_for_event(mask, k, m) :
    return _lib.TCOD_sys_check_for_event(mask, k._struct, m._struct)

def sys_wait_for_event(mask, k, m, flush) :
    return _lib.TCOD_sys_wait_for_event(mask, k._struct, m._struct, flush)
    
__all__ = [_name for _name in list(globals()) if _name[0] != '_']
