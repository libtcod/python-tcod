
from .libtcod import _lib, _ffi, _int

# high precision time functions
def set_fps(fps):
    _lib.TCOD_sys_set_fps(fps)

def get_fps():
    return _lib.TCOD_sys_get_fps()

def get_last_frame_length():
    return _lib.TCOD_sys_get_last_frame_length()

def sleep_milli(val):
    _lib.TCOD_sys_sleep_milli(val)

def elapsed_milli():
    return _lib.TCOD_sys_elapsed_milli()

def elapsed_seconds():
    return _lib.TCOD_sys_elapsed_seconds()

def set_renderer(renderer):
    _lib.TCOD_sys_set_renderer(renderer)

def get_renderer():
    return _lib.TCOD_sys_get_renderer()

# easy screenshots
def save_screenshot(name=_ffi.NULL):
    _lib.TCOD_sys_save_screenshot(nam)

# custom fullscreen resolution
def force_fullscreen_resolution(width, height):
    _lib.TCOD_sys_force_fullscreen_resolution(width, height)

def get_current_resolution():
    w = _ffi.new('int *')
    h = _ffi.new('int *')
    _lib.TCOD_sys_get_current_resolution(w, h)
    return w[0], h[0]

def get_char_size():
    w = _ffi.new('int *')
    h = _ffi.new('int *')
    _lib.TCOD_sys_get_char_size(w, h)
    return w[0], h[0]

# update font bitmap
def update_char(asciiCode, fontx, fonty, img, x, y) :
    _lib.TCOD_sys_update_char(_int(asciiCode), fontx, fonty, img, x, y)

# custom SDL post renderer
SDL_RENDERER_FUNC = _ffi.callback('void(void*)')
def register_SDL_renderer(callback):
    global _sdl_renderer_func
    _sdl_renderer_func = SDL_RENDERER_FUNC(callback)
    _lib.TCOD_sys_register_SDL_renderer(_sdl_renderer_func)

def check_for_event(mask, k, m) :
    return _lib.TCOD_sys_check_for_event(mask, k._struct, m._struct)

def wait_for_event(mask, k, m, flush) :
    return _lib.TCOD_sys_wait_for_event(mask, k._struct, m._struct, flush)
    
__all__ = [name for name in list(globals()) if name[0] != '_']
