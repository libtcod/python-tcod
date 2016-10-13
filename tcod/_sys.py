
from .libtcod import _lib, _ffi, _int, _bytes, _unpack_char_p, _PropagateException

# high precision time functions
def sys_set_fps(fps):
    """Set the maximum frame rate.

    You can disable the frame limit again by setting fps to 0.

    :param int fps: A frame rate limit (i.e. 60)
    """
    _lib.TCOD_sys_set_fps(fps)

def sys_get_fps():
    """Return the current frames per second.

    This the actual frame rate, not the frame limit set by
    :any:`tcod.sys_set_fps`.

    This number is updated every second.

    :rtype: int
    """
    return _lib.TCOD_sys_get_fps()

def sys_get_last_frame_length():
    """Return the delta time of the last rendered frame in seconds.

    :rtype: float
    """
    return _lib.TCOD_sys_get_last_frame_length()

def sys_sleep_milli(val):
    """Sleep for 'val' milliseconds.

    :param int val: Time to sleep for in milliseconds.

    .. deprecated:: 2.0
       Use :any:`time.sleep` instead.
    """
    _lib.TCOD_sys_sleep_milli(val)

def sys_elapsed_milli():
    """Get number of milliseconds since the start of the program.

    :rtype: int

    .. deprecated:: 2.0
       Use :any:`time.clock` instead.
    """
    return _lib.TCOD_sys_elapsed_milli()

def sys_elapsed_seconds():
    """Get number of seconds since the start of the program.

    :rtype: float

    .. deprecated:: 2.0
       Use :any:`time.clock` instead.
    """
    return _lib.TCOD_sys_elapsed_seconds()

def sys_set_renderer(renderer):
    """Change the current rendering mode to renderer.

    .. deprecated:: 2.0
       RENDERER_GLSL and RENDERER_OPENGL are not currently available.
    """
    _lib.TCOD_sys_set_renderer(renderer)

def sys_get_renderer():
    """Return the current rendering mode.

    """
    return _lib.TCOD_sys_get_renderer()

# easy screenshots
def sys_save_screenshot(name=None):
    """Save a screenshot to a file.

    By default this will automatically save screenshots in the working
    directory.

    The automatic names are formatted as screenshotNNN.png.  For example:
    screenshot000.png, screenshot001.png, etc.  Whichever is available first.

    :param str file: File path to save screenshot.

    """
    if name is not None:
        name = _bytes(name)
    _lib.TCOD_sys_save_screenshot(name or _ffi.NULL)

# custom fullscreen resolution
def sys_force_fullscreen_resolution(width, height):
    """Force a specific resolution in fullscreen.

    Will use the smallest available resolution so that:

    * resolution width >= width and
      resolution width >= root console width * font char width
    * resolution height >= height and
      resolution height >= root console height * font char height
    """
    _lib.TCOD_sys_force_fullscreen_resolution(width, height)

def sys_get_current_resolution():
    """Return the current resolution as (width, height)"""
    w = _ffi.new('int *')
    h = _ffi.new('int *')
    _lib.TCOD_sys_get_current_resolution(w, h)
    return w[0], h[0]

def sys_get_char_size():
    """Return the current fonts character size as (width, height)"""
    w = _ffi.new('int *')
    h = _ffi.new('int *')
    _lib.TCOD_sys_get_char_size(w, h)
    return w[0], h[0]

# update font bitmap
def sys_update_char(asciiCode, fontx, fonty, img, x, y) :
    """Dynamically update the current frot with img.

    All cells using this asciiCode will be updated
    at the next call to :any:`tcod.console_flush`.

    :param int asciiCode: Ascii code corresponding to the character to update.
    :param int fontx: Left coordinate of the character
                      in the bitmap font (in tiles)
    :param int fonty: Top coordinate of the character
                      in the bitmap font (in tiles)
    :param img: An image containing the new character bitmap.
    :param int x: Left pixel of the character in the image.
    :param int y: Top pixel of the character in the image.
    """
    _lib.TCOD_sys_update_char(_int(asciiCode), fontx, fonty, img, x, y)

def sys_register_SDL_renderer(callback):
    """Register a custom randering function with libtcod.

    The callack will receive a :any:`CData <ffi-cdata>` void* to an
    SDL_Surface* struct.

    The callback is called on every call to :any:`tcod.console_flush`.

    :param callable callback: A function which takes a single argument.
    """
    with _PropagateException() as propagate:
        @_ffi.def_extern(onerror=propagate)
        def _pycall_sdl_hook(sdl_surface):
            callback(sdl_surface)
        _lib.TCOD_sys_register_SDL_renderer(_lib._pycall_sdl_hook)

def sys_check_for_event(mask, k, m):
    """Check for events.

    mask can be any of the following:

    * tcod.EVENT_NONE
    * tcod.EVENT_KEY_PRESS
    * tcod.EVENT_KEY_RELEASE
    * tcod.EVENT_KEY
    * tcod.EVENT_MOUSE_MOVE
    * tcod.EVENT_MOUSE_PRESS
    * tcod.EVENT_MOUSE_RELEASE
    * tcod.EVENT_MOUSE
    * tcod.EVENT_FINGER_MOVE
    * tcod.EVENT_FINGER_PRESS
    * tcod.EVENT_FINGER_RELEASE
    * tcod.EVENT_FINGER
    * tcod.EVENT_ANY

    :param mask: Event types to wait for.
    :param Key k: :any:`tcod.Key` instance which might be updated with
                  an event.  Can be None.

    :param Mouse m: :any:`tcod.Mouse` instance which might be updated
                    with an event.  Can be None.
    """
    k = _ffi.NULL if k is None else k._struct
    m = _ffi.NULL if m is None else m._struct
    return _lib.TCOD_sys_check_for_event(mask, k, m)

def sys_wait_for_event(mask, k, m, flush) :
    """Wait for events.

    mask can be any of the following:

    * tcod.EVENT_NONE
    * tcod.EVENT_KEY_PRESS
    * tcod.EVENT_KEY_RELEASE
    * tcod.EVENT_KEY
    * tcod.EVENT_MOUSE_MOVE
    * tcod.EVENT_MOUSE_PRESS
    * tcod.EVENT_MOUSE_RELEASE
    * tcod.EVENT_MOUSE
    * tcod.EVENT_FINGER_MOVE
    * tcod.EVENT_FINGER_PRESS
    * tcod.EVENT_FINGER_RELEASE
    * tcod.EVENT_FINGER
    * tcod.EVENT_ANY

    If flush is True then the buffer will be cleared before waiting. Otherwise
    each available event will be returned in the order they're recieved.

    :param mask: Event types to wait for.
    :param Key k: :any:`tcod.Key` instance which might be updated with
                  an event.  Can be None.

    :param Mouse m: :any:`tcod.Mouse` instance which might be updated
                    with an event.  Can be None.

    :param bool flush: Clear the buffer before waiting.
    """
    k = _ffi.NULL if k is None else k._struct
    m = _ffi.NULL if m is None else m._struct
    return _lib.TCOD_sys_wait_for_event(mask, k, m, flush)

def clipboard_set(string):
    """Set the clipboard contents to string.

    .. versionadded:: 2.0
    """
    _lib.TCOD_sys_clipboard_set(_bytes(string))

def clipboard_get():
    """Return the current contents of the clipboard.

    .. versionadded:: 2.0
    """
    return _unpack_char_p(_lib.TCOD_sys_clipboard_get())

__all__ = [_name for _name in list(globals()) if _name[0] != '_']
