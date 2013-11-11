"""
    This module is used internally to access libtcod functions from ctypes.
    It should never be accessed directly.
"""
import sys
import os
import platform
import pkgutil
from ctypes import *

try: # decide how files are unpacked depending on if we have the pkg_resources module
    from pkg_resources import resource_filename
    def _unpackfile(filename):
        return resource_filename(__name__, filename)
except ImportError:
    from tdl import __path__
    def _unpackfile(filename):
        return os.path.abspath(os.path.join(__path__[0], filename))

def _unpackFramework(framework, path):
    """get framework.tar file, remove ".tar" and add path"""
    return os.path.abspath(os.path.join(_unpackfile(framework)[:-4], path))

def _loadDLL(dll):
    """shorter version of file unpacking and linking"""
    return cdll.LoadLibrary(_unpackfile(dll))
    
    
def _get_library_crossplatform():
    bits, linkage = platform.architecture()
    libpath = None
    if 'win32' in sys.platform:
        libSDL = _loadDLL('lib/win32/SDL.dll')
        _loadDLL('lib/win32/zlib1.dll')
        libTCOD = _loadDLL('lib/win32/libtcod-VS.dll')
    elif 'linux' in sys.platform:
        libSDL = None
        #libSDL = cdll.LoadLibrary('SDL.so')
        if bits == '32bit':
            libTCOD = _loadDLL('lib/linux32/libtcod.so')
        elif bits == '64bit':
            libTCOD = _loadDLL('lib/linux64/libtcod.so')
    elif 'darwin' in sys.platform:
        libSDL = _loadDLL('lib/darwin/SDL.dylib')
        libTCOD = _loadDLL('lib/darwin/libtcod.dylib')
        
    else:
        raise ImportError('Operating system "%s" has no supported dynamic link libarary. (%s, %s)' % (sys.platform, bits, linkage))
    return libTCOD, libSDL

_lib, _libSDL = _get_library_crossplatform()

try:
    c_bool
except NameError:
    c_bool = c_byte

# ---------------- SDL ----------------

class SDL_Rect(Structure):
    _fields_ = [('x', c_int16),
                ('y', c_int16),
                ('width', c_uint16),
                ('height', c_uint16)]
   
class SDL_PixelFormat(Structure):
   _fields_ = [('palette', c_void_p),
                ('BitsPerPixel', c_uint8),
                ('BytesPerPixel', c_uint8),
                ('Rloss', c_uint8),
                ('Gloss', c_uint8),
                ('Bloss', c_uint8),
                ('Aloss', c_uint8),
                ('Rshift', c_uint8),
                ('Gshift', c_uint8),
                ('Bshift', c_uint8),
                ('Ashift', c_uint8),
                ('Rmask', c_uint32),
                ('Gmask', c_uint32),
                ('Bmask', c_uint32),
                ('Amask', c_uint32),
                ('colorkey', c_uint32),
                ('alpha', c_uint8),
                ]
   
class SDL_Surface(Structure):
    _fields_ = [('flags', c_uint32),
                ('PixelFormat', POINTER(SDL_PixelFormat)),
                ('width', c_int),
                ('height', c_int),
                ('pitch', c_uint16),
                ('pixels', c_void_p), # Read-write
                ('rect', SDL_Rect),
                ('refcount', c_int)
                ]

# ---------------- libtcod ----------------
    
# COLOR

class _Color(Structure):
    _fields_ = [('r', c_uint8), ('g', c_uint8), ('b', c_uint8)]
    
    @staticmethod
    def new(color):
        if color is None or color is False:
            return color
        if isinstance(color, _Color):
            return color
        if isinstance(color, int):
            # format a web style color with the format 0xRRGGBB
            return _Color(color >> 16 & 0xff, color >> 8 & 0xff, color & 0xff)
        return _Color(*color)
    
    def set(self, r, g, b):
        """perform a slightly faster in-place assignment of this object"""
        self.r = r
        self.g = g
        self.b = b
        return self
    
    def __iter__(self):
        'to make this class more tuple-like'
        return iter((self.r, self.g, self.b))
        
    def __len__(self):
        return 3
        
    def __int__(self):
        "useful to convert back into web format: 0xRRGGBB"
        return (self.r << 16 | self.g << 8 | self.b)
    
    def __getitem__(self, key):
        if key == 0:
            return self.r
        if key == 1:
            return self.g
        if key == 2:
            return self.b
        raise IndexError()
    
    def __eq__(self, other):
        return (self.r, self.g, self.b) == other
        
    def __hash__(self):
        return hash((self.r, self.g, self.b))
    
    def __repr__(self):
        return '<TCODColor %s>' % repr(tuple(self))

# not needed
#_lib.TCOD_color_equals.restype = c_bool
#_lib.TCOD_color_equals.argtypes = (_Color, _Color)
#_lib.TCOD_color_multiply.restype = _Color
#_lib.TCOD_color_multiply.argtypes = (_Color, _Color)
#_lib.TCOD_color_multiply_scalar.restype = _Color
#_lib.TCOD_color_multiply_scalar.argtypes = (_Color , c_float)
#_lib.TCOD_color_lerp.restype = _Color
#_lib.TCOD_color_lerp.argtypes = (_Color, _Color, c_float)
#_lib.TCOD_color_set_HSV.restype = None
#_lib.TCOD_color_set_HSV.argtypes = (POINTER(_Color), c_float, c_float, c_float)
#_lib.TCOD_color_get_HSV.restype = None
#_lib.TCOD_color_get_HSV.argtypes = (_Color, POINTER(c_float), POINTER(c_float), POINTER(c_float))

# CONSOLE

K_NONE = 0
K_ESCAPE = 1
K_BACKSPACE = 2
K_TAB = 3
K_ENTER = 4
K_SHIFT = 5
K_CONTROL = 6
K_ALT = 7
K_PAUSE = 8
K_CAPSLOCK = 9
K_PAGEUP = 10
K_PAGEDOWN = 11
K_END = 12
K_HOME = 13
K_UP = 14
K_LEFT = 15
K_RIGHT = 16
K_DOWN = 17
K_PRINTSCREEN = 18
K_INSERT = 19
K_DELETE = 20
K_LWIN = 21
K_RWIN = 22
K_APPS = 23
K_0 = 24
K_1 = 25
K_2 = 26
K_3 = 27
K_4 = 28
K_5 = 29
K_6 = 30
K_7 = 31
K_8 = 32
K_9 = 33
K_KP0 = 34
K_KP1 = 35
K_KP2 = 36
K_KP3 = 37
K_KP4 = 38
K_KP5 = 39
K_KP6 = 40
K_KP7 = 41
K_KP8 = 42
K_KP9 = 43
K_KPADD = 44
K_KPSUB = 45
K_KPDIV = 46
K_KPMUL = 47
K_KPDEC = 48
K_KPENTER = 49
K_F1 = 50
K_F2 = 51
K_F3 = 52
K_F4 = 53
K_F5 = 54
K_F6 = 55
K_F7 = 56
K_F8 = 57
K_F9 = 58
K_F10 = 59
K_F11 = 60
K_F12 = 61
K_NUMLOCK = 62
K_SCROLLLOCK = 63
K_SPACE = 64
K_CHAR = 65

TCOD_keycode_t = c_int

class _Key(Structure):
    _fields_ = [('vk', TCOD_keycode_t),
                ('c', c_char),
                ('pressed', c_bool),
                ('lalt', c_bool),
                ('lctrl', c_bool),
                ('ralt', c_bool),
                ('rctrl', c_bool),
                ('shift', c_bool),
                ]
                
    def __iter__(self):
        return (getattr(self, attr) for attr, c_type in self._fields_ if attr != 'pressed')


CHAR_HLINE = 196
CHAR_VLINE = 179
CHAR_NE = 191
CHAR_NW = 218
CHAR_SE = 217
CHAR_SW = 192
CHAR_DHLINE = 205
CHAR_DVLINE = 186
CHAR_DNE = 187
CHAR_DNW = 201
CHAR_DSE = 188
CHAR_DSW = 200
CHAR_TEEW = 180
CHAR_TEEE = 195
CHAR_TEEN = 193
CHAR_TEES = 194
CHAR_DTEEW = 181
CHAR_DTEEE = 198
CHAR_DTEEN = 208
CHAR_DTEES = 210
CHAR_CHECKER = 178
CHAR_BLOCK = 219
CHAR_BLOCK1 = 178
CHAR_BLOCK2 = 177
CHAR_BLOCK3 = 176
CHAR_BLOCK_B = 220
CHAR_BLOCK_T = 223
CHAR_DS_CROSSH = 216
CHAR_DS_CROSSV = 215
CHAR_CROSS = 197
CHAR_LIGHT = 15
CHAR_TREE = 5
CHAR_ARROW_N = 24
CHAR_ARROW_S = 25
CHAR_ARROW_E = 26
CHAR_ARROW_W = 27

COLCTRL_1 = 1
COLCTRL_2 = 2
COLCTRL_3 = 3
COLCTRL_4 = 4
COLCTRL_5 = 5
COLCTRL_NUMBER = 5
COLCTRL_FORE_RGB = 6
COLCTRL_BACK_RGB = 7
COLCTRL_STOP = 8

TCOD_colctrl_t = c_int

BKGND_NONE = 0
BKGND_SET = 1
BKGND_MULTIPLY = 2
BKGND_LIGHTEN = 3
BKGND_DARKEN = 4
BKGND_SCREEN = 5
BKGND_COLOR_DODGE = 6
BKGND_COLOR_BURN = 7
BKGND_ADD = 8
BKGND_ADDA = 9
BKGND_BURN = 10
BKGND_OVERLAY = 11
BKGND_ALPH = 12
# just for reference:
#define TCOD_BKGND_ALPHA(alpha) ((TCOD_bkgnd_flag_t)(TCOD_BKGND_ALPH|(((uint8)(alpha*255))<<8)))
#define TCOD_BKGND_ADDALPHA(alpha) ((TCOD_bkgnd_flag_t)(TCOD_BKGND_ADDA|(((uint8)(alpha*255))<<8)))
# these are already in tdl

TCOD_bkgnd_flag_t = c_int
TCOD_renderer_t = c_int

KEY_PRESSED = 1
KEY_RELEASED = 2


#class char_t(Structure):
#    _fields_ = [('c', c_int), # character ascii code
#                ('cf', c_int), # character number in font
#                ('fore', _Color), ('back', _Color),
#                ('dirty', c_uint8)]
                
                
#class TCOD_console_t(Structure):
#    _fields_ = [('buf', POINTER(char_t)), # current console
#                ('oldbuf', POINTER(char_t)), # console for last frame
#                ('w', c_int), ('h', c_int), # console width and height (in characters,not pixels)
#                ]
                
#	/* default background operator for print & print_rect functions */
#	TCOD_bkgnd_flag_t bkgnd_flag;
#	/* default alignment for print & print_rect functions */
#	TCOD_alignment_t alignment;
#	/* foreground (text), background and key colors */
#	TCOD_color_t fore,back,key;
#	uint8 fade;
#	bool haskey; /* a key color has been defined */
#} TCOD_console_data_t;

TCOD_console_t = c_void_p

_lib.TCOD_console_init_root.restype = None
_lib.TCOD_console_init_root.argtypes = (c_int, c_int, c_char_p, c_bool, TCOD_renderer_t)
_lib.TCOD_console_set_custom_font.restype = None
_lib.TCOD_console_set_custom_font.argtypes = (c_char_p, c_int, c_int, c_int)
_lib.TCOD_console_set_window_title.restype = None
_lib.TCOD_console_set_window_title.argtypes = (c_void_p,)
_lib.TCOD_console_set_fullscreen.restype = None
_lib.TCOD_console_set_fullscreen.argtypes = (c_bool,)
_lib.TCOD_console_is_fullscreen.restype = c_bool
_lib.TCOD_console_is_fullscreen.argtypes = None
_lib.TCOD_console_is_window_closed.restype = c_bool
_lib.TCOD_console_is_window_closed.argtypes = None

_lib.TCOD_console_set_default_background.restype = None
_lib.TCOD_console_set_default_background.argtypes = (TCOD_console_t, _Color)
_lib.TCOD_console_set_default_foreground.restype = None
_lib.TCOD_console_set_default_foreground.argtypes = (TCOD_console_t, _Color)
_lib.TCOD_console_clear.restype = None
_lib.TCOD_console_clear.argtypes = (TCOD_console_t,)
_lib.TCOD_console_set_char_background.restype = None
_lib.TCOD_console_set_char_background.argtypes = (TCOD_console_t, c_int, c_int, _Color, TCOD_bkgnd_flag_t)
_lib.TCOD_console_set_char_foreground.restype = None
_lib.TCOD_console_set_char_foreground.argtypes = (TCOD_console_t, c_int, c_int, _Color)
_lib.TCOD_console_set_char.restype = None
_lib.TCOD_console_set_char.argtypes = (TCOD_console_t, c_int, c_int, c_int)
_lib.TCOD_console_put_char.restype = None
_lib.TCOD_console_put_char.argtypes = (TCOD_console_t, c_int, c_int, c_int, TCOD_bkgnd_flag_t)
_lib.TCOD_console_put_char_ex.restype = None
_lib.TCOD_console_put_char_ex.argtypes = (TCOD_console_t, c_int, c_int, c_int, _Color, _Color)

## uncomment if needed later
#_lib.TCOD_console_rect.restype = None
#_lib.TCOD_console_rect.argtypes = (TCOD_console_t, c_int, c_int, c_int, c_int, c_bool, TCOD_bkgnd_flag_t)
#_lib.TCOD_console_hline.restype = None
#_lib.TCOD_console_hline.argtypes = (TCOD_console_t, c_int, c_int, c_int, TCOD_bkgnd_flag_t)
#_lib.TCOD_console_vline.restype = None
#_lib.TCOD_console_vline.argtypes = (TCOD_console_t, c_int, c_int, c_int, TCOD_bkgnd_flag_t)
#_lib.TCOD_console_print_frame.restype = None
#_lib.TCOD_console_print_frame.argtypes = (TCOD_console_t, c_int, c_int, c_int, c_int, c_bool, c_char_p)#...?

#_lib.TCOD_console_get_background_color.restype = _Color
#_lib.TCOD_console_get_background_color.argtypes = (TCOD_console_t,)
#_lib.TCOD_console_get_foreground_color.restype = _Color
#_lib.TCOD_console_get_foreground_color.argtypes = (TCOD_console_t,)
_lib.TCOD_console_get_char_background_wrapper.restype = _Color
_lib.TCOD_console_get_char_background_wrapper.argtypes = (TCOD_console_t, c_int, c_int)
_lib.TCOD_console_get_char_foreground_wrapper.restype = _Color
_lib.TCOD_console_get_char_foreground_wrapper.argtypes = (TCOD_console_t, c_int, c_int)
_lib.TCOD_console_get_char.restype = c_int
_lib.TCOD_console_get_char.argtypes = (TCOD_console_t, c_int, c_int)

_lib.TCOD_console_set_fade.restype = None
_lib.TCOD_console_set_fade.argtypes = (c_uint8, _Color)
_lib.TCOD_console_get_fade.restype = c_uint8
_lib.TCOD_console_get_fade.argtypes = ()
_lib.TCOD_console_get_fading_color.restype = _Color
_lib.TCOD_console_get_fading_color.argtypes = ()

_lib.TCOD_console_flush.restype = None
_lib.TCOD_console_flush.argtypes = ()

_lib.TCOD_console_set_color_control.restype = None
_lib.TCOD_console_set_color_control.argtypes = (TCOD_colctrl_t, _Color, _Color)

_lib.TCOD_console_check_for_keypress.restype = _Key
_lib.TCOD_console_check_for_keypress.argtypes = (c_int,)
_lib.TCOD_console_wait_for_keypress.restype = _Key
_lib.TCOD_console_wait_for_keypress.argtypes = (c_bool,)
_lib.TCOD_console_wait_for_keypress_wrapper.restype = None
_lib.TCOD_console_wait_for_keypress_wrapper.argtypes = (POINTER(_Key), c_bool)
_lib.TCOD_console_set_keyboard_repeat.restype = None
_lib.TCOD_console_set_keyboard_repeat.argtypes = (c_int, c_int)
_lib.TCOD_console_disable_keyboard_repeat.restype = None
_lib.TCOD_console_disable_keyboard_repeat.argtypes = ()
_lib.TCOD_console_is_key_pressed.restype = c_bool
_lib.TCOD_console_is_key_pressed.argtypes = (TCOD_keycode_t,)

_lib.TCOD_console_new.restype = TCOD_console_t
_lib.TCOD_console_new.argtypes = (c_int, c_int)
_lib.TCOD_console_get_width.restype = c_int
_lib.TCOD_console_get_width.argtypes = (TCOD_console_t,)
_lib.TCOD_console_get_height.restype = c_int
_lib.TCOD_console_get_height.argtypes = (TCOD_console_t,)
_lib.TCOD_console_blit.restype = None
_lib.TCOD_console_blit.argtypes = (TCOD_console_t, c_int, c_int, c_int, c_int, TCOD_console_t, c_int, c_int, c_float, c_float)
_lib.TCOD_console_delete.restype = None
_lib.TCOD_console_delete.argtypes = (TCOD_console_t,)


# SYS

_lib.TCOD_sys_elapsed_milli.restype = c_uint32
_lib.TCOD_sys_elapsed_milli.argtypes = ()
_lib.TCOD_sys_elapsed_seconds.restype = c_float
_lib.TCOD_sys_elapsed_seconds.argtypes = ()
_lib.TCOD_sys_sleep_milli.restype = None
_lib.TCOD_sys_sleep_milli.argtypes = (c_uint32,)
_lib.TCOD_sys_save_screenshot.restype = None
_lib.TCOD_sys_save_screenshot.argtypes = (c_char_p,)
_lib.TCOD_sys_force_fullscreen_resolution.restype = None
_lib.TCOD_sys_force_fullscreen_resolution.argtypes = (c_int, c_int)
_lib.TCOD_sys_set_fps.restype = None
_lib.TCOD_sys_set_fps.argtypes = (c_int,)
_lib.TCOD_sys_get_fps.restype = c_int
_lib.TCOD_sys_get_fps.argtypes = ()
_lib.TCOD_sys_get_last_frame_length.restype = c_float
_lib.TCOD_sys_get_last_frame_length.argtypes = ()
_lib.TCOD_sys_get_current_resolution.restype = None
_lib.TCOD_sys_get_current_resolution.argtypes = (POINTER(c_int), POINTER(c_int))

_lib.TCOD_sys_clipboard_set.restype = None
_lib.TCOD_sys_clipboard_set.argtypes = (c_char_p,)
_lib.TCOD_sys_clipboard_get.restype = c_char_p
_lib.TCOD_sys_clipboard_get.argtypes = ()

_RENDERCALL = CFUNCTYPE(None, POINTER(SDL_Surface))
_lib.TCOD_sys_register_SDL_renderer.restype = None
_lib.TCOD_sys_register_SDL_renderer.argtypes = None

# IMAGE

# TCOD_image_t = c_void_p

# _lib.TCOD_image_new.restype = TCOD_image_t
# _lib.TCOD_image_new.argtypes = (c_int, c_int)
# _lib.TCOD_image_from_console.restype = TCOD_image_t
# _lib.TCOD_image_from_console.argtypes = (TCOD_console_t,)
# _lib.TCOD_image_load.restype = TCOD_image_t
# _lib.TCOD_image_load.argtypes = (c_char_p,)
# _lib.TCOD_image_clear.restype = None
# _lib.TCOD_image_clear.argtypes = (TCOD_image_t, _Color)
# _lib.TCOD_image_save.restype = None
# _lib.TCOD_image_save.argtypes = (TCOD_image_t, c_char_p)
# _lib.TCOD_image_get_size.restype = None
# _lib.TCOD_image_get_size.argtypes = (TCOD_image_t, POINTER(c_int), POINTER(c_int))
# _lib.TCOD_image_get_pixel.restype = _Color
# _lib.TCOD_image_get_pixel.argtypes = (TCOD_image_t, c_int, c_int)
# _lib.TCOD_image_get_mipmap_pixel.restype = _Color
# _lib.TCOD_image_get_mipmap_pixel.argtypes = (TCOD_image_t, c_float, c_float, c_float, c_float)
# _lib.TCOD_image_put_pixel.restype = None
# _lib.TCOD_image_put_pixel.argtypes = (TCOD_image_t, c_int, c_int, _Color)
# _lib.TCOD_image_blit.restype = None
# _lib.TCOD_image_blit.argtypes = (TCOD_image_t, TCOD_console_t, c_float, c_float, TCOD_bkgnd_flag_t, c_float, c_float, c_float)
# _lib.TCOD_image_blit_rect.restype = None
# _lib.TCOD_image_blit_rect.argtypes = (TCOD_image_t, TCOD_console_t, c_int, c_int, c_int, c_int, TCOD_bkgnd_flag_t)
# _lib.TCOD_image_delete.restype = None
# _lib.TCOD_image_delete.argtypes = (TCOD_image_t,)
# _lib.TCOD_image_set_key_color.restype = None
# _lib.TCOD_image_set_key_color.argtypes = (TCOD_image_t, _Color)
# _lib.TCOD_image_is_pixel_transparent.restype = c_bool
# _lib.TCOD_image_is_pixel_transparent.argtypes = (TCOD_image_t, c_int, c_int)

# MOUSE

class _Mouse(Structure):
    _fields_ = [('x', c_int), # absolute position
                ('y', c_int),
                ('dx', c_int), # movement since last update
                ('dy', c_int),
                ('cx', c_int), # cell coordinates in the root console
                ('cy', c_int),
                ('dcx', c_int), # movement since last update in console cells
                ('dcy', c_int),
                ('lbutton', c_bool),
                ('rbutton', c_bool),
                ('mbutton', c_bool),
                ('lbutton_pressed', c_bool),
                ('rbutton_pressed', c_bool),
                ('mbutton_pressed', c_bool),
                ('wheel_up', c_bool),
                ('wheel_down', c_bool),
                ]
    
    @property
    def motion(self):
        return (self.x, self.y), (self.cx, self.cy), (self.dx, self.dy), (self.dcx, self.dcy)
    
    @property
    def button(self):
        return self.lbutton, self.mbutton, self.rbutton
    
    @property
    def button_pressed(self):
        return self.lbutton_pressed, self.mbutton_pressed, self.rbutton_pressed
    
    def __repr__(self):
        return '<TCOD_Mouse %s>' % str(self.motion + self.button)

_lib.TCOD_mouse_get_status_wrapper.restype = None
_lib.TCOD_mouse_get_status_wrapper.argtypes = (POINTER(_Mouse),)
_lib.TCOD_mouse_show_cursor.restype = None
_lib.TCOD_mouse_show_cursor.argtypes = (c_bool,)
_lib.TCOD_mouse_is_cursor_visible.restype = c_bool
_lib.TCOD_mouse_is_cursor_visible.argtypes = ()
_lib.TCOD_mouse_move.restype = None
_lib.TCOD_mouse_move.argtypes = (c_int, c_int)

TCOD_EVENT_KEY_PRESS=1
TCOD_EVENT_KEY_RELEASE=2
TCOD_EVENT_KEY=TCOD_EVENT_KEY_PRESS|TCOD_EVENT_KEY_RELEASE
TCOD_EVENT_MOUSE_MOVE=4
TCOD_EVENT_MOUSE_PRESS=8
TCOD_EVENT_MOUSE_RELEASE=16
TCOD_EVENT_MOUSE=TCOD_EVENT_MOUSE_MOVE|TCOD_EVENT_MOUSE_PRESS|TCOD_EVENT_MOUSE_RELEASE
TCOD_EVENT_ANY=TCOD_EVENT_KEY|TCOD_EVENT_MOUSE

_lib.TCOD_sys_wait_for_event.restype = c_int
_lib.TCOD_sys_wait_for_event.argtypes = (c_int, POINTER(_Key), POINTER(_Mouse))
_lib.TCOD_sys_check_for_event.restype = c_int
_lib.TCOD_sys_check_for_event.argtypes = (c_int, POINTER(_Key), POINTER(_Mouse))

# FOV

TCOD_map_t = c_void_p

_lib.TCOD_map_new.restype = TCOD_map_t
_lib.TCOD_map_new.argtypes = (c_int, c_int)
_lib.TCOD_map_delete.restype = None
_lib.TCOD_map_delete.argtypes = (TCOD_map_t,)
_lib.TCOD_map_set_properties.restype = None
_lib.TCOD_map_set_properties.argtypes = (TCOD_map_t, c_int, c_int, c_bool, c_bool)

_lib.TCOD_map_compute_fov.restype = None
_lib.TCOD_map_compute_fov.argtypes = (TCOD_map_t, c_int, c_int, c_int, c_bool, c_int)
_lib.TCOD_map_is_in_fov.restype = c_bool
_lib.TCOD_map_is_in_fov.argtypes = (TCOD_map_t, c_int, c_int)

# PATH

TCOD_path_t = c_void_p
_PATHCALL = CFUNCTYPE(c_float, c_int, c_int, c_int, c_int, py_object)

_lib.TCOD_path_new_using_map.restype = TCOD_path_t
_lib.TCOD_path_new_using_map.argtypes = (TCOD_map_t, c_float)
_lib.TCOD_path_new_using_function.restype = TCOD_path_t
_lib.TCOD_path_new_using_function.argtypes = (c_int, c_int, _PATHCALL, py_object, c_float)
_lib.TCOD_path_compute.restype = c_bool
_lib.TCOD_path_compute.argtypes = (TCOD_path_t, c_int, c_int, c_int, c_int)
_lib.TCOD_path_walk.restype = c_bool
_lib.TCOD_path_walk.argtypes = (TCOD_path_t, POINTER(c_int), POINTER(c_int), c_bool)
_lib.TCOD_path_size.restype = c_int
_lib.TCOD_path_size.argtypes = (TCOD_path_t,)
_lib.TCOD_path_delete.restype = None
_lib.TCOD_path_delete.argtypes = (TCOD_path_t,)

# Dijkstra

TCOD_dijkstra_t = c_void_p

_lib.TCOD_dijkstra_new.restype = TCOD_dijkstra_t
_lib.TCOD_dijkstra_new.argtypes = (TCOD_map_t, c_float)
_lib.TCOD_dijkstra_new_using_function.restype = TCOD_dijkstra_t
_lib.TCOD_dijkstra_new_using_function.argtypes = (c_int, c_int, POINTER(_PATHCALL), py_object, c_float)
_lib.TCOD_dijkstra_compute.restype = None
_lib.TCOD_dijkstra_compute.argtypes = (TCOD_dijkstra_t, c_int, c_int)
_lib.TCOD_dijkstra_get_distance.restype = c_float
_lib.TCOD_dijkstra_get_distance.argtypes = (TCOD_dijkstra_t, c_int, c_int)
_lib.TCOD_dijkstra_path_set.restype = c_bool
_lib.TCOD_dijkstra_path_set.argtypes = (TCOD_dijkstra_t, c_int, c_int)
_lib.TCOD_dijkstra_size.restype = c_int
_lib.TCOD_dijkstra_size.argtypes = (TCOD_dijkstra_t,)
_lib.TCOD_dijkstra_reverse.restype = None
_lib.TCOD_dijkstra_reverse.argtypes = (TCOD_dijkstra_t,)
_lib.TCOD_dijkstra_path_walk.restype = None
_lib.TCOD_dijkstra_path_walk.argtypes = (TCOD_path_t, POINTER(c_int), POINTER(c_int))
_lib.TCOD_dijkstra_delete.restype = None
_lib.TCOD_dijkstra_delete.argtypes = (TCOD_dijkstra_t,)

# RANDOM


RNG_MT = 1 # Mersenne twister
RNG_CMWC = 2 # Multiply-with-carry

TCOD_random_t = c_void_p

#TCODLIB_API TCOD_random_t TCOD_random_new_from_seed(TCOD_random_algo_t algo, uint32 seed);
_lib.TCOD_random_new_from_seed.restype = TCOD_random_t
_lib.TCOD_random_new_from_seed.argtypes = (c_int, c_uint32)
#TCODLIB_API void TCOD_random_delete(TCOD_random_t mersenne);
_lib.TCOD_random_delete.restype = None
_lib.TCOD_random_delete.argtypes = (TCOD_random_t,)


# NOISE

NOISE_PERLIN = 1
NOISE_SIMPLEX = 2
NOISE_WAVELET = 4

TCOD_noise_t = c_void_p

#TCODLIB_API TCOD_noise_t TCOD_noise_new(int dimensions, float hurst, float lacunarity, TCOD_random_t random);
_lib.TCOD_noise_new.restype = TCOD_noise_t
_lib.TCOD_noise_new.argtypes = (c_int, c_float, c_float)

#TCODLIB_API void TCOD_noise_set_type (TCOD_noise_t noise, TCOD_noise_type_t type);
_lib.TCOD_noise_set_type.restype = None
_lib.TCOD_noise_set_type.argtypes = (TCOD_noise_t, c_int)
#TCODLIB_API float TCOD_noise_get (TCOD_noise_t noise, float *f);
_lib.TCOD_noise_get.restype = c_float
_lib.TCOD_noise_get.argtypes = (TCOD_noise_t, POINTER(c_float))
#TCODLIB_API float TCOD_noise_get_fbm (TCOD_noise_t noise, float *f, float octaves);
_lib.TCOD_noise_get_fbm.restype = c_float
_lib.TCOD_noise_get_fbm.argtypes = (TCOD_noise_t, POINTER(c_float), c_float)
#TCODLIB_API float TCOD_noise_get_turbulence (TCOD_noise_t noise, float *f, float octaves);
_lib.TCOD_noise_get_turbulence.restype = c_float
_lib.TCOD_noise_get_turbulence.argtypes = (TCOD_noise_t, POINTER(c_float), c_float)
#TCODLIB_API void TCOD_noise_delete(TCOD_noise_t noise);
_lib.TCOD_noise_delete.restype = None
_lib.TCOD_noise_delete.argtypes = (TCOD_noise_t,)


#_lib..restype = None
#_lib..argtypes = ()

# these functions should perform ultra fast once I set them up
# after testing it seems that converting python types to ctypes is slow no
# matter how they are passed to C
_lib.TCOD_console_fill_background.restype = None
_lib.TCOD_console_fill_background.argtypes = (TCOD_console_t, POINTER(c_int), POINTER(c_int), POINTER(c_int))
_lib.TCOD_console_fill_foreground.restype = None
_lib.TCOD_console_fill_foreground.argtypes = (TCOD_console_t, POINTER(c_int), POINTER(c_int), POINTER(c_int))
_lib.TCOD_console_fill_char.restype = None
_lib.TCOD_console_fill_char.argtypes = (TCOD_console_t, POINTER(c_int));

__all__ = ()
