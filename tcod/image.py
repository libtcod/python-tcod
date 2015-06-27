
from .libtcod import _lib, _ffi, _str


def new(width, height):
    return _lib.TCOD_image_new(width, height)

def clear(image,col) :
    _lib.TCOD_image_clear(image,col)

def invert(image) :
    _lib.TCOD_image_invert(image)

def hflip(image) :
    _lib.TCOD_image_hflip(image)

def rotate90(image, num=1) :
    _lib.TCOD_image_rotate90(image,num)

def vflip(image) :
    _lib.TCOD_image_vflip(image)

def scale(image, neww, newh) :
    _lib.TCOD_image_scale(image, neww, newh)

def set_key_color(image,col) :
    _lib.TCOD_image_set_key_color(image,col)

def get_alpha(image,x,y) :
    return _lib.TCOD_image_get_alpha(image, x, y)

def is_pixel_transparent(image,x,y) :
    return _lib.TCOD_image_is_pixel_transparent(image, x, y)

def load(filename):
    return _lib.TCOD_image_load(_str(filename))

def from_console(console):
    return _lib.TCOD_image_from_console(console)

def refresh_console(image, console):
    _lib.TCOD_image_refresh_console(image, console)

def get_size(image):
    w = _ffi.new('int *')
    h = _ffi.new('int *')
    _lib.TCOD_image_get_size(image, w, h)
    return w.value, h.value

def get_pixel(image, x, y):
    return _lib.TCOD_image_get_pixel(image, x, y)

def get_mipmap_pixel(image, x0, y0, x1, y1):
    return _lib.TCOD_image_get_mipmap_pixel(image, x0, y0, x1, y1)
def put_pixel(image, x, y, col):
    _lib.TCOD_image_put_pixel(image, x, y, col)
    ##_lib.TCOD_image_put_pixel_wrapper(image, x, y, col)

def blit(image, console, x, y, bkgnd_flag, scalex, scaley, angle):
    _lib.TCOD_image_blit(image, console or _ffi.NULL, x, y, bkgnd_flag,
                         scalex, scaley, angle)

def blit_rect(image, console, x, y, w, h, bkgnd_flag):
    _lib.TCOD_image_blit_rect(image, console or _ffi.NULL, x, y, w, h, bkgnd_flag)

def blit_2x(image, console, dx, dy, sx=0, sy=0, w=-1, h=-1):
    _lib.TCOD_image_blit_2x(image, console or _ffi.NULL, dx,dy,sx,sy,w,h)

def save(image, filename):
    _lib.TCOD_image_save(image, _str(filename))

def delete(image):
    _lib.TCOD_image_delete(image)


__all__ = [name for name in list(globals()) if name[0] != '_']