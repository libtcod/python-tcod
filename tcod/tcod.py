"""This module focuses on improvements to the Python libtcod API.
"""
from __future__ import absolute_import as _

import os as _os
import sys as _sys

import platform as _platform
import weakref as _weakref
import functools as _functools

import numpy as _np

from tcod.libtcod import lib, ffi, BKGND_DEFAULT, BKGND_SET

def _unpack_char_p(char_p):
    if char_p == ffi.NULL:
        return ''
    return ffi.string(char_p).decode()

def _int(int_or_str):
    'return an integer where a single character string may be expected'
    if isinstance(int_or_str, str):
        return ord(int_or_str)
    if isinstance(int_or_str, bytes):
        return int_or_str[0]
    return int(int_or_str) # check for __count__

def _cdata(cdata):
    """covert value into a cffi.CData instance"""
    try: # first check for _CDataWrapper
        cdata = cdata.cdata
    except AttributeError: # assume cdata is valid
        pass
    except KeyError:
        pass
    if cdata is None or cdata == 0: # convert None to NULL
        cdata = ffi.NULL
    return cdata

if _sys.version_info[0] == 2: # Python 2
    def _bytes(string):
        if isinstance(string, unicode):
            return string.encode()
        return string

    def _unicode(string):
        if not isinstance(string, unicode):
            return string.decode()
        return string

else: # Python 3
    def _bytes(string):
        if isinstance(string, str):
            return string.encode()
        return string

    def _unicode(string):
        if isinstance(string, bytes):
            return string.decode()
        return string

def _fmt_bytes(string):
    return _bytes(string).replace(b'%', b'%%')

def _fmt_unicode(string):
    return _unicode(string).replace(u'%', u'%%')

class _PropagateException():
    """ context manager designed to propagate exceptions outside of a cffi
    callback context.  normally cffi suppresses the exception

    when propagate is called this class will hold onto the error until the
    control flow leaves the context, then the error will be raised

    with _PropagateException as propagate:
    # give propagate as onerror parameter for ffi.def_extern
    """

    def __init__(self):
        self.exc_info = None # (exception, exc_value, traceback)

    def propagate(self, *exc_info):
        """ set an exception to be raised once this context exits

        if multiple errors are caught, only keep the first exception raised
        """
        if not self.exc_info:
            self.exc_info = exc_info

    def __enter__(self):
        """ once in context, only the propagate call is needed to use this
        class effectively
        """
        return self.propagate

    def __exit__(self, type, value, traceback):
        """ if we're holding on to an exception, raise it now

        prefers our held exception over any current raising error

        self.exc_info is reset now in case of nested manager shenanigans
        """
        if self.exc_info:
            type, value, traceback = self.exc_info
            self.exc_info = None
        if type:
            # Python 2/3 compatible throw
            exception = type(value)
            exception.__traceback__ = traceback
            raise exception

class _CDataWrapper(object):

    def __init__(self, *args, **kargs):
        self.cdata = self._get_cdata_from_args(*args, **kargs)
        if self.cdata == None:
            self.cdata = ffi.NULL
        super(_CDataWrapper, self).__init__()

    def _get_cdata_from_args(self, *args, **kargs):
        if len(args) == 1 and isinstance(args[0], ffi.CData) and not kargs:
            return args[0]
        else:
            return None


    def __hash__(self):
        return hash(self.cdata)

    def __eq__(self, other):
        try:
            return self.cdata == other.cdata
        except AttributeError:
            return NotImplemented

    def __getattr__(self, attr):
        if 'cdata' in self.__dict__:
            return getattr(self.__dict__['cdata'], attr)
        raise AttributeError(attr)

    def __setattr__(self, attr, value):
        if hasattr(self, 'cdata') and hasattr(self.cdata, attr):
            setattr(self.cdata, attr, value)
        else:
            super(_CDataWrapper, self).__setattr__(attr, value)

def _assert_cdata_is_not_null(func):
    """Any BSP methods which use a cdata object in a TCOD call need to have
    a sanity check, otherwise it may end up passing a NULL pointer"""
    if __debug__:
        @_functools.wraps(func)
        def check_sanity(*args, **kargs):
            assert self.cdata != ffi.NULL and self.cdata is not None, \
                   'cannot use function, cdata is %r' % self.cdata
            return func(*args, **kargs)
    return func

class Key(_CDataWrapper):
    """Key Event instance

    Attributes:
        vk (int): TCOD_keycode_t key code
        c (int): character if vk == TCODK_CHAR else 0
        text (Text): text[TCOD_KEY_TEXT_SIZE]; text if vk == TCODK_TEXT else text[0] == '\0'
        pressed (bool): does this correspond to a key press or key release event ?
        lalt (bool): True when left alt is held.
        lctrl (bool): True when left control is held.
        lmeta (bool): True when left meta key is held.
        ralt (bool): True when right alt is held.
        rctrl (bool): True when right control is held.
        rmeta (bool): True when right meta key is held.
        shift (bool): True when any shift is held.
    """

    _BOOL_ATTRIBUTES = ('lalt', 'lctrl', 'lmeta',
                        'ralt', 'rctrl', 'rmeta', 'pressed', 'shift')

    def __init__(self, *args, **kargs):
        super(Key, self).__init__(*args, **kargs)
        if self.cdata == ffi.NULL:
            self.cdata = ffi.new('TCOD_key_t*')

    def __getattr__(self, attr):
        if attr in self._BOOL_ATTRIBUTES:
            return bool(getattr(self.cdata, attr))
        if attr == 'c':
            return ord(getattr(self.cdata, attr))
        if attr == 'text':
            return _unpack_char_p(getattr(self.cdata, attr))
        return super(Key, self).__getattr__(attr)

class Mouse(_CDataWrapper):
    """Mouse event instance

    Attributes:
        x (int): Absolute mouse position at pixel x.
        y (int):
        dx (int): Movement since last update in pixels.
        dy (int):
        cx (int): Cell coordinates in the root console.
        cy (int):
        dcx (int): Movement since last update in console cells.
        dcy (int):
        lbutton (bool): Left button status.
        rbutton (bool): Right button status.
        mbutton (bool): Middle button status.
        lbutton_pressed (bool): Left button pressed event.
        rbutton_pressed (bool): Right button pressed event.
        mbutton_pressed (bool): Middle button pressed event.
        wheel_up (bool): Wheel up event.
        wheel_down (bool): Wheel down event.
    """

    def __init__(self, *args, **kargs):
        super(Mouse, self).__init__(*args, **kargs)
        if self.cdata == ffi.NULL:
            self.cdata = ffi.new('TCOD_mouse_t*')

class _ChBufferArray(_np.ndarray):
    """Numpy subclass designed to access libtcod's character buffer.

    This class needs to modify the char_t.cf attribute as a side effect so that
    libtcod will select the correct characters on flush.
    """

    def __new__(cls, ch_array, cf_array):
        self = ch_array.view(cls)
        self._cf_array = cf_array
        return self

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self._cf_array = None

    def __repr__(self):
        return repr(self.view(_np.ndarray))

    def __getitem__(self, index):
        """Slicing this array also slices its _cf_array attribute."""
        array = _np.ndarray.__getitem__(self, index)
        if self._cf_array is None or array.size == 1:
            return array.view(_np.ndarray)
        array._cf_array = self._cf_array[index]
        return array

    def _covert_ch_to_cf(self, index, ch_arr):
        """Apply a set of Unicode variables to libtcod's special format.

        _cf_array should be the same shape as ch_arr after being sliced by
        index.
        """
        if lib.TCOD_ctx.max_font_chars == 0:
            return # libtcod not initialized
        ch_table = ffi.buffer(
            lib.TCOD_ctx.ascii_to_tcod[0:lib.TCOD_ctx.max_font_chars])
        ch_table = _np.frombuffer(ch_table, _np.intc)
        self._cf_array[index] = ch_table[ch_arr.ravel()].reshape(ch_arr.shape)

    def __setitem__(self, index, value):
        """Properly set up the char_t.cf variables as a side effect."""
        _np.ndarray.__setitem__(self, index, value)
        if self._cf_array is not None:
            self._covert_ch_to_cf(index, self[index])

class Console(_CDataWrapper):
    """
    Args:
        width (int): Width of the new Console.
        height (int): Height of the new Console.

    .. versionadded:: 2.0
    """

    def __init__(self, *args, **kargs):
        self.cdata = self._get_cdata_from_args(*args, **kargs)
        if self.cdata is None:
            self._init(*args, **kargs)
        self._init_setup_console_data()

    def _init(self, width, height):
        self.cdata = ffi.gc(lib.TCOD_console_new(width, height),
                            lib.TCOD_console_delete)

    def _init_setup_console_data(self):
        """Setup numpy arrays over libtcod data buffers."""
        import numpy
        if self.cdata == ffi.NULL:
            self._console_data = lib.TCOD_ctx.root
        else:
            self._console_data = ffi.cast('TCOD_console_data_t *', self.cdata)

        def unpack_color(image_cdata):
            """return a (height, width, 3) shaped array from an image struct"""
            color_data = lib.TCOD_image_get_colors(image_cdata)
            color_buffer = ffi.buffer(color_data[0:self.width * self.height])
            array = _np.frombuffer(color_buffer, _np.uint8)
            return array.reshape((self.height, self.width, 3))

        self._fg = unpack_color(self._console_data.state.fg_colors)
        self._bg = unpack_color(self._console_data.state.bg_colors)

        buf = self._console_data.state.buf
        buf = ffi.buffer(buf[0:self.width * self.height])
        if ffi.sizeof('char_t') != 12:
            # I'm expecting some compiler to have this at 9.
            raise RuntimeError("Expected ffi.sizeof('char_t') to be 12. "
                               "Got %i instead." % ffi.sizeof('char_t'))
        buf = _np.frombuffer(buf, [('c', _np.intc),
                                   ('cf', _np.intc),
                                   ('dirty', _np.intc)])
        self._buf = buf.reshape((self.height, self.width))
        self._ch = _ChBufferArray(self._buf['c'], self._buf['cf'])

    @property
    def width(self):
        """int: The width of this Console. (read-only)"""
        return lib.TCOD_console_get_width(self.cdata)

    @property
    def height(self):
        """int: The height of this Console. (read-only)"""
        return lib.TCOD_console_get_height(self.cdata)

    @property
    def bg(self):
        """A numpy array with the shape (height, width, 3).

        You can change the background color by using this array.

        Index this array with ``console.bg[y, x, channel]``
        """
        return self._bg

    @property
    def fg(self):
        """A numpy array with the shape (height, width, 3).

        You can change the foreground color by using this array.

        Index this array with ``console.fg[y, x, channel]``
        """
        return self._fg

    @property
    def ch(self):
        """A numpy array with the shape (height, width).

        You can change the character tiles by using this array.

        Index this array with ``console.ch[y, x]``
        """
        return self._ch

    @property
    def default_bg(self):
        """Tuple[int, int, int]: The default background color."""
        color = self._console_data.back
        return color.r, color.g, color.b
    @default_bg.setter
    def default_bg(self, color):
        self._console_data.back = color

    @property
    def default_fg(self):
        """Tuple[int, int, int]: The default foreground color."""
        color = self._console_data.fore
        return color.r, color.g, color.b
    @default_fg.setter
    def default_fg(self, color):
        self._console_data.fore = color

    @property
    def default_blend(self):
        """int: The default blending mode."""
        return self._console_data.bkgnd_flag
    @default_blend.setter
    def default_blend(self, value):
        self._console_data.bkgnd_flag = value

    @property
    def default_alignment(self):
        """int: The default text alignment."""
        return self._console_data.alignment
    @default_alignment.setter
    def default_alignment(self, value):
        self._console_data.alignment = value

    def clear(self):
        """Reset this console to its default colors and the space character.
        """
        lib.TCOD_console_clear(self.cdata)

    def put_char(self, x, y, ch, flag=BKGND_DEFAULT):
        """Draw the character c at x,y using the default colors and a blend mode.

        Args:
            x (int): Character x position from the left.
            y (int): Character y position from the top.
            c (Union[int, AnyStr]): Character to draw, can be an integer or string.
            flag (int): Blending mode to use, defaults to BKGND_DEFAULT.
        """
        lib.TCOD_console_put_char(self.cdata, x, y, _int(ch), flag)

    def put_char_ex(self, x, y, ch, fore, back):
        """Draw the character c at x,y using the colors fore and back.

        Args:
            x (int): Character x position from the left.
            y (int): Character y position from the top.
            c (Union[int, AnyStr]): Character to draw, can be an integer or string.
            fore (Union[Tuple[int, int, int], Sequence[int]]):
                An (r, g, b) sequence or Color instance.
            back (Union[Tuple[int, int, int], Sequence[int]]):
                An (r, g, b) sequence or Color instance.
        """
        lib.TCOD_console_put_char_ex(self.cdata, x, y,
                                 _int(ch), fore, back)

    def set_char_bg(self, x, y, col, flag=BKGND_SET):
        """Change the background color of x,y to col using a blend mode.

        Args:
            x (int): Character x position from the left.
            y (int): Character y position from the top.
            col (Union[Tuple[int, int, int], Sequence[int]]):
                An (r, g, b) sequence or Color instance.
            flag (int): Blending mode to use, defaults to BKGND_SET.
        """
        lib.TCOD_console_set_char_background(self.cdata, x, y, col, flag)

    def set_char_fg(self, x, y, color):
        """Change the foreground color of x,y to col.

        Args:
            x (int): Character x position from the left.
            y (int): Character y position from the top.
            color (Union[Tuple[int, int, int], Sequence[int]]):
                An (r, g, b) sequence or Color instance.
        """
        lib.TCOD_console_set_char_foreground(self.cdata, x, y, col)

    def set_char(self, x, y, ch):
        """Change the character at x,y to c, keeping the current colors.

        Args:
            x (int): Character x position from the left.
            y (int): Character y position from the top.
            c (Union[int, AnyStr]): Character to draw, can be an integer or string.
        """
        lib.TCOD_console_set_char(self.cdata, x, y, _int(ch))

    def print_str(self, x, y, fmt):
        """Print a color formatted string on a console.

        Args:
            x (int): Character x position from the left.
            y (int): Character y position from the top.
            fmt (AnyStr): A unicode or bytes string optionaly using color codes.
        """
        lib.TCOD_console_print_utf(self.cdata, x, y, _fmt_unicode(fmt))

    def print_ex(self, x, y, flag, alignment, fmt):
        """Print a string on a console using a blend mode and alignment mode.

        Args:
            x (int): Character x position from the left.
            y (int): Character y position from the top.
        """
        lib.TCOD_console_print_ex_utf(self.cdata, x, y,
                                      flag, alignment, _fmt_unicode(fmt))

    def print_rect(self, x, y, w, h, fmt):
        """Print a string constrained to a rectangle.

        If h > 0 and the bottom of the rectangle is reached,
        the string is truncated. If h = 0,
        the string is only truncated if it reaches the bottom of the console.



        Returns:
            int: The number of lines of text once word-wrapped.
        """
        return lib.TCOD_console_print_rect_utf(
            self.cdata, x, y, width, height, _fmt_unicode(fmt))

    def print_rect_ex(self, x, y, w, h, flag, alignment, fmt):
        """Print a string constrained to a rectangle with blend and alignment.

        Returns:
            int: The number of lines of text once word-wrapped.
        """
        return lib.TCOD_console_print_rect_ex_utf(self.cdata,
            x, y, width, height, flag, alignment, _fmt_unicode(fmt))

    def get_height_rect(self, x, y, width, height, fmt):
        """Return the height of this text once word-wrapped into this rectangle.

        Returns:
            int: The number of lines of text once word-wrapped.
        """
        return lib.TCOD_console_get_height_rect_utf(
            self.cdata, x, y, width, height, _fmt_unicode(fmt))

    def rect(self, x, y, w, h, clr, flag=BKGND_DEFAULT):
        """Draw a the background color on a rect optionally clearing the text.

        If clr is True the affected tiles are changed to space character.
        """
        lib.TCOD_console_rect(self.cdata, x, y, w, h, clr, flag)

    def hline(self, x, y, width, flag=BKGND_DEFAULT):
        """Draw a horizontal line on the console.

        This always uses the character 196, the horizontal line character.
        """
        lib.TCOD_console_hline(self.cdata, x, y, width, flag)

    def vline(self, x, y, height, flag=BKGND_DEFAULT):
        """Draw a vertical line on the console.

        This always uses the character 179, the vertical line character.
        """
        lib.TCOD_console_vline(self.cdata, x, y, height, flag)

    def print_frame(self, x, y, w, h, clear=True, flag=BKGND_DEFAULT, fmt=b''):
        """Draw a framed rectangle with optinal text.

        This uses the default background color and blend mode to fill the
        rectangle and the default foreground to draw the outline.

        fmt will be printed on the inside of the rectangle, word-wrapped.
        """
        lib.TCOD_console_print_frame(self.cdata, x, y, w, h, clear, flag,
                                  _fmt_bytes(fmt))

    def blit(self, x, y, w, h,
             dest, dest_x, dest_y, fg_alpha=1, bg_alpha=1):
        """Blit this console from x,y,w,h to the console dst at xdst,ydst."""
        lib.TCOD_console_blit(self.cdata, x, y, w, h,
                              _cdata(dst), dest_x, dest_y, fg_alpha, bg_alpha)

    def set_key_color(self, color):
        """Set a consoles blit transparent color."""
        lib.TCOD_console_set_key_color(self.cdata, color)


def clipboard_set(string):
    """Set the clipboard contents to string.

    Args:
        string (AnyStr): A Unicode or UTF-8 encoded string.

    .. versionadded:: 2.0
    """
    lib.TCOD_sys_clipboard_set(_bytes(string))

def clipboard_get():
    """Return the current contents of the clipboard.

    Returns:
        Text: The clipboards current contents.

    .. versionadded:: 2.0
    """
    return _unpack_char_p(lib.TCOD_sys_clipboard_get())
