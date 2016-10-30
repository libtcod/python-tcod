"""This module focuses on improvements to the Python libtcod API.
"""
from __future__ import absolute_import as _

import os as _os
import sys as _sys

import platform as _platform
import weakref as _weakref
import functools as _functools

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

class BSP(_CDataWrapper):
    """


    Attributes:
        x (int): Rectangle left coordinate.
        y (int): Rectangle top coordinate.
        w (int): Rectangle width.
        h (int): Rectangle height.

    Args:
        x (int): Rectangle left coordinate.
        y (int): Rectangle top coordinate.
        w (int): Rectangle width.
        h (int): Rectangle height.

    .. versionchanged:: 2.0
       You can create BSP's with this class contructor instead of using
       :any:`bsp_new_with_size`.
    """

    def __init__(self, *args, **kargs):
        self._reference = None # to prevent garbage collection
        self._children = _weakref.WeakSet() # used by _invalidate_children
        super(BSP, self).__init__(*args, **kargs)
        if self._get_cdata_from_args(*args, **kargs) is None:
            self._init(*args, **kargs)

    def _init(self, x, y, w, h):
        self.cdata = ffi.gc(lib.TCOD_bsp_new_with_size(x, y, w, h),
                             lib.TCOD_bsp_delete)

    def _pass_reference(self, reference):
        self._reference = reference
        self._reference._children.add(self)
        return self

    def _invalidate_children(self):
        """Invalidates BSP instances known to be based off of this one."""
        for child in self._children:
            child._reference = None
            child._invalidate_children()
            child.cdata = ffi.NULL
        self._children.clear()

    def __str__(self):
        """Provide a useful readout when printed."""
        if not self.cdata:
            return '<%s NULL!>' % self.__class__.__name__

        status = 'leaf'
        if not self.is_leaf():
            status = ('split at dicision=%i,orientation=%r' %
                      (self.get_division(), self.get_orientation()))

        return ('<%s(x=%i,y=%i,w=%i,h=%i)depth=%i,%s>' %
                (self.__class__.__name__,
                 self.x, self.y, self.w, self.h, self.get_depth(), status))

    def get_depth(self):
        """Return the depth of this node.

        Returns:
            int: This nodes depth.

        .. versionadded:: 2.0
        """
        return self.cdata.level

    def get_division(self):
        """Return the point where this node was divided into parts.

        Returns:
            Optional[int]: The integer of where the node was split or None.

        .. versionadded:: 2.0
        """
        if self.is_leaf():
            return None
        return self.cdata.position

    def get_orientation(self):
        """Return this nodes split orientation.

        Returns:
            Optional[Text]: 'horizontal', 'vertical', or None

        .. versionadded:: 2.0
        """
        if self.is_leaf():
            return None
        elif self.cdata.horizontal:
            return 'horizontal'
        else:
            return 'vertical'

    @_assert_cdata_is_not_null
    def split_once(self, orientation, position):
        """

        .. versionadded:: 2.0
        """
        # orientation = horz
        if orientation[:1].lower() == 'h':
            lib.TCOD_bsp_split_once(self.cdata, True, position)
        elif orientation[:1].lower() == 'v':
            lib.TCOD_bsp_split_once(self.cdata, False, position)
        else:
            raise ValueError("orientation must be 'horizontal' or 'vertical'"
                             "\nNot %r" % orientation)

    @_assert_cdata_is_not_null
    def split_recursive(self, depth, min_width, min_height,
                        max_horz_ratio, max_vert_raito, random=None):
        """

        .. versionadded:: 2.0
        """
        lib.TCOD_bsp_split_recursive(self.cdata, random or ffi.NULL,
                                      depth, min_width, min_height,
                                      max_horz_ratio, max_vert_raito)

    @_assert_cdata_is_not_null
    def resize(self, x, y, w, h):
        """Resize this BSP to the provided rectangle.

        Args:
            x (int): Rectangle left coordinate.
            y (int): Rectangle top coordinate.
            w (int): Rectangle width.
            h (int): Rectangle height.

        .. versionadded:: 2.0
        """
        lib.TCOD_bsp_resize(self.cdata, x, y, w, h)

    @_assert_cdata_is_not_null
    def get_left(self):
        """Return this BSP's 'left' child.

        Returns None if this BSP is a leaf node.

        Returns:
            Optional[BSP]: This nodes left/top child or None.

        .. versionadded:: 2.0
        """
        if self.is_leaf():
            return None
        return BSP(lib.TCOD_bsp_left(self.cdata))._pass_reference(self)

    @_assert_cdata_is_not_null
    def get_right(self):
        """Return this BSP's 'right' child.

        Returns None if this BSP is a leaf node.

        Returns:
            Optional[BSP]: This nodes right/bottom child or None.

        .. versionadded:: 2.0
        """
        if self.is_leaf():
            return None
        return BSP(lib.TCOD_bsp_right(self.cdata))._pass_reference(self)

    @_assert_cdata_is_not_null
    def get_parent(self):
        """Return this BSP's parent node.

        Returns:
            Optional[BSP]: Returns the parent node as a BSP instance.
                           Returns None if this BSP has no parent.

        .. versionadded:: 2.0
        """
        node = BSP(lib.TCOD_bsp_father(self.cdata))._pass_reference(self)
        if node.cdata == ffi.NULL:
            return None
        return node

    @_assert_cdata_is_not_null
    def get_children(self):
        """Return this instances immediate children, if any.

        Returns:
            Optional[Tuple[BSP, BSP]]:
                Returns a tuple of (left, right) BSP instances.
                Returns None if this BSP has no children.

        .. versionadded:: 2.0
        """
        if self.is_leaf():
            return None
        return (BSP(lib.TCOD_bsp_left(self.cdata))._pass_reference(self),
                BSP(lib.TCOD_bsp_right(self.cdata))._pass_reference(self))

    @_assert_cdata_is_not_null
    def walk(self):
        """Iterate over this BSP's hieracrhy.

        The iterator will include the instance which called it.
        It will traverse its own children and grandchildren, in no particular
        order.

        Returns:
            Iterator[BSP]: An iterator of BSP nodes.

        .. versionadded:: 2.0
        """
        children = self.get_children() or ()
        for child in children:
            for grandchild in child.walk():
                yield grandchild
        yield self

    @_assert_cdata_is_not_null
    def is_leaf(self):
        """Returns True if this node is a leaf.

        Returns:
            bool:
                True if this node is a leaf.
                False when this node has children.

        .. versionadded:: 2.0
        """
        return bool(lib.TCOD_bsp_is_leaf(self.cdata))

    @_assert_cdata_is_not_null
    def contains(self, x, y):
        """Returns True if this node contains these coordinates.

        Args:
            x (int): X position to check.
            y (int): Y position to check.

        Returns:
            bool: True if this node contains these coordinates.
                  Otherwise False.

        .. versionadded:: 2.0
        """
        return bool(lib.TCOD_bsp_contains(self.cdata, x, y))

    @_assert_cdata_is_not_null
    def find_node(self, x, y):
        """Return the deepest node which contains these coordinates.

        Returns:
            Optional[BSP]: BSP object or None.

        .. versionadded:: 2.0
        """
        node = BSP(lib.TCOD_bsp_find_node(self.cdata,
                                           x, y))._pass_reference(self)
        if node.cdata == ffi.NULL:
            node = None
        return node

class HeightMap(_CDataWrapper):
    """libtcod HeightMap instance
    """

    def __init__(self, *args, **kargs):
        super(HeightMap, self).__init__(*args, **kargs)
        if not self.cdata:
            self._init(*args, **kargs)

    def _init(self, width, height):
        self.cdata = ffi.gc(lib.TCOD_heightmap_new(width, height),
                            lib.TCOD_heightmap_delete)


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

    def _init(self, width, height):
        self.cdata = ffi.gc(lib.TCOD_console_new(width, height),
                            lib.TCOD_console_delete)

    def get_width(self):
        """Return the width of this console.

        Returns:
            int: The width of a Console.
        """
        return lib.TCOD_console_get_width(self.cdata)

    def get_height(self):
        """Return the height of this console.

        Returns:
            int: The height of a Console.
        """
        return lib.TCOD_console_get_height(self.cdata)

    def set_default_bg(self, color):
        """Change the default backround color for this console.

        Args:
            color (Union[Tuple[int, int, int], Sequence[int]]):
                An (r, g, b) sequence or Color instance.
        """
        lib.TCOD_console_set_default_background(self.cdata, color)

    def set_default_fg(self, color):
        """Change the default foreground color for this console.

        Args:
            color (Union[Tuple[int, int, int], Sequence[int]]):
                An (r, g, b) sequence or Color instance.
        """
        lib.TCOD_console_set_default_foreground(self.cdata, color)

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

    def set_char_fg(self, x, y, col):
        """Change the foreground color of x,y to col.

        Args:
            x (int): Character x position from the left.
            y (int): Character y position from the top.
            col (Union[Tuple[int, int, int], Sequence[int]]):
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

    def set_default_bg_blend(self, flag):
        """Change the default blend mode for this console.

        Args:
            flag (int): Blend mode to use by default.
        """
        lib.TCOD_console_set_background_flag(self.cdata, flag)

    def get_default_bg_blend(self):
        """Return this consoles current blend mode.
        """
        return lib.TCOD_console_get_background_flag(self.cdata)

    def set_alignment(self, alignment):
        """Change this consoles current alignment mode.

        * tcod.LEFT
        * tcod.CENTER
        * tcod.RIGHT

        Args:
            alignment (int):
        """
        lib.TCOD_console_set_alignment(self.cdata, alignment)

    def get_alignment(self):
        """Return this consoles current alignment mode.
        """
        return lib.TCOD_console_get_alignment(self.cdata)

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

    def get_default_bg(self):
        """Return this consoles default background color."""
        return Color.from_cdata(
            lib.TCOD_console_get_default_background(self.cdata))

    def get_default_fg(self):
        """Return this consoles default foreground color."""
        return Color.from_cdata(
            lib.TCOD_console_get_default_foreground(self.cdata))

    def get_char_bg(self, x, y):
        """Return the background color at the x,y of this console."""
        return Color.from_cdata(
            lib.TCOD_console_get_char_background(self.cdata, x, y))

    def get_char_fg(self, x, y):
        """Return the foreground color at the x,y of this console."""
        return Color.from_cdata(
            lib.TCOD_console_get_char_foreground(self.cdata, x, y))

    def get_char(self, x, y):
        """Return the character at the x,y of this console."""
        return lib.TCOD_console_get_char(self.cdata, x, y)

    def blit(self, x, y, w, h,
             dest, dest_x, dest_y, fg_alpha=1, bg_alpha=1):
        """Blit this console from x,y,w,h to the console dst at xdst,ydst."""
        lib.TCOD_console_blit(self.cdata, x, y, w, h,
                              _cdata(dst), dest_x, dest_y, fg_alpha, bg_alpha)

    def set_key_color(self, color):
        """Set a consoles blit transparent color."""
        lib.TCOD_console_set_key_color(self.cdata, color)

    def fill(self, ch=None, fg=None, bg=None):
        """Fill this console with the given numpy array values.

        Args:
            ch (Optional[:any:`numpy.ndarray`]):
                A numpy integer array with a shape of (width, height)
            fg (Optional[:any:`numpy.ndarray`]):
                A numpy integer array with a shape of (width, height, 3)
            bg (Optional[:any:`numpy.ndarray`]):
                A numpy integer array with a shape of (width, height, 3)
        """
        import numpy
        if ch:
            ch = numpy.ascontiguousarray(ch, dtype=numpy.intc)
            ch_array = ffi.cast('int *', ch.ctypes.data)
            lib.TCOD_console_fill_char(self.cdata, ch_array)
        if fg:
            r = numpy.ascontiguousarray(fg[:,:,0], dtype=numpy.intc)
            g = numpy.ascontiguousarray(fg[:,:,1], dtype=numpy.intc)
            b = numpy.ascontiguousarray(fg[:,:,2], dtype=numpy.intc)
            cr = ffi.cast('int *', r.ctypes.data)
            cg = ffi.cast('int *', g.ctypes.data)
            cb = ffi.cast('int *', b.ctypes.data)
            lib.TCOD_console_fill_foreground(self.cdata, cr, cg, cb)
        if bg:
            r = numpy.ascontiguousarray(bg[:,:,0], dtype=numpy.intc)
            g = numpy.ascontiguousarray(bg[:,:,1], dtype=numpy.intc)
            b = numpy.ascontiguousarray(bg[:,:,2], dtype=numpy.intc)
            cr = ffi.cast('int *', r.ctypes.data)
            cg = ffi.cast('int *', g.ctypes.data)
            cb = ffi.cast('int *', b.ctypes.data)
            lib.TCOD_console_fill_background(self.cdata, cr, cg, cb)


class Image(_CDataWrapper):
    """
    Args:
        width (int): Width of the new Image.
        height (int): Height of the new Image.

    .. versionadded:: 2.0
    """
    def __init__(self, *args, **kargs):
        super(Console, self).__init__(*args, **kargs)
        if not self.cdata:
            self._init(*args, **kargs)

    def _init(self, width, height):
        self.cdata = ffi.gc(lib.TCOD_image_new(width, height),
                            lib.TCOD_image_delete)

def clipboard_set(string):
    """Set the clipboard contents to string.

    Args:
        string (AnyStr): The string to set the clipboard to.

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

@ffi.def_extern()
def _pycall_bsp_callback(node, handle):
    """static bool _pycall_bsp_callback(TCOD_bsp_t *node, void *userData);"""
    func, userData, propagate = ffi.from_handle(handle)
    try:
        return func(BSP(node), userData)
    except BaseException:
        propagate(*_sys.exc_info())
        return False

__all__ = [_name for _name in list(globals()) if _name[0] != '_']
