"""
libtcod works with a special 'root' console.  You create this console using
the :any:`tcod.console_init_root` function.  Usually after setting the font
with :any:`console_set_custom_font` first.

Example::

    # Make sure 'arial10x10.png' is in the same directory as this script.
    import time

    import tcod

    # Setup the font.
    tcod.console_set_custom_font(
        'arial10x10.png',
        tcod.FONT_LAYOUT_TCOD,
        )
    # Initialize the root console in a context.
    with tcod.console_init_root(80, 60, 'title') as root_console:
        root_console.print_(x=0, y=0, string='Hello World!')
        tcod.console_flush() # Show the console.
        time.sleep(3) # Wait 3 seconds.
    # The window is closed here, after the above context exits.
"""

from __future__ import absolute_import

import sys

import warnings

import numpy as np

import tcod.libtcod
from tcod.libtcod import ffi, lib
import tcod._internal

if sys.version_info[0] == 2: # Python 2
    def _fmt(string):
        if not isinstance(string, unicode):
            string = string.decode('latin-1')
        return string.replace(u'%', u'%%')
else:
    def _fmt(string):
        """Return a string that escapes 'C printf' side effects."""
        return string.replace('%', '%%')

_root_console = None

class Console(object):
    """A console object containing a grid of characters with
    foreground/background colors.

    .. versionchanged:: 4.3
        Added `order` parameter.

    Args:
        width (int): Width of the new Console.
        height (int): Height of the new Console.
        order (str): Which numpy memory order to use.

    Attributes:
        console_c (CData): A cffi pointer to a TCOD_console_t object.
    """

    def __init__(self, width, height, order='C'):
        self._key_color = None
        self._ch = np.zeros((height, width), dtype=np.intc)
        self._fg = np.zeros((height, width), dtype='(3,)u1')
        self._bg = np.zeros((height, width), dtype='(3,)u1')
        self._order = tcod._internal.verify_order(order)

        # libtcod uses the root console for defaults.
        bkgnd_flag = alignment = 0
        if lib.TCOD_ctx.root != ffi.NULL:
            bkgnd_flag = lib.TCOD_ctx.root.bkgnd_flag
            alignment = lib.TCOD_ctx.root.alignment

        self._console_data = self.console_c = ffi.new(
            'struct TCOD_Console*',
            {
            'w': width, 'h': height,
            'ch_array': ffi.cast('int*', self._ch.ctypes.data),
            'fg_array': ffi.cast('TCOD_color_t*', self._fg.ctypes.data),
            'bg_array': ffi.cast('TCOD_color_t*', self._bg.ctypes.data),
            'bkgnd_flag': bkgnd_flag,
            'alignment': alignment,
            'fore': (255, 255, 255),
            'back': (0, 0, 0),
            },
        )

    @classmethod
    def _from_cdata(cls, cdata, order='C'):
        if isinstance(cdata, cls):
            return cdata
        self = object.__new__(cls)
        self.console_c = cdata
        self._init_setup_console_data(order)
        return self

    @classmethod
    def _get_root(cls, order=None):
        """Return a root console singleton with valid buffers.

        This function will also update an already active root console.
        """
        global _root_console
        if _root_console is None:
            _root_console = object.__new__(cls)
        self = _root_console
        if order is not None:
            self._order = order
        self.console_c = ffi.NULL
        self._init_setup_console_data(self._order)
        return self

    def _init_setup_console_data(self, order='C'):
        """Setup numpy arrays over libtcod data buffers."""
        global _root_console
        self._key_color = None
        if self.console_c == ffi.NULL:
            _root_console = self
            self._console_data = lib.TCOD_ctx.root
        else:
            self._console_data = ffi.cast('struct TCOD_Console*', self.console_c)

        def unpack_color(color_data):
            """return a (height, width, 3) shaped array from an image struct"""
            color_buffer = ffi.buffer(color_data[0:self.width * self.height])
            array = np.frombuffer(color_buffer, np.uint8)
            return array.reshape((self.height, self.width, 3))

        self._fg = unpack_color(self._console_data.fg_array)
        self._bg = unpack_color(self._console_data.bg_array)

        buf = self._console_data.ch_array
        buf = ffi.buffer(buf[0:self.width * self.height])
        self._ch = np.frombuffer(buf, np.intc).reshape((self.height,
                                                        self.width))

        self._order = tcod._internal.verify_order(order)

    @property
    def width(self):
        """int: The width of this Console. (read-only)"""
        return lib.TCOD_console_get_width(self.console_c)

    @property
    def height(self):
        """int: The height of this Console. (read-only)"""
        return lib.TCOD_console_get_height(self.console_c)

    @property
    def bg(self):
        """A uint8 array with the shape (height, width, 3).

        You can change the consoles background colors by using this array.

        Index this array with ``console.bg[i, j, channel] # order='C'`` or
        ``console.bg[x, y, channel] # order='F'``.

        """
        return self._bg.transpose(1, 0, 2) if self._order == 'F' else self._bg

    @property
    def fg(self):
        """A uint8 array with the shape (height, width, 3).

        You can change the consoles foreground colors by using this array.

        Index this array with ``console.fg[i, j, channel] # order='C'`` or
        ``console.fg[x, y, channel] # order='F'``.
        """
        return self._fg.transpose(1, 0, 2) if self._order == 'F' else self._fg

    @property
    def ch(self):
        """An integer array with the shape (height, width).

        You can change the consoles character codes by using this array.

        Index this array with ``console.ch[i, j] # order='C'`` or
        ``console.ch[x, y] # order='F'``.
        """
        return self._ch.T if self._order == 'F' else self._ch

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
    def default_bg_blend(self):
        """int: The default blending mode."""
        return self._console_data.bkgnd_flag
    @default_bg_blend.setter
    def default_bg_blend(self, value):
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
        lib.TCOD_console_clear(self.console_c)

    def put_char(self, x, y, ch, bg_blend=tcod.libtcod.BKGND_DEFAULT):
        """Draw the character c at x,y using the default colors and a blend mode.

        Args:
            x (int): The x coordinate from the left.
            y (int): The y coordinate from the top.
            ch (int): Character code to draw.  Must be in integer form.
            bg_blend (int): Blending mode to use, defaults to BKGND_DEFAULT.
        """
        lib.TCOD_console_put_char(self.console_c, x, y, ch, bg_blend)

    def print_(self, x, y, string, bg_blend=tcod.libtcod.BKGND_DEFAULT,
               alignment=None):
        """Print a color formatted string on a console.

        Args:
            x (int): The x coordinate from the left.
            y (int): The y coordinate from the top.
            string (Text): A Unicode string optionaly using color codes.
            bg_blend (int): Blending mode to use, defaults to BKGND_DEFAULT.
            alignment (Optinal[int]): Text alignment.
        """
        alignment = self.default_alignment if alignment is None else alignment

        lib.TCOD_console_print_ex_utf(self.console_c, x, y,
                                      bg_blend, alignment, _fmt(string))

    def print_rect(self, x, y, width, height, string,
                   bg_blend=tcod.libtcod.BKGND_DEFAULT, alignment=None):
        """Print a string constrained to a rectangle.

        If h > 0 and the bottom of the rectangle is reached,
        the string is truncated. If h = 0,
        the string is only truncated if it reaches the bottom of the console.

        Args:
            x (int): The x coordinate from the left.
            y (int): The y coordinate from the top.
            width (int): Maximum width to render the text.
            height (int): Maximum lines to render the text.
            string (Text): A Unicode string.
            bg_blend (int): Background blending flag.
            alignment (Optional[int]): Alignment flag.

        Returns:
            int: The number of lines of text once word-wrapped.
        """
        alignment = self.default_alignment if alignment is None else alignment
        return lib.TCOD_console_print_rect_ex_utf(self.console_c,
            x, y, width, height, bg_blend, alignment, _fmt(string))

    def get_height_rect(self, x, y, width, height, string):
        """Return the height of this text word-wrapped into this rectangle.

        Args:
            x (int): The x coordinate from the left.
            y (int): The y coordinate from the top.
            width (int): Maximum width to render the text.
            height (int): Maximum lines to render the text.
            string (Text): A Unicode string.

        Returns:
            int: The number of lines of text once word-wrapped.
        """
        return lib.TCOD_console_get_height_rect_utf(
            self.console_c, x, y, width, height, _fmt(string))

    def rect(self, x, y, width, height, clear,
             bg_blend=tcod.libtcod.BKGND_DEFAULT):
        """Draw a the background color on a rect optionally clearing the text.

        If clr is True the affected tiles are changed to space character.

        Args:
            x (int): The x coordinate from the left.
            y (int): The y coordinate from the top.
            width (int): Maximum width to render the text.
            height (int): Maximum lines to render the text.
            clear (bool): If True all text in the affected area will be
                          removed.
            bg_blend (int): Background blending flag.
        """
        lib.TCOD_console_rect(self.console_c, x, y, width, height, clear,
                              bg_blend)

    def hline(self, x, y, width, bg_blend=tcod.libtcod.BKGND_DEFAULT):
        """Draw a horizontal line on the console.

        This always uses the character 196, the horizontal line character.

        Args:
            x (int): The x coordinate from the left.
            y (int): The y coordinate from the top.
            width (int): The horozontal length of this line.
            bg_blend (int): The background blending flag.
        """
        lib.TCOD_console_hline(self.console_c, x, y, width, bg_blend)

    def vline(self, x, y, height, bg_blend=tcod.libtcod.BKGND_DEFAULT):
        """Draw a vertical line on the console.

        This always uses the character 179, the vertical line character.

        Args:
            x (int): The x coordinate from the left.
            y (int): The y coordinate from the top.
            height (int): The horozontal length of this line.
            bg_blend (int): The background blending flag.
        """
        lib.TCOD_console_vline(self.console_c, x, y, height, bg_blend)

    def print_frame(self, x: int, y: int, width: int, height: int,
                    string: str='', clear: bool=True,
                    bg_blend: int=tcod.libtcod.BKGND_DEFAULT):
        """Draw a framed rectangle with optinal text.

        This uses the default background color and blend mode to fill the
        rectangle and the default foreground to draw the outline.

        `string` will be printed on the inside of the rectangle, word-wrapped.
        If `string` is empty then no title will be drawn.

        Args:
            x (int): The x coordinate from the left.
            y (int): The y coordinate from the top.
            width (int): The width if the frame.
            height (int): The height of the frame.
            string (str): A Unicode string to print.
            clear (bool): If True all text in the affected area will be
                          removed.
            bg_blend (int): The background blending flag.

        Note:
            This method does not support Unicode outside of the 0-255 range.
        """
        if string:
            string = string.encode('latin-1')
        else:
            string = ffi.NULL
        lib.TCOD_console_print_frame(self.console_c, x, y, width, height,
                                     clear, bg_blend, string)

    def blit(self, dest, dest_x=0, dest_y=0,
             src_x=0, src_y=0, width=0, height=0,
             fg_alpha=1.0, bg_alpha=1.0, key_color=None):
        """Blit from this console onto the ``dest`` console.

        Args:
            dest (Console): The destintaion console to blit onto.
            dest_x (int): Leftmost coordinate of the destintaion console.
            dest_y (int): Topmost coordinate of the destintaion console.
            src_x (int): X coordinate from this console to blit, from the left.
            src_y (int): Y coordinate from this console to blit, from the top.
            width (int): The width of the region to blit.

                If this is 0 the maximum possible width will be used.
            height (int): The height of the region to blit.

                If this is 0 the maximum possible height will be used.
            fg_alpha (float): Foreground color alpha vaule.
            bg_alpha (float): Background color alpha vaule.
            key_color (Optional[Tuple[int, int, int]]):
                None, or a (red, green, blue) tuple with values of 0-255.

        .. versionchanged:: 4.0
            Parameters were rearraged and made optional.

            Previously they were:
            `(x, y, width, height, dest, dest_x, dest_y, *)`
        """
        # The old syntax is easy to detect and correct.
        if hasattr(src_y, 'console_c'):
            src_x, src_y, width, height, dest, dest_x, dest_y = \
                dest, dest_x, dest_y, src_x, src_y, width, height
            warnings.warn(
                "Parameter names have been moved around, see documentation.",
                DeprecationWarning,
                stacklevel=2,
                )

        if key_color or self._key_color:
            key_color = ffi.new('TCOD_color_t*', key_color)
            lib.TCOD_console_blit_key_color(
            self.console_c, src_x, src_y, width, height,
            dest.console_c, dest_x, dest_y, fg_alpha, bg_alpha, key_color
            )
        else:
            lib.TCOD_console_blit(
                self.console_c, src_x, src_y, width, height,
                dest.console_c, dest_x, dest_y, fg_alpha, bg_alpha
            )

    def set_key_color(self, color):
        """Set a consoles blit transparent color.

        Args:
            color (Tuple[int, int, int]):
        """
        self._key_color = color

    def __enter__(self):
        """Returns this console in a managed context.

        When the root console is used as a context, the graphical window will
        close once the context is left as if :any:`tcod.console_delete` was
        called on it.

        This is useful for some Python IDE's like IDLE, where the window would
        not be closed on its own otherwise.
        """
        if self.console_c != ffi.NULL:
            raise NotImplementedError('Only the root console has a context.')
        return self

    def __exit__(self, *args):
        """Closes the graphical window on exit.

        Some tcod functions may have undefined behavior after this point.
        """
        lib.TCOD_console_delete(self.console_c)

    def __bool__(self):
        """Returns False if this is the root console.

        This mimics libtcodpy behavior.
        """
        return self.console_c != ffi.NULL

    __nonzero__ = __bool__

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['console_c']
        state['_console_data'] = {
            'w': self.width, 'h': self.height,
            'bkgnd_flag': self.default_bg_blend,
            'alignment': self.default_alignment,
            'fore': self.default_fg,
            'back': self.default_bg,
        }
        if self.console_c == ffi.NULL:
            state['_ch'] = np.copy(self._ch)
            state['_fg'] = np.copy(self._fg)
            state['_bg'] = np.copy(self._bg)
        return state

    def __setstate__(self, state):
        self._key_color = None
        self.__dict__.update(state)
        self._console_data.update(
            {
            'ch_array': ffi.cast('int*', self._ch.ctypes.data),
            'fg_array': ffi.cast('TCOD_color_t*', self._fg.ctypes.data),
            'bg_array': ffi.cast('TCOD_color_t*', self._bg.ctypes.data),
            }
        )
        self._console_data = self.console_c = ffi.new(
            'struct TCOD_Console*', self._console_data)
