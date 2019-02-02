"""
    .. deprecated:: 8.4
        This module has been deprecated.

    Getting Started
    ===============
      Once the library is imported you can load the font you want to use with
      :any:`tdl.set_font`.
      This is optional and when skipped will use a decent default font.

      After that you call :any:`tdl.init` to set the size of the window and
      get the root console in return.
      This console is the canvas to what will appear on the screen.

    Indexing Consoles
    =================
      For most methods taking a position you can use Python-style negative
      indexes to refer to the opposite side of a console with (-1, -1)
      starting at the bottom right.
      You can also check if a point is part of a console using containment
      logic i.e. ((x, y) in console).

      You may also iterate over a console using a for statement.  This returns
      every x,y coordinate available to draw on but it will be extremely slow
      to actually operate on every coordinate individualy.
      Try to minimize draws by using an offscreen :any:`Console`, only drawing
      what needs to be updated, and using :any:`Console.blit`.

    Drawing and Colors
    ==================

      Once you have the root console from :any:`tdl.init` you can start drawing
      on it using a method such as :any:`Console.draw_char`.
      When using this method you can have the char parameter be an integer or a
      single character string.

      The fg and bg parameters expect a variety of types.
      The parameters default to Ellipsis which will tell the function to
      use the colors previously set by the :any:`Console.set_colors` method.
      The colors set by :any`Console.set_colors` are per each
      :any:`Console`/:any:`Window` and default to white on black.
      You can use a 3-item list/tuple of [red, green, blue] with integers in
      the 0-255 range with [0, 0, 0] being black and [255, 255, 255] being
      white.
      You can even use a single integer of 0xRRGGBB if you like.

      Using None in the place of any of the three parameters (char, fg, bg)
      will tell the function to not overwrite that color or character.

      After the drawing functions are called a call to :any:`tdl.flush` will
      update the screen.
"""

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import sys as _sys
import os as _os

import array as _array
import weakref as _weakref
import itertools as _itertools
import textwrap as _textwrap
import struct as _struct
import re as _re
import warnings as _warnings

from tcod import ffi as _ffi
from tcod import lib as _lib

import tdl.event
import tdl.noise
import tdl.map
import tdl.style as _style

from tcod import __version__

_warnings.warn(
    "The tdl module has been deprecated.",
    DeprecationWarning,
    stacklevel=2,
)

_IS_PYTHON3 = (_sys.version_info[0] == 3)

if _IS_PYTHON3: # some type lists to use with isinstance
    _INTTYPES = (int,)
    _NUMTYPES = (int, float)
    _STRTYPES = (str, bytes)
else:
    _INTTYPES = (int, long)
    _NUMTYPES = (int, long, float)
    _STRTYPES = (str, unicode)

def _encodeString(string): # still used for filepaths, and that's about it
    "changes string into bytes if running in python 3, for sending to ctypes"
    if isinstance(string, _STRTYPES):
        return string.encode()
    return string

def _format_char(char):
    """Prepares a single character for passing to ctypes calls, needs to return
    an integer but can also pass None which will keep the current character
    instead of overwriting it.

    This is called often and needs to be optimized whenever possible.
    """
    if char is None:
        return -1
    if isinstance(char, _STRTYPES) and len(char) == 1:
        return ord(char)
    try:
        return int(char) # allow all int-like objects
    except:
        raise TypeError('char single character string, integer, or None\nReceived: ' + repr(char))

_utf32_codec = {'little': 'utf-32le', 'big': 'utf-32le'}[_sys.byteorder]

def _format_str(string):
    """Attempt fast string handing by decoding directly into an array."""
    if isinstance(string, _STRTYPES):
        if _IS_PYTHON3:
            array = _array.array('I')
            array.frombytes(string.encode(_utf32_codec))
        else: # Python 2
            if isinstance(string, unicode):
                array = _array.array(b'I')
                array.fromstring(string.encode(_utf32_codec))
            else:
                array = _array.array(b'B')
                array.fromstring(string)
        return array
    return string

_fontinitialized = False
_rootinitialized = False
_rootConsoleRef = None

_put_char_ex = _lib.TDL_console_put_char_ex

# python 2 to 3 workaround
if _sys.version_info[0] == 2:
    int_types = (int, long)
else:
    int_types = int


def _format_color(color, default=Ellipsis):
        if color is Ellipsis:
            return default
        if color is None:
            return -1
        if isinstance(color, (tuple, list)) and len(color) == 3:
            return (color[0] << 16) + (color[1] << 8) + color[2]
        try:
            return int(color) # allow all int-like objects
        except:
            raise TypeError('fg and bg must be a 3 item tuple, integer, Ellipsis, or None\nReceived: ' + repr(color))

def _to_tcod_color(color):
    return _ffi.new('TCOD_color_t *', (color >> 16 & 0xff,
                                       color >> 8 & 0xff,
                                       color & 0xff))

def _getImageSize(filename):
    """Try to get the width and height of a bmp of png image file"""
    result = None
    file = open(filename, 'rb')
    if file.read(8) == b'\x89PNG\r\n\x1a\n': # PNG
        while 1:
            length, = _struct.unpack('>i', file.read(4))
            chunkID = file.read(4)
            if chunkID == '': # EOF
                break
            if chunkID == b'IHDR':
                # return width, height
                result = _struct.unpack('>ii', file.read(8))
                break
            file.seek(4 + length, 1)
        file.close()
        return result
    file.seek(0)
    if file.read(8) == b'BM': # Bitmap
        file.seek(18, 0) # skip to size data
        result = _struct.unpack('<ii', file.read(8))
    file.close()
    return result # (width, height), or None

class TDLError(Exception):
    """
    The catch all for most TDL specific errors.
    """

class _BaseConsole(object):
    """You will not use this class directly.  The methods in this class are
    inherited by the :any:`tdl.Console` and :any:`tdl.Window` subclasses.
    """
    __slots__ = ('width', 'height', 'console', '_cursor', '_fg',
                 '_bg', '_blend', '__weakref__', '__dict__')

    def __init__(self):
        self._cursor = (0, 0)
        self._scrollMode = 'error'
        self._fg = _format_color((255, 255, 255))
        self._bg = _format_color((0, 0, 0))
        self._blend = _lib.TCOD_BKGND_SET

    def _normalizePoint(self, x, y):
        """Check if a point is in bounds and make minor adjustments.

        Respects Pythons negative indexes.  -1 starts at the bottom right.
        Replaces the _drawable function
        """
        # cast to int, always faster than type checking
        x = int(x)
        y = int(y)

        assert (-self.width <= x < self.width) and \
               (-self.height <= y < self.height), \
               ('(%i, %i) is an invalid postition on %s' % (x, y, self))

        # handle negative indexes
        return (x % self.width, y % self.height)

    def _normalizeRect(self, x, y, width, height):
        """Check if the rectangle is in bounds and make minor adjustments.
        raise AssertionError's for any problems
        """
        x, y = self._normalizePoint(x, y) # inherit _normalizePoint logic

        assert width is None or isinstance(width, _INTTYPES), 'width must be an integer or None, got %s' % repr(width)
        assert height is None or isinstance(height, _INTTYPES), 'height must be an integer or None, got %s' % repr(height)

        # if width or height are None then extend them to the edge
        if width is None:
            width = self.width - x
        elif width < 0: # handle negative numbers
            width += self.width
            width = max(0, width) # a 'too big' negative is clamped zero
        if height is None:
            height = self.height - y
            height = max(0, height)
        elif height < 0:
            height += self.height

        # reduce rect size to bounds
        width = min(width, self.width - x)
        height = min(height, self.height - y)

        return x, y, width, height

    def _normalizeCursor(self, x, y):
        """return the normalized the cursor position."""
        width, height = self.get_size()
        assert width != 0 and height != 0, 'can not print on a console with a width or height of zero'
        while x >= width:
            x -= width
            y += 1
        while y >= height:
            if self._scrollMode == 'scroll':
                y -= 1
                self.scroll(0, -1)
            elif self._scrollMode == 'error':
                # reset the cursor on error
                self._cursor = (0, 0)
                raise TDLError('Cursor has reached the end of the console')
        return (x, y)

    def set_mode(self, mode):
        """Configure how this console will react to the cursor writing past the
        end if the console.

        This is for methods that use the virtual cursor, such as
        :any:`print_str`.

        Args:
            mode (Text): The mode to set.

            Possible settings are:

           - 'error' - A TDLError will be raised once the cursor
             reaches the end of the console.  Everything up until
             the error will still be drawn.
             This is the default setting.
           - 'scroll' - The console will scroll up as stuff is
             written to the end.
             You can restrict the region with :any:`tdl.Window` when
             doing this.

        ..seealso:: :any:`write`, :any:`print_str`
        """
        MODES = ['error', 'scroll']
        if mode.lower() not in MODES:
            raise TDLError('mode must be one of %s, got %s' % (MODES, repr(mode)))
        self._scrollMode = mode.lower()

    def set_colors(self, fg=None, bg=None):
        """Sets the colors to be used with the L{print_str} and draw_* methods.

        Values of None will only leave the current values unchanged.

        Args:
            fg (Optional[Union[Tuple[int, int, int], int, Ellipsis]])
            bg (Optional[Union[Tuple[int, int, int], int, Ellipsis]])
        .. seealso:: :any:`move`, :any:`print_str`
        """
        if fg is not None:
            self._fg = _format_color(fg, self._fg)
        if bg is not None:
            self._bg = _format_color(bg, self._bg)

    def print_str(self, string):
        """Print a string at the virtual cursor.

        Handles special characters such as '\\n' and '\\r'.
        Printing past the bottom of the console will scroll everything upwards
        if :any:`set_mode` is set to 'scroll'.

        Colors can be set with :any:`set_colors` and the virtual cursor can
        be moved with :any:`move`.

        Args:
            string (Text): The text to print.

        .. seealso:: :any:`draw_str`, :any:`move`, :any:`set_colors`,
                     :any:`set_mode`, :any:`write`, :any:`Window`
        """
        x, y = self._cursor
        for char in string:
            if char == '\n': # line break
                x = 0
                y += 1
                continue
            if char == '\r': # return
                x = 0
                continue
            x, y = self._normalizeCursor(x, y)
            self.draw_char(x, y, char, self._fg, self._bg)
            x += 1
        self._cursor = (x, y)

    def write(self, string):
        """This method mimics basic file-like behaviour.

        Because of this method you can replace sys.stdout or sys.stderr with
        a :any:`Console` or :any:`Window` instance.

        This is a convoluted process and behaviour seen now can be excepted to
        change on later versions.

        Args:
            string (Text): The text to write out.

        .. seealso:: :any:`set_colors`, :any:`set_mode`, :any:`Window`
        """
        # some 'basic' line buffer stuff.
        # there must be an easier way to do this.  The textwrap module didn't
        # help much.
        x, y = self._normalizeCursor(*self._cursor)
        width, height = self.get_size()
        wrapper = _textwrap.TextWrapper(initial_indent=(' '*x), width=width)
        writeLines = []
        for line in string.split('\n'):
            if line:
                writeLines += wrapper.wrap(line)
                wrapper.initial_indent = ''
            else:
                writeLines.append([])

        for line in writeLines:
            x, y = self._normalizeCursor(x, y)
            self.draw_str(x, y, line[x:], self._fg, self._bg)
            y += 1
            x = 0
        y -= 1
        self._cursor = (x, y)

    def draw_char(self, x, y, char, fg=Ellipsis, bg=Ellipsis):
        """Draws a single character.

        Args:
            x (int): x-coordinate to draw on.
            y (int): y-coordinate to draw on.
            char (Optional[Union[int, Text]]): An integer, single character
                string, or None.

                You can set the char parameter as None if you only want to change
                the colors of the tile.
            fg (Optional[Union[Tuple[int, int, int], int, Ellipsis]])
            bg (Optional[Union[Tuple[int, int, int], int, Ellipsis]])

        Raises: AssertionError: Having x or y values that can't be placed
            inside of the console will raise an AssertionError.
            You can use always use ((x, y) in console) to
            check if a tile is drawable.

        .. seealso:: :any:`get_char`
        """
        #x, y = self._normalizePoint(x, y)
        _put_char_ex(self.console_c, x, y, _format_char(char),
                     _format_color(fg, self._fg), _format_color(bg, self._bg), 1)

    def draw_str(self, x, y, string, fg=Ellipsis, bg=Ellipsis):
        """Draws a string starting at x and y.

        A string that goes past the right side will wrap around.  A string
        wrapping to below the console will raise :any:`tdl.TDLError` but will
        still be written out.
        This means you can safely ignore the errors with a
        try..except block if you're fine with partially written strings.

        \\r and \\n are drawn on the console as normal character tiles.  No
        special encoding is done and any string will translate to the character
        table as is.

        For a string drawing operation that respects special characters see
        :any:`print_str`.

        Args:
            x (int): x-coordinate to start at.
            y (int): y-coordinate to start at.
            string (Union[Text, Iterable[int]]): A string or an iterable of
                numbers.

                Special characters are ignored and rendered as any other
                character.
            fg (Optional[Union[Tuple[int, int, int], int, Ellipsis]])
            bg (Optional[Union[Tuple[int, int, int], int, Ellipsis]])

        Raises:
            AssertionError: Having x or y values that can't be placed inside
                            of the console will raise an AssertionError.

                You can use always use ``((x, y) in console)`` to check if
                a tile is drawable.

        .. seealso:: :any:`print_str`
        """

        x, y = self._normalizePoint(x, y)
        fg, bg = _format_color(fg, self._fg), _format_color(bg, self._bg)
        width, height = self.get_size()
        def _drawStrGen(x=x, y=y, string=string, width=width, height=height):
            """Generator for draw_str

            Iterates over ((x, y), ch) data for _set_batch, raising an
            error if the end of the console is reached.
            """
            for char in _format_str(string):
                if y == height:
                    raise TDLError('End of console reached.')
                yield((x, y), char)
                x += 1 # advance cursor
                if x == width: # line break
                    x = 0
                    y += 1
        self._set_batch(_drawStrGen(), fg, bg)

    def draw_rect(self, x, y, width, height, string, fg=Ellipsis, bg=Ellipsis):
        """Draws a rectangle starting from x and y and extending to width and height.

        If width or height are None then it will extend to the edge of the console.

        Args:
            x (int): x-coordinate for the top side of the rect.
            y (int): y-coordinate for the left side of the rect.
            width (Optional[int]): The width of the rectangle.

                Can be None to extend to the bottom right of the
                console or can be a negative number to be sized reltive
                to the total size of the console.
            height (Optional[int]): The height of the rectangle.
            string (Optional[Union[Text, int]]): An integer, single character
                string, or None.

                You can set the string parameter as None if you only want
                to change the colors of an area.
            fg (Optional[Union[Tuple[int, int, int], int, Ellipsis]])
            bg (Optional[Union[Tuple[int, int, int], int, Ellipsis]])

        Raises:
            AssertionError: Having x or y values that can't be placed inside
                            of the console will raise an AssertionError.

            You can use always use ``((x, y) in console)`` to check if a tile
            is drawable.
        .. seealso:: :any:`clear`, :any:`draw_frame`
        """
        x, y, width, height = self._normalizeRect(x, y, width, height)
        fg, bg = _format_color(fg, self._fg), _format_color(bg, self._bg)
        char = _format_char(string)
        # use itertools to make an x,y grid
        # using ctypes here reduces type converstions later
        #grid = _itertools.product((_ctypes.c_int(x) for x in range(x, x + width)),
        #                          (_ctypes.c_int(y) for y in range(y, y + height)))
        grid = _itertools.product((x for x in range(x, x + width)),
                                  (y for y in range(y, y + height)))
        # zip the single character in a batch variable
        batch = zip(grid, _itertools.repeat(char, width * height))
        self._set_batch(batch, fg, bg, nullChar=(char is None))

    def draw_frame(self, x, y, width, height, string, fg=Ellipsis, bg=Ellipsis):
        """Similar to L{draw_rect} but only draws the outline of the rectangle.

        `width or `height` can be None to extend to the bottom right of the
        console or can be a negative number to be sized reltive
        to the total size of the console.

        Args:
            x (int): The x-coordinate to start on.
            y (int): The y-coordinate to start on.
            width (Optional[int]): Width of the rectangle.
            height (Optional[int]): Height of the rectangle.
            string (Optional[Union[Text, int]]): An integer, single character
                string, or None.

                You can set this parameter as None if you only want
                to change the colors of an area.
            fg (Optional[Union[Tuple[int, int, int], int, Ellipsis]])
            bg (Optional[Union[Tuple[int, int, int], int, Ellipsis]])

        Raises:
            AssertionError: Having x or y values that can't be placed inside
                             of the console will raise an AssertionError.

             You can use always use ``((x, y) in console)`` to check if a tile
             is drawable.
        .. seealso:: :any:`draw_rect`, :any:`Window`
        """
        x, y, width, height = self._normalizeRect(x, y, width, height)
        fg, bg = _format_color(fg, self._fg), _format_color(bg, self._bg)
        char = _format_char(string)
        if width == 1 or height == 1: # it's just a single width line here
            return self.draw_rect(x, y, width, height, char, fg, bg)

        # draw sides of frame with draw_rect
        self.draw_rect(x, y, 1, height, char, fg, bg)
        self.draw_rect(x, y, width, 1, char, fg, bg)
        self.draw_rect(x + width - 1, y, 1, height, char, fg, bg)
        self.draw_rect(x, y + height - 1, width, 1, char, fg, bg)

    def blit(self, source, x=0, y=0, width=None, height=None, srcX=0, srcY=0,
             fg_alpha=1.0, bg_alpha=1.0):
        """Blit another console or Window onto the current console.

        By default it blits the entire source to the topleft corner.

        Args:
            source (Union[tdl.Console, tdl.Window]): The blitting source.
                A console can blit to itself without any problems.
            x (int): x-coordinate of this console to blit on.
            y (int): y-coordinate of this console to blit on.
            width (Optional[int]): Width of the rectangle.

                Can be None to extend as far as possible to the
                bottom right corner of the blit area or can be a negative
                number to be sized reltive to the total size of the
                B{destination} console.
            height (Optional[int]): Height of the rectangle.
            srcX (int):  x-coordinate of the source region to blit.
            srcY (int):  y-coordinate of the source region to blit.
            fg_alpha (float): The foreground alpha.
        """
        assert isinstance(source, (Console, Window)), "source muse be a Window or Console instance"

        # handle negative indexes and rects
        # negative width and height will be set realtive to the destination
        # and will also be clamped to the smallest Console
        x, y, width, height = self._normalizeRect(x, y, width, height)
        srcX, srcY, width, height = source._normalizeRect(srcX, srcY, width, height)

        # translate source and self if any of them are Window instances
        srcX, srcY = source._translate(srcX, srcY)
        source = source.console
        x, y = self._translate(x, y)
        self = self.console

        if self == source:
            # if we are the same console then we need a third console to hold
            # onto the data, otherwise it tries to copy into itself and
            # starts destroying everything
            tmp = Console(width, height)
            _lib.TCOD_console_blit(source.console_c,
                                   srcX, srcY, width, height,
                                   tmp.console_c, 0, 0, fg_alpha, bg_alpha)
            _lib.TCOD_console_blit(tmp.console_c, 0, 0, width, height,
                                   self.console_c, x, y, fg_alpha, bg_alpha)
        else:
            _lib.TCOD_console_blit(source.console_c,
                                   srcX, srcY, width, height,
                                   self.console_c, x, y, fg_alpha, bg_alpha)

    def get_cursor(self):
        """Return the virtual cursor position.

        The cursor can be moved with the :any:`move` method.

        Returns:
            Tuple[int, int]: The (x, y) coordinate of where :any:`print_str`
                will continue from.

        .. seealso:: :any:move`
        """
        x, y = self._cursor
        width, height = self.parent.get_size()
        while x >= width:
            x -= width
            y += 1
        if y >= height and self.scrollMode == 'scroll':
            y = height - 1
        return x, y

    def get_size(self):
        """Return the size of the console as (width, height)

        Returns:
            Tuple[int, int]: A (width, height) tuple.
        """
        return self.width, self.height

    def __iter__(self):
        """Return an iterator with every possible (x, y) value for this console.

        It goes without saying that working on the console this way is a
        slow process, especially for Python, and should be minimized.

        Returns:
            Iterator[Tuple[int, int]]: An ((x, y), ...) iterator.
        """
        return _itertools.product(range(self.width), range(self.height))

    def move(self, x, y):
        """Move the virtual cursor.

        Args:
            x (int): x-coordinate to place the cursor.
            y (int): y-coordinate to place the cursor.

        .. seealso:: :any:`get_cursor`, :any:`print_str`, :any:`write`
        """
        self._cursor = self._normalizePoint(x, y)

    def scroll(self, x, y):
        """Scroll the contents of the console in the direction of x,y.

        Uncovered areas will be cleared to the default background color.
        Does not move the virutal cursor.

        Args:
            x (int): Distance to scroll along the x-axis.
            y (int): Distance to scroll along the y-axis.

        Returns:
            Iterator[Tuple[int, int]]: An iterator over the (x, y) coordinates
            of any tile uncovered after scrolling.

        .. seealso:: :any:`set_colors`
        """
        assert isinstance(x, _INTTYPES), "x must be an integer, got %s" % repr(x)
        assert isinstance(y, _INTTYPES), "y must be an integer, got %s" % repr(x)
        def getSlide(x, length):
            """get the parameters needed to scroll the console in the given
            direction with x
            returns (x, length, srcx)
            """
            if x > 0:
                srcx = 0
                length -= x
            elif x < 0:
                srcx = abs(x)
                x = 0
                length -= srcx
            else:
                srcx = 0
            return x, length, srcx
        def getCover(x, length):
            """return the (x, width) ranges of what is covered and uncovered"""
            cover = (0, length) # everything covered
            uncover = None  # nothing uncovered
            if x > 0: # left side uncovered
                cover = (x, length - x)
                uncover = (0, x)
            elif x < 0: # right side uncovered
                x = abs(x)
                cover = (0, length - x)
                uncover = (length - x, x)
            return cover, uncover

        width, height = self.get_size()
        if abs(x) >= width or abs(y) >= height:
            return self.clear() # just clear the console normally

        # get the ranges of the areas that will be uncovered
        coverX, uncoverX = getCover(x, width)
        coverY, uncoverY = getCover(y, height)
        # so at this point we know that coverX and coverY makes a rect that
        # encases the area that we end up blitting to.  uncoverX/Y makes a
        # rect in the corner of the uncovered area.  So we need to combine
        # the uncoverX/Y with coverY/X to make what's left of the uncovered
        # area.  Explaining it makes it mush easier to do now.

        # But first we need to blit.
        x, width, srcx = getSlide(x, width)
        y, height, srcy = getSlide(y, height)
        self.blit(self, x, y, width, height, srcx, srcy)
        if uncoverX: # clear sides (0x20 is space)
            self.draw_rect(uncoverX[0], coverY[0], uncoverX[1], coverY[1],
                           0x20, self._fg, self._bg)
        if uncoverY: # clear top/bottom
            self.draw_rect(coverX[0], uncoverY[0], coverX[1], uncoverY[1],
                           0x20, self._fg, self._bg)
        if uncoverX and uncoverY: # clear corner
            self.draw_rect(uncoverX[0], uncoverY[0], uncoverX[1], uncoverY[1],
                           0x20, self._fg, self._bg)

    def clear(self, fg=Ellipsis, bg=Ellipsis):
        """Clears the entire L{Console}/L{Window}.

        Unlike other drawing functions, fg and bg can not be None.

        Args:
            fg (Union[Tuple[int, int, int], int, Ellipsis])
            bg (Union[Tuple[int, int, int], int, Ellipsis])

        .. seealso:: :any:`draw_rect`
        """
        raise NotImplementedError('this method is overwritten by subclasses')

    def get_char(self, x, y):
        """Return the character and colors of a tile as (ch, fg, bg)

        This method runs very slowly as is not recommended to be called
        frequently.

        Args:
            x (int): The x-coordinate to pick.
            y (int): The y-coordinate to pick.

        Returns:
            Tuple[int, Tuple[int, int, int], Tuple[int, int, int]]: A 3-item
                tuple: `(int, fg, bg)`

                The first item is an integer of the
                character at the position (x, y) the second and third are the
                foreground and background colors respectfully.

        .. seealso:: :any:`draw_char`
        """
        raise NotImplementedError('Method here only exists for the docstring')

    def __contains__(self, position):
        """Use ``((x, y) in console)`` to check if a position is drawable on
        this console.
        """
        x, y = position
        return (0 <= x < self.width) and (0 <= y < self.height)

class Console(_BaseConsole):
    """Contains character and color data and can be drawn to.

    The console created by the :any:`tdl.init` function is the root console
    and is the console that is rendered to the screen with :any:`flush`.

    Any console created from the Console class is an off-screen console that
    can be drawn on before being :any:`blit` to the root console.

    Args:
        width (int): Width of the new console in tiles
        height (int): Height of the new console in tiles
    """

    def __init__(self, width, height):
        _BaseConsole.__init__(self)
        self.console_c = _lib.TCOD_console_new(width, height)
        self.console = self
        self.width = width
        self.height = height

    @property
    def tcod_console(self):
        return self.console_c
    @tcod_console.setter
    def tcod_console(self, value):
        self.console_c = value

    @classmethod
    def _newConsole(cls, console):
        """Make a Console instance, from a console ctype"""
        self = cls.__new__(cls)
        _BaseConsole.__init__(self)
        self.console_c = console
        self.console = self
        self.width = _lib.TCOD_console_get_width(console)
        self.height = _lib.TCOD_console_get_height(console)
        return self

    def _root_unhook(self):
        """Change this root console into a normal Console object and
        delete the root console from TCOD
        """
        global _rootinitialized, _rootConsoleRef
        # do we recognise this as the root console?
        # if not then assume the console has already been taken care of
        if(_rootConsoleRef and _rootConsoleRef() is self):
            # turn this console into a regular console
            unhooked = _lib.TCOD_console_new(self.width, self.height)
            _lib.TCOD_console_blit(self.console_c,
                                   0, 0, self.width, self.height,
                                   unhooked, 0, 0, 1, 1)
            # delete root console from TDL and TCOD
            _rootinitialized = False
            _rootConsoleRef = None
            _lib.TCOD_console_delete(self.console_c)
            # this Console object is now a regular console
            self.console_c = unhooked

    def __del__(self):
        """
        If the main console is garbage collected then the window will be closed as well
        """
        if self.console_c is None:
            return # this console was already deleted
        if self.console_c is _ffi.NULL:
            # a pointer to the special root console
            self._root_unhook() # unhook the console and leave it to the GC
            return
        # this is a normal console pointer and can be safely deleted
        _lib.TCOD_console_delete(self.console_c)
        self.console_c = None

    def __copy__(self):
        # make a new class and blit
        clone = self.__class__(self.width, self.height)
        clone.blit(self)
        return clone

    def __getstate__(self):
        # save data from get_char
        data = [self.get_char(x, y) for x,y in
                _itertools.product(range(self.width), range(self.height))]
        return self.width, self.height, data

    def __setstate__(self, state):
        # make console from __init__ and unpack a get_char array
        width, height, data = state
        self.__init__(width, height)
        for (x, y), graphic in zip(_itertools.product(range(width),
                                                      range(height)), data):
            self.draw_char(x, y, *graphic)

    @staticmethod
    def _translate(x, y):
        """Convertion x and y to their position on the root Console for this Window

        Because this is a Console instead of a Window we return the paramaters
        untouched"""
        return x, y

    def clear(self, fg=Ellipsis, bg=Ellipsis):
        # inherit docstring
        assert fg is not None and bg is not None, 'Can not use None with clear'
        fg = _format_color(fg, self._fg)
        bg = _format_color(bg, self._bg)
        _lib.TCOD_console_set_default_foreground(self.console_c,
                                                 _to_tcod_color(fg)[0])
        _lib.TCOD_console_set_default_background(self.console_c,
                                                 _to_tcod_color(bg)[0])
        _lib.TCOD_console_clear(self.console_c)


    def _set_char(self, x, y, char, fg=None, bg=None,
                  bgblend=_lib.TCOD_BKGND_SET):
        """
        Sets a character.
        This is called often and is designed to be as fast as possible.

        Because of the need for speed this function will do NO TYPE CHECKING
        AT ALL, it's up to the drawing functions to use the functions:
        _format_char and _format_color before passing to this."""
        # values are already formatted, honestly this function is redundant
        return _put_char_ex(self.console_c, x, y, char, fg, bg, bgblend)

    def _set_batch(self, batch, fg, bg, bgblend=1, nullChar=False):
        """
        Try to perform a batch operation otherwise fall back to _set_char.
        If fg and bg are defined then this is faster but not by very
        much.

        if any character is None then nullChar is True

        batch is a iterable of [(x, y), ch] items
        """
        for (x, y), char in batch:
            self._set_char(x, y, char, fg, bg, bgblend)

    def get_char(self, x, y):
        # inherit docstring
        x, y = self._normalizePoint(x, y)
        char = _lib.TCOD_console_get_char(self.console_c, x, y)
        bg = _lib.TCOD_console_get_char_background(self.console_c, x, y)
        fg = _lib.TCOD_console_get_char_foreground(self.console_c, x, y)
        return char, (fg.r, fg.g, fg.b), (bg.r, bg.g, bg.b)

    # Copy docstrings for Sphinx
    clear.__doc__ = _BaseConsole.clear.__doc__
    get_char.__doc__ = _BaseConsole.get_char.__doc__

    def __repr__(self):
        return "<Console (Width=%i Height=%i)>" % (self.width, self.height)


class Window(_BaseConsole):
    """Isolate part of a :any:`Console` or :any:`Window` instance.

    This classes methods are the same as :any:`tdl.Console`

    Making a Window and setting its width or height to None will extend it to
    the edge of the console.

    This follows the normal rules for indexing so you can use a
    negative integer to place the Window relative to the bottom
    right of the parent Console instance.

    `width` or `height` can be set to None to extend as far as possible
    to the bottom right corner of the parent Console or can be a
    negative number to be sized reltive to the Consoles total size.

    Args:
        console (Union(tdl.Console, tdl.Window)): The parent object.
        x (int): x-coordinate to place the Window.
        y (int): y-coordinate to place the Window.
        width (Optional[int]): Width of the Window.
        height (Optional[int]): Height of the Window.
    """

    __slots__ = ('parent', 'x', 'y')

    def __init__(self, console, x, y, width, height):
        _BaseConsole.__init__(self)
        assert isinstance(console, (Console, Window)), 'console parameter must be a Console or Window instance, got %s' % repr(console)
        self.parent = console
        self.x, self.y, self.width, self.height = console._normalizeRect(x, y, width, height)
        if isinstance(console, Console):
            self.console = console
        else:
            self.console = self.parent.console

    def _translate(self, x, y):
        """Convertion x and y to their position on the root Console"""
        # we add our position relative to our parent and then call then next parent up
        return self.parent._translate((x + self.x), (y + self.y))

    def clear(self, fg=Ellipsis, bg=Ellipsis):
        # inherit docstring
        assert fg is not None and bg is not None, 'Can not use None with clear'
        if fg is Ellipsis:
            fg = self._fg
        if bg is Ellipsis:
            bg = self._bg
        self.draw_rect(0, 0, None, None, 0x20, fg, bg)

    def _set_char(self, x, y, char=None, fg=None, bg=None, bgblend=1):
        self.parent._set_char((x + self.x), (y + self.y), char, fg, bg, bgblend)

    def _set_batch(self, batch, *args, **kargs):
        # positional values will need to be translated to the parent console
        myX = self.x # remove dots for speed up
        myY = self.y
        self.parent._set_batch((((x + myX, y + myY), ch)
                                   for ((x, y), ch) in batch), *args, **kargs)


    def draw_char(self, x, y, char, fg=Ellipsis, bg=Ellipsis):
        # inherit docstring
        x, y = self._normalizePoint(x, y)
        if fg is Ellipsis:
            fg = self._fg
        if bg is Ellipsis:
            bg = self._bg
        self.parent.draw_char(x + self.x, y + self.y, char, fg, bg)

    def draw_rect(self, x, y, width, height, string, fg=Ellipsis, bg=Ellipsis):
        # inherit docstring
        x, y, width, height = self._normalizeRect(x, y, width, height)
        if fg is Ellipsis:
            fg = self._fg
        if bg is Ellipsis:
            bg = self._bg
        self.parent.draw_rect(x + self.x, y + self.y, width, height,
                              string, fg, bg)

    def draw_frame(self, x, y, width, height, string, fg=Ellipsis, bg=Ellipsis):
        # inherit docstring
        x, y, width, height = self._normalizeRect(x, y, width, height)
        if fg is Ellipsis:
            fg = self._fg
        if bg is Ellipsis:
            bg = self._bg
        self.parent.draw_frame(x + self.x, y + self.y, width, height,
                               string, fg, bg)

    def get_char(self, x, y):
        # inherit docstring
        x, y = self._normalizePoint(x, y)
        xtrans, ytrans = self._translate(x, y)
        return self.console.get_char(xtrans, ytrans)

    def __repr__(self):
        return "<Window(X=%i Y=%i Width=%i Height=%i)>" % (self.x, self.y,
                                                          self.width,
                                                          self.height)


def init(width, height, title=None, fullscreen=False, renderer='SDL'):
    """Start the main console with the given width and height and return the
    root console.

    Call the consoles drawing functions.  Then remember to use L{tdl.flush} to
    make what's drawn visible on the console.

    Args:
        width (int): width of the root console (in tiles)
        height (int): height of the root console (in tiles)
        title (Optiona[Text]):
            Text to display as the window title.

            If left None it defaults to the running scripts filename.
        fullscreen (bool): Can be set to True to start in fullscreen mode.
        renderer (Text): Can be one of 'GLSL', 'OPENGL', or 'SDL'.

        Due to way Python works you're unlikely to see much of an
        improvement by using 'GLSL' over 'OPENGL' as most of the
        time Python is slow interacting with the console and the
        rendering itself is pretty fast even on 'SDL'.

    Returns:
        tdl.Console: The root console.

            Only what is drawn on the root console is
            what's visible after a call to :any:`tdl.flush`.
            After the root console is garbage collected, the window made by
            this function will close.

    .. seealso::
        :any:`Console` :any:`set_font`
    """
    RENDERERS = {'GLSL': 0, 'OPENGL': 1, 'SDL': 2}
    global _rootinitialized, _rootConsoleRef
    if not _fontinitialized: # set the default font to the one that comes with tdl
        set_font(_os.path.join(__path__[0], 'terminal8x8.png'),
                 None, None, True, True)

    if renderer.upper() not in RENDERERS:
        raise TDLError('No such render type "%s", expected one of "%s"' % (renderer, '", "'.join(RENDERERS)))
    renderer = RENDERERS[renderer.upper()]

    # If a console already exists then make a clone to replace it
    if _rootConsoleRef and _rootConsoleRef():
        # unhook the root console, turning into a regular console and deleting
        # the root console from libTCOD
        _rootConsoleRef()._root_unhook()

    if title is None: # use a default title
        if _sys.argv:
            # Use the script filename as the title.
            title = _os.path.basename(_sys.argv[0])
        else:
            title = 'python-tdl'

    _lib.TCOD_console_init_root(width, height, _encodeString(title), fullscreen, renderer)

    #event.get() # flush the libtcod event queue to fix some issues
    # issues may be fixed already

    event._eventsflushed = False
    _rootinitialized = True
    rootconsole = Console._newConsole(_ffi.NULL)
    _rootConsoleRef = _weakref.ref(rootconsole)

    return rootconsole

def flush():
    """Make all changes visible and update the screen.

    Remember to call this function after drawing operations.
    Calls to flush will enfore the frame rate limit set by :any:`tdl.set_fps`.

    This function can only be called after :any:`tdl.init`
    """
    if not _rootinitialized:
        raise TDLError('Cannot flush without first initializing with tdl.init')
    # flush the OS event queue, preventing lock-ups if not done manually
    event.get()
    _lib.TCOD_console_flush()

def set_font(path, columns=None, rows=None, columnFirst=False,
             greyscale=False, altLayout=False):
    """Changes the font to be used for this session.
    This should be called before :any:`tdl.init`

    If the font specifies its size in its filename (i.e. font_NxN.png) then this
    function can auto-detect the tileset formatting and the parameters columns
    and rows can be left None.

    While it's possible you can change the font mid program it can sometimes
    break in rare circumstances.  So use caution when doing this.

    Args:
        path (Text): A file path to a `.bmp` or `.png` file.
        columns (int):
            Number of columns in the tileset.

            Can be left None for auto-detection.
        rows (int):
            Number of rows in the tileset.

            Can be left None for auto-detection.
        columnFirst (bool):
            Defines if the characer order goes along the rows or colomns.

            It should be True if the charater codes 0-15 are in the
            first column, and should be False if the characters 0-15
            are in the first row.
        greyscale (bool):
            Creates an anti-aliased font from a greyscale bitmap.
            Otherwise it uses the alpha channel for anti-aliasing.

            Unless you actually need anti-aliasing from a font you
            know uses a smooth greyscale channel you should leave
            this on False.
        altLayout (bool)
            An alternative layout with space in the upper left corner.
            The colomn parameter is ignored if this is True,
            find examples of this layout in the `font/libtcod/`
            directory included with the python-tdl source.

    Raises:
        TDLError: Will be raised if no file is found at path or if auto-
                  detection fails.
    """
    # put up some constants that are only used here
    FONT_LAYOUT_ASCII_INCOL = 1
    FONT_LAYOUT_ASCII_INROW = 2
    FONT_TYPE_GREYSCALE = 4
    FONT_LAYOUT_TCOD = 8
    global _fontinitialized
    _fontinitialized = True
    flags = 0
    if altLayout:
        flags |= FONT_LAYOUT_TCOD
    elif columnFirst:
        flags |= FONT_LAYOUT_ASCII_INCOL
    else:
        flags |= FONT_LAYOUT_ASCII_INROW
    if greyscale:
        flags |= FONT_TYPE_GREYSCALE
    if not _os.path.exists(path):
        raise TDLError('Font %r not found.' % _os.path.abspath(path))
    path = _os.path.abspath(path)

    # and the rest is the auto-detect script
    imgSize = _getImageSize(path) # try to find image size
    if imgSize:
        fontWidth, fontHeight = None, None
        imgWidth, imgHeight = imgSize
        # try to get font size from filename
        match = _re.match('.*?([0-9]+)[xX]([0-9]+)', _os.path.basename(path))
        if match:
            fontWidth, fontHeight = match.groups()
            fontWidth, fontHeight = int(fontWidth), int(fontHeight)

            # estimate correct tileset size
            estColumns, remC = divmod(imgWidth, fontWidth)
            estRows, remR = divmod(imgHeight, fontHeight)
            if remC or remR:
                _warnings.warn("Font may be incorrectly formatted.")

            if not columns:
                columns = estColumns
            if not rows:
                rows = estRows
        else:
            # filename doesn't contain NxN, but we can still estimate the fontWidth
            # and fontHeight given number of columns and rows.
            if columns and rows:
                fontWidth, remC = divmod(imgWidth, columns)
                fontHeight, remR = divmod(imgHeight, rows)
                if remC or remR:
                    _warnings.warn("Font may be incorrectly formatted.")

            # the font name excluded the fonts size
            if not (columns and rows):
                # no matched font size and no tileset is given
                raise TDLError('%s has no font size in filename' % _os.path.basename(path))

        if columns and rows:
            # confirm user set options
            if (fontWidth * columns != imgWidth or
                fontHeight * rows != imgHeight):
                _warnings.warn("set_font parameters are set as if the image size is (%d, %d) when the detected size is actually (%i, %i)"
                             % (fontWidth * columns, fontHeight * rows,
                                imgWidth, imgHeight))
    else:
        _warnings.warn("%s is probably not an image." % _os.path.basename(path))

    if not (columns and rows):
        # didn't auto-detect
        raise TDLError('Can not auto-detect the tileset of %s' % _os.path.basename(path))

    _lib.TCOD_console_set_custom_font(_encodeString(path), flags, columns, rows)

def get_fullscreen():
    """Returns True if program is fullscreen.

    Returns:
        bool:  Returns True if the application is in full-screen mode.
               Otherwise returns False.
    """
    if not _rootinitialized:
        raise TDLError('Initialize first with tdl.init')
    return _lib.TCOD_console_is_fullscreen()

def set_fullscreen(fullscreen):
    """Changes the fullscreen state.

    Args:
        fullscreen (bool): True for full-screen, False for windowed mode.
    """
    if not _rootinitialized:
        raise TDLError('Initialize first with tdl.init')
    _lib.TCOD_console_set_fullscreen(fullscreen)

def set_title(title):
    """Change the window title.

    Args:
        title (Text): The new title text.
    """
    if not _rootinitialized:
        raise TDLError('Not initilized.  Set title with tdl.init')
    _lib.TCOD_console_set_window_title(_encodeString(title))

def screenshot(path=None):
    """Capture the screen and save it as a png file.

    If path is None then the image will be placed in the current
    folder with the names:
    ``screenshot001.png, screenshot002.png, ...``

    Args:
        path (Optional[Text]): The file path to save the screenshot.
    """
    if not _rootinitialized:
        raise TDLError('Initialize first with tdl.init')
    if isinstance(path, str):
        _lib.TCOD_sys_save_screenshot(_encodeString(path))
    elif path is None: # save to screenshot001.png, screenshot002.png, ...
        filelist = _os.listdir('.')
        n = 1
        filename = 'screenshot%.3i.png' % n
        while filename in filelist:
            n += 1
            filename = 'screenshot%.3i.png' % n
        _lib.TCOD_sys_save_screenshot(_encodeString(filename))
    else: # assume file like obj
        #save to temp file and copy to file-like obj
        tmpname = _os.tempnam()
        _lib.TCOD_sys_save_screenshot(_encodeString(tmpname))
        with tmpname as tmpfile:
            path.write(tmpfile.read())
        _os.remove(tmpname)
    #else:
    #    raise TypeError('path is an invalid type: %s' % type(path))

def set_fps(fps):
    """Set the maximum frame rate.

    Further calls to :any:`tdl.flush` will limit the speed of
    the program to run at `fps` frames per second. This can
    also be set to None to remove the frame rate limit.

    Args:
        fps (optional[int]): The frames per second limit, or None.
    """
    _lib.TCOD_sys_set_fps(fps or 0)

def get_fps():
    """Return the current frames per second of the running program set by
    :any:`set_fps`

    Returns:
        int: The frame rate set by :any:`set_fps`.
             If there is no current limit, this will return 0.
    """
    return _lib.TCOD_sys_get_fps()

def force_resolution(width, height):
    """Change the fullscreen resoulution.

    Args:
        width (int): Width in pixels.
        height (int): Height in pixels.
    """
    _lib.TCOD_sys_force_fullscreen_resolution(width, height)


__all__ = [_var for _var in locals().keys() if _var[0] != '_'] # remove modules from __all__
__all__ += ['_BaseConsole'] # keep this object public to show the documentation in epydoc
__all__.remove('absolute_import')
__all__.remove('division')
__all__.remove('print_function')
__all__.remove('unicode_literals')

# backported function names
_BaseConsole.setMode = _style.backport(_BaseConsole.set_mode)
_BaseConsole.setColors = _style.backport(_BaseConsole.set_colors)
_BaseConsole.printStr = _style.backport(_BaseConsole.print_str)
_BaseConsole.drawChar = _style.backport(_BaseConsole.draw_char)
_BaseConsole.drawStr = _style.backport(_BaseConsole.draw_str)
_BaseConsole.drawRect = _style.backport(_BaseConsole.draw_rect)
_BaseConsole.drawFrame = _style.backport(_BaseConsole.draw_frame)
_BaseConsole.getCursor = _style.backport(_BaseConsole.get_cursor)
_BaseConsole.getSize = _style.backport(_BaseConsole.get_size)
_BaseConsole.getChar = _style.backport(_BaseConsole.get_char)

Console.getChar = _style.backport(Console.get_char)

Window.drawChar = _style.backport(Window.draw_char)
Window.drawRect = _style.backport(Window.draw_rect)
Window.drawFrame = _style.backport(Window.draw_frame)
Window.getChar = _style.backport(Window.get_char)

setFont = _style.backport(set_font)
getFullscreen = _style.backport(get_fullscreen)
setFullscreen = _style.backport(set_fullscreen)
setTitle = _style.backport(set_title)
setFPS = _style.backport(set_fps)
getFPS = _style.backport(get_fps)
forceResolution = _style.backport(force_resolution)
