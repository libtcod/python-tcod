"""
    The documentation for python-tdl.  A Pythonic port of
    U{libtcod<http://doryen.eptalys.net/libtcod/>}.
    
    You can find the project page on Google Code
    U{here<http://code.google.com/p/python-tdl/>}.

    Getting Started
    ===============
      Once the library is imported you can load the font you want to use with
      L{tdl.setFont}.
      This is optional and when skipped will use a decent default font.
      
      After that you call L{tdl.init} to set the size of the window and get the
      root console in return.
      This console is the canvas to what will appear on the screen.

    Indexing Consoles
    =================
      For most methods taking a position you can use Python-style negative
      indexes to refer to the opposite side of a console with (-1, -1)
      starting at the bottom right.
      You can also check if a point is part of a console using containment
      logic i.e. ((x, y) in console).
      
    Drawing
    =======
      Once you have the root console from L{tdl.init} you can start drawing on
      it using a method such as L{Console.drawChar}.
      When using this method you can have the char parameter be an integer or a
      single character string.
      The fgcolor and bgcolor parameters expect a three item list
      [red, green, blue] with integers in the 0-255 range with [0, 0, 0] being
      black and [255, 255, 255] being white.
      Or instead you can use None for any of the three parameters to tell the
      library to keep what is at that spot instead of overwriting it.
      After the drawing functions are called a call to L{tdl.flush} will update
      the screen.
"""

import sys
import os
import ctypes
import weakref
import array
import itertools
import textwrap

from . import event, map, noise
from .__tcod import _lib, _Color, _unpackfile

_IS_PYTHON3 = (sys.version_info[0] == 3)

if _IS_PYTHON3: # some type lists to use with isinstance
    _INTTYPES = (int,)
    _NUMTYPES = (int, float)
    _STRTYPES = (str, bytes)
else:
    _INTTYPES = (int, long)
    _NUMTYPES = (int, long, float)
    _STRTYPES = (str,)

def _encodeString(string): # still used for filepaths, and that's about it
    "changes string into bytes if running in python 3, for sending to ctypes"
    if _IS_PYTHON3 and isinstance(string, str):
        return string.encode()
    return string

#def _formatString(string):
#    pass

def _formatChar(char):
    """Prepares a single character for passing to ctypes calls, needs to return
    an integer but can also pass None which will keep the current character
    instead of overwriting it.

    This is called often and needs to be optimized whenever possible.
    """
    if char is None:
        return None
    if isinstance(char, _INTTYPES):
        return char
    if isinstance(char, _STRTYPES) and len(char) == 1:
        return ord(char)
    raise TypeError('Expected char parameter to be a single character string, number, or None, got: %s' % repr(char))

_fontinitialized = False
_rootinitialized = False
_rootConsoleRef = None
# remove dots from common functions
_setchar = _lib.TCOD_console_set_char
_setfore = _lib.TCOD_console_set_char_foreground
_setback = _lib.TCOD_console_set_char_background
_setcharEX = _lib.TCOD_console_put_char_ex
def _verify_colors(*colors):
    """Used internally.
    Raise an assertion error if the parameters can not be converted into colors.
    """
    for color in colors:
        assert _iscolor(color), 'a color must be a 3 item tuple, web format, or None, received %s' % repr(color)
    return True

def _iscolor(color):
    """Used internally.
    A debug function to see if an object can be used as a TCOD color struct.
    None counts as a parameter to keep the current colors instead.

    This function is often part of an inner-loop and can slow a program down.
    It has been made to work with assert and can be skipped with the -O flag.
    Still it's called often and must be optimized.
    """
    if color is None:
        return True
    if isinstance(color, (tuple, list, _Color)):
        return len(color) == 3
    if isinstance(color, _INTTYPES):
        return True
    return False

## not using this for now
#class Color(object):
#    
#    def __init__(self, r, g, b):
#        self._color = (r, g, b)
#        self._ctype = None
#        
#    def _getCType(self):
#        if not self._ctype:
#            self._ctype = _Color(*self._color)
#        return self._ctype
#        
#    def __len__(self):
#        return 3
    
def _formatColor(color):
    """Format the color to ctypes
    """
    if color is None:
        return None
    if isinstance(color, _Color):
        return color
    #if isinstance(color, Color):
    #    return color._getCType()
    if isinstance(color, _INTTYPES):
        # format a web style color with the format 0xRRGGBB
        return _Color(color >> 16 & 0xff, color >> 8 & 0xff, color & 0xff)
    return _Color(*color)

class TDLError(Exception):
    """
    The catch all for most TDL specific errors.
    """

class _MetaConsole(object):
    """
    Contains methods shared by both the L{Console} and L{Window} classes.
    """
    __slots__ = ('width', 'height', 'console', '__weakref__', '__dict__')

    def drawChar(self, x, y, char, fgcolor=(255, 255, 255), bgcolor=(0, 0, 0)):
        """Draws a single character.

        @type x: int
        @param x: X coordinate to draw at.
        @type y: int
        @param y: Y coordinate to draw at.
        
        @type char: int, string, or None
        @param char: Should be an integer, single character string, or None.

                     You can set the char parameter as None if you only want to change
                     the colors of the tile.

        @type fgcolor: (r, g, b) or None
        @param fgcolor: For fgcolor and bgcolor you use a 3 item list with
                        integers ranging 0-255 or None.
                        
                        None will keep the current color at this position unchanged.
        @type bgcolor: (r, g, b) or None
        @param bgcolor: Background color.  See fgcolor

        @raise AssertionError: Having x or y values that can't be placed inside
                               of the console will raise an AssertionError.
                               You can use always use ((x, y) in console) to
                               check if a tile is drawable.
        """

        assert _verify_colors(fgcolor, bgcolor)
        x, y = self._normalizePoint(x, y)
        x, y = ctypes.c_int(x), ctypes.c_int(y)
        self._setChar(x, y, _formatChar(char),
                      _formatColor(fgcolor), _formatColor(bgcolor))

    def drawStr(self, x, y, string, fgcolor=(255, 255, 255), bgcolor=(0, 0, 0)):
        """Draws a string starting at x and y.  Optinally colored.

        A string that goes past the right side will wrap around.  A string
        wraping to below the console will raise a L{TDLError} but will still be
        written out.  This means you can safely ignore the errors with a
        try... except block if you're fine with partily written strings.

        \\r and \\n are drawn on the console as normal character tiles.  No
        special encoding is done and any string will translate to the character
        table as is.
        
        For a string drawing operation that respects special characters see the
        L{Typewriter} class.

        @type x: int
        @param x: X coordinate to draw at.
        @type y: int
        @param y: Y coordinate to draw at.
        
        @type string: string or iterable
        @param string: Can be a string or an iterable of numbers.
                       
                       Special characters are ignored and rendered as any other
                       character.
        
        @type fgcolor: (r, g, b) or None
        @param fgcolor: For fgcolor and bgcolor you use a 3 item list with
                        integers ranging 0-255 or None.
                        
                        None will keep the current color at this position unchanged.
        @type bgcolor: (r, g, b) or None
        @param bgcolor: Background color.  See fgcolor
        
        @raise AssertionError: Having x or y values that can't be placed inside
                               of the console will raise an AssertionError.
                               
                               You can use always use ((x, y) in console) to
                               check if a tile is drawable.
        """

        x, y = self._normalizePoint(x, y)
        assert _verify_colors(fgcolor, bgcolor)
        fgcolor, bgcolor = _formatColor(fgcolor), _formatColor(bgcolor)
        width, height = self.getSize()
        batch = [] # prepare a batch operation
        def _drawStrGen(x=x, y=y, string=string, width=width, height=height):
            """Generator for drawStr

            Iterates over ((x, y), ch) data for _setCharBatch, raising an
            error if the end of the console is reached.
            """
            for char in string:
                if y == height:
                    raise TDLError('End of console reached.')
                #batch.append(((x, y), _formatChar(char))) # ((x, y), ch)
                yield((x, y), _formatChar(char))
                x += 1 # advance cursor
                if x == width: # line break
                    x = 0
                    y += 1
        self._setCharBatch(_drawStrGen(), fgcolor, bgcolor)

    def drawRect(self, x, y, width, height, string, fgcolor=(255, 255, 255), bgcolor=(0, 0, 0)):
        """Draws a rectangle starting from x and y and extending to width and height.
        
        If width or height are None then it will extend to the edge of the console.

        @type x: int
        @param x: x coordinate to draw at.
        @type y: int
        @param y: y coordinate to draw at.
        
        @type width: int or None
        @param width: Width of the rectangle.
                      
                      Can be None to extend to the bottom right of the
                      console or can be a negative number to be sized reltive
                      to the total size of the console.
        @type height: int or None
        @param height: Height of the rectangle.  See width.
        
        @type string: int, string, or None
        @param string: Should be an integer, single character string, or None.

                       You can set the char parameter as None if you only want
                       to change the colors of an area.
        
        @type fgcolor: (r, g, b) or None
        @param fgcolor: For fgcolor and bgcolor you use a 3 item list with
                        integers ranging 0-255 or None.
                        
                        None will keep the current color at this position unchanged.
        @type bgcolor: (r, g, b) or None
        @param bgcolor: Background color.  See fgcolor
        
        @raise AssertionError: Having x or y values that can't be placed inside
                               of the console will raise an AssertionError.
                               
                               You can use always use ((x, y) in console) to
                               check if a tile is drawable.
        """
        x, y, width, height = self._normalizeRect(x, y, width, height)
        assert _verify_colors(fgcolor, bgcolor)
        fgcolor, bgcolor = _formatColor(fgcolor), _formatColor(bgcolor)
        char = _formatChar(string)
        # use itertools to make an x,y grid
        # using ctypes here reduces type converstions later
        grid = itertools.product((ctypes.c_int(x) for x in range(x, x + width)),
                                 (ctypes.c_int(y) for y in range(y, y + height)))
        # zip the single character in a batch variable
        batch = zip(grid, itertools.repeat(char, width * height))
        self._setCharBatch(batch, fgcolor, bgcolor, nullChar=(char is None))

    def drawFrame(self, x, y, width, height, string, fgcolor=(255, 255, 255), bgcolor=(0, 0, 0)):
        """Similar to L{drawRect} but only draws the outline of the rectangle.

        @type x: int
        @param x: x coordinate to draw at.
        @type y: int
        @param y: y coordinate to draw at.
        
        @type width: int or None
        @param width: Width of the rectangle.
                      
                      Can be None to extend to the bottom right of the
                      console or can be a negative number to be sized reltive
                      to the total size of the console.
        @type height: int or None
        @param height: Height of the rectangle.  See width.
        
        @type string: int, string, or None
        @param string: Should be an integer, single character string, or None.

                       You can set the char parameter as None if you only want
                       to change the colors of an area.
        
        @type fgcolor: (r, g, b) or None
        @param fgcolor: For fgcolor and bgcolor you use a 3 item list with
                        integers ranging 0-255 or None.
                        
                        None will keep the current color at this position unchanged.
        @type bgcolor: (r, g, b) or None
        @param bgcolor: Background color.  See fgcolor
        
        @raise AssertionError: Having x or y values that can't be placed inside
                               of the console will raise an AssertionError.
                               
                               You can use always use ((x, y) in console) to
                               check if a tile is drawable.
        """
        x, y, width, height = self._normalizeRect(x, y, width, height)
        assert _verify_colors(fgcolor, bgcolor)
        fgcolor, bgcolor = _formatColor(fgcolor), _formatColor(bgcolor)
        char = _formatChar(string)
        if width == 1 or height == 1: # it's just a single width line here
            return self.drawRect(x, y, width, height, char, fgcolor, bgcolor)

        # draw sides of frame with drawRect
        self.drawRect(x, y, 1, height, char, fgcolor, bgcolor)
        self.drawRect(x, y, width, 1, char, fgcolor, bgcolor)
        self.drawRect(x + width - 1, y, 1, height, char, fgcolor, bgcolor)
        self.drawRect(x, y + height - 1, width, 1, char, fgcolor, bgcolor)

    def _normalizePoint(self, x, y):
        """Check if a point is in bounds and make minor adjustments.
        
        Respects Pythons negative indexes.  -1 starts at the bottom right.
        Replaces the _drawable function
        """
        assert isinstance(x, _INTTYPES), 'x must be an integer, got %s' % repr(x)
        assert isinstance(y, _INTTYPES), 'y must be an integer, got %s' % repr(y)

        assert (-self.width <= x < self.width) and (-self.height <= y < self.height), \
                ('(%i, %i) is an invalid postition on %s' % (x, y, self))
                 
        # handle negative indexes
        if x < 0:
            x += self.width
        if y < 0:
            y += self.height
        return (x, y)

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

    def blit(self, source, x=0, y=0, width=None, height=None, srcX=0, srcY=0):
        """Blit another console or Window onto the current console.

        By default it blits the entire source to the topleft corner.

        @type source: L{Console} or L{Window}
        @param source: Source window can be a L{Console} or L{Window} instance.
                       It can even blit to itself without any problems.
        
        @type x: int
        @param x: X coordinate to blit to.
        @type y: int
        @param y: Y coordinate to blit to.
        
        @type width: int or None
        @param width: Width of the rectangle.
                      
                      Can be None to extend as far as possible to the
                      bottom right corner of the blit area or can be a negative
                      number to be sized reltive to the total size of the
                      B{destination} console.
        @type height: int or None
        @param height: Height of the rectangle.  See width.
        
        @type srcX: int
        @param srcX: The source consoles x coordinate to blit from.
        @type srcY: int
        @param srcY: The source consoles y coordinate to blit from.
        """
        # hardcode alpha settings for now
        fgalpha=1.0
        bgalpha=1.0

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
            _lib.TCOD_console_blit(source, srcX, srcY, width, height, tmp, 0, 0, fgalpha, bgalpha)
            _lib.TCOD_console_blit(tmp, 0, 0, width, height, self, x, y, fgalpha, bgalpha)
        else:
            _lib.TCOD_console_blit(source, srcX, srcY, width, height, self, x, y, fgalpha, bgalpha)

    def getSize(self):
        """Return the size of the console as (width, height)

        @rtype: (int, int)
        """
        return self.width, self.height

    def scroll(self, x, y):
        """Scroll the contents of the console in the direction of x,y.

        Uncovered areas will be cleared.
        @type x: int
        @param x: Distance to scroll along x-axis
        @type y: int
        @param y: Distance to scroll along y-axis
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

        width, height = self.getSize()
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
            self.drawRect(uncoverX[0], coverY[0], uncoverX[1], coverY[1], 0x20)
        if uncoverY: # clear top/bottom
            self.drawRect(coverX[0], uncoverY[0], coverX[1], uncoverY[1], 0x20)
        if uncoverX and uncoverY: # clear corner
            self.drawRect(uncoverX[0], uncoverY[0], uncoverX[1], uncoverY[1], 0x20)

    def getChar(self, x, y):
        """Return the character and colors of a tile as (ch, fg, bg)
        
        This method runs very slowly as is not recommended to be called
        frequently.

        @rtype: (int, (r, g, b), (r, g, b))
        @returns: Returns a 3-item tuple.  The first item is an integer of the
                  character at the position (x, y) the second and third are the
                  foreground and background colors respectfully.
        """
        raise NotImplementedError('Method here only exists for the docstring')
            
    def __contains__(self, position):
        """Use ((x, y) in console) to check if a position is drawable on this console.
        """
        x, y = position
        return (0 <= x < self.width) and (0 <= y < self.height)

class Console(_MetaConsole):
    """Contains character and color data and can be drawn to.

    The console created by the L{tdl.init} function is the root console and is the
    console that is rendered to the screen with L{flush}.

    Any console created from the Console class is an off-screen console that
    can be drawn on before being L{blit} to the root console.
    """

    __slots__ = ('_as_parameter_', '_typewriter')

    def __init__(self, width, height):
        """Create a new offscreen console.
        
        @type width: int
        @param width: Width of the console in tiles
        @type height: int
        @param height: Height of the console in tiles
        """
        if not _rootinitialized:
            raise TDLError('Can not create Console\'s before tdl.init')
        self._as_parameter_ = _lib.TCOD_console_new(width, height)
        self.console = self
        self.width = width
        self.height = height
        self._typewriter = None # "typewriter lock", makes sure the colors are set to the typewriter

    @classmethod
    def _newConsole(cls, console):
        """Make a Console instance, from a console ctype"""
        self = cls.__new__(cls)
        self._as_parameter_ = console
        self.console = self
        self.width = _lib.TCOD_console_get_width(self)
        self.height = _lib.TCOD_console_get_height(self)
        self._typewriter = None
        return self

    def __del__(self):
        """
        If the main console is garbage collected then the window will be closed as well
        """
        # If this is the root console the window will close when collected
        try:
            if isinstance(self._as_parameter_, ctypes.c_void_p):
                global _rootinitialized, _rootConsoleRef
                _rootinitialized = False
                _rootConsoleRef = None
            _lib.TCOD_console_delete(self)
        except StandardError:
            pass # I forget why I put this here but I'm to afraid to delete it

    def _replace(self, console):
        """Used internally

        Mostly used just to replace this Console object with the root console
        If another Console object is used then they are swapped
        """
        if isinstance(console, Console):
            self._as_parameter_, console._as_parameter_ = \
              console._as_parameter_, self._as_parameter_ # swap tcod consoles
        else:
            self._as_parameter_ = console
        self.width = _lib.TCOD_console_get_width(self)
        self.height = _lib.TCOD_console_get_height(self)
        return self

    def _translate(self, x, y):
        """Convertion x and y to their position on the root Console for this Window

        Because this is a Console instead of a Window we return the paramaters
        untouched"""
        return x, y

    def clear(self, fgcolor=(0, 0, 0), bgcolor=(0, 0, 0)):
        """Clears the entire Console.

        @type fgcolor: (r, g, b)
        @param fgcolor: Foreground color.
        
                        Must be a 3-item list with integers that range 0-255.
                        
                        Unlike most other operations you can not use None here.
        @type bgcolor: (r, g, b)
        @param bgcolor: Background color.  See fgcolor.
        """
        assert _verify_colors(fgcolor, bgcolor)
        assert fgcolor and bgcolor, 'Can not use None with clear'
        self._typewriter = None
        _lib.TCOD_console_set_default_background(self, _formatColor(bgcolor))
        _lib.TCOD_console_set_default_foreground(self, _formatColor(fgcolor))
        _lib.TCOD_console_clear(self)

    def _setChar(self, x, y, char, fgcolor=None, bgcolor=None, bgblend=1):
        """
        Sets a character.
        This is called often and is designed to be as fast as possible.

        Because of the need for speed this function will do NO TYPE CHECKING
        AT ALL, it's up to the drawing functions to use the functions:
        _formatChar and _formatColor before passing to this."""
        # buffer values as ctypes objects
        console = self._as_parameter_

        if char is not None and fgcolor is not None and bgcolor is not None:
            _setcharEX(console, x, y, char, fgcolor, bgcolor)
            return
        if char is not None:
            _setchar(console, x, y, char)
        if fgcolor is not None:
            _setfore(console, x, y, fgcolor)
        if bgcolor is not None:
            _setback(console, x, y, bgcolor, bgblend)

    def _setCharBatch(self, batch, fgcolor, bgcolor, bgblend=1, nullChar=False):
        """
        Try to perform a batch operation otherwise fall back to _setChar.
        If fgcolor and bgcolor are defined then this is faster but not by very
        much.

        batch is a iterable of [(x, y), ch] items
        """
        if fgcolor and not nullChar:
            # buffer values as ctypes objects
            self._typewriter = None # clear the typewriter as colors will be set
            console = self._as_parameter_
            if not bgcolor:
                bgblend = 0
            bgblend = ctypes.c_int(bgblend)

            _lib.TCOD_console_set_default_background(console, bgcolor)
            _lib.TCOD_console_set_default_foreground(console, fgcolor)
            _putChar = _lib.TCOD_console_put_char # remove dots and make local
            for (x, y), char in batch:
                _putChar(console, x, y, char, bgblend)
        else:
            for (x, y), char in batch:
                self._setChar(x, y, char, fgcolor, bgcolor, bgblend)

    def getChar(self, x, y):
        # inherit docstring
        x, y = self._normalizePoint(x, y)
        char = _lib.TCOD_console_get_char(self, x, y)
        bgcolor = _lib.TCOD_console_get_char_background_wrapper(self, x, y)
        fgcolor = _lib.TCOD_console_get_char_foreground_wrapper(self, x, y)
        return char, tuple(fgcolor), tuple(bgcolor)

    def __repr__(self):
        return "<Console (Width=%i Height=%i)>" % (self.width, self.height)


class Window(_MetaConsole):
    """A Window contains a small isolated part of a Console.

    Drawing on the Window draws on the Console.

    Making a Window and setting its width or height to None will extend it to
    the edge of the console.
    """

    __slots__ = ('parent', 'x', 'y')

    def __init__(self, console, x, y, width, height):
        """Isolate part of a L{Console} or L{Window} instance.
        
        @type console: L{Console} or L{Window}
        @param console: The parent object which can be a L{Console} or another
                        L{Window} instance.
        
        @type x: int
        @param x: X coordinate to place the Window.
                  
                  This follows the normal rules for indexing so you can use a
                  negative integer to place the Window relative to the bottom
                  right of the parent Console instance.
        @type y: int
        @param y: Y coordinate to place the Window.
        
                  See x.
        
        @type width: int or None
        @param width: Width of the Window.
                      
                      Can be None to extend as far as possible to the
                      bottom right corner of the parent Console or can be a
                      negative number to be sized reltive to the Consoles total
                      size.
        @type height: int or None
        @param height: Height of the Window.
        
                       See width.
        """
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

    def clear(self, fgcolor=(0, 0, 0), bgcolor=(0, 0, 0)):
        """Clears the entire Window.

        @type fgcolor: (r, g, b)
        @param fgcolor: Foreground color.
        
                        Must be a 3-item list with integers that range 0-255.
                        
                        Unlike most other operations you can not use None here.
        @type bgcolor: (r, g, b)
        @param bgcolor: Background color.  See fgcolor.
        """
        assert _verify_colors(fgcolor, bgcolor)
        assert fgcolor and bgcolor, 'Can not use None with clear'
        self.draw_rect(0, 0, None, None, 0x20, fgcolor, bgcolor)

    def _setChar(self, x, y, char=None, fgcolor=None, bgcolor=None, bgblend=1):
        self.parent._setChar((x + self.x), (y + self.y), char, fgcolor, bgcolor, bgblend)

    def _setCharBatch(self, batch, fgcolor, bgcolor, bgblend=1):
        myX = self.x # remove dots for speed up
        myY = self.y
        self.parent._setCharBatch((((x + myX, y + myY), ch) for ((x, y), ch) in batch),
                                  fgcolor, bgcolor, bgblend)
    
    
    def drawChar(self, x, y, char, fgcolor=(255, 255, 255), bgcolor=(0, 0, 0)):
        # inherit docstring
        x, y = self._normalizePoint(x, y)
        self.parent.drawChar(x + self.x, y + self.y, char, fgcolor, bgcolor)
    
    def drawRect(self, x, y, width, height, string, fgcolor=(255, 255, 255), bgcolor=(0, 0, 0)):
        # inherit docstring
        x, y, width, height = self._normalizeRect(x, y, width, height)
        self.parent.drawRect(x + self.x, y + self.y, width, height, string, fgcolor, bgcolor)
        
    def drawFrame(self, x, y, width, height, string, fgcolor=(255, 255, 255), bgcolor=(0, 0, 0)):
        # inherit docstring
        x, y, width, height = self._normalizeRect(x, y, width, height)
        self.parent.drawFrame(x + self.x, y + self.y, width, height, string, fgcolor, bgcolor)

    def getChar(self, x, y):
        # inherit docstring
        x, y = self._normalizePoint(x, y)
        return self.console.getChar(self._translate(x, y))

    def __repr__(self):
        return "<Window(X=%i Y=%i Width=%i Height=%i)>" % (self.x, self.y,
                                                          self.width,
                                                          self.height)


class Typewriter(object):
    """Converts a console into a scrolling text log that respects special
    characters.
    
    This class works best on a L{Window} or off-screen L{Console} instance.
    In a L{Window} for example the scrolling text is limited to the L{Window}'s
    isolated area.
    """

    def __init__(self, console):
        """Add a virtual cursor to a L{Console} or L{Window} instance.
        
        @type console: L{Console} or L{Window}
        """
        assert isinstance(console, (Console, Window)), 'console parameter must be a Console or Window instance, got %s' % repr(console)
        self.parent = console
        if isinstance(self.parent, Console):
            self.console = self.parent
        else:
            self.console = self.parent.console
        self.cursor = (0, 0) # cursor position
        self.scrollMode = 'scroll' #can be 'scroll', 'error'
        self.fgcolor = _formatColor((255, 255, 255))
        self.bgcolor = _formatColor((0, 0, 0))
        self._bgblend = 1 # SET

    def _normalize(self, x, y):
        """return the normalized the cursor position."""
        width, height = self.parent.getSize()
        while x >= width:
            x -= width
            y += 1
        while y >= height:
            if self.scrollMode == 'scroll':
                y -= 1
                self.parent.scroll(0, -1)
            elif self.scrollMode == 'error':
                # reset the cursor on error
                self.cursor = (0, 0)
                raise TDLError('Typewriter cursor has reached the end of the console')
        return (x, y)

    def getCursor(self):
        """Return the virtual cursor position.
        
        @rtype: (int, int)
        @return: Returns (x, y) a 2-integer tuple containing where the next
                 L{addChar} or L{addStr} will start at.
                 
                 This can be changed with the L{move} method."""
        x, y = self.cursor
        width, height = self.parent.getSize()
        while x >= width:
            x -= width
            y += 1
        if y >= height and self.scrollMode == 'scroll':
            y = height - 1
        return x, y

    def move(self, x, y):
        """Move the virtual cursor.
        
        @type x: int
        @param x: X position to place the cursor.
        @type y: int
        @param y: Y position to place the cursor.
        """
        self.cursor = self.parent._normalizePoint(x, y)
        
    def setFG(self, color):
        """Change the foreground color"""
        assert _iscolor(color)
        assert color is not None
        self.fgcolor = _formatColor(color)
        if self.console._typewriter is self:
            _lib.TCOD_console_set_default_foreground(self.console, self.fgcolor)
        
    def setBG(self, color):
        """Change the background color"""
        assert _iscolor(color)
        assert color is not None
        self.bgcolor = _formatColor(color)
        if self.console._typewriter is self:
            _lib.TCOD_console_set_default_background(self.console, self.bgcolor)
        
    def _updateConsole(self):
        """Make sure the colors on a console match the Typewriter instance"""
        if self.console._typewriter is not self:
            self.console._typewriter = self
            
            _lib.TCOD_console_set_default_background(self.console, self.bgcolor)
            _lib.TCOD_console_set_default_foreground(self.console, self.fgcolor)
        

    def addChar(self, char):
        """Draw a single character at the cursor."""
        if char == '\n': # line break
            x = 0
            y += 1
            return
        if char == '\r': # return
            x = 0
            return
        x, y = self._normalize(*self.cursor)
        self.cursor = [x + 1, y] # advance cursor on next draw
        self._updateConsole()
        x, y = self.parent._translate(x, y)
        _lib.TCOD_console_put_char(self.console._as_parameter_, x, y, _formatChar(char), self._bgblend)
        

    def addStr(self, string):
        """Write a string at the cursor.  Handles special characters such as newlines.
        
        @type string: string
        @param string: 
        """
        x, y = self.cursor
        for char in string:
            if char == '\n': # line break
                x = 0
                y += 1
                continue
            if char == '\r': # return
                x = 0
                continue
            x, y = self._normalize(x, y)
            self.parent.drawChar(x, y, char, self.fgcolor, self.bgcolor)
            x += 1
        self.cursor = (x, y)

    def write(self, string):
        """This method mimics basic file-like behaviour.
        
        Because of this method you can replace sys.stdout or sys.stderr with
        a L{Typewriter} instance.
        
        @type string: string
        """
        # some 'basic' line buffer stuff.
        # there must be an easier way to do this.  The textwrap module didn't
        # help much.
        x, y = self._normalize(*self.cursor)
        width, height = self.parent.getSize()
        wrapper = textwrap.TextWrapper(initial_indent=(' '*x), width=width)
        writeLines = []
        for line in string.split('\n'):
            if line:
                writeLines += wrapper.wrap(line)
                wrapper.initial_indent = ''
            else:
                writeLines.append([])

        for line in writeLines:
            x, y = self._normalize(x, y)
            self.parent.drawStr(x, y, line[x:], self.fgcolor, self.bgcolor)
            y += 1
            x = 0
        y -= 1
        self.cursor = (x, y)
        

def init(width, height, title=None, fullscreen=False, renderer='OPENGL'):
    """Start the main console with the given width and height and return the
    root console.

    Call the consoles drawing functions.  Then remember to use L{tdl.flush} to
    make what's drawn visible on the console.

    @type width: int
    @param width: width of the root console (in tiles)
    
    @type height: int
    @param height: height of the root console (in tiles)
    
    @type title: string
    @param title: Text to display as the window title.
                  
                  If left None it defaults to the running scripts filename.
    
    @type fullscreen: boolean
    @param fullscreen: Can be set to True to start in fullscreen mode.

    @type renderer: string
    @param renderer: Can be one of 'GLSL', 'OPENGL', or 'SDL'.

                     Due to way Python works you're unlikely to see much of an
                     improvement by using 'GLSL' or 'OPENGL' as most of the
                     time Python is slow interacting with the console and the
                     rendering itself is pretty fast even on 'SDL'.

                     This should be left at default or switched to 'SDL' for
                     better reliability and an instantaneous start up time.

    @rtype: L{Console}
    @return: The root console.  Only what is drawn on the root console is
             what's visible after a call to L{tdl.flush}.
             After the root console is garbage collected, the window made by
             this function will close.
    """
    RENDERERS = {'GLSL': 0, 'OPENGL': 1, 'SDL': 2}
    global _rootinitialized, _rootConsoleRef
    if not _fontinitialized: # set the default font to the one that comes with tdl
        setFont(_unpackfile('terminal.png'), 16, 16, colomn=True)

    if renderer.upper() not in RENDERERS:
        raise TDLError('No such render type "%s", expected one of "%s"' % (renderer, '", "'.join(RENDERERS)))
    renderer = RENDERERS[renderer.upper()]

    # If a console already exists then make a clone to replace it
    if _rootConsoleRef and _rootConsoleRef():
        oldroot = _rootConsoleRef()
        rootreplacement = Console(oldroot.width, oldroot.height)
        rootreplacement.blit(oldroot)
        oldroot._replace(rootreplacement)
        del rootreplacement
        
    if title is None: # use a default title
        if sys.argv:
            # Use the script filename as the title.
            title = os.path.basename(sys.argv[0])
        else:
            title = 'python-tdl'

    _lib.TCOD_console_init_root(width, height, _encodeString(title), fullscreen, renderer)

    #event.get() # flush the libtcod event queue to fix some issues
    # issues may be fixed already

    event._eventsflushed = False
    _rootinitialized = True
    rootconsole = Console._newConsole(ctypes.c_void_p())
    _rootConsoleRef = weakref.ref(rootconsole)

    return rootconsole

def flush():
    """Make all changes visible and update the screen.

    Remember to call this function after drawing operations.
    Calls to flush will enfore the frame rate limit set by L{tdl.setFPS}.

    This function can only be called after L{tdl.init}
    """
    if not _rootinitialized:
        raise TDLError('Cannot flush without first initializing with tdl.init')

    _lib.TCOD_console_flush()

def setFont(path, tileWidth, tileHeight, colomn=False,
            greyscale=False, altLayout=False):
    """Changes the font to be used for this session.
    This should be called before L{tdl.init}

    While it's possible you can change the font mid program it can sometimes
    break in rare circumstances.  So use caution when doing this.

    @type path: string
    @param path: Must be a string filepath where a bmp or png file is found.

    @type tileWidth: int
    @param tileWidth: The width of an individual tile.

    @type tileHeight: int
    @param tileHeight: The height of an individual tile.

    @type colomn: boolean
    @param colomn: Defines if the characer order goes along the rows or
                   colomns.
                   It should be True if the charater codes 0-15 are in the
                   first column.  And should be False if the characters 0-15
                   are in the first row.

    @type greyscale: boolean
    @param greyscale: Creates an anti-aliased font from a greyscale bitmap.
                      Otherwise it uses the alpha channel for anti-aliasing.

    @type altLayout: boolean
    @param altLayout: An alternative layout with space in the upper left
                      corner.  The colomn parameter is ignored if this is
                      True, find examples of this layout in the font/
                      directory included with the python-tdl source.

    @raise TDLError: Will be raised if no file is found at path.

    @note: A png file that's been optimized can fail to load correctly on
           MAC OS X creating a garbled mess when rendering.
           Don't use a program like optipng or just use bmp files instead if
           you want your program to work on macs.
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
    elif colomn:
        flags |= FONT_LAYOUT_ASCII_INCOL
    else:
        flags |= FONT_LAYOUT_ASCII_INROW
    if greyscale:
        flags |= FONT_TYPE_GREYSCALE
    if not os.path.exists(path):
        raise TDLError('no file exists at: "%s"' % path)
    _lib.TCOD_console_set_custom_font(_encodeString(path), flags, tileWidth, tileHeight)

def getFullscreen():
    """Returns True if program is fullscreen.

    @rtype: boolean
    @return: Returns True if the window is in fullscreen mode.
             Otherwise returns False.
    """
    if not _rootinitialized:
        raise TDLError('Initialize first with tdl.init')
    return _lib.TCOD_console_is_fullscreen()

def setFullscreen(fullscreen):
    """Changes the fullscreen state.

    @type fullscreen: boolean
    """
    if not _rootinitialized:
        raise TDLError('Initialize first with tdl.init')
    _lib.TCOD_console_set_fullscreen(fullscreen)

def setTitle(title):
    """Change the window title.

    @type title: string
    """
    if not _rootinitialized:
        raise TDLError('Not initilized.  Set title with tdl.init')
    _lib.TCOD_console_set_window_title(_encodeString(title))

def screenshot(path=None):
    """Capture the screen and save it as a png file

    @type path: string
    @param path: The filepath to save the screenshot.

                 If path is None then the image will be placed in the current
                 folder with the names:
                 screenshot001.png, screenshot002.png, ...
    """
    if not _rootinitialized:
        raise TDLError('Initialize first with tdl.init')
    if isinstance(fileobj, str):
        _lib.TCOD_sys_save_screenshot(_encodeString(fileobj))
    elif isinstance(fileobj, file): # save to temp file and copy to file-like obj
        tmpname = os.tempnam()
        _lib.TCOD_sys_save_screenshot(_encodeString(tmpname))
        with tmpname as tmpfile:
            fileobj.write(tmpfile.read())
        os.remove(tmpname)
    elif fileobj is None: # save to screenshot001.png, screenshot002.png, ...
        filelist = os.listdir('.')
        n = 1
        filename = 'screenshot%.3i.png' % n
        while filename in filelist:
            n += 1
            filename = 'screenshot%.4i.png' % n
        _lib.TCOD_sys_save_screenshot(_encodeString(filename))
    else:
        raise TypeError('fileobj is an invalid type: %s' % type(fileobj))

def setFPS(frameRate):
    """Set the maximum frame rate.

    @type frameRate: int
    @param frameRate: Further calls to L{tdl.flush} will limit the speed of
                      the program to run at <frameRate> frames per second. Can
                      also be set to 0 to run without a limit.

                      Defaults to None.
    """
    if frameRate is None:
        frameRate = 0
    assert isinstance(frameRate, _INTTYPES), 'frameRate must be an integer or None, got: %s' % repr(frameRate)
    _lib.TCOD_sys_set_fps(frameRate)

def getFPS():
    """Return the current frames per second of the running program set by
    L{setFPS}

    @rtype: int
    @return: Returns the frameRate set by setFPS.
             If set to no limit, this will return 0.
    """
    return _lib.TCOD_sys_get_fps()

def forceResolution(width, height):
    """Change the fullscreen resoulution

    @type width: int
    @type height: int
    """
    _lib.TCOD_sys_force_fullscreen_resolution(width, height)
        
__all__ = [_var for _var in locals().keys() if _var[0] != '_' and _var not in ['sys', 'os', 'ctypes', 'array', 'weakref', 'itertools', 'textwrap']]
__all__ += ['_MetaConsole'] # keep this object public to show the documentation in epydoc