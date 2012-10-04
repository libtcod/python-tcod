"""
    The documentation for python-tdl.  A Pythonic port of
    U{libtcod<http://doryen.eptalys.net/libtcod/>}.
    
    Getting Started
    ===============
      Once the library is imported you can load the font you want to use with
      L{tdl.setFont}.  This is optional and can be skipped to get a decent
      default font.  After that you call L{tdl.init} to set the size of the
      window and get the root console in return.  This console is the canvas
      to what will appear on the screen.
    
    Drawing
    =======
      Once you have the root console from L{tdl.init} you can start drawing on
      it using a method such as L{Console.drawChar}.  When using this method
      you can have the char parameter be an intiger or a single character
      string.  The fgcolor and bgcolor parameters expect a three item list
      [red, green, blue] with integers in the 0-255 range with [0, 0, 0] being
      black and [255, 255, 255] being white.  Or instead you can use None for
      any of the three parameters to tell the library to keep what is at that
      spot instead of overwriting it.  After the drawing functions are called
      a call to L{tdl.flush} will update the screen.
"""

import sys
import os
import ctypes
import weakref
import array
import itertools

from . import event
from .__tcod import _lib, _Color, _unpackfile

_IS_PYTHON3 = (sys.version_info[0] == 3)
_USE_FILL = False
'Set to True to use the libtcod fill optimization.  This is actually slower than the normal mode.'

def _format_string(string): # still used for filepaths, and that's about it
    "changes string into bytes if running in python 3, for sending to ctypes"
    if _IS_PYTHON3 and isinstance(string, str):
        return string.encode()
    return string

#def _encodeString(string):
#    pass

def _formatChar(char):
    """Prepares a single character for passing to ctypes calls, needs to return
    an integer but can also pass None which will keep the current character
    instead of overrwriting it.
    
    This is called often and needs to be optimized whenever possible.
    """
    if char is None:
        return None
    if isinstance(char, int) or not _IS_PYTHON3 and isinstance(char, long):
        return char
    if isinstance(char, (str, bytes)) and len(char) == 1:
        return ord(char)
    raise TypeError('Expected char parameter to be a single character string, number, or None, got: %s' % repr(char))

_fontinitialized = False
_rootinitialized = False
_rootconsole = None
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
    if isinstance(color, int) or not _IS_PYTHON3 and isinstance(color, long):
        return True
    return False

def _formatColor(color):
    """Format the color to ctypes
    """
    if color is None:
        return None
    # avoid isinstance, checking __class__ gives a small speed increase
    if color.__class__ is _Color:
        return color
    if isinstance(color, int) or not _IS_PYTHON3 and isinstance(color, long):
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
    __slots__ = ('width', 'height', '__weakref__', '__dict__')
    
    def drawChar(self, x, y, char, fgcolor=(255, 255, 255), bgcolor=(0, 0, 0)):
        """Draws a single character.

        @type x: int
        @type y: int
        @type char: int, string, or None
        @type fgcolor: 3-item list or None
        @type bgcolor: 3-item list or None
        @param char: Should be an integer, single character string, or None.
        
                     You can set the char parameter as None if you only want to change
                     the colors of the tile.

        @param fgcolor: For fgcolor and bgcolor you use a 3 item list with integers ranging 0 - 255 or None.
                        None will keep the current color at this position unchanged.
        
        
        @raise AssertionError: Having the x or y values outside of the console will raise an
                               AssertionError.  You can use ((x, y) in console)
                               to check if a cell is drawable.
        """
        
        assert _verify_colors(fgcolor, bgcolor)
        assert self._drawable(x, y)
        
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

        fgcolor and bgcolor can be set to None to keep the colors unchanged.
        
        @type x: int
        @type y: int
        @type string: string or iterable
        @type fgcolor: 3-item list or None
        @type bgcolor: 3-item list or None
        
        """
        
        assert self._drawable(x, y)
        assert _verify_colors(fgcolor, bgcolor)
        fgcolor, bgcolor = _formatColor(fgcolor), _formatColor(bgcolor)
        width, height = self.getSize()
        for char in string:
            if y == height:
                raise TDLError('End of console reached.')
            self._setChar(x, y, _formatChar(char), fgcolor, bgcolor)
            x += 1 # advance cursor
            if x == width: # line break
                x = 0
                y += 1
    
    def drawRect(self, x, y, width, height, string, fgcolor=(255, 255, 255), bgcolor=(0, 0, 0)):
        """Draws a rectangle starting from x and y and extending to width and
        height.  If width or height are None then it will extend to the edge
        of the console.  The rest are the same as drawChar.
        
        @type x: int
        @type y: int
        @type width: int or None
        @type height: int or None
        @type string: int, string, or None
        @type fgcolor: 3-item list or None
        @type bgcolor: 3-item list or None
        """
        x, y, width, height = self._normalizeRect(x, y, width, height)
        assert _verify_colors(fgcolor, bgcolor)
        fgcolor, bgcolor = _formatColor(fgcolor), _formatColor(bgcolor)
        char = _formatChar(string)
        for cellY in range(y, y + height):
            for cellX in range(x, x + width):
                self._setChar(cellX, cellY, char, fgcolor, bgcolor)
        
    def drawFrame(self, x, y, width, height, string, fgcolor=(255, 255, 255), bgcolor=(0, 0, 0)):
        """Similar to drawRect but only draws the outline of the rectangle.
        
        @type x: int
        @type y: int
        @type width: int or None
        @type height: int or None
        @type string: int, string, or None
        @type fgcolor: 3-item list or None
        @type bgcolor: 3-item list or None
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
    
    
    def _normalizeRect(self, x, y, width, height):
        """Check if the rectangle is in bounds and make minor adjustments.
        raise AssertionError's for any problems
        """
        old = x, y, width, height
        assert isinstance(x, int), 'x must be an integer, got %s' % repr(x)
        assert isinstance(y, int), 'y must be an integer, got %s' % repr(y)
        if width == None: # if width or height are None then extend them to the edge
            width = self.width - x
        if height == None:
            height = self.height - y
        assert isinstance(width, int), 'width must be an integer or None, got %s' % repr(width)
        assert isinstance(height, int), 'height must be an integer or None, got %s' % repr(height)
        
        assert width >= 0 and height >= 0, 'width and height cannot be negative'
        # later idea, negative numbers work like Python list indexing
        
        assert x >= 0 and y >= 0 and x + width <= self.width and y + height <= self.height, \
        'Rect is out of bounds at (x=%i y=%i width=%i height=%i), Console bounds are (width=%i, height=%i)' % (old + self.getSize())
        return x, y, width, height
    
    def _rectInBounds(self, x, y, width, height):
        "check the rect so see if it's within the bounds of this console"
        if width is None:
            width = 0
        if height is None:
            height = 0
        if (x < 0 or y < 0 or
            x + width > self.width or y + height > self.height):
            return False
            
        return True
    
    def _clampRect(self, x=0, y=0, width=None, height=None):
        """Alter a rectange to fit inside of this console
        
        width and hight of None will extend to the edge and
        an area out of bounds will end with a width and height of 0
        """
        # extend any width and height of None to the end of the console
        if width is None:
            width = self.width - x
        if height is None:
            height = self.height - y
        # move x and y within bounds, shrinking the width and height to match
        if x < 0:
            width += x
            x = 0
        if y < 0:
            height += y
            y = 0
        # move width and height within bounds
        width = min(width, self.width - x)
        height = min(height, self.height - y)
        # a rect that was out of bounds will have a 0 or negative width or height at this point
        if width <= 0 or height <= 0:
            width = height = 0
        return x, y, width, height
    
    def blit(self, source, x=0, y=0, width=None, height=None, srcX=0, srcY=0):
        """Blit another console or Window onto the current console.

        By default it blits the entire source to the topleft corner.
        
        @type source: Console or Window
        @type x: int
        @type y: int
        @type width: int or None
        @type height: int or None
        @type srcX: int
        @type srcY: int
        """
        # hardcode alpha settings for now
        fgalpha=1.0
        bgalpha=1.0
        
        assert isinstance(source, (Console, Window)), "source muse be a Window or Console instance"
        
        assert width is None or isinstance(width, (int)), "width must be a number or None, got %s" % repr(width)
        assert height is None or isinstance(height, (int)), "height must be a number or None, got %s" % repr(height)
        
        # fill in width, height
        if width == None:
            width = min(self.width - x, source.width - srcX)
        if height == None:
            height = min(self.height - y, source.height - srcY)
        
        x, y, width, height = self._normalizeRect(x, y, width, height)
        srcX, srcY, width, height = source._normalizeRect(srcX, srcY, width, height)
        
        # translate source and self if any of them are Window instances
        if isinstance(source, Window):
            srcX, srcY = source._translate(srcX, srcY)
            source = source.console
        
        if isinstance(self, Window):
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
        @type y: int
        """
        assert isinstance(x, int), "x must be an integer, got %s" % repr(x)
        assert isinstance(y, int), "y must be an integer, got %s" % repr(x)
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
        
    def __contains__(self, position):
        """Use ((x, y) in console) to check if a position is drawable on this console.
        """
        x, y = position
        return (0 <= x < self.width) and (0 <= y < self.height)
        
    def _drawable(self, x, y):
        """Used internally
        Checks if a cell is part of the console.
        Raises an AssertionError if it can not be used.
        """
        assert isinstance(x, int), 'x must be an integer, got %s' % repr(x)
        assert isinstance(y, int), 'y must be an integer, got %s' % repr(y)
        
        assert (0 <= x < self.width) and (0 <= y < self.height), \
                ('(%i, %i) is an invalid postition.  %s size is (%i, %i)' %
                 (x, y, self.__class__.__name__, self.width, self.height))
        return True

class Console(_MetaConsole):
    """The Console is the main class of the tdl library.

    The console created by the L{tdl.init} function is the root console and is the
    consle that is rendered to the screen with flush.

    Any console made from Console is an off-screen console that can be drawn
    on and then L{blit} to the root console.
    """

    __slots__ = ('_as_parameter_',)

    def __init__(self, width, height):
        """Create a new offscreen console
        """
        self._as_parameter_ = _lib.TCOD_console_new(width, height)
        self.width = width
        self.height = height
        self._initArrays()
        #self.clear()
        
    @classmethod
    def _newConsole(cls, console):
        """Make a Console instance, from a console ctype"""
        self = cls.__new__(cls)
        self._as_parameter_ = console
        self.width = _lib.TCOD_console_get_width(self)
        self.height = _lib.TCOD_console_get_height(self)
        self._initArrays()
        #self.clear()
        return self
        
    def _initArrays(self):
        if not _USE_FILL:
            return
        # used for the libtcod fill optimization
        IntArray = ctypes.c_int * (self.width * self.height)
        self.chArray = IntArray()
        self.fgArrays = (IntArray(),
                         IntArray(),
                         IntArray())
        self.bgArrays = (IntArray(),
                         IntArray(),
                         IntArray())
        
    def __del__(self):
        """
        If the main console is garbage collected then the window will be closed as well
        """
        # If this is the root console the window will close when collected
        try:
            if isinstance(self._as_parameter_, ctypes.c_void_p):
                global _rootinitialized, _rootconsole
                _rootinitialized = False
                _rootconsole = None
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
        
    def clear(self, fgcolor=(255, 255, 255), bgcolor=(0, 0, 0)):
        """Clears the entire console.
        
        @type fgcolor: 3-item list
        @type bgcolor: 3-item list
        """
        assert _verify_colors(fgcolor, bgcolor)
        assert fgcolor and bgcolor, 'Can not use None with clear'
        _lib.TCOD_console_set_default_background(self, _formatColor(bgcolor))
        _lib.TCOD_console_set_default_foreground(self, _formatColor(fgcolor))
        _lib.TCOD_console_clear(self)
    
    def _setCharFill(self, x, y, char, fgcolor=None, bgcolor=None):
        """An optimized version using the fill wrappers that didn't work out to be any faster"""
        index = x + y * self.width
        self.chArray[index] = char
        for channel, color in zip(itertools.chain(self.fgArrays, self.bgArrays),
                                  itertools.chain(fgcolor, bgcolor)):
            channel[index] = color

    def _setCharCall(self, x, y, char, fgcolor=None, bgcolor=None, bgblend=1):
        """
        Sets a character.
        This is called often and is designed to be as fast as possible.
        
        Because of the need for speed this function will do NO TYPE CHECKING
        AT ALL, it's up to the drawing functions to use the functions:
        _formatChar and _formatColor before passing to this."""
        if char is not None and fgcolor is not None and bgcolor is not None:
            return _setcharEX(self, x, y, char, fgcolor, bgcolor)
        if char is not None:
            _setchar(self, x, y, char)
        if fgcolor is not None:
            _setfore(self, x, y, fgcolor)
        if bgcolor is not None:
            _setback(self, x, y, bgcolor, bgblend)
            
    if _USE_FILL:
        _setChar = _setCharFill
    else:
        _setChar = _setCharCall

    def getChar(self, x, y):
        """Return the character and colors of a cell as
        (char, fgcolor, bgcolor)

        The charecter is returned as a number.
        Each color is returned as a tuple.
        
        @rtype: (int, 3-item tuple, 3-item tuple)
        """
        self._drawable(x, y)
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

    __slots__ = ('console', 'parent', 'x', 'y')

    def __init__(self, console, x, y, width, height):
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

    def clear(self, fgcolor=(255, 255, 255), bgcolor=(0, 0, 0)):
        """Clears the entire Window.
        
        @type fgcolor: 3-item list
        @type bgcolor: 3-item list
        """
        assert _verify_colors(fgcolor, bgcolor)
        assert fgcolor and bgcolor, 'Can not use None with clear'
        self.draw_rect(0, 0, None, None, 0x20, fgcolor, bgcolor)

    def _setChar(self, x, y, char=None, fgcolor=None, bgcolor=None, bgblend=1):
        self.parent._setChar((x + self.x), (y + self.y), char, fgcolor, bgcolor, bgblend)
    
    def getChar(self, x, y):
        """Return the character and colors of a cell as (ch, fg, bg)
        
        @rtype: (int, 3-item tuple, 3-item tuple)
        """
        self._drawable(x, y)
        return self.console.getChar(self._translate(x, y))

    def __repr__(self):
        return "<Window(X=%i Y=%i Width=%i Height=%i)>" % (self.x, self.y,
                                                          self.width,
                                                          self.height)


def init(width, height, title='python-tdl', fullscreen=False, renderer='OPENGL'):
    """Start the main console with the given width and height and return the
    root console.

    Call the consoles drawing functions.  Then remember to use L{tdl.flush} to
    make what's drawn visible on the console.

    @type width: int
    @type height: int
    
    @type title: string
    
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
    global _rootinitialized, _rootconsole
    if not _fontinitialized: # set the default font to the one that comes with tdl
        setFont(_unpackfile('terminal.png'), 16, 16, colomn=True)

    if renderer.upper() not in RENDERERS:
        raise TDLError('No such render type "%s", expected one of "%s"' % (renderer, '", "'.join(RENDERERS)))
    renderer = RENDERERS[renderer.upper()]
    
    # If a console already exists then make a clone to replace it
    if _rootconsole is not None:
        oldroot = _rootconsole()
        rootreplacement = Console(oldroot.width, oldroot.height)
        rootreplacement.blit(oldroot)
        oldroot._replace(rootreplacement)
        del rootreplacement

    _lib.TCOD_console_init_root(width, height, _format_string(title), fullscreen, renderer)

    #event.get() # flush the libtcod event queue to fix some issues
    # issues may be fixed already

    event._eventsflushed = False
    _rootinitialized = True
    rootconsole = Console._newConsole(ctypes.c_void_p())
    _rootconsole = weakref.ref(rootconsole)

    return rootconsole

def flush():
    """Make all changes visible and update the screen.
    
    Remember to call this function after drawing operations.
    Calls to flush will enfore the frame rate limit set by L{tdl.setFPS}.
    
    This function can only be called after L{tdl.init}
    """
    if not _rootinitialized:
        raise TDLError('Cannot flush without first initializing with tdl.init')
        
    if _USE_FILL:
        console = _rootconsole()
        _lib.TCOD_console_fill_background(console, *console.bgArrays)
        _lib.TCOD_console_fill_foreground(console, *console.fgArrays)
        _lib.TCOD_console_fill_char(console, console.chArray)
        
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
    _lib.TCOD_console_set_custom_font(_format_string(path), flags, tileWidth, tileHeight)

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
    _lib.TCOD_console_set_window_title(_format_string(title))

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
        _lib.TCOD_sys_save_screenshot(_format_string(fileobj))
    elif isinstance(fileobj, file): # save to temp file and copy to file-like obj
        tmpname = os.tempnam()
        _lib.TCOD_sys_save_screenshot(_format_string(tmpname))
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
        _lib.TCOD_sys_save_screenshot(_format_string(filename))
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
    assert isinstance(frameRate, int), 'frameRate must be an integer or None, got: %s' % repr(frameRate)
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

__all__ = [_var for _var in locals().keys() if _var[0] != '_' and _var not in ['sys', 'os', 'ctypes', 'array', 'weakref', 'itertools']]
__all__ += ['_MetaConsole'] # keep this object public to show the documentation in epydoc
