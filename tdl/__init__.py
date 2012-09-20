"""
    
"""

import sys
import os
import ctypes
import weakref
import array

from . import event
from .tcod import _lib, _Color, _unpackfile

_IS_PYTHON3 = (sys.version_info[0] == 3)
#_encoding = 'cp437'

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
    if color.__class__ is int or not _IS_PYTHON3 and color.__class__ is long:
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
    if color.__class__ is int or not _IS_PYTHON3 and color.__class__ is long:
        # format a web style color with the format 0xRRGGBB
        return _Color(color >> 16 & 0xff, color >> 8 & 0xff, color & 0xff)
    return _Color(*color)

class TDLError(Exception):
    """
    The catch all for most TDL specific errors.
    """

class _MetaConsole(object):
    """
    Contains methods shared by both the Console and Window classes.
    """
    __slots__ = ('width', 'height', '__weakref__', '__dict__')
    
    def drawChar(self, x, y, char=None, fgcolor=(255, 255, 255), bgcolor=(0, 0, 0)):
        """Draws a single character.

        char should be an integer, single character string, or None
        you can set the char parameter as None if you only want to change
        the colors of the tile.

        For fgcolor and bgcolor you use a 3 item list or None.  None will
        keep the current color at this position unchanged.
        

        Having the x or y values outside of the console will raise an
        AssertionError.
        """
        
        assert _verify_colors(fgcolor, bgcolor)
        assert self._drawable(x, y)
        
        self._setChar(x, y, _formatChar(char),
                      _formatColor(fgcolor), _formatColor(bgcolor))

    def drawStr(self, x, y, string, fgcolor=(255, 255, 255), bgcolor=(0, 0, 0)):
        """Draws a string starting at x and y.  Optinally colored.

        A string that goes past the right side will wrap around.  A string
        wraping to below the console will raise a TDLError but will still be
        written out.  This means you can safely ignore the errors with a
        try... except block if you're fine with partily written strings.

        \\r and \\n are drawn on the console as normal character tiles.  No
        special encoding is done and any string will translate to the character
        table as is.

        fgcolor and bgcolor can be set to None to keep the colors unchanged.
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
    
    def drawRect(self, x, y, width, height, string=None, fgcolor=(255, 255, 255), bgcolor=(0, 0, 0)):
        """Draws a rectangle starting from x and y and extending to width and
        height.  If width or height are None then it will extend to the edge
        of the console.  The rest are the same as drawChar.
        """
        x, y, width, height = self._normalizeRect(x, y, width, height)
        assert _verify_colors(fgcolor, bgcolor)
        fgcolor, bgcolor = _formatColor(fgcolor), _formatColor(bgcolor)
        char = _formatChar(string)
        for cellY in range(y, y + height):
            for cellX in range(x, x + width):
                self._setChar(cellX, cellY, char, fgcolor, bgcolor)
        
    def drawFrame(self, x, y, width, height, string=None, fgcolor=(255, 255, 255), bgcolor=(0, 0, 0)):
        "Similar to drawRect but only draws the outline of the rectangle"
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
    
    def blit(self, source, x=0, y=0, width=None, height=None, srcx=0, srcy=0):
        """Blit another console or Window onto the current console.

        By default it blits the entire source to the topleft corner.
        
        If nothing is blited then TDLError is raised
        """
        # hardcode alpha settings for now
        fgalpha=1.0
        bgalpha=1.0
        
        assert isinstance(source, (Console, Window)), "source muse be a Window or Console instance"
        
        assert width is None or isinstance(width, (int)), "width must be a number or None, got %s" % repr(width)
        assert height is None or isinstance(height, (int)), "height must be a number or None, got %s" % repr(height)
        
        # fill in width, height
        if width == None:
            width = min(self.width - x, source.width - srcx)
        if height == None:
            height = min(self.height - y, source.height - srcy)
        
        x, y, width, height = self._normalizeRect(x, y, width, height)
        srcx, srcy, width, height = source._normalizeRect(srcx, srcy, width, height)
        
        # translate source and self if any of them are Window instances
        if isinstance(source, Window):
            srcx, srcy = source._translate(srcx, srcy)
            source = source.console
        
        if isinstance(self, Window):
            x, y = self._translate(x, y)
            self = self.console
        
        if self == source:
            # if we are the same console then we need a third console to hold
            # onto the data, otherwise it tries to copy into itself and
            # starts destroying everything
            tmp = Console(width, height)
            _lib.TCOD_console_blit(source, srcx, srcy, width, height, tmp, 0, 0, fgalpha, bgalpha)
            _lib.TCOD_console_blit(tmp, 0, 0, width, height, self, x, y, fgalpha, bgalpha)
        else:
            _lib.TCOD_console_blit(source, srcx, srcy, width, height, self, x, y, fgalpha, bgalpha)

    def getSize(self):
        """Return the size of the console as (width, height)
        """
        return self.width, self.height
        
    def scroll(self, x, y):
        """Scroll the contents of the console in the direction of x,y.
        
        Uncovered areas will be cleared.
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
        
        # But first we need to blit so we have an area to clear.
        x, width, srcx = getSlide(x, width)
        y, height, srcy = getSlide(y, height)
        self.blit(self, x, y, width, height, srcx, srcy)
        
        # let's do this
        if uncoverX: # clear sides (0x20 is space)
            self.drawRect(uncoverX[0], coverY[0], uncoverX[1], coverY[1], 0x20)
        if uncoverY: # clear top/bottom
            self.drawRect(coverX[0], uncoverY[0], coverX[1], uncoverY[1], 0x20)
        if uncoverX and uncoverY: # clear corner
            self.drawRect(uncoverX[0], uncoverY[0], uncoverX[1], uncoverY[1], 0x20)
        
        # you know, now that I think about it.  I could of just copied it
        # into another console instance and cleared the whole thing.
        # not only would that have been a better idea.  It would of been
        # faster too. (but only faster for Console's)
        
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

    The console created by the init function is the root console and is the
    consle that is rendered to the screen with flush.

    Any console made from Console is an off-screen console that can be drawn
    on and then blited to the root console.
    """

    __slots__ = ('_as_parameter_',)

    def __init__(self, width, height):
        self._as_parameter_ = _lib.TCOD_console_new(width, height)
        self.width = width
        self.height = height
        self.clear()

    @classmethod
    def _newConsole(cls, console):
        """Make a Console instance, from a console ctype"""
        self = cls.__new__(cls)
        self._as_parameter_ = console
        self.width = _lib.TCOD_console_get_width(self)
        self.height = _lib.TCOD_console_get_height(self)
        self.cursorX = self.cursorY = 0
        self.clear()
        return self
        
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
            pass

    def _replace(self, console):
        """Used internally
        
        Mostly used just to replace this Console object with the root console
        If another Console object is used then they are swapped
        
        Soon to be removed and replaced by the _newConsole method
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
        """
        assert _verify_colors(fgcolor, bgcolor)
        _lib.TCOD_console_set_default_background(self, _formatColor(bgcolor))
        _lib.TCOD_console_set_default_foreground(self, _formatColor(fgcolor))
        _lib.TCOD_console_clear(self)
    
    def _setChar(self, x, y, char, fgcolor=None, bgcolor=None, bgblend=1):
        """
        Sets a character without moving the virtual cursor, this is called
        often and is designed to be as fast as possible.
        
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

    def getChar(self, x, y):
        """Return the character and colors of a cell as (ch, fg, bg)

        The charecter is returned as a number.
        each color is returned as a tuple
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
        """
        assert _verify_colors(fgcolor, bgcolor)
        self.draw_rect(0, 0, None, None, 0x20, fgcolor, bgcolor)

    def _setChar(self, x, y, char=None, fgcolor=None, bgcolor=None, bgblend=1):
        self.parent._setChar((x + self.x), (y + self.y), char, fgcolor, bgcolor, bgblend)
    
    def getChar(self, x, y):
        """Return the character and colors of a cell as (ch, fg, bg)
        """
        self._drawable(x, y)
        return self.console.getChar(self._translate(x, y))

    def __repr__(self):
        return "<Window(X=%i Y=%i Width=%i Height=%i)>" % (self.x, self.y,
                                                          self.width,
                                                          self.height)


def init(width, height, title='TDL', fullscreen=False, renderer='OPENGL'):
    """Start the main console with the given width and height and return the
    root console.

    Remember to use tdl.flush() to make what's drawn visible on the console.

    After the root console is garbage collected, the window made by this
    function will close.
    
    renderer can be one of 'GLSL', 'OPENGL', or 'SDL'
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
    """
    if not _rootinitialized:
        raise TDLError('Cannot flush without first initializing with tdl.init')
    
    # old hack to prevent locking up on old libtcod
    # you can probably delete all autoflush releated stuff
    #if event._autoflush and not event._eventsflushed:
    #    event.get()
    #else: # do not flush events after the user starts using them
    #    event._autoflush = False
    
    #event._eventsflushed = False
    _lib.TCOD_console_flush()

def setFont(path, tileWidth, tileHeight, colomn=False, greyscale=False, altLayout=False):
    """Changes the font to be used for this session
    This should be called before tdl.init

    path - must be a string for where a bitmap file is found.

    tileWidth, tileHeight - is the size of an individual tile.

    colomn - defines if the characer order goes along the rows or colomns.  It
    should be True if the codes are 0-15 in the first column.  And should be
    False if the codes are 0-15 in the first row.
    
    greyscale - creates an anti-aliased font from a greyscale bitmap.
    Unnecessary when a font has an alpha channel for anti-aliasing.
    
    altLayout - a alternative layout with space in the upper left corner.  The
    colomn parameter is ignored if this is True, find examples of this layout
    in the font/ directory included with the TDL source.
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
    """Returns if the window is fullscreen

    The user can't make the window fullscreen so you must use the
    setFullscreen function
    """
    if not _rootinitialized:
        raise TDLError('Initialize first with tdl.init')
    return _lib.TCOD_console_is_fullscreen()

def setFullscreen(fullscreen):
    """Sets the fullscreen state to the boolen value
    """
    if not _rootinitialized:
        raise TDLError('Initialize first with tdl.init')
    _lib.TCOD_console_set_fullscreen(fullscreen)

def setTitle(title):
    """Change the window title.
    """
    if not _rootinitialized:
        raise TDLError('Not initilized.  Set title with tdl.init')
    _lib.TCOD_console_set_window_title(_format_string(title))

def screenshot(file=None):
    """Capture the screen and place it in file.

    file can be a file-like object or a filepath to save the screenshot
    if file is none then file will be placed in the current folder with
    the names: screenshot001.png, screenshot002.png, ...
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
    """Set the frame rate.

    You can set this to have no limit by using 0.
    """
    if frameRate is None:
        frameRate = 0
    assert isinstance(frameRate, int), 'frameRate must be an integer or None, got: %s' % repr(frameRate)
    _lib.TCOD_sys_set_fps(frameRate)

def getFPS():
    """Return the current frames per second of the running program.
    """
    return _lib.TCOD_sys_get_fps()

def forceResolution(width, height):
    """Change the fullscreen resoulution
    """
    _lib.TCOD_sys_force_fullscreen_resolution(width, height)

__all__ = [var for var in locals().keys() if not '_' in var[0]]

