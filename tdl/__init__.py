"""
    tdl is a ctypes port of The Doryen Library.
"""

import sys
import os
import ctypes
import weakref
import array

from . import event
from .tcod import _lib, _Color, _unpackfile
from .local import *

_IS_PYTHON3 = (sys.version_info[0] == 3)
def _format_string(string):
    "changes string into bytes if running in python 3, for sending to ctypes"
    if _IS_PYTHON3 and isinstance(string, str):
        return string.encode()
    return string

def _formatChar(char):
    """Prepares a single character for passing to ctypes calls, needs to return
    an integer but can also pass None which will keep the current character
    instead of overrwriting it.
    
    This is called often and needs to be optimized whenever possible.
    """
    if char is None:
        return None
    if isinstance(char, (str, bytes)) and len(char) == 1:
        return ord(char)
    if isinstance(char, int) or not _IS_PYTHON3 and isinstance(char, long):
        return char
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
        assert _iscolor(color), 'a color must be a 3 item tuple, web format, or None, received %s' % repr()
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
    if color.__class__ is int:
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
    if color.__class__ is int:
        # format a web style color with the format 0xRRGGBB
        return _Color(color >> 16 & 0xff, color >> 8 & 0xff, color & 0xff)
    return _Color(*color)

class TDLError(Exception):
    """
    The catch all for most TDL specific errors.
    """

#class TDLDrawError(TDLError):
#    pass

#class TDLBlitError(TDLError):
#    pass

#class TDLIndexError(TDLError):
#    pass

class Console(object):
    """The Console is the main class of the tdl library.

    The console created by the init function is the root console and is the
    consle that is rendered to the screen with flush.

    Any console made from Console is an off screen console that must be blited
    to the root console to be visisble.
    """

    __slots__ = ('_as_parameter_', 'width', 'height', '__weakref__', '__dict__')

    def __init__(self, width, height):
        self._as_parameter_ = _lib.TCOD_console_new(width, height)
        self.width = width
        self.height = height
        self.cursorX = self.cursorY = 0
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
    
    def _rectInBounds(self, x, y, width, height):
        "check the rect so see if it's within the bounds of this console"
        if width is None:
            width = 0
        if height is None:
            height = 0
        if x < 0 or y < 0:
            return False
        if x + width > self.width or y + height > self.height:
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
        
        # if None is used, get the cursor position
        x, y = self._cursor_get(x, y)
        srcx, srcy = source._cursor_get(srcx, srcy)

        assert width is None or isinstance(width, (int)), "width must be a number or None"
        assert height is None or isinstance(height, (int)), "height must be a number or None"
        
        if not self._rectInBounds(srcx, srcy, width, height):
            raise TDLError('Source is out of bounds')
        if not self._rectInBounds(x, y, width, height):
            raise TDLError('Destination is out of bounds')
        
        x, y, width, height = self._clampRect(x, y, width, height)
        srcx, srcy, width, height = source._clampRect(srcx, srcy, width, height)
        

        # translate source and self if any of them are Window instances
        if isinstance(source, Window):
            source._translate(srcx, srcy)
            source = source.console
        
        if isinstance(self, Window):
            self._translate(x, y)
            self = self.console
        
        _lib.TCOD_console_blit(source, srcx, srcy, width, height, self, x, y, fgalpha, bgalpha)

    def getSize(self):
        """Return the size of the console as (width, height)
        """
        return self.width, self.height

    def _cursor_normalize(self):
        if self.cursorX >= self.width:
            self.cursorX = 0
            self.cursorY += 1
            if self.cursorY >= self.height:
                self.cursorY = 0
        
    def _cursor_advance(self, rate=1):
        "Move the virtual cursor forward, one step by default"
        self.cursorX += rate
    
    def _cursor_newline(self):
        "Move the cursor down and set x=0"
        self.cursorX = 0
        self.cursorY += 1
        self._cursor_normalize()
    
    def _cursor_get(self, x=None, y=None):
        "Fill in blanks with the cursor position"
        if x is None or y is None:
            self._cursor_normalize() # make sure cursor is valid first
        if x is None:
            x = self.cursorX
        if y is None:
            y = self.cursorY
        return x, y
    
    def _cursor_move(self, x=None, y=None):
        "Changes the cursor position, checks if the position is valid, and returns the new position"
        x, y = self._cursor_get(x, y)
        # replace None with cursor position
        # check if position is valid, raise an error if not
        self._drawable(x, y)
        # change cursor position
        self.cursorX, self.cursorY = x, y
        # return new position
        return x, y
        
    def _drawable(self, x, y):
        """Used internally
        Checks if a cell is part of the console.
        Raises an exception if it can not be used.
        """
        #if x is not None and not isinstance(x, int):
        #    raise TypeError('x must be an integer, got %s' % repr(x))
        #if y is not None and not isinstance(y, int):
        #    raise TypeError('y must be an integer, got %s' % repr(y))
        
        if (0 <= x < self.width) and (0 <= y < self.height):
            return True
        raise TDLError('(%i, %i) is an invalid postition.  %s size is (%i, %i)' %
                            (x, y, self.__class__.__name__, self.width, self.height))

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
    
    
    def _setChar(self, x, y, char, fgcolor=None, bgcolor=None, bgblend=BND_SET):
        """Sets a character without moving the virtual cursor, this is called often
        and is designed to be as fast as possible"""
        if char is not None and fgcolor is not None and bgcolor is not None:
            return _setcharEX(self, x, y, _formatChar(char), _formatColor(fgcolor), _formatColor(bgcolor))
        if char is not None:
            _setchar(self, x, y, _formatChar(char))
        if fgcolor is not None:
            _setfore(self, x, y, _formatColor(fgcolor))
        if bgcolor is not None:
            _setback(self, x, y, _formatColor(bgcolor), bgblend)
    
    def drawChar(self, x, y, char=None, fgcolor=(255, 255, 255), bgcolor=(0, 0, 0)):
        """Draws a single character.

        char should be an integer, single character string, or None
        you can set the char parameter as None if you only want to change
        the colors of the tile.

        For fgcolor and bgcolor you use a 3 item list or None
        None will keep the color unchanged.

        Having the x or y values outside of the console will raise a TDLError.
        """
        # hardcode alpha settings for now
        #bgblend=BND_SET
        
        assert _verify_colors(fgcolor, bgcolor)
        x, y = self._cursor_move(x, y)
        
        self._setChar(x, y, char, fgcolor, bgcolor)
        #self._cursor_advance()

    def drawStr(self, x, y, string, fgcolor=(255, 255, 255), bgcolor=(0, 0, 0)):
        """Draws a string starting at x and y.

        A string that goes past the end will wrap around.  No warning will be
        made if it reaches the end of the console.

        \\r and \\n are drawn on the console as normal character tiles.

        For fgcolor and bgcolor, None will keep the color unchanged.

        If a large enough tileset is loaded you can use a unicode string.
        """
        # hardcode alpha settings for now
        bgblend=BND_SET
        
        #x, y = self._cursor_move(x, y)
        assert _verify_colors(fgcolor, bgcolor)
        fgcolor, bgcolor = _formatColor(fgcolor), _formatColor(bgcolor)
        width, height = self.getSize()
        for char in string:
            self._setChar(x, y, char, fgcolor, bgcolor)
            x += 1 # advance cursor
            if x == width: # line break
                x = 0
                y += 1
                if y == height:
                    return # just clip the rest of the string and exit
    
    def drawRect(self, x, y, width, height, string=None, fgcolor=(255, 255, 255), bgcolor=(0, 0, 0)):
        """Draws a rectangle starting from x and y and extending to width and
        height.  If width or height are None then it will extend to the edge
        of the console.  The rest are the same as drawChar.
        """
        # hardcode alpha settings for now
        bgblend=BND_SET
        
        ## replace None with cursor position
        #x, y = self._cursor_get(x, y)
        ## clamp rect to bounds
        #if (width, height) == (0, 0):
        #    raise TDLError('Rectange is out of bounds at (%i, %i), bounds are (%i, %i)' % ((x, y) + self.getSize()))
        if not self._rectInBounds(x, y, width, height):
            raise TDLError('Rectange is out of bounds at (x=%i, y=%i, width=%s, height=%s), bounds are (%i, %i)' %
                           ((x, y, width, height) + self.getSize()))
        x, y, width, height = self._clampRect(x, y, width, height) # fill in width height
        _verify_colors(fgcolor, bgcolor)
        fgcolor, bgcolor = _formatColor(fgcolor), _formatColor(bgcolor)
        for cellY in range(y, y + height):
            for cellX in range(x, x + width):
                self._setChar(cellX, cellY, string, fgcolor, bgcolor, bgblend)
        
    def drawFrame(self, x, y, width, height, string=None, fgcolor=(255, 255, 255), bgcolor=(0, 0, 0)):
        "Similar to drawRect but only draws the outline of the rectangle"
        # hardcode alpha settings for now
        bgblend=BND_SET
        
        #x, y = self._cursor_get(x, y)
        #x, y, width, height = self._clampRect(x, y, width, height)
        #if (width, height) == (0, 0):
        #    raise TDLError('Rectange is out of bounds at (%i, %i), bounds are (%i, %i)' % ((x, y) + self.getSize()))
        if not self._rectInBounds(x, y, width, height):
            raise TDLError('Frame is out of bounds at (x=%i, y=%i, width=%i, height=%i), bounds are (width=%i, height=%i)' %
                           ((x, y, width, height) + self.getSize()))
        
        x, y, width, height = self._clampRect(x, y, width, height) # fill in width height
        _verify_colors(fgcolor, bgcolor)
        fgcolor, bgcolor = _formatColor(fgcolor), _formatColor(bgcolor)
        if width == 1 or height == 1: # it's just a single width line here
            self.drawRect(x, y, width, height, string, fgcolor, bgcolor, bgblend)
            return
        # draw frame with drawRect
        self.drawRect(x, y, 1, height, string, fgcolor, bgcolor)
        self.drawRect(x, y, width, 1, string, fgcolor, bgcolor)
        self.drawRect(x + width - 1, y, 1, height, string, fgcolor, bgcolor)
        self.drawRect(x, y + height - 1, width, 1, string, fgcolor, bgcolor)

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


class Window(object):
    """A Window contains a small isolated part of a Console.

    Drawing on the Window draws on the Console.  This works both ways.

    If you make a Window without setting its size (or set the width and height
    as None) it will extend to the edge of the console.

    You can't blit Window instances but drawing works as expected.
    """

    __slots__ = ('console', 'parent', 'x', 'y', 'width', 'height', '__weakref__', '__dict__')

    def __init__(self, console, x, y, width, height):
        if not isinstance(console, (Console, Window)):
            raise TypeError('console parameter must be a Console or Window instance, got %s' % repr(console))
        assert isinstance(x, int), 'expeted integer got: %s' % repr(x)
        assert isinstance(y, int), 'expeted integer got: %s' % repr(y)
        assert (isinstance(width, int) or width is None), 'expeted integer or None got: %s' % repr(width)
        assert (isinstance(height, int) or height is None), 'expeted integer or None got: %s' % repr(height)
        if not console._rectInBounds(x, y, width, height):
            raise TDLError("New Window is not within bounds of its parent, Window is (x=%i y=%i width=%i height=%i), Parent is (width=%i, height=%i)" %
                           ((x, y, width, height) + console.getSize()))
        self.parent = console
        self.x, self.y, self.width, self.height = console._clampRect(x, y, width, height) # fill width height params
        self.cursorX = self.cursorY = 0
        if isinstance(console, Console):
            self.console = console
        else:
            self.console = self.parent.console
        
    _cursor_normalize = Console._cursor_normalize
    _cursor_advance = Console._cursor_advance
    _cursor_newline = Console._cursor_newline
    _drawable = Console._drawable
    _rectInBounds = Console._rectInBounds
    _clampRect = Console._clampRect
    
    def _translate(self, x, y):
        """Convertion x and y to their position on the root Console"""
        # we add our position relative to our parent and then call then next parent up
        return self.parent._translate((x + self.x), (y + self.y))
    
    blit = Console.blit

    def clear(self, fgcolor=(255, 255, 255), bgcolor=(0, 0, 0)):
        """Clears the entire Window.
        """
        assert _verify_colors(fgcolor, bgcolor)
        self.draw_rect(0, 0, None, None, 0x20, fgcolor, bgcolor)

    def _setChar(self, x, y, char=None, fgcolor=None, bgcolor=None, bgblend=BND_SET):
        parent._setChar((x + self.x), (y + self.y), char, fgcolor, bgcolor, bgblend)
        
    drawChar = Console.drawChar
    drawStr = Console.drawStr
    drawRect = Console.drawRect
    drawFrame = Console.drawFrame
    

    def getChar(self, x, y):
        """Return the character and colors of a cell as (ch, fg, bg)
        """
        self._drawable(x, y)
        return self.console.getChar(self._translate(x, y))

    getSize = Console.getSize

    def __repr__(self):
        return "<Window(X=%i Y=%i Width=%i Height=%i)>" % (self.x, self.y,
                                                          self.width,
                                                          self.height)


def init(width, height, title='TDL', fullscreen=False, renderer=RENDERER_SDL):
    """Start the main console with the given width and height and return the
    root console.

    Remember to use tdl.flush() to make what's drawn visible on the console.

    After the root console is garbage collected the window made by this function
    will close.
    """
    global _rootinitialized, _rootconsole
    if not _fontinitialized: # set the default font to the one that comes with tdl
        setFont(_unpackfile('terminal.png'),
                 16, 16, FONT_LAYOUT_ASCII_INCOL)

    if renderer>RENDERER_SDL:
        raise TDLError('Render type out of range: "%i"' % renderer)
    
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

def setFont(path, width, height, flags):
    """Changes the font to be used for this session
    This should be called before tdl.init

    path must be a string for where a bitmap file is found.

    width and height should be the size of an individual tile.

    flags are used to define the characters layout in the bitmap and the font type :
    FONT_LAYOUT_ASCII_INCOL : characters in ASCII order, code 0-15 in the first column
    FONT_LAYOUT_ASCII_INROW : characters in ASCII order, code 0-15 in the first row
    FONT_LAYOUT_TCOD : simplified layout, see libtcod documents
    FONT_TYPE_GREYSCALE : create an anti-aliased font from a greyscale bitmap
    """
    global _fontinitialized
    _fontinitialized = True
    assert os.path.exists(path), 'no file exists at "%s"' % path
    _lib.TCOD_console_set_custom_font(_format_string(path), flags, width, height)

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

def setFPS(fps):
    """Set the frames per second.

    You can set this to have no limit by using 0.
    """
    if fps is None:
        fps = 0
    assert isinstance(fps, int), 'fps must be an integer or None, got: %s' % repr(fps)
    _lib.TCOD_sys_set_fps(fps)

def getFPS():
    """Return the current frames per second of the running program.
    """
    return _lib.TCOD_sys_get_fps()

def forceResolution(width, height):
    """Change the fullscreen resoulution
    """
    _lib.TCOD_sys_force_fullscreen_resolution(width, height)

__all__ = [var for var in locals().keys() if not '_' in var[0]]

