"""This module handles backward compatibility with the ctypes libtcodpy module.
"""

from __future__ import absolute_import as _

import threading as _threading

from tcod.libtcod import *

from tcod.tcod import _int, _cdata, _unpack_char_p
from tcod.tcod import _bytes, _unicode, _fmt_bytes, _fmt_unicode
from tcod.tcod import _CDataWrapper
from tcod.tcod import _PropagateException
from tcod.tcod import BSP as Bsp
from tcod.tcod import Key, Mouse, HeightMap, Console, Image

class ConsoleBuffer(object):
    """Simple console that allows direct (fast) access to cells. simplifies
    use of the "fill" functions.

    Args:
        width (int): Width of the new ConsoleBuffer.
        height (int): Height of the new ConsoleBuffer.
        back_r (int): Red background color, from 0 to 255.
        back_g (int): Green background color, from 0 to 255.
        back_b (int): Blue background color, from 0 to 255.
        fore_r (int): Red foreground color, from 0 to 255.
        fore_g (int): Green foreground color, from 0 to 255.
        fore_b (int): Blue foreground color, from 0 to 255.
        char (AnyStr): A single character str or bytes object.
    """
    def __init__(self, width, height, back_r=0, back_g=0, back_b=0, fore_r=0, fore_g=0, fore_b=0, char=' '):
        """initialize with given width and height. values to fill the buffer
        are optional, defaults to black with no characters.
        """
        n = width * height
        self.width = width
        self.height = height
        self.clear(back_r, back_g, back_b, fore_r, fore_g, fore_b, char)

    def clear(self, back_r=0, back_g=0, back_b=0, fore_r=0, fore_g=0, fore_b=0, char=' '):
        """Clears the console.  Values to fill it with are optional, defaults
        to black with no characters.

        Args:
            back_r (int): Red background color, from 0 to 255.
            back_g (int): Green background color, from 0 to 255.
            back_b (int): Blue background color, from 0 to 255.
            fore_r (int): Red foreground color, from 0 to 255.
            fore_g (int): Green foreground color, from 0 to 255.
            fore_b (int): Blue foreground color, from 0 to 255.
            char (AnyStr): A single character str or bytes object.
        """
        n = self.width * self.height
        self.back_r = [back_r] * n
        self.back_g = [back_g] * n
        self.back_b = [back_b] * n
        self.fore_r = [fore_r] * n
        self.fore_g = [fore_g] * n
        self.fore_b = [fore_b] * n
        self.char = [ord(char)] * n

    def copy(self):
        """Returns a copy of this ConsoleBuffer.

        Returns:
            ConsoleBuffer: A new ConsoleBuffer copy.
        """
        other = ConsoleBuffer(0, 0)
        other.width = self.width
        other.height = self.height
        other.back_r = list(self.back_r)  # make explicit copies of all lists
        other.back_g = list(self.back_g)
        other.back_b = list(self.back_b)
        other.fore_r = list(self.fore_r)
        other.fore_g = list(self.fore_g)
        other.fore_b = list(self.fore_b)
        other.char = list(self.char)
        return other

    def set_fore(self, x, y, r, g, b, char):
        """Set the character and foreground color of one cell.

        Args:
            x (int): X position to change.
            y (int): Y position to change.
            r (int): Red foreground color, from 0 to 255.
            g (int): Green foreground color, from 0 to 255.
            b (int): Blue foreground color, from 0 to 255.
            char (AnyStr): A single character str or bytes object.
        """
        i = self.width * y + x
        self.fore_r[i] = r
        self.fore_g[i] = g
        self.fore_b[i] = b
        self.char[i] = ord(char)

    def set_back(self, x, y, r, g, b):
        """Set the background color of one cell.

        Args:
            x (int): X position to change.
            y (int): Y position to change.
            r (int): Red background color, from 0 to 255.
            g (int): Green background color, from 0 to 255.
            b (int): Blue background color, from 0 to 255.
            char (AnyStr): A single character str or bytes object.
        """
        i = self.width * y + x
        self.back_r[i] = r
        self.back_g[i] = g
        self.back_b[i] = b

    def set(self, x, y, back_r, back_g, back_b, fore_r, fore_g, fore_b, char):
        """Set the background color, foreground color and character of one cell.

        Args:
            x (int): X position to change.
            y (int): Y position to change.
            back_r (int): Red background color, from 0 to 255.
            back_g (int): Green background color, from 0 to 255.
            back_b (int): Blue background color, from 0 to 255.
            fore_r (int): Red foreground color, from 0 to 255.
            fore_g (int): Green foreground color, from 0 to 255.
            fore_b (int): Blue foreground color, from 0 to 255.
            char (AnyStr): A single character str or bytes object.
        """
        i = self.width * y + x
        self.back_r[i] = back_r
        self.back_g[i] = back_g
        self.back_b[i] = back_b
        self.fore_r[i] = fore_r
        self.fore_g[i] = fore_g
        self.fore_b[i] = fore_b
        self.char[i] = ord(char)

    def blit(self, dest, fill_fore=True, fill_back=True):
        """Use libtcod's "fill" functions to write the buffer to a console.

        Args:
            dest (Console): Console object to modify.
            fill_fore (bool):
                If True, fill the foreground color and characters.
            fill_back (bool):
                If True, fill the background color.
        """
        dest = _cdata(dest)
        if (console_get_width(dest) != self.width or
            console_get_height(dest) != self.height):
            raise ValueError('ConsoleBuffer.blit: Destination console has an incorrect size.')

        if fill_back:
            lib.TCOD_console_fill_background(dest or ffi.NULL,
                                              ffi.new('int[]', self.back_r),
                                              ffi.new('int[]', self.back_g),
                                              ffi.new('int[]', self.back_b))
        if fill_fore:
            lib.TCOD_console_fill_foreground(dest or ffi.NULL,
                                              ffi.new('int[]', self.fore_r),
                                              ffi.new('int[]', self.fore_g),
                                              ffi.new('int[]', self.fore_b))
            lib.TCOD_console_fill_char(dest or ffi.NULL,
                                        ffi.new('int[]', self.char))

class Dice(_CDataWrapper):
    """

    Args:
        nb_dices (int): Number of dice.
        nb_faces (int): Number of sides on a die.
        multiplier (float): Multiplier.
        addsub (float): Addition.

    .. versionchanged:: 2.0
        This class now acts like the other CData wrapped classes
        and no longer acts like a list.

    .. deprecated:: 2.0
        You should make your own dice functions instead of using this class
        which is tied to a CData object.
    """

    def __init__(self, *args, **kargs):
        super(Dice, self).__init__(*args, **kargs)
        if self.cdata == ffi.NULL:
            self._init(*args, **kargs)

    def _init(self, nb_dices=0, nb_faces=0, multiplier=0, addsub=0):
        self.cdata = ffi.new('TCOD_dice_t*')
        self.nb_dices = nb_dices
        self.nb_faces = nb_faces
        self.multiplier = multiplier
        self.addsub = addsub

    def _get_nb_dices(self):
        return self.nb_rolls
    def _set_nb_dices(self, value):
        self.nb_rolls = value
    nb_dices = property(_get_nb_dices, _set_nb_dices)

    def __str__(self):
        add = '+(%s)' % self.addsub if self.addsub != 0 else ''
        return '%id%ix%s%s' % (self.nb_dices, self.nb_faces,
                               self.multiplier, add)

    def __repr__(self):
        return ('%s(nb_dices=%r,nb_faces=%r,multiplier=%r,addsub=%r)' %
                (self.__class__.__name__, self.nb_dices, self.nb_faces,
                 self.multiplier, self.addsub))

def bsp_new_with_size(x, y, w, h):
    """Create a new BSP instance with the given rectangle.

    Args:
        x (int): Rectangle left coordinate.
        y (int): Rectangle top coordinate.
        w (int): Rectangle width.
        h (int): Rectangle height.

    Returns:
        BSP: A new BSP instance.

    .. deprecated:: 2.0
       Call the :any:`BSP` class instead.
    """
    return Bsp(x, y, w, h)

def bsp_split_once(node, horizontal, position):
    """
    .. deprecated:: 2.0
       Use :any:`BSP.split_once` instead.
    """
    node.split_once('h' if horizontal else 'v', position)

def bsp_split_recursive(node, randomizer, nb, minHSize, minVSize, maxHRatio,
                        maxVRatio):
    node.split_recursive(nb, minHSize, minVSize,
                         maxHRatio, maxVRatio, randomizer)

def bsp_resize(node, x, y, w, h):
    """
    .. deprecated:: 2.0
       Use :any:`BSP.resize` instead.
    """
    node.resize(x, y, w, h)

def bsp_left(node):
    """
    .. deprecated:: 2.0
       Use :any:`BSP.get_left` instead.
    """
    return node.get_left()

def bsp_right(node):
    """
    .. deprecated:: 2.0
       Use :any:`BSP.get_right` instead.
    """
    return node.get_right()

def bsp_father(node):
    """
    .. deprecated:: 2.0
       Use :any:`BSP.get_parent` instead.
    """
    return node.get_parent()

def bsp_is_leaf(node):
    """
    .. deprecated:: 2.0
       Use :any:`BSP.is_leaf` instead.
    """
    return node.is_leaf()

def bsp_contains(node, cx, cy):
    """
    .. deprecated:: 2.0
       Use :any:`BSP.contains` instead.
    """
    return node.contains(cx, cy)

def bsp_find_node(node, cx, cy):
    """
    .. deprecated:: 2.0
       Use :any:`BSP.find_node` instead.
    """
    return node.find_node(cx, cy)

def _bsp_traverse(node, func, callback, userData):
    """pack callback into a handle for use with the callback
    _pycall_bsp_callback
    """
    with _PropagateException() as propagate:
        handle = ffi.new_handle((callback, userData, propagate))
        func(node.cdata, lib._pycall_bsp_callback, handle)

def bsp_traverse_pre_order(node, callback, userData=0):
    """Traverse this nodes hierarchy with a callback.

    .. deprecated:: 2.0
       Use :any:`BSP.walk` instead.
    """
    _bsp_traverse(node, lib.TCOD_bsp_traverse_pre_order, callback, userData)

def bsp_traverse_in_order(node, callback, userData=0):
    """Traverse this nodes hierarchy with a callback.

    .. deprecated:: 2.0
       Use :any:`BSP.walk` instead.
    """
    _bsp_traverse(node, lib.TCOD_bsp_traverse_in_order, callback, userData)

def bsp_traverse_post_order(node, callback, userData=0):
    """Traverse this nodes hierarchy with a callback.

    .. deprecated:: 2.0
       Use :any:`BSP.walk` instead.
    """
    _bsp_traverse(node, lib.TCOD_bsp_traverse_post_order, callback, userData)

def bsp_traverse_level_order(node, callback, userData=0):
    """Traverse this nodes hierarchy with a callback.

    .. deprecated:: 2.0
       Use :any:`BSP.walk` instead.
    """
    _bsp_traverse(node, lib.TCOD_bsp_traverse_level_order, callback, userData)

def bsp_traverse_inverted_level_order(node, callback, userData=0):
    """Traverse this nodes hierarchy with a callback.

    .. deprecated:: 2.0
       Use :any:`BSP.walk` instead.
    """
    _bsp_traverse(node, lib.TCOD_bsp_traverse_inverted_level_order,
                  callback, userData)

def bsp_remove_sons(node):
    """Delete all children of a given node.  Not recommended.

    .. note::
       This function will add unnecessary complexity to your code.
       Don't use it.

    .. deprecated:: 2.0
       BSP deletion is automatic.
    """
    node._invalidate_children()
    lib.TCOD_bsp_remove_sons(node.cdata)

def bsp_delete(node):
    """Exists for backward compatibility.  Does nothing.

    BSP's created by this library are automatically garbage collected once
    there are no references to the tree.
    This function exists for backwards compatibility.

    .. deprecated:: 2.0
       BSP deletion is automatic.
    """
    pass

def color_lerp(c1, c2, a):
    """Return the linear interpolation between two colors.

    ``a`` is the interpolation value, with 0 returing ``c1``,
    1 returning ``c2``, and 0.5 returing a color halfway between both.

    Args:
        c1 (Union[Tuple[int, int, int], Sequence[int]]):
            The first color.  At a=0.
        c2 (Union[Tuple[int, int, int], Sequence[int]]):
            The second color.  At a=1.
        a (float): The interpolation value,

    Returns:
        Color: The interpolated Color.
    """
    return Color.from_cdata(lib.TCOD_color_lerp(c1, c2, a))

def color_set_hsv(c, h, s, v):
    """Set a color using: hue, saturation, and value parameters.

    Does not return a new Color.  Instead the provided color is modified.

    Args:
        c (Color): Must be a color instance.
        h (float): Hue, from 0 to 1.
        s (float): Saturation, from 0 to 1.
        v (float): Value, from 0 to 1.
    """
    new_color = ffi.new('TCOD_color_t*')
    lib.TCOD_color_set_HSV(new_color, h, s, v)
    c.r = new_color.r
    c.g = new_color.g
    c.b = new_color.b

def color_get_hsv(c):
    """Return the (hue, saturation, value) of a color.

    Args:
        c (Union[Tuple[int, int, int], Sequence[int]]):
            An (r, g, b) sequence or Color instance.

    Returns:
        Tuple[float, float, float]:
            A tuple with (hue, saturation, value) values, from 0 to 1.
    """
    h = ffi.new('float *')
    s = ffi.new('float *')
    v = ffi.new('float *')
    lib.TCOD_color_get_HSV(c, h, s, v)
    return h[0], s[0], v[0]

def color_scale_HSV(c, scoef, vcoef):
    """Scale a color's saturation and value.

    Does not return a new Color.  Instead the provided color is modified.

    Args:
        c (Color): Must be a Color instance.
        scoef (float): Saturation multiplier, from 0 to 1.
                       Use 1 to keep current saturation.
        vcoef (float): Value multiplier, from 0 to 1.
                       Use 1 to keep current value.
    """
    color_p = ffi.new('TCOD_color_t*')
    color_p.r, color_p.g, color_p.b = c.r, c.g, c.b
    lib.TCOD_color_scale_HSV(color_p, scoef, vcoef)
    c.r, c.g, c.b = color_p.r, color_p.g, color_p.b

def color_gen_map(colors, indexes):
    """Return a smoothly defined scale of colors.

    If ``indexes`` is [0, 3, 9] for example, the first color from ``colors``
    will be returned at 0, the 2nd will be at 3, and the 3rd will be at 9.
    All in-betweens will be filled with a gradient.

    Args:
        colors (Iterable[Union[Tuple[int, int, int], Sequence[int]]]):
            Array of colors to be sampled.
        indexes (Iterable[int]): A list of indexes.

    Returns:
        List[Color]: A list of Color instances.

    Example:
        >>> tcod.color_gen_map([(0, 0, 0), (255, 128, 0)], [0, 5])
        [Color(0,0,0), Color(51,25,0), Color(102,51,0), Color(153,76,0), \
Color(204,102,0), Color(255,128,0)]
    """
    ccolors = ffi.new('TCOD_color_t[]', colors)
    cindexes = ffi.new('int[]', indexes)
    cres = ffi.new('TCOD_color_t[]', max(indexes) + 1)
    lib.TCOD_color_gen_map(cres, len(colors), ccolors, cindexes)
    return [Color.from_cdata(cdata) for cdata in cres]

_numpy = None

def _numpy_available():
    'check if numpy is available and lazily load it when needed'
    global _numpy
    if _numpy is None:
        try:
            import numpy as _numpy
        except ImportError:
            _numpy = False
    return _numpy

# initializing the console
def console_init_root(w, h, title, fullscreen=False,
                      renderer=RENDERER_SDL):
    """Set up the primary display and return the root console.

    Note:
        Currently only the SDL renderer is supported at the moment.
        Do not attempt to change it.

    Args:
        w (int): Width in character tiles for the root console.
        h (int): Height in character tiles for the root console.
        title (AnyStr):
            This string will be displayed on the created windows title bar.
        renderer: Rendering mode for libtcod to use.

    Returns:
        Console:
            Returns a special Console instance representing the root console.
    """
    lib.TCOD_console_init_root(w, h, _bytes(title), fullscreen, renderer)
    return Console(ffi.NULL) # root console is null


def console_set_custom_font(fontFile, flags=FONT_LAYOUT_ASCII_INCOL,
                            nb_char_horiz=0, nb_char_vertic=0):
    """Load a custom font file.

    Call this before function before calling :any:`tcod.console_init_root`.

    Flags can be a mix of the following:

    * tcod.FONT_LAYOUT_ASCII_INCOL
    * tcod.FONT_LAYOUT_ASCII_INROW
    * tcod.FONT_TYPE_GREYSCALE
    * tcod.FONT_TYPE_GRAYSCALE
    * tcod.FONT_LAYOUT_TCOD

    Args:
        fontFile (AnyStr): Path to a font file.
        flags (int):
        nb_char_horiz (int):
        nb_char_vertic (int):
    """
    lib.TCOD_console_set_custom_font(_bytes(fontFile), flags,
                                     nb_char_horiz, nb_char_vertic)


def console_get_width(con):
    """Return the width of a console.

    Args:
        con (Console): Any Console instance.

    Returns:
        int: The width of a Console.

    .. deprecated:: 2.0
        Use `Console.get_width` instead.
    """
    return lib.TCOD_console_get_width(_cdata(con))

def console_get_height(con):
    """Return the height of a console.

    Args:
        con (Console): Any Console instance.

    Returns:
        int: The height of a Console.

    .. deprecated:: 2.0
        Use `Console.get_hright` instead.
    """
    return lib.TCOD_console_get_height(_cdata(con))

def console_map_ascii_code_to_font(asciiCode, fontCharX, fontCharY):
    lib.TCOD_console_map_ascii_code_to_font(_int(asciiCode), fontCharX,
                                                              fontCharY)

def console_map_ascii_codes_to_font(firstAsciiCode, nbCodes, fontCharX,
                                    fontCharY):
    lib.TCOD_console_map_ascii_codes_to_font(_int(firstAsciiCode), nbCodes,
                                              fontCharX, fontCharY)

def console_map_string_to_font(s, fontCharX, fontCharY):
    lib.TCOD_console_map_string_to_font_utf(_unicode(s), fontCharX, fontCharY)

def console_is_fullscreen():
    """Returns True if the display is fullscreen.

    Returns:
        bool: True if the display is fullscreen, otherwise False.
    """
    return bool(lib.TCOD_console_is_fullscreen())

def console_set_fullscreen(fullscreen):
    """Change the display to be fullscreen or windowed.

    Args:
        fullscreen (bool): Use True to change to fullscreen.
                           Use False to change to windowed.
    """
    lib.TCOD_console_set_fullscreen(fullscreen)

def console_is_window_closed():
    """Returns True if the window has received and exit event."""
    return lib.TCOD_console_is_window_closed()

def console_set_window_title(title):
    """Change the current title bar string.

    Args:
        title (AnyStr): A string to change the title bar to.
    """
    lib.TCOD_console_set_window_title(_bytes(title))

def console_credits():
    lib.TCOD_console_credits()

def console_credits_reset():
    lib.TCOD_console_credits_reset()

def console_credits_render(x, y, alpha):
    return lib.TCOD_console_credits_render(x, y, alpha)

def console_flush():
    """Update the display to represent the root consoles current state."""
    lib.TCOD_console_flush()

# drawing on a console
def console_set_default_background(con, col):
    """Change the default background color for a console.

    Args:
        con (Console): Any Console instance.
        col (Union[Tuple[int, int, int], Sequence[int]]):
            An (r, g, b) sequence or Color instance.
    """
    lib.TCOD_console_set_default_background(_cdata(con), col)

def console_set_default_foreground(con, col):
    """Change the default foreground color for a console.

    Args:
        con (Console): Any Console instance.
        col (Union[Tuple[int, int, int], Sequence[int]]):
            An (r, g, b) sequence or Color instance.
    """
    lib.TCOD_console_set_default_foreground(_cdata(con), col)

def console_clear(con):
    """Reset a console to its default colors and the space character.

    Args:
        con (Console): Any Console instance.

    .. seealso::
       :any:`console_set_default_background`
       :any:`console_set_default_foreground`
    """
    return lib.TCOD_console_clear(_cdata(con))

def console_put_char(con, x, y, c, flag=BKGND_DEFAULT):
    """Draw the character c at x,y using the default colors and a blend mode.

    Args:
        con (Console): Any Console instance.
        x (int): Character x position from the left.
        y (int): Character y position from the top.
        c (Union[int, AnyStr]): Character to draw, can be an integer or string.
        flag (int): Blending mode to use, defaults to BKGND_DEFAULT.
    """
    lib.TCOD_console_put_char(_cdata(con), x, y, _int(c), flag)

def console_put_char_ex(con, x, y, c, fore, back):
    """Draw the character c at x,y using the colors fore and back.

    Args:
        con (Console): Any Console instance.
        x (int): Character x position from the left.
        y (int): Character y position from the top.
        c (Union[int, AnyStr]): Character to draw, can be an integer or string.
        fore (Union[Tuple[int, int, int], Sequence[int]]):
            An (r, g, b) sequence or Color instance.
        back (Union[Tuple[int, int, int], Sequence[int]]):
            An (r, g, b) sequence or Color instance.
    """
    lib.TCOD_console_put_char_ex(_cdata(con), x, y,
                                 _int(c), fore, back)

def console_set_char_background(con, x, y, col, flag=BKGND_SET):
    """Change the background color of x,y to col using a blend mode.

    Args:
        con (Console): Any Console instance.
        x (int): Character x position from the left.
        y (int): Character y position from the top.
        col (Union[Tuple[int, int, int], Sequence[int]]):
            An (r, g, b) sequence or Color instance.
        flag (int): Blending mode to use, defaults to BKGND_SET.
    """
    lib.TCOD_console_set_char_background(_cdata(con), x, y, col, flag)

def console_set_char_foreground(con, x, y, col):
    """Change the foreground color of x,y to col.

    Args:
        con (Console): Any Console instance.
        x (int): Character x position from the left.
        y (int): Character y position from the top.
        col (Union[Tuple[int, int, int], Sequence[int]]):
            An (r, g, b) sequence or Color instance.
    """
    lib.TCOD_console_set_char_foreground(_cdata(con), x, y, col)

def console_set_char(con, x, y, c):
    """Change the character at x,y to c, keeping the current colors.

    Args:
        con (Console): Any Console instance.
        x (int): Character x position from the left.
        y (int): Character y position from the top.
        c (Union[int, AnyStr]): Character to draw, can be an integer or string.
    """
    lib.TCOD_console_set_char(_cdata(con), x, y, _int(c))

def console_set_background_flag(con, flag):
    """Change the default blend mode for this console.

    Args:
        con (Console): Any Console instance.
        flag (int): Blend mode to use by default.
    """
    lib.TCOD_console_set_background_flag(_cdata(con), flag)

def console_get_background_flag(con):
    """Return this consoles current blend mode.

    Args:
        con (Console): Any Console instance.
    """
    return lib.TCOD_console_get_background_flag(_cdata(con))

def console_set_alignment(con, alignment):
    """Change this consoles current alignment mode.

    * tcod.LEFT
    * tcod.CENTER
    * tcod.RIGHT

    Args:
        con (Console): Any Console instance.
        alignment (int):
    """
    lib.TCOD_console_set_alignment(_cdata(con), alignment)

def console_get_alignment(con):
    """Return this consoles current alignment mode.

    Args:
        con (Console): Any Console instance.
    """
    return lib.TCOD_console_get_alignment(_cdata(con))

def console_print(con, x, y, fmt):
    """Print a color formatted string on a console.

    Args:
        con (Console): Any Console instance.
        x (int): Character x position from the left.
        y (int): Character y position from the top.
        fmt (AnyStr): A unicode or bytes string optionaly using color codes.
    """
    lib.TCOD_console_print_utf(_cdata(con), x, y, _fmt_unicode(fmt))

def console_print_ex(con, x, y, flag, alignment, fmt):
    """Print a string on a console using a blend mode and alignment mode.

    Args:
        con (Console): Any Console instance.
        x (int): Character x position from the left.
        y (int): Character y position from the top.
    """
    lib.TCOD_console_print_ex_utf(_cdata(con), x, y,
                                   flag, alignment, _fmt_unicode(fmt))

def console_print_rect(con, x, y, w, h, fmt):
    """Print a string constrained to a rectangle.

    If h > 0 and the bottom of the rectangle is reached,
    the string is truncated. If h = 0,
    the string is only truncated if it reaches the bottom of the console.



    Returns:
        int: The number of lines of text once word-wrapped.
    """
    return lib.TCOD_console_print_rect_utf(_cdata(con), x, y, w, h,
                                            _fmt_unicode(fmt))

def console_print_rect_ex(con, x, y, w, h, flag, alignment, fmt):
    """Print a string constrained to a rectangle with blend and alignment.

    Returns:
        int: The number of lines of text once word-wrapped.
    """
    return lib.TCOD_console_print_rect_ex_utf(_cdata(con), x, y, w, h,
                                              flag, alignment,
                                              _fmt_unicode(fmt))

def console_get_height_rect(con, x, y, w, h, fmt):
    """Return the height of this text once word-wrapped into this rectangle.

    Returns:
        int: The number of lines of text once word-wrapped.
    """
    return lib.TCOD_console_get_height_rect_utf(_cdata(con), x, y, w, h,
                                                 _fmt_unicode(fmt))

def console_rect(con, x, y, w, h, clr, flag=BKGND_DEFAULT):
    """Draw a the background color on a rect optionally clearing the text.

    If clr is True the affected tiles are changed to space character.
    """
    lib.TCOD_console_rect(_cdata(con), x, y, w, h, clr, flag)

def console_hline(con, x, y, l, flag=BKGND_DEFAULT):
    """Draw a horizontal line on the console.

    This always uses the character 196, the horizontal line character.
    """
    lib.TCOD_console_hline(_cdata(con), x, y, l, flag)

def console_vline(con, x, y, l, flag=BKGND_DEFAULT):
    """Draw a vertical line on the console.

    This always uses the character 179, the vertical line character.
    """
    lib.TCOD_console_vline(_cdata(con), x, y, l, flag)

def console_print_frame(con, x, y, w, h, clear=True, flag=BKGND_DEFAULT, fmt=b''):
    """Draw a framed rectangle with optinal text.

    This uses the default background color and blend mode to fill the
    rectangle and the default foreground to draw the outline.

    fmt will be printed on the inside of the rectangle, word-wrapped.
    """
    lib.TCOD_console_print_frame(_cdata(con), x, y, w, h, clear, flag,
                                  _fmt_bytes(fmt))

def console_set_color_control(con, fore, back):
    """Configure color control codes.

    Args:
        con (int): Color control constant to modify.
        fore (Union[Tuple[int, int, int], Sequence[int]]):
            An (r, g, b) sequence or Color instance.
        back (Union[Tuple[int, int, int], Sequence[int]]):
            An (r, g, b) sequence or Color instance.
    """
    lib.TCOD_console_set_color_control(_cdata(con), fore, back)

def console_get_default_background(con):
    """Return this consoles default background color."""
    return Color.from_cdata(lib.TCOD_console_get_default_background(_cdata(con)))

def console_get_default_foreground(con):
    """Return this consoles default foreground color."""
    return Color.from_cdata(lib.TCOD_console_get_default_foreground(_cdata(con)))

def console_get_char_background(con, x, y):
    """Return the background color at the x,y of this console."""
    return Color.from_cdata(lib.TCOD_console_get_char_background(_cdata(con), x, y))

def console_get_char_foreground(con, x, y):
    """Return the foreground color at the x,y of this console."""
    return Color.from_cdata(lib.TCOD_console_get_char_foreground(_cdata(con), x, y))

def console_get_char(con, x, y):
    """Return the character at the x,y of this console."""
    return lib.TCOD_console_get_char(_cdata(con), x, y)

def console_set_fade(fade, fadingColor):
    lib.TCOD_console_set_fade(fade, fadingColor)

def console_get_fade():
    return lib.TCOD_console_get_fade()

def console_get_fading_color():
    return Color.from_cdata(lib.TCOD_console_get_fading_color())

# handling keyboard input
def console_wait_for_keypress(flush):
    """Block until the user presses a key, then returns a new Key.

    Args:
        flush bool: If True then the event queue is cleared before waiting
                    for the next event.

    Returns:
        Key: A new Key instance.
    """
    k=Key()
    lib.TCOD_console_wait_for_keypress_wrapper(k.cdata, flush)
    return k

def console_check_for_keypress(flags=KEY_RELEASED):
    k=Key()
    lib.TCOD_console_check_for_keypress_wrapper(k.cdata, flags)
    return k

def console_is_key_pressed(key):
    return lib.TCOD_console_is_key_pressed(key)

def console_set_keyboard_repeat(initial_delay, interval):
    lib.TCOD_console_set_keyboard_repeat(initial_delay, interval)

def console_disable_keyboard_repeat():
    lib.TCOD_console_disable_keyboard_repeat()

# using offscreen consoles
def console_new(w, h):
    """Return an offscreen console of size: w,h."""
    return Console(ffi.gc(lib.TCOD_console_new(w, h), lib.TCOD_console_delete))
def console_from_file(filename):
    return Console(lib.TCOD_console_from_file(_bytes(filename)))

def console_blit(src, x, y, w, h, dst, xdst, ydst, ffade=1.0,bfade=1.0):
    """Blit the console src from x,y,w,h to console dst at xdst,ydst."""
    lib.TCOD_console_blit(_cdata(src), x, y, w, h,
                          _cdata(dst), xdst, ydst, ffade, bfade)

def console_set_key_color(con, col):
    """Set a consoles blit transparent color."""
    lib.TCOD_console_set_key_color(_cdata(con), col)

def console_delete(con):
    con = _cdata(con)
    if con == ffi.NULL:
        lib.TCOD_console_delete(con)

# fast color filling
def console_fill_foreground(con, r, g, b):
    """Fill the foregound of a console with r,g,b.

    Args:
        con (Console): Any Console instance.
        r (Sequence[int]): An array of integers with a length of width*height.
        g (Sequence[int]): An array of integers with a length of width*height.
        b (Sequence[int]): An array of integers with a length of width*height.
    """
    if len(r) != len(g) or len(r) != len(b):
        raise TypeError('R, G and B must all have the same size.')
    if (_numpy_available() and isinstance(r, _numpy.ndarray) and
        isinstance(g, _numpy.ndarray) and isinstance(b, _numpy.ndarray)):
        #numpy arrays, use numpy's ctypes functions
        r = _numpy.ascontiguousarray(r, dtype=_numpy.intc)
        g = _numpy.ascontiguousarray(g, dtype=_numpy.intc)
        b = _numpy.ascontiguousarray(b, dtype=_numpy.intc)
        cr = ffi.cast('int *', r.ctypes.data)
        cg = ffi.cast('int *', g.ctypes.data)
        cb = ffi.cast('int *', b.ctypes.data)
    else:
        # otherwise convert using ffi arrays
        cr = ffi.new('int[]', r)
        cg = ffi.new('int[]', g)
        cb = ffi.new('int[]', b)

    lib.TCOD_console_fill_foreground(_cdata(con), cr, cg, cb)

def console_fill_background(con, r, g, b):
    """Fill the backgound of a console with r,g,b.

    Args:
        con (Console): Any Console instance.
        r (Sequence[int]): An array of integers with a length of width*height.
        g (Sequence[int]): An array of integers with a length of width*height.
        b (Sequence[int]): An array of integers with a length of width*height.
    """
    if len(r) != len(g) or len(r) != len(b):
        raise TypeError('R, G and B must all have the same size.')
    if (_numpy_available() and isinstance(r, _numpy.ndarray) and
        isinstance(g, _numpy.ndarray) and isinstance(b, _numpy.ndarray)):
        #numpy arrays, use numpy's ctypes functions
        r = _numpy.ascontiguousarray(r, dtype=_numpy.intc)
        g = _numpy.ascontiguousarray(g, dtype=_numpy.intc)
        b = _numpy.ascontiguousarray(b, dtype=_numpy.intc)
        cr = ffi.cast('int *', r.ctypes.data)
        cg = ffi.cast('int *', g.ctypes.data)
        cb = ffi.cast('int *', b.ctypes.data)
    else:
        # otherwise convert using ffi arrays
        cr = ffi.new('int[]', r)
        cg = ffi.new('int[]', g)
        cb = ffi.new('int[]', b)

    lib.TCOD_console_fill_background(_cdata(con), cr, cg, cb)

def console_fill_char(con,arr):
    """Fill the character tiles of a console with an array.

    Args:
        con (Console): Any Console instance.
        arr (Sequence[int]): An array of integers with a length of width*height.
    """
    if (_numpy_available() and isinstance(arr, _numpy.ndarray) ):
        #numpy arrays, use numpy's ctypes functions
        arr = _numpy.ascontiguousarray(arr, dtype=_numpy.intc)
        carr = ffi.cast('int *', arr.ctypes.data)
    else:
        #otherwise convert using the ffi module
        carr = ffi.new('int[]', arr)

    lib.TCOD_console_fill_char(_cdata(con), carr)

def console_load_asc(con, filename):
    return lib.TCOD_console_load_asc(_cdata(con), _bytes(filename))

def console_save_asc(con, filename):
    lib.TCOD_console_save_asc(_cdata(con),_bytes(filename))

def console_load_apf(con, filename):
    return lib.TCOD_console_load_apf(_cdata(con),_bytes(filename))

def console_save_apf(con, filename):
    lib.TCOD_console_save_apf(_cdata(con),_bytes(filename))

@ffi.def_extern()
def _pycall_path_func(x1, y1, x2, y2, handle):
    '''static float _pycall_path_func( int xFrom, int yFrom, int xTo, int yTo, void *user_data );
    '''
    func, propagate_manager, user_data = ffi.from_handle(handle)
    try:
        return func(x1, y1, x2, y2, *user_data)
    except BaseException:
        propagate_manager.propagate(*_sys.exc_info())
        return None

def path_new_using_map(m, dcost=1.41):
    return (ffi.gc(lib.TCOD_path_new_using_map(m, dcost),
                    lib.TCOD_path_delete), _PropagateException())

def path_new_using_function(w, h, func, userData=0, dcost=1.41):
    propagator = _PropagateException()
    handle = ffi.new_handle((func, propagator, (userData,)))
    return (ffi.gc(lib.TCOD_path_new_using_function(w, h, lib._pycall_path_func,
            handle, dcost), lib.TCOD_path_delete), propagator, handle)

def path_compute(p, ox, oy, dx, dy):
    with p[1]:
        return lib.TCOD_path_compute(p[0], ox, oy, dx, dy)

def path_get_origin(p):
    x = ffi.new('int *')
    y = ffi.new('int *')
    lib.TCOD_path_get_origin(p[0], x, y)
    return x[0], y[0]

def path_get_destination(p):
    x = ffi.new('int *')
    y = ffi.new('int *')
    lib.TCOD_path_get_destination(p[0], x, y)
    return x[0], y[0]

def path_size(p):
    return lib.TCOD_path_size(p[0])

def path_reverse(p):
    lib.TCOD_path_reverse(p[0])

def path_get(p, idx):
    x = ffi.new('int *')
    y = ffi.new('int *')
    lib.TCOD_path_get(p[0], idx, x, y)
    return x[0], y[0]

def path_is_empty(p):
    return lib.TCOD_path_is_empty(p[0])

def path_walk(p, recompute):
    x = ffi.new('int *')
    y = ffi.new('int *')
    with p[1]:
        if lib.TCOD_path_walk(p[0], x, y, recompute):
            return x[0], y[0]
    return None,None

def path_delete(p):
    pass

def dijkstra_new(m, dcost=1.41):
    return (ffi.gc(lib.TCOD_dijkstra_new(m, dcost),
                    lib.TCOD_dijkstra_delete), _PropagateException())

def dijkstra_new_using_function(w, h, func, userData=0, dcost=1.41):
    propagator = _PropagateException()
    handle = ffi.new_handle((func, propagator, (userData,)))
    return (ffi.gc(lib.TCOD_dijkstra_new_using_function(w, h,
                    lib._pycall_path_func, handle, dcost),
                    lib.TCOD_dijkstra_delete), propagator, handle)

def dijkstra_compute(p, ox, oy):
    with p[1]:
        lib.TCOD_dijkstra_compute(p[0], ox, oy)

def dijkstra_path_set(p, x, y):
    return lib.TCOD_dijkstra_path_set(p[0], x, y)

def dijkstra_get_distance(p, x, y):
    return lib.TCOD_dijkstra_get_distance(p[0], x, y)

def dijkstra_size(p):
    return lib.TCOD_dijkstra_size(p[0])

def dijkstra_reverse(p):
    lib.TCOD_dijkstra_reverse(p[0])

def dijkstra_get(p, idx):
    x = ffi.new('int *')
    y = ffi.new('int *')
    lib.TCOD_dijkstra_get(p[0], idx, x, y)
    return x[0], y[0]

def dijkstra_is_empty(p):
    return lib.TCOD_dijkstra_is_empty(p[0])

def dijkstra_path_walk(p):
    x = ffi.new('int *')
    y = ffi.new('int *')
    if lib.TCOD_dijkstra_path_walk(p[0], x, y):
        return x[0], y[0]
    return None,None

def dijkstra_delete(p):
    pass

def heightmap_new(w, h):
    return HeightMap(w, h)

def heightmap_set_value(hm, x, y, value):
    lib.TCOD_heightmap_set_value(hm.cdata, x, y, value)

def heightmap_add(hm, value):
    lib.TCOD_heightmap_add(hm.cdata, value)

def heightmap_scale(hm, value):
    lib.TCOD_heightmap_scale(hm.cdata, value)

def heightmap_clear(hm):
    lib.TCOD_heightmap_clear(hm.cdata)

def heightmap_clamp(hm, mi, ma):
    lib.TCOD_heightmap_clamp(hm.cdata, mi, ma)

def heightmap_copy(hm1, hm2):
    lib.TCOD_heightmap_copy(hm1.cdata, hm2.cdata)

def heightmap_normalize(hm,  mi=0.0, ma=1.0):
    lib.TCOD_heightmap_normalize(hm.cdata, mi, ma)

def heightmap_lerp_hm(hm1, hm2, hm3, coef):
    lib.TCOD_heightmap_lerp_hm(hm1.cdata, hm2.cdata, hm3.cdata, coef)

def heightmap_add_hm(hm1, hm2, hm3):
    lib.TCOD_heightmap_add_hm(hm1.cdata, hm2.cdata, hm3.cdata)

def heightmap_multiply_hm(hm1, hm2, hm3):
    lib.TCOD_heightmap_multiply_hm(hm1.cdata, hm2.cdata, hm3.cdata)

def heightmap_add_hill(hm, x, y, radius, height):
    lib.TCOD_heightmap_add_hill(hm.cdata, x, y, radius, height)

def heightmap_dig_hill(hm, x, y, radius, height):
    lib.TCOD_heightmap_dig_hill(hm.cdata, x, y, radius, height)

def heightmap_rain_erosion(hm, nbDrops, erosionCoef, sedimentationCoef, rnd=None):
    lib.TCOD_heightmap_rain_erosion(hm.cdata, nbDrops, erosionCoef,
                                     sedimentationCoef, rnd or ffi.NULL)

def heightmap_kernel_transform(hm, kernelsize, dx, dy, weight, minLevel,
                               maxLevel):
    cdx = ffi.new('int[]', dx)
    cdy = ffi.new('int[]', dy)
    cweight = ffi.new('float[]', weight)
    lib.TCOD_heightmap_kernel_transform(hm.cdata, kernelsize, cdx, cdy, cweight,
                                         minLevel, maxLevel)

def heightmap_add_voronoi(hm, nbPoints, nbCoef, coef, rnd=None):
    ccoef = ffi.new('float[]', coef)
    lib.TCOD_heightmap_add_voronoi(hm.cdata, nbPoints, nbCoef, ccoef, rnd or ffi.NULL)

def heightmap_add_fbm(hm, noise, mulx, muly, addx, addy, octaves, delta, scale):
    lib.TCOD_heightmap_add_fbm(hm.cdata, noise, mulx, muly, addx, addy,
                                octaves, delta, scale)
def heightmap_scale_fbm(hm, noise, mulx, muly, addx, addy, octaves, delta,
                        scale):
    lib.TCOD_heightmap_scale_fbm(hm.cdata, noise, mulx, muly, addx, addy,
                                  octaves, delta, scale)

def heightmap_dig_bezier(hm, px, py, startRadius, startDepth, endRadius,
                         endDepth):
    #IARRAY = c_int * 4
    cpx = ffi.new('int[4]', px)
    cpy = ffi.new('int[4]', py)
    lib.TCOD_heightmap_dig_bezier(hm.cdata, cpx, cpy, startRadius,
                                   startDepth, endRadius,
                                   endDepth)

def heightmap_get_value(hm, x, y):
    return lib.TCOD_heightmap_get_value(hm.cdata, x, y)

def heightmap_get_interpolated_value(hm, x, y):
    return lib.TCOD_heightmap_get_interpolated_value(hm.cdata, x, y)

def heightmap_get_slope(hm, x, y):
    return lib.TCOD_heightmap_get_slope(hm.cdata, x, y)

def heightmap_get_normal(hm, x, y, waterLevel):
    #FARRAY = c_float * 3
    cn = ffi.new('float[3]')
    lib.TCOD_heightmap_get_normal(hm.cdata, x, y, cn, waterLevel)
    return tuple(cn)

def heightmap_count_cells(hm, mi, ma):
    return lib.TCOD_heightmap_count_cells(hm.cdata, mi, ma)

def heightmap_has_land_on_border(hm, waterlevel):
    return lib.TCOD_heightmap_has_land_on_border(hm.cdata, waterlevel)

def heightmap_get_minmax(hm):
    mi = ffi.new('float *')
    ma = ffi.new('float *')
    lib.TCOD_heightmap_get_minmax(hm.cdata, mi, ma)
    return mi[0], ma[0]

def heightmap_delete(hm):
    pass

def image_new(width, height):
    return ffi.gc(lib.TCOD_image_new(width, height), lib.TCOD_image_delete)

def image_clear(image,col):
    lib.TCOD_image_clear(image,col)

def image_invert(image):
    lib.TCOD_image_invert(image)

def image_hflip(image):
    lib.TCOD_image_hflip(image)

def image_rotate90(image, num=1):
    lib.TCOD_image_rotate90(image,num)

def image_vflip(image):
    lib.TCOD_image_vflip(image)

def image_scale(image, neww, newh):
    lib.TCOD_image_scale(image, neww, newh)

def image_set_key_color(image,col):
    lib.TCOD_image_set_key_color(image, col)

def image_get_alpha(image,x,y):
    return lib.TCOD_image_get_alpha(image, x, y)

def image_is_pixel_transparent(image,x,y):
    return lib.TCOD_image_is_pixel_transparent(image, x, y)

def image_load(filename):
    return ffi.gc(lib.TCOD_image_load(_bytes(filename)),
                   lib.TCOD_image_delete)

def image_from_console(console):
    return ffi.gc(lib.TCOD_image_from_console(_cdata(console)),
                   lib.TCOD_image_delete)

def image_refresh_console(image, console):
    lib.TCOD_image_refresh_console(image, _cdata(console))

def image_get_size(image):
    w = ffi.new('int *')
    h = ffi.new('int *')
    lib.TCOD_image_get_size(image, w, h)
    return w[0], h[0]

def image_get_pixel(image, x, y):
    return lib.TCOD_image_get_pixel(image, x, y)

def image_get_mipmap_pixel(image, x0, y0, x1, y1):
    return lib.TCOD_image_get_mipmap_pixel(image, x0, y0, x1, y1)

def image_put_pixel(image, x, y, col):
    lib.TCOD_image_put_pixel(image, x, y, col)
    ##lib.TCOD_image_put_pixel_wrapper(image, x, y, col)

def image_blit(image, console, x, y, bkgnd_flag, scalex, scaley, angle):
    lib.TCOD_image_blit(image, _cdata(console), x, y, bkgnd_flag,
                         scalex, scaley, angle)

def image_blit_rect(image, console, x, y, w, h, bkgnd_flag):
    lib.TCOD_image_blit_rect(image, _cdata(console), x, y, w, h, bkgnd_flag)

def image_blit_2x(image, console, dx, dy, sx=0, sy=0, w=-1, h=-1):
    lib.TCOD_image_blit_2x(image, _cdata(console), dx,dy,sx,sy,w,h)

def image_save(image, filename):
    lib.TCOD_image_save(image, _bytes(filename))

def image_delete(image):
    pass

def line_init(xo, yo, xd, yd):
    """Initilize a line whose points will be returned by `line_step`.

    This function does not return anything on its own.

    Does not include the origin point.

    Args:
        xo (int): X starting point.
        yo (int): Y starting point.
        xd (int): X destination point.
        yd (int): Y destination point.

    .. deprecated:: 2.0
       Use `line_iter` instead.
    """
    lib.TCOD_line_init(xo, yo, xd, yd)

def line_step():
    """After calling line_init returns (x, y) points of the line.

    Once all points are exhausted this function will return (None, None)

    Returns:
        Union[Tuple[int, int], Tuple[None, None]]:
            The next (x, y) point of the line setup by line_init,
            or (None, None) if there are no more points.

    .. deprecated:: 2.0
       Use `line_iter` instead.
    """
    x = ffi.new('int *')
    y = ffi.new('int *')
    ret = lib.TCOD_line_step(x, y)
    if not ret:
        return x[0], y[0]
    return None,None

_line_listener_lock = _threading.Lock()

def line(xo, yo, xd, yd, py_callback):
    """ Iterate over a line using a callback function.

    Your callback function will take x and y parameters and return True to
    continue iteration or False to stop iteration and return.

    This function includes both the start and end points.

    Args:
        xo (int): X starting point.
        yo (int): Y starting point.
        xd (int): X destination point.
        yd (int): Y destination point.
        py_callback (Callable[[int, int], bool]):
            A callback which takes x and y parameters and returns bool.

    Returns:
        bool: False if the callback cancels the line interation by
              returning False or None, otherwise True.

    .. deprecated:: 2.0
       Use `line_iter` instead.
    """
    with _PropagateException() as propagate:
        with _line_listener_lock:
            @ffi.def_extern(onerror=propagate)
            def _pycall_line_listener(x, y):
                return py_callback(x, y)
            return bool(lib.TCOD_line(xo, yo, xd, yd,
                                       lib._pycall_line_listener))

def line_iter(xo, yo, xd, yd):
    """ returns an iterator

    This iterator does not include the origin point.

    Args:
        xo (int): X starting point.
        yo (int): Y starting point.
        xd (int): X destination point.
        yd (int): Y destination point.

    Returns:
        Iterator[Tuple[int,int]]: An iterator of (x,y) points.
    """
    data = ffi.new('TCOD_bresenham_data_t *')
    lib.TCOD_line_init_mt(xo, yo, xd, yd, data)
    x = ffi.new('int *')
    y = ffi.new('int *')
    done = False
    while not lib.TCOD_line_step_mt(x, y, data):
        yield (x[0], y[0])

FOV_BASIC = 0
FOV_DIAMOND = 1
FOV_SHADOW = 2
FOV_PERMISSIVE_0 = 3
FOV_PERMISSIVE_1 = 4
FOV_PERMISSIVE_2 = 5
FOV_PERMISSIVE_3 = 6
FOV_PERMISSIVE_4 = 7
FOV_PERMISSIVE_5 = 8
FOV_PERMISSIVE_6 = 9
FOV_PERMISSIVE_7 = 10
FOV_PERMISSIVE_8 = 11
FOV_RESTRICTIVE = 12
NB_FOV_ALGORITHMS = 13

def map_new(w, h):
    return ffi.gc(lib.TCOD_map_new(w, h), lib.TCOD_map_delete)

def map_copy(source, dest):
    return lib.TCOD_map_copy(source, dest)

def map_set_properties(m, x, y, isTrans, isWalk):
    lib.TCOD_map_set_properties(m, x, y, isTrans, isWalk)

def map_clear(m,walkable=False,transparent=False):
    lib.TCOD_map_clear(m, walkable, transparent)

def map_compute_fov(m, x, y, radius=0, light_walls=True, algo=FOV_RESTRICTIVE ):
    lib.TCOD_map_compute_fov(m, x, y, radius, light_walls, algo)

def map_is_in_fov(m, x, y):
    return lib.TCOD_map_is_in_fov(m, x, y)

def map_is_transparent(m, x, y):
    return lib.TCOD_map_is_transparent(m, x, y)

def map_is_walkable(m, x, y):
    return lib.TCOD_map_is_walkable(m, x, y)

def map_delete(m):
    pass

def map_get_width(map):
    return lib.TCOD_map_get_width(map)

def map_get_height(map):
    return lib.TCOD_map_get_height(map)

def mouse_show_cursor(visible):
    lib.TCOD_mouse_show_cursor(visible)

def mouse_is_cursor_visible():
    return lib.TCOD_mouse_is_cursor_visible()

def mouse_move(x, y):
    lib.TCOD_mouse_move(x, y)

def mouse_get_status():
    return Mouse(lib.TCOD_mouse_get_status())

def namegen_parse(filename,random=None):
    lib.TCOD_namegen_parse(_bytes(filename), random or ffi.NULL)

def namegen_generate(name):
    return _unpack_char_p(lib.TCOD_namegen_generate(_bytes(name), False))

def namegen_generate_custom(name, rule):
    return _unpack_char_p(lib.TCOD_namegen_generate(_bytes(name),
                                                     _bytes(rule), False))

def namegen_get_sets():
    sets = lib.TCOD_namegen_get_sets()
    try:
        lst = []
        while not lib.TCOD_list_is_empty(sets):
            lst.append(_unpack_char_p(ffi.cast('char *', lib.TCOD_list_pop(sets))))
    finally:
        lib.TCOD_list_delete(sets)
    return lst

def namegen_destroy():
    lib.TCOD_namegen_destroy()

def noise_new(dim, h=NOISE_DEFAULT_HURST, l=NOISE_DEFAULT_LACUNARITY,
        random=None):
    return ffi.gc(lib.TCOD_noise_new(dim, h, l, random or ffi.NULL),
                   lib.TCOD_noise_delete)

def noise_set_type(n, typ):
    lib.TCOD_noise_set_type(n,typ)

def noise_get(n, f, typ=NOISE_DEFAULT):
    return lib.TCOD_noise_get_ex(n, ffi.new('float[]', f), typ)

def noise_get_fbm(n, f, oc, typ=NOISE_DEFAULT):
    return lib.TCOD_noise_get_fbm_ex(n, ffi.new('float[]', f), oc, typ)

def noise_get_turbulence(n, f, oc, typ=NOISE_DEFAULT):
    return lib.TCOD_noise_get_turbulence_ex(n, ffi.new('float[]', f), oc, typ)

def noise_delete(n):
    pass

_chr = chr
try:
    _chr = unichr # Python 2
except NameError:
    pass

def _unpack_union(type, union):
    '''
        unpack items from parser new_property (value_converter)
    '''
    if type == lib.TCOD_TYPE_BOOL:
        return bool(union.b)
    elif type == lib.TCOD_TYPE_CHAR:
        return _unicode(union.c)
    elif type == lib.TCOD_TYPE_INT:
        return union.i
    elif type == lib.TCOD_TYPE_FLOAT:
        return union.f
    elif (type == lib.TCOD_TYPE_STRING or
         lib.TCOD_TYPE_VALUELIST15 >= type >= lib.TCOD_TYPE_VALUELIST00):
         return _unpack_char_p(union.s)
    elif type == lib.TCOD_TYPE_COLOR:
        return Color.from_cdata(union.col)
    elif type == lib.TCOD_TYPE_DICE:
        return Dice(union.dice)
    elif type & lib.TCOD_TYPE_LIST:
        return _convert_TCODList(union.list, type & 0xFF)
    else:
        raise RuntimeError('Unknown libtcod type: %i' % type)

def _convert_TCODList(clist, type):
    return [_unpack_union(type, lib.TDL_list_get_union(clist, i))
            for i in range(lib.TCOD_list_size(clist))]

def parser_new():
    return ffi.gc(lib.TCOD_parser_new(), lib.TCOD_parser_delete)

def parser_new_struct(parser, name):
    return lib.TCOD_parser_new_struct(parser, name)

# prevent multiple threads from messing with def_extern callbacks
_parser_callback_lock = _threading.Lock()

def parser_run(parser, filename, listener=None):
    if not listener:
        lib.TCOD_parser_run(parser, _bytes(filename), ffi.NULL)
        return

    propagate_manager = _PropagateException()
    propagate = propagate_manager.propagate

    with _parser_callback_lock:
        clistener = ffi.new('TCOD_parser_listener_t *')

        @ffi.def_extern(onerror=propagate)
        def pycall_parser_new_struct(struct, name):
            return listener.end_struct(struct, _unpack_char_p(name))

        @ffi.def_extern(onerror=propagate)
        def pycall_parser_new_flag(name):
            return listener.new_flag(_unpack_char_p(name))

        @ffi.def_extern(onerror=propagate)
        def pycall_parser_new_property(propname, type, value):
            return listener.new_property(_unpack_char_p(propname), type,
                                         _unpack_union(type, value))

        @ffi.def_extern(onerror=propagate)
        def pycall_parser_end_struct(struct, name):
            return listener.end_struct(struct, _unpack_char_p(name))

        @ffi.def_extern(onerror=propagate)
        def pycall_parser_error(msg):
            listener.error(_unpack_char_p(msg))

        clistener.new_struct = lib.pycall_parser_new_struct
        clistener.new_flag = lib.pycall_parser_new_flag
        clistener.new_property = lib.pycall_parser_new_property
        clistener.end_struct = lib.pycall_parser_end_struct
        clistener.error = lib.pycall_parser_error

        with propagate_manager:
            lib.TCOD_parser_run(parser, _bytes(filename), clistener)

def parser_delete(parser):
    pass

def parser_get_bool_property(parser, name):
    return bool(lib.TCOD_parser_get_bool_property(parser, _bytes(name)))

def parser_get_int_property(parser, name):
    return lib.TCOD_parser_get_int_property(parser, _bytes(name))

def parser_get_char_property(parser, name):
    return _chr(lib.TCOD_parser_get_char_property(parser, _bytes(name)))

def parser_get_float_property(parser, name):
    return lib.TCOD_parser_get_float_property(parser, _bytes(name))

def parser_get_string_property(parser, name):
    return _unpack_char_p(lib.TCOD_parser_get_string_property(parser, _bytes(name)))

def parser_get_color_property(parser, name):
    return Color.from_cdata(lib.TCOD_parser_get_color_property(parser, _bytes(name)))

def parser_get_dice_property(parser, name):
    d = ffi.new('TCOD_dice_t *')
    lib.TCOD_parser_get_dice_property_py(parser, _bytes(name), d)
    return Dice(d)

def parser_get_list_property(parser, name, type):
    clist = lib.TCOD_parser_get_list_property(parser, _bytes(name), type)
    return _convert_TCODList(clist, type)

RNG_MT = 0
RNG_CMWC = 1

DISTRIBUTION_LINEAR = 0
DISTRIBUTION_GAUSSIAN = 1
DISTRIBUTION_GAUSSIAN_RANGE = 2
DISTRIBUTION_GAUSSIAN_INVERSE = 3
DISTRIBUTION_GAUSSIAN_RANGE_INVERSE = 4

def random_get_instance():
    return lib.TCOD_random_get_instance()

def random_new(algo=RNG_CMWC):
    return ffi.gc(lib.TCOD_random_new(algo), lib.TCOD_random_delete)

def random_new_from_seed(seed, algo=RNG_CMWC):
    return ffi.gc(lib.TCOD_random_new_from_seed(algo, seed),
                   lib.TCOD_random_delete)

def random_set_distribution(rnd, dist):
	lib.TCOD_random_set_distribution(rnd or ffi.NULL, dist)

def random_get_int(rnd, mi, ma):
    return lib.TCOD_random_get_int(rnd or ffi.NULL, mi, ma)

def random_get_float(rnd, mi, ma):
    return lib.TCOD_random_get_float(rnd or ffi.NULL, mi, ma)

def random_get_double(rnd, mi, ma):
    return lib.TCOD_random_get_double(rnd or ffi.NULL, mi, ma)

def random_get_int_mean(rnd, mi, ma, mean):
    return lib.TCOD_random_get_int_mean(rnd or ffi.NULL, mi, ma, mean)

def random_get_float_mean(rnd, mi, ma, mean):
    return lib.TCOD_random_get_float_mean(rnd or ffi.NULL, mi, ma, mean)

def random_get_double_mean(rnd, mi, ma, mean):
    return lib.TCOD_random_get_double_mean(rnd or ffi.NULL, mi, ma, mean)

def random_save(rnd):
    return ffi.gc(lib.TCOD_random_save(rnd or ffi.NULL),
                   lib.TCOD_random_delete)

def random_restore(rnd, backup):
    lib.TCOD_random_restore(rnd or ffi.NULL, backup)

def random_delete(rnd):
    pass

def struct_add_flag(struct, name):
    lib.TCOD_struct_add_flag(struct, name)

def struct_add_property(struct, name, typ, mandatory):
    lib.TCOD_struct_add_property(struct, name, typ, mandatory)

def struct_add_value_list(struct, name, value_list, mandatory):
    CARRAY = c_char_p * (len(value_list) + 1)
    cvalue_list = CARRAY()
    for i in range(len(value_list)):
        cvalue_list[i] = cast(value_list[i], c_char_p)
    cvalue_list[len(value_list)] = 0
    lib.TCOD_struct_add_value_list(struct, name, cvalue_list, mandatory)

def struct_add_list_property(struct, name, typ, mandatory):
    lib.TCOD_struct_add_list_property(struct, name, typ, mandatory)

def struct_add_structure(struct, sub_struct):
    lib.TCOD_struct_add_structure(struct, sub_struct)

def struct_get_name(struct):
    return _unpack_char_p(lib.TCOD_struct_get_name(struct))

def struct_is_mandatory(struct, name):
    return lib.TCOD_struct_is_mandatory(struct, name)

def struct_get_type(struct, name):
    return lib.TCOD_struct_get_type(struct, name)

# high precision time functions
def sys_set_fps(fps):
    """Set the maximum frame rate.

    You can disable the frame limit again by setting fps to 0.

    Args:
        fps (int): A frame rate limit (i.e. 60)
    """
    lib.TCOD_sys_set_fps(fps)

def sys_get_fps():
    """Return the current frames per second.

    This the actual frame rate, not the frame limit set by
    :any:`tcod.sys_set_fps`.

    This number is updated every second.

    Returns:
        int: The currently measured frame rate.
    """
    return lib.TCOD_sys_get_fps()

def sys_get_last_frame_length():
    """Return the delta time of the last rendered frame in seconds.

    Returns:
        float: The delta time of the last rendered frame.
    """
    return lib.TCOD_sys_get_last_frame_length()

def sys_sleep_milli(val):
    """Sleep for 'val' milliseconds.

    Args:
        val (int): Time to sleep for in milliseconds.

    .. deprecated:: 2.0
       Use :any:`time.sleep` instead.
    """
    lib.TCOD_sys_sleep_milli(val)

def sys_elapsed_milli():
    """Get number of milliseconds since the start of the program.

    Returns:
        int: Time since the progeam has started in milliseconds.

    .. deprecated:: 2.0
       Use :any:`time.clock` instead.
    """
    return lib.TCOD_sys_elapsed_milli()

def sys_elapsed_seconds():
    """Get number of seconds since the start of the program.

    Returns:
        float: Time since the progeam has started in seconds.

    .. deprecated:: 2.0
       Use :any:`time.clock` instead.
    """
    return lib.TCOD_sys_elapsed_seconds()

def sys_set_renderer(renderer):
    """Change the current rendering mode to renderer.

    .. deprecated:: 2.0
       RENDERER_GLSL and RENDERER_OPENGL are not currently available.
    """
    lib.TCOD_sys_set_renderer(renderer)

def sys_get_renderer():
    """Return the current rendering mode.

    """
    return lib.TCOD_sys_get_renderer()

# easy screenshots
def sys_save_screenshot(name=None):
    """Save a screenshot to a file.

    By default this will automatically save screenshots in the working
    directory.

    The automatic names are formatted as screenshotNNN.png.  For example:
    screenshot000.png, screenshot001.png, etc.  Whichever is available first.

    Args:
        file Optional[AnyStr]: File path to save screenshot.
    """
    if name is not None:
        name = _bytes(name)
    lib.TCOD_sys_save_screenshot(name or ffi.NULL)

# custom fullscreen resolution
def sys_force_fullscreen_resolution(width, height):
    """Force a specific resolution in fullscreen.

    Will use the smallest available resolution so that:

    * resolution width >= width and
      resolution width >= root console width * font char width
    * resolution height >= height and
      resolution height >= root console height * font char height

    Args:
        width (int): The desired resolution width.
        height (int): The desired resolution height.
    """
    lib.TCOD_sys_force_fullscreen_resolution(width, height)

def sys_get_current_resolution():
    """Return the current resolution as (width, height)

    Returns:
        Tuple[int,int]: The current resolution.
    """
    w = ffi.new('int *')
    h = ffi.new('int *')
    lib.TCOD_sys_get_current_resolution(w, h)
    return w[0], h[0]

def sys_get_char_size():
    """Return the current fonts character size as (width, height)

    Returns:
        Tuple[int,int]: The current font glyph size in (width, height)
    """
    w = ffi.new('int *')
    h = ffi.new('int *')
    lib.TCOD_sys_get_char_size(w, h)
    return w[0], h[0]

# update font bitmap
def sys_update_char(asciiCode, fontx, fonty, img, x, y):
    """Dynamically update the current frot with img.

    All cells using this asciiCode will be updated
    at the next call to :any:`tcod.console_flush`.

    Args:
        asciiCode (int): Ascii code corresponding to the character to update.
        fontx (int): Left coordinate of the character
                     in the bitmap font (in tiles)
        fonty (int): Top coordinate of the character
                     in the bitmap font (in tiles)
        img (Image): An image containing the new character bitmap.
        x (int): Left pixel of the character in the image.
        y (int): Top pixel of the character in the image.
    """
    lib.TCOD_sys_update_char(_int(asciiCode), fontx, fonty, img, x, y)

def sys_register_SDL_renderer(callback):
    """Register a custom randering function with libtcod.

    The callack will receive a :any:`CData <ffi-cdata>` void* to an
    SDL_Surface* struct.

    The callback is called on every call to :any:`tcod.console_flush`.

    Args:
        callback Callable[[CData], None]:
            A function which takes a single argument.
    """
    with _PropagateException() as propagate:
        @ffi.def_extern(onerror=propagate)
        def _pycall_sdl_hook(sdl_surface):
            callback(sdl_surface)
        lib.TCOD_sys_register_SDL_renderer(lib._pycall_sdl_hook)

def sys_check_for_event(mask, k, m):
    """Check for events.

    mask can be any of the following:

    * tcod.EVENT_NONE
    * tcod.EVENT_KEY_PRESS
    * tcod.EVENT_KEY_RELEASE
    * tcod.EVENT_KEY
    * tcod.EVENT_MOUSE_MOVE
    * tcod.EVENT_MOUSE_PRESS
    * tcod.EVENT_MOUSE_RELEASE
    * tcod.EVENT_MOUSE
    * tcod.EVENT_FINGER_MOVE
    * tcod.EVENT_FINGER_PRESS
    * tcod.EVENT_FINGER_RELEASE
    * tcod.EVENT_FINGER
    * tcod.EVENT_ANY

    Args:
        mask (int): Event types to wait for.
        k (Optional[Key]): A tcod.Key instance which might be updated with
                           an event.  Can be None.
        m (Optional[Mouse]): A tcod.Mouse instance which might be updated
                             with an event.  Can be None.
    """
    return lib.TCOD_sys_check_for_event(mask, _cdata(k), _cdata(m))

def sys_wait_for_event(mask, k, m, flush):
    """Wait for events.

    mask can be any of the following:

    * tcod.EVENT_NONE
    * tcod.EVENT_KEY_PRESS
    * tcod.EVENT_KEY_RELEASE
    * tcod.EVENT_KEY
    * tcod.EVENT_MOUSE_MOVE
    * tcod.EVENT_MOUSE_PRESS
    * tcod.EVENT_MOUSE_RELEASE
    * tcod.EVENT_MOUSE
    * tcod.EVENT_FINGER_MOVE
    * tcod.EVENT_FINGER_PRESS
    * tcod.EVENT_FINGER_RELEASE
    * tcod.EVENT_FINGER
    * tcod.EVENT_ANY

    If flush is True then the buffer will be cleared before waiting. Otherwise
    each available event will be returned in the order they're recieved.

    Args:
        mask (int): Event types to wait for.
        k (Optional[Key]): A tcod.Key instance which might be updated with
                           an event.  Can be None.
        m (Optional[Mouse]): A tcod.Mouse instance which might be updated
                             with an event.  Can be None.
        flush (bool): Clear the event buffer before waiting.
    """
    return lib.TCOD_sys_wait_for_event(mask, _cdata(k), _cdata(m), flush)

__all__ = [_name for _name in list(globals()) if _name[0] != '_']
