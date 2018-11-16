"""This module handles backward compatibility with the ctypes libtcodpy module.
"""

from __future__ import absolute_import as _

import os
import sys

import threading as _threading
from typing import Any, AnyStr, Optional, Sequence, Tuple
import warnings

import numpy as _np
import numpy as np

from tcod.libtcod import *

from tcod._internal import deprecate

from tcod.tcod import _int, _unpack_char_p
from tcod.tcod import _bytes, _unicode, _fmt_bytes, _fmt_unicode
from tcod.tcod import _CDataWrapper
from tcod.tcod import _PropagateException
from tcod.tcod import _console

import tcod.bsp
from tcod.color import Color
import tcod.console
import tcod.image
import tcod.map
import tcod.noise
import tcod.path
import tcod.random

Bsp = tcod.bsp.BSP


class ConsoleBuffer(object):
    """Simple console that allows direct (fast) access to cells. simplifies
    use of the "fill" functions.

    .. deprecated:: 6.0
        Console array attributes perform better than this class.

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
        warnings.warn(
            "Console array attributes perform better than this class.",
            DeprecationWarning,
            stacklevel=2,
        )
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
        if not dest:
            dest = tcod.console.Console._from_cdata(ffi.NULL)
        if (dest.width != self.width or
            dest.height != self.height):
            raise ValueError('ConsoleBuffer.blit: Destination console has an incorrect size.')

        if fill_back:
            bg = dest.bg.ravel()
            bg[0::3] = self.back_r
            bg[1::3] = self.back_g
            bg[2::3] = self.back_b

        if fill_fore:
            fg = dest.fg.ravel()
            fg[0::3] = self.fore_r
            fg[1::3] = self.fore_g
            fg[2::3] = self.fore_b
            dest.ch.ravel()[:] = self.char

class Dice(_CDataWrapper):
    """

    Args:
        nb_dices (int): Number of dice.
        nb_faces (int): Number of sides on a die.
        multiplier (float): Multiplier.
        addsub (float): Addition.

    .. deprecated:: 2.0
        You should make your own dice functions instead of using this class
        which is tied to a CData object.
    """

    def __init__(self, *args, **kargs):
        warnings.warn("Using this class is not recommended.",
                      DeprecationWarning, stacklevel=2)
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


# reverse lookup table for KEY_X attributes, used by Key.__repr__
_LOOKUP_VK = {
    value:'KEY_%s' % key[6:] for key,value in lib.__dict__.items()
    if key.startswith('TCODK')
    }

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

    def __init__(self, vk=0, c=0, text='', pressed=False,
                 lalt=False, lctrl=False, lmeta=False,
                 ralt=False, rctrl=False, rmeta=False, shift=False):
        if isinstance(vk, ffi.CData):
            self.cdata = vk
            return
        self.cdata = ffi.new('TCOD_key_t*')
        self.vk = vk
        self.c = c
        self.text = text
        self.pressed = pressed
        self.lalt = lalt
        self.lctrl = lctrl
        self.lmeta = lmeta
        self.ralt = ralt
        self.rctrl = rctrl
        self.rmeta = rmeta
        self.shift = shift

    def __getattr__(self, attr):
        if attr in self._BOOL_ATTRIBUTES:
            return bool(getattr(self.cdata, attr))
        if attr == 'c':
            return ord(self.cdata.c)
        if attr == 'text':
            return ffi.string(self.cdata.text).decode()
        return super(Key, self).__getattr__(attr)

    def __setattr__(self, attr, value):
        if attr == 'c':
            self.cdata.c = chr(value).encode('latin-1')
        elif attr == 'text':
            self.cdata.text = value.encode()
        else:
            super(Key, self).__setattr__(attr, value)

    def __repr__(self):
        """Return a representation of this Key object."""
        params = []
        params.append('pressed=%r, vk=tcod.%s' %
                      (self.pressed, _LOOKUP_VK[self.vk]))
        if self.c:
            params.append('c=ord(%r)' % chr(self.c))
        if self.text:
            params.append('text=%r' % self.text)
        for attr in ['shift', 'lalt', 'lctrl', 'lmeta',
                     'ralt', 'rctrl', 'rmeta']:
            if getattr(self, attr):
                params.append('%s=%r' % (attr, getattr(self, attr)))
        return 'tcod.Key(%s)' % ', '.join(params)


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

    def __init__(self, x=0, y=0, dx=0, dy=0, cx=0, cy=0, dcx=0, dcy=0,
                 **kargs):
        if isinstance(x, ffi.CData):
            self.cdata = x
            return
        self.cdata = ffi.new('TCOD_mouse_t*')
        self.x = x
        self.y = y
        self.dx = dx
        self.dy = dy
        self.cx = cx
        self.cy = cy
        self.dcx = dcx
        self.dcy = dcy
        for attr, value in kargs.items():
            setattr(self, attr, value)

    def __repr__(self):
        """Return a representation of this Mouse object."""
        params = []
        for attr in ['x', 'y', 'dx', 'dy', 'cx', 'cy', 'dcx', 'dcy']:
            if getattr(self, attr) == 0:
                continue
            params.append('%s=%r' % (attr, getattr(self, attr)))
        for attr in ['lbutton', 'rbutton', 'mbutton',
                     'lbutton_pressed', 'rbutton_pressed', 'mbutton_pressed',
                     'wheel_up', 'wheel_down']:
            if getattr(self, attr):
                params.append('%s=%r' % (attr, getattr(self, attr)))
        return 'tcod.Mouse(%s)' % ', '.join(params)


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
    node.split_once(horizontal, position)

def bsp_split_recursive(node, randomizer, nb, minHSize, minVSize, maxHRatio,
                        maxVRatio):
    """
    .. deprecated:: 2.0
       Use :any:`BSP.split_recursive` instead.
    """
    node.split_recursive(nb, minHSize, minVSize,
                         maxHRatio, maxVRatio, randomizer)

def bsp_resize(node, x, y, w, h):
    """
    .. deprecated:: 2.0
        Assign directly to :any:`BSP` attributes instead.
    """
    node.x = x
    node.y = y
    node.width = w
    node.height = h

def bsp_left(node):
    """
    .. deprecated:: 2.0
       Use :any:`BSP.children` instead.
    """
    return None if not node.children else node.children[0]

def bsp_right(node):
    """
    .. deprecated:: 2.0
       Use :any:`BSP.children` instead.
    """
    return None if not node.children else node.children[1]

def bsp_father(node):
    """
    .. deprecated:: 2.0
       Use :any:`BSP.parent` instead.
    """
    return node.parent

def bsp_is_leaf(node):
    """
    .. deprecated:: 2.0
       Use :any:`BSP.children` instead.
    """
    return not node.children

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

def _bsp_traverse(node_iter, callback, userData):
    """pack callback into a handle for use with the callback
    _pycall_bsp_callback
    """
    for node in node_iter:
        callback(node, userData)

def bsp_traverse_pre_order(node, callback, userData=0):
    """Traverse this nodes hierarchy with a callback.

    .. deprecated:: 2.0
       Use :any:`BSP.walk` instead.
    """
    _bsp_traverse(node._iter_pre_order(), callback, userData)

def bsp_traverse_in_order(node, callback, userData=0):
    """Traverse this nodes hierarchy with a callback.

    .. deprecated:: 2.0
       Use :any:`BSP.walk` instead.
    """
    _bsp_traverse(node._iter_in_order(), callback, userData)

def bsp_traverse_post_order(node, callback, userData=0):
    """Traverse this nodes hierarchy with a callback.

    .. deprecated:: 2.0
       Use :any:`BSP.walk` instead.
    """
    _bsp_traverse(node._iter_post_order(), callback, userData)

def bsp_traverse_level_order(node, callback, userData=0):
    """Traverse this nodes hierarchy with a callback.

    .. deprecated:: 2.0
       Use :any:`BSP.walk` instead.
    """
    _bsp_traverse(node._iter_level_order(), callback, userData)

def bsp_traverse_inverted_level_order(node, callback, userData=0):
    """Traverse this nodes hierarchy with a callback.

    .. deprecated:: 2.0
       Use :any:`BSP.walk` instead.
    """
    _bsp_traverse(node._iter_inverted_level_order(), callback, userData)

def bsp_remove_sons(node):
    """Delete all children of a given node.  Not recommended.

    .. note::
       This function will add unnecessary complexity to your code.
       Don't use it.

    .. deprecated:: 2.0
       BSP deletion is automatic.
    """
    node.children = ()

def bsp_delete(node):
    # type: (Any) -> None
    """Exists for backward compatibility.  Does nothing.

    BSP's created by this library are automatically garbage collected once
    there are no references to the tree.
    This function exists for backwards compatibility.

    .. deprecated:: 2.0
       BSP deletion is automatic.
    """

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
    return Color._new_from_cdata(lib.TCOD_color_lerp(c1, c2, a))

def color_set_hsv(c, h, s, v):
    """Set a color using: hue, saturation, and value parameters.

    Does not return a new Color.  ``c`` is modified inplace.

    Args:
        c (Union[Color, List[Any]]): A Color instance, or a list of any kind.
        h (float): Hue, from 0 to 360.
        s (float): Saturation, from 0 to 1.
        v (float): Value, from 0 to 1.
    """
    new_color = ffi.new('TCOD_color_t*')
    lib.TCOD_color_set_HSV(new_color, h, s, v)
    c[:] = new_color.r, new_color.g, new_color.b

def color_get_hsv(c):
    """Return the (hue, saturation, value) of a color.

    Args:
        c (Union[Tuple[int, int, int], Sequence[int]]):
            An (r, g, b) sequence or Color instance.

    Returns:
        Tuple[float, float, float]:
            A tuple with (hue, saturation, value) values, from 0 to 1.
    """
    hsv = ffi.new('float [3]')
    lib.TCOD_color_get_HSV(c, hsv, hsv + 1, hsv + 2)
    return hsv[0], hsv[1], hsv[2]

def color_scale_HSV(c, scoef, vcoef):
    """Scale a color's saturation and value.

    Does not return a new Color.  ``c`` is modified inplace.

    Args:
        c (Union[Color, List[int]]): A Color instance, or an [r, g, b] list.
        scoef (float): Saturation multiplier, from 0 to 1.
                       Use 1 to keep current saturation.
        vcoef (float): Value multiplier, from 0 to 1.
                       Use 1 to keep current value.
    """
    color_p = ffi.new('TCOD_color_t*')
    color_p.r, color_p.g, color_p.b = c.r, c.g, c.b
    lib.TCOD_color_scale_HSV(color_p, scoef, vcoef)
    c[:] = color_p.r, color_p.g, color_p.b

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
    return [Color._new_from_cdata(cdata) for cdata in cres]


def console_init_root(
        w: int, h: int, title: Optional[AnyStr]=None, fullscreen: bool=False,
        renderer: Optional[int]=None, order: str='C') -> tcod.console.Console:
    """Set up the primary display and return the root console.

    `w` and `h` are the columns and rows of the new window (in tiles.)

    `title` is an optional string to display on the windows title bar.

    `fullscreen` determines if the window will start in fullscreen.  Fullscreen
    mode is unreliable unless the renderer is set to `RENDERER_SDL2` or
    `RENDERER_OPENGL2`.

    `renderer` is the rendering back-end that libtcod will use.  Options are:

    * `tcod.RENDERER_SDL`
    * `tcod.RENDERER_OPENGL`
    * `tcod.RENDERER_GLSL`
    * `tcod.RENDERER_SDL2`
    * `tcod.RENDERER_OPENGL2`

    `order` will affect how the array attributes of the returned root console
    are indexed.  `order='C'` is the default, but `order='F'` is recommended.

    .. versionchanged:: 4.3
        Added `order` parameter.
        `title` parameter is now optional.

    .. versionchanged:: 8.0
        The default `renderer` is now automatic instead of always being
        `RENDERER_SDL`.
    """
    if title is None:
        # Use the scripts filename as the title.
        title = os.path.basename(sys.argv[0])
    if renderer is None:
        renderer = RENDERER_GLSL # Stable for now.
    lib.TCOD_console_init_root(w, h, _bytes(title), fullscreen, renderer)
    return tcod.console.Console._get_root(order)


def console_set_custom_font(fontFile: AnyStr, flags:
                            int=FONT_LAYOUT_ASCII_INCOL,
                            nb_char_horiz: int=0, nb_char_vertic: int=0):
    """Load the custom font file at `fontFile`.

    Call this before function before calling :any:`tcod.console_init_root`.

    Flags can be a mix of the following:

    * tcod.FONT_LAYOUT_ASCII_INCOL
    * tcod.FONT_LAYOUT_ASCII_INROW
    * tcod.FONT_TYPE_GREYSCALE
    * tcod.FONT_TYPE_GRAYSCALE
    * tcod.FONT_LAYOUT_TCOD

    `nb_char_horiz` and `nb_char_vertic` are the columns and rows of the font
    file respectfully.
    """
    if not os.path.exists(fontFile):
        raise RuntimeError("File not found:\n\t%s"
                           % (os.path.realpath(fontFile),))
    lib.TCOD_console_set_custom_font(_bytes(fontFile), flags,
                                     nb_char_horiz, nb_char_vertic)


def console_get_width(con):
    """Return the width of a console.

    Args:
        con (Console): Any Console instance.

    Returns:
        int: The width of a Console.

    .. deprecated:: 2.0
        Use `Console.width` instead.
    """
    return lib.TCOD_console_get_width(_console(con))

def console_get_height(con):
    """Return the height of a console.

    Args:
        con (Console): Any Console instance.

    Returns:
        int: The height of a Console.

    .. deprecated:: 2.0
        Use `Console.height` instead.
    """
    return lib.TCOD_console_get_height(_console(con))

def console_map_ascii_code_to_font(asciiCode, fontCharX, fontCharY):
    """Set a character code to new coordinates on the tile-set.

    `asciiCode` must be within the bounds created during the initialization of
    the loaded tile-set.  For example, you can't use 255 here unless you have a
    256 tile tile-set loaded.  This applies to all functions in this group.

    Args:
        asciiCode (int): The character code to change.
        fontCharX (int): The X tile coordinate on the loaded tileset.
                         0 is the leftmost tile.
        fontCharY (int): The Y tile coordinate on the loaded tileset.
                         0 is the topmost tile.
    """
    lib.TCOD_console_map_ascii_code_to_font(_int(asciiCode), fontCharX,
                                                              fontCharY)

def console_map_ascii_codes_to_font(firstAsciiCode, nbCodes, fontCharX,
                                    fontCharY):
    """Remap a contiguous set of codes to a contiguous set of tiles.

    Both the tile-set and character codes must be contiguous to use this
    function.  If this is not the case you may want to use
    :any:`console_map_ascii_code_to_font`.

    Args:
        firstAsciiCode (int): The starting character code.
        nbCodes (int): The length of the contiguous set.
        fontCharX (int): The starting X tile coordinate on the loaded tileset.
                         0 is the leftmost tile.
        fontCharY (int): The starting Y tile coordinate on the loaded tileset.
                         0 is the topmost tile.

    """
    lib.TCOD_console_map_ascii_codes_to_font(_int(firstAsciiCode), nbCodes,
                                              fontCharX, fontCharY)

def console_map_string_to_font(s, fontCharX, fontCharY):
    """Remap a string of codes to a contiguous set of tiles.

    Args:
        s (AnyStr): A string of character codes to map to new values.
                    The null character `'\\x00'` will prematurely end this
                    function.
        fontCharX (int): The starting X tile coordinate on the loaded tileset.
                         0 is the leftmost tile.
        fontCharY (int): The starting Y tile coordinate on the loaded tileset.
                         0 is the topmost tile.
    """
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

def console_has_mouse_focus():
    """Return True if the window has mouse focus."""
    return lib.TCOD_console_has_mouse_focus()

def console_is_active():
    """Return True if the window has keyboard focus."""
    return lib.TCOD_console_is_active()

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
    lib.TCOD_console_set_default_background(_console(con), col)

def console_set_default_foreground(con, col):
    """Change the default foreground color for a console.

    Args:
        con (Console): Any Console instance.
        col (Union[Tuple[int, int, int], Sequence[int]]):
            An (r, g, b) sequence or Color instance.
    """
    lib.TCOD_console_set_default_foreground(_console(con), col)

def console_clear(con):
    """Reset a console to its default colors and the space character.

    Args:
        con (Console): Any Console instance.

    .. seealso::
       :any:`console_set_default_background`
       :any:`console_set_default_foreground`
    """
    return lib.TCOD_console_clear(_console(con))

def console_put_char(con, x, y, c, flag=BKGND_DEFAULT):
    """Draw the character c at x,y using the default colors and a blend mode.

    Args:
        con (Console): Any Console instance.
        x (int): Character x position from the left.
        y (int): Character y position from the top.
        c (Union[int, AnyStr]): Character to draw, can be an integer or string.
        flag (int): Blending mode to use, defaults to BKGND_DEFAULT.
    """
    lib.TCOD_console_put_char(_console(con), x, y, _int(c), flag)

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
    lib.TCOD_console_put_char_ex(_console(con), x, y, _int(c), fore, back)

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
    lib.TCOD_console_set_char_background(_console(con), x, y, col, flag)

@deprecate("Directly access a consoles foreground color with `console.fg`")
def console_set_char_foreground(con, x, y, col):
    """Change the foreground color of x,y to col.

    Args:
        con (Console): Any Console instance.
        x (int): Character x position from the left.
        y (int): Character y position from the top.
        col (Union[Tuple[int, int, int], Sequence[int]]):
            An (r, g, b) sequence or Color instance.
    """
    lib.TCOD_console_set_char_foreground(_console(con), x, y, col)

@deprecate("Directly access a consoles characters with `console.ch`")
def console_set_char(con, x, y, c):
    """Change the character at x,y to c, keeping the current colors.

    Args:
        con (Console): Any Console instance.
        x (int): Character x position from the left.
        y (int): Character y position from the top.
        c (Union[int, AnyStr]): Character to draw, can be an integer or string.
    """
    lib.TCOD_console_set_char(_console(con), x, y, _int(c))

def console_set_background_flag(con, flag):
    """Change the default blend mode for this console.

    Args:
        con (Console): Any Console instance.
        flag (int): Blend mode to use by default.
    """
    lib.TCOD_console_set_background_flag(_console(con), flag)

def console_get_background_flag(con):
    """Return this consoles current blend mode.

    Args:
        con (Console): Any Console instance.
    """
    return lib.TCOD_console_get_background_flag(_console(con))

def console_set_alignment(con, alignment):
    """Change this consoles current alignment mode.

    * tcod.LEFT
    * tcod.CENTER
    * tcod.RIGHT

    Args:
        con (Console): Any Console instance.
        alignment (int):
    """
    lib.TCOD_console_set_alignment(_console(con), alignment)

def console_get_alignment(con):
    """Return this consoles current alignment mode.

    Args:
        con (Console): Any Console instance.
    """
    return lib.TCOD_console_get_alignment(_console(con))

def console_print(con, x, y, fmt):
    """Print a color formatted string on a console.

    Args:
        con (Console): Any Console instance.
        x (int): Character x position from the left.
        y (int): Character y position from the top.
        fmt (AnyStr): A unicode or bytes string optionaly using color codes.
    """
    lib.TCOD_console_print(_console(con), x, y, _fmt_unicode(fmt))

def console_print_ex(con, x, y, flag, alignment, fmt):
    """Print a string on a console using a blend mode and alignment mode.

    Args:
        con (Console): Any Console instance.
        x (int): Character x position from the left.
        y (int): Character y position from the top.
    """
    lib.TCOD_console_print_ex(_console(con),
                               x, y, flag, alignment, _fmt_unicode(fmt))

def console_print_rect(con, x, y, w, h, fmt):
    """Print a string constrained to a rectangle.

    If h > 0 and the bottom of the rectangle is reached,
    the string is truncated. If h = 0,
    the string is only truncated if it reaches the bottom of the console.



    Returns:
        int: The number of lines of text once word-wrapped.
    """
    return lib.TCOD_console_print_rect(
        _console(con), x, y, w, h, _fmt_unicode(fmt))

def console_print_rect_ex(con, x, y, w, h, flag, alignment, fmt):
    """Print a string constrained to a rectangle with blend and alignment.

    Returns:
        int: The number of lines of text once word-wrapped.
    """
    return lib.TCOD_console_print_rect_ex(
        _console(con), x, y, w, h, flag, alignment, _fmt_unicode(fmt))

def console_get_height_rect(con, x, y, w, h, fmt):
    """Return the height of this text once word-wrapped into this rectangle.

    Returns:
        int: The number of lines of text once word-wrapped.
    """
    return lib.TCOD_console_get_height_rect(
        _console(con), x, y, w, h, _fmt_unicode(fmt))

def console_rect(con, x, y, w, h, clr, flag=BKGND_DEFAULT):
    """Draw a the background color on a rect optionally clearing the text.

    If clr is True the affected tiles are changed to space character.
    """
    lib.TCOD_console_rect(_console(con), x, y, w, h, clr, flag)

def console_hline(con, x, y, l, flag=BKGND_DEFAULT):
    """Draw a horizontal line on the console.

    This always uses the character 196, the horizontal line character.
    """
    lib.TCOD_console_hline(_console(con), x, y, l, flag)

def console_vline(con, x, y, l, flag=BKGND_DEFAULT):
    """Draw a vertical line on the console.

    This always uses the character 179, the vertical line character.
    """
    lib.TCOD_console_vline(_console(con), x, y, l, flag)

def console_print_frame(con, x: int, y: int, w: int, h: int, clear: bool=True,
                        flag: int=BKGND_DEFAULT, fmt: bytes=b''):
    """Draw a framed rectangle with optinal text.

    This uses the default background color and blend mode to fill the
    rectangle and the default foreground to draw the outline.

    `fmt` will be printed on the inside of the rectangle, word-wrapped.
    If `fmt` is empty then no title will be drawn.
    """
    if fmt:
        fmt = _fmt_bytes(fmt)
    else:
        fmt = ffi.NULL
    lib.TCOD_console_print_frame(_console(con), x, y, w, h, clear, flag, fmt)

def console_set_color_control(con, fore, back):
    """Configure :any:`color controls`.

    Args:
        con (int): :any:`Color control` constant to modify.
        fore (Union[Tuple[int, int, int], Sequence[int]]):
            An (r, g, b) sequence or Color instance.
        back (Union[Tuple[int, int, int], Sequence[int]]):
            An (r, g, b) sequence or Color instance.
    """
    lib.TCOD_console_set_color_control(con, fore, back)

def console_get_default_background(con):
    """Return this consoles default background color."""
    return Color._new_from_cdata(
        lib.TCOD_console_get_default_background(_console(con)))

def console_get_default_foreground(con):
    """Return this consoles default foreground color."""
    return Color._new_from_cdata(
        lib.TCOD_console_get_default_foreground(_console(con)))

@deprecate("Directly access a consoles background color with `console.bg`")
def console_get_char_background(con, x, y):
    """Return the background color at the x,y of this console."""
    return Color._new_from_cdata(
        lib.TCOD_console_get_char_background(_console(con), x, y))

@deprecate("Directly access a consoles foreground color with `console.fg`")
def console_get_char_foreground(con, x, y):
    """Return the foreground color at the x,y of this console."""
    return Color._new_from_cdata(
        lib.TCOD_console_get_char_foreground(_console(con), x, y))

@deprecate("Directly access a consoles characters with `console.ch`")
def console_get_char(con, x, y):
    """Return the character at the x,y of this console."""
    return lib.TCOD_console_get_char(_console(con), x, y)

def console_set_fade(fade, fadingColor):
    lib.TCOD_console_set_fade(fade, fadingColor)

def console_get_fade():
    return lib.TCOD_console_get_fade()

def console_get_fading_color():
    return Color._new_from_cdata(lib.TCOD_console_get_fading_color())

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

# using offscreen consoles
def console_new(w, h):
    """Return an offscreen console of size: w,h."""
    return tcod.console.Console(w, h)

def console_from_file(filename):
    """Return a new console object from a filename.

    The file format is automactially determined.  This can load REXPaint `.xp`,
    ASCII Paint `.apf`, or Non-delimited ASCII `.asc` files.

    Args:
        filename (Text): The path to the file, as a string.

    Returns: A new :any`Console` instance.
    """
    return tcod.console.Console._from_cdata(
        lib.TCOD_console_from_file(filename.encode('utf-8')))

def console_blit(src, x, y, w, h, dst, xdst, ydst, ffade=1.0,bfade=1.0):
    """Blit the console src from x,y,w,h to console dst at xdst,ydst."""
    lib.TCOD_console_blit(
        _console(src), x, y, w, h,
        _console(dst), xdst, ydst, ffade, bfade)

def console_set_key_color(con, col):
    """Set a consoles blit transparent color."""
    lib.TCOD_console_set_key_color(_console(con), col)
    if hasattr(con, 'set_key_color'):
        con.set_key_color(col)

def console_delete(con):
    # type: (Console) -> None
    """Closes the window if `con` is the root console.

    libtcod objects are automatically garbage collected once they go out of
    scope.

    This function exists for backwards compatibility.
    """
    # type: (Any) -> None
    con = _console(con)
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
    if (isinstance(r, _np.ndarray) and isinstance(g, _np.ndarray) and
            isinstance(b, _np.ndarray)):
        #numpy arrays, use numpy's ctypes functions
        r = _np.ascontiguousarray(r, dtype=_np.intc)
        g = _np.ascontiguousarray(g, dtype=_np.intc)
        b = _np.ascontiguousarray(b, dtype=_np.intc)
        cr = ffi.cast('int *', r.ctypes.data)
        cg = ffi.cast('int *', g.ctypes.data)
        cb = ffi.cast('int *', b.ctypes.data)
    else:
        # otherwise convert using ffi arrays
        cr = ffi.new('int[]', r)
        cg = ffi.new('int[]', g)
        cb = ffi.new('int[]', b)

    lib.TCOD_console_fill_foreground(_console(con), cr, cg, cb)

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
    if (isinstance(r, _np.ndarray) and isinstance(g, _np.ndarray) and
            isinstance(b, _np.ndarray)):
        #numpy arrays, use numpy's ctypes functions
        r = _np.ascontiguousarray(r, dtype=_np.intc)
        g = _np.ascontiguousarray(g, dtype=_np.intc)
        b = _np.ascontiguousarray(b, dtype=_np.intc)
        cr = ffi.cast('int *', r.ctypes.data)
        cg = ffi.cast('int *', g.ctypes.data)
        cb = ffi.cast('int *', b.ctypes.data)
    else:
        # otherwise convert using ffi arrays
        cr = ffi.new('int[]', r)
        cg = ffi.new('int[]', g)
        cb = ffi.new('int[]', b)

    lib.TCOD_console_fill_background(_console(con), cr, cg, cb)

def console_fill_char(con,arr):
    """Fill the character tiles of a console with an array.

    Args:
        con (Console): Any Console instance.
        arr (Sequence[int]): An array of integers with a length of width*height.
    """
    if isinstance(arr, _np.ndarray):
        #numpy arrays, use numpy's ctypes functions
        arr = _np.ascontiguousarray(arr, dtype=_np.intc)
        carr = ffi.cast('int *', arr.ctypes.data)
    else:
        #otherwise convert using the ffi module
        carr = ffi.new('int[]', arr)

    lib.TCOD_console_fill_char(_console(con), carr)

def console_load_asc(con, filename):
    """Update a console from a non-delimited ASCII `.asc` file."""
    return lib.TCOD_console_load_asc(_console(con), filename.encode('utf-8'))

def console_save_asc(con, filename):
    """Save a console to a non-delimited ASCII `.asc` file."""
    return lib.TCOD_console_save_asc(_console(con), filename.encode('utf-8'))

def console_load_apf(con, filename):
    """Update a console from an ASCII Paint `.apf` file."""
    return lib.TCOD_console_load_apf(_console(con), filename.encode('utf-8'))

def console_save_apf(con, filename):
    """Save a console to an ASCII Paint `.apf` file."""
    return lib.TCOD_console_save_apf(_console(con), filename.encode('utf-8'))

def console_load_xp(con, filename):
    """Update a console from a REXPaint `.xp` file."""
    return lib.TCOD_console_load_xp(_console(con), filename.encode('utf-8'))

def console_save_xp(con, filename, compress_level=9):
    """Save a console to a REXPaint `.xp` file."""
    return lib.TCOD_console_save_xp(
        _console(con),
        filename.encode('utf-8'),
        compress_level,
        )

def console_from_xp(filename):
    """Return a single console from a REXPaint `.xp` file."""
    return tcod.console.Console._from_cdata(
        lib.TCOD_console_from_xp(filename.encode('utf-8')))

def console_list_load_xp(filename):
    """Return a list of consoles from a REXPaint `.xp` file."""
    tcod_list = lib.TCOD_console_list_from_xp(filename.encode('utf-8'))
    if tcod_list == ffi.NULL:
        return None
    try:
        python_list = []
        lib.TCOD_list_reverse(tcod_list)
        while not lib.TCOD_list_is_empty(tcod_list):
            python_list.append(
                tcod.console.Console._from_cdata(lib.TCOD_list_pop(tcod_list)),
                )
        return python_list
    finally:
        lib.TCOD_list_delete(tcod_list)

def console_list_save_xp(console_list, filename, compress_level=9):
    """Save a list of consoles to a REXPaint `.xp` file."""
    tcod_list = lib.TCOD_list_new()
    try:
        for console in console_list:
            lib.TCOD_list_push(tcod_list, _console(console))
        return lib.TCOD_console_list_save_xp(
            tcod_list, filename.encode('utf-8'), compress_level
            )
    finally:
        lib.TCOD_list_delete(tcod_list)

def path_new_using_map(m, dcost=1.41):
    """Return a new AStar using the given Map.

    Args:
        m (Map): A Map instance.
        dcost (float): The path-finding cost of diagonal movement.
                       Can be set to 0 to disable diagonal movement.
    Returns:
        AStar: A new AStar instance.
    """
    return tcod.path.AStar(m, dcost)

def path_new_using_function(w, h, func, userData=0, dcost=1.41):
    """Return a new AStar using the given callable function.

    Args:
        w (int): Clipping width.
        h (int): Clipping height.
        func (Callable[[int, int, int, int, Any], float]):
        userData (Any):
        dcost (float): A multiplier for the cost of diagonal movement.
                       Can be set to 0 to disable diagonal movement.
    Returns:
        AStar: A new AStar instance.
    """
    return tcod.path.AStar(
        tcod.path._EdgeCostFunc((func, userData), (w, h)),
        dcost,
        )

def path_compute(p, ox, oy, dx, dy):
    """Find a path from (ox, oy) to (dx, dy).  Return True if path is found.

    Args:
        p (AStar): An AStar instance.
        ox (int): Starting x position.
        oy (int): Starting y position.
        dx (int): Destination x position.
        dy (int): Destination y position.
    Returns:
        bool: True if a valid path was found.  Otherwise False.
    """
    return lib.TCOD_path_compute(p._path_c, ox, oy, dx, dy)

def path_get_origin(p):
    """Get the current origin position.

    This point moves when :any:`path_walk` returns the next x,y step.

    Args:
        p (AStar): An AStar instance.
    Returns:
        Tuple[int, int]: An (x, y) point.
    """
    x = ffi.new('int *')
    y = ffi.new('int *')
    lib.TCOD_path_get_origin(p._path_c, x, y)
    return x[0], y[0]

def path_get_destination(p):
    """Get the current destination position.

    Args:
        p (AStar): An AStar instance.
    Returns:
        Tuple[int, int]: An (x, y) point.
    """
    x = ffi.new('int *')
    y = ffi.new('int *')
    lib.TCOD_path_get_destination(p._path_c, x, y)
    return x[0], y[0]

def path_size(p):
    """Return the current length of the computed path.

    Args:
        p (AStar): An AStar instance.
    Returns:
        int: Length of the path.
    """
    return lib.TCOD_path_size(p._path_c)

def path_reverse(p):
    """Reverse the direction of a path.

    This effectively swaps the origin and destination points.

    Args:
        p (AStar): An AStar instance.
    """
    lib.TCOD_path_reverse(p._path_c)

def path_get(p, idx):
    """Get a point on a path.

    Args:
        p (AStar): An AStar instance.
        idx (int): Should be in range: 0 <= inx < :any:`path_size`
    """
    x = ffi.new('int *')
    y = ffi.new('int *')
    lib.TCOD_path_get(p._path_c, idx, x, y)
    return x[0], y[0]

def path_is_empty(p):
    """Return True if a path is empty.

    Args:
        p (AStar): An AStar instance.
    Returns:
        bool: True if a path is empty.  Otherwise False.
    """
    return lib.TCOD_path_is_empty(p._path_c)

def path_walk(p, recompute):
    """Return the next (x, y) point in a path, or (None, None) if it's empty.

    When ``recompute`` is True and a previously valid path reaches a point
    where it is now blocked, a new path will automatically be found.

    Args:
        p (AStar): An AStar instance.
        recompute (bool): Recompute the path automatically.
    Returns:
        Union[Tuple[int, int], Tuple[None, None]]:
            A single (x, y) point, or (None, None)
    """
    x = ffi.new('int *')
    y = ffi.new('int *')
    if lib.TCOD_path_walk(p._path_c, x, y, recompute):
        return x[0], y[0]
    return None,None

def path_delete(p):
    # type (Any) -> None
    """Does nothing. libtcod objects are managed by Python's garbage collector.

    This function exists for backwards compatibility with libtcodpy.
    """

def dijkstra_new(m, dcost=1.41):
    return tcod.path.Dijkstra(m, dcost)

def dijkstra_new_using_function(w, h, func, userData=0, dcost=1.41):
    return tcod.path.Dijkstra(
        tcod.path._EdgeCostFunc((func, userData), (w, h)),
        dcost,
        )

def dijkstra_compute(p, ox, oy):
    lib.TCOD_dijkstra_compute(p._path_c, ox, oy)

def dijkstra_path_set(p, x, y):
    return lib.TCOD_dijkstra_path_set(p._path_c, x, y)

def dijkstra_get_distance(p, x, y):
    return lib.TCOD_dijkstra_get_distance(p._path_c, x, y)

def dijkstra_size(p):
    return lib.TCOD_dijkstra_size(p._path_c)

def dijkstra_reverse(p):
    lib.TCOD_dijkstra_reverse(p._path_c)

def dijkstra_get(p, idx):
    x = ffi.new('int *')
    y = ffi.new('int *')
    lib.TCOD_dijkstra_get(p._path_c, idx, x, y)
    return x[0], y[0]

def dijkstra_is_empty(p):
    return lib.TCOD_dijkstra_is_empty(p._path_c)

def dijkstra_path_walk(p):
    x = ffi.new('int *')
    y = ffi.new('int *')
    if lib.TCOD_dijkstra_path_walk(p._path_c, x, y):
        return x[0], y[0]
    return None,None

def dijkstra_delete(p):
    # type (Any) -> None
    """Does nothing. libtcod objects are managed by Python's garbage collector.

    This function exists for backwards compatibility with libtcodpy.
    """

def _heightmap_cdata(array: np.ndarray) -> ffi.CData:
    """Return a new TCOD_heightmap_t instance using an array.

    Formatting is verified during this function.
    """
    if array.flags['F_CONTIGUOUS']:
        array = array.transpose()
    if not array.flags['C_CONTIGUOUS']:
        raise ValueError('array must be a contiguous segment.')
    if array.dtype != _np.float32:
        raise ValueError('array dtype must be float32, not %r' % array.dtype)
    width, height = array.shape
    pointer = ffi.cast('float *', array.ctypes.data)
    return ffi.new('TCOD_heightmap_t *', (width, height, pointer))

def heightmap_new(w: int, h: int, order: str='C') -> np.ndarray:
    """Return a new numpy.ndarray formatted for use with heightmap functions.

    `w` and `h` are the width and height of the array.

    `order` is given to the new NumPy array, it can be 'C' or 'F'.

    You can pass a NumPy array to any heightmap function as long as all the
    following are true::
    * The array is 2 dimensional.
    * The array has the C_CONTIGUOUS or F_CONTIGUOUS flag.
    * The array's dtype is :any:`dtype.float32`.

    The returned NumPy array will fit all these conditions.

    .. versionchanged:: 8.1
        Added the `order` parameter.
    """
    if order == 'C':
        return np.zeros((h, w), _np.float32, order='C')
    elif order == 'F':
        return np.zeros((w, h), _np.float32, order='F')
    else:
        raise ValueError("Invalid order parameter, should be 'C' or 'F'.")

def heightmap_set_value(hm: np.ndarray, x: int, y: int, value: float) -> None:
    """Set the value of a point on a heightmap.

    .. deprecated:: 2.0
        `hm` is a NumPy array, so values should be assigned to it directly.
    """
    if hm.flags['C_CONTIGUOUS']:
        warnings.warn("Assign to this heightmap with hm[i,j] = value\n"
                      "consider using order='F'",
                      DeprecationWarning, stacklevel=2)
        hm[y, x] = value
    elif hm.flags['F_CONTIGUOUS']:
        warnings.warn("Assign to this heightmap with hm[x,y] = value",
                      DeprecationWarning, stacklevel=2)
        hm[x, y] = value
    else:
        raise ValueError("This array is not contiguous.")

@deprecate("Add a scalar to an array using `hm[:] += value`")
def heightmap_add(hm: np.ndarray, value: float) -> None:
    """Add value to all values on this heightmap.

    Args:
        hm (numpy.ndarray): A numpy.ndarray formatted for heightmap functions.
        value (float): A number to add to this heightmap.

    .. deprecated:: 2.0
        Do ``hm[:] += value`` instead.
    """
    hm[:] += value

@deprecate("Multiply an array with a scaler using `hm[:] *= value`")
def heightmap_scale(hm: np.ndarray, value: float) -> None:
    """Multiply all items on this heightmap by value.

    Args:
        hm (numpy.ndarray): A numpy.ndarray formatted for heightmap functions.
        value (float): A number to scale this heightmap by.

    .. deprecated:: 2.0
        Do ``hm[:] *= value`` instead.
    """
    hm[:] *= value

@deprecate("Clear an array with`hm[:] = 0`")
def heightmap_clear(hm: np.ndarray) -> None:
    """Add value to all values on this heightmap.

    Args:
        hm (numpy.ndarray): A numpy.ndarray formatted for heightmap functions.

    .. deprecated:: 2.0
        Do ``hm.array[:] = 0`` instead.
    """
    hm[:] = 0

@deprecate("Clamp array values using `hm.clip(mi, ma)`")
def heightmap_clamp(hm: np.ndarray, mi: float, ma: float) -> None:
    """Clamp all values on this heightmap between ``mi`` and ``ma``

    Args:
        hm (numpy.ndarray): A numpy.ndarray formatted for heightmap functions.
        mi (float): The lower bound to clamp to.
        ma (float): The upper bound to clamp to.

    .. deprecated:: 2.0
        Do ``hm.clip(mi, ma)`` instead.
    """
    hm.clip(mi, ma)

@deprecate("Copy an array using `hm2[:] = hm1[:]`, or `hm1.copy()`")
def heightmap_copy(hm1: np.ndarray, hm2: np.ndarray) -> None:
    """Copy the heightmap ``hm1`` to ``hm2``.

    Args:
        hm1 (numpy.ndarray): The source heightmap.
        hm2 (numpy.ndarray): The destination heightmap.

    .. deprecated:: 2.0
        Do ``hm2[:] = hm1[:]`` instead.
    """
    hm2[:] = hm1[:]

def heightmap_normalize(hm: np.ndarray,  mi: float=0.0, ma: float=1.0) -> None:
    """Normalize heightmap values between ``mi`` and ``ma``.

    Args:
        mi (float): The lowest value after normalization.
        ma (float): The highest value after normalization.
    """
    lib.TCOD_heightmap_normalize(_heightmap_cdata(hm), mi, ma)

def heightmap_lerp_hm(hm1: np.ndarray, hm2: np.ndarray, hm3: np.ndarray,
                      coef: float) -> None:
    """Perform linear interpolation between two heightmaps storing the result
    in ``hm3``.

    This is the same as doing ``hm3[:] = hm1[:] + (hm2[:] - hm1[:]) * coef``

    Args:
        hm1 (numpy.ndarray): The first heightmap.
        hm2 (numpy.ndarray): The second heightmap to add to the first.
        hm3 (numpy.ndarray): A destination heightmap to store the result.
        coef (float): The linear interpolation coefficient.
    """
    lib.TCOD_heightmap_lerp_hm(_heightmap_cdata(hm1), _heightmap_cdata(hm2),
                               _heightmap_cdata(hm3), coef)

@deprecate("Add 2 arrays using `hm3 = hm1 + hm2`")
def heightmap_add_hm(hm1: np.ndarray, hm2: np.ndarray,
                     hm3: np.ndarray) -> None:
    """Add two heightmaps together and stores the result in ``hm3``.

    Args:
        hm1 (numpy.ndarray): The first heightmap.
        hm2 (numpy.ndarray): The second heightmap to add to the first.
        hm3 (numpy.ndarray): A destination heightmap to store the result.

    .. deprecated:: 2.0
        Do ``hm3[:] = hm1[:] + hm2[:]`` instead.
    """
    hm3[:] = hm1[:] + hm2[:]

@deprecate("Multiply 2 arrays using `hm3 = hm1 * hm2`")
def heightmap_multiply_hm(hm1: np.ndarray, hm2: np.ndarray,
                          hm3: np.ndarray) -> None:
    """Multiplies two heightmap's together and stores the result in ``hm3``.

    Args:
        hm1 (numpy.ndarray): The first heightmap.
        hm2 (numpy.ndarray): The second heightmap to multiply with the first.
        hm3 (numpy.ndarray): A destination heightmap to store the result.

    .. deprecated:: 2.0
        Do ``hm3[:] = hm1[:] * hm2[:]`` instead.
        Alternatively you can do ``HeightMap(hm1.array[:] * hm2.array[:])``.
    """
    hm3[:] = hm1[:] * hm2[:]

def heightmap_add_hill(hm: np.ndarray, x: float, y: float, radius: float,
                       height: float) -> None:
    """Add a hill (a half spheroid) at given position.

    If height == radius or -radius, the hill is a half-sphere.

    Args:
        hm (numpy.ndarray): A numpy.ndarray formatted for heightmap functions.
        x (float): The x position at the center of the new hill.
        y (float): The y position at the center of the new hill.
        radius (float): The size of the new hill.
        height (float): The height or depth of the new hill.
    """
    lib.TCOD_heightmap_add_hill(_heightmap_cdata(hm), x, y, radius, height)

def heightmap_dig_hill(hm: np.ndarray, x: float, y: float, radius: float,
                       height: float) -> None:
    """

    This function takes the highest value (if height > 0) or the lowest
    (if height < 0) between the map and the hill.

    It's main goal is to carve things in maps (like rivers) by digging hills along a curve.

    Args:
        hm (numpy.ndarray): A numpy.ndarray formatted for heightmap functions.
        x (float): The x position at the center of the new carving.
        y (float): The y position at the center of the new carving.
        radius (float): The size of the carving.
        height (float): The height or depth of the hill to dig out.
    """
    lib.TCOD_heightmap_dig_hill(_heightmap_cdata(hm), x, y, radius, height)

def heightmap_rain_erosion(hm: np.ndarray, nbDrops: int, erosionCoef: float,
                           sedimentationCoef: float,
                           rnd: Optional[tcod.random.Random]=None) -> None:
    """Simulate the effect of rain drops on the terrain, resulting in erosion.

    ``nbDrops`` should be at least hm.size.

    Args:
        hm (numpy.ndarray): A numpy.ndarray formatted for heightmap functions.
        nbDrops (int): Number of rain drops to simulate.
        erosionCoef (float): Amount of ground eroded on the drop's path.
        sedimentationCoef (float): Amount of ground deposited when the drops
                                   stops to flow.
        rnd (Optional[Random]): A tcod.Random instance, or None.
    """
    lib.TCOD_heightmap_rain_erosion(
        _heightmap_cdata(hm), nbDrops, erosionCoef,
        sedimentationCoef, rnd.random_c if rnd else ffi.NULL
    )

def heightmap_kernel_transform(hm: np.ndarray, kernelsize: int,
                               dx: Sequence[int], dy: Sequence[int],
                               weight: Sequence[float],
                               minLevel: float, maxLevel: float) -> None:
    """Apply a generic transformation on the map, so that each resulting cell
    value is the weighted sum of several neighbour cells.

    This can be used to smooth/sharpen the map.

    Args:
        hm (numpy.ndarray): A numpy.ndarray formatted for heightmap functions.
        kernelsize (int): Should be set to the length of the parameters::
                          dx, dy, and weight.
        dx (Sequence[int]): A sequence of x coorinates.
        dy (Sequence[int]): A sequence of y coorinates.
        weight (Sequence[float]): A sequence of kernelSize cells weight.
                                  The value of each neighbour cell is scaled by
                                  its corresponding weight
        minLevel (float): No transformation will apply to cells
                          below this value.
        maxLevel (float): No transformation will apply to cells
                          above this value.

    See examples below for a simple horizontal smoothing kernel :
    replace value(x,y) with
    0.33*value(x-1,y) + 0.33*value(x,y) + 0.33*value(x+1,y).
    To do this, you need a kernel of size 3
    (the sum involves 3 surrounding cells).
    The dx,dy array will contain:

    * dx=-1, dy=0 for cell (x-1, y)
    * dx=1, dy=0 for cell (x+1, y)
    * dx=0, dy=0 for cell (x, y)
    * The weight array will contain 0.33 for each cell.

    Example:
        >>> import numpy as np
        >>> heightmap = np.zeros((3, 3), dtype=np.float32)
        >>> heightmap[:,1] = 1
        >>> dx = [-1, 1, 0]
        >>> dy = [0, 0, 0]
        >>> weight = [0.33, 0.33, 0.33]
        >>> tcod.heightmap_kernel_transform(heightmap, 3, dx, dy, weight,
        ...                                 0.0, 1.0)
    """
    cdx = ffi.new('int[]', dx)
    cdy = ffi.new('int[]', dy)
    cweight = ffi.new('float[]', weight)
    lib.TCOD_heightmap_kernel_transform(_heightmap_cdata(hm), kernelsize,
                                        cdx, cdy, cweight, minLevel, maxLevel)

def heightmap_add_voronoi(hm: np.ndarray, nbPoints: Any, nbCoef: int,
                          coef: Sequence[float],
                          rnd: Optional[tcod.random.Random]=None) -> None:
    """Add values from a Voronoi diagram to the heightmap.

    Args:
        hm (numpy.ndarray): A numpy.ndarray formatted for heightmap functions.
        nbPoints (Any): Number of Voronoi sites.
        nbCoef (int): The diagram value is calculated from the nbCoef
                      closest sites.
        coef (Sequence[float]): The distance to each site is scaled by the
                                corresponding coef.
                                Closest site : coef[0],
                                second closest site : coef[1], ...
        rnd (Optional[Random]): A Random instance, or None.
    """
    nbPoints = len(coef)
    ccoef = ffi.new('float[]', coef)
    lib.TCOD_heightmap_add_voronoi(
        _heightmap_cdata(hm), nbPoints,
        nbCoef, ccoef, rnd.random_c if rnd else ffi.NULL)

@deprecate("Arrays of noise should be sampled using the tcod.noise module.")
def heightmap_add_fbm(hm: np.ndarray, noise: tcod.noise.Noise,
                      mulx: float, muly: float, addx: float, addy: float,
                      octaves: float, delta: float, scale: float) -> None:
    """Add FBM noise to the heightmap.

    The noise coordinate for each map cell is
    `((x + addx) * mulx / width, (y + addy) * muly / height)`.

    The value added to the heightmap is `delta + noise * scale`.

    Args:
        hm (numpy.ndarray): A numpy.ndarray formatted for heightmap functions.
        noise (Noise): A Noise instance.
        mulx (float): Scaling of each x coordinate.
        muly (float): Scaling of each y coordinate.
        addx (float): Translation of each x coordinate.
        addy (float): Translation of each y coordinate.
        octaves (float): Number of octaves in the FBM sum.
        delta (float): The value added to all heightmap cells.
        scale (float): The noise value is scaled with this parameter.

    .. deprecated:: 8.1
        An equivalent array of noise samples can be taken using a method such
        as :any:`Noise.sample_ogrid`.
    """
    noise = noise.noise_c if noise is not None else ffi.NULL
    lib.TCOD_heightmap_add_fbm(_heightmap_cdata(hm), noise,
                               mulx, muly, addx, addy, octaves, delta, scale)

@deprecate("Arrays of noise should be sampled using the tcod.noise module.")
def heightmap_scale_fbm(hm: np.ndarray, noise: tcod.noise.Noise,
                        mulx: float, muly: float, addx: float, addy: float,
                        octaves: float, delta: float, scale: float) -> None:
    """Multiply the heighmap values with FBM noise.

    Args:
        hm (numpy.ndarray): A numpy.ndarray formatted for heightmap functions.
        noise (Noise): A Noise instance.
        mulx (float): Scaling of each x coordinate.
        muly (float): Scaling of each y coordinate.
        addx (float): Translation of each x coordinate.
        addy (float): Translation of each y coordinate.
        octaves (float): Number of octaves in the FBM sum.
        delta (float): The value added to all heightmap cells.
        scale (float): The noise value is scaled with this parameter.

    .. deprecated:: 8.1
        An equivalent array of noise samples can be taken using a method such
        as :any:`Noise.sample_ogrid`.
    """
    noise = noise.noise_c if noise is not None else ffi.NULL
    lib.TCOD_heightmap_scale_fbm(_heightmap_cdata(hm), noise,
                                 mulx, muly, addx, addy, octaves, delta, scale)

def heightmap_dig_bezier(hm: np.ndarray,
                         px: Tuple[int, int, int, int],
                         py: Tuple[int, int, int, int],
                         startRadius: float, startDepth: float,
                         endRadius: float, endDepth: float) -> None:
    """Carve a path along a cubic Bezier curve.

    Both radius and depth can vary linearly along the path.

    Args:
        hm (numpy.ndarray): A numpy.ndarray formatted for heightmap functions.
        px (Sequence[int]): The 4 `x` coordinates of the Bezier curve.
        py (Sequence[int]): The 4 `y` coordinates of the Bezier curve.
        startRadius (float): The starting radius size.
        startDepth (float): The starting depth.
        endRadius (float): The ending radius size.
        endDepth (float): The ending depth.
    """
    lib.TCOD_heightmap_dig_bezier(_heightmap_cdata(hm), px, py, startRadius,
                                   startDepth, endRadius,
                                   endDepth)

def heightmap_get_value(hm: np.ndarray, x: int, y: int) -> float:
    """Return the value at ``x``, ``y`` in a heightmap.

    .. deprecated:: 2.0
        Access `hm` as a NumPy array instead.
    """
    if hm.flags['C_CONTIGUOUS']:
        warnings.warn("Get a value from this heightmap with hm[i,j]\n"
                      "consider using order='F'",
                      DeprecationWarning, stacklevel=2)
        return hm[y, x]
    elif hm.flags['F_CONTIGUOUS']:
        warnings.warn("Get a value from this heightmap with hm[x,y]",
                      DeprecationWarning, stacklevel=2)
        return hm[x, y]
    else:
        raise ValueError("This array is not contiguous.")


def heightmap_get_interpolated_value(hm: np.ndarray,
                                     x: float, y: float) -> float:
    """Return the interpolated height at non integer coordinates.

    Args:
        hm (numpy.ndarray): A numpy.ndarray formatted for heightmap functions.
        x (float): A floating point x coordinate.
        y (float): A floating point y coordinate.

    Returns:
        float: The value at ``x``, ``y``.
    """
    return lib.TCOD_heightmap_get_interpolated_value(_heightmap_cdata(hm),
                                                     x, y)

def heightmap_get_slope(hm: np.ndarray, x: int, y: int) -> float:
    """Return the slope between 0 and (pi / 2) at given coordinates.

    Args:
        hm (numpy.ndarray): A numpy.ndarray formatted for heightmap functions.
        x (int): The x coordinate.
        y (int): The y coordinate.

    Returns:
        float: The steepness at ``x``, ``y``.  From 0 to (pi / 2)
    """
    return lib.TCOD_heightmap_get_slope(_heightmap_cdata(hm), x, y)

def heightmap_get_normal(hm: np.ndarray, x: float, y: float,
                         waterLevel: float) -> Tuple[float, float, float]:
    """Return the map normal at given coordinates.

    Args:
        hm (numpy.ndarray): A numpy.ndarray formatted for heightmap functions.
        x (float): The x coordinate.
        y (float): The y coordinate.
        waterLevel (float): The heightmap is considered flat below this value.

    Returns:
        Tuple[float, float, float]: An (x, y, z) vector normal.
    """
    cn = ffi.new('float[3]')
    lib.TCOD_heightmap_get_normal(_heightmap_cdata(hm), x, y, cn, waterLevel)
    return tuple(cn)

@deprecate("This function is deprecated, see documentation.")
def heightmap_count_cells(hm: np.ndarray, mi: float, ma: float) -> int:
    """Return the number of map cells which value is between ``mi`` and ``ma``.

    Args:
        hm (numpy.ndarray): A numpy.ndarray formatted for heightmap functions.
        mi (float): The lower bound.
        ma (float): The upper bound.

    Returns:
        int: The count of values which fall between ``mi`` and ``ma``.

    .. deprecated:: 8.1
        Can be replaced by an equivalent NumPy function such as:
        ``numpy.count_nonzero((mi <= hm) & (hm < ma))``
    """
    return lib.TCOD_heightmap_count_cells(_heightmap_cdata(hm), mi, ma)

def heightmap_has_land_on_border(hm: np.ndarray, waterlevel: float) -> bool:
    """Returns True if the map edges are below ``waterlevel``, otherwise False.

    Args:
        hm (numpy.ndarray): A numpy.ndarray formatted for heightmap functions.
        waterLevel (float): The water level to use.

    Returns:
        bool: True if the map edges are below ``waterlevel``, otherwise False.
    """
    return lib.TCOD_heightmap_has_land_on_border(_heightmap_cdata(hm),
                                                 waterlevel)

@deprecate("Use `hm.min()` and `hm.max()` instead.")
def heightmap_get_minmax(hm: np.ndarray) -> Tuple[float, float]:
    """Return the min and max values of this heightmap.

    Args:
        hm (numpy.ndarray): A numpy.ndarray formatted for heightmap functions.

    Returns:
        Tuple[float, float]: The (min, max) values.

    .. deprecated:: 2.0
        Use ``hm.min()`` or ``hm.max()`` instead.
    """
    mi = ffi.new('float *')
    ma = ffi.new('float *')
    lib.TCOD_heightmap_get_minmax(_heightmap_cdata(hm), mi, ma)
    return mi[0], ma[0]

@deprecate("libtcod objects are deleted automatically.")
def heightmap_delete(hm: Any) -> None:
    """Does nothing. libtcod objects are managed by Python's garbage collector.

    This function exists for backwards compatibility with libtcodpy.

    .. deprecated:: 2.0
        libtcod-cffi deletes heightmaps automatically.
    """

def image_new(width, height):
    return tcod.image.Image(width, height)

def image_clear(image, col):
    image.clear(col)

def image_invert(image):
    image.invert()

def image_hflip(image):
    image.hflip()

def image_rotate90(image, num=1):
    image.rotate90(num)

def image_vflip(image):
    image.vflip()

def image_scale(image, neww, newh):
    image.scale(neww, newh)

def image_set_key_color(image, col):
    image.set_key_color(col)

def image_get_alpha(image, x, y):
    image.get_alpha(x, y)

def image_is_pixel_transparent(image, x, y):
    lib.TCOD_image_is_pixel_transparent(image.image_c, x, y)

def image_load(filename):
    """Load an image file into an Image instance and return it.

    Args:
        filename (AnyStr): Path to a .bmp or .png image file.
    """
    return tcod.image.Image._from_cdata(
        ffi.gc(lib.TCOD_image_load(_bytes(filename)),
               lib.TCOD_image_delete)
        )

def image_from_console(console):
    """Return an Image with a Consoles pixel data.

    This effectively takes a screen-shot of the Console.

    Args:
        console (Console): Any Console instance.
    """
    return tcod.image.Image._from_cdata(
        ffi.gc(
            lib.TCOD_image_from_console(_console(console)),
            lib.TCOD_image_delete,
            )
        )

def image_refresh_console(image, console):
    image.refresh_console(console)

def image_get_size(image):
    return image.width, image.height

def image_get_pixel(image, x, y):
    return image.get_pixel(x, y)

def image_get_mipmap_pixel(image, x0, y0, x1, y1):
    return image.get_mipmap_pixel(x0, y0, x1, y1)

def image_put_pixel(image, x, y, col):
    image.put_pixel(x, y, col)

def image_blit(image, console, x, y, bkgnd_flag, scalex, scaley, angle):
    image.blit(console, x, y, bkgnd_flag, scalex, scaley, angle)

def image_blit_rect(image, console, x, y, w, h, bkgnd_flag):
    image.blit_rect(console, x, y, w, h, bkgnd_flag)

def image_blit_2x(image, console, dx, dy, sx=0, sy=0, w=-1, h=-1):
    image.blit_2x(console, dx, dy, sx, sy, w, h)

def image_save(image, filename):
    image.save_as(filename)

def image_delete(image):
    # type (Any) -> None
    """Does nothing. libtcod objects are managed by Python's garbage collector.

    This function exists for backwards compatibility with libtcodpy.
    """

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
    for x, y in line_iter(xo, yo, xd, yd):
        if not py_callback(x, y):
            break
    else:
        return True
    return False


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
    yield xo, yo
    while not lib.TCOD_line_step_mt(x, y, data):
        yield (x[0], y[0])


def line_where(x1, y1, x2, y2, inclusive=True):
    # type: (int, int, int, int, bool) -> tuple[np.ndarray, np.ndarray]
    """Return a NumPy index array following a Bresenham line.

    If `inclusive` is true then the start point is included in the result.

    Example:
        >>> where = tcod.line_where(1, 0, 3, 4)
        >>> where
        (array([1, 1, 2, 2, 3]...), array([0, 1, 2, 3, 4]...))
        >>> array = np.zeros((5, 5), dtype=np.int32)
        >>> array[where] = np.arange(len(where[0])) + 1
        >>> array
        array([[0, 0, 0, 0, 0],
               [1, 2, 0, 0, 0],
               [0, 0, 3, 4, 0],
               [0, 0, 0, 0, 5],
               [0, 0, 0, 0, 0]]...)

    .. versionadded:: 4.6
    """
    length = max(abs(x1 - x2), abs(y1 - y2)) + 1
    array = np.ndarray((2, length), dtype=np.intc)
    x = ffi.cast('int*', array[0].ctypes.data)
    y = ffi.cast('int*', array[1].ctypes.data)
    lib.LineWhere(x1, y1, x2, y2, x, y)
    if not inclusive:
        array = array[:, 1:]
    return tuple(array)


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
    # type: (int, int) -> tcod.map.Map
    """Return a :any:`tcod.map.Map` with a width and height.

    .. deprecated:: 4.5
        Use the :any:`tcod.map` module for working with field-of-view,
        or :any:`tcod.path` for working with path-finding.
    """
    return tcod.map.Map(w, h)

def map_copy(source, dest):
    # type: (tcod.map.Map, tcod.map.Map) -> None
    """Copy map data from `source` to `dest`.

    .. deprecated:: 4.5
        Use Python's copy module, or see :any:`tcod.map.Map` and assign between
        array attributes manually.
    """
    if source.width != dest.width or source.height != dest.height:
        dest.__init__(source.width, source.height, source._order)
    dest._Map__buffer[:] = source._Map__buffer[:]

def map_set_properties(m, x, y, isTrans, isWalk):
    # type: (tcod.map.Map, int, int, bool, bool) -> None
    """Set the properties of a single cell.

    .. note::
        This function is slow.
    .. deprecated:: 4.5
        Use :any:`tcod.map.Map.transparent` and :any:`tcod.map.Map.walkable`
        arrays to set these properties.
    """
    lib.TCOD_map_set_properties(m.map_c, x, y, isTrans, isWalk)

def map_clear(m, transparent=False, walkable=False):
    # type: (tcod.map.Map, bool, bool) -> None
    """Change all map cells to a specific value.

    .. deprecated:: 4.5
        Use :any:`tcod.map.Map.transparent` and :any:`tcod.map.Map.walkable`
        arrays to set these properties.
    """
    m.transparent[:] = transparent
    m.walkable[:] = walkable

def map_compute_fov(m, x, y, radius=0, light_walls=True, algo=FOV_RESTRICTIVE ):
    # type: (tcod.map.Map, int, int, int, bool, int) -> None
    """Compute the field-of-view for a map instance.

    .. deprecated:: 4.5
        Use :any:`tcod.map.Map.compute_fov` instead.
    """
    m.compute_fov(x, y, radius, light_walls, algo)

def map_is_in_fov(m, x, y):
    # type: (tcod.map.Map, int, int) -> bool
    """Return True if the cell at x,y is lit by the last field-of-view
    algorithm.

    .. note::
        This function is slow.
    .. deprecated:: 4.5
        Use :any:`tcod.map.Map.fov` to check this property.
    """
    return lib.TCOD_map_is_in_fov(m.map_c, x, y)

def map_is_transparent(m, x, y):
    # type: (tcod.map.Map, int, int) -> bool
    """
    .. note::
        This function is slow.
    .. deprecated:: 4.5
        Use :any:`tcod.map.Map.transparent` to check this property.
    """
    return lib.TCOD_map_is_transparent(m.map_c, x, y)

def map_is_walkable(m, x, y):
    # type: (tcod.map.Map, int, int) -> bool
    """
    .. note::
        This function is slow.
    .. deprecated:: 4.5
        Use :any:`tcod.map.Map.walkable` to check this property.
    """
    return lib.TCOD_map_is_walkable(m.map_c, x, y)

def map_delete(m):
    # type (Any) -> None
    """Does nothing. libtcod objects are managed by Python's garbage collector.

    This function exists for backwards compatibility with libtcodpy.
    """

def map_get_width(map):
    # type: (tcod.map.Map) -> int
    """Return the width of a map.

    .. deprecated:: 4.5
        Check the :any:`tcod.map.Map.width` attribute instead.
    """
    return map.width

def map_get_height(map):
    # type: (tcod.map.Map) -> int
    """Return the height of a map.

    .. deprecated:: 4.5
        Check the :any:`tcod.map.Map.height` attribute instead.
    """
    return map.height

def mouse_show_cursor(visible):
    # type: (bool) -> None
    """Change the visibility of the mouse cursor."""
    lib.TCOD_mouse_show_cursor(visible)

def mouse_is_cursor_visible():
    # type: () -> bool
    """Return True if the mouse cursor is visible."""
    return lib.TCOD_mouse_is_cursor_visible()

def mouse_move(x, y):
    # type (int, int) -> None
    lib.TCOD_mouse_move(x, y)

def mouse_get_status():
    # type: () -> Mouse
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
    """Return a new Noise instance.

    Args:
        dim (int): Number of dimensions.  From 1 to 4.
        h (float): The hurst exponent.  Should be in the 0.0-1.0 range.
        l (float): The noise lacunarity.
        random (Optional[Random]): A Random instance, or None.

    Returns:
        Noise: The new Noise instance.
    """
    return tcod.noise.Noise(dim, hurst=h, lacunarity=l, seed=random)

def noise_set_type(n, typ):
    """Set a Noise objects default noise algorithm.

    Args:
        typ (int): Any NOISE_* constant.
    """
    n.algorithm = typ

def noise_get(n, f, typ=NOISE_DEFAULT):
    """Return the noise value sampled from the ``f`` coordinate.

    ``f`` should be a tuple or list with a length matching
    :any:`Noise.dimensions`.
    If ``f`` is shoerter than :any:`Noise.dimensions` the missing coordinates
    will be filled with zeros.

    Args:
        n (Noise): A Noise instance.
        f (Sequence[float]): The point to sample the noise from.
        typ (int): The noise algorithm to use.

    Returns:
        float: The sampled noise value.
    """
    return lib.TCOD_noise_get_ex(n.noise_c, ffi.new('float[4]', f), typ)

def noise_get_fbm(n, f, oc, typ=NOISE_DEFAULT):
    """Return the fractal Brownian motion sampled from the ``f`` coordinate.

    Args:
        n (Noise): A Noise instance.
        f (Sequence[float]): The point to sample the noise from.
        typ (int): The noise algorithm to use.
        octaves (float): The level of level.  Should be more than 1.

    Returns:
        float: The sampled noise value.
    """
    return lib.TCOD_noise_get_fbm_ex(n.noise_c, ffi.new('float[4]', f),
                                     oc, typ)

def noise_get_turbulence(n, f, oc, typ=NOISE_DEFAULT):
    """Return the turbulence noise sampled from the ``f`` coordinate.

    Args:
        n (Noise): A Noise instance.
        f (Sequence[float]): The point to sample the noise from.
        typ (int): The noise algorithm to use.
        octaves (float): The level of level.  Should be more than 1.

    Returns:
        float: The sampled noise value.
    """
    return lib.TCOD_noise_get_turbulence_ex(n.noise_c, ffi.new('float[4]', f),
                                            oc, typ)

def noise_delete(n):
    # type (Any) -> None
    """Does nothing. libtcod objects are managed by Python's garbage collector.

    This function exists for backwards compatibility with libtcodpy.
    """

_chr = chr
try:
    _chr = unichr # Python 2
except NameError:
    pass

def _unpack_union(type_, union):
    '''
        unpack items from parser new_property (value_converter)
    '''
    if type_ == lib.TCOD_TYPE_BOOL:
        return bool(union.b)
    elif type_ == lib.TCOD_TYPE_CHAR:
        return union.c.decode('latin-1')
    elif type_ == lib.TCOD_TYPE_INT:
        return union.i
    elif type_ == lib.TCOD_TYPE_FLOAT:
        return union.f
    elif (type_ == lib.TCOD_TYPE_STRING or
         lib.TCOD_TYPE_VALUELIST15 >= type_ >= lib.TCOD_TYPE_VALUELIST00):
         return _unpack_char_p(union.s)
    elif type_ == lib.TCOD_TYPE_COLOR:
        return Color._new_from_cdata(union.col)
    elif type_ == lib.TCOD_TYPE_DICE:
        return Dice(union.dice)
    elif type_ & lib.TCOD_TYPE_LIST:
        return _convert_TCODList(union.list, type_ & 0xFF)
    else:
        raise RuntimeError('Unknown libtcod type: %i' % type_)

def _convert_TCODList(clist, type_):
    return [_unpack_union(type_, lib.TDL_list_get_union(clist, i))
            for i in range(lib.TCOD_list_size(clist))]

def parser_new():
    return ffi.gc(lib.TCOD_parser_new(), lib.TCOD_parser_delete)

def parser_new_struct(parser, name):
    return lib.TCOD_parser_new_struct(parser, name)

# prevent multiple threads from messing with def_extern callbacks
_parser_callback_lock = _threading.Lock()
_parser_listener = None # temporary global pointer to a listener instance

@ffi.def_extern()
def _pycall_parser_new_struct(struct, name):
    return _parser_listener.end_struct(struct, _unpack_char_p(name))

@ffi.def_extern()
def _pycall_parser_new_flag(name):
    return _parser_listener.new_flag(_unpack_char_p(name))

@ffi.def_extern()
def _pycall_parser_new_property(propname, type, value):
    return _parser_listener.new_property(_unpack_char_p(propname), type,
                                 _unpack_union(type, value))

@ffi.def_extern()
def _pycall_parser_end_struct(struct, name):
    return _parser_listener.end_struct(struct, _unpack_char_p(name))

@ffi.def_extern()
def _pycall_parser_error(msg):
    _parser_listener.error(_unpack_char_p(msg))

def parser_run(parser, filename, listener=None):
    global _parser_listener
    if not listener:
        lib.TCOD_parser_run(parser, _bytes(filename), ffi.NULL)
        return

    propagate_manager = _PropagateException()
    propagate = propagate_manager.propagate

    clistener = ffi.new(
        'TCOD_parser_listener_t *',
        {
            'new_struct': lib._pycall_parser_new_struct,
            'new_flag': lib._pycall_parser_new_flag,
            'new_property': lib._pycall_parser_new_property,
            'end_struct': lib._pycall_parser_end_struct,
            'error': lib._pycall_parser_error,
        },
    )

    with _parser_callback_lock:
        _parser_listener = listener
        with propagate_manager:
            lib.TCOD_parser_run(parser, _bytes(filename), clistener)

def parser_delete(parser):
    # type (Any) -> None
    """Does nothing. libtcod objects are managed by Python's garbage collector.

    This function exists for backwards compatibility with libtcodpy.
    """

def parser_get_bool_property(parser, name):
    return bool(lib.TCOD_parser_get_bool_property(parser, _bytes(name)))

def parser_get_int_property(parser, name):
    return lib.TCOD_parser_get_int_property(parser, _bytes(name))

def parser_get_char_property(parser, name):
    return _chr(lib.TCOD_parser_get_char_property(parser, _bytes(name)))

def parser_get_float_property(parser, name):
    return lib.TCOD_parser_get_float_property(parser, _bytes(name))

def parser_get_string_property(parser, name):
    return _unpack_char_p(
        lib.TCOD_parser_get_string_property(parser, _bytes(name)))

def parser_get_color_property(parser, name):
    return Color._new_from_cdata(
        lib.TCOD_parser_get_color_property(parser, _bytes(name)))

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
    """Return the default Random instance.

    Returns:
        Random: A Random instance using the default random number generator.
    """
    return tcod.random.Random._new_from_cdata(
        ffi.cast('mersenne_data_t*', lib.TCOD_random_get_instance()))

def random_new(algo=RNG_CMWC):
    """Return a new Random instance.  Using ``algo``.

    Args:
        algo (int): The random number algorithm to use.

    Returns:
        Random: A new Random instance using the given algorithm.
    """
    return tcod.random.Random(algo)

def random_new_from_seed(seed, algo=RNG_CMWC):
    """Return a new Random instance.  Using the given ``seed`` and ``algo``.

    Args:
        seed (Hashable): The RNG seed.  Should be a 32-bit integer, but any
                         hashable object is accepted.
        algo (int): The random number algorithm to use.

    Returns:
        Random: A new Random instance using the given algorithm.
    """
    return tcod.random.Random(algo, seed)

def random_set_distribution(rnd, dist):
    """Change the distribution mode of a random number generator.

    Args:
        rnd (Optional[Random]): A Random instance, or None to use the default.
        dist (int): The distribution mode to use.  Should be DISTRIBUTION_*.
    """
    lib.TCOD_random_set_distribution(rnd.random_c if rnd else ffi.NULL, dist)

def random_get_int(rnd, mi, ma):
    """Return a random integer in the range: ``mi`` <= n <= ``ma``.

    The result is affacted by calls to :any:`random_set_distribution`.

    Args:
        rnd (Optional[Random]): A Random instance, or None to use the default.
        low (int): The lower bound of the random range, inclusive.
        high (int): The upper bound of the random range, inclusive.

    Returns:
        int: A random integer in the range ``mi`` <= n <= ``ma``.
    """
    return lib.TCOD_random_get_int(rnd.random_c if rnd else ffi.NULL, mi, ma)

def random_get_float(rnd, mi, ma):
    """Return a random float in the range: ``mi`` <= n <= ``ma``.

    The result is affacted by calls to :any:`random_set_distribution`.

    Args:
        rnd (Optional[Random]): A Random instance, or None to use the default.
        low (float): The lower bound of the random range, inclusive.
        high (float): The upper bound of the random range, inclusive.

    Returns:
        float: A random double precision float
               in the range ``mi`` <= n <= ``ma``.
    """
    return lib.TCOD_random_get_double(
        rnd.random_c if rnd else ffi.NULL, mi, ma)

def random_get_double(rnd, mi, ma):
    """Return a random float in the range: ``mi`` <= n <= ``ma``.

    .. deprecated:: 2.0
        Use :any:`random_get_float` instead.
        Both funtions return a double precision float.
    """
    return lib.TCOD_random_get_double(
        rnd.random_c if rnd else ffi.NULL, mi, ma)

def random_get_int_mean(rnd, mi, ma, mean):
    """Return a random weighted integer in the range: ``mi`` <= n <= ``ma``.

    The result is affacted by calls to :any:`random_set_distribution`.

    Args:
        rnd (Optional[Random]): A Random instance, or None to use the default.
        low (int): The lower bound of the random range, inclusive.
        high (int): The upper bound of the random range, inclusive.
        mean (int): The mean return value.

    Returns:
        int: A random weighted integer in the range ``mi`` <= n <= ``ma``.
    """
    return lib.TCOD_random_get_int_mean(
        rnd.random_c if rnd else ffi.NULL, mi, ma, mean)

def random_get_float_mean(rnd, mi, ma, mean):
    """Return a random weighted float in the range: ``mi`` <= n <= ``ma``.

    The result is affacted by calls to :any:`random_set_distribution`.

    Args:
        rnd (Optional[Random]): A Random instance, or None to use the default.
        low (float): The lower bound of the random range, inclusive.
        high (float): The upper bound of the random range, inclusive.
        mean (float): The mean return value.

    Returns:
        float: A random weighted double precision float
               in the range ``mi`` <= n <= ``ma``.
    """
    return lib.TCOD_random_get_double_mean(
        rnd.random_c if rnd else ffi.NULL, mi, ma, mean)

def random_get_double_mean(rnd, mi, ma, mean):
    """Return a random weighted float in the range: ``mi`` <= n <= ``ma``.

    .. deprecated:: 2.0
        Use :any:`random_get_float_mean` instead.
        Both funtions return a double precision float.
    """
    return lib.TCOD_random_get_double_mean(
        rnd.random_c if rnd else ffi.NULL, mi, ma, mean)

def random_save(rnd):
    """Return a copy of a random number generator.

    Args:
        rnd (Optional[Random]): A Random instance, or None to use the default.

    Returns:
        Random: A Random instance with a copy of the random generator.
    """
    return tcod.random.Random._new_from_cdata(
        ffi.gc(
            ffi.cast('mersenne_data_t*',
                     lib.TCOD_random_save(rnd.random_c if rnd else ffi.NULL)),
            lib.TCOD_random_delete),
        )

def random_restore(rnd, backup):
    """Restore a random number generator from a backed up copy.

    Args:
        rnd (Optional[Random]): A Random instance, or None to use the default.
        backup (Random): The Random instance which was used as a backup.
    """
    lib.TCOD_random_restore(rnd.random_c if rnd else ffi.NULL,
                            backup.random_c)

def random_delete(rnd):
    # type (Any) -> None
    """Does nothing. libtcod objects are managed by Python's garbage collector.

    This function exists for backwards compatibility with libtcodpy.
    """

def struct_add_flag(struct, name):
    lib.TCOD_struct_add_flag(struct, name)

def struct_add_property(struct, name, typ, mandatory):
    lib.TCOD_struct_add_property(struct, name, typ, mandatory)

def struct_add_value_list(struct, name, value_list, mandatory):
    CARRAY = c_char_p * (len(value_list) + 1)
    cvalue_list = CARRAY()
    for i, value in enumerate(value_list):
        cvalue_list[i] = cast(value, c_char_p)
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

@deprecate("Use Python's standard 'time' module instead of this function.")
def sys_sleep_milli(val):
    """Sleep for 'val' milliseconds.

    Args:
        val (int): Time to sleep for in milliseconds.

    .. deprecated:: 2.0
       Use :any:`time.sleep` instead.
    """
    lib.TCOD_sys_sleep_milli(val)

@deprecate("Use Python's standard 'time' module instead of this function.")
def sys_elapsed_milli():
    """Get number of milliseconds since the start of the program.

    Returns:
        int: Time since the progeam has started in milliseconds.

    .. deprecated:: 2.0
       Use :any:`time.clock` instead.
    """
    return lib.TCOD_sys_elapsed_milli()

@deprecate("Use Python's standard 'time' module instead of this function.")
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
    if tcod.console._root_console is not None:
        tcod.console.Console._get_root()

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

    Note:
        This callback will only be called by the SDL renderer.

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
    """Check for and return an event.

    Args:
        mask (int): :any:`Event types` to wait for.
        k (Optional[Key]): A tcod.Key instance which might be updated with
                           an event.  Can be None.
        m (Optional[Mouse]): A tcod.Mouse instance which might be updated
                             with an event.  Can be None.
    """
    return lib.TCOD_sys_check_for_event(
        mask, k.cdata if k else ffi.NULL, m.cdata if m else ffi.NULL)

def sys_wait_for_event(mask, k, m, flush):
    """Wait for an event then return.

    If flush is True then the buffer will be cleared before waiting. Otherwise
    each available event will be returned in the order they're recieved.

    Args:
        mask (int): :any:`Event types` to wait for.
        k (Optional[Key]): A tcod.Key instance which might be updated with
                           an event.  Can be None.
        m (Optional[Mouse]): A tcod.Mouse instance which might be updated
                             with an event.  Can be None.
        flush (bool): Clear the event buffer before waiting.
    """
    return lib.TCOD_sys_wait_for_event(
        mask, k.cdata if k else ffi.NULL, m.cdata if m else ffi.NULL, flush)

@deprecate("This function does not provide reliable access to the clipboard.")
def sys_clipboard_set(text):
    """Sets the clipboard to `text`.

    .. deprecated:: 6.0
       This function does not provide reliable access to the clipboard.
    """
    return lib.TCOD_sys_clipboard_set(text.encode('utf-8'))

@deprecate("This function does not provide reliable access to the clipboard.")
def sys_clipboard_get():
    """Return the current value of the clipboard.

    .. deprecated:: 6.0
       This function does not provide reliable access to the clipboard.
    """
    return ffi.string(lib.TCOD_sys_clipboard_get()).decode('utf-8')
