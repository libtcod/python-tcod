"""This module handles backward compatibility with the ctypes libtcodpy module.
"""
import os
import sys

import atexit
import threading
from typing import (
    Any,
    AnyStr,
    Callable,
    Hashable,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)
import warnings

import numpy as np

from tcod.libtcod import ffi, lib

from tcod.constants import *  # noqa: F4
from tcod.constants import (
    BKGND_DEFAULT,
    BKGND_SET,
    BKGND_ALPH,
    BKGND_ADDA,
    FONT_LAYOUT_ASCII_INCOL,
    FOV_RESTRICTIVE,
    FOV_PERMISSIVE_0,
    NOISE_DEFAULT,
    KEY_RELEASED,
)

from tcod._internal import deprecate, pending_deprecate, _check

from tcod.tcod import _int, _unpack_char_p
from tcod.tcod import _bytes, _unicode, _fmt
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

NB_FOV_ALGORITHMS = 13

NOISE_DEFAULT_HURST = 0.5
NOISE_DEFAULT_LACUNARITY = 2.0


def FOV_PERMISSIVE(p: int) -> int:
    return FOV_PERMISSIVE_0 + p


def BKGND_ALPHA(a: int) -> int:
    return BKGND_ALPH | (int(a * 255) << 8)


def BKGND_ADDALPHA(a: int) -> int:
    return BKGND_ADDA | (int(a * 255) << 8)


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

    def __init__(
        self,
        width: int,
        height: int,
        back_r: int = 0,
        back_g: int = 0,
        back_b: int = 0,
        fore_r: int = 0,
        fore_g: int = 0,
        fore_b: int = 0,
        char: str = " ",
    ) -> None:
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

    def clear(
        self,
        back_r: int = 0,
        back_g: int = 0,
        back_b: int = 0,
        fore_r: int = 0,
        fore_g: int = 0,
        fore_b: int = 0,
        char: str = " ",
    ) -> None:
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

    def copy(self) -> "ConsoleBuffer":
        """Returns a copy of this ConsoleBuffer.

        Returns:
            ConsoleBuffer: A new ConsoleBuffer copy.
        """
        other = ConsoleBuffer(0, 0)  # type: "ConsoleBuffer"
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

    def set_fore(
        self, x: int, y: int, r: int, g: int, b: int, char: str
    ) -> None:
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

    def set_back(self, x: int, y: int, r: int, g: int, b: int) -> None:
        """Set the background color of one cell.

        Args:
            x (int): X position to change.
            y (int): Y position to change.
            r (int): Red background color, from 0 to 255.
            g (int): Green background color, from 0 to 255.
            b (int): Blue background color, from 0 to 255.
        """
        i = self.width * y + x
        self.back_r[i] = r
        self.back_g[i] = g
        self.back_b[i] = b

    def set(
        self,
        x: int,
        y: int,
        back_r: int,
        back_g: int,
        back_b: int,
        fore_r: int,
        fore_g: int,
        fore_b: int,
        char: str,
    ) -> None:
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

    def blit(
        self,
        dest: tcod.console.Console,
        fill_fore: bool = True,
        fill_back: bool = True,
    ) -> None:
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
        if dest.width != self.width or dest.height != self.height:
            raise ValueError(
                "ConsoleBuffer.blit: "
                "Destination console has an incorrect size."
            )

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

    def __init__(self, *args: Any, **kargs: Any) -> None:
        warnings.warn(
            "Using this class is not recommended.",
            DeprecationWarning,
            stacklevel=2,
        )
        super(Dice, self).__init__(*args, **kargs)
        if self.cdata == ffi.NULL:
            self._init(*args, **kargs)

    def _init(
        self,
        nb_dices: int = 0,
        nb_faces: int = 0,
        multiplier: float = 0,
        addsub: float = 0,
    ) -> None:
        self.cdata = ffi.new("TCOD_dice_t*")
        self.nb_dices = nb_dices
        self.nb_faces = nb_faces
        self.multiplier = multiplier
        self.addsub = addsub

    @property
    def nb_dices(self) -> int:
        return self.nb_rolls

    @nb_dices.setter
    def nb_dices(self, value: int) -> None:
        self.nb_rolls = value

    def __str__(self) -> str:
        add = "+(%s)" % self.addsub if self.addsub != 0 else ""
        return "%id%ix%s%s" % (
            self.nb_dices,
            self.nb_faces,
            self.multiplier,
            add,
        )

    def __repr__(self) -> str:
        return "%s(nb_dices=%r,nb_faces=%r,multiplier=%r,addsub=%r)" % (
            self.__class__.__name__,
            self.nb_dices,
            self.nb_faces,
            self.multiplier,
            self.addsub,
        )


# reverse lookup table for KEY_X attributes, used by Key.__repr__
_LOOKUP_VK = {
    value: "KEY_%s" % key[6:]
    for key, value in lib.__dict__.items()
    if key.startswith("TCODK")
}


class Key(_CDataWrapper):
    """Key Event instance

    Attributes:
        vk (int): TCOD_keycode_t key code
        c (int): character if vk == TCODK_CHAR else 0
        text (Text): text[TCOD_KEY_TEXT_SIZE];
                     text if vk == TCODK_TEXT else text[0] == '\0'
        pressed (bool): does this correspond to a key press or key release
                        event?
        lalt (bool): True when left alt is held.
        lctrl (bool): True when left control is held.
        lmeta (bool): True when left meta key is held.
        ralt (bool): True when right alt is held.
        rctrl (bool): True when right control is held.
        rmeta (bool): True when right meta key is held.
        shift (bool): True when any shift is held.

    .. deprecated:: 9.3
        Use events from the :any:`tcod.event` module instead.
    """

    _BOOL_ATTRIBUTES = (
        "lalt",
        "lctrl",
        "lmeta",
        "ralt",
        "rctrl",
        "rmeta",
        "pressed",
        "shift",
    )

    def __init__(
        self,
        vk: int = 0,
        c: int = 0,
        text: str = "",
        pressed: bool = False,
        lalt: bool = False,
        lctrl: bool = False,
        lmeta: bool = False,
        ralt: bool = False,
        rctrl: bool = False,
        rmeta: bool = False,
        shift: bool = False,
    ):
        if isinstance(vk, ffi.CData):
            self.cdata = vk
            return
        self.cdata = ffi.new("TCOD_key_t*")
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

    def __getattr__(self, attr: str) -> Any:
        if attr in self._BOOL_ATTRIBUTES:
            return bool(getattr(self.cdata, attr))
        if attr == "c":
            return ord(self.cdata.c)
        if attr == "text":
            return ffi.string(self.cdata.text).decode()
        return super(Key, self).__getattr__(attr)

    def __setattr__(self, attr: str, value: Any) -> None:
        if attr == "c":
            self.cdata.c = chr(value).encode("latin-1")
        elif attr == "text":
            self.cdata.text = value.encode()
        else:
            super(Key, self).__setattr__(attr, value)

    def __repr__(self) -> str:
        """Return a representation of this Key object."""
        params = []
        params.append(
            "pressed=%r, vk=tcod.%s" % (self.pressed, _LOOKUP_VK[self.vk])
        )
        if self.c:
            params.append("c=ord(%r)" % chr(self.c))
        if self.text:
            params.append("text=%r" % self.text)
        for attr in [
            "shift",
            "lalt",
            "lctrl",
            "lmeta",
            "ralt",
            "rctrl",
            "rmeta",
        ]:
            if getattr(self, attr):
                params.append("%s=%r" % (attr, getattr(self, attr)))
        return "tcod.Key(%s)" % ", ".join(params)

    @property
    def key_p(self) -> Any:
        return self.cdata


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

    .. deprecated:: 9.3
        Use events from the :any:`tcod.event` module instead.
    """

    def __init__(
        self,
        x: int = 0,
        y: int = 0,
        dx: int = 0,
        dy: int = 0,
        cx: int = 0,
        cy: int = 0,
        dcx: int = 0,
        dcy: int = 0,
        **kargs: Any
    ):
        if isinstance(x, ffi.CData):
            self.cdata = x
            return
        self.cdata = ffi.new("TCOD_mouse_t*")
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

    def __repr__(self) -> str:
        """Return a representation of this Mouse object."""
        params = []
        for attr in ["x", "y", "dx", "dy", "cx", "cy", "dcx", "dcy"]:
            if getattr(self, attr) == 0:
                continue
            params.append("%s=%r" % (attr, getattr(self, attr)))
        for attr in [
            "lbutton",
            "rbutton",
            "mbutton",
            "lbutton_pressed",
            "rbutton_pressed",
            "mbutton_pressed",
            "wheel_up",
            "wheel_down",
        ]:
            if getattr(self, attr):
                params.append("%s=%r" % (attr, getattr(self, attr)))
        return "tcod.Mouse(%s)" % ", ".join(params)

    @property
    def mouse_p(self) -> Any:
        return self.cdata


@deprecate("Call tcod.bsp.BSP(x, y, width, height) instead.")
def bsp_new_with_size(x: int, y: int, w: int, h: int) -> tcod.bsp.BSP:
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


@deprecate("Call node.split_once instead.")
def bsp_split_once(
    node: tcod.bsp.BSP, horizontal: bool, position: int
) -> None:
    """
    .. deprecated:: 2.0
       Use :any:`BSP.split_once` instead.
    """
    node.split_once(horizontal, position)


@deprecate("Call node.split_recursive instead.")
def bsp_split_recursive(
    node: tcod.bsp.BSP,
    randomizer: Optional[tcod.random.Random],
    nb: int,
    minHSize: int,
    minVSize: int,
    maxHRatio: int,
    maxVRatio: int,
) -> None:
    """
    .. deprecated:: 2.0
       Use :any:`BSP.split_recursive` instead.
    """
    node.split_recursive(
        nb, minHSize, minVSize, maxHRatio, maxVRatio, randomizer
    )


@deprecate("Assign values via attribute instead.")
def bsp_resize(node: tcod.bsp.BSP, x: int, y: int, w: int, h: int) -> None:
    """
    .. deprecated:: 2.0
        Assign directly to :any:`BSP` attributes instead.
    """
    node.x = x
    node.y = y
    node.width = w
    node.height = h


@deprecate("Access children with 'node.children' instead.")
def bsp_left(node: tcod.bsp.BSP) -> Optional[tcod.bsp.BSP]:
    """
    .. deprecated:: 2.0
       Use :any:`BSP.children` instead.
    """
    return None if not node.children else node.children[0]


@deprecate("Access children with 'node.children' instead.")
def bsp_right(node: tcod.bsp.BSP) -> Optional[tcod.bsp.BSP]:
    """
    .. deprecated:: 2.0
       Use :any:`BSP.children` instead.
    """
    return None if not node.children else node.children[1]


@deprecate("Get the parent with 'node.parent' instead.")
def bsp_father(node: tcod.bsp.BSP) -> Optional[tcod.bsp.BSP]:
    """
    .. deprecated:: 2.0
       Use :any:`BSP.parent` instead.
    """
    return node.parent


@deprecate("Check for children with 'bool(node.children)' instead.")
def bsp_is_leaf(node: tcod.bsp.BSP) -> bool:
    """
    .. deprecated:: 2.0
       Use :any:`BSP.children` instead.
    """
    return not node.children


@deprecate("Use 'node.contains' instead.")
def bsp_contains(node: tcod.bsp.BSP, cx: int, cy: int) -> bool:
    """
    .. deprecated:: 2.0
       Use :any:`BSP.contains` instead.
    """
    return node.contains(cx, cy)


@deprecate("Use 'node.find_node' instead.")
def bsp_find_node(
    node: tcod.bsp.BSP, cx: int, cy: int
) -> Optional[tcod.bsp.BSP]:
    """
    .. deprecated:: 2.0
       Use :any:`BSP.find_node` instead.
    """
    return node.find_node(cx, cy)


def _bsp_traverse(
    node_iter: Iterable[tcod.bsp.BSP],
    callback: Callable[[tcod.bsp.BSP, Any], None],
    userData: Any,
) -> None:
    """pack callback into a handle for use with the callback
    _pycall_bsp_callback
    """
    for node in node_iter:
        callback(node, userData)


@deprecate("Iterate over nodes using 'for n in node.pre_order():' instead.")
def bsp_traverse_pre_order(
    node: tcod.bsp.BSP,
    callback: Callable[[tcod.bsp.BSP, Any], None],
    userData: Any = 0,
) -> None:
    """Traverse this nodes hierarchy with a callback.

    .. deprecated:: 2.0
       Use :any:`BSP.pre_order` instead.
    """
    _bsp_traverse(node.pre_order(), callback, userData)


@deprecate("Iterate over nodes using 'for n in node.in_order():' instead.")
def bsp_traverse_in_order(
    node: tcod.bsp.BSP,
    callback: Callable[[tcod.bsp.BSP, Any], None],
    userData: Any = 0,
) -> None:
    """Traverse this nodes hierarchy with a callback.

    .. deprecated:: 2.0
       Use :any:`BSP.in_order` instead.
    """
    _bsp_traverse(node.in_order(), callback, userData)


@deprecate("Iterate over nodes using 'for n in node.post_order():' instead.")
def bsp_traverse_post_order(
    node: tcod.bsp.BSP,
    callback: Callable[[tcod.bsp.BSP, Any], None],
    userData: Any = 0,
) -> None:
    """Traverse this nodes hierarchy with a callback.

    .. deprecated:: 2.0
       Use :any:`BSP.post_order` instead.
    """
    _bsp_traverse(node.post_order(), callback, userData)


@deprecate("Iterate over nodes using 'for n in node.level_order():' instead.")
def bsp_traverse_level_order(
    node: tcod.bsp.BSP,
    callback: Callable[[tcod.bsp.BSP, Any], None],
    userData: Any = 0,
) -> None:
    """Traverse this nodes hierarchy with a callback.

    .. deprecated:: 2.0
       Use :any:`BSP.level_order` instead.
    """
    _bsp_traverse(node.level_order(), callback, userData)


@deprecate(
    "Iterate over nodes using "
    "'for n in node.inverted_level_order():' instead."
)
def bsp_traverse_inverted_level_order(
    node: tcod.bsp.BSP,
    callback: Callable[[tcod.bsp.BSP, Any], None],
    userData: Any = 0,
) -> None:
    """Traverse this nodes hierarchy with a callback.

    .. deprecated:: 2.0
       Use :any:`BSP.inverted_level_order` instead.
    """
    _bsp_traverse(node.inverted_level_order(), callback, userData)


@deprecate("Delete bsp children using 'node.children = ()' instead.")
def bsp_remove_sons(node: tcod.bsp.BSP) -> None:
    """Delete all children of a given node.  Not recommended.

    .. note::
       This function will add unnecessary complexity to your code.
       Don't use it.

    .. deprecated:: 2.0
       BSP deletion is automatic.
    """
    node.children = ()


@deprecate("libtcod objects are deleted automatically.")
def bsp_delete(node: tcod.bsp.BSP) -> None:
    """Exists for backward compatibility.  Does nothing.

    BSP's created by this library are automatically garbage collected once
    there are no references to the tree.
    This function exists for backwards compatibility.

    .. deprecated:: 2.0
       BSP deletion is automatic.
    """


@pending_deprecate()
def color_lerp(
    c1: Tuple[int, int, int], c2: Tuple[int, int, int], a: float
) -> Color:
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


@pending_deprecate()
def color_set_hsv(c: Color, h: float, s: float, v: float) -> None:
    """Set a color using: hue, saturation, and value parameters.

    Does not return a new Color.  ``c`` is modified inplace.

    Args:
        c (Union[Color, List[Any]]): A Color instance, or a list of any kind.
        h (float): Hue, from 0 to 360.
        s (float): Saturation, from 0 to 1.
        v (float): Value, from 0 to 1.
    """
    new_color = ffi.new("TCOD_color_t*")
    lib.TCOD_color_set_HSV(new_color, h, s, v)
    c[:] = new_color.r, new_color.g, new_color.b


@pending_deprecate()
def color_get_hsv(c: Tuple[int, int, int]) -> Tuple[float, float, float]:
    """Return the (hue, saturation, value) of a color.

    Args:
        c (Union[Tuple[int, int, int], Sequence[int]]):
            An (r, g, b) sequence or Color instance.

    Returns:
        Tuple[float, float, float]:
            A tuple with (hue, saturation, value) values, from 0 to 1.
    """
    hsv = ffi.new("float [3]")
    lib.TCOD_color_get_HSV(c, hsv, hsv + 1, hsv + 2)
    return hsv[0], hsv[1], hsv[2]


@pending_deprecate()
def color_scale_HSV(c: Color, scoef: float, vcoef: float) -> None:
    """Scale a color's saturation and value.

    Does not return a new Color.  ``c`` is modified inplace.

    Args:
        c (Union[Color, List[int]]): A Color instance, or an [r, g, b] list.
        scoef (float): Saturation multiplier, from 0 to 1.
                       Use 1 to keep current saturation.
        vcoef (float): Value multiplier, from 0 to 1.
                       Use 1 to keep current value.
    """
    color_p = ffi.new("TCOD_color_t*")
    color_p.r, color_p.g, color_p.b = c.r, c.g, c.b
    lib.TCOD_color_scale_HSV(color_p, scoef, vcoef)
    c[:] = color_p.r, color_p.g, color_p.b


@pending_deprecate()
def color_gen_map(
    colors: Iterable[Tuple[int, int, int]], indexes: Iterable[int]
) -> List[Color]:
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
        [Color(0, 0, 0), Color(51, 25, 0), Color(102, 51, 0), \
Color(153, 76, 0), Color(204, 102, 0), Color(255, 128, 0)]
    """
    ccolors = ffi.new("TCOD_color_t[]", colors)
    cindexes = ffi.new("int[]", indexes)
    cres = ffi.new("TCOD_color_t[]", max(indexes) + 1)
    lib.TCOD_color_gen_map(cres, len(ccolors), ccolors, cindexes)
    return [Color._new_from_cdata(cdata) for cdata in cres]


def console_init_root(
    w: int,
    h: int,
    title: Optional[str] = None,
    fullscreen: bool = False,
    renderer: Optional[int] = None,
    order: str = "C",
    vsync: Optional[bool] = None,
) -> tcod.console.Console:
    """Set up the primary display and return the root console.

    `w` and `h` are the columns and rows of the new window (in tiles.)

    `title` is an optional string to display on the windows title bar.

    `fullscreen` determines if the window will start in fullscreen.  Fullscreen
    mode is unreliable unless the renderer is set to `tcod.RENDERER_SDL2` or
    `tcod.RENDERER_OPENGL2`.

    `renderer` is the rendering back-end that libtcod will use.
    If you don't know which to pick, then use `tcod.RENDERER_SDL2`.
    Options are:

    * `tcod.RENDERER_SDL`:
      A deprecated software/SDL2 renderer.
    * `tcod.RENDERER_OPENGL`:
      A deprecated SDL2/OpenGL1 renderer.
    * `tcod.RENDERER_GLSL`:
      A deprecated SDL2/OpenGL2 renderer.
    * `tcod.RENDERER_SDL2`:
      The recommended SDL2 renderer.  Rendering is decided by SDL2 and can be
      changed by using an SDL2 hint.
    * `tcod.RENDERER_OPENGL2`:
      An SDL2/OPENGL2 renderer.  Usually faster than regular SDL2.
      Requires OpenGL 2.0 Core.

    `order` will affect how the array attributes of the returned root console
    are indexed.  `order='C'` is the default, but `order='F'` is recommended.

    If `vsync` is True then the frame-rate will be synchronized to the monitors
    vertical refresh rate.  This prevents screen tearing and avoids wasting
    computing power on overdraw.  If `vsync` is False then the frame-rate will
    be uncapped.  The default is False but will change to True in the future.
    This option only works with the SDL2 or OPENGL2 renderers, any other
    renderer will always have `vsync` disabled.

    .. versionchanged:: 4.3
        Added `order` parameter.
        `title` parameter is now optional.

    .. versionchanged:: 8.0
        The default `renderer` is now automatic instead of always being
        `RENDERER_SDL`.

    .. versionchanged:: 10.1
        Added the `vsync` parameter.
    """
    if title is None:
        # Use the scripts filename as the title.
        title = os.path.basename(sys.argv[0])
    if renderer is None:
        warnings.warn(
            "A renderer should be given, see the online documentation.",
            DeprecationWarning,
            stacklevel=2,
        )
        renderer = tcod.constants.RENDERER_SDL
    elif renderer in (
        tcod.constants.RENDERER_SDL,
        tcod.constants.RENDERER_OPENGL,
        tcod.constants.RENDERER_GLSL,
    ):
        warnings.warn(
            "The SDL, OPENGL, and GLSL renderers are deprecated.",
            DeprecationWarning,
            stacklevel=2,
        )
    NEW_RENDERER = renderer in (
        tcod.constants.RENDERER_SDL2,
        tcod.constants.RENDERER_OPENGL2,
    )
    if vsync is None and NEW_RENDERER:
        vsync = False
        warnings.warn(
            "vsync defaults to False, but the default will change to True in "
            "the future.  Provide a value for vsync to suppress this warning.",
            DeprecationWarning,
            stacklevel=2,
        )
    if not NEW_RENDERER:
        vsync = False
    _check(
        lib.TCOD_console_init_root_(
            w, h, _bytes(title), fullscreen, renderer, vsync
        )
    )
    console = tcod.console.Console._get_root(order)
    console.clear()
    return console


def console_set_custom_font(
    fontFile: AnyStr,
    flags: int = FONT_LAYOUT_ASCII_INCOL,
    nb_char_horiz: int = 0,
    nb_char_vertic: int = 0,
) -> None:
    """Load the custom font file at `fontFile`.

    Call this before function before calling :any:`tcod.console_init_root`.

    Flags can be a mix of the following:

    * tcod.FONT_LAYOUT_ASCII_INCOL:
      Decode tileset raw in column-major order.
    * tcod.FONT_LAYOUT_ASCII_INROW:
      Decode tileset raw in row-major order.
    * tcod.FONT_TYPE_GREYSCALE:
      Force tileset to be read as greyscale.
    * tcod.FONT_TYPE_GRAYSCALE
    * tcod.FONT_LAYOUT_TCOD:
      Unique layout used by libtcod.
    * tcod.FONT_LAYOUT_CP437:
      Decode a row-major Code Page 437 tileset into Unicode.

    `nb_char_horiz` and `nb_char_vertic` are the columns and rows of the font
    file respectfully.
    """
    if not os.path.exists(fontFile):
        raise RuntimeError(
            "File not found:\n\t%s" % (os.path.realpath(fontFile),)
        )
    _check(
        lib.TCOD_console_set_custom_font(
            _bytes(fontFile), flags, nb_char_horiz, nb_char_vertic
        )
    )


@deprecate("Check `con.width` instead.")
def console_get_width(con: tcod.console.Console) -> int:
    """Return the width of a console.

    Args:
        con (Console): Any Console instance.

    Returns:
        int: The width of a Console.

    .. deprecated:: 2.0
        Use `Console.width` instead.
    """
    return int(lib.TCOD_console_get_width(_console(con)))


@deprecate("Check `con.height` instead.")
def console_get_height(con: tcod.console.Console) -> int:
    """Return the height of a console.

    Args:
        con (Console): Any Console instance.

    Returns:
        int: The height of a Console.

    .. deprecated:: 2.0
        Use `Console.height` instead.
    """
    return int(lib.TCOD_console_get_height(_console(con)))


@pending_deprecate()
def console_map_ascii_code_to_font(
    asciiCode: int, fontCharX: int, fontCharY: int
) -> None:
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
    lib.TCOD_console_map_ascii_code_to_font(
        _int(asciiCode), fontCharX, fontCharY
    )


@pending_deprecate()
def console_map_ascii_codes_to_font(
    firstAsciiCode: int, nbCodes: int, fontCharX: int, fontCharY: int
) -> None:
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
    lib.TCOD_console_map_ascii_codes_to_font(
        _int(firstAsciiCode), nbCodes, fontCharX, fontCharY
    )


@pending_deprecate()
def console_map_string_to_font(s: str, fontCharX: int, fontCharY: int) -> None:
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


def console_is_fullscreen() -> bool:
    """Returns True if the display is fullscreen.

    Returns:
        bool: True if the display is fullscreen, otherwise False.
    """
    return bool(lib.TCOD_console_is_fullscreen())


def console_set_fullscreen(fullscreen: bool) -> None:
    """Change the display to be fullscreen or windowed.

    Args:
        fullscreen (bool): Use True to change to fullscreen.
                           Use False to change to windowed.
    """
    lib.TCOD_console_set_fullscreen(fullscreen)


@deprecate('Use the tcod.event module to check for "QUIT" type events.')
def console_is_window_closed() -> bool:
    """Returns True if the window has received and exit event.

    .. deprecated:: 9.3
        Use the :any:`tcod.event` module to check for "QUIT" type events.
    """
    return bool(lib.TCOD_console_is_window_closed())


@pending_deprecate()
def console_has_mouse_focus() -> bool:
    """Return True if the window has mouse focus."""
    return bool(lib.TCOD_console_has_mouse_focus())


@pending_deprecate()
def console_is_active() -> bool:
    """Return True if the window has keyboard focus."""
    return bool(lib.TCOD_console_is_active())


def console_set_window_title(title: str) -> None:
    """Change the current title bar string.

    Args:
        title (AnyStr): A string to change the title bar to.
    """
    lib.TCOD_console_set_window_title(_bytes(title))


@pending_deprecate()
def console_credits() -> None:
    lib.TCOD_console_credits()


def console_credits_reset() -> None:
    lib.TCOD_console_credits_reset()


def console_credits_render(x: int, y: int, alpha: bool) -> bool:
    return bool(lib.TCOD_console_credits_render(x, y, alpha))


def console_flush() -> None:
    """Update the display to represent the root consoles current state."""
    lib.TCOD_console_flush()


# drawing on a console
@deprecate("Set the `con.default_bg` attribute instead.")
def console_set_default_background(
    con: tcod.console.Console, col: Tuple[int, int, int]
) -> None:
    """Change the default background color for a console.

    Args:
        con (Console): Any Console instance.
        col (Union[Tuple[int, int, int], Sequence[int]]):
            An (r, g, b) sequence or Color instance.

    .. deprecated:: 8.5
        Use :any:`Console.default_bg` instead.
    """
    lib.TCOD_console_set_default_background(_console(con), col)


@deprecate("Set the `con.default_fg` attribute instead.")
def console_set_default_foreground(
    con: tcod.console.Console, col: Tuple[int, int, int]
) -> None:
    """Change the default foreground color for a console.

    Args:
        con (Console): Any Console instance.
        col (Union[Tuple[int, int, int], Sequence[int]]):
            An (r, g, b) sequence or Color instance.

    .. deprecated:: 8.5
        Use :any:`Console.default_fg` instead.
    """
    lib.TCOD_console_set_default_foreground(_console(con), col)


@deprecate("Call the `con.clear()` method instead.")
def console_clear(con: tcod.console.Console) -> None:
    """Reset a console to its default colors and the space character.

    Args:
        con (Console): Any Console instance.

    .. seealso::
       :any:`console_set_default_background`
       :any:`console_set_default_foreground`

    .. deprecated:: 8.5
        Call the :any:`Console.clear` method instead.
    """
    lib.TCOD_console_clear(_console(con))


@pending_deprecate()
def console_put_char(
    con: tcod.console.Console,
    x: int,
    y: int,
    c: Union[int, str],
    flag: int = BKGND_DEFAULT,
) -> None:
    """Draw the character c at x,y using the default colors and a blend mode.

    Args:
        con (Console): Any Console instance.
        x (int): Character x position from the left.
        y (int): Character y position from the top.
        c (Union[int, AnyStr]): Character to draw, can be an integer or string.
        flag (int): Blending mode to use, defaults to BKGND_DEFAULT.
    """
    lib.TCOD_console_put_char(_console(con), x, y, _int(c), flag)


@pending_deprecate()
def console_put_char_ex(
    con: tcod.console.Console,
    x: int,
    y: int,
    c: Union[int, str],
    fore: Tuple[int, int, int],
    back: Tuple[int, int, int],
) -> None:
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


@pending_deprecate()
def console_set_char_background(
    con: tcod.console.Console,
    x: int,
    y: int,
    col: Tuple[int, int, int],
    flag: int = BKGND_SET,
) -> None:
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
def console_set_char_foreground(
    con: tcod.console.Console, x: int, y: int, col: Tuple[int, int, int]
) -> None:
    """Change the foreground color of x,y to col.

    Args:
        con (Console): Any Console instance.
        x (int): Character x position from the left.
        y (int): Character y position from the top.
        col (Union[Tuple[int, int, int], Sequence[int]]):
            An (r, g, b) sequence or Color instance.

    .. deprecated:: 8.4
        Array access performs significantly faster than using this function.
        See :any:`Console.fg`.
    """
    lib.TCOD_console_set_char_foreground(_console(con), x, y, col)


@deprecate("Directly access a consoles characters with `console.ch`")
def console_set_char(
    con: tcod.console.Console, x: int, y: int, c: Union[int, str]
) -> None:
    """Change the character at x,y to c, keeping the current colors.

    Args:
        con (Console): Any Console instance.
        x (int): Character x position from the left.
        y (int): Character y position from the top.
        c (Union[int, AnyStr]): Character to draw, can be an integer or string.

    .. deprecated:: 8.4
        Array access performs significantly faster than using this function.
        See :any:`Console.ch`.
    """
    lib.TCOD_console_set_char(_console(con), x, y, _int(c))


@deprecate("Set the `con.default_bg_blend` attribute instead.")
def console_set_background_flag(con: tcod.console.Console, flag: int) -> None:
    """Change the default blend mode for this console.

    Args:
        con (Console): Any Console instance.
        flag (int): Blend mode to use by default.

    .. deprecated:: 8.5
        Set :any:`Console.default_bg_blend` instead.
    """
    lib.TCOD_console_set_background_flag(_console(con), flag)


@deprecate("Check the `con.default_bg_blend` attribute instead.")
def console_get_background_flag(con: tcod.console.Console) -> int:
    """Return this consoles current blend mode.

    Args:
        con (Console): Any Console instance.

    .. deprecated:: 8.5
        Check :any:`Console.default_bg_blend` instead.
    """
    return int(lib.TCOD_console_get_background_flag(_console(con)))


@deprecate("Set the `con.default_alignment` attribute instead.")
def console_set_alignment(con: tcod.console.Console, alignment: int) -> None:
    """Change this consoles current alignment mode.

    * tcod.LEFT
    * tcod.CENTER
    * tcod.RIGHT

    Args:
        con (Console): Any Console instance.
        alignment (int):

    .. deprecated:: 8.5
        Set :any:`Console.default_alignment` instead.
    """
    lib.TCOD_console_set_alignment(_console(con), alignment)


@deprecate("Check the `con.default_alignment` attribute instead.")
def console_get_alignment(con: tcod.console.Console) -> int:
    """Return this consoles current alignment mode.

    Args:
        con (Console): Any Console instance.

    .. deprecated:: 8.5
        Check :any:`Console.default_alignment` instead.
    """
    return int(lib.TCOD_console_get_alignment(_console(con)))


@deprecate("Call the `con.print_` method instead.")
def console_print(con: tcod.console.Console, x: int, y: int, fmt: str) -> None:
    """Print a color formatted string on a console.

    Args:
        con (Console): Any Console instance.
        x (int): Character x position from the left.
        y (int): Character y position from the top.
        fmt (AnyStr): A unicode or bytes string optionaly using color codes.

    .. deprecated:: 8.5
        Use :any:`Console.print_` instead.
    """
    lib.TCOD_console_printf(_console(con), x, y, _fmt(fmt))


@deprecate("Call the `con.print_` method instead.")
def console_print_ex(
    con: tcod.console.Console,
    x: int,
    y: int,
    flag: int,
    alignment: int,
    fmt: str,
) -> None:
    """Print a string on a console using a blend mode and alignment mode.

    Args:
        con (Console): Any Console instance.
        x (int): Character x position from the left.
        y (int): Character y position from the top.

    .. deprecated:: 8.5
        Use :any:`Console.print_` instead.
    """
    lib.TCOD_console_printf_ex(_console(con), x, y, flag, alignment, _fmt(fmt))


@deprecate("Call the `con.print_rect` method instead.")
def console_print_rect(
    con: tcod.console.Console, x: int, y: int, w: int, h: int, fmt: str
) -> int:
    """Print a string constrained to a rectangle.

    If h > 0 and the bottom of the rectangle is reached,
    the string is truncated. If h = 0,
    the string is only truncated if it reaches the bottom of the console.



    Returns:
        int: The number of lines of text once word-wrapped.

    .. deprecated:: 8.5
        Use :any:`Console.print_rect` instead.
    """
    return int(
        lib.TCOD_console_printf_rect(_console(con), x, y, w, h, _fmt(fmt))
    )


@deprecate("Call the `con.print_rect` method instead.")
def console_print_rect_ex(
    con: tcod.console.Console,
    x: int,
    y: int,
    w: int,
    h: int,
    flag: int,
    alignment: int,
    fmt: str,
) -> int:
    """Print a string constrained to a rectangle with blend and alignment.

    Returns:
        int: The number of lines of text once word-wrapped.

    .. deprecated:: 8.5
        Use :any:`Console.print_rect` instead.
    """
    return int(
        lib.TCOD_console_printf_rect_ex(
            _console(con), x, y, w, h, flag, alignment, _fmt(fmt)
        )
    )


@deprecate("Call the `con.get_height_rect` method instead.")
def console_get_height_rect(
    con: tcod.console.Console, x: int, y: int, w: int, h: int, fmt: str
) -> int:
    """Return the height of this text once word-wrapped into this rectangle.

    Returns:
        int: The number of lines of text once word-wrapped.

    .. deprecated:: 8.5
        Use :any:`Console.get_height_rect` instead.
    """
    return int(
        lib.TCOD_console_get_height_rect_fmt(
            _console(con), x, y, w, h, _fmt(fmt)
        )
    )


@deprecate("Call the `con.rect` method instead.")
def console_rect(
    con: tcod.console.Console,
    x: int,
    y: int,
    w: int,
    h: int,
    clr: bool,
    flag: int = BKGND_DEFAULT,
) -> None:
    """Draw a the background color on a rect optionally clearing the text.

    If clr is True the affected tiles are changed to space character.

    .. deprecated:: 8.5
        Use :any:`Console.rect` instead.
    """
    lib.TCOD_console_rect(_console(con), x, y, w, h, clr, flag)


@deprecate("Call the `con.hline` method instead.")
def console_hline(
    con: tcod.console.Console,
    x: int,
    y: int,
    l: int,
    flag: int = BKGND_DEFAULT,
) -> None:
    """Draw a horizontal line on the console.

    This always uses the character 196, the horizontal line character.

    .. deprecated:: 8.5
        Use :any:`Console.hline` instead.
    """
    lib.TCOD_console_hline(_console(con), x, y, l, flag)


@deprecate("Call the `con.vline` method instead.")
def console_vline(
    con: tcod.console.Console,
    x: int,
    y: int,
    l: int,
    flag: int = BKGND_DEFAULT,
) -> None:
    """Draw a vertical line on the console.

    This always uses the character 179, the vertical line character.

    .. deprecated:: 8.5
        Use :any:`Console.vline` instead.
    """
    lib.TCOD_console_vline(_console(con), x, y, l, flag)


@deprecate("Call the `con.print_frame` method instead.")
def console_print_frame(
    con: tcod.console.Console,
    x: int,
    y: int,
    w: int,
    h: int,
    clear: bool = True,
    flag: int = BKGND_DEFAULT,
    fmt: str = "",
) -> None:
    """Draw a framed rectangle with optinal text.

    This uses the default background color and blend mode to fill the
    rectangle and the default foreground to draw the outline.

    `fmt` will be printed on the inside of the rectangle, word-wrapped.
    If `fmt` is empty then no title will be drawn.

    .. versionchanged:: 8.2
        Now supports Unicode strings.

    .. deprecated:: 8.5
        Use :any:`Console.print_frame` instead.
    """
    fmt = _fmt(fmt) if fmt else ffi.NULL
    lib.TCOD_console_printf_frame(_console(con), x, y, w, h, clear, flag, fmt)


@pending_deprecate()
def console_set_color_control(
    con: int, fore: Tuple[int, int, int], back: Tuple[int, int, int]
) -> None:
    """Configure :any:`color controls`.

    Args:
        con (int): :any:`Color control` constant to modify.
        fore (Union[Tuple[int, int, int], Sequence[int]]):
            An (r, g, b) sequence or Color instance.
        back (Union[Tuple[int, int, int], Sequence[int]]):
            An (r, g, b) sequence or Color instance.
    """
    lib.TCOD_console_set_color_control(con, fore, back)


@deprecate("Check the `con.default_bg` attribute instead.")
def console_get_default_background(con: tcod.console.Console) -> Color:
    """Return this consoles default background color.

    .. deprecated:: 8.5
        Use :any:`Console.default_bg` instead.
    """
    return Color._new_from_cdata(
        lib.TCOD_console_get_default_background(_console(con))
    )


@deprecate("Check the `con.default_fg` attribute instead.")
def console_get_default_foreground(con: tcod.console.Console) -> Color:
    """Return this consoles default foreground color.

    .. deprecated:: 8.5
        Use :any:`Console.default_fg` instead.
    """
    return Color._new_from_cdata(
        lib.TCOD_console_get_default_foreground(_console(con))
    )


@deprecate("Directly access a consoles background color with `console.bg`")
def console_get_char_background(
    con: tcod.console.Console, x: int, y: int
) -> Color:
    """Return the background color at the x,y of this console.

    .. deprecated:: 8.4
        Array access performs significantly faster than using this function.
        See :any:`Console.bg`.
    """
    return Color._new_from_cdata(
        lib.TCOD_console_get_char_background(_console(con), x, y)
    )


@deprecate("Directly access a consoles foreground color with `console.fg`")
def console_get_char_foreground(
    con: tcod.console.Console, x: int, y: int
) -> Color:
    """Return the foreground color at the x,y of this console.

    .. deprecated:: 8.4
        Array access performs significantly faster than using this function.
        See :any:`Console.fg`.
    """
    return Color._new_from_cdata(
        lib.TCOD_console_get_char_foreground(_console(con), x, y)
    )


@deprecate("Directly access a consoles characters with `console.ch`")
def console_get_char(con: tcod.console.Console, x: int, y: int) -> int:
    """Return the character at the x,y of this console.

    .. deprecated:: 8.4
        Array access performs significantly faster than using this function.
        See :any:`Console.ch`.
    """
    return lib.TCOD_console_get_char(_console(con), x, y)  # type: ignore


@pending_deprecate()
def console_set_fade(fade: int, fadingColor: Tuple[int, int, int]) -> None:
    lib.TCOD_console_set_fade(fade, fadingColor)


@pending_deprecate()
def console_get_fade() -> int:
    return int(lib.TCOD_console_get_fade())


@pending_deprecate()
def console_get_fading_color() -> Color:
    return Color._new_from_cdata(lib.TCOD_console_get_fading_color())


# handling keyboard input
@deprecate("Use the tcod.event.wait function to wait for events.")
def console_wait_for_keypress(flush: bool) -> Key:
    """Block until the user presses a key, then returns a new Key.

    Args:
        flush bool: If True then the event queue is cleared before waiting
                    for the next event.

    Returns:
        Key: A new Key instance.

    .. deprecated:: 9.3
        Use the :any:`tcod.event.wait` function to wait for events.
    """
    key = Key()
    lib.TCOD_console_wait_for_keypress_wrapper(key.key_p, flush)
    return key


@deprecate("Use the tcod.event.get function to check for events.")
def console_check_for_keypress(flags: int = KEY_RELEASED) -> Key:
    """
    .. deprecated:: 9.3
        Use the :any:`tcod.event.get` function to check for events.
    """
    key = Key()
    lib.TCOD_console_check_for_keypress_wrapper(key.key_p, flags)
    return key


@pending_deprecate()
def console_is_key_pressed(key: int) -> bool:
    return bool(lib.TCOD_console_is_key_pressed(key))


# using offscreen consoles
@deprecate("Create a console using `tcod.console.Console(...)` instead.")
def console_new(w: int, h: int) -> tcod.console.Console:
    """Return an offscreen console of size: w,h.

    .. deprecated:: 8.5
        Create new consoles using :any:`tcod.console.Console` instead of this
        function.
    """
    return tcod.console.Console(w, h)


def console_from_file(filename: str) -> tcod.console.Console:
    """Return a new console object from a filename.

    The file format is automactially determined.  This can load REXPaint `.xp`,
    ASCII Paint `.apf`, or Non-delimited ASCII `.asc` files.

    Args:
        filename (Text): The path to the file, as a string.

    Returns: A new :any`Console` instance.
    """
    return tcod.console.Console._from_cdata(
        lib.TCOD_console_from_file(filename.encode("utf-8"))
    )


@deprecate("Call the `Console.blit` method instead.")
def console_blit(
    src: tcod.console.Console,
    x: int,
    y: int,
    w: int,
    h: int,
    dst: tcod.console.Console,
    xdst: int,
    ydst: int,
    ffade: float = 1.0,
    bfade: float = 1.0,
) -> None:
    """Blit the console src from x,y,w,h to console dst at xdst,ydst.

    .. deprecated:: 8.5
        Call the :any:`Console.blit` method instead.
    """
    lib.TCOD_console_blit(
        _console(src), x, y, w, h, _console(dst), xdst, ydst, ffade, bfade
    )


@deprecate(
    "Pass the key color to `Console.blit` instead of calling this function."
)
def console_set_key_color(
    con: tcod.console.Console, col: Tuple[int, int, int]
) -> None:
    """Set a consoles blit transparent color.

    .. deprecated:: 8.5
        Pass the key color to :any:`tcod.console.Console.blit` instead of
        calling this function.
    """
    lib.TCOD_console_set_key_color(_console(con), col)
    if hasattr(con, "set_key_color"):
        con.set_key_color(col)


def console_delete(con: tcod.console.Console) -> None:
    """Closes the window if `con` is the root console.

    libtcod objects are automatically garbage collected once they go out of
    scope.

    This function exists for backwards compatibility.

    .. deprecated:: 9.3
        This function is not needed for normal :any:`tcod.console.Console`'s.
        The root console should be used in a with statement instead to ensure
        that it closes.
    """
    con = _console(con)
    if con == ffi.NULL:
        lib.TCOD_console_delete(con)
        warnings.warn(
            "Instead of this call you should use a with statement to ensure"
            " the root console closes, for example:"
            "\n    with tcod.console_init_root(...) as root_console:"
            "\n        ...",
            DeprecationWarning,
            stacklevel=2,
        )
    else:
        warnings.warn(
            "You no longer need to make this call, "
            "Console's are deleted when they go out of scope.",
            DeprecationWarning,
            stacklevel=2,
        )


@deprecate("Assign to the console.fg array instead.")
def console_fill_foreground(
    con: tcod.console.Console,
    r: Sequence[int],
    g: Sequence[int],
    b: Sequence[int],
) -> None:
    """Fill the foregound of a console with r,g,b.

    Args:
        con (Console): Any Console instance.
        r (Sequence[int]): An array of integers with a length of width*height.
        g (Sequence[int]): An array of integers with a length of width*height.
        b (Sequence[int]): An array of integers with a length of width*height.

    .. deprecated:: 8.4
        You should assign to :any:`tcod.console.Console.fg` instead.
    """
    if len(r) != len(g) or len(r) != len(b):
        raise TypeError("R, G and B must all have the same size.")
    if (
        isinstance(r, np.ndarray)
        and isinstance(g, np.ndarray)
        and isinstance(b, np.ndarray)
    ):
        # numpy arrays, use numpy's ctypes functions
        r_ = np.ascontiguousarray(r, dtype=np.intc)
        g_ = np.ascontiguousarray(g, dtype=np.intc)
        b_ = np.ascontiguousarray(b, dtype=np.intc)
        cr = ffi.cast("int *", r_.ctypes.data)
        cg = ffi.cast("int *", g_.ctypes.data)
        cb = ffi.cast("int *", b_.ctypes.data)
    else:
        # otherwise convert using ffi arrays
        cr = ffi.new("int[]", r)
        cg = ffi.new("int[]", g)
        cb = ffi.new("int[]", b)

    lib.TCOD_console_fill_foreground(_console(con), cr, cg, cb)


@deprecate("Assign to the console.bg array instead.")
def console_fill_background(
    con: tcod.console.Console,
    r: Sequence[int],
    g: Sequence[int],
    b: Sequence[int],
) -> None:
    """Fill the backgound of a console with r,g,b.

    Args:
        con (Console): Any Console instance.
        r (Sequence[int]): An array of integers with a length of width*height.
        g (Sequence[int]): An array of integers with a length of width*height.
        b (Sequence[int]): An array of integers with a length of width*height.

    .. deprecated:: 8.4
        You should assign to :any:`tcod.console.Console.bg` instead.
    """
    if len(r) != len(g) or len(r) != len(b):
        raise TypeError("R, G and B must all have the same size.")
    if (
        isinstance(r, np.ndarray)
        and isinstance(g, np.ndarray)
        and isinstance(b, np.ndarray)
    ):
        # numpy arrays, use numpy's ctypes functions
        r_ = np.ascontiguousarray(r, dtype=np.intc)
        g_ = np.ascontiguousarray(g, dtype=np.intc)
        b_ = np.ascontiguousarray(b, dtype=np.intc)
        cr = ffi.cast("int *", r_.ctypes.data)
        cg = ffi.cast("int *", g_.ctypes.data)
        cb = ffi.cast("int *", b_.ctypes.data)
    else:
        # otherwise convert using ffi arrays
        cr = ffi.new("int[]", r)
        cg = ffi.new("int[]", g)
        cb = ffi.new("int[]", b)

    lib.TCOD_console_fill_background(_console(con), cr, cg, cb)


@deprecate("Assign to the console.ch array instead.")
def console_fill_char(con: tcod.console.Console, arr: Sequence[int]) -> None:
    """Fill the character tiles of a console with an array.

    `arr` is an array of integers with a length of the consoles width and
    height.

    .. deprecated:: 8.4
        You should assign to :any:`tcod.console.Console.ch` instead.
    """
    if isinstance(arr, np.ndarray):
        # numpy arrays, use numpy's ctypes functions
        np_array = np.ascontiguousarray(arr, dtype=np.intc)
        carr = ffi.cast("int *", np_array.ctypes.data)
    else:
        # otherwise convert using the ffi module
        carr = ffi.new("int[]", arr)

    lib.TCOD_console_fill_char(_console(con), carr)


@pending_deprecate()
def console_load_asc(con: tcod.console.Console, filename: str) -> bool:
    """Update a console from a non-delimited ASCII `.asc` file."""
    return bool(
        lib.TCOD_console_load_asc(_console(con), filename.encode("utf-8"))
    )


@pending_deprecate()
def console_save_asc(con: tcod.console.Console, filename: str) -> bool:
    """Save a console to a non-delimited ASCII `.asc` file."""
    return bool(
        lib.TCOD_console_save_asc(_console(con), filename.encode("utf-8"))
    )


@pending_deprecate()
def console_load_apf(con: tcod.console.Console, filename: str) -> bool:
    """Update a console from an ASCII Paint `.apf` file."""
    return bool(
        lib.TCOD_console_load_apf(_console(con), filename.encode("utf-8"))
    )


@pending_deprecate()
def console_save_apf(con: tcod.console.Console, filename: str) -> bool:
    """Save a console to an ASCII Paint `.apf` file."""
    return bool(
        lib.TCOD_console_save_apf(_console(con), filename.encode("utf-8"))
    )


def console_load_xp(con: tcod.console.Console, filename: str) -> bool:
    """Update a console from a REXPaint `.xp` file."""
    return bool(
        lib.TCOD_console_load_xp(_console(con), filename.encode("utf-8"))
    )


def console_save_xp(
    con: tcod.console.Console, filename: str, compress_level: int = 9
) -> bool:
    """Save a console to a REXPaint `.xp` file."""
    return bool(
        lib.TCOD_console_save_xp(
            _console(con), filename.encode("utf-8"), compress_level
        )
    )


def console_from_xp(filename: str) -> tcod.console.Console:
    """Return a single console from a REXPaint `.xp` file."""
    return tcod.console.Console._from_cdata(
        lib.TCOD_console_from_xp(filename.encode("utf-8"))
    )


def console_list_load_xp(
    filename: str
) -> Optional[List[tcod.console.Console]]:
    """Return a list of consoles from a REXPaint `.xp` file."""
    tcod_list = lib.TCOD_console_list_from_xp(filename.encode("utf-8"))
    if tcod_list == ffi.NULL:
        return None
    try:
        python_list = []
        lib.TCOD_list_reverse(tcod_list)
        while not lib.TCOD_list_is_empty(tcod_list):
            python_list.append(
                tcod.console.Console._from_cdata(lib.TCOD_list_pop(tcod_list))
            )
        return python_list
    finally:
        lib.TCOD_list_delete(tcod_list)


def console_list_save_xp(
    console_list: Sequence[tcod.console.Console],
    filename: str,
    compress_level: int = 9,
) -> bool:
    """Save a list of consoles to a REXPaint `.xp` file."""
    tcod_list = lib.TCOD_list_new()
    try:
        for console in console_list:
            lib.TCOD_list_push(tcod_list, _console(console))
        return bool(
            lib.TCOD_console_list_save_xp(
                tcod_list, filename.encode("utf-8"), compress_level
            )
        )
    finally:
        lib.TCOD_list_delete(tcod_list)


@pending_deprecate()
def path_new_using_map(
    m: tcod.map.Map, dcost: float = 1.41
) -> tcod.path.AStar:
    """Return a new AStar using the given Map.

    Args:
        m (Map): A Map instance.
        dcost (float): The path-finding cost of diagonal movement.
                       Can be set to 0 to disable diagonal movement.
    Returns:
        AStar: A new AStar instance.
    """
    return tcod.path.AStar(m, dcost)


@pending_deprecate()
def path_new_using_function(
    w: int,
    h: int,
    func: Callable[[int, int, int, int, Any], float],
    userData: Any = 0,
    dcost: float = 1.41,
) -> tcod.path.AStar:
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
        tcod.path._EdgeCostFunc((func, userData), (w, h)), dcost
    )


@pending_deprecate()
def path_compute(
    p: tcod.path.AStar, ox: int, oy: int, dx: int, dy: int
) -> bool:
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
    return bool(lib.TCOD_path_compute(p._path_c, ox, oy, dx, dy))


@pending_deprecate()
def path_get_origin(p: tcod.path.AStar) -> Tuple[int, int]:
    """Get the current origin position.

    This point moves when :any:`path_walk` returns the next x,y step.

    Args:
        p (AStar): An AStar instance.
    Returns:
        Tuple[int, int]: An (x, y) point.
    """
    x = ffi.new("int *")
    y = ffi.new("int *")
    lib.TCOD_path_get_origin(p._path_c, x, y)
    return x[0], y[0]


@pending_deprecate()
def path_get_destination(p: tcod.path.AStar) -> Tuple[int, int]:
    """Get the current destination position.

    Args:
        p (AStar): An AStar instance.
    Returns:
        Tuple[int, int]: An (x, y) point.
    """
    x = ffi.new("int *")
    y = ffi.new("int *")
    lib.TCOD_path_get_destination(p._path_c, x, y)
    return x[0], y[0]


@pending_deprecate()
def path_size(p: tcod.path.AStar) -> int:
    """Return the current length of the computed path.

    Args:
        p (AStar): An AStar instance.
    Returns:
        int: Length of the path.
    """
    return int(lib.TCOD_path_size(p._path_c))


@pending_deprecate()
def path_reverse(p: tcod.path.AStar) -> None:
    """Reverse the direction of a path.

    This effectively swaps the origin and destination points.

    Args:
        p (AStar): An AStar instance.
    """
    lib.TCOD_path_reverse(p._path_c)


@pending_deprecate()
def path_get(p: tcod.path.AStar, idx: int) -> Tuple[int, int]:
    """Get a point on a path.

    Args:
        p (AStar): An AStar instance.
        idx (int): Should be in range: 0 <= inx < :any:`path_size`
    """
    x = ffi.new("int *")
    y = ffi.new("int *")
    lib.TCOD_path_get(p._path_c, idx, x, y)
    return x[0], y[0]


@pending_deprecate()
def path_is_empty(p: tcod.path.AStar) -> bool:
    """Return True if a path is empty.

    Args:
        p (AStar): An AStar instance.
    Returns:
        bool: True if a path is empty.  Otherwise False.
    """
    return bool(lib.TCOD_path_is_empty(p._path_c))


@pending_deprecate()
def path_walk(
    p: tcod.path.AStar, recompute: bool
) -> Union[Tuple[int, int], Tuple[None, None]]:
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
    x = ffi.new("int *")
    y = ffi.new("int *")
    if lib.TCOD_path_walk(p._path_c, x, y, recompute):
        return x[0], y[0]
    return None, None


@deprecate("libtcod objects are deleted automatically.")
def path_delete(p: tcod.path.AStar) -> None:
    """Does nothing. libtcod objects are managed by Python's garbage collector.

    This function exists for backwards compatibility with libtcodpy.
    """


@pending_deprecate()
def dijkstra_new(m: tcod.map.Map, dcost: float = 1.41) -> tcod.path.Dijkstra:
    return tcod.path.Dijkstra(m, dcost)


@pending_deprecate()
def dijkstra_new_using_function(
    w: int,
    h: int,
    func: Callable[[int, int, int, int, Any], float],
    userData: Any = 0,
    dcost: float = 1.41,
) -> tcod.path.Dijkstra:
    return tcod.path.Dijkstra(
        tcod.path._EdgeCostFunc((func, userData), (w, h)), dcost
    )


@pending_deprecate()
def dijkstra_compute(p: tcod.path.Dijkstra, ox: int, oy: int) -> None:
    lib.TCOD_dijkstra_compute(p._path_c, ox, oy)


@pending_deprecate()
def dijkstra_path_set(p: tcod.path.Dijkstra, x: int, y: int) -> bool:
    return bool(lib.TCOD_dijkstra_path_set(p._path_c, x, y))


@pending_deprecate()
def dijkstra_get_distance(p: tcod.path.Dijkstra, x: int, y: int) -> int:
    return int(lib.TCOD_dijkstra_get_distance(p._path_c, x, y))


@pending_deprecate()
def dijkstra_size(p: tcod.path.Dijkstra) -> int:
    return int(lib.TCOD_dijkstra_size(p._path_c))


@pending_deprecate()
def dijkstra_reverse(p: tcod.path.Dijkstra) -> None:
    lib.TCOD_dijkstra_reverse(p._path_c)


@pending_deprecate()
def dijkstra_get(p: tcod.path.Dijkstra, idx: int) -> Tuple[int, int]:
    x = ffi.new("int *")
    y = ffi.new("int *")
    lib.TCOD_dijkstra_get(p._path_c, idx, x, y)
    return x[0], y[0]


@pending_deprecate()
def dijkstra_is_empty(p: tcod.path.Dijkstra) -> bool:
    return bool(lib.TCOD_dijkstra_is_empty(p._path_c))


@pending_deprecate()
def dijkstra_path_walk(
    p: tcod.path.Dijkstra
) -> Union[Tuple[int, int], Tuple[None, None]]:
    x = ffi.new("int *")
    y = ffi.new("int *")
    if lib.TCOD_dijkstra_path_walk(p._path_c, x, y):
        return x[0], y[0]
    return None, None


@deprecate("libtcod objects are deleted automatically.")
def dijkstra_delete(p: tcod.path.Dijkstra) -> None:
    """Does nothing. libtcod objects are managed by Python's garbage collector.

    This function exists for backwards compatibility with libtcodpy.
    """


def _heightmap_cdata(array: np.ndarray) -> ffi.CData:
    """Return a new TCOD_heightmap_t instance using an array.

    Formatting is verified during this function.
    """
    if array.flags["F_CONTIGUOUS"]:
        array = array.transpose()
    if not array.flags["C_CONTIGUOUS"]:
        raise ValueError("array must be a contiguous segment.")
    if array.dtype != np.float32:
        raise ValueError("array dtype must be float32, not %r" % array.dtype)
    width, height = array.shape
    pointer = ffi.cast("float *", array.ctypes.data)
    return ffi.new("TCOD_heightmap_t *", (width, height, pointer))


@pending_deprecate()
def heightmap_new(w: int, h: int, order: str = "C") -> np.ndarray:
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
    if order == "C":
        return np.zeros((h, w), np.float32, order="C")
    elif order == "F":
        return np.zeros((w, h), np.float32, order="F")
    else:
        raise ValueError("Invalid order parameter, should be 'C' or 'F'.")


@deprecate("Assign to heightmaps as a NumPy array instead.")
def heightmap_set_value(hm: np.ndarray, x: int, y: int, value: float) -> None:
    """Set the value of a point on a heightmap.

    .. deprecated:: 2.0
        `hm` is a NumPy array, so values should be assigned to it directly.
    """
    if hm.flags["C_CONTIGUOUS"]:
        warnings.warn(
            "Assign to this heightmap with hm[i,j] = value\n"
            "consider using order='F'",
            DeprecationWarning,
            stacklevel=2,
        )
        hm[y, x] = value
    elif hm.flags["F_CONTIGUOUS"]:
        warnings.warn(
            "Assign to this heightmap with hm[x,y] = value",
            DeprecationWarning,
            stacklevel=2,
        )
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


@pending_deprecate()
def heightmap_normalize(
    hm: np.ndarray, mi: float = 0.0, ma: float = 1.0
) -> None:
    """Normalize heightmap values between ``mi`` and ``ma``.

    Args:
        mi (float): The lowest value after normalization.
        ma (float): The highest value after normalization.
    """
    lib.TCOD_heightmap_normalize(_heightmap_cdata(hm), mi, ma)


@pending_deprecate()
def heightmap_lerp_hm(
    hm1: np.ndarray, hm2: np.ndarray, hm3: np.ndarray, coef: float
) -> None:
    """Perform linear interpolation between two heightmaps storing the result
    in ``hm3``.

    This is the same as doing ``hm3[:] = hm1[:] + (hm2[:] - hm1[:]) * coef``

    Args:
        hm1 (numpy.ndarray): The first heightmap.
        hm2 (numpy.ndarray): The second heightmap to add to the first.
        hm3 (numpy.ndarray): A destination heightmap to store the result.
        coef (float): The linear interpolation coefficient.
    """
    lib.TCOD_heightmap_lerp_hm(
        _heightmap_cdata(hm1),
        _heightmap_cdata(hm2),
        _heightmap_cdata(hm3),
        coef,
    )


@deprecate("Add 2 arrays using `hm3 = hm1 + hm2`")
def heightmap_add_hm(
    hm1: np.ndarray, hm2: np.ndarray, hm3: np.ndarray
) -> None:
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
def heightmap_multiply_hm(
    hm1: np.ndarray, hm2: np.ndarray, hm3: np.ndarray
) -> None:
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


@pending_deprecate()
def heightmap_add_hill(
    hm: np.ndarray, x: float, y: float, radius: float, height: float
) -> None:
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


@pending_deprecate()
def heightmap_dig_hill(
    hm: np.ndarray, x: float, y: float, radius: float, height: float
) -> None:
    """

    This function takes the highest value (if height > 0) or the lowest
    (if height < 0) between the map and the hill.

    It's main goal is to carve things in maps (like rivers) by digging hills
    along a curve.

    Args:
        hm (numpy.ndarray): A numpy.ndarray formatted for heightmap functions.
        x (float): The x position at the center of the new carving.
        y (float): The y position at the center of the new carving.
        radius (float): The size of the carving.
        height (float): The height or depth of the hill to dig out.
    """
    lib.TCOD_heightmap_dig_hill(_heightmap_cdata(hm), x, y, radius, height)


@pending_deprecate()
def heightmap_rain_erosion(
    hm: np.ndarray,
    nbDrops: int,
    erosionCoef: float,
    sedimentationCoef: float,
    rnd: Optional[tcod.random.Random] = None,
) -> None:
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
        _heightmap_cdata(hm),
        nbDrops,
        erosionCoef,
        sedimentationCoef,
        rnd.random_c if rnd else ffi.NULL,
    )


@pending_deprecate()
def heightmap_kernel_transform(
    hm: np.ndarray,
    kernelsize: int,
    dx: Sequence[int],
    dy: Sequence[int],
    weight: Sequence[float],
    minLevel: float,
    maxLevel: float,
) -> None:
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
    cdx = ffi.new("int[]", dx)
    cdy = ffi.new("int[]", dy)
    cweight = ffi.new("float[]", weight)
    lib.TCOD_heightmap_kernel_transform(
        _heightmap_cdata(hm), kernelsize, cdx, cdy, cweight, minLevel, maxLevel
    )


@pending_deprecate()
def heightmap_add_voronoi(
    hm: np.ndarray,
    nbPoints: Any,
    nbCoef: int,
    coef: Sequence[float],
    rnd: Optional[tcod.random.Random] = None,
) -> None:
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
    ccoef = ffi.new("float[]", coef)
    lib.TCOD_heightmap_add_voronoi(
        _heightmap_cdata(hm),
        nbPoints,
        nbCoef,
        ccoef,
        rnd.random_c if rnd else ffi.NULL,
    )


@deprecate("Arrays of noise should be sampled using the tcod.noise module.")
def heightmap_add_fbm(
    hm: np.ndarray,
    noise: tcod.noise.Noise,
    mulx: float,
    muly: float,
    addx: float,
    addy: float,
    octaves: float,
    delta: float,
    scale: float,
) -> None:
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
    lib.TCOD_heightmap_add_fbm(
        _heightmap_cdata(hm),
        noise,
        mulx,
        muly,
        addx,
        addy,
        octaves,
        delta,
        scale,
    )


@deprecate("Arrays of noise should be sampled using the tcod.noise module.")
def heightmap_scale_fbm(
    hm: np.ndarray,
    noise: tcod.noise.Noise,
    mulx: float,
    muly: float,
    addx: float,
    addy: float,
    octaves: float,
    delta: float,
    scale: float,
) -> None:
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
    lib.TCOD_heightmap_scale_fbm(
        _heightmap_cdata(hm),
        noise,
        mulx,
        muly,
        addx,
        addy,
        octaves,
        delta,
        scale,
    )


@pending_deprecate()
def heightmap_dig_bezier(
    hm: np.ndarray,
    px: Tuple[int, int, int, int],
    py: Tuple[int, int, int, int],
    startRadius: float,
    startDepth: float,
    endRadius: float,
    endDepth: float,
) -> None:
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
    lib.TCOD_heightmap_dig_bezier(
        _heightmap_cdata(hm),
        px,
        py,
        startRadius,
        startDepth,
        endRadius,
        endDepth,
    )


@deprecate("This object can be accessed as a NumPy array.")
def heightmap_get_value(hm: np.ndarray, x: int, y: int) -> float:
    """Return the value at ``x``, ``y`` in a heightmap.

    .. deprecated:: 2.0
        Access `hm` as a NumPy array instead.
    """
    if hm.flags["C_CONTIGUOUS"]:
        warnings.warn(
            "Get a value from this heightmap with hm[i,j]\n"
            "consider using order='F'",
            DeprecationWarning,
            stacklevel=2,
        )
        return hm[y, x]  # type: ignore
    elif hm.flags["F_CONTIGUOUS"]:
        warnings.warn(
            "Get a value from this heightmap with hm[x,y]",
            DeprecationWarning,
            stacklevel=2,
        )
        return hm[x, y]  # type: ignore
    else:
        raise ValueError("This array is not contiguous.")


@pending_deprecate()
def heightmap_get_interpolated_value(
    hm: np.ndarray, x: float, y: float
) -> float:
    """Return the interpolated height at non integer coordinates.

    Args:
        hm (numpy.ndarray): A numpy.ndarray formatted for heightmap functions.
        x (float): A floating point x coordinate.
        y (float): A floating point y coordinate.

    Returns:
        float: The value at ``x``, ``y``.
    """
    return float(
        lib.TCOD_heightmap_get_interpolated_value(_heightmap_cdata(hm), x, y)
    )


@pending_deprecate()
def heightmap_get_slope(hm: np.ndarray, x: int, y: int) -> float:
    """Return the slope between 0 and (pi / 2) at given coordinates.

    Args:
        hm (numpy.ndarray): A numpy.ndarray formatted for heightmap functions.
        x (int): The x coordinate.
        y (int): The y coordinate.

    Returns:
        float: The steepness at ``x``, ``y``.  From 0 to (pi / 2)
    """
    return float(lib.TCOD_heightmap_get_slope(_heightmap_cdata(hm), x, y))


@pending_deprecate()
def heightmap_get_normal(
    hm: np.ndarray, x: float, y: float, waterLevel: float
) -> Tuple[float, float, float]:
    """Return the map normal at given coordinates.

    Args:
        hm (numpy.ndarray): A numpy.ndarray formatted for heightmap functions.
        x (float): The x coordinate.
        y (float): The y coordinate.
        waterLevel (float): The heightmap is considered flat below this value.

    Returns:
        Tuple[float, float, float]: An (x, y, z) vector normal.
    """
    cn = ffi.new("float[3]")
    lib.TCOD_heightmap_get_normal(_heightmap_cdata(hm), x, y, cn, waterLevel)
    return tuple(cn)  # type: ignore


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
    return int(lib.TCOD_heightmap_count_cells(_heightmap_cdata(hm), mi, ma))


@pending_deprecate()
def heightmap_has_land_on_border(hm: np.ndarray, waterlevel: float) -> bool:
    """Returns True if the map edges are below ``waterlevel``, otherwise False.

    Args:
        hm (numpy.ndarray): A numpy.ndarray formatted for heightmap functions.
        waterLevel (float): The water level to use.

    Returns:
        bool: True if the map edges are below ``waterlevel``, otherwise False.
    """
    return bool(
        lib.TCOD_heightmap_has_land_on_border(_heightmap_cdata(hm), waterlevel)
    )


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
    mi = ffi.new("float *")
    ma = ffi.new("float *")
    lib.TCOD_heightmap_get_minmax(_heightmap_cdata(hm), mi, ma)
    return mi[0], ma[0]


@deprecate("libtcod objects are deleted automatically.")
def heightmap_delete(hm: Any) -> None:
    """Does nothing. libtcod objects are managed by Python's garbage collector.

    This function exists for backwards compatibility with libtcodpy.

    .. deprecated:: 2.0
        libtcod-cffi deletes heightmaps automatically.
    """


@pending_deprecate()
def image_new(width: int, height: int) -> tcod.image.Image:
    return tcod.image.Image(width, height)


@pending_deprecate()
def image_clear(image: tcod.image.Image, col: Tuple[int, int, int]) -> None:
    image.clear(col)


@pending_deprecate()
def image_invert(image: tcod.image.Image) -> None:
    image.invert()


@pending_deprecate()
def image_hflip(image: tcod.image.Image) -> None:
    image.hflip()


@pending_deprecate()
def image_rotate90(image: tcod.image.Image, num: int = 1) -> None:
    image.rotate90(num)


@pending_deprecate()
def image_vflip(image: tcod.image.Image) -> None:
    image.vflip()


@pending_deprecate()
def image_scale(image: tcod.image.Image, neww: int, newh: int) -> None:
    image.scale(neww, newh)


@pending_deprecate()
def image_set_key_color(
    image: tcod.image.Image, col: Tuple[int, int, int]
) -> None:
    image.set_key_color(col)


@pending_deprecate()
def image_get_alpha(image: tcod.image.Image, x: int, y: int) -> int:
    return image.get_alpha(x, y)


@pending_deprecate()
def image_is_pixel_transparent(
    image: tcod.image.Image, x: int, y: int
) -> bool:
    return bool(lib.TCOD_image_is_pixel_transparent(image.image_c, x, y))


@pending_deprecate()
def image_load(filename: str) -> tcod.image.Image:
    """Load an image file into an Image instance and return it.

    Args:
        filename (AnyStr): Path to a .bmp or .png image file.
    """
    return tcod.image.Image._from_cdata(
        ffi.gc(lib.TCOD_image_load(_bytes(filename)), lib.TCOD_image_delete)
    )


@pending_deprecate()
def image_from_console(console: tcod.console.Console) -> tcod.image.Image:
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


@pending_deprecate()
def image_refresh_console(
    image: tcod.image.Image, console: tcod.console.Console
) -> None:
    image.refresh_console(console)


@pending_deprecate()
def image_get_size(image: tcod.image.Image) -> Tuple[int, int]:
    return image.width, image.height


@pending_deprecate()
def image_get_pixel(
    image: tcod.image.Image, x: int, y: int
) -> Tuple[int, int, int]:
    return image.get_pixel(x, y)


@pending_deprecate()
def image_get_mipmap_pixel(
    image: tcod.image.Image, x0: float, y0: float, x1: float, y1: float
) -> Tuple[int, int, int]:
    return image.get_mipmap_pixel(x0, y0, x1, y1)


@pending_deprecate()
def image_put_pixel(
    image: tcod.image.Image, x: int, y: int, col: Tuple[int, int, int]
) -> None:
    image.put_pixel(x, y, col)


@pending_deprecate()
def image_blit(
    image: tcod.image.Image,
    console: tcod.console.Console,
    x: float,
    y: float,
    bkgnd_flag: int,
    scalex: float,
    scaley: float,
    angle: float,
) -> None:
    image.blit(console, x, y, bkgnd_flag, scalex, scaley, angle)


@pending_deprecate()
def image_blit_rect(
    image: tcod.image.Image,
    console: tcod.console.Console,
    x: int,
    y: int,
    w: int,
    h: int,
    bkgnd_flag: int,
) -> None:
    image.blit_rect(console, x, y, w, h, bkgnd_flag)


@pending_deprecate()
def image_blit_2x(
    image: tcod.image.Image,
    console: tcod.console.Console,
    dx: int,
    dy: int,
    sx: int = 0,
    sy: int = 0,
    w: int = -1,
    h: int = -1,
) -> None:
    image.blit_2x(console, dx, dy, sx, sy, w, h)


@pending_deprecate()
def image_save(image: tcod.image.Image, filename: str) -> None:
    image.save_as(filename)


@deprecate("libtcod objects are deleted automatically.")
def image_delete(image: tcod.image.Image) -> None:
    """Does nothing. libtcod objects are managed by Python's garbage collector.

    This function exists for backwards compatibility with libtcodpy.
    """


@deprecate("Use tcod.line_iter instead.")
def line_init(xo: int, yo: int, xd: int, yd: int) -> None:
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


@deprecate("Use tcod.line_iter instead.")
def line_step() -> Union[Tuple[int, int], Tuple[None, None]]:
    """After calling line_init returns (x, y) points of the line.

    Once all points are exhausted this function will return (None, None)

    Returns:
        Union[Tuple[int, int], Tuple[None, None]]:
            The next (x, y) point of the line setup by line_init,
            or (None, None) if there are no more points.

    .. deprecated:: 2.0
       Use `line_iter` instead.
    """
    x = ffi.new("int *")
    y = ffi.new("int *")
    ret = lib.TCOD_line_step(x, y)
    if not ret:
        return x[0], y[0]
    return None, None


@deprecate("Use tcod.line_iter instead.")
def line(
    xo: int, yo: int, xd: int, yd: int, py_callback: Callable[[int, int], bool]
) -> bool:
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


def line_iter(xo: int, yo: int, xd: int, yd: int) -> Iterator[Tuple[int, int]]:
    """ returns an Iterable

    This Iterable does not include the origin point.

    Args:
        xo (int): X starting point.
        yo (int): Y starting point.
        xd (int): X destination point.
        yd (int): Y destination point.

    Returns:
        Iterable[Tuple[int,int]]: An Iterable of (x,y) points.
    """
    data = ffi.new("TCOD_bresenham_data_t *")
    lib.TCOD_line_init_mt(xo, yo, xd, yd, data)
    x = ffi.new("int *")
    y = ffi.new("int *")
    yield xo, yo
    while not lib.TCOD_line_step_mt(x, y, data):
        yield (x[0], y[0])


def line_where(
    x1: int, y1: int, x2: int, y2: int, inclusive: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
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
    x = ffi.cast("int*", array[0].ctypes.data)
    y = ffi.cast("int*", array[1].ctypes.data)
    lib.LineWhere(x1, y1, x2, y2, x, y)
    if not inclusive:
        array = array[:, 1:]
    return tuple(array)  # type: ignore


@deprecate("Call tcod.map.Map(width, height) instead.")
def map_new(w: int, h: int) -> tcod.map.Map:
    """Return a :any:`tcod.map.Map` with a width and height.

    .. deprecated:: 4.5
        Use the :any:`tcod.map` module for working with field-of-view,
        or :any:`tcod.path` for working with path-finding.
    """
    return tcod.map.Map(w, h)


@deprecate("Use Python's standard copy module instead.")
def map_copy(source: tcod.map.Map, dest: tcod.map.Map) -> None:
    """Copy map data from `source` to `dest`.

    .. deprecated:: 4.5
        Use Python's copy module, or see :any:`tcod.map.Map` and assign between
        array attributes manually.
    """
    if source.width != dest.width or source.height != dest.height:
        dest.__init__(  # type: ignore
            source.width, source.height, source._order
        )
    dest._Map__buffer[:] = source._Map__buffer[:]  # type: ignore


@deprecate("Set properties using the m.transparent and m.walkable arrays.")
def map_set_properties(
    m: tcod.map.Map, x: int, y: int, isTrans: bool, isWalk: bool
) -> None:
    """Set the properties of a single cell.

    .. note::
        This function is slow.
    .. deprecated:: 4.5
        Use :any:`tcod.map.Map.transparent` and :any:`tcod.map.Map.walkable`
        arrays to set these properties.
    """
    lib.TCOD_map_set_properties(m.map_c, x, y, isTrans, isWalk)


@deprecate("Clear maps using NumPy broadcast rules instead.")
def map_clear(
    m: tcod.map.Map, transparent: bool = False, walkable: bool = False
) -> None:
    """Change all map cells to a specific value.

    .. deprecated:: 4.5
        Use :any:`tcod.map.Map.transparent` and :any:`tcod.map.Map.walkable`
        arrays to set these properties.
    """
    m.transparent[:] = transparent
    m.walkable[:] = walkable


@deprecate("Call the map.compute_fov method instead.")
def map_compute_fov(
    m: tcod.map.Map,
    x: int,
    y: int,
    radius: int = 0,
    light_walls: bool = True,
    algo: int = FOV_RESTRICTIVE,
) -> None:
    """Compute the field-of-view for a map instance.

    .. deprecated:: 4.5
        Use :any:`tcod.map.Map.compute_fov` instead.
    """
    m.compute_fov(x, y, radius, light_walls, algo)


@deprecate("Use map.fov to check for this property.")
def map_is_in_fov(m: tcod.map.Map, x: int, y: int) -> bool:
    """Return True if the cell at x,y is lit by the last field-of-view
    algorithm.

    .. note::
        This function is slow.
    .. deprecated:: 4.5
        Use :any:`tcod.map.Map.fov` to check this property.
    """
    return bool(lib.TCOD_map_is_in_fov(m.map_c, x, y))


@deprecate("Use map.transparent to check for this property.")
def map_is_transparent(m: tcod.map.Map, x: int, y: int) -> bool:
    """
    .. note::
        This function is slow.
    .. deprecated:: 4.5
        Use :any:`tcod.map.Map.transparent` to check this property.
    """
    return bool(lib.TCOD_map_is_transparent(m.map_c, x, y))


@deprecate("Use map.walkable to check for this property.")
def map_is_walkable(m: tcod.map.Map, x: int, y: int) -> bool:
    """
    .. note::
        This function is slow.
    .. deprecated:: 4.5
        Use :any:`tcod.map.Map.walkable` to check this property.
    """
    return bool(lib.TCOD_map_is_walkable(m.map_c, x, y))


@deprecate("libtcod objects are deleted automatically.")
def map_delete(m: tcod.map.Map) -> None:
    """Does nothing. libtcod objects are managed by Python's garbage collector.

    This function exists for backwards compatibility with libtcodpy.
    """


@deprecate("Check the map.width attribute instead.")
def map_get_width(map: tcod.map.Map) -> int:
    """Return the width of a map.

    .. deprecated:: 4.5
        Check the :any:`tcod.map.Map.width` attribute instead.
    """
    return map.width


@deprecate("Check the map.height attribute instead.")
def map_get_height(map: tcod.map.Map) -> int:
    """Return the height of a map.

    .. deprecated:: 4.5
        Check the :any:`tcod.map.Map.height` attribute instead.
    """
    return map.height


@pending_deprecate()
def mouse_show_cursor(visible: bool) -> None:
    """Change the visibility of the mouse cursor."""
    lib.TCOD_mouse_show_cursor(visible)


@pending_deprecate()
def mouse_is_cursor_visible() -> bool:
    """Return True if the mouse cursor is visible."""
    return bool(lib.TCOD_mouse_is_cursor_visible())


@pending_deprecate()
def mouse_move(x: int, y: int) -> None:
    lib.TCOD_mouse_move(x, y)


@deprecate("Use tcod.event.get_mouse_state() instead.")
def mouse_get_status() -> Mouse:
    return Mouse(lib.TCOD_mouse_get_status())


@pending_deprecate()
def namegen_parse(
    filename: str, random: Optional[tcod.random.Random] = None
) -> None:
    lib.TCOD_namegen_parse(_bytes(filename), random or ffi.NULL)


@pending_deprecate()
def namegen_generate(name: str) -> str:
    return _unpack_char_p(lib.TCOD_namegen_generate(_bytes(name), False))


@pending_deprecate()
def namegen_generate_custom(name: str, rule: str) -> str:
    return _unpack_char_p(
        lib.TCOD_namegen_generate_custom(_bytes(name), _bytes(rule), False)
    )


@pending_deprecate()
def namegen_get_sets() -> List[str]:
    sets = lib.TCOD_namegen_get_sets()
    try:
        lst = []
        while not lib.TCOD_list_is_empty(sets):
            lst.append(
                _unpack_char_p(ffi.cast("char *", lib.TCOD_list_pop(sets)))
            )
    finally:
        lib.TCOD_list_delete(sets)
    return lst


@pending_deprecate()
def namegen_destroy() -> None:
    lib.TCOD_namegen_destroy()


@pending_deprecate()
def noise_new(
    dim: int,
    h: float = NOISE_DEFAULT_HURST,
    l: float = NOISE_DEFAULT_LACUNARITY,  # noqa: E741
    random: Optional[tcod.random.Random] = None,
) -> tcod.noise.Noise:
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


@pending_deprecate()
def noise_set_type(n: tcod.noise.Noise, typ: int) -> None:
    """Set a Noise objects default noise algorithm.

    Args:
        typ (int): Any NOISE_* constant.
    """
    n.algorithm = typ


@pending_deprecate()
def noise_get(
    n: tcod.noise.Noise, f: Sequence[float], typ: int = NOISE_DEFAULT
) -> float:
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
    return float(lib.TCOD_noise_get_ex(n.noise_c, ffi.new("float[4]", f), typ))


@pending_deprecate()
def noise_get_fbm(
    n: tcod.noise.Noise,
    f: Sequence[float],
    oc: float,
    typ: int = NOISE_DEFAULT,
) -> float:
    """Return the fractal Brownian motion sampled from the ``f`` coordinate.

    Args:
        n (Noise): A Noise instance.
        f (Sequence[float]): The point to sample the noise from.
        typ (int): The noise algorithm to use.
        octaves (float): The level of level.  Should be more than 1.

    Returns:
        float: The sampled noise value.
    """
    return float(
        lib.TCOD_noise_get_fbm_ex(n.noise_c, ffi.new("float[4]", f), oc, typ)
    )


@pending_deprecate()
def noise_get_turbulence(
    n: tcod.noise.Noise,
    f: Sequence[float],
    oc: float,
    typ: int = NOISE_DEFAULT,
) -> float:
    """Return the turbulence noise sampled from the ``f`` coordinate.

    Args:
        n (Noise): A Noise instance.
        f (Sequence[float]): The point to sample the noise from.
        typ (int): The noise algorithm to use.
        octaves (float): The level of level.  Should be more than 1.

    Returns:
        float: The sampled noise value.
    """
    return float(
        lib.TCOD_noise_get_turbulence_ex(
            n.noise_c, ffi.new("float[4]", f), oc, typ
        )
    )


@deprecate("libtcod objects are deleted automatically.")
def noise_delete(n: tcod.noise.Noise) -> None:
    # type (Any) -> None
    """Does nothing. libtcod objects are managed by Python's garbage collector.

    This function exists for backwards compatibility with libtcodpy.
    """


def _unpack_union(type_: int, union: Any) -> Any:
    """
        unpack items from parser new_property (value_converter)
    """
    if type_ == lib.TCOD_TYPE_BOOL:
        return bool(union.b)
    elif type_ == lib.TCOD_TYPE_CHAR:
        return union.c.decode("latin-1")
    elif type_ == lib.TCOD_TYPE_INT:
        return union.i
    elif type_ == lib.TCOD_TYPE_FLOAT:
        return union.f
    elif (
        type_ == lib.TCOD_TYPE_STRING
        or lib.TCOD_TYPE_VALUELIST15 >= type_ >= lib.TCOD_TYPE_VALUELIST00
    ):
        return _unpack_char_p(union.s)
    elif type_ == lib.TCOD_TYPE_COLOR:
        return Color._new_from_cdata(union.col)
    elif type_ == lib.TCOD_TYPE_DICE:
        return Dice(union.dice)
    elif type_ & lib.TCOD_TYPE_LIST:
        return _convert_TCODList(union.list, type_ & 0xFF)
    else:
        raise RuntimeError("Unknown libtcod type: %i" % type_)


def _convert_TCODList(clist: Any, type_: int) -> Any:
    return [
        _unpack_union(type_, lib.TDL_list_get_union(clist, i))
        for i in range(lib.TCOD_list_size(clist))
    ]


@deprecate("Parser functions have been deprecated.")
def parser_new() -> Any:
    return ffi.gc(lib.TCOD_parser_new(), lib.TCOD_parser_delete)


@deprecate("Parser functions have been deprecated.")
def parser_new_struct(parser: Any, name: str) -> Any:
    return lib.TCOD_parser_new_struct(parser, name)


# prevent multiple threads from messing with def_extern callbacks
_parser_callback_lock = threading.Lock()
# temporary global pointer to a listener instance
_parser_listener = None  # type: Any


@ffi.def_extern()  # type: ignore
def _pycall_parser_new_struct(struct: Any, name: str) -> Any:
    return _parser_listener.new_struct(struct, _unpack_char_p(name))


@ffi.def_extern()  # type: ignore
def _pycall_parser_new_flag(name: str) -> Any:
    return _parser_listener.new_flag(_unpack_char_p(name))


@ffi.def_extern()  # type: ignore
def _pycall_parser_new_property(propname: Any, type: Any, value: Any) -> Any:
    return _parser_listener.new_property(
        _unpack_char_p(propname), type, _unpack_union(type, value)
    )


@ffi.def_extern()  # type: ignore
def _pycall_parser_end_struct(struct: Any, name: Any) -> Any:
    return _parser_listener.end_struct(struct, _unpack_char_p(name))


@ffi.def_extern()  # type: ignore
def _pycall_parser_error(msg: Any) -> None:
    _parser_listener.error(_unpack_char_p(msg))


@deprecate("Parser functions have been deprecated.")
def parser_run(parser: Any, filename: str, listener: Any = None) -> None:
    global _parser_listener
    if not listener:
        lib.TCOD_parser_run(parser, _bytes(filename), ffi.NULL)
        return

    propagate_manager = _PropagateException()

    clistener = ffi.new(
        "TCOD_parser_listener_t *",
        {
            "new_struct": lib._pycall_parser_new_struct,
            "new_flag": lib._pycall_parser_new_flag,
            "new_property": lib._pycall_parser_new_property,
            "end_struct": lib._pycall_parser_end_struct,
            "error": lib._pycall_parser_error,
        },
    )

    with _parser_callback_lock:
        _parser_listener = listener
        with propagate_manager:
            lib.TCOD_parser_run(parser, _bytes(filename), clistener)


@deprecate("libtcod objects are deleted automatically.")
def parser_delete(parser: Any) -> None:
    # type (Any) -> None
    """Does nothing. libtcod objects are managed by Python's garbage collector.

    This function exists for backwards compatibility with libtcodpy.
    """


@deprecate("Parser functions have been deprecated.")
def parser_get_bool_property(parser: Any, name: str) -> bool:
    return bool(lib.TCOD_parser_get_bool_property(parser, _bytes(name)))


@deprecate("Parser functions have been deprecated.")
def parser_get_int_property(parser: Any, name: str) -> int:
    return int(lib.TCOD_parser_get_int_property(parser, _bytes(name)))


@deprecate("Parser functions have been deprecated.")
def parser_get_char_property(parser: Any, name: str) -> str:
    return chr(lib.TCOD_parser_get_char_property(parser, _bytes(name)))


@deprecate("Parser functions have been deprecated.")
def parser_get_float_property(parser: Any, name: str) -> float:
    return float(lib.TCOD_parser_get_float_property(parser, _bytes(name)))


@deprecate("Parser functions have been deprecated.")
def parser_get_string_property(parser: Any, name: str) -> str:
    return _unpack_char_p(
        lib.TCOD_parser_get_string_property(parser, _bytes(name))
    )


@deprecate("Parser functions have been deprecated.")
def parser_get_color_property(parser: Any, name: str) -> Color:
    return Color._new_from_cdata(
        lib.TCOD_parser_get_color_property(parser, _bytes(name))
    )


@deprecate("Parser functions have been deprecated.")
def parser_get_dice_property(parser: Any, name: str) -> Dice:
    d = ffi.new("TCOD_dice_t *")
    lib.TCOD_parser_get_dice_property_py(parser, _bytes(name), d)
    return Dice(d)


@deprecate("Parser functions have been deprecated.")
def parser_get_list_property(parser: Any, name: str, type: Any) -> Any:
    clist = lib.TCOD_parser_get_list_property(parser, _bytes(name), type)
    return _convert_TCODList(clist, type)


RNG_MT = 0
RNG_CMWC = 1

DISTRIBUTION_LINEAR = 0
DISTRIBUTION_GAUSSIAN = 1
DISTRIBUTION_GAUSSIAN_RANGE = 2
DISTRIBUTION_GAUSSIAN_INVERSE = 3
DISTRIBUTION_GAUSSIAN_RANGE_INVERSE = 4


@pending_deprecate()
def random_get_instance() -> tcod.random.Random:
    """Return the default Random instance.

    Returns:
        Random: A Random instance using the default random number generator.
    """
    return tcod.random.Random._new_from_cdata(
        ffi.cast("mersenne_data_t*", lib.TCOD_random_get_instance())
    )


@pending_deprecate()
def random_new(algo: int = RNG_CMWC) -> tcod.random.Random:
    """Return a new Random instance.  Using ``algo``.

    Args:
        algo (int): The random number algorithm to use.

    Returns:
        Random: A new Random instance using the given algorithm.
    """
    return tcod.random.Random(algo)


@pending_deprecate()
def random_new_from_seed(
    seed: Hashable, algo: int = RNG_CMWC
) -> tcod.random.Random:
    """Return a new Random instance.  Using the given ``seed`` and ``algo``.

    Args:
        seed (Hashable): The RNG seed.  Should be a 32-bit integer, but any
                         hashable object is accepted.
        algo (int): The random number algorithm to use.

    Returns:
        Random: A new Random instance using the given algorithm.
    """
    return tcod.random.Random(algo, seed)


@pending_deprecate()
def random_set_distribution(
    rnd: Optional[tcod.random.Random], dist: int
) -> None:
    """Change the distribution mode of a random number generator.

    Args:
        rnd (Optional[Random]): A Random instance, or None to use the default.
        dist (int): The distribution mode to use.  Should be DISTRIBUTION_*.
    """
    lib.TCOD_random_set_distribution(rnd.random_c if rnd else ffi.NULL, dist)


@pending_deprecate()
def random_get_int(rnd: Optional[tcod.random.Random], mi: int, ma: int) -> int:
    """Return a random integer in the range: ``mi`` <= n <= ``ma``.

    The result is affected by calls to :any:`random_set_distribution`.

    Args:
        rnd (Optional[Random]): A Random instance, or None to use the default.
        low (int): The lower bound of the random range, inclusive.
        high (int): The upper bound of the random range, inclusive.

    Returns:
        int: A random integer in the range ``mi`` <= n <= ``ma``.
    """
    return int(
        lib.TCOD_random_get_int(rnd.random_c if rnd else ffi.NULL, mi, ma)
    )


@pending_deprecate()
def random_get_float(
    rnd: Optional[tcod.random.Random], mi: float, ma: float
) -> float:
    """Return a random float in the range: ``mi`` <= n <= ``ma``.

    The result is affected by calls to :any:`random_set_distribution`.

    Args:
        rnd (Optional[Random]): A Random instance, or None to use the default.
        low (float): The lower bound of the random range, inclusive.
        high (float): The upper bound of the random range, inclusive.

    Returns:
        float: A random double precision float
               in the range ``mi`` <= n <= ``ma``.
    """
    return float(
        lib.TCOD_random_get_double(rnd.random_c if rnd else ffi.NULL, mi, ma)
    )


@deprecate("Call tcod.random_get_float instead.")
def random_get_double(
    rnd: Optional[tcod.random.Random], mi: float, ma: float
) -> float:
    """Return a random float in the range: ``mi`` <= n <= ``ma``.

    .. deprecated:: 2.0
        Use :any:`random_get_float` instead.
        Both funtions return a double precision float.
    """
    return float(
        lib.TCOD_random_get_double(rnd.random_c if rnd else ffi.NULL, mi, ma)
    )


@pending_deprecate()
def random_get_int_mean(
    rnd: Optional[tcod.random.Random], mi: int, ma: int, mean: int
) -> int:
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
    return int(
        lib.TCOD_random_get_int_mean(
            rnd.random_c if rnd else ffi.NULL, mi, ma, mean
        )
    )


@pending_deprecate()
def random_get_float_mean(
    rnd: Optional[tcod.random.Random], mi: float, ma: float, mean: float
) -> float:
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
    return float(
        lib.TCOD_random_get_double_mean(
            rnd.random_c if rnd else ffi.NULL, mi, ma, mean
        )
    )


@deprecate("Call tcod.random_get_float_mean instead.")
def random_get_double_mean(
    rnd: Optional[tcod.random.Random], mi: float, ma: float, mean: float
) -> float:
    """Return a random weighted float in the range: ``mi`` <= n <= ``ma``.

    .. deprecated:: 2.0
        Use :any:`random_get_float_mean` instead.
        Both funtions return a double precision float.
    """
    return float(
        lib.TCOD_random_get_double_mean(
            rnd.random_c if rnd else ffi.NULL, mi, ma, mean
        )
    )


@deprecate("Use the standard library 'copy' module instead.")
def random_save(rnd: Optional[tcod.random.Random]) -> tcod.random.Random:
    """Return a copy of a random number generator.

    .. deprecated:: 8.4
        You can use the standard library copy and pickle modules to save a
        random state.
    """
    return tcod.random.Random._new_from_cdata(
        ffi.gc(
            ffi.cast(
                "mersenne_data_t*",
                lib.TCOD_random_save(rnd.random_c if rnd else ffi.NULL),
            ),
            lib.TCOD_random_delete,
        )
    )


@deprecate("This function is deprecated.")
def random_restore(
    rnd: Optional[tcod.random.Random], backup: tcod.random.Random
) -> None:
    """Restore a random number generator from a backed up copy.

    Args:
        rnd (Optional[Random]): A Random instance, or None to use the default.
        backup (Random): The Random instance which was used as a backup.

    .. deprecated:: 8.4
        You can use the standard library copy and pickle modules to save a
        random state.
    """
    lib.TCOD_random_restore(rnd.random_c if rnd else ffi.NULL, backup.random_c)


@deprecate("libtcod objects are deleted automatically.")
def random_delete(rnd: tcod.random.Random) -> None:
    """Does nothing. libtcod objects are managed by Python's garbage collector.

    This function exists for backwards compatibility with libtcodpy.
    """


@deprecate("This function is deprecated.")
def struct_add_flag(struct, name):  # type: ignore
    lib.TCOD_struct_add_flag(struct, name)


@deprecate("This function is deprecated.")
def struct_add_property(struct, name, typ, mandatory):  # type: ignore
    lib.TCOD_struct_add_property(struct, name, typ, mandatory)


@deprecate("This function is deprecated.")
def struct_add_value_list(struct, name, value_list, mandatory):  # type: ignore
    c_strings = [
        ffi.new("char[]", value.encode("utf-8")) for value in value_list
    ]
    c_value_list = ffi.new("char*[]", c_strings)
    lib.TCOD_struct_add_value_list(struct, name, c_value_list, mandatory)


@deprecate("This function is deprecated.")
def struct_add_list_property(struct, name, typ, mandatory):  # type: ignore
    lib.TCOD_struct_add_list_property(struct, name, typ, mandatory)


@deprecate("This function is deprecated.")
def struct_add_structure(struct, sub_struct):  # type: ignore
    lib.TCOD_struct_add_structure(struct, sub_struct)


@deprecate("This function is deprecated.")
def struct_get_name(struct):  # type: ignore
    return _unpack_char_p(lib.TCOD_struct_get_name(struct))


@deprecate("This function is deprecated.")
def struct_is_mandatory(struct, name):  # type: ignore
    return lib.TCOD_struct_is_mandatory(struct, name)


@deprecate("This function is deprecated.")
def struct_get_type(struct, name):  # type: ignore
    return lib.TCOD_struct_get_type(struct, name)


# high precision time functions
def sys_set_fps(fps: int) -> None:
    """Set the maximum frame rate.

    You can disable the frame limit again by setting fps to 0.

    Args:
        fps (int): A frame rate limit (i.e. 60)
    """
    lib.TCOD_sys_set_fps(fps)


def sys_get_fps() -> int:
    """Return the current frames per second.

    This the actual frame rate, not the frame limit set by
    :any:`tcod.sys_set_fps`.

    This number is updated every second.

    Returns:
        int: The currently measured frame rate.
    """
    return int(lib.TCOD_sys_get_fps())


def sys_get_last_frame_length() -> float:
    """Return the delta time of the last rendered frame in seconds.

    Returns:
        float: The delta time of the last rendered frame.
    """
    return float(lib.TCOD_sys_get_last_frame_length())


@deprecate("Use Python's standard 'time' module instead of this function.")
def sys_sleep_milli(val: int) -> None:
    """Sleep for 'val' milliseconds.

    Args:
        val (int): Time to sleep for in milliseconds.

    .. deprecated:: 2.0
       Use :any:`time.sleep` instead.
    """
    lib.TCOD_sys_sleep_milli(val)


@deprecate("Use Python's standard 'time' module instead of this function.")
def sys_elapsed_milli() -> int:
    """Get number of milliseconds since the start of the program.

    Returns:
        int: Time since the progeam has started in milliseconds.

    .. deprecated:: 2.0
       Use :any:`time.clock` instead.
    """
    return int(lib.TCOD_sys_elapsed_milli())


@deprecate("Use Python's standard 'time' module instead of this function.")
def sys_elapsed_seconds() -> float:
    """Get number of seconds since the start of the program.

    Returns:
        float: Time since the progeam has started in seconds.

    .. deprecated:: 2.0
       Use :any:`time.clock` instead.
    """
    return float(lib.TCOD_sys_elapsed_seconds())


def sys_set_renderer(renderer: int) -> None:
    """Change the current rendering mode to renderer.

    .. deprecated:: 2.0
       RENDERER_GLSL and RENDERER_OPENGL are not currently available.
    """
    _check(lib.TCOD_sys_set_renderer(renderer))
    if tcod.console._root_console is not None:
        tcod.console.Console._get_root()


def sys_get_renderer() -> int:
    """Return the current rendering mode.

    """
    return int(lib.TCOD_sys_get_renderer())


# easy screenshots
def sys_save_screenshot(name: Optional[str] = None) -> None:
    """Save a screenshot to a file.

    By default this will automatically save screenshots in the working
    directory.

    The automatic names are formatted as screenshotNNN.png.  For example:
    screenshot000.png, screenshot001.png, etc.  Whichever is available first.

    Args:
        file Optional[AnyStr]: File path to save screenshot.
    """
    lib.TCOD_sys_save_screenshot(
        _bytes(name) if name is not None else ffi.NULL
    )


# custom fullscreen resolution
@pending_deprecate()
def sys_force_fullscreen_resolution(width: int, height: int) -> None:
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


def sys_get_current_resolution() -> Tuple[int, int]:
    """Return the current resolution as (width, height)

    Returns:
        Tuple[int,int]: The current resolution.
    """
    w = ffi.new("int *")
    h = ffi.new("int *")
    lib.TCOD_sys_get_current_resolution(w, h)
    return w[0], h[0]


def sys_get_char_size() -> Tuple[int, int]:
    """Return the current fonts character size as (width, height)

    Returns:
        Tuple[int,int]: The current font glyph size in (width, height)
    """
    w = ffi.new("int *")
    h = ffi.new("int *")
    lib.TCOD_sys_get_char_size(w, h)
    return w[0], h[0]


# update font bitmap
@pending_deprecate()
def sys_update_char(
    asciiCode: int,
    fontx: int,
    fonty: int,
    img: tcod.image.Image,
    x: int,
    y: int,
) -> None:
    """Dynamically update the current font with img.

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


@pending_deprecate()
def sys_register_SDL_renderer(callback: Callable[[Any], None]) -> None:
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

        @ffi.def_extern(onerror=propagate)  # type: ignore
        def _pycall_sdl_hook(sdl_surface: Any) -> None:
            callback(sdl_surface)

        lib.TCOD_sys_register_SDL_renderer(lib._pycall_sdl_hook)


@deprecate("Use tcod.event.get to check for events.")
def sys_check_for_event(
    mask: int, k: Optional[Key], m: Optional[Mouse]
) -> int:
    """Check for and return an event.

    Args:
        mask (int): :any:`Event types` to wait for.
        k (Optional[Key]): A tcod.Key instance which might be updated with
                           an event.  Can be None.
        m (Optional[Mouse]): A tcod.Mouse instance which might be updated
                             with an event.  Can be None.

    .. deprecated:: 9.3
        Use the :any:`tcod.event.get` function to check for events.
    """
    return int(
        lib.TCOD_sys_check_for_event(
            mask, k.key_p if k else ffi.NULL, m.mouse_p if m else ffi.NULL
        )
    )


@deprecate("Use tcod.event.wait to wait for events.")
def sys_wait_for_event(
    mask: int, k: Optional[Key], m: Optional[Mouse], flush: bool
) -> int:
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

    .. deprecated:: 9.3
        Use the :any:`tcod.event.wait` function to wait for events.
    """
    return int(
        lib.TCOD_sys_wait_for_event(
            mask,
            k.key_p if k else ffi.NULL,
            m.mouse_p if m else ffi.NULL,
            flush,
        )
    )


@deprecate("This function does not provide reliable access to the clipboard.")
def sys_clipboard_set(text: str) -> bool:
    """Sets the clipboard to `text`.

    .. deprecated:: 6.0
       This function does not provide reliable access to the clipboard.
    """
    return bool(lib.TCOD_sys_clipboard_set(text.encode("utf-8")))


@deprecate("This function does not provide reliable access to the clipboard.")
def sys_clipboard_get() -> str:
    """Return the current value of the clipboard.

    .. deprecated:: 6.0
       This function does not provide reliable access to the clipboard.
    """
    return str(ffi.string(lib.TCOD_sys_clipboard_get()).decode("utf-8"))


@atexit.register
def _atexit_verify() -> None:
    """Warns if the libtcod root console is implicitly deleted."""
    if lib.TCOD_ctx.root:
        warnings.warn(
            "The libtcod root console was implicitly deleted.\n"
            "Make sure the 'with' statement is used with the root console to"
            " ensure that it closes properly.",
            ResourceWarning,
            stacklevel=2,
        )
        lib.TCOD_console_delete(ffi.NULL)
