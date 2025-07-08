"""Libtcod tile-based Consoles and printing functions.

Libtcod consoles are a strictly tile-based representation of colored glyphs/tiles.
To render a console you need a tileset and a window to render to.
See :ref:`getting-started` for info on how to set those up.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, overload

import numpy as np
from typing_extensions import Self, deprecated

import tcod._internal
import tcod.constants
import tcod.image
from tcod._internal import _check, _path_encode
from tcod.cffi import ffi, lib

if TYPE_CHECKING:
    from os import PathLike

    from numpy.typing import ArrayLike, NDArray


def _fmt(string: str) -> bytes:
    """Return a string that escapes 'C printf' side effects."""
    return string.encode("utf-8").replace(b"%", b"%%")


_root_console = None

rgba_graphic: np.dtype[Any] = np.dtype([("ch", np.intc), ("fg", "4B"), ("bg", "4B")])
"""A NumPy :any:`dtype` compatible with :any:`Console.rgba`.

This dtype is: ``np.dtype([("ch", np.intc), ("fg", "4B"), ("bg", "4B")])``

.. versionadded:: 12.3
"""

rgb_graphic: np.dtype[Any] = np.dtype([("ch", np.intc), ("fg", "3B"), ("bg", "3B")])
"""A NumPy :any:`dtype` compatible with :any:`Console.rgb`.

This dtype is: ``np.dtype([("ch", np.intc), ("fg", "3B"), ("bg", "3B")])``

.. versionadded:: 12.3
"""


class Console:
    """A console object containing a grid of characters with foreground/background colors.

    `width` and `height` are the size of the console (in tiles.)

    `order` determines how the axes of NumPy array attributes are arranged.
    With the default setting of `"C"` you use [y, x] to index a consoles
    arrays such as :any:`Console.rgb`.
    `order="F"` will swap the first two axes which allows for more intuitive
    `[x, y]` indexing.  To be consistent you may have to do this for every
    NumPy array to create.

    With `buffer` the console can be initialized from another array. The
    `buffer` should be compatible with the `width`, `height`, and `order`
    given; and should also have a dtype compatible with :any:`Console.DTYPE`.

    .. versionchanged:: 4.3
        Added `order` parameter.

    .. versionchanged:: 8.5
        Added `buffer`, `copy`, and default parameters.
        Arrays are initialized as if the :any:`clear` method was called.

    .. versionchanged:: 10.0
        `DTYPE` changed, `buffer` now requires colors with an alpha channel.

    Attributes:
        console_c: A python-cffi "TCOD_Console*" object.
        DTYPE:
            A class attribute which provides a dtype compatible with this
            class.

            ``[("ch", np.intc), ("fg", "(4,)u1"), ("bg", "(4,)u1")]``

            Example::

                >>> buffer = np.zeros(
                ...     shape=(20, 3),
                ...     dtype=tcod.console.Console.DTYPE,
                ...     order="F",
                ... )
                >>> buffer["ch"] = ord('_')
                >>> buffer["ch"][:, 1] = ord('x')
                >>> c = tcod.console.Console(20, 3, order="F", buffer=buffer)
                >>> print(c)
                <____________________
                 xxxxxxxxxxxxxxxxxxxx
                 ____________________>

            .. versionadded:: 8.5

            .. versionchanged:: 10.0
                Added an alpha channel to the color types.
    """

    DTYPE = rgba_graphic

    # A structured array type with the added "fg_rgb" and "bg_rgb" fields.
    _DTYPE_RGB: np.dtype[Any] = np.dtype(
        {
            "names": ["ch", "fg", "bg"],
            "formats": [np.int32, "3u1", "3u1"],
            "offsets": [0, 4, 8],
            "itemsize": 12,
        }
    )

    def __init__(
        self,
        width: int,
        height: int,
        order: Literal["C", "F"] = "C",
        buffer: NDArray[Any] | None = None,
    ) -> None:
        """Initialize the console."""
        self._key_color: tuple[int, int, int] | None = None
        self._order: Literal["C", "F"] = tcod._internal.verify_order(order)
        if buffer is not None:
            if self._order == "F":
                buffer = buffer.transpose()
            self._tiles: NDArray[Any] = np.ascontiguousarray(buffer, self.DTYPE)
        else:
            self._tiles = np.ndarray((height, width), dtype=self.DTYPE)

        # libtcod uses the root console for defaults.
        default_bg_blend = 0
        default_alignment = 0
        if lib.TCOD_ctx.root != ffi.NULL:
            default_bg_blend = lib.TCOD_ctx.root.bkgnd_flag
            default_alignment = lib.TCOD_ctx.root.alignment

        self._console_data = self.console_c = ffi.new(
            "struct TCOD_Console*",
            {
                "w": width,
                "h": height,
                "elements": width * height,
                "tiles": ffi.from_buffer("struct TCOD_ConsoleTile*", self._tiles),
                "bkgnd_flag": default_bg_blend,
                "alignment": default_alignment,
                "fore": (255, 255, 255),
                "back": (0, 0, 0),
            },
        )

        if buffer is None:
            self.clear()

    @classmethod
    def _from_cdata(cls, cdata: Any, order: Literal["C", "F"] = "C") -> Console:  # noqa: ANN401
        """Return a Console instance which wraps this `TCOD_Console*` object."""
        if isinstance(cdata, cls):
            return cdata
        self: Console = object.__new__(cls)
        self.console_c = cdata
        self._init_setup_console_data(order)
        return self

    @classmethod
    def _get_root(cls, order: Literal["C", "F"] | None = None) -> Console:
        """Return a root console singleton with valid buffers.

        This function will also update an already active root console.
        """
        global _root_console  # noqa: PLW0603
        if _root_console is None:
            _root_console = object.__new__(cls)
        self: Console = _root_console
        if order is not None:
            self._order = order
        self.console_c = ffi.NULL
        self._init_setup_console_data(self._order)
        return self

    def _init_setup_console_data(self, order: Literal["C", "F"] = "C") -> None:
        """Setup numpy arrays over libtcod data buffers."""
        global _root_console  # noqa: PLW0603
        self._key_color = None
        if self.console_c == ffi.NULL:
            _root_console = self
            self._console_data = lib.TCOD_ctx.root
        else:
            self._console_data = ffi.cast("struct TCOD_Console*", self.console_c)

        self._tiles = np.frombuffer(
            ffi.buffer(self._console_data.tiles[0 : self.width * self.height]),
            dtype=self.DTYPE,
        ).reshape((self.height, self.width))

        self._order = tcod._internal.verify_order(order)

    @property
    def width(self) -> int:
        """The width of this Console."""
        return int(lib.TCOD_console_get_width(self.console_c))

    @property
    def height(self) -> int:
        """The height of this Console."""
        return int(lib.TCOD_console_get_height(self.console_c))

    @property
    def bg(self) -> NDArray[np.uint8]:
        """A uint8 array with the shape (height, width, 3).

        You can change the consoles background colors by using this array.

        Index this array with ``console.bg[i, j, channel]  # order='C'`` or
        ``console.bg[x, y, channel]  # order='F'``.

        """
        bg: np.ndarray[Any, np.dtype[np.uint8]] = self._tiles["bg"][..., :3]
        if self._order == "F":
            bg = bg.transpose(1, 0, 2)
        return bg

    @property
    def fg(self) -> NDArray[np.uint8]:
        """A uint8 array with the shape (height, width, 3).

        You can change the consoles foreground colors by using this array.

        Index this array with ``console.fg[i, j, channel]  # order='C'`` or
        ``console.fg[x, y, channel]  # order='F'``.
        """
        fg: np.ndarray[Any, np.dtype[np.uint8]] = self._tiles["fg"][..., :3]
        if self._order == "F":
            fg = fg.transpose(1, 0, 2)
        return fg

    @property
    def ch(self) -> NDArray[np.intc]:
        """An integer array with the shape (height, width).

        You can change the consoles character codes by using this array.

        Index this array with ``console.ch[i, j]  # order='C'`` or
        ``console.ch[x, y]  # order='F'``.
        """
        return self._tiles["ch"].T if self._order == "F" else self._tiles["ch"]

    @property
    @deprecated("This attribute has been renamed to `rgba`.", category=FutureWarning)
    def tiles(self) -> NDArray[Any]:
        """An array of this consoles raw tile data.

        This acts as a combination of the `ch`, `fg`, and `bg` attributes.
        Colors include an alpha channel but how alpha works is currently
        undefined.

        .. versionadded:: 10.0

        .. deprecated:: 12.3
            Use :any:`Console.rgba` instead.
        """
        return self.rgba

    @property
    @deprecated("This attribute has been renamed to `rgba`.", category=FutureWarning)
    def buffer(self) -> NDArray[Any]:
        """An array of this consoles raw tile data.

        .. versionadded:: 11.4

        .. deprecated:: 11.8
            Use :any:`Console.rgba` instead.
        """
        return self.rgba

    @property
    @deprecated("This attribute has been renamed to `rgb`.", category=FutureWarning)
    def tiles_rgb(self) -> NDArray[Any]:
        """An array of this consoles data without the alpha channel.

        .. versionadded:: 11.8

        .. deprecated:: 12.3
            Use :any:`Console.rgb` instead.
        """
        return self.rgb

    @property
    @deprecated("This attribute has been renamed to `rgb`.", category=FutureWarning)
    def tiles2(self) -> NDArray[Any]:
        """This name is deprecated in favour of :any:`rgb`.

        .. versionadded:: 11.3

        .. deprecated:: 11.8
            Use :any:`Console.rgb` instead.
        """
        return self.rgb

    @property
    def rgba(self) -> NDArray[Any]:
        """An array of this consoles raw tile data.

        The axes of this array is affected by the `order` parameter given to
        initialize the console.

        Example:
            >>> con = tcod.console.Console(10, 2)
            >>> WHITE, BLACK = (255, 255, 255), (0, 0, 0)
            >>> con.rgba[0, 0] = (
            ...     ord("X"),
            ...     (*WHITE, 255),
            ...     (*BLACK, 255),
            ... )
            >>> print(f"{con.rgba[0, 0]=}")
            con.rgba[0, 0]=...(88, [255, 255, 255, 255], [  0,   0,   0, 255])...

        .. versionadded:: 12.3
        """
        return self._tiles.T if self._order == "F" else self._tiles

    @property
    def rgb(self) -> NDArray[Any]:
        """An array of this consoles data without the alpha channel.

        The axes of this array is affected by the `order` parameter given to
        initialize the console.

        The :any:`rgb_graphic` dtype can be used to make arrays compatible
        with this attribute that are independent of a :any:`Console`.

        Example:
            >>> tile_graphics = np.array(  # Tile graphics lookup table
            ...     [  # (Unicode, foreground, background)
            ...         (ord("."), (255, 255, 255), (0, 0, 0)),  # Tile 0
            ...         (ord("#"), (255, 255, 255), (0, 0, 0)),  # Tile 1
            ...         (ord("^"), (255, 255, 255), (0, 0, 0)),  # Tile 2
            ...         (ord("~"), (255, 255, 255), (0, 0, 0)),  # Tile 3
            ...     ],
            ...     dtype=tcod.console.rgb_graphic,
            ... )
            >>> console = tcod.console.Console(6, 5)
            >>> console.rgb[:] = tile_graphics[  # Convert 2D array of indexes to tile graphics
            ...     [
            ...         [1, 1, 1, 1, 1, 1],
            ...         [1, 0, 2, 0, 0, 1],
            ...         [1, 0, 0, 3, 3, 1],
            ...         [1, 0, 0, 3, 3, 1],
            ...         [1, 1, 1, 1, 1, 1],
            ...     ],
            ... ]
            >>> print(console)
            <######
             #.^..#
             #..~~#
             #..~~#
             ######>

        Example:
            >>> con = tcod.console.Console(10, 2)
            >>> BLUE, YELLOW, BLACK = (0, 0, 255), (255, 255, 0), (0, 0, 0)
            >>> con.rgb[0, 0] = ord("@"), YELLOW, BLACK
            >>> print(f"{con.rgb[0, 0]=}")
            con.rgb[0, 0]=...(64, [255, 255,   0], [0, 0, 0])...
            >>> con.rgb["bg"] = BLUE
            >>> print(f"{con.rgb[0, 0]=}")
            con.rgb[0, 0]=...(64, [255, 255,   0], [  0,   0, 255])...

        .. versionadded:: 12.3
        """
        return self.rgba.view(self._DTYPE_RGB)

    _DEPRECATE_CONSOLE_DEFAULTS_MSG = """Console defaults have been deprecated.
Consider one of the following:

    # Set parameters once then pass them as kwargs
    DEFAULT_COLOR = {"bg": (0, 0, 127), "fg": (127, 127, 255)}
    console.print(x, y, string, **DEFAULT_COLOR)

    # Clear the console to a color and then skip setting colors on printing/drawing
    console.clear(fg=(127, 127, 255), bg=(0, 0, 127))
    console.print(x, y, string, fg=None)
"""

    @property  # Getters used internally, so only deprecate the setters.
    def default_bg(self) -> tuple[int, int, int]:
        """Tuple[int, int, int]: The default background color.

        .. deprecated:: 8.5
            These should not be used. Prefer passing defaults as kwargs.

            .. code-block::

                DEFAULT_COLOR = {"bg": (0, 0, 127), "fg": (127, 127, 255)}
                console.print(x, y, string, **DEFAULT_COLOR)
        """
        color = self._console_data.back
        return color.r, color.g, color.b

    @default_bg.setter
    @deprecated(_DEPRECATE_CONSOLE_DEFAULTS_MSG, category=FutureWarning)
    def default_bg(self, color: tuple[int, int, int]) -> None:
        self._console_data.back = color

    @property
    def default_fg(self) -> tuple[int, int, int]:
        """Tuple[int, int, int]: The default foreground color.

        .. deprecated:: 8.5
            These should not be used. Prefer passing defaults as kwargs.
        """
        color = self._console_data.fore
        return color.r, color.g, color.b

    @default_fg.setter
    @deprecated(_DEPRECATE_CONSOLE_DEFAULTS_MSG, category=FutureWarning)
    def default_fg(self, color: tuple[int, int, int]) -> None:
        self._console_data.fore = color

    @property
    def default_bg_blend(self) -> int:
        """int: The default blending mode.

        .. deprecated:: 8.5
            These should not be used. Prefer passing defaults as kwargs.
        """
        return int(self._console_data.bkgnd_flag)

    @default_bg_blend.setter
    @deprecated(_DEPRECATE_CONSOLE_DEFAULTS_MSG, category=FutureWarning)
    def default_bg_blend(self, value: int) -> None:
        self._console_data.bkgnd_flag = value

    @property
    def default_alignment(self) -> int:
        """int: The default text alignment.

        .. deprecated:: 8.5
            These should not be used. Prefer passing defaults as kwargs.
        """
        return int(self._console_data.alignment)

    @default_alignment.setter
    @deprecated(_DEPRECATE_CONSOLE_DEFAULTS_MSG, category=FutureWarning)
    def default_alignment(self, value: int) -> None:
        self._console_data.alignment = value

    def __clear_warning(self, name: str, value: tuple[int, int, int]) -> None:
        """Raise a warning for bad default values during calls to clear."""
        warnings.warn(
            f"Clearing with the console default values is deprecated.\nAdd {name}={value!r} to this call.",
            FutureWarning,
            stacklevel=3,
        )

    def clear(
        self,
        ch: int = 0x20,
        fg: tuple[int, int, int] = ...,  # type: ignore[assignment]
        bg: tuple[int, int, int] = ...,  # type: ignore[assignment]
    ) -> None:
        """Reset all values in this console to a single value.

        `ch` is the character to clear the console with.  Defaults to the space
        character.

        `fg` and `bg` are the colors to clear the console with.  Defaults to
        white-on-black if the console defaults are untouched.

        .. note::
            If `fg`/`bg` are not set, they will default to
            :any:`default_fg`/:any:`default_bg`.
            However, default values other than white-on-back are deprecated.

        .. versionchanged:: 8.5
            Added the `ch`, `fg`, and `bg` parameters.
            Non-white-on-black default values are deprecated.
        """
        if fg is ...:  # type: ignore[comparison-overlap]
            fg = self.default_fg
            if fg != (255, 255, 255):
                self.__clear_warning("fg", fg)
        if bg is ...:  # type: ignore[comparison-overlap]
            bg = self.default_bg
            if bg != (0, 0, 0):
                self.__clear_warning("bg", bg)
        self._tiles[...] = ch, (*fg, 255), (*bg, 255)

    def put_char(
        self,
        x: int,
        y: int,
        ch: int,
        bg_blend: int = tcod.constants.BKGND_DEFAULT,
    ) -> None:
        """Draw the character c at x,y using the default colors and a blend mode.

        Args:
            x (int): The x coordinate from the left.
            y (int): The y coordinate from the top.
            ch (int): Character code to draw.  Must be in integer form.
            bg_blend (int): Blending mode to use, defaults to BKGND_DEFAULT.
        """
        lib.TCOD_console_put_char(self.console_c, x, y, ch, bg_blend)

    __ALIGNMENT_LOOKUP = (
        "tcod.LEFT",
        "tcod.RIGHT",
        "tcod.CENTER",
    )

    __BG_BLEND_LOOKUP = (
        "tcod.BKGND_NONE",
        "tcod.BKGND_SET",
        "tcod.BKGND_MULTIPLY",
        "tcod.BKGND_LIGHTEN",
        "tcod.BKGND_DARKEN",
        "tcod.BKGND_SCREEN",
        "tcod.BKGND_COLOR_DODGE",
        "tcod.BKGND_COLOR_BURN",
        "tcod.BKGND_ADD",
        "tcod.BKGND_ADDA",
        "tcod.BKGND_BURN",
        "tcod.BKGND_OVERLAY",
        "tcod.BKGND_ALPH",
        "tcod.BKGND_DEFAULT",
    )

    def __deprecate_defaults(  # noqa: C901, PLR0912
        self,
        new_func: str,
        bg_blend: Any,  # noqa: ANN401
        alignment: Any = ...,  # noqa: ANN401
        clear: Any = ...,  # noqa: ANN401
    ) -> None:
        """Return the parameters needed to recreate the current default state."""
        if not __debug__:
            return

        fg: tuple[int, int, int] | None = self.default_fg
        bg: tuple[int, int, int] | None = self.default_bg
        if bg_blend == tcod.constants.BKGND_NONE:
            bg = None
        bg_blend = self.default_bg_blend if bg_blend == tcod.constants.BKGND_DEFAULT else None
        if bg_blend == tcod.constants.BKGND_NONE:
            bg = None
            bg_blend = None
        if bg_blend == tcod.constants.BKGND_SET:
            bg_blend = None
        if alignment is None:
            alignment = self.default_alignment
            if alignment == tcod.constants.LEFT:
                alignment = None
        else:
            alignment = None
        if clear is not ...:
            fg = None
        params = []
        if clear is True:
            params.append('ch=ord(" ")')
        if clear is False:
            params.append("ch=0")
        if fg is not None:
            params.append(f"fg={fg}")
        if bg is not None:
            params.append(f"bg={bg}")
        if bg_blend is not None:
            params.append(f"bg_blend={self.__BG_BLEND_LOOKUP[bg_blend]}")
        if alignment is not None:
            params.append(f"alignment={self.__ALIGNMENT_LOOKUP[alignment]}")
        param_str = ", ".join(params)
        param_str = "." if not param_str else f" and add the following parameters:\n{param_str}"
        warnings.warn(
            "Console functions using default values have been deprecated.\n"
            f"Replace this method with `Console.{new_func}`{param_str}",
            FutureWarning,
            stacklevel=3,
        )

    @deprecated("Switch to using keywords and then replace with 'console.print(...)'")
    def print_(
        self,
        x: int,
        y: int,
        string: str,
        bg_blend: int = tcod.constants.BKGND_DEFAULT,
        alignment: int | None = None,
    ) -> None:
        """Print a color formatted string on a console.

        Args:
            x (int): The x coordinate from the left.
            y (int): The y coordinate from the top.
            string (str): A Unicode string optionally using color codes.
            bg_blend (int): Blending mode to use, defaults to BKGND_DEFAULT.
            alignment (Optional[int]): Text alignment.

        .. deprecated:: 8.5
            Console methods which depend on console defaults have been
            deprecated.
            Use :any:`Console.print` instead, calling this function will print
            a warning detailing which default values need to be made explicit.
        """
        self.__deprecate_defaults("print", bg_blend, alignment)
        alignment = self.default_alignment if alignment is None else alignment
        lib.TCOD_console_printf_ex(self.console_c, x, y, bg_blend, alignment, _fmt(string))

    @deprecated("Switch to using keywords and then replace with 'console.print(...)'")
    def print_rect(  # noqa: PLR0913
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        string: str,
        bg_blend: int = tcod.constants.BKGND_DEFAULT,
        alignment: int | None = None,
    ) -> int:
        """Print a string constrained to a rectangle.

        If h > 0 and the bottom of the rectangle is reached,
        the string is truncated. If h = 0,
        the string is only truncated if it reaches the bottom of the console.

        Args:
            x (int): The x coordinate from the left.
            y (int): The y coordinate from the top.
            width (int): Maximum width to render the text.
            height (int): Maximum lines to render the text.
            string (str): A Unicode string.
            bg_blend (int): Background blending flag.
            alignment (Optional[int]): Alignment flag.

        Returns:
            int: The number of lines of text once word-wrapped.

        .. deprecated:: 8.5
            Console methods which depend on console defaults have been
            deprecated.
            Use :any:`Console.print_box` instead, calling this function will
            print a warning detailing which default values need to be made
            explicit.
        """
        self.__deprecate_defaults("print_box", bg_blend, alignment)
        alignment = self.default_alignment if alignment is None else alignment
        return int(
            lib.TCOD_console_printf_rect_ex(
                self.console_c,
                x,
                y,
                width,
                height,
                bg_blend,
                alignment,
                _fmt(string),
            )
        )

    def get_height_rect(self, x: int, y: int, width: int, height: int, string: str) -> int:
        """Return the height of this text word-wrapped into this rectangle.

        Args:
            x (int): The x coordinate from the left.
            y (int): The y coordinate from the top.
            width (int): Maximum width to render the text.
            height (int): Maximum lines to render the text.
            string (str): A Unicode string.

        Returns:
            int: The number of lines of text once word-wrapped.
        """
        string_ = string.encode("utf-8")
        return int(lib.TCOD_console_get_height_rect_n(self.console_c, x, y, width, height, len(string_), string_))

    @deprecated("""Replace with 'console.draw_rect(x, y, width, height, ch=..., fg=..., bg=..., bg_blend=...)'""")
    def rect(  # noqa: PLR0913
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        clear: bool,  # noqa: FBT001
        bg_blend: int = tcod.constants.BKGND_DEFAULT,
    ) -> None:
        """Draw a the background color on a rect optionally clearing the text.

        If `clear` is True the affected tiles are changed to space character.

        Args:
            x (int): The x coordinate from the left.
            y (int): The y coordinate from the top.
            width (int): Maximum width to render the text.
            height (int): Maximum lines to render the text.
            clear (bool): If True all text in the affected area will be
                          removed.
            bg_blend (int): Background blending flag.

        .. deprecated:: 8.5
            Console methods which depend on console defaults have been
            deprecated.
            Use :any:`Console.draw_rect` instead, calling this function will
            print a warning detailing which default values need to be made
            explicit.
        """
        self.__deprecate_defaults("draw_rect", bg_blend, clear=bool(clear))
        lib.TCOD_console_rect(self.console_c, x, y, width, height, clear, bg_blend)

    @deprecated(
        """Replace with 'console.draw_rect(x, y, width=width, height=1, ch=ord("─"), fg=..., bg=..., bg_blend=...)'"""
    )
    def hline(
        self,
        x: int,
        y: int,
        width: int,
        bg_blend: int = tcod.constants.BKGND_DEFAULT,
    ) -> None:
        """Draw a horizontal line on the console.

        This always uses ord("─"), the horizontal line character.

        Args:
            x (int): The x coordinate from the left.
            y (int): The y coordinate from the top.
            width (int): The horizontal length of this line.
            bg_blend (int): The background blending flag.

        .. deprecated:: 8.5
            Console methods which depend on console defaults have been
            deprecated.
            Use :any:`Console.draw_rect` instead, calling this function will
            print a warning detailing which default values need to be made
            explicit.
        """
        self.__deprecate_defaults("draw_rect", bg_blend)
        lib.TCOD_console_hline(self.console_c, x, y, width, bg_blend)

    @deprecated(
        """Replace with 'console.draw_rect(x, y, width=1, height=height, ch=ord("│"), fg=..., bg=..., bg_blend=...)'"""
    )
    def vline(
        self,
        x: int,
        y: int,
        height: int,
        bg_blend: int = tcod.constants.BKGND_DEFAULT,
    ) -> None:
        """Draw a vertical line on the console.

        This always uses ord("│"), the vertical line character.

        Args:
            x (int): The x coordinate from the left.
            y (int): The y coordinate from the top.
            height (int): The horizontal length of this line.
            bg_blend (int): The background blending flag.

        .. deprecated:: 8.5
            Console methods which depend on console defaults have been
            deprecated.
            Use :any:`Console.draw_rect` instead, calling this function will
            print a warning detailing which default values need to be made
            explicit.
        """
        self.__deprecate_defaults("draw_rect", bg_blend)
        lib.TCOD_console_vline(self.console_c, x, y, height, bg_blend)

    @deprecated("Replace with 'console.draw_frame(...)', use a separate print call for frame titles")
    def print_frame(  # noqa: PLR0913
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        string: str = "",
        clear: bool = True,  # noqa: FBT001, FBT002
        bg_blend: int = tcod.constants.BKGND_DEFAULT,
    ) -> None:
        """Draw a framed rectangle with optional text.

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

        .. versionchanged:: 8.2
            Now supports Unicode strings.

        .. deprecated:: 8.5
            Console methods which depend on console defaults have been
            deprecated.
            Use :any:`Console.draw_frame` instead, calling this function will
            print a warning detailing which default values need to be made
            explicit.
        """
        self.__deprecate_defaults("draw_frame", bg_blend)
        string_: Any = _fmt(string) if string else ffi.NULL
        _check(lib.TCOD_console_printf_frame(self.console_c, x, y, width, height, clear, bg_blend, string_))

    def blit(  # noqa: PLR0913
        self,
        dest: Console,
        dest_x: int = 0,
        dest_y: int = 0,
        src_x: int = 0,
        src_y: int = 0,
        width: int = 0,
        height: int = 0,
        fg_alpha: float = 1.0,
        bg_alpha: float = 1.0,
        key_color: tuple[int, int, int] | None = None,
    ) -> None:
        """Blit from this console onto the ``dest`` console.

        Args:
            dest (Console): The destination console to blit onto.
            dest_x (int): Leftmost coordinate of the destination console.
            dest_y (int): Topmost coordinate of the destination console.
            src_x (int): X coordinate from this console to blit, from the left.
            src_y (int): Y coordinate from this console to blit, from the top.
            width (int): The width of the region to blit.

                If this is 0 the maximum possible width will be used.
            height (int): The height of the region to blit.

                If this is 0 the maximum possible height will be used.
            fg_alpha (float): Foreground color alpha value.
            bg_alpha (float): Background color alpha value.
            key_color (Optional[Tuple[int, int, int]]):
                None, or a (red, green, blue) tuple with values of 0-255.

        .. versionchanged:: 4.0
            Parameters were rearranged and made optional.

            Previously they were:
            `(x, y, width, height, dest, dest_x, dest_y, *)`

        .. versionchanged:: 11.6
            Now supports per-cell alpha transparency.

            Use :any:`Console.buffer` to set tile alpha before blit.
        """
        # The old syntax is easy to detect and correct.
        if hasattr(src_y, "console_c"):
            (src_x, src_y, width, height, dest, dest_x, dest_y) = (
                dest,  # type: ignore[assignment]
                dest_x,
                dest_y,
                src_x,
                src_y,  # type: ignore[assignment]
                width,
                height,
            )
            warnings.warn(
                "Parameter names have been moved around, see documentation.",
                DeprecationWarning,
                stacklevel=2,
            )

        key_color = key_color or self._key_color
        if key_color:
            key_color = ffi.new("TCOD_color_t*", key_color)
            lib.TCOD_console_blit_key_color(
                self.console_c,
                src_x,
                src_y,
                width,
                height,
                dest.console_c,
                dest_x,
                dest_y,
                fg_alpha,
                bg_alpha,
                key_color,
            )
        else:
            lib.TCOD_console_blit(
                self.console_c,
                src_x,
                src_y,
                width,
                height,
                dest.console_c,
                dest_x,
                dest_y,
                fg_alpha,
                bg_alpha,
            )

    @deprecated("Pass the key color to Console.blit instead of calling this function.")
    def set_key_color(self, color: tuple[int, int, int] | None) -> None:
        """Set a consoles blit transparent color.

        `color` is the (r, g, b) color, or None to disable key color.

        .. deprecated:: 8.5
            Pass the key color to :any:`Console.blit` instead of calling this
            function.
        """
        self._key_color = color

    def __enter__(self) -> Self:
        """Return this console in a managed context.

        When the root console is used as a context, the graphical window will
        close once the context is left as if :any:`libtcodpy.console_delete` was
        called on it.

        This is useful for some Python IDE's like IDLE, where the window would
        not be closed on its own otherwise.

        .. seealso::
            :any:`libtcodpy.console_init_root`
        """
        if self.console_c != ffi.NULL:
            msg = "Only the root console has a context."
            raise NotImplementedError(msg)
        return self

    def close(self) -> None:
        """Close the active window managed by libtcod.

        This must only be called on the root console, which is returned from
        :any:`libtcodpy.console_init_root`.

        .. versionadded:: 11.11
        """
        if self.console_c != ffi.NULL:
            msg = "Only the root console can be used to close libtcod's window."
            raise NotImplementedError(msg)
        lib.TCOD_console_delete(self.console_c)

    def __exit__(self, *_: object) -> None:
        """Close the graphical window on exit.

        Some tcod functions may have undefined behavior after this point.
        """
        self.close()

    def __bool__(self) -> bool:
        """Return False if this is the root console.

        This mimics libtcodpy behavior.
        """
        return bool(self.console_c != ffi.NULL)

    def __getstate__(self) -> dict[str, Any]:
        """Support serialization via :mod:`pickle`."""
        state = self.__dict__.copy()
        del state["console_c"]
        state["_console_data"] = {
            "w": self.width,
            "h": self.height,
            "bkgnd_flag": self.default_bg_blend,
            "alignment": self.default_alignment,
            "fore": self.default_fg,
            "back": self.default_bg,
        }
        if self.console_c == ffi.NULL:
            state["_tiles"] = np.array(self._tiles, copy=True)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Support serialization via :mod:`pickle`."""
        self._key_color = None
        if "_tiles" not in state:
            tiles: NDArray[Any] = np.ndarray((self.height, self.width), dtype=self.DTYPE)
            tiles["ch"] = state["_ch"]
            tiles["fg"][..., :3] = state["_fg"]
            tiles["fg"][..., 3] = 255
            tiles["bg"][..., :3] = state["_bg"]
            tiles["bg"][..., 3] = 255
            state["_tiles"] = tiles
            del state["_ch"]
            del state["_fg"]
            del state["_bg"]

        self.__dict__.update(state)
        self._console_data["tiles"] = ffi.from_buffer("struct TCOD_ConsoleTile*", self._tiles)
        self._console_data = self.console_c = ffi.new("struct TCOD_Console*", self._console_data)

    def __repr__(self) -> str:
        """Return a string representation of this console."""
        return f"tcod.console.Console(width={self.width}, height={self.height}, order={self._order!r},buffer=\n{self.rgba!r})"

    def __str__(self) -> str:
        """Return a simplified representation of this consoles contents."""
        return "<{}>".format("\n ".join("".join(chr(c) for c in line) for line in self._tiles["ch"]))

    @overload
    def print(
        self,
        x: int,
        y: int,
        text: str,
        *,
        width: int | None = None,
        height: int | None = None,
        fg: tuple[int, int, int] | None = None,
        bg: tuple[int, int, int] | None = None,
        bg_blend: int = tcod.constants.BKGND_SET,
        alignment: int = tcod.constants.LEFT,
    ) -> int: ...

    @overload
    @deprecated(
        "Replace text, fg, bg, bg_blend, and alignment with keyword arguments."
        "\n'string' keyword should be renamed to `text`"
    )
    def print(
        self,
        x: int,
        y: int,
        text: str,
        fg: tuple[int, int, int] | None = None,
        bg: tuple[int, int, int] | None = None,
        bg_blend: int = tcod.constants.BKGND_SET,
        alignment: int = tcod.constants.LEFT,
        *,
        string: str = "",
    ) -> int: ...

    @overload
    @deprecated(
        "Replace text, fg, bg, bg_blend, and alignment with keyword arguments."
        "\n'string' keyword should be renamed to `text`"
    )
    def print(
        self,
        x: int,
        y: int,
        text: str = "",
        fg: tuple[int, int, int] | None = None,
        bg: tuple[int, int, int] | None = None,
        bg_blend: int = tcod.constants.BKGND_SET,
        alignment: int = tcod.constants.LEFT,
        *,
        string: str,
    ) -> int: ...

    def print(  # noqa: PLR0913
        self,
        x: int,
        y: int,
        text: str = "",
        fg: tuple[int, int, int] | None = None,
        bg: tuple[int, int, int] | None = None,
        bg_blend: int = tcod.constants.BKGND_SET,
        alignment: int = tcod.constants.LEFT,
        *,
        width: int | None = None,
        height: int | None = None,
        string: str = "",
    ) -> int:
        r"""Print a string of Unicode text on this console.

        Prefer using keywords for this method call to avoid ambiguous parameters.

        Args:
            x: Starting X coordinate, with the left-most tile as zero.
            y: Starting Y coordinate, with the top-most tile as zero.
            text: A Unicode string which may include color control characters.
            width: Width in tiles to constrain the printing region.
                If a `width` is given then `text` will have automatic word wrapping applied to it.
                A `width` of `None` means `text` will only have manual line breaks.
            height: Height in tiles to constrain the printing region.
            fg: RGB tuple to use as the foreground color, or `None` to leave the foreground unchanged.
                Tuple values should be 0-255.
                Must be given as a keyword argument.
            bg: RGB tuple to use as the background color, or `None` to leave the foreground unchanged.
                Tuple values should be 0-255.
                Must be given as a keyword argument.
            bg_blend: Background blend type used by libtcod.
                Typically starts with `libtcodpy.BKGND_*`.
                Must be given as a keyword argument.
            alignment: One of `libtcodpy.LEFT`, `libtcodpy.CENTER`, or `libtcodpy.RIGHT`
                Must be given as a keyword argument.
            string: Older deprecated name of the `text` parameter.

        Returns:
            The height of `text` in lines via word wrapping and line breaks.

        Example::

            >>> from tcod import libtcodpy
            >>> console = tcod.console.Console(20, 1)
            >>> console.clear(ch=ord('·'))
            >>> console.print(x=0, y=0, text="left")
            1
            >>> console.print(x=console.width, y=0, text="right", alignment=libtcodpy.RIGHT)
            1
            >>> console.print(x=10, y=0, text="center", alignment=libtcodpy.CENTER)
            1
            >>> print(console)
            <left···center··right>

            >>> console = tcod.console.Console(20, 4)
            >>> console.clear(ch=ord('·'))
            >>> console.print(x=1, y=1, text="words within bounds", width=8)
            3
            >>> print(console)
            <····················
             ·words··············
             ·within·············
             ·bounds·············>
            >>> WHITE = (255, 255, 255)
            >>> BLACK = (0, 0, 0)
            >>> console.print(x=0, y=0, text="Black text on white background", fg=BLACK, bg=WHITE)
            1

        .. versionadded:: 8.5

        .. versionchanged:: 9.0

            `fg` and `bg` now default to `None` instead of white-on-black.

        .. versionchanged:: 13.0

            `x` and `y` are now always used as an absolute position for negative values.

        .. versionchanged:: 18.0

            Deprecated giving `string`, `fg`, `bg`, and `bg_blend` as positional arguments.

            Added `text` parameter to replace `string`.

            Added `width` and `height` keyword parameters to bind text to a rectangle and replace other print functions.
            Right-aligned text with `width=None` now treats the `x` coordinate as a past-the-end index, this will shift
            the text of older calls to the left by 1 tile.

            Now returns the number of lines printed via word wrapping,
            same as previous print functions bound to rectangles.
        """
        if width is not None and width <= 0:
            return 0
        if width is None and alignment == tcod.constants.LEFT:  # Fix alignment
            width = 0x100000
        if width is None and alignment == tcod.constants.CENTER:  # Fix center alignment
            x -= 0x100000
            width = 0x200000
        if width is None and alignment == tcod.constants.RIGHT:  # Fix right alignment
            x -= 0x100000
            width = 0x100000
        rgb_fg = ffi.new("TCOD_ColorRGB*", fg) if fg is not None else ffi.NULL
        rgb_bg = ffi.new("TCOD_ColorRGB*", bg) if bg is not None else ffi.NULL
        utf8 = (string or text).encode("utf-8")
        return _check(
            int(
                lib.TCOD_printn_rgb(
                    self.console_c,
                    {
                        "x": x,
                        "y": y,
                        "width": width or 0,
                        "height": height or 0,
                        "fg": rgb_fg,
                        "bg": rgb_bg,
                        "flag": bg_blend,
                        "alignment": alignment,
                    },
                    len(utf8),
                    utf8,
                )
            )
        )

    @deprecated("Switch to using keywords and then replace with 'console.print(...)'")
    def print_box(  # noqa: PLR0913
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        string: str,
        fg: tuple[int, int, int] | None = None,
        bg: tuple[int, int, int] | None = None,
        bg_blend: int = tcod.constants.BKGND_SET,
        alignment: int = tcod.constants.LEFT,
    ) -> int:
        """Print a string constrained to a rectangle and return the height.

        `x` and `y` are the starting tile, with ``0,0`` as the upper-left
        corner of the console.

        `width` and `height` determine the bounds of the rectangle, the text
        will automatically be word-wrapped to fit within these bounds.

        `string` is a Unicode string which may include color control
        characters.

        `fg` and `bg` are the foreground text color and background tile color
        respectfully.  This is a 3-item tuple with (r, g, b) color values from
        0 to 255.  These parameters can also be set to `None` to leave the
        colors unchanged.

        `bg_blend` is the blend type used by libtcod.

        `alignment` can be `tcod.LEFT`, `tcod.CENTER`, or `tcod.RIGHT`.

        Returns the actual height of the printed area.

        .. versionadded:: 8.5

        .. versionchanged:: 9.0
            `fg` and `bg` now default to `None` instead of white-on-black.

        .. versionchanged:: 13.0
            `x` and `y` are now always used as an absolute position for negative values.
        """
        string_ = string.encode("utf-8")
        return int(
            lib.TCOD_console_printn_rect(
                self.console_c,
                x,
                y,
                width,
                height,
                len(string_),
                string_,
                (fg,) if fg is not None else ffi.NULL,
                (bg,) if bg is not None else ffi.NULL,
                bg_blend,
                alignment,
            )
        )

    @overload
    def draw_frame(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        *,
        clear: bool = True,
        fg: tuple[int, int, int] | None = None,
        bg: tuple[int, int, int] | None = None,
        bg_blend: int = tcod.constants.BKGND_SET,
        decoration: str | tuple[int, int, int, int, int, int, int, int, int] = "┌─┐│ │└─┘",
    ) -> None: ...

    @overload
    @deprecated("Parameters clear, fg, bg, bg_blend should be keyword arguments. Remove title parameter")
    def draw_frame(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        title: str = "",
        clear: bool = True,  # noqa: FBT001, FBT002
        fg: tuple[int, int, int] | None = None,
        bg: tuple[int, int, int] | None = None,
        bg_blend: int = tcod.constants.BKGND_SET,
        *,
        decoration: str | tuple[int, int, int, int, int, int, int, int, int] = "┌─┐│ │└─┘",
    ) -> None: ...

    def draw_frame(  # noqa: PLR0913
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        title: str = "",
        clear: bool = True,  # noqa: FBT001, FBT002
        fg: tuple[int, int, int] | None = None,
        bg: tuple[int, int, int] | None = None,
        bg_blend: int = tcod.constants.BKGND_SET,
        *,
        decoration: str | tuple[int, int, int, int, int, int, int, int, int] = "┌─┐│ │└─┘",
    ) -> None:
        r"""Draw a framed rectangle with an optional title.

        `x` and `y` are the starting tile, with ``0,0`` as the upper-left
        corner of the console.

        `width` and `height` determine the size of the frame.

        `title` is a Unicode string.  The title is drawn with `bg` as the text
        color and `fg` as the background.
        Using the `title` parameter is discouraged since the style it uses is
        hard-coded into libtcod.  You should print over the top or bottom
        border with :any:`Console.print_box` using your own style.

        If `clear` is True than the region inside of the frame will be cleared.
        Must be given as a keyword argument.

        `fg` and `bg` are the foreground and background colors for the frame
        border.  This is a 3-item tuple with (r, g, b) color values from
        0 to 255.  These parameters can also be set to `None` to leave the
        colors unchanged.
        Must be given as a keyword argument.

        `bg_blend` is the blend type used by libtcod.
        Must be given as a keyword argument.

        `decoration` is a sequence of glyphs to use for rendering the borders.
        This a str or tuple of int's with 9 items with the items arranged in
        row-major order.
        If a `decoration` is given then `title` can not be used because the
        style for `title` is hard-coded.  You can easily print along the upper
        or lower border of the frame manually.

        .. versionadded:: 8.5

        .. versionchanged:: 9.0
            `fg` and `bg` now default to `None` instead of white-on-black.

        .. versionchanged:: 12.6
            Added `decoration` parameter.

        .. versionchanged:: 13.0
            `x` and `y` are now always used as an absolute position for negative values.

        .. versionchanged:: 18.0
            Deprecated `clear`, `fg`, `bg`, and `bg_blend` being given as positional arguments.
            These should be keyword arguments only.

        Example::

            >>> from tcod import libtcodpy
            >>> console = tcod.console.Console(12, 6)
            >>> console.draw_frame(x=0, y=0, width=3, height=3)
            >>> console.draw_frame(x=3, y=0, width=3, height=3, decoration="╔═╗║ ║╚═╝")
            >>> console.draw_frame(x=6, y=0, width=3, height=3, decoration="123456789")
            >>> console.draw_frame(x=9, y=0, width=3, height=3, decoration="/-\\| |\\-/")
            >>> console.draw_frame(x=0, y=3, width=12, height=3)
            >>> console.print(x=0, y=3, width=12, height=1, string=" Title ", alignment=libtcodpy.CENTER)
            1
            >>> console.print(x=0, y=5, width=12, height=1, string="┤Lower├", alignment=libtcodpy.CENTER)
            1
            >>> print(console)
            <┌─┐╔═╗123/-\
             │ │║ ║456| |
             └─┘╚═╝789\-/
             ┌─ Title ──┐
             │          │
             └─┤Lower├──┘>
        """
        if title and decoration != "┌─┐│ │└─┘":
            msg = "The title and decoration parameters are mutually exclusive.  You should print the title manually."
            raise TypeError(msg)
        if title:
            warnings.warn(
                "The title parameter will be removed in the future since the style is hard-coded.",
                PendingDeprecationWarning,
                stacklevel=2,
            )
            title_ = title.encode("utf-8")
            lib.TCOD_console_printn_frame(
                self.console_c,
                x,
                y,
                width,
                height,
                len(title_),
                title_,
                (fg,) if fg is not None else ffi.NULL,
                (bg,) if bg is not None else ffi.NULL,
                bg_blend,
                clear,
            )
            return
        decoration_ = [ord(c) for c in decoration] if isinstance(decoration, str) else decoration
        if len(decoration_) != 9:  # noqa: PLR2004
            msg = f"Decoration must have a length of 9 (len(decoration) is {len(decoration_)}.)"
            raise TypeError(msg)
        _check(
            lib.TCOD_console_draw_frame_rgb(
                self.console_c,
                x,
                y,
                width,
                height,
                decoration_,
                (fg,) if fg is not None else ffi.NULL,
                (bg,) if bg is not None else ffi.NULL,
                bg_blend,
                clear,
            )
        )

    @overload
    def draw_rect(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        *,
        ch: int,
        fg: tuple[int, int, int] | None = None,
        bg: tuple[int, int, int] | None = None,
        bg_blend: int = tcod.constants.BKGND_SET,
    ) -> None: ...

    @overload
    @deprecated("Parameters ch, fg, bg, bg_blend should be keyword arguments")
    def draw_rect(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        ch: int,
        fg: tuple[int, int, int] | None = None,
        bg: tuple[int, int, int] | None = None,
        bg_blend: int = tcod.constants.BKGND_SET,
    ) -> None: ...

    def draw_rect(  # noqa: PLR0913
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        ch: int,
        fg: tuple[int, int, int] | None = None,
        bg: tuple[int, int, int] | None = None,
        bg_blend: int = tcod.constants.BKGND_SET,
    ) -> None:
        """Draw characters and colors over a rectangular region.

        `x` and `y` are the starting tile, with ``0,0`` as the upper-left
        corner of the console.

        `width` and `height` determine the size of the rectangle.

        `ch` is a Unicode integer.  You can use 0 to leave the current characters unchanged.
        Must be given as a keyword argument.

        `fg` and `bg` are the foreground text color and background tile color respectfully.
        This is a 3-item tuple with (r, g, b) color values from 0 to 255.
        These parameters can also be set to `None` to leave the colors unchanged.
        Must be given as a keyword argument.

        `bg_blend` is the blend type used by libtcod.
        Must be given as a keyword argument.

        .. versionadded:: 8.5

        .. versionchanged:: 9.0
            `fg` and `bg` now default to `None` instead of white-on-black.

        .. versionchanged:: 13.0
            `x` and `y` are now always used as an absolute position for negative values.

        .. versionchanged:: 18.0
            Deprecated `ch`, `fg`, `bg`, and `bg_blend` being given as positional arguments.
            These should be keyword arguments only.
        """
        lib.TCOD_console_draw_rect_rgb(
            self.console_c,
            x,
            y,
            width,
            height,
            ch,
            (fg,) if fg is not None else ffi.NULL,
            (bg,) if bg is not None else ffi.NULL,
            bg_blend,
        )

    def draw_semigraphics(self, pixels: ArrayLike | tcod.image.Image, x: int = 0, y: int = 0) -> None:
        """Draw a block of 2x2 semi-graphics into this console.

        `pixels` is an Image or an array-like object.  It will be down-sampled
        into 2x2 blocks when drawn.  Array-like objects must be in the shape of
        `(height, width, RGB)` and should have a `dtype` of `numpy.uint8`.

        `x` and `y` is the upper-left tile position to start drawing.

        .. versionadded:: 11.4
        """
        image = tcod.image._as_image(pixels)
        lib.TCOD_image_blit_2x(image.image_c, self.console_c, x, y, 0, 0, -1, -1)


def get_height_rect(width: int, string: str) -> int:
    """Return the number of lines which would be printed from these parameters.

    `width` is the width of the print boundary.

    `string` is a Unicode string which may include color control characters.

    .. versionadded:: 9.2
    """
    string_ = string.encode("utf-8")
    return int(lib.TCOD_console_get_height_rect_wn(width, len(string_), string_))


@deprecated("This function does not support contexts.", category=FutureWarning)
def recommended_size() -> tuple[int, int]:
    """Return the recommended size of a console for the current active window.

    The return is determined from the active tileset size and active window
    size.  This result should be used create an :any:`Console` instance.

    This function will raise RuntimeError if libtcod has not been initialized.

    .. versionadded:: 11.8

    .. seealso::
        :any:`libtcodpy.console_init_root`
        :any:`libtcodpy.console_flush`

    .. deprecated:: 11.13
        This function does not support contexts.
        Use :any:`Context.recommended_console_size` instead.
    """
    if not lib.TCOD_ctx.engine:
        msg = "The libtcod engine was not initialized first."
        raise RuntimeError(msg)
    window = lib.TCOD_sys_get_sdl_window()
    renderer = lib.TCOD_sys_get_sdl_renderer()
    with ffi.new("int[2]") as xy:
        if renderer:
            lib.SDL_GetCurrentRenderOutputSize(renderer, xy, xy + 1)
        else:  # Assume OpenGL if a renderer does not exist.
            lib.SDL_GetWindowSizeInPixels(window, xy, xy + 1)
        w = max(1, xy[0] // lib.TCOD_ctx.tileset.tile_width)
        h = max(1, xy[1] // lib.TCOD_ctx.tileset.tile_height)
    return w, h


def load_xp(path: str | PathLike[str], order: Literal["C", "F"] = "C") -> tuple[Console, ...]:
    """Load a REXPaint file as a tuple of consoles.

    `path` is the name of the REXPaint file to load.
    Usually ending with `.xp`.

    `order` is the memory order of the Console's array buffer,
    see :any:`tcod.console.Console`.

    .. versionadded:: 12.4

    Example::

        import numpy as np
        import tcod.console
        import tcod.tileset

        path = "example.xp"  # REXPaint file with one layer.

        # Load a REXPaint file with a single layer.
        # The comma after console is used to unpack a single item tuple.
        console, = tcod.console.load_xp(path, order="F")

        # Convert tcod's Code Page 437 character mapping into a NumPy array.
        CP437_TO_UNICODE = np.asarray(tcod.tileset.CHARMAP_CP437)

        # Convert from REXPaint's CP437 encoding to Unicode in-place.
        console.ch[:] = CP437_TO_UNICODE[console.ch]

        # Apply REXPaint's alpha key color.
        KEY_COLOR = (255, 0, 255)
        is_transparent = (console.rgb["bg"] == KEY_COLOR).all(axis=-1)
        console.rgba[is_transparent] = (ord(" "), (0,), (0,))
    """
    path = Path(path).resolve(strict=True)
    layers = _check(tcod.lib.TCOD_load_xp(_path_encode(path), 0, ffi.NULL))
    consoles = ffi.new("TCOD_Console*[]", layers)
    _check(tcod.lib.TCOD_load_xp(_path_encode(path), layers, consoles))
    return tuple(Console._from_cdata(console_p, order=order) for console_p in consoles)


def save_xp(
    path: str | PathLike[str],
    consoles: Iterable[Console],
    compress_level: int = 9,
) -> None:
    """Save tcod Consoles to a REXPaint file.

    `path` is where to save the file.

    `consoles` are the :any:`tcod.console.Console` objects to be saved.

    `compress_level` is the zlib compression level to be used.

    Color alpha will be lost during saving.

    Consoles will be saved as-is as much as possible.  You may need to convert
    characters from Unicode to CP437 if you want to load the file in REXPaint.

    .. versionadded:: 12.4

    Example::

        import numpy as np
        import tcod.console
        import tcod.tileset

        console = tcod.console.Console(80, 24)  # Example console.

        # Convert from Unicode to REXPaint's encoding.
        # Required to load this console correctly in the REXPaint tool.

        # Convert tcod's Code Page 437 character mapping into a NumPy array.
        CP437_TO_UNICODE = np.asarray(tcod.tileset.CHARMAP_CP437)

        # Initialize a Unicode-to-CP437 array.
        # 0x20000 is the current full range of Unicode.
        # fill_value=ord("?") means that "?" will be the result of any unknown codepoint.
        UNICODE_TO_CP437 = np.full(0x20000, fill_value=ord("?"))
        # Assign the CP437 mappings.
        UNICODE_TO_CP437[CP437_TO_UNICODE] = np.arange(len(CP437_TO_UNICODE))

        # Convert from Unicode to CP437 in-place.
        console.ch[:] = UNICODE_TO_CP437[console.ch]

        # Convert console alpha into REXPaint's alpha key color.
        KEY_COLOR = (255, 0, 255)
        is_transparent = console.rgba["bg"][:, :, 3] == 0
        console.rgb["bg"][is_transparent] = KEY_COLOR

        tcod.console.save_xp("example.xp", [console])
    """
    path = Path(path)
    consoles_c = ffi.new("TCOD_Console*[]", [c.console_c for c in consoles])
    _check(
        tcod.lib.TCOD_save_xp(
            len(consoles_c),
            consoles_c,
            _path_encode(path),
            compress_level,
        )
    )
