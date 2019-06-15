"""
libtcod works with a special 'root' console.  You create this console using
the :any:`tcod.console_init_root` function.  Usually after setting the font
with :any:`console_set_custom_font` first.

Example::

    # Make sure 'arial10x10.png' is in the same directory as this script.
    import tcod
    import tcod.event

    # Setup the font.
    tcod.console_set_custom_font(
        "arial10x10.png",
        tcod.FONT_LAYOUT_TCOD | tcod.FONT_TYPE_GREYSCALE,
    )
    # Initialize the root console in a context.
    with tcod.console_init_root(80, 60, order="F") as root_console:
        root_console.print_(x=0, y=0, string='Hello World!')
        while True:
            tcod.console_flush()  # Show the console.
            for event in tcod.event.wait():
                if event.type == "QUIT":
                    raise SystemExit()
        # The libtcod window will be closed at the end of this with-block.
"""

from typing import Any, Optional, Tuple  # noqa: F401
import warnings

import numpy as np

import tcod.constants
from tcod.libtcod import ffi, lib
import tcod._internal
from tcod._internal import deprecate


def _fmt(string: str) -> bytes:
    """Return a string that escapes 'C printf' side effects."""
    return string.encode("utf-8").replace(b"%", b"%%")


_root_console = None


class Console:
    """A console object containing a grid of characters with
    foreground/background colors.

    `width` and `height` are the size of the console (in tiles.)

    `order` determines how the axes of NumPy array attributes are arraigned.
    `order="F"` will swap the first two axes which allows for more intuitive
    `[x, y]` indexing.

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
                >>> buffer["ch"] = ord(' ')
                >>> buffer["ch"][:, 1] = ord('x')
                >>> c = tcod.console.Console(20, 3, order="F", buffer=buffer)
                >>> print(c)
                <                    |
                |xxxxxxxxxxxxxxxxxxxx|
                |                    >

            .. versionadded:: 8.5

            .. versionchanged:: 10.0
                Added an alpha channel to the color types.
    """

    DTYPE = np.dtype([("ch", np.intc), ("fg", "(4,)u1"), ("bg", "(4,)u1")])

    # A structured arrays type with the added "fg_rgb" and "bg_rgb" fields.
    _DTYPE_RGB = np.dtype(
        {
            "names": ["ch", "fg", "bg", "fg_rgb", "bg_rgb"],
            "formats": [np.int32, "(4,)u1", "(4,)u1", "(3,)u1", "(3,)u1"],
            "offsets": [0, 4, 8, 4, 8],
        }
    )

    def __init__(
        self,
        width: int,
        height: int,
        order: str = "C",
        buffer: Optional[np.array] = None,
    ):
        self._key_color = None  # type: Optional[Tuple[int, int, int]]
        self._order = tcod._internal.verify_order(order)
        if buffer is not None:
            if self._order == "F":
                buffer = buffer.transpose()
            self._tiles = np.ascontiguousarray(buffer, self.DTYPE)
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
                "tiles": ffi.cast(
                    "struct TCOD_ConsoleTile*", self._tiles.ctypes.data
                ),
                "bkgnd_flag": default_bg_blend,
                "alignment": default_alignment,
                "fore": (255, 255, 255),
                "back": (0, 0, 0),
            },
        )

        if buffer is None:
            self.clear()

    @classmethod
    def _from_cdata(cls, cdata: Any, order: str = "C") -> "Console":
        """Return a Console instance which wraps this `TCOD_Console*` object.
        """
        if isinstance(cdata, cls):
            return cdata
        self = object.__new__(cls)  # type: Console
        self.console_c = cdata
        self._init_setup_console_data(order)
        return self

    @classmethod
    def _get_root(cls, order: Optional[str] = None) -> "Console":
        """Return a root console singleton with valid buffers.

        This function will also update an already active root console.
        """
        global _root_console
        if _root_console is None:
            _root_console = object.__new__(cls)
        self = _root_console  # type: Console
        if order is not None:
            self._order = order
        self.console_c = ffi.NULL
        self._init_setup_console_data(self._order)
        return self

    def _init_setup_console_data(self, order: str = "C") -> None:
        """Setup numpy arrays over libtcod data buffers."""
        global _root_console
        self._key_color = None
        if self.console_c == ffi.NULL:
            _root_console = self
            self._console_data = lib.TCOD_ctx.root
        else:
            self._console_data = ffi.cast(
                "struct TCOD_Console*", self.console_c
            )

        self._tiles = np.frombuffer(
            ffi.buffer(self._console_data.tiles[0 : self.width * self.height]),
            dtype=self.DTYPE,
        ).reshape((self.height, self.width))

        self._order = tcod._internal.verify_order(order)

    @property
    def width(self) -> int:
        """int: The width of this Console. (read-only)"""
        return lib.TCOD_console_get_width(self.console_c)  # type: ignore

    @property
    def height(self) -> int:
        """int: The height of this Console. (read-only)"""
        return lib.TCOD_console_get_height(self.console_c)  # type: ignore

    @property
    def bg(self) -> np.array:
        """A uint8 array with the shape (height, width, 3).

        You can change the consoles background colors by using this array.

        Index this array with ``console.bg[i, j, channel]  # order='C'`` or
        ``console.bg[x, y, channel]  # order='F'``.

        """
        bg = self._tiles["bg"][..., :3]
        if self._order == "F":
            bg = bg.transpose(1, 0, 2)
        return bg

    @property
    def fg(self) -> np.array:
        """A uint8 array with the shape (height, width, 3).

        You can change the consoles foreground colors by using this array.

        Index this array with ``console.fg[i, j, channel]  # order='C'`` or
        ``console.fg[x, y, channel]  # order='F'``.
        """
        fg = self._tiles["fg"][..., :3]
        if self._order == "F":
            fg = fg.transpose(1, 0, 2)
        return fg

    @property
    def ch(self) -> np.array:
        """An integer array with the shape (height, width).

        You can change the consoles character codes by using this array.

        Index this array with ``console.ch[i, j]  # order='C'`` or
        ``console.ch[x, y]  # order='F'``.
        """
        return self._tiles["ch"].T if self._order == "F" else self._tiles["ch"]

    @property
    def tiles(self) -> np.array:
        """An array of this consoles tile data.

        This acts as a combination of the `ch`, `fg`, and `bg` attributes.
        Colors include an alpha channel but how alpha works is currently
        undefined.

        Example::
            >>> con = tcod.console.Console(10, 2, order="F")
            >>> con.tiles[0, 0] = (
            ...     ord("X"),
            ...     (*tcod.white, 255),
            ...     (*tcod.black, 255),
            ... )
            >>> con.tiles[0, 0]
            (88, [255, 255, 255, 255], [  0,   0,   0, 255])

        .. versionadded:: 10.0
        """
        return self._tiles.T if self._order == "F" else self._tiles

    @property
    def default_bg(self) -> Tuple[int, int, int]:
        """Tuple[int, int, int]: The default background color."""
        color = self._console_data.back
        return color.r, color.g, color.b

    @default_bg.setter  # type: ignore
    @deprecate("Console defaults have been deprecated.")
    def default_bg(self, color: Tuple[int, int, int]) -> None:
        self._console_data.back = color

    @property
    def default_fg(self) -> Tuple[int, int, int]:
        """Tuple[int, int, int]: The default foreground color."""
        color = self._console_data.fore
        return color.r, color.g, color.b

    @default_fg.setter  # type: ignore
    @deprecate("Console defaults have been deprecated.")
    def default_fg(self, color: Tuple[int, int, int]) -> None:
        self._console_data.fore = color

    @property
    def default_bg_blend(self) -> int:
        """int: The default blending mode."""
        return self._console_data.bkgnd_flag  # type: ignore

    @default_bg_blend.setter  # type: ignore
    @deprecate("Console defaults have been deprecated.")
    def default_bg_blend(self, value: int) -> None:
        self._console_data.bkgnd_flag = value

    @property
    def default_alignment(self) -> int:
        """int: The default text alignment."""
        return self._console_data.alignment  # type: ignore

    @default_alignment.setter  # type: ignore
    @deprecate("Console defaults have been deprecated.")
    def default_alignment(self, value: int) -> None:
        self._console_data.alignment = value

    def __clear_warning(self, name: str, value: Tuple[int, int, int]) -> None:
        """Raise a warning for bad default values during calls to clear."""
        warnings.warn(
            "Clearing with the console default values is deprecated.\n"
            "Add %s=%r to this call." % (name, value),
            DeprecationWarning,
            stacklevel=3,
        )

    def clear(  # type: ignore
        self,
        ch: int = ord(" "),
        fg: Tuple[int, int, int] = ...,
        bg: Tuple[int, int, int] = ...,
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
        if fg is ...:
            fg = self.default_fg
            if fg != (255, 255, 255):
                self.__clear_warning("fg", fg)
        if bg is ...:
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

    __ALIGNMENT_LOOKUP = {0: "tcod.LEFT", 1: "tcod.RIGHT", 2: "tcod.CENTER"}

    __BG_BLEND_LOOKUP = {
        0: "tcod.BKGND_NONE",
        1: "tcod.BKGND_SET",
        2: "tcod.BKGND_MULTIPLY",
        3: "tcod.BKGND_LIGHTEN",
        4: "tcod.BKGND_DARKEN",
        5: "tcod.BKGND_SCREEN",
        6: "tcod.BKGND_COLOR_DODGE",
        7: "tcod.BKGND_COLOR_BURN",
        8: "tcod.BKGND_ADD",
        9: "tcod.BKGND_ADDA",
        10: "tcod.BKGND_BURN",
        11: "tcod.BKGND_OVERLAY",
        12: "tcod.BKGND_ALPH",
        13: "tcod.BKGND_DEFAULT",
    }

    def __deprecate_defaults(
        self,
        new_func: str,
        bg_blend: Any,
        alignment: Any = ...,
        clear: Any = ...,
    ) -> None:
        """Return the parameters needed to recreate the current default state.
        """
        if not __debug__:
            return

        fg = self.default_fg  # type: Any
        bg = self.default_bg  # type: Any
        if bg_blend == tcod.constants.BKGND_NONE:
            bg = None
        if bg_blend == tcod.constants.BKGND_DEFAULT:
            bg_blend = self.default_bg_blend
        else:
            bg_blend = None
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
            params.append("fg=%s" % (fg,))
        if bg is not None:
            params.append("bg=%s" % (bg,))
        if bg_blend is not None:
            params.append("bg_blend=%s" % (self.__BG_BLEND_LOOKUP[bg_blend],))
        if alignment is not None:
            params.append(
                "alignment=%s" % (self.__ALIGNMENT_LOOKUP[alignment],)
            )
        param_str = ", ".join(params)
        if not param_str:
            param_str = "."
        else:
            param_str = " and add the following parameters:\n%s" % (param_str,)
        warnings.warn(
            "Console functions using default values have been deprecated.\n"
            "Replace this method with `Console.%s`%s" % (new_func, param_str),
            DeprecationWarning,
            stacklevel=3,
        )

    def print_(
        self,
        x: int,
        y: int,
        string: str,
        bg_blend: int = tcod.constants.BKGND_DEFAULT,
        alignment: Optional[int] = None,
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
        lib.TCOD_console_printf_ex(
            self.console_c, x, y, bg_blend, alignment, _fmt(string)
        )

    def print_rect(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        string: str,
        bg_blend: int = tcod.constants.BKGND_DEFAULT,
        alignment: Optional[int] = None,
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

    def get_height_rect(
        self, x: int, y: int, width: int, height: int, string: str
    ) -> int:
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
        return int(
            lib.get_height_rect(
                self.console_c, x, y, width, height, string_, len(string_)
            )
        )

    def rect(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        clear: bool,
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
        lib.TCOD_console_rect(
            self.console_c, x, y, width, height, clear, bg_blend
        )

    def hline(
        self,
        x: int,
        y: int,
        width: int,
        bg_blend: int = tcod.constants.BKGND_DEFAULT,
    ) -> None:
        """Draw a horizontal line on the console.

        This always uses ord('─'), the horizontal line character.

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

    def vline(
        self,
        x: int,
        y: int,
        height: int,
        bg_blend: int = tcod.constants.BKGND_DEFAULT,
    ) -> None:
        """Draw a vertical line on the console.

        This always uses ord('│'), the vertical line character.

        Args:
            x (int): The x coordinate from the left.
            y (int): The y coordinate from the top.
            height (int): The horozontal length of this line.
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

    def print_frame(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        string: str = "",
        clear: bool = True,
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
        string = _fmt(string) if string else ffi.NULL
        lib.TCOD_console_printf_frame(
            self.console_c, x, y, width, height, clear, bg_blend, string
        )

    def blit(
        self,
        dest: "Console",
        dest_x: int = 0,
        dest_y: int = 0,
        src_x: int = 0,
        src_y: int = 0,
        width: int = 0,
        height: int = 0,
        fg_alpha: float = 1.0,
        bg_alpha: float = 1.0,
        key_color: Optional[Tuple[int, int, int]] = None,
    ) -> None:
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
        if hasattr(src_y, "console_c"):
            (
                src_x,  # type: ignore
                src_y,
                width,
                height,
                dest,  # type: ignore
                dest_x,
                dest_y,
            ) = (dest, dest_x, dest_y, src_x, src_y, width, height)
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

    @deprecate(
        "Pass the key color to Console.blit instead of calling this function."
    )
    def set_key_color(self, color: Optional[Tuple[int, int, int]]) -> None:
        """Set a consoles blit transparent color.

        `color` is the (r, g, b) color, or None to disable key color.

        .. deprecated:: 8.5
            Pass the key color to :any:`Console.blit` instead of calling this
            function.
        """
        self._key_color = color

    def __enter__(self) -> "Console":
        """Returns this console in a managed context.

        When the root console is used as a context, the graphical window will
        close once the context is left as if :any:`tcod.console_delete` was
        called on it.

        This is useful for some Python IDE's like IDLE, where the window would
        not be closed on its own otherwise.

        .. seealso::
            :any:`tcod.console_init_root`
        """
        if self.console_c != ffi.NULL:
            raise NotImplementedError("Only the root console has a context.")
        return self

    def __exit__(self, *args: Any) -> None:
        """Closes the graphical window on exit.

        Some tcod functions may have undefined behavior after this point.
        """
        lib.TCOD_console_delete(self.console_c)

    def __bool__(self) -> bool:
        """Returns False if this is the root console.

        This mimics libtcodpy behavior.
        """
        return bool(self.console_c != ffi.NULL)

    def __getstate__(self) -> Any:
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
            state["_tiles"] = np.copy(self._tiles)
        return state

    def __setstate__(self, state: Any) -> None:
        self._key_color = None
        if "_tiles" not in state:
            tiles = np.ndarray((self.height, self.width), dtype=self.DTYPE)
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
        self._console_data["tiles"] = ffi.cast(
            "struct TCOD_ConsoleTile*", self._tiles.ctypes.data
        )
        self._console_data = self.console_c = ffi.new(
            "struct TCOD_Console*", self._console_data
        )

    def __repr__(self) -> str:
        """Return a string representation of this console."""
        return (
            "tcod.console.Console(width=%i, height=%i, "
            "order=%r,buffer=\n%r)"
            % (self.width, self.height, self._order, self.tiles)
        )

    def __str__(self) -> str:
        """Return a simplified representation of this consoles contents."""
        return "<%s>" % "|\n|".join(
            "".join(chr(c) for c in line) for line in self._tiles["ch"]
        )

    def _pythonic_index(self, x: int, y: int) -> Tuple[int, int]:
        if __debug__ and (x < 0 or y < 0):
            warnings.warn(
                "How negative indexes are handled my change in the future.",
                PendingDeprecationWarning,
                stacklevel=3,
            )
        if x < 0:
            x += self.width
        if y < 0:
            y += self.height
        return x, y

    def print(
        self,
        x: int,
        y: int,
        string: str,
        fg: Optional[Tuple[int, int, int]] = None,
        bg: Optional[Tuple[int, int, int]] = None,
        bg_blend: int = tcod.constants.BKGND_SET,
        alignment: int = tcod.constants.LEFT,
    ) -> None:
        """Print a string on a console with manual line breaks.

        `x` and `y` are the starting tile, with ``0,0`` as the upper-left
        corner of the console.  You can use negative numbers if you want to
        start printing relative to the bottom-right corner, but this behavior
        may change in future versions.

        `string` is a Unicode string which may include color control
        characters.  Strings which are too long will be truncated until the
        next newline character ``"\\n"``.

        `fg` and `bg` are the foreground text color and background tile color
        respectfully.  This is a 3-item tuple with (r, g, b) color values from
        0 to 255.  These parameters can also be set to `None` to leave the
        colors unchanged.

        `bg_blend` is the blend type used by libtcod.

        `alignment` can be `tcod.LEFT`, `tcod.CENTER`, or `tcod.RIGHT`.

        .. versionadded:: 8.5

        .. versionchanged:: 9.0
            `fg` and `bg` now default to `None` instead of white-on-black.
        """
        x, y = self._pythonic_index(x, y)
        string_ = string.encode("utf-8")  # type: bytes
        lib.console_print(
            self.console_c,
            x,
            y,
            string_,
            len(string_),
            (fg,) if fg is not None else ffi.NULL,
            (bg,) if bg is not None else ffi.NULL,
            bg_blend,
            alignment,
        )

    def print_box(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        string: str,
        fg: Optional[Tuple[int, int, int]] = None,
        bg: Optional[Tuple[int, int, int]] = None,
        bg_blend: int = tcod.constants.BKGND_SET,
        alignment: int = tcod.constants.LEFT,
    ) -> int:
        """Print a string constrained to a rectangle and return the height.

        `x` and `y` are the starting tile, with ``0,0`` as the upper-left
        corner of the console.  You can use negative numbers if you want to
        start printing relative to the bottom-right corner, but this behavior
        may change in future versions.

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
        """
        x, y = self._pythonic_index(x, y)
        string_ = string.encode("utf-8")  # type: bytes
        return int(
            lib.print_rect(
                self.console_c,
                x,
                y,
                width,
                height,
                string_,
                len(string_),
                (fg,) if fg is not None else ffi.NULL,
                (bg,) if bg is not None else ffi.NULL,
                bg_blend,
                alignment,
            )
        )

    def draw_frame(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        title: str = "",
        clear: bool = True,
        fg: Optional[Tuple[int, int, int]] = None,
        bg: Optional[Tuple[int, int, int]] = None,
        bg_blend: int = tcod.constants.BKGND_SET,
    ) -> None:
        """Draw a framed rectangle with an optional title.

        `x` and `y` are the starting tile, with ``0,0`` as the upper-left
        corner of the console.  You can use negative numbers if you want to
        start printing relative to the bottom-right corner, but this behavior
        may change in future versions.

        `width` and `height` determine the size of the frame.

        `title` is a Unicode string.

        If `clear` is True than the region inside of the frame will be cleared.

        `fg` and `bg` are the foreground text color and background tile color
        respectfully.  This is a 3-item tuple with (r, g, b) color values from
        0 to 255.  These parameters can also be set to `None` to leave the
        colors unchanged.

        `bg_blend` is the blend type used by libtcod.

        .. versionadded:: 8.5

        .. versionchanged:: 9.0
            `fg` and `bg` now default to `None` instead of white-on-black.
        """
        x, y = self._pythonic_index(x, y)
        title_ = title.encode("utf-8")  # type: bytes
        lib.print_frame(
            self.console_c,
            x,
            y,
            width,
            height,
            title_,
            len(title_),
            (fg,) if fg is not None else ffi.NULL,
            (bg,) if bg is not None else ffi.NULL,
            bg_blend,
            clear,
        )

    def draw_rect(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        ch: int,
        fg: Optional[Tuple[int, int, int]] = None,
        bg: Optional[Tuple[int, int, int]] = None,
        bg_blend: int = tcod.constants.BKGND_SET,
    ) -> None:
        """Draw characters and colors over a rectangular region.

        `x` and `y` are the starting tile, with ``0,0`` as the upper-left
        corner of the console.  You can use negative numbers if you want to
        start printing relative to the bottom-right corner, but this behavior
        may change in future versions.

        `width` and `height` determine the size of the rectangle.

        `ch` is a Unicode integer.  You can use 0 to leave the current
        characters unchanged.

        `fg` and `bg` are the foreground text color and background tile color
        respectfully.  This is a 3-item tuple with (r, g, b) color values from
        0 to 255.  These parameters can also be set to `None` to leave the
        colors unchanged.

        `bg_blend` is the blend type used by libtcod.

        .. versionadded:: 8.5

        .. versionchanged:: 9.0
            `fg` and `bg` now default to `None` instead of white-on-black.
        """
        x, y = self._pythonic_index(x, y)
        lib.draw_rect(
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


def get_height_rect(width: int, string: str) -> int:
    """Return the number of lines which would be printed from these parameters.

    `width` is the width of the print boundary.

    `string` is a Unicode string which may include color control characters.

    .. versionadded:: 9.2
    """
    string_ = string.encode("utf-8")  # type: bytes
    return int(lib.get_height_rect2(width, string_, len(string_)))
