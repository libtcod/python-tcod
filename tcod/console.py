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

    `copy` is a placeholder.  In the future this will determine if the
    `buffer` is copied or used as-is.  So set it to None or True as
    appropriate.

    `default_bg`, `default_bg`, `default_bg_blend`, and `default_alignment` are
    the default values used in some methods.  The `default_bg` and `default_bg`
    will affect the starting foreground and background color of the console.

    .. versionchanged:: 4.3
        Added `order` parameter.

    .. versionchanged:: 8.5
        Added `buffer`, `copy`, and default parameters.
        Arrays are initialized as if the :any:`clear` method was called.

    Attributes:
        console_c: A python-cffi "TCOD_Console*" object.
        DTYPE:
            A class attribute which provides a dtype compatible with this
            class.

            ``[("ch", np.intc), ("fg", "(3,)u1"), ("bg", "(3,)u1")]``

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
    """

    DTYPE = [("ch", np.intc), ("fg", "(3,)u1"), ("bg", "(3,)u1")]  # type: Any

    def __init__(
        self,
        width: int,
        height: int,
        order: str = "C",
        buffer: Optional[np.array] = None,
        copy: Optional[bool] = None,
        default_bg: Tuple[int, int, int] = (0, 0, 0),
        default_fg: Tuple[int, int, int] = (255, 255, 255),
        default_bg_blend: Optional[int] = None,
        default_alignment: Optional[int] = None,
    ):
        self._key_color = None  # type: Optional[Tuple[int, int, int]]
        self._order = tcod._internal.verify_order(order)
        if copy is not None and not copy:
            raise ValueError("copy=False is not supported in this version.")
        if buffer is not None:
            if self._order == "F":
                buffer = buffer.transpose()
            self._ch = np.ascontiguousarray(buffer["ch"], np.intc)
            self._fg = np.ascontiguousarray(buffer["fg"], "u1")
            self._bg = np.ascontiguousarray(buffer["bg"], "u1")
        else:
            self._ch = np.ndarray((height, width), dtype=np.intc)
            self._fg = np.ndarray((height, width), dtype="(3,)u1")
            self._bg = np.ndarray((height, width), dtype="(3,)u1")

        # libtcod uses the root console for defaults.
        if default_bg_blend is None:
            default_bg_blend = 0
            if lib.TCOD_ctx.root != ffi.NULL:
                default_bg_blend = lib.TCOD_ctx.root.bkgnd_flag
        if default_alignment is None:
            default_alignment = 0
            if lib.TCOD_ctx.root != ffi.NULL:
                default_alignment = lib.TCOD_ctx.root.alignment

        self._console_data = self.console_c = ffi.new(
            "struct TCOD_Console*",
            {
                "w": width,
                "h": height,
                "ch_array": ffi.cast("int*", self._ch.ctypes.data),
                "fg_array": ffi.cast("TCOD_color_t*", self._fg.ctypes.data),
                "bg_array": ffi.cast("TCOD_color_t*", self._bg.ctypes.data),
                "bkgnd_flag": default_bg_blend,
                "alignment": default_alignment,
                "fore": default_fg,
                "back": default_bg,
            },
        )

        if buffer is None:
            self.clear(fg=default_fg, bg=default_bg)

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

        def unpack_color(color_data: Any) -> np.array:
            """return a (height, width, 3) shaped array from an image struct"""
            color_buffer = ffi.buffer(color_data[0 : self.width * self.height])
            array = np.frombuffer(color_buffer, np.uint8)
            return array.reshape((self.height, self.width, 3))

        self._fg = unpack_color(self._console_data.fg_array)
        self._bg = unpack_color(self._console_data.bg_array)

        buf = self._console_data.ch_array
        buf = ffi.buffer(buf[0 : self.width * self.height])
        self._ch = np.frombuffer(buf, np.intc).reshape(
            (self.height, self.width)
        )

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
        return self._bg.transpose(1, 0, 2) if self._order == "F" else self._bg

    @property
    def fg(self) -> np.array:
        """A uint8 array with the shape (height, width, 3).

        You can change the consoles foreground colors by using this array.

        Index this array with ``console.fg[i, j, channel]  # order='C'`` or
        ``console.fg[x, y, channel]  # order='F'``.
        """
        return self._fg.transpose(1, 0, 2) if self._order == "F" else self._fg

    @property
    def ch(self) -> np.array:
        """An integer array with the shape (height, width).

        You can change the consoles character codes by using this array.

        Index this array with ``console.ch[i, j]  # order='C'`` or
        ``console.ch[x, y]  # order='F'``.
        """
        return self._ch.T if self._order == "F" else self._ch

    @property
    def default_bg(self) -> Tuple[int, int, int]:
        """Tuple[int, int, int]: The default background color."""
        color = self._console_data.back
        return color.r, color.g, color.b

    @default_bg.setter
    def default_bg(self, color: Tuple[int, int, int]) -> None:
        self._console_data.back = color

    @property
    def default_fg(self) -> Tuple[int, int, int]:
        """Tuple[int, int, int]: The default foreground color."""
        color = self._console_data.fore
        return color.r, color.g, color.b

    @default_fg.setter
    def default_fg(self, color: Tuple[int, int, int]) -> None:
        self._console_data.fore = color

    @property
    def default_bg_blend(self) -> int:
        """int: The default blending mode."""
        return self._console_data.bkgnd_flag  # type: ignore

    @default_bg_blend.setter
    def default_bg_blend(self, value: int) -> None:
        self._console_data.bkgnd_flag = value

    @property
    def default_alignment(self) -> int:
        """int: The default text alignment."""
        return self._console_data.alignment  # type: ignore

    @default_alignment.setter
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
        self.ch[...] = ch
        self.fg[...] = fg
        self.bg[...] = bg

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
        """
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
        """
        alignment = self.default_alignment if alignment is None else alignment
        return lib.TCOD_console_printf_rect_ex(  # type: ignore
            self.console_c,
            x,
            y,
            width,
            height,
            bg_blend,
            alignment,
            _fmt(string),
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
        return lib.TCOD_console_get_height_rect_fmt(  # type: ignore
            self.console_c, x, y, width, height, _fmt(string)
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

        If clr is True the affected tiles are changed to space character.

        Args:
            x (int): The x coordinate from the left.
            y (int): The y coordinate from the top.
            width (int): Maximum width to render the text.
            height (int): Maximum lines to render the text.
            clear (bool): If True all text in the affected area will be
                          removed.
            bg_blend (int): Background blending flag.
        """
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

        This always uses the character 196, the horizontal line character.

        Args:
            x (int): The x coordinate from the left.
            y (int): The y coordinate from the top.
            width (int): The horizontal length of this line.
            bg_blend (int): The background blending flag.
        """
        lib.TCOD_console_hline(self.console_c, x, y, width, bg_blend)

    def vline(
        self,
        x: int,
        y: int,
        height: int,
        bg_blend: int = tcod.constants.BKGND_DEFAULT,
    ) -> None:
        """Draw a vertical line on the console.

        This always uses the character 179, the vertical line character.

        Args:
            x (int): The x coordinate from the left.
            y (int): The y coordinate from the top.
            height (int): The horozontal length of this line.
            bg_blend (int): The background blending flag.
        """
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
        """Draw a framed rectangle with optinal text.

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
        """
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
        """
        if self.console_c != ffi.NULL:
            raise NotImplementedError("Only the root console has a context.")
        return self

    def __exit__(self, *args: Any) -> None:
        """Closes the graphical window on exit.

        Some tcod functions may have undefined behaviour after this point.
        """
        lib.TCOD_console_delete(self.console_c)

    def __bool__(self) -> bool:
        """Returns False if this is the root console.

        This mimics libtcodpy behaviour.
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
            state["_ch"] = np.copy(self._ch)
            state["_fg"] = np.copy(self._fg)
            state["_bg"] = np.copy(self._bg)
        return state

    def __setstate__(self, state: Any) -> None:
        self._key_color = None
        self.__dict__.update(state)
        self._console_data.update(
            {
                "ch_array": ffi.cast("int*", self._ch.ctypes.data),
                "fg_array": ffi.cast("TCOD_color_t*", self._fg.ctypes.data),
                "bg_array": ffi.cast("TCOD_color_t*", self._bg.ctypes.data),
            }
        )
        self._console_data = self.console_c = ffi.new(
            "struct TCOD_Console*", self._console_data
        )

    def __repr__(self) -> str:
        """Return a string representation of this console."""
        buffer = np.ndarray((self.height, self.width), dtype=self.DTYPE)
        buffer["ch"] = self.ch
        buffer["fg"] = self.fg
        buffer["bg"] = self.bg
        return (
            "tcod.console.Console(width=%i, height=%i, "
            "order=%r,buffer=\n%r,\ndefault_bg=%r, default_fg=%r, "
            "default_bg_blend=%s, default_alignment=%s)"
            % (
                self.width,
                self.height,
                self._order,
                buffer,
                self.default_bg,
                self.default_fg,
                self.default_bg_blend,
                self.default_alignment,
            )
        )

    def __str__(self) -> str:
        """Return a simplified representation of this consoles contents."""
        return "<%s>" % "|\n|".join(
            "".join(chr(c) for c in line) for line in self._ch
        )
