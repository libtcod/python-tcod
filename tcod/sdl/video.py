"""SDL2 Window and Display handling.

There are two main ways to access the SDL window.
Either you can use this module to open a window yourself bypassing libtcod's context,
or you can use :any:`Context.sdl_window` to get the window being controlled by that context (if the context has one.)

.. versionadded:: 13.4
"""

from __future__ import annotations

import enum
import sys
from typing import TYPE_CHECKING, Any

import numpy as np
from typing_extensions import Self, deprecated

import tcod.sdl.constants
from tcod.cffi import ffi, lib
from tcod.sdl._internal import Properties, _check, _check_p, _required_version

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

__all__ = (
    "Capitalization",
    "FlashOperation",
    "TextInputType",
    "Window",
    "WindowFlags",
    "get_grabbed_window",
    "new_window",
    "screen_saver_allowed",
)


class WindowFlags(enum.IntFlag):
    """Bit flags which make up a windows state.

    .. seealso::
        https://wiki.libsdl.org/SDL_WindowFlags
    """

    FULLSCREEN = int(lib.SDL_WINDOW_FULLSCREEN)
    """"""
    OPENGL = int(lib.SDL_WINDOW_OPENGL)
    """"""
    HIDDEN = int(lib.SDL_WINDOW_HIDDEN)
    """"""
    BORDERLESS = int(lib.SDL_WINDOW_BORDERLESS)
    """"""
    RESIZABLE = int(lib.SDL_WINDOW_RESIZABLE)
    """"""
    MINIMIZED = int(lib.SDL_WINDOW_MINIMIZED)
    """"""
    MAXIMIZED = int(lib.SDL_WINDOW_MAXIMIZED)
    """"""
    MOUSE_GRABBED = int(lib.SDL_WINDOW_MOUSE_GRABBED)
    """"""
    INPUT_FOCUS = int(lib.SDL_WINDOW_INPUT_FOCUS)
    """"""
    MOUSE_FOCUS = int(lib.SDL_WINDOW_MOUSE_FOCUS)
    """"""
    ALLOW_HIGHDPI = int(lib.SDL_WINDOW_HIGH_PIXEL_DENSITY)
    """"""
    MOUSE_CAPTURE = int(lib.SDL_WINDOW_MOUSE_CAPTURE)
    """"""
    ALWAYS_ON_TOP = int(lib.SDL_WINDOW_ALWAYS_ON_TOP)
    """"""
    UTILITY = int(lib.SDL_WINDOW_UTILITY)
    """"""
    TOOLTIP = int(lib.SDL_WINDOW_TOOLTIP)
    """"""
    POPUP_MENU = int(lib.SDL_WINDOW_POPUP_MENU)
    """"""
    VULKAN = int(lib.SDL_WINDOW_VULKAN)
    """"""
    METAL = int(getattr(lib, "SDL_WINDOW_METAL", 0x20000000))  # SDL >= 2.0.14
    """"""


class FlashOperation(enum.IntEnum):
    """Values for :any:`Window.flash`."""

    CANCEL = 0
    """Stop flashing."""
    BRIEFLY = 1
    """Flash briefly."""
    UNTIL_FOCUSED = 2
    """Flash until focus is gained."""


class TextInputType(enum.IntEnum):
    """SDL input types for text input.

    .. seealso::
        :any:`Window.start_text_input`
        https://wiki.libsdl.org/SDL3/SDL_TextInputType

    .. versionadded:: 19.1
    """

    TEXT = lib.SDL_TEXTINPUT_TYPE_TEXT
    """The input is text."""
    TEXT_NAME = lib.SDL_TEXTINPUT_TYPE_TEXT_NAME
    """The input is a person's name."""
    TEXT_EMAIL = lib.SDL_TEXTINPUT_TYPE_TEXT_EMAIL
    """The input is an e-mail address."""
    TEXT_USERNAME = lib.SDL_TEXTINPUT_TYPE_TEXT_USERNAME
    """The input is a username."""
    TEXT_PASSWORD_HIDDEN = lib.SDL_TEXTINPUT_TYPE_TEXT_PASSWORD_HIDDEN
    """The input is a secure password that is hidden."""
    TEXT_PASSWORD_VISIBLE = lib.SDL_TEXTINPUT_TYPE_TEXT_PASSWORD_VISIBLE
    """The input is a secure password that is visible."""
    NUMBER = lib.SDL_TEXTINPUT_TYPE_NUMBER
    """The input is a number."""
    NUMBER_PASSWORD_HIDDEN = lib.SDL_TEXTINPUT_TYPE_NUMBER_PASSWORD_HIDDEN
    """The input is a secure PIN that is hidden."""
    NUMBER_PASSWORD_VISIBLE = lib.SDL_TEXTINPUT_TYPE_NUMBER_PASSWORD_VISIBLE
    """The input is a secure PIN that is visible."""


class Capitalization(enum.IntEnum):
    """Text capitalization for text input.

    .. seealso::
        :any:`Window.start_text_input`
        https://wiki.libsdl.org/SDL3/SDL_Capitalization

    .. versionadded:: 19.1
    """

    NONE = lib.SDL_CAPITALIZE_NONE
    """No auto-capitalization will be done."""
    SENTENCES = lib.SDL_CAPITALIZE_SENTENCES
    """The first letter of sentences will be capitalized."""
    WORDS = lib.SDL_CAPITALIZE_WORDS
    """The first letter of words will be capitalized."""
    LETTERS = lib.SDL_CAPITALIZE_LETTERS
    """All letters will be capitalized."""


class _TempSurface:
    """Holds a temporary surface derived from a NumPy array."""

    def __init__(self, pixels: ArrayLike) -> None:
        self._array: NDArray[np.uint8] = np.ascontiguousarray(pixels, dtype=np.uint8)
        if len(self._array.shape) != 3:  # noqa: PLR2004
            msg = f"NumPy shape must be 3D [y, x, ch] (got {self._array.shape})"
            raise TypeError(msg)
        if not (3 <= self._array.shape[2] <= 4):  # noqa: PLR2004
            msg = f"NumPy array must have RGB or RGBA channels. (got {self._array.shape})"
            raise TypeError(msg)
        self.p = ffi.gc(
            _check_p(
                lib.SDL_CreateSurfaceFrom(
                    self._array.shape[1],
                    self._array.shape[0],
                    lib.SDL_PIXELFORMAT_RGBA32 if self._array.shape[2] == 4 else lib.SDL_PIXELFORMAT_RGB24,
                    ffi.from_buffer("void*", self._array),
                    self._array.strides[0],
                )
            ),
            lib.SDL_DestroySurface,
        )


class Window:
    """An SDL2 Window object."""

    def __init__(self, sdl_window_p: Any) -> None:  # noqa: ANN401
        if ffi.typeof(sdl_window_p) is not ffi.typeof("struct SDL_Window*"):
            msg = "sdl_window_p must be {!r} type (was {!r}).".format(
                ffi.typeof("struct SDL_Window*"), ffi.typeof(sdl_window_p)
            )
            raise TypeError(msg)
        if not sdl_window_p:
            msg = "sdl_window_p can not be a null pointer."
            raise TypeError(msg)
        self.p = sdl_window_p

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Window):
            return NotImplemented
        return bool(self.p == other.p)

    def __hash__(self) -> int:
        return hash(self.p)

    def _as_property_pointer(self) -> Any:  # noqa: ANN401
        return self.p

    @classmethod
    def _from_property_pointer(cls, raw_cffi_pointer: Any, /) -> Self:  # noqa: ANN401
        return cls(raw_cffi_pointer)

    def set_icon(self, pixels: ArrayLike) -> None:
        """Set the window icon from an image.

        Args:
            pixels: A row-major array of RGB or RGBA pixel values.
        """
        surface = _TempSurface(pixels)
        lib.SDL_SetWindowIcon(self.p, surface.p)

    @property
    def position(self) -> tuple[int, int]:
        """Get or set the (x, y) position of the window.

        This attribute can be set the move the window.
        The constants tcod.lib.SDL_WINDOWPOS_CENTERED or tcod.lib.SDL_WINDOWPOS_UNDEFINED may be used.
        """
        xy = ffi.new("int[2]")
        lib.SDL_GetWindowPosition(self.p, xy, xy + 1)
        return xy[0], xy[1]

    @position.setter
    def position(self, xy: tuple[int, int]) -> None:
        x, y = xy
        lib.SDL_SetWindowPosition(self.p, x, y)

    @property
    def size(self) -> tuple[int, int]:
        """Get or set the pixel (width, height) of the window client area.

        This attribute can be set to change the size of the window but the given size must be greater than (1, 1) or
        else ValueError will be raised.
        """
        xy = ffi.new("int[2]")
        lib.SDL_GetWindowSize(self.p, xy, xy + 1)
        return xy[0], xy[1]

    @size.setter
    def size(self, xy: tuple[int, int]) -> None:
        if any(i <= 0 for i in xy):
            msg = f"Window size must be greater than zero, not {xy}"
            raise ValueError(msg)
        x, y = xy
        lib.SDL_SetWindowSize(self.p, x, y)

    @property
    def min_size(self) -> tuple[int, int]:
        """Get or set this windows minimum client area."""
        xy = ffi.new("int[2]")
        lib.SDL_GetWindowMinimumSize(self.p, xy, xy + 1)
        return xy[0], xy[1]

    @min_size.setter
    def min_size(self, xy: tuple[int, int]) -> None:
        lib.SDL_SetWindowMinimumSize(self.p, xy[0], xy[1])

    @property
    def max_size(self) -> tuple[int, int]:
        """Get or set this windows maximum client area."""
        xy = ffi.new("int[2]")
        lib.SDL_GetWindowMaximumSize(self.p, xy, xy + 1)
        return xy[0], xy[1]

    @max_size.setter
    def max_size(self, xy: tuple[int, int]) -> None:
        lib.SDL_SetWindowMaximumSize(self.p, xy[0], xy[1])

    @property
    def title(self) -> str:
        """Get or set the title of the window."""
        return str(ffi.string(lib.SDL_GetWindowTitle(self.p)), encoding="utf-8")

    @title.setter
    def title(self, value: str) -> None:
        lib.SDL_SetWindowTitle(self.p, value.encode("utf-8"))

    @property
    def flags(self) -> WindowFlags:
        """The current flags of this window, read-only."""
        return WindowFlags(lib.SDL_GetWindowFlags(self.p))

    @property
    def fullscreen(self) -> bool:
        """Get or set the fullscreen status of this window.

        Example::

            # Toggle fullscreen.
            window: tcod.sdl.video.Window
            window.fullscreen = not window.fullscreen
        """
        return bool(self.flags & WindowFlags.FULLSCREEN)

    @fullscreen.setter
    def fullscreen(self, value: bool) -> None:
        _check(lib.SDL_SetWindowFullscreen(self.p, value))

    @property
    def resizable(self) -> bool:
        """Get or set if this window can be resized."""
        return bool(self.flags & WindowFlags.RESIZABLE)

    @resizable.setter
    def resizable(self, value: bool) -> None:
        lib.SDL_SetWindowResizable(self.p, value)

    @property
    def border_size(self) -> tuple[int, int, int, int]:
        """Get the (top, left, bottom, right) size of the window decorations around the client area.

        If this fails or the window doesn't have decorations yet then the value will be (0, 0, 0, 0).

        .. seealso::
            https://wiki.libsdl.org/SDL_GetWindowBordersSize
        """
        borders = ffi.new("int[4]")
        # The return code is ignored.
        _ = lib.SDL_GetWindowBordersSize(self.p, borders, borders + 1, borders + 2, borders + 3)
        return borders[0], borders[1], borders[2], borders[3]

    @property
    def opacity(self) -> float:
        """Get or set this windows opacity.  0.0 is fully transparent and 1.0 is fully opaque.

        Will error if you try to set this and opacity isn't supported.
        """
        return float(lib.SDL_GetWindowOpacity(self.p))

    @opacity.setter
    def opacity(self, value: float) -> None:
        _check(lib.SDL_SetWindowOpacity(self.p, value))

    @property
    @deprecated("This attribute as been split into mouse_grab and keyboard_grab")
    def grab(self) -> bool:
        """Get or set this windows input grab mode.

        .. deprecated:: 19.0
            This attribute as been split into :any:`mouse_grab` and :any:`keyboard_grab`.
        """
        return self.mouse_grab

    @grab.setter
    def grab(self, value: bool) -> None:
        self.mouse_grab = value

    @property
    def mouse_grab(self) -> bool:
        """Get or set this windows mouse input grab mode.

        .. versionadded:: 19.0
        """
        return bool(lib.SDL_GetWindowMouseGrab(self.p))

    @mouse_grab.setter
    def mouse_grab(self, value: bool, /) -> None:
        lib.SDL_SetWindowMouseGrab(self.p, value)

    @property
    def keyboard_grab(self) -> bool:
        """Get or set this windows keyboard input grab mode.

        https://wiki.libsdl.org/SDL3/SDL_SetWindowKeyboardGrab

        .. versionadded:: 19.0
        """
        return bool(lib.SDL_GetWindowKeyboardGrab(self.p))

    @keyboard_grab.setter
    def keyboard_grab(self, value: bool, /) -> None:
        lib.SDL_SetWindowKeyboardGrab(self.p, value)

    @property
    def mouse_rect(self) -> tuple[int, int, int, int] | None:
        """Get or set the mouse confinement area when the window has mouse focus.

        Setting this will not automatically grab the cursor.

        .. versionadded:: 13.5
        """
        rect = lib.SDL_GetWindowMouseRect(self.p)
        return (rect.x, rect.y, rect.w, rect.h) if rect else None

    @mouse_rect.setter
    def mouse_rect(self, rect: tuple[int, int, int, int] | None) -> None:
        _check(lib.SDL_SetWindowMouseRect(self.p, (rect,) if rect else ffi.NULL))

    @_required_version((2, 0, 16))
    def flash(self, operation: FlashOperation = FlashOperation.UNTIL_FOCUSED) -> None:
        """Get the users attention."""
        _check(lib.SDL_FlashWindow(self.p, operation))

    def raise_window(self) -> None:
        """Raise the window and set input focus."""
        lib.SDL_RaiseWindow(self.p)

    def restore(self) -> None:
        """Restore a minimized or maximized window to its original size and position."""
        lib.SDL_RestoreWindow(self.p)

    def maximize(self) -> None:
        """Make the window as big as possible."""
        lib.SDL_MaximizeWindow(self.p)

    def minimize(self) -> None:
        """Minimize the window to an iconic state."""
        lib.SDL_MinimizeWindow(self.p)

    def show(self) -> None:
        """Show this window."""
        lib.SDL_ShowWindow(self.p)

    def hide(self) -> None:
        """Hide this window."""
        lib.SDL_HideWindow(self.p)

    @property
    def relative_mouse_mode(self) -> bool:
        """Enable or disable relative mouse mode which will lock and hide the mouse and only report mouse motion.

        .. seealso::
            :any:`tcod.sdl.mouse.capture`
            https://wiki.libsdl.org/SDL_SetWindowRelativeMouseMode
        """
        return bool(lib.SDL_GetWindowRelativeMouseMode(self.p))

    @relative_mouse_mode.setter
    def relative_mouse_mode(self, enable: bool, /) -> None:
        _check(lib.SDL_SetWindowRelativeMouseMode(self.p, enable))

    def start_text_input(
        self,
        *,
        type: TextInputType = TextInputType.TEXT,  # noqa: A002
        capitalization: Capitalization | None = None,
        autocorrect: bool = True,
        multiline: bool | None = None,
        android_type: int | None = None,
    ) -> None:
        """Start receiving text input events supporting Unicode. This may open an on-screen keyboard.

        This method is meant to be paired with :any:`set_text_input_area`.

        Args:
            type: Type of text being inputted, see :any:`TextInputType`
            capitalization: Capitalization hint, default is based on `type` given, see :any:`Capitalization`.
            autocorrect: Enable auto completion and auto correction.
            multiline: Allow multiple lines of text.
            android_type: Input type for Android, see SDL docs.

        .. seealso::
            :any:`stop_text_input`
            :any:`set_text_input_area`
            https://wiki.libsdl.org/SDL3/SDL_StartTextInputWithProperties

        .. versionadded:: 19.1
        """
        props = Properties()
        props[("SDL_PROP_TEXTINPUT_TYPE_NUMBER", int)] = int(type)
        if capitalization is not None:
            props[("SDL_PROP_TEXTINPUT_CAPITALIZATION_NUMBER", int)] = int(capitalization)
        props[("SDL_PROP_TEXTINPUT_AUTOCORRECT_BOOLEAN", bool)] = autocorrect
        if multiline is not None:
            props[("SDL_PROP_TEXTINPUT_MULTILINE_BOOLEAN", bool)] = multiline
        if android_type is not None:
            props[("SDL_PROP_TEXTINPUT_ANDROID_INPUTTYPE_NUMBER", int)] = int(android_type)
        _check(lib.SDL_StartTextInputWithProperties(self.p, props.p))

    def set_text_input_area(self, rect: tuple[int, int, int, int], cursor: int) -> None:
        """Assign the area used for entering Unicode text input.

        Args:
            rect: `(x, y, width, height)` rectangle used for text input
            cursor: Cursor X position, relative to `rect[0]`

        .. seealso::
            :any:`start_text_input`
            https://wiki.libsdl.org/SDL3/SDL_SetTextInputArea

        .. versionadded:: 19.1
        """
        _check(lib.SDL_SetTextInputArea(self.p, (rect,), cursor))

    def stop_text_input(self) -> None:
        """Stop receiving text events for this window and close relevant on-screen keyboards.

        .. seealso::
            :any:`start_text_input`

        .. versionadded:: 19.1
        """
        _check(lib.SDL_StopTextInput(self.p))


def new_window(  # noqa: PLR0913
    width: int,
    height: int,
    *,
    x: int | None = None,
    y: int | None = None,
    title: str | None = None,
    flags: int = 0,
) -> Window:
    """Initialize and return a new SDL Window.

    Args:
        width: The requested pixel width of the window.
        height: The requested pixel height of the window.
        x: The left-most position of the window.
        y: The top-most position of the window.
        title: The title text of the new window.  If no option is given then `sys.arg[0]` will be used as the title.
        flags: The SDL flags to use for this window, such as `tcod.sdl.video.WindowFlags.RESIZABLE`.
               See :any:`WindowFlags` for more options.

    Example::

        import tcod.sdl.video
        # Create a new resizable window with a custom title.
        window = tcod.sdl.video.new_window(640, 480, title="Title bar text", flags=tcod.sdl.video.WindowFlags.RESIZABLE)

    .. seealso::
        :func:`tcod.sdl.render.new_renderer`
    """
    if title is None:
        title = sys.argv[0]
    window_props = Properties()
    window_props[(tcod.sdl.constants.SDL_PROP_WINDOW_CREATE_FLAGS_NUMBER, int)] = flags
    window_props[(tcod.sdl.constants.SDL_PROP_WINDOW_CREATE_TITLE_STRING, str)] = title
    if x is not None:
        window_props[(tcod.sdl.constants.SDL_PROP_WINDOW_CREATE_X_NUMBER, int)] = x
    if y is not None:
        window_props[(tcod.sdl.constants.SDL_PROP_WINDOW_CREATE_Y_NUMBER, int)] = y
    window_props[(tcod.sdl.constants.SDL_PROP_WINDOW_CREATE_WIDTH_NUMBER, int)] = width
    window_props[(tcod.sdl.constants.SDL_PROP_WINDOW_CREATE_HEIGHT_NUMBER, int)] = height
    window_p = ffi.gc(lib.SDL_CreateWindowWithProperties(window_props.p), lib.SDL_DestroyWindow)
    return Window(_check_p(window_p))


def get_grabbed_window() -> Window | None:
    """Return the window which has input grab enabled, if any."""
    sdl_window_p = lib.SDL_GetGrabbedWindow()
    return Window(sdl_window_p) if sdl_window_p else None


def screen_saver_allowed(allow: bool | None = None) -> bool:
    """Allow or prevent a screen saver from being displayed and return the current allowed status.

    If `allow` is `None` then only the current state is returned.
    Otherwise it will change the state before checking it.

    SDL typically disables the screensaver by default.
    If you're unsure, then don't touch this.

    Example::

        import tcod.sdl.video

        print(f"Screen saver was allowed: {tcod.sdl.video.screen_saver_allowed()}")
        # Allow the screen saver.
        # Might be okay for some turn-based games which don't use a gamepad.
        tcod.sdl.video.screen_saver_allowed(True)
    """
    if allow is None:
        pass
    elif allow:
        lib.SDL_EnableScreenSaver()
    else:
        lib.SDL_DisableScreenSaver()
    return bool(lib.SDL_ScreenSaverEnabled())
