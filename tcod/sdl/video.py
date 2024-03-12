"""SDL2 Window and Display handling.

There are two main ways to access the SDL window.
Either you can use this module to open a window yourself bypassing libtcod's context,
or you can use :any:`Context.sdl_window` to get the window being controlled by that context (if the context has one.)

.. versionadded:: 13.4
"""

from __future__ import annotations

import enum
import sys
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tcod.cffi import ffi, lib
from tcod.sdl._internal import _check, _check_p, _required_version, _version_at_least

__all__ = (
    "WindowFlags",
    "FlashOperation",
    "Window",
    "new_window",
    "get_grabbed_window",
    "screen_saver_allowed",
)


class WindowFlags(enum.IntFlag):
    """Bit flags which make up a windows state.

    .. seealso::
        https://wiki.libsdl.org/SDL_WindowFlags
    """

    FULLSCREEN = int(lib.SDL_WINDOW_FULLSCREEN)
    """"""
    FULLSCREEN_DESKTOP = int(lib.SDL_WINDOW_FULLSCREEN_DESKTOP)
    """"""
    OPENGL = int(lib.SDL_WINDOW_OPENGL)
    """"""
    SHOWN = int(lib.SDL_WINDOW_SHOWN)
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
    MOUSE_GRABBED = int(lib.SDL_WINDOW_INPUT_GRABBED)
    """"""
    INPUT_FOCUS = int(lib.SDL_WINDOW_INPUT_FOCUS)
    """"""
    MOUSE_FOCUS = int(lib.SDL_WINDOW_MOUSE_FOCUS)
    """"""
    FOREIGN = int(lib.SDL_WINDOW_FOREIGN)
    """"""
    ALLOW_HIGHDPI = int(lib.SDL_WINDOW_ALLOW_HIGHDPI)
    """"""
    MOUSE_CAPTURE = int(lib.SDL_WINDOW_MOUSE_CAPTURE)
    """"""
    ALWAYS_ON_TOP = int(lib.SDL_WINDOW_ALWAYS_ON_TOP)
    """"""
    SKIP_TASKBAR = int(lib.SDL_WINDOW_SKIP_TASKBAR)
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
            lib.SDL_CreateRGBSurfaceFrom(
                ffi.from_buffer("void*", self._array),
                self._array.shape[1],  # Width.
                self._array.shape[0],  # Height.
                self._array.shape[2] * 8,  # Bit depth.
                self._array.strides[1],  # Pitch.
                0x000000FF,
                0x0000FF00,
                0x00FF0000,
                0xFF000000 if self._array.shape[2] == 4 else 0,  # noqa: PLR2004
            ),
            lib.SDL_FreeSurface,
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

    def __eq__(self, other: Any) -> bool:
        return bool(self.p == other.p)

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
    def fullscreen(self) -> int:
        """Get or set the fullscreen status of this window.

        Can be set to the :any:`WindowFlags.FULLSCREEN` or :any:`WindowFlags.FULLSCREEN_DESKTOP` flags.

        Example::

            # Toggle fullscreen.
            window: tcod.sdl.video.Window
            if window.fullscreen:
                window.fullscreen = False  # Set windowed mode.
            else:
                window.fullscreen = tcod.sdl.video.WindowFlags.FULLSCREEN_DESKTOP
        """
        return self.flags & (WindowFlags.FULLSCREEN | WindowFlags.FULLSCREEN_DESKTOP)

    @fullscreen.setter
    def fullscreen(self, value: int) -> None:
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
        out = ffi.new("float*")
        _check(lib.SDL_GetWindowOpacity(self.p, out))
        return float(out[0])

    @opacity.setter
    def opacity(self, value: float) -> None:
        _check(lib.SDL_SetWindowOpacity(self.p, value))

    @property
    def grab(self) -> bool:
        """Get or set this windows input grab mode.

        .. seealso::
            https://wiki.libsdl.org/SDL_SetWindowGrab
        """
        return bool(lib.SDL_GetWindowGrab(self.p))

    @grab.setter
    def grab(self, value: bool) -> None:
        lib.SDL_SetWindowGrab(self.p, value)

    @property
    def mouse_rect(self) -> tuple[int, int, int, int] | None:
        """Get or set the mouse confinement area when the window has mouse focus.

        Setting this will not automatically grab the cursor.

        .. versionadded:: 13.5
        """
        _version_at_least((2, 0, 18))
        rect = lib.SDL_GetWindowMouseRect(self.p)
        return (rect.x, rect.y, rect.w, rect.h) if rect else None

    @mouse_rect.setter
    def mouse_rect(self, rect: tuple[int, int, int, int] | None) -> None:
        _version_at_least((2, 0, 18))
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
    x = x if x is not None else int(lib.SDL_WINDOWPOS_UNDEFINED)
    y = y if y is not None else int(lib.SDL_WINDOWPOS_UNDEFINED)
    if title is None:
        title = sys.argv[0]
    window_p = ffi.gc(lib.SDL_CreateWindow(title.encode("utf-8"), x, y, width, height, flags), lib.SDL_DestroyWindow)
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
    return bool(lib.SDL_IsScreenSaverEnabled())
