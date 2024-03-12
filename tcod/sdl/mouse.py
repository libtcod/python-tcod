"""SDL mouse and cursor functions.

You can use this module to move or capture the cursor.

You can also set the cursor icon to an OS-defined or custom icon.

.. versionadded:: 13.5
"""

from __future__ import annotations

import enum
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

import tcod.event
import tcod.sdl.video
from tcod.cffi import ffi, lib
from tcod.sdl._internal import _check, _check_p


class Cursor:
    """A cursor icon for use with :any:`set_cursor`."""

    def __init__(self, sdl_cursor_p: Any) -> None:  # noqa: ANN401
        if ffi.typeof(sdl_cursor_p) is not ffi.typeof("struct SDL_Cursor*"):
            msg = f"Expected a {ffi.typeof('struct SDL_Cursor*')} type (was {ffi.typeof(sdl_cursor_p)})."
            raise TypeError(msg)
        if not sdl_cursor_p:
            msg = "C pointer must not be null."
            raise TypeError(msg)
        self.p = sdl_cursor_p

    def __eq__(self, other: Any) -> bool:
        return bool(self.p == getattr(other, "p", None))

    @classmethod
    def _claim(cls, sdl_cursor_p: Any) -> Cursor:
        """Verify and wrap this pointer in a garbage collector before returning a Cursor."""
        return cls(ffi.gc(_check_p(sdl_cursor_p), lib.SDL_FreeCursor))


class SystemCursor(enum.IntEnum):
    """An enumerator of system cursor icons."""

    ARROW = 0
    """"""
    IBEAM = enum.auto()
    """"""
    WAIT = enum.auto()
    """"""
    CROSSHAIR = enum.auto()
    """"""
    WAITARROW = enum.auto()
    """"""
    SIZENWSE = enum.auto()
    """"""
    SIZENESW = enum.auto()
    """"""
    SIZEWE = enum.auto()
    """"""
    SIZENS = enum.auto()
    """"""
    SIZEALL = enum.auto()
    """"""
    NO = enum.auto()
    """"""
    HAND = enum.auto()
    """"""


def new_cursor(data: NDArray[np.bool_], mask: NDArray[np.bool_], hot_xy: tuple[int, int] = (0, 0)) -> Cursor:
    """Return a new non-color Cursor from the provided parameters.

    Args:
        data: A row-major boolean array for the data parameters.  See the SDL docs for more info.
        mask: A row-major boolean array for the mask parameters.  See the SDL docs for more info.
        hot_xy: The position of the pointer relative to the mouse sprite, starting from the upper-left at (0, 0).

    .. seealso::
        :any:`set_cursor`
        https://wiki.libsdl.org/SDL_CreateCursor
    """
    if len(data.shape) != 2:  # noqa: PLR2004
        msg = "Data and mask arrays must be 2D."
        raise TypeError(msg)
    if data.shape != mask.shape:
        msg = "Data and mask arrays must have the same shape."
        raise TypeError(msg)
    height, width = data.shape
    data_packed = np.packbits(data, axis=0, bitorder="big")
    mask_packed = np.packbits(mask, axis=0, bitorder="big")
    return Cursor._claim(
        lib.SDL_CreateCursor(
            ffi.from_buffer("uint8_t*", data_packed), ffi.from_buffer("uint8_t*", mask_packed), width, height, *hot_xy
        )
    )


def new_color_cursor(pixels: ArrayLike, hot_xy: tuple[int, int]) -> Cursor:
    """Create a new color cursor.

    Args:
        pixels: A row-major array of RGB or RGBA pixels.
        hot_xy: The position of the pointer relative to the mouse sprite, starting from the upper-left at (0, 0).

    .. seealso::
        :any:`set_cursor`
    """
    surface = tcod.sdl.video._TempSurface(pixels)
    return Cursor._claim(lib.SDL_CreateColorCursor(surface.p, *hot_xy))


def new_system_cursor(cursor: SystemCursor) -> Cursor:
    """Return a new Cursor from one of the system cursors labeled by SystemCursor.

    .. seealso::
        :any:`set_cursor`
    """
    return Cursor._claim(lib.SDL_CreateSystemCursor(cursor))


def set_cursor(cursor: Cursor | SystemCursor | None) -> None:
    """Change the active cursor to the one provided.

    Args:
        cursor: A cursor created from :any:`new_cursor`, :any:`new_color_cursor`, or :any:`new_system_cursor`.
                Can also take values of :any:`SystemCursor` directly.
                None will force the current cursor to be redrawn.
    """
    if isinstance(cursor, SystemCursor):
        cursor = new_system_cursor(cursor)
    lib.SDL_SetCursor(cursor.p if cursor is not None else ffi.NULL)


def get_default_cursor() -> Cursor:
    """Return the default cursor."""
    return Cursor(_check_p(lib.SDL_GetDefaultCursor()))


def get_cursor() -> Cursor | None:
    """Return the active cursor, or None if these is no mouse."""
    cursor_p = lib.SDL_GetCursor()
    return Cursor(cursor_p) if cursor_p else None


def capture(enable: bool) -> None:
    """Enable or disable mouse capture to track the mouse outside of a window.

    It is highly recommended to read the related remarks section in the SDL docs before using this.

    Example::

        # Make mouse button presses capture the mouse until all buttons are released.
        # This means that dragging the mouse outside of the window will not cause an interruption in motion events.
        for event in tcod.event.get():
            match event:
                case tcod.event.MouseButtonDown(button=button, pixel=pixel):  # Clicking the window captures the mouse.
                    tcod.sdl.mouse.capture(True)
                case tcod.event.MouseButtonUp():  # When all buttons are released then the mouse is released.
                    if tcod.event.mouse.get_global_state().state == 0:
                        tcod.sdl.mouse.capture(False)
                case tcod.event.MouseMotion(pixel=pixel, pixel_motion=pixel_motion, state=state):
                    pass  # While a button is held this event is still captured outside of the window.

    .. seealso::
        :any:`tcod.sdl.mouse.set_relative_mode`
        https://wiki.libsdl.org/SDL_CaptureMouse
    """
    _check(lib.SDL_CaptureMouse(enable))


def set_relative_mode(enable: bool) -> None:
    """Enable or disable relative mouse mode which will lock and hide the mouse and only report mouse motion.

    .. seealso::
        :any:`tcod.sdl.mouse.capture`
        https://wiki.libsdl.org/SDL_SetRelativeMouseMode
    """
    _check(lib.SDL_SetRelativeMouseMode(enable))


def get_relative_mode() -> bool:
    """Return True if relative mouse mode is enabled."""
    return bool(lib.SDL_GetRelativeMouseMode())


def get_global_state() -> tcod.event.MouseState:
    """Return the mouse state relative to the desktop.

    .. seealso::
        https://wiki.libsdl.org/SDL_GetGlobalMouseState
    """
    xy = ffi.new("int[2]")
    state = lib.SDL_GetGlobalMouseState(xy, xy + 1)
    return tcod.event.MouseState((xy[0], xy[1]), state=state)


def get_relative_state() -> tcod.event.MouseState:
    """Return the mouse state, the coordinates are relative to the last time this function was called.

    .. seealso::
        https://wiki.libsdl.org/SDL_GetRelativeMouseState
    """
    xy = ffi.new("int[2]")
    state = lib.SDL_GetRelativeMouseState(xy, xy + 1)
    return tcod.event.MouseState((xy[0], xy[1]), state=state)


def get_state() -> tcod.event.MouseState:
    """Return the mouse state relative to the window with mouse focus.

    .. seealso::
        https://wiki.libsdl.org/SDL_GetMouseState
    """
    xy = ffi.new("int[2]")
    state = lib.SDL_GetMouseState(xy, xy + 1)
    return tcod.event.MouseState((xy[0], xy[1]), state=state)


def get_focus() -> tcod.sdl.video.Window | None:
    """Return the window which currently has mouse focus."""
    window_p = lib.SDL_GetMouseFocus()
    return tcod.sdl.video.Window(window_p) if window_p else None


def warp_global(x: int, y: int) -> None:
    """Move the mouse cursor to a position on the desktop."""
    _check(lib.SDL_WarpMouseGlobal(x, y))


def warp_in_window(window: tcod.sdl.video.Window, x: int, y: int) -> None:
    """Move the mouse cursor to a position within a window."""
    lib.SDL_WarpMouseInWindow(window.p, x, y)


def show(visible: bool | None = None) -> bool:
    """Optionally show or hide the mouse cursor then return the state of the cursor.

    Args:
        visible: If None then only return the current state. Otherwise set the mouse visibility.

    Returns:
        True if the cursor is visible.

    .. versionadded:: 16.0
    """
    _OPTIONS = {None: lib.SDL_QUERY, False: lib.SDL_DISABLE, True: lib.SDL_ENABLE}
    return _check(lib.SDL_ShowCursor(_OPTIONS[visible])) == int(lib.SDL_ENABLE)
