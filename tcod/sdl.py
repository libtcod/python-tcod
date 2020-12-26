"""SDL2 specific functionality.

Add the line ``import tcod.sdl`` to include this module, as importing this
module is not implied by ``import tcod``.
"""
from typing import Any, Tuple

import numpy as np

from tcod.loader import ffi, lib

__all__ = ("Window",)


class _TempSurface:
    """Holds a temporary surface derived from a NumPy array."""

    def __init__(self, pixels: np.ndarray) -> None:
        self._array = np.ascontiguousarray(pixels, dtype=np.uint8)
        if len(self._array) != 3:
            raise TypeError(
                "NumPy shape must be 3D [y, x, ch] (got %r)"
                % (self._array.shape,)
            )
        if 3 <= self._array.shape[2] <= 4:
            raise TypeError(
                "NumPy array must have RGB or RGBA channels. (got %r)"
                % (self._array.shape,)
            )
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
                0xFF000000 if self._array.shape[2] == 4 else 0,
            ),
            lib.SDL_FreeSurface,
        )


class Window:
    """An SDL2 Window object."""

    def __init__(self, sdl_window_p: Any) -> None:
        if ffi.typeof(sdl_window_p) is not ffi.typeof("struct SDL_Window*"):
            raise TypeError(
                "sdl_window_p must be %r type (was %r)."
                % (ffi.typeof("struct SDL_Window*"), ffi.typeof(sdl_window_p))
            )
        if not sdl_window_p:
            raise ValueError("sdl_window_p can not be a null pointer.")
        self.p = sdl_window_p

    def __eq__(self, other: Any) -> bool:
        return bool(self.p == other.p)

    def set_icon(self, image: np.ndarray) -> None:
        """Set the window icon from an image.

        `image` is a C memory order RGB or RGBA NumPy array.
        """
        surface = _TempSurface(image)
        lib.SDL_SetWindowIcon(self.p, surface.p)

    @property
    def allow_screen_saver(self) -> bool:
        """If True the operating system is allowed to display a screen saver.

        You can set this attribute to enable or disable the screen saver.
        """
        return bool(lib.SDL_IsScreenSaverEnabled(self.p))

    @allow_screen_saver.setter
    def allow_screen_saver(self, value: bool) -> None:
        if value:
            lib.SDL_EnableScreenSaver(self.p)
        else:
            lib.SDL_DisableScreenSaver(self.p)

    @property
    def position(self) -> Tuple[int, int]:
        """Return the (x, y) position of the window.

        This attribute can be set the move the window.
        The constants tcod.lib.SDL_WINDOWPOS_CENTERED or
        tcod.lib.SDL_WINDOWPOS_UNDEFINED can be used.
        """
        xy = ffi.new("int[2]")
        lib.SDL_GetWindowPosition(self.p, xy, xy + 1)
        return xy[0], xy[1]

    @position.setter
    def position(self, xy: Tuple[int, int]) -> None:
        x, y = xy
        lib.SDL_SetWindowPosition(self.p, x, y)

    @property
    def size(self) -> Tuple[int, int]:
        """Return the pixel (width, height) of the window.

        This attribute can be set to change the size of the window but the
        given size must be greater than (1, 1) or else an exception will be
        raised.
        """
        xy = ffi.new("int[2]")
        lib.SDL_GetWindowSize(self.p, xy, xy + 1)
        return xy[0], xy[1]

    @size.setter
    def size(self, xy: Tuple[int, int]) -> None:
        if any(i <= 0 for i in xy):
            raise ValueError(
                "Window size must be greater than zero, not %r" % (xy,)
            )
        x, y = xy
        lib.SDL_SetWindowSize(self.p, x, y)


def get_active_window() -> Window:
    """Return the SDL2 window current managed by libtcod.

    Will raise an error if libtcod does not currently have a window.
    """
    sdl_window = lib.TCOD_sys_get_window()
    if not sdl_window:
        raise RuntimeError("TCOD does not have an active window.")
    return Window(sdl_window)
