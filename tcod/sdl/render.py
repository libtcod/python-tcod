from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from tcod.loader import ffi, lib
from tcod.sdl import _check


class Texture:
    def __init__(self, sdl_texture_p: Any, sdl_renderer_p: Any = None) -> None:
        self.p = sdl_texture_p
        self._sdl_renderer_p = sdl_renderer_p  # Keep alive.

    def __eq__(self, other: Any) -> bool:
        return bool(self.p == getattr(other, "p", None))

    def _query(self) -> Tuple[int, int, int, int]:
        """Return (format, access, width, height)."""
        format = ffi.new("uint32_t*")
        buffer = ffi.new("int[3]")
        lib.SDL_QueryTexture(self.p, format, buffer, buffer + 1, buffer + 2)
        return int(format), int(buffer[0]), int(buffer[1]), int(buffer[2])

    @property
    def format(self) -> int:
        """Texture format, read only."""
        buffer = ffi.new("uint32_t*")
        lib.SDL_QueryTexture(self.p, buffer, ffi.NULL, ffi.NULL, ffi.NULL)
        return int(buffer[0])

    @property
    def access(self) -> int:
        """Texture access mode, read only."""
        buffer = ffi.new("int*")
        lib.SDL_QueryTexture(self.p, ffi.NULL, buffer, ffi.NULL, ffi.NULL)
        return int(buffer[0])

    @property
    def width(self) -> int:
        """Texture pixel width, read only."""
        buffer = ffi.new("int*")
        lib.SDL_QueryTexture(self.p, ffi.NULL, ffi.NULL, buffer, ffi.NULL)
        return int(buffer[0])

    @property
    def height(self) -> int:
        """Texture pixel height, read only."""
        buffer = ffi.new("int*")
        lib.SDL_QueryTexture(self.p, ffi.NULL, ffi.NULL, ffi.NULL, buffer)
        return int(buffer[0])

    @property
    def alpha_mod(self) -> int:
        """Texture alpha modulate value, can be set to: 0 - 255."""
        return int(lib.SDL_GetTextureAlphaMod(self.p))

    @alpha_mod.setter
    def alpha_mod(self, value: int) -> None:
        _check(lib.SDL_SetTextureAlphaMod(self.p, value))

    @property
    def blend_mode(self) -> int:
        """Texture blend mode, can be set."""
        return int(lib.SDL_GetTextureBlendMode(self.p))

    @blend_mode.setter
    def blend_mode(self, value: int) -> None:
        _check(lib.SDL_SetTextureBlendMode(self.p, value))

    @property
    def rgb_mod(self) -> Tuple[int, int, int]:
        """Texture RGB color modulate values, can be set."""
        rgb = ffi.new("uint8_t[3]")
        _check(lib.SDL_GetTextureColorMod(self.p, rgb, rgb + 1, rgb + 2))
        return int(rgb[0]), int(rgb[1]), int(rgb[2])

    @rgb_mod.setter
    def rgb_mod(self, rgb: Tuple[int, int, int]) -> None:
        _check(lib.SDL_SetTextureColorMod(self.p, rgb[0], rgb[1], rgb[2]))


class Renderer:
    def __init__(self, sdl_renderer_p: Any) -> None:
        if ffi.typeof(sdl_renderer_p) is not ffi.typeof("struct SDL_Renderer*"):
            raise TypeError(f"Expected a {ffi.typeof('struct SDL_Window*')} type (was {ffi.typeof(sdl_renderer_p)}).")
        self.p = sdl_renderer_p

    def __eq__(self, other: Any) -> bool:
        return bool(self.p == getattr(other, "p", None))

    def new_texture(
        self, width: int, height: int, *, format: Optional[int] = None, access: Optional[int] = None
    ) -> Texture:
        """Allocate and return a new Texture for this renderer."""
        if format is None:
            format = 0
        if access is None:
            access = int(lib.SDL_TEXTUREACCESS_STATIC)
        format = int(lib.SDL_PIXELFORMAT_RGBA32)
        access = int(lib.SDL_TEXTUREACCESS_STATIC)
        texture_p = ffi.gc(lib.SDL_CreateTexture(self.p, format, access, width, height), lib.SDL_DestroyTexture)
        return Texture(texture_p, self.p)

    def upload_texture(
        self, pixels: NDArray[Any], *, format: Optional[int] = None, access: Optional[int] = None
    ) -> Texture:
        """Return a new Texture from an array of pixels."""
        if format is None:
            assert len(pixels.shape) == 3
            assert pixels.dtype == np.uint8
            if pixels.shape[2] == 4:
                format = int(lib.SDL_PIXELFORMAT_RGBA32)
            elif pixels.shape[2] == 3:
                format = int(lib.SDL_PIXELFORMAT_RGB32)
            else:
                assert False

        texture = self.new_texture(pixels.shape[1], pixels.shape[0], format=format, access=access)
        if not pixels[0].flags["C_CONTIGUOUS"]:
            pixels = np.ascontiguousarray(pixels)
        _check(
            lib.SDL_UpdateTexture(texture.p, ffi.NULL, ffi.cast("const void*", pixels.ctypes.data), pixels.strides[0])
        )
        return texture
