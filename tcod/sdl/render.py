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


class _RestoreTargetContext:
    """A context manager which tracks the current render target and restores it on exiting."""

    def __init__(self, renderer: Renderer) -> None:
        self.renderer = renderer
        self.old_texture_p = lib.SDL_GetRenderTarget(renderer.p)

    def __enter__(self) -> None:
        pass

    def __exit__(self, *_: Any) -> None:
        _check(lib.SDL_SetRenderTarget(self.renderer.p, self.old_texture_p))


class Renderer:
    def __init__(self, sdl_renderer_p: Any) -> None:
        if ffi.typeof(sdl_renderer_p) is not ffi.typeof("struct SDL_Renderer*"):
            raise TypeError(f"Expected a {ffi.typeof('struct SDL_Window*')} type (was {ffi.typeof(sdl_renderer_p)}).")
        self.p = sdl_renderer_p

    def __eq__(self, other: Any) -> bool:
        return bool(self.p == getattr(other, "p", None))

    def copy(
        self,
        texture: Texture,
        source: Optional[Tuple[int, int, int, int]] = None,
        dest: Optional[Tuple[int, int, int, int]] = None,
    ) -> None:
        """Copy a texture to the rendering target.

        `source` and `dest` are (x, y, width, height) regions of the texture parameter and target texture respectively.
        """
        source_ = ffi.NULL if source is None else ffi.new("SDL_Rect*", source)
        dest_ = ffi.NULL if dest is None else ffi.new("SDL_Rect*", dest)
        _check(lib.SDL_RenderCopy(self.p, texture.p, source_, dest_))

    def present(self) -> None:
        """Present the currently rendered image to the screen."""
        lib.SDL_RenderPresent(self.p)

    def new_texture(
        self, width: int, height: int, *, format: Optional[int] = None, access: Optional[int] = None
    ) -> Texture:
        """Allocate and return a new Texture for this renderer."""
        if format is None:
            format = 0
        if access is None:
            access = int(lib.SDL_TEXTUREACCESS_STATIC)
        texture_p = ffi.gc(lib.SDL_CreateTexture(self.p, format, access, width, height), lib.SDL_DestroyTexture)
        return Texture(texture_p, self.p)

    def set_render_target(self, texture: Texture) -> _RestoreTargetContext:
        """Change the render target to `texture`, returns a context that will restore the original target when exited."""
        restore = _RestoreTargetContext(self)
        _check(lib.SDL_SetRenderTarget(self.p, texture.p))
        return restore

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
                raise TypeError(f"Can't determine the format required for an array of shape {pixels.shape}.")

        texture = self.new_texture(pixels.shape[1], pixels.shape[0], format=format, access=access)
        if not pixels[0].flags["C_CONTIGUOUS"]:
            pixels = np.ascontiguousarray(pixels)
        _check(
            lib.SDL_UpdateTexture(texture.p, ffi.NULL, ffi.cast("const void*", pixels.ctypes.data), pixels.strides[0])
        )
        return texture
