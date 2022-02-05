"""SDL2 Rendering functionality.

.. versionadded:: 13.4
"""
from __future__ import annotations

import enum
from typing import Any, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

import tcod.sdl.video
from tcod.loader import ffi, lib
from tcod.sdl import _check, _check_p


class TextureAccess(enum.IntEnum):
    """Determines how a texture is expected to be used."""

    STATIC = lib.SDL_TEXTUREACCESS_STATIC or 0
    """Texture rarely changes."""
    STREAMING = lib.SDL_TEXTUREACCESS_STREAMING or 0
    """Texture frequently changes."""
    TARGET = lib.SDL_TEXTUREACCESS_TARGET or 0
    """Texture will be used as a render target."""


class Texture:
    """SDL hardware textures."""

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
        out = ffi.new("uint8_t*")
        _check(lib.SDL_GetTextureAlphaMod(self.p, out))
        return int(out[0])

    @alpha_mod.setter
    def alpha_mod(self, value: int) -> None:
        _check(lib.SDL_SetTextureAlphaMod(self.p, value))

    @property
    def blend_mode(self) -> int:
        """Texture blend mode, can be set."""
        out = ffi.new("SDL_BlendMode*")
        _check(lib.SDL_GetTextureBlendMode(self.p, out))
        return int(out[0])

    @blend_mode.setter
    def blend_mode(self, value: int) -> None:
        _check(lib.SDL_SetTextureBlendMode(self.p, value))

    @property
    def color_mod(self) -> Tuple[int, int, int]:
        """Texture RGB color modulate values, can be set."""
        rgb = ffi.new("uint8_t[3]")
        _check(lib.SDL_GetTextureColorMod(self.p, rgb, rgb + 1, rgb + 2))
        return int(rgb[0]), int(rgb[1]), int(rgb[2])

    @color_mod.setter
    def color_mod(self, rgb: Tuple[int, int, int]) -> None:
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
    """SDL Renderer."""

    def __init__(self, sdl_renderer_p: Any) -> None:
        if ffi.typeof(sdl_renderer_p) is not ffi.typeof("struct SDL_Renderer*"):
            raise TypeError(f"Expected a {ffi.typeof('struct SDL_Window*')} type (was {ffi.typeof(sdl_renderer_p)}).")
        if not sdl_renderer_p:
            raise TypeError("C pointer must not be null.")
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

    def set_render_target(self, texture: Texture) -> _RestoreTargetContext:
        """Change the render target to `texture`, returns a context that will restore the original target when exited."""
        restore = _RestoreTargetContext(self)
        _check(lib.SDL_SetRenderTarget(self.p, texture.p))
        return restore

    def new_texture(
        self, width: int, height: int, *, format: Optional[int] = None, access: Optional[int] = None
    ) -> Texture:
        """Allocate and return a new Texture for this renderer.

        Args:
            width: The pixel width of the new texture.
            height: The pixel height of the new texture.
            format: The format the new texture.
            access: The access mode of the texture.  Defaults to :any:`TextureAccess.STATIC`.
                    See :any:`TextureAccess` for more options.
        """
        if format is None:
            format = 0
        if access is None:
            access = int(lib.SDL_TEXTUREACCESS_STATIC)
        texture_p = ffi.gc(lib.SDL_CreateTexture(self.p, format, access, width, height), lib.SDL_DestroyTexture)
        return Texture(texture_p, self.p)

    def upload_texture(
        self, pixels: NDArray[Any], *, format: Optional[int] = None, access: Optional[int] = None
    ) -> Texture:
        """Return a new Texture from an array of pixels.

        Args:
            pixels: An RGB or RGBA array of pixels in row-major order.
            format: The format of `pixels` when it isn't a simple RGB or RGBA array.
            access: The access mode of the texture.  Defaults to :any:`TextureAccess.STATIC`.
                    See :any:`TextureAccess` for more options.
        """
        if format is None:
            assert len(pixels.shape) == 3
            assert pixels.dtype == np.uint8
            if pixels.shape[2] == 4:
                format = int(lib.SDL_PIXELFORMAT_RGBA32)
            elif pixels.shape[2] == 3:
                format = int(lib.SDL_PIXELFORMAT_RGB24)
            else:
                raise TypeError(f"Can't determine the format required for an array of shape {pixels.shape}.")

        texture = self.new_texture(pixels.shape[1], pixels.shape[0], format=format, access=access)
        if not pixels[0].flags["C_CONTIGUOUS"]:
            pixels = np.ascontiguousarray(pixels)
        _check(
            lib.SDL_UpdateTexture(texture.p, ffi.NULL, ffi.cast("const void*", pixels.ctypes.data), pixels.strides[0])
        )
        return texture


def new_renderer(
    window: tcod.sdl.video.Window,
    *,
    driver: Optional[int] = None,
    software: bool = False,
    vsync: bool = True,
    target_textures: bool = False,
) -> Renderer:
    """Initialize and return a new SDL Renderer.

    Args:
        window: The window that this renderer will be attached to.
        driver: Force SDL to use a specific video driver.
        software: If True then a software renderer will be forced.  By default a hardware renderer is used.
        vsync: If True then Vsync will be enabled.
        target_textures: If True then target textures can be used by the renderer.

    Example::

        # Start by creating a window.
        sdl_window = tcod.sdl.video.new_window(640, 480)
        # Create a renderer with target texture support.
        sdl_renderer = tcod.sdl.render.new_renderer(sdl_window, target_textures=True)

    .. seealso::
        :func:`tcod.sdl.video.new_window`
    """
    driver = driver if driver is not None else -1
    flags = 0
    if vsync:
        flags |= int(lib.SDL_RENDERER_PRESENTVSYNC)
    if target_textures:
        flags |= int(lib.SDL_RENDERER_TARGETTEXTURE)
    flags |= int(lib.SDL_RENDERER_SOFTWARE) if software else int(lib.SDL_RENDERER_ACCELERATED)
    renderer_p = _check_p(ffi.gc(lib.SDL_CreateRenderer(window.p, driver, flags), lib.SDL_DestroyRenderer))
    return Renderer(renderer_p)
