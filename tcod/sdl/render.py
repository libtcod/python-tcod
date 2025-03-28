"""SDL2 Rendering functionality.

.. versionadded:: 13.4
"""

from __future__ import annotations

import enum
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Literal

import tcod.sdl.video
from tcod.cffi import ffi, lib
from tcod.sdl._internal import _check, _check_p, _required_version


class TextureAccess(enum.IntEnum):
    """Determines how a texture is expected to be used."""

    STATIC = 0
    """Texture rarely changes."""
    STREAMING = 1
    """Texture frequently changes."""
    TARGET = 2
    """Texture will be used as a render target."""


class RendererFlip(enum.IntFlag):
    """Flip parameter for :any:`Renderer.copy`."""

    NONE = 0
    """Default value, no flip."""
    HORIZONTAL = 1
    """Flip the image horizontally."""
    VERTICAL = 2
    """Flip the image vertically."""


class BlendFactor(enum.IntEnum):
    """SDL blend factors.

    .. seealso::
        :any:`compose_blend_mode`
        https://wiki.libsdl.org/SDL_BlendFactor

    .. versionadded:: 13.5
    """

    ZERO = 0x1
    """"""
    ONE = 0x2
    """"""
    SRC_COLOR = 0x3
    """"""
    ONE_MINUS_SRC_COLOR = 0x4
    """"""
    SRC_ALPHA = 0x5
    """"""
    ONE_MINUS_SRC_ALPHA = 0x6
    """"""
    DST_COLOR = 0x7
    """"""
    ONE_MINUS_DST_COLOR = 0x8
    """"""
    DST_ALPHA = 0x9
    """"""
    ONE_MINUS_DST_ALPHA = 0xA
    """"""


class BlendOperation(enum.IntEnum):
    """SDL blend operations.

    .. seealso::
        :any:`compose_blend_mode`
        https://wiki.libsdl.org/SDL_BlendOperation

    .. versionadded:: 13.5
    """

    ADD = 0x1
    """dest + source"""
    SUBTRACT = 0x2
    """dest - source"""
    REV_SUBTRACT = 0x3
    """source - dest"""
    MINIMUM = 0x4
    """min(dest, source)"""
    MAXIMUM = 0x5
    """max(dest, source)"""


class BlendMode(enum.IntEnum):
    """SDL blend modes.

    .. seealso::
        :any:`Texture.blend_mode`
        :any:`Renderer.draw_blend_mode`
        :any:`compose_blend_mode`

    .. versionadded:: 13.5
    """

    NONE = 0x00000000
    """"""
    BLEND = 0x00000001
    """"""
    ADD = 0x00000002
    """"""
    MOD = 0x00000004
    """"""
    INVALID = 0x7FFFFFFF
    """"""


def compose_blend_mode(  # noqa: PLR0913
    source_color_factor: BlendFactor,
    dest_color_factor: BlendFactor,
    color_operation: BlendOperation,
    source_alpha_factor: BlendFactor,
    dest_alpha_factor: BlendFactor,
    alpha_operation: BlendOperation,
) -> BlendMode:
    """Return a custom blend mode composed of the given factors and operations.

    .. seealso::
        https://wiki.libsdl.org/SDL_ComposeCustomBlendMode

    .. versionadded:: 13.5
    """
    return BlendMode(
        lib.SDL_ComposeCustomBlendMode(
            source_color_factor,
            dest_color_factor,
            color_operation,
            source_alpha_factor,
            dest_alpha_factor,
            alpha_operation,
        )
    )


class Texture:
    """SDL hardware textures.

    Create a new texture using :any:`Renderer.new_texture` or :any:`Renderer.upload_texture`.
    """

    def __init__(self, sdl_texture_p: Any, sdl_renderer_p: Any = None) -> None:
        """Encapsulate an SDL_Texture pointer. This function is private."""
        self.p = sdl_texture_p
        self._sdl_renderer_p = sdl_renderer_p  # Keep alive.
        query = self._query()
        self.format: Final[int] = query[0]
        """Texture format, read only."""
        self.access: Final[TextureAccess] = TextureAccess(query[1])
        """Texture access mode, read only.

        .. versionchanged:: 13.5
            Attribute is now a :any:`TextureAccess` value.
        """
        self.width: Final[int] = query[2]
        """Texture pixel width, read only."""
        self.height: Final[int] = query[3]
        """Texture pixel height, read only."""

    def __eq__(self, other: object) -> bool:
        """Return True if compared to the same texture."""
        if isinstance(other, Texture):
            return bool(self.p == other.p)
        return NotImplemented

    def _query(self) -> tuple[int, int, int, int]:
        """Return (format, access, width, height)."""
        format = ffi.new("uint32_t*")
        buffer = ffi.new("int[3]")
        lib.SDL_QueryTexture(self.p, format, buffer, buffer + 1, buffer + 2)
        return int(format[0]), int(buffer[0]), int(buffer[1]), int(buffer[2])

    def update(self, pixels: NDArray[Any], rect: tuple[int, int, int, int] | None = None) -> None:
        """Update the pixel data of this texture.

        .. versionadded:: 13.5
        """
        if rect is None:
            rect = (0, 0, self.width, self.height)
        assert pixels.shape[:2] == (self.height, self.width)
        if not pixels[0].flags.c_contiguous:
            pixels = np.ascontiguousarray(pixels)
        _check(lib.SDL_UpdateTexture(self.p, (rect,), ffi.cast("void*", pixels.ctypes.data), pixels.strides[0]))

    @property
    def alpha_mod(self) -> int:
        """Texture alpha modulate value, can be set to 0 - 255."""
        out = ffi.new("uint8_t*")
        _check(lib.SDL_GetTextureAlphaMod(self.p, out))
        return int(out[0])

    @alpha_mod.setter
    def alpha_mod(self, value: int) -> None:
        _check(lib.SDL_SetTextureAlphaMod(self.p, value))

    @property
    def blend_mode(self) -> BlendMode:
        """Texture blend mode, can be set.

        .. versionchanged:: 13.5
            Property now returns a BlendMode instance.
        """
        out = ffi.new("SDL_BlendMode*")
        _check(lib.SDL_GetTextureBlendMode(self.p, out))
        return BlendMode(out[0])

    @blend_mode.setter
    def blend_mode(self, value: int) -> None:
        _check(lib.SDL_SetTextureBlendMode(self.p, value))

    @property
    def color_mod(self) -> tuple[int, int, int]:
        """Texture RGB color modulate values, can be set."""
        rgb = ffi.new("uint8_t[3]")
        _check(lib.SDL_GetTextureColorMod(self.p, rgb, rgb + 1, rgb + 2))
        return int(rgb[0]), int(rgb[1]), int(rgb[2])

    @color_mod.setter
    def color_mod(self, rgb: tuple[int, int, int]) -> None:
        _check(lib.SDL_SetTextureColorMod(self.p, rgb[0], rgb[1], rgb[2]))


class _RestoreTargetContext:
    """A context manager which tracks the current render target and restores it on exiting."""

    def __init__(self, renderer: Renderer) -> None:
        self.renderer = renderer
        self.old_texture_p = lib.SDL_GetRenderTarget(renderer.p)

    def __enter__(self) -> None:
        pass

    def __exit__(self, *_: object) -> None:
        _check(lib.SDL_SetRenderTarget(self.renderer.p, self.old_texture_p))


class Renderer:
    """SDL Renderer."""

    def __init__(self, sdl_renderer_p: Any) -> None:
        """Encapsulate an SDL_Renderer pointer. This function is private."""
        if ffi.typeof(sdl_renderer_p) is not ffi.typeof("struct SDL_Renderer*"):
            msg = f"Expected a {ffi.typeof('struct SDL_Window*')} type (was {ffi.typeof(sdl_renderer_p)})."
            raise TypeError(msg)
        if not sdl_renderer_p:
            msg = "C pointer must not be null."
            raise TypeError(msg)
        self.p = sdl_renderer_p

    def __eq__(self, other: object) -> bool:
        """Return True if compared to the same renderer."""
        if isinstance(other, Renderer):
            return bool(self.p == other.p)
        return NotImplemented

    def copy(  # noqa: PLR0913
        self,
        texture: Texture,
        source: tuple[float, float, float, float] | None = None,
        dest: tuple[float, float, float, float] | None = None,
        angle: float = 0,
        center: tuple[float, float] | None = None,
        flip: RendererFlip = RendererFlip.NONE,
    ) -> None:
        """Copy a texture to the rendering target.

        Args:
            texture: The texture to copy onto the current texture target.
            source: The (x, y, width, height) region of `texture` to copy.  If None then the entire texture is copied.
            dest: The (x, y, width, height) region of the target.  If None then the entire target is drawn over.
            angle: The angle in degrees to rotate the image clockwise.
            center: The (x, y) point where rotation is applied.  If None then the center of `dest` is used.
            flip: Flips the `texture` when drawing it.

        .. versionchanged:: 13.5
            `source` and `dest` can now be float tuples.
            Added the `angle`, `center`, and `flip` parameters.
        """
        _check(
            lib.SDL_RenderCopyExF(
                self.p,
                texture.p,
                (source,) if source is not None else ffi.NULL,
                (dest,) if dest is not None else ffi.NULL,
                angle,
                (center,) if center is not None else ffi.NULL,
                flip,
            )
        )

    def present(self) -> None:
        """Present the currently rendered image to the screen."""
        lib.SDL_RenderPresent(self.p)

    def set_render_target(self, texture: Texture) -> _RestoreTargetContext:
        """Change the render target to `texture`, returns a context that will restore the original target when exited."""
        restore = _RestoreTargetContext(self)
        _check(lib.SDL_SetRenderTarget(self.p, texture.p))
        return restore

    def new_texture(self, width: int, height: int, *, format: int | None = None, access: int | None = None) -> Texture:
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

    def upload_texture(self, pixels: NDArray[Any], *, format: int | None = None, access: int | None = None) -> Texture:
        """Return a new Texture from an array of pixels.

        Args:
            pixels: An RGB or RGBA array of pixels in row-major order.
            format: The format of `pixels` when it isn't a simple RGB or RGBA array.
            access: The access mode of the texture.  Defaults to :any:`TextureAccess.STATIC`.
                    See :any:`TextureAccess` for more options.
        """
        if format is None:
            assert len(pixels.shape) == 3  # noqa: PLR2004
            assert pixels.dtype == np.uint8
            if pixels.shape[2] == 4:  # noqa: PLR2004
                format = int(lib.SDL_PIXELFORMAT_RGBA32)
            elif pixels.shape[2] == 3:  # noqa: PLR2004
                format = int(lib.SDL_PIXELFORMAT_RGB24)
            else:
                msg = f"Can't determine the format required for an array of shape {pixels.shape}."
                raise TypeError(msg)

        texture = self.new_texture(pixels.shape[1], pixels.shape[0], format=format, access=access)
        if not pixels[0].flags["C_CONTIGUOUS"]:
            pixels = np.ascontiguousarray(pixels)
        _check(
            lib.SDL_UpdateTexture(texture.p, ffi.NULL, ffi.cast("const void*", pixels.ctypes.data), pixels.strides[0])
        )
        return texture

    @property
    def draw_color(self) -> tuple[int, int, int, int]:
        """Get or set the active RGBA draw color for this renderer.

        .. versionadded:: 13.5
        """
        rgba = ffi.new("uint8_t[4]")
        _check(lib.SDL_GetRenderDrawColor(self.p, rgba, rgba + 1, rgba + 2, rgba + 3))
        return tuple(rgba)

    @draw_color.setter
    def draw_color(self, rgba: tuple[int, int, int, int]) -> None:
        _check(lib.SDL_SetRenderDrawColor(self.p, *rgba))

    @property
    def draw_blend_mode(self) -> BlendMode:
        """Get or set the active blend mode of this renderer.

        .. versionadded:: 13.5
        """
        out = ffi.new("SDL_BlendMode*")
        _check(lib.SDL_GetRenderDrawBlendMode(self.p, out))
        return BlendMode(out[0])

    @draw_blend_mode.setter
    def draw_blend_mode(self, value: int) -> None:
        _check(lib.SDL_SetRenderDrawBlendMode(self.p, value))

    @property
    def output_size(self) -> tuple[int, int]:
        """Get the (width, height) pixel resolution of the rendering context.

        .. seealso::
            https://wiki.libsdl.org/SDL_GetRendererOutputSize

        .. versionadded:: 13.5
        """
        out = ffi.new("int[2]")
        _check(lib.SDL_GetRendererOutputSize(self.p, out, out + 1))
        return out[0], out[1]

    @property
    def clip_rect(self) -> tuple[int, int, int, int] | None:
        """Get or set the clipping rectangle of this renderer.

        Set to None to disable clipping.

        .. versionadded:: 13.5
        """
        if not lib.SDL_RenderIsClipEnabled(self.p):
            return None
        rect = ffi.new("SDL_Rect*")
        lib.SDL_RenderGetClipRect(self.p, rect)
        return rect.x, rect.y, rect.w, rect.h

    @clip_rect.setter
    def clip_rect(self, rect: tuple[int, int, int, int] | None) -> None:
        rect_p = ffi.NULL if rect is None else ffi.new("SDL_Rect*", rect)
        _check(lib.SDL_RenderSetClipRect(self.p, rect_p))

    @property
    def integer_scaling(self) -> bool:
        """Get or set if this renderer enforces integer scaling.

        .. seealso::
            https://wiki.libsdl.org/SDL_RenderSetIntegerScale

        .. versionadded:: 13.5
        """
        return bool(lib.SDL_RenderGetIntegerScale(self.p))

    @integer_scaling.setter
    def integer_scaling(self, enable: bool) -> None:
        _check(lib.SDL_RenderSetIntegerScale(self.p, enable))

    @property
    def logical_size(self) -> tuple[int, int]:
        """Get or set a device independent (width, height) resolution.

        Might be (0, 0) if a resolution was never assigned.

        .. seealso::
            https://wiki.libsdl.org/SDL_RenderSetLogicalSize

        .. versionadded:: 13.5
        """
        out = ffi.new("int[2]")
        lib.SDL_RenderGetLogicalSize(self.p, out, out + 1)
        return out[0], out[1]

    @logical_size.setter
    def logical_size(self, size: tuple[int, int]) -> None:
        _check(lib.SDL_RenderSetLogicalSize(self.p, *size))

    @property
    def scale(self) -> tuple[float, float]:
        """Get or set an (x_scale, y_scale) multiplier for drawing.

        .. seealso::
            https://wiki.libsdl.org/SDL_RenderSetScale

        .. versionadded:: 13.5
        """
        out = ffi.new("float[2]")
        lib.SDL_RenderGetScale(self.p, out, out + 1)
        return out[0], out[1]

    @scale.setter
    def scale(self, scale: tuple[float, float]) -> None:
        _check(lib.SDL_RenderSetScale(self.p, *scale))

    @property
    def viewport(self) -> tuple[int, int, int, int] | None:
        """Get or set the drawing area for the current rendering target.

        .. seealso::
            https://wiki.libsdl.org/SDL_RenderSetViewport

        .. versionadded:: 13.5
        """
        rect = ffi.new("SDL_Rect*")
        lib.SDL_RenderGetViewport(self.p, rect)
        return rect.x, rect.y, rect.w, rect.h

    @viewport.setter
    def viewport(self, rect: tuple[int, int, int, int] | None) -> None:
        _check(lib.SDL_RenderSetViewport(self.p, (rect,)))

    @_required_version((2, 0, 18))
    def set_vsync(self, enable: bool) -> None:
        """Enable or disable VSync for this renderer.

        .. versionadded:: 13.5
        """
        _check(lib.SDL_RenderSetVSync(self.p, enable))

    def read_pixels(
        self,
        *,
        rect: tuple[int, int, int, int] | None = None,
        format: int | Literal["RGB", "RGBA"] = "RGBA",
        out: NDArray[np.uint8] | None = None,
    ) -> NDArray[np.uint8]:
        """Fetch the pixel contents of the current rendering target to an array.

        By default returns an RGBA pixel array of the full target in the shape: ``(height, width, rgba)``.
        The target can be changed with :any:`set_render_target`

        Args:
            rect: The ``(left, top, width, height)`` region of the target to fetch, or None for the entire target.
            format: The pixel format.  Defaults to ``"RGBA"``.
            out: The output array.
                Can be None or must be an ``np.uint8`` array of shape: ``(height, width, channels)``.
                Must be C contiguous along the ``(width, channels)`` axes.

        This operation is slow due to coping from VRAM to RAM.
        When reading the main rendering target this should be called after rendering and before :any:`present`.
        See https://wiki.libsdl.org/SDL2/SDL_RenderReadPixels

        Returns:
            The output uint8 array of shape: ``(height, width, channels)`` with the fetched pixels.

        .. versionadded:: 15.0
        """
        FORMATS: Final = {"RGB": lib.SDL_PIXELFORMAT_RGB24, "RGBA": lib.SDL_PIXELFORMAT_RGBA32}
        sdl_format = FORMATS.get(format) if isinstance(format, str) else format
        if rect is None:
            texture_p = lib.SDL_GetRenderTarget(self.p)
            if texture_p:
                texture = Texture(texture_p)
                rect = (0, 0, texture.width, texture.height)
            else:
                rect = (0, 0, *self.output_size)
        width, height = rect[2:4]
        if out is None:
            if sdl_format == lib.SDL_PIXELFORMAT_RGBA32:
                out = np.empty((height, width, 4), dtype=np.uint8)
            elif sdl_format == lib.SDL_PIXELFORMAT_RGB24:
                out = np.empty((height, width, 3), dtype=np.uint8)
            else:
                msg = f"Pixel format {format!r} not supported by tcod."
                raise TypeError(msg)
        if out.dtype != np.uint8:
            msg = "`out` must be a uint8 array."
            raise TypeError(msg)
        expected_shape = (height, width, {lib.SDL_PIXELFORMAT_RGB24: 3, lib.SDL_PIXELFORMAT_RGBA32: 4}[sdl_format])
        if out.shape != expected_shape:
            msg = f"Expected `out` to be an array of shape {expected_shape}, got {out.shape} instead."
            raise TypeError(msg)
        if not out[0].flags.c_contiguous:
            msg = "`out` array must be C contiguous."
        _check(
            lib.SDL_RenderReadPixels(
                self.p,
                (rect,),
                sdl_format,
                ffi.cast("void*", out.ctypes.data),
                out.strides[0],
            )
        )
        return out

    def clear(self) -> None:
        """Clear the current render target with :any:`draw_color`.

        .. versionadded:: 13.5
        """
        _check(lib.SDL_RenderClear(self.p))

    def fill_rect(self, rect: tuple[float, float, float, float]) -> None:
        """Fill a rectangle with :any:`draw_color`.

        .. versionadded:: 13.5
        """
        _check(lib.SDL_RenderFillRectF(self.p, (rect,)))

    def draw_rect(self, rect: tuple[float, float, float, float]) -> None:
        """Draw a rectangle outline.

        .. versionadded:: 13.5
        """
        _check(lib.SDL_RenderDrawRectF(self.p, (rect,)))

    def draw_point(self, xy: tuple[float, float]) -> None:
        """Draw a point.

        .. versionadded:: 13.5
        """
        x, y = xy
        _check(lib.SDL_RenderDrawPointF(self.p, x, y))

    def draw_line(self, start: tuple[float, float], end: tuple[float, float]) -> None:
        """Draw a single line.

        .. versionadded:: 13.5
        """
        x1, y1 = start
        x2, y2 = end
        _check(lib.SDL_RenderDrawLineF(self.p, x1, y1, x2, y2))

    @staticmethod
    def _convert_array(array: NDArray[np.number]) -> NDArray[np.intc] | NDArray[np.float32]:
        """Convert ndarray for a SDL function expecting a C contiguous array of either intc or float32."""
        if array.dtype in (np.intc, np.int8, np.int16, np.int32, np.uint8, np.uint16):
            return np.ascontiguousarray(array, np.intc)
        return np.ascontiguousarray(array, np.float32)

    def fill_rects(self, rects: NDArray[np.number]) -> None:
        """Fill multiple rectangles from an array.

        .. versionadded:: 13.5
        """
        assert len(rects.shape) == 2  # noqa: PLR2004
        assert rects.shape[1] == 4  # noqa: PLR2004
        rects = self._convert_array(rects)
        if rects.dtype == np.intc:
            _check(lib.SDL_RenderFillRects(self.p, tcod.ffi.from_buffer("SDL_Rect*", rects), rects.shape[0]))
            return
        _check(lib.SDL_RenderFillRectsF(self.p, tcod.ffi.from_buffer("SDL_FRect*", rects), rects.shape[0]))

    def draw_rects(self, rects: NDArray[np.number]) -> None:
        """Draw multiple outlined rectangles from an array.

        .. versionadded:: 13.5
        """
        assert len(rects.shape) == 2  # noqa: PLR2004
        assert rects.shape[1] == 4  # noqa: PLR2004
        rects = self._convert_array(rects)
        if rects.dtype == np.intc:
            _check(lib.SDL_RenderDrawRects(self.p, tcod.ffi.from_buffer("SDL_Rect*", rects), rects.shape[0]))
            return
        _check(lib.SDL_RenderDrawRectsF(self.p, tcod.ffi.from_buffer("SDL_FRect*", rects), rects.shape[0]))

    def draw_points(self, points: NDArray[np.number]) -> None:
        """Draw an array of points.

        .. versionadded:: 13.5
        """
        assert len(points.shape) == 2  # noqa: PLR2004
        assert points.shape[1] == 2  # noqa: PLR2004
        points = self._convert_array(points)
        if points.dtype == np.intc:
            _check(lib.SDL_RenderDrawPoints(self.p, tcod.ffi.from_buffer("SDL_Point*", points), points.shape[0]))
            return
        _check(lib.SDL_RenderDrawPointsF(self.p, tcod.ffi.from_buffer("SDL_FPoint*", points), points.shape[0]))

    def draw_lines(self, points: NDArray[np.intc | np.float32]) -> None:
        """Draw a connected series of lines from an array.

        .. versionadded:: 13.5
        """
        assert len(points.shape) == 2  # noqa: PLR2004
        assert points.shape[1] == 2  # noqa: PLR2004
        points = self._convert_array(points)
        if points.dtype == np.intc:
            _check(lib.SDL_RenderDrawLines(self.p, tcod.ffi.from_buffer("SDL_Point*", points), points.shape[0] - 1))
            return
        _check(lib.SDL_RenderDrawLinesF(self.p, tcod.ffi.from_buffer("SDL_FPoint*", points), points.shape[0] - 1))

    @_required_version((2, 0, 18))
    def geometry(
        self,
        texture: Texture | None,
        xy: NDArray[np.float32],
        color: NDArray[np.uint8],
        uv: NDArray[np.float32],
        indices: NDArray[np.uint8 | np.uint16 | np.uint32] | None = None,
    ) -> None:
        """Render triangles from texture and vertex data.

        .. versionadded:: 13.5
        """
        assert xy.dtype == np.float32
        assert len(xy.shape) == 2  # noqa: PLR2004
        assert xy.shape[1] == 2  # noqa: PLR2004
        assert xy[0].flags.c_contiguous

        assert color.dtype == np.uint8
        assert len(color.shape) == 2  # noqa: PLR2004
        assert color.shape[1] == 4  # noqa: PLR2004
        assert color[0].flags.c_contiguous

        assert uv.dtype == np.float32
        assert len(uv.shape) == 2  # noqa: PLR2004
        assert uv.shape[1] == 2  # noqa: PLR2004
        assert uv[0].flags.c_contiguous
        if indices is not None:
            assert indices.dtype.type in (np.uint8, np.uint16, np.uint32, np.int8, np.int16, np.int32)
            indices = np.ascontiguousarray(indices)
            assert len(indices.shape) == 1
        assert xy.shape[0] == color.shape[0] == uv.shape[0]
        _check(
            lib.SDL_RenderGeometryRaw(
                self.p,
                texture.p if texture else ffi.NULL,
                ffi.cast("float*", xy.ctypes.data),
                xy.strides[0],
                ffi.cast("uint8_t*", color.ctypes.data),
                color.strides[0],
                ffi.cast("float*", uv.ctypes.data),
                uv.strides[0],
                xy.shape[0],  # Number of vertices.
                ffi.cast("void*", indices.ctypes.data) if indices is not None else ffi.NULL,
                indices.size if indices is not None else 0,
                indices.itemsize if indices is not None else 0,
            )
        )


def new_renderer(
    window: tcod.sdl.video.Window,
    *,
    driver: int | None = None,
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
