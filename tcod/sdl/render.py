"""SDL2 Rendering functionality.

.. versionadded:: 13.4
"""

from __future__ import annotations

import enum
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Final, Literal

import numpy as np
from typing_extensions import deprecated

import tcod.sdl.constants
import tcod.sdl.video
from tcod.cffi import ffi, lib
from tcod.sdl._internal import Properties, _check, _check_p

if TYPE_CHECKING:
    from numpy.typing import NDArray


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


class LogicalPresentation(enum.IntEnum):
    """SDL logical presentation modes.

    See https://wiki.libsdl.org/SDL3/SDL_RendererLogicalPresentation

    .. versionadded:: 19.0
    """

    DISABLED = 0
    """"""
    STRETCH = 1
    """"""
    LETTERBOX = 2
    """"""
    OVERSCAN = 3
    """"""
    INTEGER_SCALE = 4
    """"""


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

        props = Properties(lib.SDL_GetTextureProperties(self.p))
        self.format: Final[int] = props[(tcod.sdl.constants.SDL_PROP_TEXTURE_FORMAT_NUMBER, int)]
        """Texture format, read only."""
        self.access: Final[TextureAccess] = TextureAccess(
            props[(tcod.sdl.constants.SDL_PROP_TEXTURE_ACCESS_NUMBER, int)]
        )
        """Texture access mode, read only.

        .. versionchanged:: 13.5
            Attribute is now a :any:`TextureAccess` value.
        """
        self.width: Final[int] = props[(tcod.sdl.constants.SDL_PROP_TEXTURE_WIDTH_NUMBER, int)]
        """Texture pixel width, read only."""
        self.height: Final[int] = props[(tcod.sdl.constants.SDL_PROP_TEXTURE_HEIGHT_NUMBER, int)]
        """Texture pixel height, read only."""

    def __eq__(self, other: object) -> bool:
        """Return True if compared to the same texture."""
        if isinstance(other, Texture):
            return bool(self.p == other.p)
        return NotImplemented

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
            lib.SDL_RenderTextureRotated(
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
        _check(lib.SDL_GetCurrentRenderOutputSize(self.p, out, out + 1))
        return out[0], out[1]

    @property
    def clip_rect(self) -> tuple[int, int, int, int] | None:
        """Get or set the clipping rectangle of this renderer.

        Set to None to disable clipping.

        .. versionadded:: 13.5
        """
        if not lib.SDL_RenderClipEnabled(self.p):
            return None
        rect = ffi.new("SDL_Rect*")
        lib.SDL_GetRenderClipRect(self.p, rect)
        return rect.x, rect.y, rect.w, rect.h

    @clip_rect.setter
    def clip_rect(self, rect: tuple[int, int, int, int] | None) -> None:
        rect_p = ffi.NULL if rect is None else ffi.new("SDL_Rect*", rect)
        _check(lib.SDL_SetRenderClipRect(self.p, rect_p))

    def set_logical_presentation(self, resolution: tuple[int, int], mode: LogicalPresentation) -> None:
        """Set this renderers device independent resolution.

        .. seealso::
            https://wiki.libsdl.org/SDL3/SDL_SetRenderLogicalPresentation

        .. versionadded:: 19.0
        """
        width, height = resolution
        _check(lib.SDL_SetRenderLogicalPresentation(self.p, width, height, mode))

    @property
    def logical_size(self) -> tuple[int, int]:
        """Get current independent (width, height) resolution.

        Might be (0, 0) if a resolution was never assigned.

        .. seealso::
            https://wiki.libsdl.org/SDL3/SDL_GetRenderLogicalPresentation

        .. versionadded:: 13.5

        .. versionchanged:: 19.0
            Setter is deprecated, use :any:`set_logical_presentation` instead.
        """
        out = ffi.new("int[2]")
        lib.SDL_GetRenderLogicalPresentation(self.p, out, out + 1, ffi.NULL)
        return out[0], out[1]

    @logical_size.setter
    @deprecated("Use set_logical_presentation method to correctly setup logical size.")
    def logical_size(self, size: tuple[int, int]) -> None:
        width, height = size
        _check(lib.SDL_SetRenderLogicalPresentation(self.p, width, height, lib.SDL_LOGICAL_PRESENTATION_STRETCH))

    @property
    def scale(self) -> tuple[float, float]:
        """Get or set an (x_scale, y_scale) multiplier for drawing.

        .. seealso::
            https://wiki.libsdl.org/SDL_RenderSetScale

        .. versionadded:: 13.5
        """
        out = ffi.new("float[2]")
        lib.SDL_GetRenderScale(self.p, out, out + 1)
        return out[0], out[1]

    @scale.setter
    def scale(self, scale: tuple[float, float]) -> None:
        _check(lib.SDL_SetRenderScale(self.p, *scale))

    @property
    def viewport(self) -> tuple[int, int, int, int] | None:
        """Get or set the drawing area for the current rendering target.

        .. seealso::
            https://wiki.libsdl.org/SDL_RenderSetViewport

        .. versionadded:: 13.5
        """
        rect = ffi.new("SDL_Rect*")
        lib.SDL_GetRenderViewport(self.p, rect)
        return rect.x, rect.y, rect.w, rect.h

    @viewport.setter
    def viewport(self, rect: tuple[int, int, int, int] | None) -> None:
        _check(lib.SDL_SetRenderViewport(self.p, (rect,)))

    def set_vsync(self, enable: bool) -> None:
        """Enable or disable VSync for this renderer.

        .. versionadded:: 13.5
        """
        _check(lib.SDL_SetRenderVSync(self.p, enable))

    def read_pixels(
        self,
        *,
        rect: tuple[int, int, int, int] | None = None,
        format: Literal["RGB", "RGBA"] = "RGBA",  # noqa: A002
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
        See https://wiki.libsdl.org/SDL3/SDL_RenderReadPixels

        Returns:
            The output uint8 array of shape: ``(height, width, channels)`` with the fetched pixels.

        .. versionadded:: 15.0

        .. versionchanged:: 19.0
            `format` no longer accepts `int` values.
        """
        surface = _check_p(
            ffi.gc(lib.SDL_RenderReadPixels(self.p, (rect,) if rect is not None else ffi.NULL), lib.SDL_DestroySurface)
        )
        width, height = rect[2:4] if rect is not None else (int(surface.w), int(surface.h))
        depth = {"RGB": 3, "RGBA": 4}.get(format)
        if depth is None:
            msg = f"Pixel format {format!r} not supported by tcod."
            raise TypeError(msg)
        expected_shape = height, width, depth
        if out is None:
            out = np.empty(expected_shape, dtype=np.uint8)
        if out.dtype != np.uint8:
            msg = "`out` must be a uint8 array."
            raise TypeError(msg)
        if out.shape != expected_shape:
            msg = f"Expected `out` to be an array of shape {expected_shape}, got {out.shape} instead."
            raise TypeError(msg)
        if not out[0].flags.c_contiguous:
            msg = "`out` array must be C contiguous."
        out_surface = tcod.sdl.video._TempSurface(out)
        _check(lib.SDL_BlitSurface(surface, ffi.NULL, out_surface.p, ffi.NULL))
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
        _check(lib.SDL_RenderFillRect(self.p, (rect,)))

    def draw_rect(self, rect: tuple[float, float, float, float]) -> None:
        """Draw a rectangle outline.

        .. versionadded:: 13.5
        """
        _check(lib.SDL_RenderFillRects(self.p, (rect,), 1))

    def draw_point(self, xy: tuple[float, float]) -> None:
        """Draw a point.

        .. versionadded:: 13.5
        """
        x, y = xy
        _check(lib.SDL_RenderPoint(self.p, x, y))

    def draw_line(self, start: tuple[float, float], end: tuple[float, float]) -> None:
        """Draw a single line.

        .. versionadded:: 13.5
        """
        x1, y1 = start
        x2, y2 = end
        _check(lib.SDL_RenderLine(self.p, x1, y1, x2, y2))

    @staticmethod
    def _convert_array(array: NDArray[np.number] | Sequence[Sequence[float]], item_length: int) -> NDArray[np.float32]:
        """Convert ndarray for a SDL function expecting a C contiguous array of either intc or float32.

        Array shape is enforced to be (n, item_length)
        """
        out = np.ascontiguousarray(array, np.float32)
        if len(out.shape) != 2:  # noqa: PLR2004
            msg = f"Array must have 2 axes, but shape is {out.shape!r}"
            raise TypeError(msg)
        if out.shape[1] != item_length:
            msg = f"Array shape[1] must be {item_length}, but shape is {out.shape!r}"
            raise TypeError(msg)
        return out

    def fill_rects(self, rects: NDArray[np.number] | Sequence[tuple[float, float, float, float]]) -> None:
        """Fill multiple rectangles from an array.

        Args:
            rects: A sequence or array of (x, y, width, height) rectangles.

        .. versionadded:: 13.5
        """
        rects = self._convert_array(rects, item_length=4)
        _check(lib.SDL_RenderFillRects(self.p, tcod.ffi.from_buffer("SDL_FRect*", rects), rects.shape[0]))

    def draw_rects(self, rects: NDArray[np.number] | Sequence[tuple[float, float, float, float]]) -> None:
        """Draw multiple outlined rectangles from an array.

        Args:
            rects: A sequence or array of (x, y, width, height) rectangles.

        .. versionadded:: 13.5
        """
        rects = self._convert_array(rects, item_length=4)
        assert len(rects.shape) == 2  # noqa: PLR2004
        assert rects.shape[1] == 4  # noqa: PLR2004
        _check(lib.SDL_RenderRects(self.p, tcod.ffi.from_buffer("SDL_FRect*", rects), rects.shape[0]))

    def draw_points(self, points: NDArray[np.number] | Sequence[tuple[float, float]]) -> None:
        """Draw an array of points.

        Args:
            points: A sequence or array of (x, y) points.

        .. versionadded:: 13.5
        """
        points = self._convert_array(points, item_length=2)
        _check(lib.SDL_RenderPoints(self.p, tcod.ffi.from_buffer("SDL_FPoint*", points), points.shape[0]))

    def draw_lines(self, points: NDArray[np.number] | Sequence[tuple[float, float]]) -> None:
        """Draw a connected series of lines from an array.

        Args:
            points: A sequence or array of (x, y) points.

        .. versionadded:: 13.5
        """
        points = self._convert_array(points, item_length=2)
        _check(lib.SDL_RenderLines(self.p, tcod.ffi.from_buffer("SDL_FPoint*", points), points.shape[0]))

    def geometry(
        self,
        texture: Texture | None,
        xy: NDArray[np.float32] | Sequence[tuple[float, float]],
        color: NDArray[np.float32] | Sequence[tuple[float, float, float, float]],
        uv: NDArray[np.float32] | Sequence[tuple[float, float]],
        indices: NDArray[np.uint8 | np.uint16 | np.uint32] | None = None,
    ) -> None:
        """Render triangles from texture and vertex data.

        Args:
            texture: The SDL texture to render from.
            xy: A sequence of (x, y) points to buffer.
            color: A sequence of (r, g, b, a) colors to buffer.
            uv: A sequence of (x, y) coordinates to buffer.
            indices: A sequence of indexes referring to the buffered data, every 3 indexes is a triangle to render.

        .. versionadded:: 13.5

        .. versionchanged:: 19.0
            `color` now takes float values instead of 8-bit integers.
        """
        xy = np.ascontiguousarray(xy, np.float32)
        assert len(xy.shape) == 2  # noqa: PLR2004
        assert xy.shape[1] == 2  # noqa: PLR2004

        color = np.ascontiguousarray(color, np.float32)
        assert len(color.shape) == 2  # noqa: PLR2004
        assert color.shape[1] == 4  # noqa: PLR2004

        uv = np.ascontiguousarray(uv, np.float32)
        assert len(uv.shape) == 2  # noqa: PLR2004
        assert uv.shape[1] == 2  # noqa: PLR2004
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
                ffi.cast("SDL_FColor*", color.ctypes.data),
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
    driver: str | None = None,
    vsync: int = True,
) -> Renderer:
    """Initialize and return a new SDL Renderer.

    Args:
        window: The window that this renderer will be attached to.
        driver: Force SDL to use a specific video driver.
        vsync: If True then Vsync will be enabled.

    Example::

        # Start by creating a window.
        sdl_window = tcod.sdl.video.new_window(640, 480)
        # Create a renderer with target texture support.
        sdl_renderer = tcod.sdl.render.new_renderer(sdl_window)

    .. seealso::
        :func:`tcod.sdl.video.new_window`

    .. versionchanged:: 19.0
        Removed `software` and `target_textures` parameters.
        `vsync` now takes an integer.
        `driver` now take a string.
    """
    props = Properties()
    props[(tcod.sdl.constants.SDL_PROP_RENDERER_CREATE_PRESENT_VSYNC_NUMBER, int)] = vsync
    props[(tcod.sdl.constants.SDL_PROP_RENDERER_CREATE_WINDOW_POINTER, tcod.sdl.video.Window)] = window
    if driver is not None:
        props[(tcod.sdl.constants.SDL_PROP_RENDERER_CREATE_NAME_STRING, str)] = driver
    renderer_p = _check_p(ffi.gc(lib.SDL_CreateRendererWithProperties(props.p), lib.SDL_DestroyRenderer))
    return Renderer(renderer_p)
