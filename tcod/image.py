"""Libtcod functionality for handling images.

This module is generally seen as outdated.
To load images you should typically use `Pillow <https://pillow.readthedocs.io/en/stable/>`_ or
`imageio <https://imageio.readthedocs.io/en/stable/>`_ unless you need to use a feature exclusive to libtcod.

**Python-tcod is unable to render pixels to consoles.**
The best it can do with consoles is convert an image into semigraphics which can be shown on non-emulated terminals.
For true pixel-based rendering you'll want to access the SDL rendering port at :any:`tcod.sdl.render`.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from typing_extensions import deprecated

from tcod._internal import _console, _path_encode
from tcod.cffi import ffi, lib

if TYPE_CHECKING:
    from os import PathLike

    from numpy.typing import ArrayLike, NDArray

    import tcod.console


class Image:
    """A libtcod image.

    Args:
        width (int): Width of the new Image.
        height (int): Height of the new Image.

    Attributes:
        width (int): Read only width of this Image.
        height (int): Read only height of this Image.
    """

    def __init__(self, width: int, height: int) -> None:
        """Initialize a blank image."""
        self.width, self.height = width, height
        self.image_c = ffi.gc(lib.TCOD_image_new(width, height), lib.TCOD_image_delete)
        if self.image_c == ffi.NULL:
            msg = "Failed to allocate image."
            raise MemoryError(msg)

    @classmethod
    def _from_cdata(cls, cdata: Any) -> Image:  # noqa: ANN401
        self: Image = object.__new__(cls)
        if cdata == ffi.NULL:
            msg = "Pointer must not be NULL."
            raise RuntimeError(msg)
        self.image_c = cdata
        self.width, self.height = self._get_size()
        return self

    @classmethod
    def from_array(cls, array: ArrayLike) -> Image:
        """Create a new Image from a copy of an array-like object.

        Example:
            >>> import numpy as np
            >>> import tcod
            >>> array = np.zeros((5, 5, 3), dtype=np.uint8)
            >>> image = tcod.image.Image.from_array(array)

        .. versionadded:: 11.4
        """
        array = np.asarray(array, dtype=np.uint8)
        height, width, depth = array.shape
        image = cls(width, height)
        image_array: NDArray[np.uint8] = np.asarray(image)
        image_array[...] = array
        return image

    @classmethod
    def from_file(cls, path: str | PathLike[str]) -> Image:
        """Return a new Image loaded from the given `path`.

        .. versionadded:: 16.0
        """
        path = Path(path).resolve(strict=True)
        return cls._from_cdata(ffi.gc(lib.TCOD_image_load(_path_encode(path)), lib.TCOD_image_delete))

    def clear(self, color: tuple[int, int, int]) -> None:
        """Fill this entire Image with color.

        Args:
            color (Union[Tuple[int, int, int], Sequence[int]]):
                An (r, g, b) sequence or Color instance.
        """
        lib.TCOD_image_clear(self.image_c, color)

    def invert(self) -> None:
        """Invert all colors in this Image."""
        lib.TCOD_image_invert(self.image_c)

    def hflip(self) -> None:
        """Horizontally flip this Image."""
        lib.TCOD_image_hflip(self.image_c)

    def rotate90(self, rotations: int = 1) -> None:
        """Rotate this Image clockwise in 90 degree steps.

        Args:
            rotations (int): Number of 90 degree clockwise rotations.
        """
        lib.TCOD_image_rotate90(self.image_c, rotations)

    def vflip(self) -> None:
        """Vertically flip this Image."""
        lib.TCOD_image_vflip(self.image_c)

    def scale(self, width: int, height: int) -> None:
        """Scale this Image to the new width and height.

        Args:
            width (int): The new width of the Image after scaling.
            height (int): The new height of the Image after scaling.
        """
        lib.TCOD_image_scale(self.image_c, width, height)
        self.width, self.height = width, height

    def set_key_color(self, color: tuple[int, int, int]) -> None:
        """Set a color to be transparent during blitting functions.

        Args:
            color (Union[Tuple[int, int, int], Sequence[int]]):
                An (r, g, b) sequence or Color instance.
        """
        lib.TCOD_image_set_key_color(self.image_c, color)

    def get_alpha(self, x: int, y: int) -> int:
        """Get the Image alpha of the pixel at x, y.

        Args:
            x (int): X pixel of the image.  Starting from the left at 0.
            y (int): Y pixel of the image.  Starting from the top at 0.

        Returns:
            int: The alpha value of the pixel.
            With 0 being fully transparent and 255 being fully opaque.
        """
        return int(lib.TCOD_image_get_alpha(self.image_c, x, y))

    def refresh_console(self, console: tcod.console.Console) -> None:
        """Update an Image created with :any:`libtcodpy.image_from_console`.

        The console used with this function should have the same width and
        height as the Console given to :any:`libtcodpy.image_from_console`.
        The font width and height must also be the same as when
        :any:`libtcodpy.image_from_console` was called.

        Args:
            console (Console): A Console with a pixel width and height
                               matching this Image.
        """
        lib.TCOD_image_refresh_console(self.image_c, _console(console))

    def _get_size(self) -> tuple[int, int]:
        """Return the (width, height) for this Image.

        Returns:
            Tuple[int, int]: The (width, height) of this Image
        """
        w = ffi.new("int *")
        h = ffi.new("int *")
        lib.TCOD_image_get_size(self.image_c, w, h)
        return w[0], h[0]

    def get_pixel(self, x: int, y: int) -> tuple[int, int, int]:
        """Get the color of a pixel in this Image.

        Args:
            x (int): X pixel of the Image.  Starting from the left at 0.
            y (int): Y pixel of the Image.  Starting from the top at 0.

        Returns:
            Tuple[int, int, int]:
                An (r, g, b) tuple containing the pixels color value.
                Values are in a 0 to 255 range.
        """
        color = lib.TCOD_image_get_pixel(self.image_c, x, y)
        return color.r, color.g, color.b

    def get_mipmap_pixel(self, left: float, top: float, right: float, bottom: float) -> tuple[int, int, int]:
        """Get the average color of a rectangle in this Image.

        Parameters should stay within the following limits:
        * 0 <= left < right < Image.width
        * 0 <= top < bottom < Image.height

        Args:
            left (float): Left corner of the region.
            top (float): Top corner of the region.
            right (float): Right corner of the region.
            bottom (float): Bottom corner of the region.

        Returns:
            Tuple[int, int, int]:
                An (r, g, b) tuple containing the averaged color value.
                Values are in a 0 to 255 range.
        """
        color = lib.TCOD_image_get_mipmap_pixel(self.image_c, left, top, right, bottom)
        return (color.r, color.g, color.b)

    def put_pixel(self, x: int, y: int, color: tuple[int, int, int]) -> None:
        """Change a pixel on this Image.

        Args:
            x (int): X pixel of the Image.  Starting from the left at 0.
            y (int): Y pixel of the Image.  Starting from the top at 0.
            color (Union[Tuple[int, int, int], Sequence[int]]):
                An (r, g, b) sequence or Color instance.
        """
        lib.TCOD_image_put_pixel(self.image_c, x, y, color)

    def blit(  # noqa: PLR0913
        self,
        console: tcod.console.Console,
        x: float,
        y: float,
        bg_blend: int,
        scale_x: float,
        scale_y: float,
        angle: float,
    ) -> None:
        """Blit onto a Console using scaling and rotation.

        Args:
            console (Console): Blit destination Console.
            x (float): Console X position for the center of the Image blit.
            y (float): Console Y position for the center of the Image blit.
                     The Image blit is centered on this position.
            bg_blend (int): Background blending mode to use.
            scale_x (float): Scaling along Image x axis.
                             Set to 1 for no scaling.  Must be over 0.
            scale_y (float): Scaling along Image y axis.
                             Set to 1 for no scaling.  Must be over 0.
            angle (float): Rotation angle in radians. (Clockwise?)
        """
        lib.TCOD_image_blit(
            self.image_c,
            _console(console),
            x,
            y,
            bg_blend,
            scale_x,
            scale_y,
            angle,
        )

    def blit_rect(  # noqa: PLR0913
        self,
        console: tcod.console.Console,
        x: int,
        y: int,
        width: int,
        height: int,
        bg_blend: int,
    ) -> None:
        """Blit onto a Console without scaling or rotation.

        Args:
            console (Console): Blit destination Console.
            x (int): Console tile X position starting from the left at 0.
            y (int): Console tile Y position starting from the top at 0.
            width (int): Use -1 for Image width.
            height (int): Use -1 for Image height.
            bg_blend (int): Background blending mode to use.
        """
        lib.TCOD_image_blit_rect(self.image_c, _console(console), x, y, width, height, bg_blend)

    def blit_2x(  # noqa: PLR0913
        self,
        console: tcod.console.Console,
        dest_x: int,
        dest_y: int,
        img_x: int = 0,
        img_y: int = 0,
        img_width: int = -1,
        img_height: int = -1,
    ) -> None:
        """Blit onto a Console with double resolution.

        Args:
            console (Console): Blit destination Console.
            dest_x (int): Console tile X position starting from the left at 0.
            dest_y (int): Console tile Y position starting from the top at 0.
            img_x (int): Left corner pixel of the Image to blit
            img_y (int): Top corner pixel of the Image to blit
            img_width (int): Width of the Image to blit.
                             Use -1 for the full Image width.
            img_height (int): Height of the Image to blit.
                              Use -1 for the full Image height.
        """
        lib.TCOD_image_blit_2x(
            self.image_c,
            _console(console),
            dest_x,
            dest_y,
            img_x,
            img_y,
            img_width,
            img_height,
        )

    def save_as(self, filename: str | PathLike[str]) -> None:
        """Save the Image to a 32-bit .bmp or .png file.

        Args:
            filename (Text): File path to same this Image.

        .. versionchanged:: 16.0
            Added PathLike support.
        """
        lib.TCOD_image_save(self.image_c, _path_encode(Path(filename)))

    @property
    def __array_interface__(self) -> dict[str, Any]:
        """Return an interface for this images pixel buffer.

        Use :any:`numpy.asarray` to get the read-write array of this Image.

        This will often return an RGB array, but could also return an RGBA
        array or fail silently.  Future versions might change what type of
        array is returned.

        You can use ``dtype=numpy.uint8`` to ensure that errors are not ignored
        by NumPy.

        .. versionadded:: 11.4
        """
        strides = None
        if self.image_c.mipmaps:  # Libtcod RGB array.
            depth = 3
            data = int(ffi.cast("size_t", self.image_c.mipmaps[0].buf))
        else:
            msg = "Image has no initialized data."
            raise TypeError(msg)
        return {
            "shape": (self.height, self.width, depth),
            "typestr": "|u1",
            "data": (data, False),
            "strides": strides,
            "version": 3,
        }


@deprecated(
    "This function may be removed in the future."
    "  It's recommended to load images with a more complete image library such as python-Pillow or python-imageio.",
)
def load(filename: str | PathLike[str]) -> NDArray[np.uint8]:
    """Load a PNG file as an RGBA array.

    `filename` is the name of the file to load.

    The returned array is in the shape: `(height, width, RGBA)`.

    .. versionadded:: 11.4
    """
    filename = Path(filename).resolve(strict=True)
    image = Image._from_cdata(ffi.gc(lib.TCOD_image_load(_path_encode(filename)), lib.TCOD_image_delete))
    array: NDArray[np.uint8] = np.asarray(image, dtype=np.uint8)
    height, width, depth = array.shape
    if depth == 3:  # noqa: PLR2004
        array = np.concatenate(
            (
                array,
                np.full((height, width, 1), fill_value=255, dtype=np.uint8),
            ),
            axis=2,
        )
    return array


class _TempImage:
    """An Image-like container for NumPy arrays."""

    def __init__(self, array: ArrayLike) -> None:
        """Initialize an image from the given array.  May copy or reference the array."""
        self._array: NDArray[np.uint8] = np.ascontiguousarray(array, dtype=np.uint8)
        height, width, depth = self._array.shape
        if depth != 3:  # noqa: PLR2004
            msg = f"Array must have RGB channels.  Shape is: {self._array.shape!r}"
            raise TypeError(msg)
        self._buffer = ffi.from_buffer("TCOD_color_t[]", self._array)
        self._mipmaps = ffi.new(
            "struct TCOD_mipmap_*",
            {
                "width": width,
                "height": height,
                "fwidth": width,
                "fheight": height,
                "buf": self._buffer,
                "dirty": True,
            },
        )
        self.image_c = ffi.new(
            "TCOD_Image*",
            {
                "nb_mipmaps": 1,
                "mipmaps": self._mipmaps,
                "has_key_color": False,
            },
        )


def _as_image(image: ArrayLike | Image | _TempImage) -> _TempImage | Image:
    """Convert this input into an Image-like object."""
    if isinstance(image, (Image, _TempImage)):
        return image
    return _TempImage(image)
