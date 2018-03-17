
from __future__ import absolute_import

import numpy as np

from tcod.libtcod import ffi, lib
from tcod.tcod import _console

class _ImageBufferArray(np.ndarray):

    def __new__(cls, image):
        size = image.height * image.width
        self = np.frombuffer(ffi.buffer(lib.TCOD_image_get_colors()[size]),
                             np.uint8)
        self = self.reshape((image.height, image.width, 3)).view(cls)
        self._image_c = image.cdata
        return self

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self._image_c = getattr(obj, '_image_c', None)

    def __repr__(self):
        return repr(self.view(np.ndarray))

    def __setitem__(self, index, value):
        """Must invalidate mipmaps on any write."""
        np.ndarray.__setitem__(self, index, value)
        if self._image_c is not None:
            lib.TCOD_image_invalidate_mipmaps(self._image_c)


class Image(object):
    """
    Args:
        width (int): Width of the new Image.
        height (int): Height of the new Image.

    Attributes:
        width (int): Read only width of this Image.
        height (int): Read only height of this Image.
    """
    def __init__(self, width, height):
        self.width, self.height = width, height
        self.image_c = ffi.gc(lib.TCOD_image_new(width, height),
                              lib.TCOD_image_delete)

    @classmethod
    def _from_cdata(cls, cdata):
        self = object.__new__(cls)
        self.image_c = cdata
        self.width, self.height = self._get_size()
        return self

    def clear(self, color):
        """Fill this entire Image with color.

        Args:
            color (Union[Tuple[int, int, int], Sequence[int]]):
                An (r, g, b) sequence or Color instance.
        """
        lib.TCOD_image_clear(self.image_c, color)

    def invert(self):
        """Invert all colors in this Image."""
        lib.TCOD_image_invert(self.image_c)

    def hflip(self):
        """Horizontally flip this Image."""
        lib.TCOD_image_hflip(self.image_c)

    def rotate90(self, rotations=1):
        """Rotate this Image clockwise in 90 degree steps.

        Args:
            rotations (int): Number of 90 degree clockwise rotations.
        """
        lib.TCOD_image_rotate90(self.image_c, rotations)

    def vflip(self):
        """Vertically flip this Image."""
        lib.TCOD_image_vflip(self.image_c)

    def scale(self, width, height):
        """Scale this Image to the new width and height.

        Args:
            width (int): The new width of the Image after scaling.
            height (int): The new height of the Image after scaling.
        """
        lib.TCOD_image_scale(self.image_c, width, height)
        self.width, self.height = width, height

    def set_key_color(self, color):
        """Set a color to be transparent during blitting functions.

        Args:
            color (Union[Tuple[int, int, int], Sequence[int]]):
                An (r, g, b) sequence or Color instance.
        """
        lib.TCOD_image_set_key_color(self.image_c, color)

    def get_alpha(self, x, y):
        """Get the Image alpha of the pixel at x, y.

        Args:
            x (int): X pixel of the image.  Starting from the left at 0.
            y (int): Y pixel of the image.  Starting from the top at 0.

        Returns:
            int: The alpha value of the pixel.
            With 0 being fully transparent and 255 being fully opaque.
        """
        return lib.TCOD_image_get_alpha(self.image_c, x, y)

    def refresh_console(self, console):
        """Update an Image created with :any:`tcod.image_from_console`.

        The console used with this function should have the same width and
        height as the Console given to :any:`tcod.image_from_console`.
        The font width and height must also be the same as when
        :any:`tcod.image_from_console` was called.

        Args:
            console (Console): A Console with a pixel width and height
                               matching this Image.
        """
        lib.TCOD_image_refresh_console(self.image_c, _console(console))

    def _get_size(self):
        """Return the (width, height) for this Image.

        Returns:
            Tuple[int, int]: The (width, height) of this Image
        """
        w = ffi.new('int *')
        h = ffi.new('int *')
        lib.TCOD_image_get_size(self.image_c, w, h)
        return w[0], h[0]

    def get_pixel(self, x, y):
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

    def get_mipmap_pixel(self, left, top, right, bottom):
        """Get the average color of a rectangle in this Image.

        Parameters should stay within the following limits:
        * 0 <= left < right < Image.width
        * 0 <= top < bottom < Image.height

        Args:
            left (int): Left corner of the region.
            top (int): Top corner of the region.
            right (int): Right corner of the region.
            bottom (int): Bottom corner of the region.

        Returns:
            Tuple[int, int, int]:
                An (r, g, b) tuple containing the averaged color value.
                Values are in a 0 to 255 range.
        """
        color = lib.TCOD_image_get_mipmap_pixel(self.image_c,
                                                left, top, right, bottom)
        return (color.r, color.g, color.b)

    def put_pixel(self, x, y, color):
        """Change a pixel on this Image.

        Args:
            x (int): X pixel of the Image.  Starting from the left at 0.
            y (int): Y pixel of the Image.  Starting from the top at 0.
            color (Union[Tuple[int, int, int], Sequence[int]]):
                An (r, g, b) sequence or Color instance.
        """
        lib.TCOD_image_put_pixel(self.image_c, x, y, color)

    def blit(self, console, x, y, bg_blend, scale_x, scale_y, angle):
        """Blit onto a Console using scaling and rotation.

        Args:
            console (Console): Blit destination Console.
            x (int): Console X position for the center of the Image blit.
            y (int): Console Y position for the center of the Image blit.
                     The Image blit is centered on this position.
            bg_blend (int): Background blending mode to use.
            scale_x (float): Scaling along Image x axis.
                             Set to 1 for no scaling.  Must be over 0.
            scale_y (float): Scaling along Image y axis.
                             Set to 1 for no scaling.  Must be over 0.
            angle (float): Rotation angle in radians. (Clockwise?)
        """
        lib.TCOD_image_blit(
            self.image_c, _console(console),
            x, y, bg_blend, scale_x, scale_y, angle)

    def blit_rect(self, console, x, y, width, height, bg_blend):
        """Blit onto a Console without scaling or rotation.

        Args:
            console (Console): Blit destination Console.
            x (int): Console tile X position starting from the left at 0.
            y (int): Console tile Y position starting from the top at 0.
            width (int): Use -1 for Image width.
            height (int): Use -1 for Image height.
            bg_blend (int): Background blending mode to use.
        """
        lib.TCOD_image_blit_rect(
            self.image_c, _console(console), x, y, width, height, bg_blend)

    def blit_2x(self, console, dest_x, dest_y,
                img_x=0, img_y=0, img_width=-1, img_height=-1):
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
            self.image_c, _console(console),
            dest_x, dest_y, img_x, img_y, img_width, img_height)

    def save_as(self, filename):
        """Save the Image to a 32-bit .bmp or .png file.

        Args:
            filename (Text): File path to same this Image.
        """
        lib.TCOD_image_save(self.image_c, filename.encode('utf-8'))
