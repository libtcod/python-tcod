"""Tileset and font related functions.
"""
import os

from typing import Any, Tuple

import numpy as np

from tcod.libtcod import lib, ffi


class Tileset:
    """A collection of graphical tiles.

    This class is provisional, the API may change in the future.
    """

    def __init__(self, tile_width: int, tile_height: int) -> None:
        self._tileset_p = ffi.gc(
            lib.TCOD_tileset_new(tile_width, tile_height),
            lib.TCOD_tileset_delete,
        )

    @classmethod
    def _claim(cls, cdata: Any) -> "Tileset":
        """Return a new Tileset that owns the provided TCOD_Tileset* object."""
        self = object.__new__(cls)  # type: Tileset
        if cdata == ffi.NULL:
            raise RuntimeError("Tileset initialized with nullptr.")
        self._tileset_p = ffi.gc(cdata, lib.TCOD_tileset_delete)
        return self

    @property
    def tile_width(self) -> int:
        """The width of the tile in pixels."""
        return int(lib.TCOD_tileset_get_tile_width_(self._tileset_p))

    @property
    def tile_height(self) -> int:
        """The height of the tile in pixels."""
        return int(lib.TCOD_tileset_get_tile_height_(self._tileset_p))

    @property
    def tile_shape(self) -> Tuple[int, int]:
        """The shape (height, width) of the tile in pixels."""
        return self.tile_height, self.tile_width

    def __contains__(self, codepoint: int) -> bool:
        """Test if a tileset has a codepoint with ``n in tileset``."""
        return bool(
            lib.TCOD_tileset_get_tile_(self._tileset_p, codepoint, ffi.NULL)
            == 0
        )

    def get_tile(self, codepoint: int) -> np.ndarray:
        """Return a copy of a tile for the given codepoint.

        If the tile does not exist yet then a blank array will be returned.

        The tile will have a shape of (height, width, rgba) and a dtype of
        uint8.  Note that most grey-scale tiles will only use the alpha
        channel and will usually have a solid white color channel.
        """
        tile = np.zeros(self.tile_shape + (4,), dtype=np.uint8)
        lib.TCOD_tileset_get_tile_(
            self._tileset_p,
            codepoint,
            ffi.cast("struct TCOD_ColorRGBA*", tile.ctypes.data),
        )
        return tile

    def set_tile(self, codepoint: int, tile: np.ndarray) -> None:
        """Upload a tile into this array.

        The tile can be in 32-bit color (height, width, rgba), or grey-scale
        (height, width).  The tile should have a dtype of ``np.uint8``.

        This data may need to be sent to graphics card memory, this is a slow
        operation.
        """
        tile = np.ascontiguousarray(tile, dtype=np.uint8)
        if tile.shape == self.tile_shape:
            full_tile = np.empty(self.tile_shape + (4,), dtype=np.uint8)
            full_tile[:, :, :3] = 255
            full_tile[:, :, 3] = tile
            return self.set_tile(codepoint, full_tile)
        required = self.tile_shape + (4,)
        if tile.shape != required:
            raise ValueError(
                "Tile shape must be %r or %r, got %r."
                % (required, self.tile_shape, tile.shape)
            )
        lib.TCOD_tileset_set_tile_(
            self._tileset_p,
            codepoint,
            ffi.cast("struct TCOD_ColorRGBA*", tile.ctypes.data),
        )


def get_default() -> Tileset:
    """Return a reference to the default Tileset.

    This function is provisional.  The API may change.
    """
    return Tileset._claim(lib.TCOD_get_default_tileset())


def set_default(tileset: Tileset) -> None:
    """Set the default tileset.

    The display will use this new tileset immediately.

    This function only affects the `SDL2` and `OPENGL2` renderers.

    This function is provisional.  The API may change.
    """
    lib.TCOD_set_default_tileset(tileset._tileset_p)


def load_truetype_font(
    path: str, tile_width: int, tile_height: int
) -> Tileset:
    """Return a new Tileset from a `.ttf` or `.otf` file.

    Same as :any:`set_truetype_font`, but returns a :any:`Tileset` instead.
    You can send this Tileset to :any:`set_default`.

    This function is provisional.  The API may change.
    """
    if not os.path.exists(path):
        raise RuntimeError("File not found:\n\t%s" % (os.path.realpath(path),))
    cdata = lib.TCOD_load_truetype_font_(
        path.encode(), tile_width, tile_height
    )
    if not cdata:
        raise RuntimeError(ffi.string(lib.TCOD_get_error()))
    return Tileset._claim(cdata)


def set_truetype_font(path: str, tile_width: int, tile_height: int) -> None:
    """Set the default tileset from a `.ttf` or `.otf` file.

    `path` is the file path for the font file.

    `tile_width` and `tile_height` are the desired size of the tiles in the new
    tileset.  The font will be scaled to fit the given `tile_height` and
    `tile_width`.

    This function will only affect the `SDL2` and `OPENGL2` renderers.

    This function must be called before :any:`tcod.console_init_root`.  Once
    the root console is setup you may call this funtion again to change the
    font.  The tileset can be changed but the window will not be resized
    automatically.

    .. versionadded:: 9.2
    """
    if not os.path.exists(path):
        raise RuntimeError("File not found:\n\t%s" % (os.path.realpath(path),))
    if lib.TCOD_tileset_load_truetype_(path.encode(), tile_width, tile_height):
        raise RuntimeError(ffi.string(lib.TCOD_get_error()))
