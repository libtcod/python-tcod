"""Tileset and font related functions.

Tilesets can be loaded as a whole from tile-sheets or True-Type fonts, or they
can be put together from multiple tile images by loading them separately
using :any:`Tileset.set_tile`.

A major restriction with libtcod is that all tiles must be the same size and
tiles can't overlap when rendered.  For sprite-based rendering it can be
useful to use `an alternative library for graphics rendering
<https://wiki.python.org/moin/PythonGameLibraries>`_ while continuing to use
python-tcod's pathfinding and field-of-view algorithms.
"""
import itertools
import os
from typing import Any, Iterable, Optional, Tuple

import numpy as np

import tcod.console
from tcod._internal import _check, _console, _raise_tcod_error, deprecate
from tcod.loader import ffi, lib


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
            ffi.from_buffer("struct TCOD_ColorRGBA*", tile),
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
            ffi.from_buffer("struct TCOD_ColorRGBA*", tile),
        )

    def render(self, console: tcod.console.Console) -> np.ndarray:
        """Render an RGBA array, using console with this tileset.

        `console` is the Console object to render, this can not be the root
        console.

        The output array will be a np.uint8 array with the shape of:
        ``(con_height * tile_height, con_width * tile_width, 4)``.

        .. versionadded:: 11.9
        """
        if not console:
            raise ValueError("'console' must not be the root console.")
        width = console.width * self.tile_width
        height = console.height * self.tile_height
        out = np.empty((height, width, 4), np.uint8)
        out[:] = 9
        surface_p = ffi.gc(
            lib.SDL_CreateRGBSurfaceWithFormatFrom(
                ffi.from_buffer("void*", out),
                width,
                height,
                32,
                out.strides[0],
                lib.SDL_PIXELFORMAT_RGBA32,
            ),
            lib.SDL_FreeSurface,
        )
        with surface_p:
            with ffi.new("SDL_Surface**", surface_p) as surface_p_p:
                _check(
                    lib.TCOD_tileset_render_to_surface(
                        self._tileset_p,
                        _console(console),
                        ffi.NULL,
                        surface_p_p,
                    )
                )
        return out

    def remap(self, codepoint: int, x: int, y: int = 0) -> None:
        """Reassign a codepoint to a character in this tileset.

        `codepoint` is the Unicode codepoint to assign.

        `x` and `y` is the position on the tilesheet to assign to `codepoint`.
        Large values of `x` will wrap to the next row, so `y` isn't necessary
        if you think of the tilesheet as a 1D array.

        This is normally used on loaded tilesheets.  Other methods of Tileset
        creation won't have reliable tile indexes.

        .. versionadded:: 11.12
        """
        tile_i = x + y * self._tileset_p.virtual_columns
        if not (0 <= tile_i < self._tileset_p.tiles_count):
            raise IndexError(
                "Tile %i is non-existent and can't be assigned."
                " (Tileset has %i tiles.)"
                % (tile_i, self._tileset_p.tiles_count)
            )
        _check(
            lib.TCOD_tileset_assign_tile(
                self._tileset_p,
                tile_i,
                codepoint,
            )
        )


@deprecate("Using the default tileset is deprecated.")
def get_default() -> Tileset:
    """Return a reference to the default Tileset.

    .. versionadded:: 11.10

    .. deprecated:: 11.13
        The default tileset is deprecated.
        With contexts this is no longer needed.
    """
    return Tileset._claim(lib.TCOD_get_default_tileset())


@deprecate("Using the default tileset is deprecated.")
def set_default(tileset: Tileset) -> None:
    """Set the default tileset.

    The display will use this new tileset immediately.

    .. versionadded:: 11.10

    .. deprecated:: 11.13
        The default tileset is deprecated.
        With contexts this is no longer needed.
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


@deprecate("Accessing the default tileset is deprecated.")
def set_truetype_font(path: str, tile_width: int, tile_height: int) -> None:
    """Set the default tileset from a `.ttf` or `.otf` file.

    `path` is the file path for the font file.

    `tile_width` and `tile_height` are the desired size of the tiles in the new
    tileset.  The font will be scaled to fit the given `tile_height` and
    `tile_width`.

    This function must be called before :any:`tcod.console_init_root`.  Once
    the root console is setup you may call this funtion again to change the
    font.  The tileset can be changed but the window will not be resized
    automatically.

    .. versionadded:: 9.2

    .. deprecated:: 11.13
        This function does not support contexts.
        Use :any:`load_truetype_font` instead.
    """
    if not os.path.exists(path):
        raise RuntimeError("File not found:\n\t%s" % (os.path.realpath(path),))
    if lib.TCOD_tileset_load_truetype_(path.encode(), tile_width, tile_height):
        raise RuntimeError(ffi.string(lib.TCOD_get_error()))


def load_bdf(path: str) -> Tileset:
    """Return a new Tileset from a `.bdf` file.

    For the best results the font should be monospace, cell-based, and
    single-width.  As an example, a good set of fonts would be the
    `Unicode fonts and tools for X11 <https://www.cl.cam.ac.uk/~mgk25/ucs-fonts.html>`_
    package.

    Pass the returned Tileset to :any:`tcod.tileset.set_default` and it will
    take effect when `tcod.console_init_root` is called.

    .. versionadded:: 11.10
    """  # noqa: E501
    if not os.path.exists(path):
        raise RuntimeError("File not found:\n\t%s" % (os.path.realpath(path),))
    cdata = lib.TCOD_load_bdf(path.encode())
    if not cdata:
        raise RuntimeError(ffi.string(lib.TCOD_get_error()).decode())
    return Tileset._claim(cdata)


def load_tilesheet(
    path: str, columns: int, rows: int, charmap: Optional[Iterable[int]]
) -> Tileset:
    """Return a new Tileset from a simple tilesheet image.

    `path` is the file path to a PNG file with the tileset.

    `columns` and `rows` is the shape of the tileset.  Tiles are assumed to
    take up the entire space of the image.

    `charmap` is the character mapping to use.  This is a list or generator
    of codepoints which map the tiles like this:
    ``charmap[tile_index] = codepoint``.
    For common tilesets `charmap` should be :any:`tcod.tileset.CHARMAP_CP437`.
    Generators will be sliced so :any:`itertools.count` can be used which will
    give all tiles the same codepoint as their index, but this will not map
    tiles onto proper Unicode.
    If `None` is used then no tiles will be mapped, you will need to use
    :any:`Tileset.remap` to assign codepoints to this Tileset.

    .. versionadded:: 11.12
    """
    if not os.path.exists(path):
        raise RuntimeError("File not found:\n\t%s" % (os.path.realpath(path),))
    mapping = []
    if charmap is not None:
        mapping = list(itertools.islice(charmap, columns * rows))
    cdata = lib.TCOD_tileset_load(
        path.encode(), columns, rows, len(mapping), mapping
    )
    if not cdata:
        _raise_tcod_error()
    return Tileset._claim(cdata)


CHARMAP_CP437 = [
    0x0000,
    0x263A,
    0x263B,
    0x2665,
    0x2666,
    0x2663,
    0x2660,
    0x2022,
    0x25D8,
    0x25CB,
    0x25D9,
    0x2642,
    0x2640,
    0x266A,
    0x266B,
    0x263C,
    0x25BA,
    0x25C4,
    0x2195,
    0x203C,
    0x00B6,
    0x00A7,
    0x25AC,
    0x21A8,
    0x2191,
    0x2193,
    0x2192,
    0x2190,
    0x221F,
    0x2194,
    0x25B2,
    0x25BC,
    0x0020,
    0x0021,
    0x0022,
    0x0023,
    0x0024,
    0x0025,
    0x0026,
    0x0027,
    0x0028,
    0x0029,
    0x002A,
    0x002B,
    0x002C,
    0x002D,
    0x002E,
    0x002F,
    0x0030,
    0x0031,
    0x0032,
    0x0033,
    0x0034,
    0x0035,
    0x0036,
    0x0037,
    0x0038,
    0x0039,
    0x003A,
    0x003B,
    0x003C,
    0x003D,
    0x003E,
    0x003F,
    0x0040,
    0x0041,
    0x0042,
    0x0043,
    0x0044,
    0x0045,
    0x0046,
    0x0047,
    0x0048,
    0x0049,
    0x004A,
    0x004B,
    0x004C,
    0x004D,
    0x004E,
    0x004F,
    0x0050,
    0x0051,
    0x0052,
    0x0053,
    0x0054,
    0x0055,
    0x0056,
    0x0057,
    0x0058,
    0x0059,
    0x005A,
    0x005B,
    0x005C,
    0x005D,
    0x005E,
    0x005F,
    0x0060,
    0x0061,
    0x0062,
    0x0063,
    0x0064,
    0x0065,
    0x0066,
    0x0067,
    0x0068,
    0x0069,
    0x006A,
    0x006B,
    0x006C,
    0x006D,
    0x006E,
    0x006F,
    0x0070,
    0x0071,
    0x0072,
    0x0073,
    0x0074,
    0x0075,
    0x0076,
    0x0077,
    0x0078,
    0x0079,
    0x007A,
    0x007B,
    0x007C,
    0x007D,
    0x007E,
    0x007F,
    0x00C7,
    0x00FC,
    0x00E9,
    0x00E2,
    0x00E4,
    0x00E0,
    0x00E5,
    0x00E7,
    0x00EA,
    0x00EB,
    0x00E8,
    0x00EF,
    0x00EE,
    0x00EC,
    0x00C4,
    0x00C5,
    0x00C9,
    0x00E6,
    0x00C6,
    0x00F4,
    0x00F6,
    0x00F2,
    0x00FB,
    0x00F9,
    0x00FF,
    0x00D6,
    0x00DC,
    0x00A2,
    0x00A3,
    0x00A5,
    0x20A7,
    0x0192,
    0x00E1,
    0x00ED,
    0x00F3,
    0x00FA,
    0x00F1,
    0x00D1,
    0x00AA,
    0x00BA,
    0x00BF,
    0x2310,
    0x00AC,
    0x00BD,
    0x00BC,
    0x00A1,
    0x00AB,
    0x00BB,
    0x2591,
    0x2592,
    0x2593,
    0x2502,
    0x2524,
    0x2561,
    0x2562,
    0x2556,
    0x2555,
    0x2563,
    0x2551,
    0x2557,
    0x255D,
    0x255C,
    0x255B,
    0x2510,
    0x2514,
    0x2534,
    0x252C,
    0x251C,
    0x2500,
    0x253C,
    0x255E,
    0x255F,
    0x255A,
    0x2554,
    0x2569,
    0x2566,
    0x2560,
    0x2550,
    0x256C,
    0x2567,
    0x2568,
    0x2564,
    0x2565,
    0x2559,
    0x2558,
    0x2552,
    0x2553,
    0x256B,
    0x256A,
    0x2518,
    0x250C,
    0x2588,
    0x2584,
    0x258C,
    0x2590,
    0x2580,
    0x03B1,
    0x00DF,
    0x0393,
    0x03C0,
    0x03A3,
    0x03C3,
    0x00B5,
    0x03C4,
    0x03A6,
    0x0398,
    0x03A9,
    0x03B4,
    0x221E,
    0x03C6,
    0x03B5,
    0x2229,
    0x2261,
    0x00B1,
    0x2265,
    0x2264,
    0x2320,
    0x2321,
    0x00F7,
    0x2248,
    0x00B0,
    0x2219,
    0x00B7,
    0x221A,
    0x207F,
    0x00B2,
    0x25A0,
    0x00A0,
]
"""This is one of the more common character mappings.

https://en.wikipedia.org/wiki/Code_page_437

.. versionadded:: 11.12
"""

CHARMAP_TCOD = [
    0x20,
    0x21,
    0x22,
    0x23,
    0x24,
    0x25,
    0x26,
    0x27,
    0x28,
    0x29,
    0x2A,
    0x2B,
    0x2C,
    0x2D,
    0x2E,
    0x2F,
    0x30,
    0x31,
    0x32,
    0x33,
    0x34,
    0x35,
    0x36,
    0x37,
    0x38,
    0x39,
    0x3A,
    0x3B,
    0x3C,
    0x3D,
    0x3E,
    0x3F,
    0x40,
    0x5B,
    0x5C,
    0x5D,
    0x5E,
    0x5F,
    0x60,
    0x7B,
    0x7C,
    0x7D,
    0x7E,
    0x2591,
    0x2592,
    0x2593,
    0x2502,
    0x2500,
    0x253C,
    0x2524,
    0x2534,
    0x251C,
    0x252C,
    0x2514,
    0x250C,
    0x2510,
    0x2518,
    0x2598,
    0x259D,
    0x2580,
    0x2596,
    0x259A,
    0x2590,
    0x2597,
    0x2191,
    0x2193,
    0x2190,
    0x2192,
    0x25B2,
    0x25BC,
    0x25C4,
    0x25BA,
    0x2195,
    0x2194,
    0x2610,
    0x2611,
    0x25CB,
    0x25C9,
    0x2551,
    0x2550,
    0x256C,
    0x2563,
    0x2569,
    0x2560,
    0x2566,
    0x255A,
    0x2554,
    0x2557,
    0x255D,
    0x00,
    0x00,
    0x00,
    0x00,
    0x00,
    0x00,
    0x00,
    0x41,
    0x42,
    0x43,
    0x44,
    0x45,
    0x46,
    0x47,
    0x48,
    0x49,
    0x4A,
    0x4B,
    0x4C,
    0x4D,
    0x4E,
    0x4F,
    0x50,
    0x51,
    0x52,
    0x53,
    0x54,
    0x55,
    0x56,
    0x57,
    0x58,
    0x59,
    0x5A,
    0x00,
    0x00,
    0x00,
    0x00,
    0x00,
    0x00,
    0x61,
    0x62,
    0x63,
    0x64,
    0x65,
    0x66,
    0x67,
    0x68,
    0x69,
    0x6A,
    0x6B,
    0x6C,
    0x6D,
    0x6E,
    0x6F,
    0x70,
    0x71,
    0x72,
    0x73,
    0x74,
    0x75,
    0x76,
    0x77,
    0x78,
    0x79,
    0x7A,
    0x00,
    0x00,
    0x00,
    0x00,
    0x00,
    0x00,
]
"""The layout used by older libtcod fonts, in Unicode.

This layout is non-standard, and it's not recommend to make a font for it, but
you might need it to load an existing font.

This character map is in Unicode, so old code using the non-Unicode
`tcod.CHAR_*` constants will need to be updated.

.. versionadded:: 11.12
"""
