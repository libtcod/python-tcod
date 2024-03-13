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

from __future__ import annotations

import itertools
from os import PathLike
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray

import tcod.console
from tcod._internal import _check, _console, _path_encode, _raise_tcod_error, deprecate
from tcod.cffi import ffi, lib


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
    def _claim(cls, cdata: Any) -> Tileset:  # noqa: ANN401
        """Return a new Tileset that owns the provided TCOD_Tileset* object."""
        self = object.__new__(cls)
        if cdata == ffi.NULL:
            msg = "Tileset initialized with nullptr."
            raise RuntimeError(msg)
        self._tileset_p = ffi.gc(cdata, lib.TCOD_tileset_delete)
        return self

    @classmethod
    def _from_ref(cls, tileset_p: Any) -> Tileset:  # noqa: ANN401
        self = object.__new__(cls)
        self._tileset_p = tileset_p
        return self

    @property
    def tile_width(self) -> int:
        """Width of the tile in pixels."""
        return int(lib.TCOD_tileset_get_tile_width_(self._tileset_p))

    @property
    def tile_height(self) -> int:
        """Height of the tile in pixels."""
        return int(lib.TCOD_tileset_get_tile_height_(self._tileset_p))

    @property
    def tile_shape(self) -> tuple[int, int]:
        """Shape (height, width) of the tile in pixels."""
        return self.tile_height, self.tile_width

    def __contains__(self, codepoint: int) -> bool:
        """Test if a tileset has a codepoint with ``n in tileset``."""
        return bool(lib.TCOD_tileset_get_tile_(self._tileset_p, codepoint, ffi.NULL) == 0)

    def get_tile(self, codepoint: int) -> NDArray[np.uint8]:
        """Return a copy of a tile for the given codepoint.

        If the tile does not exist yet then a blank array will be returned.

        The tile will have a shape of (height, width, rgba) and a dtype of
        uint8.  Note that most grey-scale tiles will only use the alpha
        channel and will usually have a solid white color channel.
        """
        tile: NDArray[np.uint8] = np.zeros((*self.tile_shape, 4), dtype=np.uint8)
        lib.TCOD_tileset_get_tile_(
            self._tileset_p,
            codepoint,
            ffi.from_buffer("struct TCOD_ColorRGBA*", tile),
        )
        return tile

    def set_tile(self, codepoint: int, tile: ArrayLike | NDArray[np.uint8]) -> None:
        """Upload a tile into this array.

        Args:
            codepoint (int): The Unicode codepoint you are assigning to.
                If the tile is a sprite rather than a common glyph then consider assigning it to a
                `Private Use Area <https://en.wikipedia.org/wiki/Private_Use_Areas>`_.
            tile (Union[ArrayLike, NDArray[np.uint8]]):
                The pixels to use for this tile in row-major order and must be in the same shape as :any:`tile_shape`.
                `tile` can be an RGBA array with the shape of ``(height, width, rgba)``, or a grey-scale array with the
                shape ``(height, width)``.
                The `tile` array will be converted to a dtype of ``np.uint8``.

        An RGB array as an input is too ambiguous and an alpha channel must be added, for example if an image has a key
        color than the key color pixels must have their alpha channel set to zero.

        This data may be immediately sent to VRAM, which can be a slow operation.

        Example::

            # Examples use imageio for image loading, see https://imageio.readthedocs.io
            tileset: tcod.tileset.Tileset  # This example assumes you are modifying an existing tileset.

            # Normal usage when a tile already has its own alpha channel.
            # The loaded tile must be the correct shape for the tileset you assign it to.
            # The tile is assigned to a private use area and will not conflict with any exiting codepoint.
            tileset.set_tile(0x100000, imageio.load("rgba_tile.png"))

            # Load a greyscale tile.
            tileset.set_tile(0x100001, imageio.load("greyscale_tile.png"), pilmode="L")
            # If you are stuck with an RGB array then you can use the red channel as the input: `rgb[:, :, 0]`

            # Loads an RGB sprite without a background.
            tileset.set_tile(0x100002, imageio.load("rgb_no_background.png", pilmode="RGBA"))
            # If you're stuck with an RGB array then you can pad the channel axis with an alpha of 255:
            #   rgba = np.pad(rgb, pad_width=((0, 0), (0, 0), (0, 1)), constant_values=255)

            # Loads an RGB sprite with a key color background.
            KEY_COLOR = np.asarray((255, 0, 255), dtype=np.uint8)
            sprite_rgb = imageio.load("rgb_tile.png")
            # Compare the RGB colors to KEY_COLOR, compress full matches to a 2D mask.
            sprite_mask = (sprite_rgb != KEY_COLOR).all(axis=2)
            # Generate the alpha array, with 255 as the foreground and 0 as the background.
            sprite_alpha = sprite_mask.astype(np.uint8) * 255
            # Combine the RGB and alpha arrays into an RGBA array.
            sprite_rgba = np.append(sprite_rgb, sprite_alpha, axis=2)
            tileset.set_tile(0x100003, sprite_rgba)

        """
        tile = np.ascontiguousarray(tile, dtype=np.uint8)
        if tile.shape == self.tile_shape:
            full_tile: NDArray[np.uint8] = np.empty((*self.tile_shape, 4), dtype=np.uint8)
            full_tile[:, :, :3] = 255
            full_tile[:, :, 3] = tile
            return self.set_tile(codepoint, full_tile)
        required = (*self.tile_shape, 4)
        if tile.shape != required:
            note = ""
            if len(tile.shape) == 3 and tile.shape[2] == 3:  # noqa: PLR2004
                note = (
                    "\nNote: An RGB array is too ambiguous,"
                    " an alpha channel must be added to this array to divide the background/foreground areas."
                )
            msg = f"Tile shape must be {required} or {self.tile_shape}, got {tile.shape}.{note}"
            raise ValueError(msg)
        lib.TCOD_tileset_set_tile_(
            self._tileset_p,
            codepoint,
            ffi.from_buffer("struct TCOD_ColorRGBA*", tile),
        )
        return None

    def render(self, console: tcod.console.Console) -> NDArray[np.uint8]:
        """Render an RGBA array, using console with this tileset.

        `console` is the Console object to render, this can not be the root
        console.

        The output array will be a np.uint8 array with the shape of:
        ``(con_height * tile_height, con_width * tile_width, 4)``.

        .. versionadded:: 11.9
        """
        if not console:
            msg = "'console' must not be the root console."
            raise ValueError(msg)
        width = console.width * self.tile_width
        height = console.height * self.tile_height
        out: NDArray[np.uint8] = np.empty((height, width, 4), np.uint8)
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
        with surface_p, ffi.new("SDL_Surface**", surface_p) as surface_p_p:
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

        `x` and `y` is the position of the tilesheet to assign to `codepoint`.
        This is the tile position itself, not the pixel position of the tile.
        Large values of `x` will wrap to the next row, so using `x` by itself
        is equivalent to `Tile Index` in the :any:`charmap-reference`.

        This is normally used on loaded tilesheets.  Other methods of Tileset
        creation won't have reliable tile indexes.

        .. versionadded:: 11.12
        """
        tile_i = x + y * self._tileset_p.virtual_columns
        if not (0 <= tile_i < self._tileset_p.tiles_count):
            raise IndexError(
                "Tile %i is non-existent and can't be assigned."
                " (Tileset has %i tiles.)" % (tile_i, self._tileset_p.tiles_count)
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


def load_truetype_font(path: str | PathLike[str], tile_width: int, tile_height: int) -> Tileset:
    """Return a new Tileset from a `.ttf` or `.otf` file.

    Same as :any:`set_truetype_font`, but returns a :any:`Tileset` instead.
    You can send this Tileset to :any:`set_default`.

    This function is provisional.  The API may change.
    """
    path = Path(path).resolve(strict=True)
    cdata = lib.TCOD_load_truetype_font_(_path_encode(path), tile_width, tile_height)
    if not cdata:
        raise RuntimeError(ffi.string(lib.TCOD_get_error()))
    return Tileset._claim(cdata)


@deprecate("Accessing the default tileset is deprecated.")
def set_truetype_font(path: str | PathLike[str], tile_width: int, tile_height: int) -> None:
    """Set the default tileset from a `.ttf` or `.otf` file.

    `path` is the file path for the font file.

    `tile_width` and `tile_height` are the desired size of the tiles in the new
    tileset.  The font will be scaled to fit the given `tile_height` and
    `tile_width`.

    This function must be called before :any:`libtcodpy.console_init_root`.  Once
    the root console is setup you may call this function again to change the
    font.  The tileset can be changed but the window will not be resized
    automatically.

    .. versionadded:: 9.2

    .. deprecated:: 11.13
        This function does not support contexts.
        Use :any:`load_truetype_font` instead.
    """
    path = Path(path).resolve(strict=True)
    if lib.TCOD_tileset_load_truetype_(_path_encode(path), tile_width, tile_height):
        raise RuntimeError(ffi.string(lib.TCOD_get_error()))


def load_bdf(path: str | PathLike[str]) -> Tileset:
    """Return a new Tileset from a `.bdf` file.

    For the best results the font should be monospace, cell-based, and
    single-width.  As an example, a good set of fonts would be the
    `Unicode fonts and tools for X11 <https://www.cl.cam.ac.uk/~mgk25/ucs-fonts.html>`_
    package.

    Pass the returned Tileset to :any:`tcod.tileset.set_default` and it will
    take effect when `libtcodpy.console_init_root` is called.

    .. versionadded:: 11.10
    """
    path = Path(path).resolve(strict=True)
    cdata = lib.TCOD_load_bdf(_path_encode(path))
    if not cdata:
        raise RuntimeError(ffi.string(lib.TCOD_get_error()).decode())
    return Tileset._claim(cdata)


def load_tilesheet(path: str | PathLike[str], columns: int, rows: int, charmap: Iterable[int] | None) -> Tileset:
    """Return a new Tileset from a simple tilesheet image.

    `path` is the file path to a PNG file with the tileset.

    `columns` and `rows` is the shape of the tileset.  Tiles are assumed to
    take up the entire space of the image.

    `charmap` is a sequence of codepoints to map the tilesheet to in row-major order.
    This is a list or generator of codepoints which map the tiles like this: ``charmap[tile_index] = codepoint``.
    For common tilesets `charmap` should be :any:`tcod.tileset.CHARMAP_CP437`.
    Generators will be sliced so :any:`itertools.count` can be used which will
    give all tiles the same codepoint as their index, but this will not map
    tiles onto proper Unicode.
    If `None` is used then no tiles will be mapped, you will need to use
    :any:`Tileset.remap` to assign codepoints to this Tileset.

    .. versionadded:: 11.12
    """
    path = Path(path).resolve(strict=True)
    mapping = []
    if charmap is not None:
        mapping = list(itertools.islice(charmap, columns * rows))
    cdata = lib.TCOD_tileset_load(_path_encode(path), columns, rows, len(mapping), mapping)
    if not cdata:
        _raise_tcod_error()
    return Tileset._claim(cdata)


def procedural_block_elements(*, tileset: Tileset) -> None:
    """Overwrite the block element codepoints in `tileset` with procedurally generated glyphs.

    Args:
        tileset (Tileset): A :any:`Tileset` with tiles of any shape.

    This will overwrite all of the codepoints `listed here <https://en.wikipedia.org/wiki/Block_Elements>`_
    except for the shade glyphs.

    This function is useful for other functions such as :any:`Console.draw_semigraphics` which use more types of block
    elements than are found in Code Page 437.

    .. versionadded:: 13.1

    Example::

        >>> tileset = tcod.tileset.Tileset(8, 8)
        >>> tcod.tileset.procedural_block_elements(tileset=tileset)
        >>> tileset.get_tile(0x259E)[:, :, 3]  # "▞" Quadrant upper right and lower left.
        array([[  0,   0,   0,   0, 255, 255, 255, 255],
               [  0,   0,   0,   0, 255, 255, 255, 255],
               [  0,   0,   0,   0, 255, 255, 255, 255],
               [  0,   0,   0,   0, 255, 255, 255, 255],
               [255, 255, 255, 255,   0,   0,   0,   0],
               [255, 255, 255, 255,   0,   0,   0,   0],
               [255, 255, 255, 255,   0,   0,   0,   0],
               [255, 255, 255, 255,   0,   0,   0,   0]], dtype=uint8)
        >>> tileset.get_tile(0x2581)[:, :, 3]  # "▁" Lower one eighth block.
        array([[  0,   0,   0,   0,   0,   0,   0,   0],
               [  0,   0,   0,   0,   0,   0,   0,   0],
               [  0,   0,   0,   0,   0,   0,   0,   0],
               [  0,   0,   0,   0,   0,   0,   0,   0],
               [  0,   0,   0,   0,   0,   0,   0,   0],
               [  0,   0,   0,   0,   0,   0,   0,   0],
               [  0,   0,   0,   0,   0,   0,   0,   0],
               [255, 255, 255, 255, 255, 255, 255, 255]], dtype=uint8)
       >>> tileset.get_tile(0x258D)[:, :, 3]  # "▍" Left three eighths block.
       array([[255, 255, 255,   0,   0,   0,   0,   0],
              [255, 255, 255,   0,   0,   0,   0,   0],
              [255, 255, 255,   0,   0,   0,   0,   0],
              [255, 255, 255,   0,   0,   0,   0,   0],
              [255, 255, 255,   0,   0,   0,   0,   0],
              [255, 255, 255,   0,   0,   0,   0,   0],
              [255, 255, 255,   0,   0,   0,   0,   0],
              [255, 255, 255,   0,   0,   0,   0,   0]], dtype=uint8)
    """
    quadrants: NDArray[np.uint8] = np.zeros(tileset.tile_shape, dtype=np.uint8)
    half_height = tileset.tile_height // 2
    half_width = tileset.tile_width // 2
    quadrants[:half_height, :half_width] = 0b1000  # Top-left.
    quadrants[:half_height, half_width:] = 0b0100  # Top-right.
    quadrants[half_height:, :half_width] = 0b0010  # Bottom-left.
    quadrants[half_height:, half_width:] = 0b0001  # Bottom-right.

    for codepoint, quad_mask in (
        (0x2580, 0b1100),  # "▀" Upper half block.
        (0x2584, 0b0011),  # "▄" Lower half block.
        (0x2588, 0b1111),  # "█" Full block.
        (0x258C, 0b1010),  # "▌" Left half block.
        (0x2590, 0b0101),  # "▐" Right half block.
        (0x2596, 0b0010),  # "▖" Quadrant lower left.
        (0x2597, 0b0001),  # "▗" Quadrant lower right.
        (0x2598, 0b1000),  # "▘" Quadrant upper left.
        (0x2599, 0b1011),  # "▙" Quadrant upper left and lower left and lower right.
        (0x259A, 0b1001),  # "▚" Quadrant upper left and lower right.
        (0x259B, 0b1110),  # "▛" Quadrant upper left and upper right and lower left.
        (0x259C, 0b1101),  # "▜" Quadrant upper left and upper right and lower right.
        (0x259D, 0b0100),  # "▝" Quadrant upper right.
        (0x259E, 0b0110),  # "▞" Quadrant upper right and lower left.
        (0x259F, 0b0111),  # "▟" Quadrant upper right and lower left and lower right.
    ):
        alpha: NDArray[np.uint8] = np.asarray((quadrants & quad_mask) != 0, dtype=np.uint8)
        alpha *= 255
        tileset.set_tile(codepoint, alpha)

    for codepoint, axis, fraction, negative in (
        (0x2581, 0, 7, True),  # "▁" Lower one eighth block.
        (0x2582, 0, 6, True),  # "▂" Lower one quarter block.
        (0x2583, 0, 5, True),  # "▃" Lower three eighths block.
        (0x2585, 0, 3, True),  # "▅" Lower five eighths block.
        (0x2586, 0, 2, True),  # "▆" Lower three quarters block.
        (0x2587, 0, 1, True),  # "▇" Lower seven eighths block.
        (0x2589, 1, 7, False),  # "▉" Left seven eighths block.
        (0x258A, 1, 6, False),  # "▊" Left three quarters block.
        (0x258B, 1, 5, False),  # "▋" Left five eighths block.
        (0x258D, 1, 3, False),  # "▍" Left three eighths block.
        (0x258E, 1, 2, False),  # "▎" Left one quarter block.
        (0x258F, 1, 1, False),  # "▏" Left one eighth block.
        (0x2594, 0, 1, False),  # "▔" Upper one eighth block.
        (0x2595, 1, 7, True),  # "▕" Right one eighth block .
    ):
        indexes = [slice(None), slice(None)]
        divide = tileset.tile_shape[axis] * fraction // 8
        # If negative then shade from the far corner, otherwise shade from the near corner.
        indexes[axis] = slice(divide, None) if negative else slice(None, divide)
        alpha = np.zeros(tileset.tile_shape, dtype=np.uint8)
        alpha[tuple(indexes)] = 255
        tileset.set_tile(codepoint, alpha)


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
    0x2302,
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
"""A code page 437 character mapping.

See :ref:`code-page-437` for more info and a table of glyphs.

.. versionadded:: 11.12

.. versionchanged:: 14.0
    Character at index ``0x7F`` was changed from value ``0x7F`` to the HOUSE ``⌂`` glyph ``0x2302``.
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
you might need it to load an existing font made for libtcod.

This character map is in Unicode, so old code using the non-Unicode
`tcod.CHAR_*` constants will need to be updated.

See :ref:`deprecated-tcod-layout` for a table of glyphs used in this character
map.

.. versionadded:: 11.12
"""
