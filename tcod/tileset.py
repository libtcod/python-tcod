"""Tileset and font related functions.
"""
import os

from tcod.libtcod import lib


def set_truetype_font(path: str, tile_width: int, tile_height: int) -> None:
    """Set the default tileset from a `.ttf` or `.otf` file.

    `path` is the file path for the font file.

    `tile_width` and `tile_height` are the desired size of the tiles in the new
    tileset.  The font will be scaled to fit the `tile_height` and may be
    clipped to fit inside of the `tile_width`.

    This function will only affect the `SDL2` and `OPENGL2` renderers.

    This function must be called before :any:`tcod.console_init_root`.  Once
    the root console is setup you may call this funtion again to change the
    font.  The tileset can be changed but the window will not be resized
    automatically.

    .. versionadded:: 9.2
    """
    if not os.path.exists(path):
        raise RuntimeError("File not found:\n\t%s" % (os.path.realpath(path),))
    lib.TCOD_tileset_load_truetype_(path.encode(), tile_width, tile_height)
