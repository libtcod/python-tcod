#!/usr/bin/env python3
"""A TrueType Font example using the FreeType library.

You will need to get this external library from PyPI:

    pip install freetype-py
"""
# To the extent possible under law, the libtcod maintainers have waived all
# copyright and related or neighboring rights to this example script.
# https://creativecommons.org/publicdomain/zero/1.0/
from typing import Tuple

import freetype  # type: ignore  # pip install freetype-py
import numpy as np
import tcod
from numpy.typing import NDArray

FONT = "VeraMono.ttf"


def load_ttf(path: str, size: Tuple[int, int]) -> tcod.tileset.Tileset:
    """Load a TTF file and return a tcod Tileset.

    `path` is the file path to the font, this can be any font supported by the
    FreeType library.

    `size` is the (width, height) of the Tileset in pixels.

    Feel free to use this function in your own code.
    """
    ttf = freetype.Face(path)
    ttf.set_pixel_sizes(*size)

    tileset = tcod.tileset.Tileset(*size)
    for codepoint, glyph_index in ttf.get_chars():
        # Add every glyph to the Tileset.
        ttf.load_glyph(glyph_index)
        bitmap = ttf.glyph.bitmap
        assert bitmap.pixel_mode == freetype.FT_PIXEL_MODE_GRAY
        bitmap_array: NDArray[np.uint8] = np.asarray(bitmap.buffer).reshape((bitmap.width, bitmap.rows), order="F")
        if bitmap_array.size == 0:
            continue  # Skip blank glyphs.
        output_image: NDArray[np.uint8] = np.zeros(size, dtype=np.uint8, order="F")
        out_slice = output_image

        # Adjust the position to center this glyph on the tile.
        left = (size[0] - bitmap.width) // 2
        top = size[1] - ttf.glyph.bitmap_top + ttf.size.descender // 64

        # `max` is used because I was too lazy to properly slice the array.
        out_slice = out_slice[max(0, left) :, max(0, top) :]
        out_slice[: bitmap_array.shape[0], : bitmap_array.shape[1]] = bitmap_array[
            : out_slice.shape[0], : out_slice.shape[1]
        ]

        tileset.set_tile(codepoint, output_image.transpose())
    return tileset


def main() -> None:
    console = tcod.Console(16, 12, order="F")
    with tcod.context.new(
        columns=console.width,
        rows=console.height,
        tileset=load_ttf(FONT, (24, 24)),
    ) as context:
        while True:
            console.clear()
            # Draw checkerboard.
            console.tiles_rgb["bg"][::2, ::2] = 0x20
            console.tiles_rgb["bg"][1::2, 1::2] = 0x20
            # Print ASCII characters.
            console.tiles_rgb["ch"][:16, :6] = np.arange(0x20, 0x80).reshape(0x10, -1, order="F")
            console.print(0, 7, "Example text.")
            context.present(console, integer_scaling=True)
            for event in tcod.event.wait():
                if isinstance(event, tcod.event.Quit):
                    raise SystemExit()
                if isinstance(event, tcod.event.WindowResized) and event.type == "WINDOWSIZECHANGED":
                    # Resize the Tileset to match the new screen size.
                    context.change_tileset(
                        load_ttf(
                            path=FONT,
                            size=(
                                event.width // console.width,
                                event.height // console.height,
                            ),
                        )
                    )


if __name__ == "__main__":
    tcod_version = tuple(int(n) for n in tcod.__version__.split(".") if n.isdigit())
    assert tcod_version[:2] >= (12, 1), "Must be using tcod 12.1 or later."
    main()
