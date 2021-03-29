#!/usr/bin/env python3
"""A TrueType Font example using the FreeType library.

You will need to get this external library from PyPI:

    pip install freetype-py

This script has known issues and may crash when the window is resized.
"""
from typing import Tuple

import freetype  # type: ignore  # pip install freetype-py
import numpy as np
import tcod

FONT = "VeraMono.ttf"


def load_ttf(path: str, size: Tuple[int, int]) -> tcod.tileset.Tileset:
    """Load a TTF file as a tcod tileset."""
    ttf = freetype.Face(path)
    ttf.set_pixel_sizes(*size)
    half_advance = size[0] - (ttf.bbox.xMax - ttf.bbox.xMin) // 64

    tileset = tcod.tileset.Tileset(*size)
    for codepoint, glyph_index in ttf.get_chars():
        ttf.load_glyph(glyph_index)
        bitmap = ttf.glyph.bitmap
        assert bitmap.pixel_mode == freetype.FT_PIXEL_MODE_GRAY
        bitmap_array = np.asarray(bitmap.buffer).reshape(
            (bitmap.width, bitmap.rows), order="F"
        )
        if bitmap_array.size == 0:
            continue
        output_image = np.zeros(size, dtype=np.uint8, order="F")
        out_slice = output_image
        left = ttf.glyph.bitmap_left + half_advance
        top = size[1] - ttf.glyph.bitmap_top + ttf.bbox.yMin // 64
        out_slice = out_slice[max(0, left) :, max(0, top) :]
        out_slice[
            : bitmap_array.shape[0], : bitmap_array.shape[1]
        ] = bitmap_array[: out_slice.shape[0], : out_slice.shape[1]]
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
            console.tiles_rgb["bg"][::2, ::2] = 0x20
            console.tiles_rgb["bg"][1::2, 1::2] = 0x20
            console.tiles_rgb["ch"][:16, :6] = np.arange(0x20, 0x80).reshape(
                0x10, -1, order="F"
            )
            console.print(0, 7, "Example text.")
            context.present(console, integer_scaling=True)
            for event in tcod.event.wait():
                if isinstance(event, tcod.event.Quit):
                    raise SystemExit()
                if (
                    isinstance(event, tcod.event.WindowResized)
                    and event.type == "WINDOWSIZECHANGED"
                ):
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
    main()
