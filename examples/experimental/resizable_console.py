#!/usr/bin/env python3
# To the extent possible under law, the libtcod maintainers have waived all
# copyright and related or neighboring rights for this script.  This work is
# published from: United States.
# https://creativecommons.org/publicdomain/zero/1.0/
"""An example showing the console being resized to fit the window."""
from typing import Tuple

import tcod
import tcod.event
import tcod.tileset

import custrender  # Using the custom renderer engine.


def fit_console(width: int, height: int) -> Tuple[int, int]:
    """Return a console resolution the fits the given pixel resolution."""
    # Use the current active tileset as a reference.
    tileset = tcod.tileset.get_default()
    return width // tileset.tile_width, height // tileset.tile_height


def main() -> None:
    window_flags = (
        tcod.lib.SDL_WINDOW_RESIZABLE | tcod.lib.SDL_WINDOW_MAXIMIZED
    )
    renderer_flags = tcod.lib.SDL_RENDERER_PRESENTVSYNC
    with custrender.init_sdl2(640, 480, None, window_flags, renderer_flags):
        console = tcod.console.Console(
            *fit_console(*custrender.get_renderer_size())
        )
        TEXT = "Resizable console with no stretching."
        while True:
            console.clear()

            # Draw the checkerboard pattern.
            console.tiles["bg"][::2, ::2] = (32, 32, 32, 255)
            console.tiles["bg"][1::2, 1::2] = (32, 32, 32, 255)

            console.print_box(0, 0, 0, 0, TEXT)

            # These functions are explained in `custrender.py`.
            custrender.clear((0, 0, 0))
            custrender.accumulate(
                console, custrender.get_viewport(console, True, True)
            )
            custrender.present()

            for event in tcod.event.wait():
                if event.type == "QUIT":
                    raise SystemExit()
                elif event.type == "WINDOWRESIZED":
                    # Replace `console` with a new one of the correct size.
                    console = tcod.console.Console(
                        *fit_console(event.width, event.height)
                    )


if __name__ == "__main__":
    main()
