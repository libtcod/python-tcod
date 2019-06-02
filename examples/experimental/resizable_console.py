#!/usr/bin/env python3
# To the extent possible under law, the libtcod maintainers have waived all
# copyright and related or neighboring rights for this script.  This work is
# published from: United States.
# https://creativecommons.org/publicdomain/zero/1.0/
"""An example showing the console being resized to fit the window."""
import tcod
import tcod.event
import tcod.tileset

import custrender  # Using the custom renderer engine.


def main() -> None:
    with tcod.console_init_root(
        20, 4, renderer=tcod.RENDERER_SDL2, vsync=True
    ) as console:
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
                    # Use the current active tileset as a reference.
                    tileset = tcod.tileset.get_default()
                    # Replace `console` with a new one of the correct size.
                    console = tcod.console.Console(
                        event.width // tileset.tile_width,
                        event.height // tileset.tile_height,
                    )


if __name__ == "__main__":
    main()
