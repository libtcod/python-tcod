#!/usr/bin/env python
"""cx_Freeze main script example."""

import tcod.console
import tcod.context
import tcod.event
import tcod.tileset

WIDTH, HEIGHT = 80, 60
console = None


def main() -> None:
    """Entry point function."""
    tileset = tcod.tileset.load_tilesheet("data/terminal8x8_gs_ro.png", 16, 16, tcod.tileset.CHARMAP_CP437)
    with tcod.context.new(columns=WIDTH, rows=HEIGHT, tileset=tileset) as context:
        while True:
            console = tcod.console.Console(WIDTH, HEIGHT)
            console.print(0, 0, "Hello World")
            context.present(console)
            for event in tcod.event.wait():
                if isinstance(event, tcod.event.Quit):
                    raise SystemExit()


if __name__ == "__main__":
    main()
