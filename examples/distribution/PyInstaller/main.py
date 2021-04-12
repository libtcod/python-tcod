#!/usr/bin/env python3
# To the extent possible under law, the libtcod maintainers have waived all
# copyright and related or neighboring rights for the "hello world" PyInstaller
# example script.  This work is published from: United States.
# https://creativecommons.org/publicdomain/zero/1.0/
import os.path
import sys

import tcod

WIDTH, HEIGHT = 80, 60

# The base directory, this is sys._MEIPASS when in one-file mode.
BASE_DIR = getattr(sys, "_MEIPASS", ".")

FONT_PATH = os.path.join(BASE_DIR, "data/terminal8x8_gs_ro.png")


def main():
    tileset = tcod.tileset.load_tilesheet(
        FONT_PATH, 16, 16, tcod.tileset.CHARMAP_CP437
    )
    with tcod.context.new(
        columns=WIDTH, rows=HEIGHT, tileset=tileset
    ) as context:
        while True:
            console = tcod.console.Console(WIDTH, HEIGHT)
            console.print(0, 0, "Hello World")
            context.present(console)
            for event in tcod.event.wait():
                if isinstance(event, tcod.event.Quit):
                    raise SystemExit()


if __name__ == "__main__":
    main()
