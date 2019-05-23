#!/usr/bin/env python3
# To the extent possible under law, the libtcod maintainers have waived all
# copyright and related or neighboring rights for the "hello world" PyInstaller
# example script.  This work is published from: United States.
# https://creativecommons.org/publicdomain/zero/1.0/
import sys
import os.path

import tcod
import tcod.event

WIDTH, HEIGHT = 80, 60

# The base directory, this is sys._MEIPASS when in one-file mode.
BASE_DIR = getattr(sys, "_MEIPASS", ".")

FONT_PATH = os.path.join(BASE_DIR, "terminal8x8_gs_ro.png")


def main():
    tcod.console_set_custom_font(FONT_PATH, tcod.FONT_LAYOUT_CP437)
    with tcod.console_init_root(
        WIDTH, HEIGHT, renderer=tcod.RENDERER_SDL2, vsync=True
    ) as console:
        while True:
            console.clear()
            console.print(0, 0, "Hello World")
            tcod.console_flush()

            for event in tcod.event.wait():
                if event.type == "QUIT":
                    raise SystemExit()


if __name__ == "__main__":
    main()
