#!/usr/bin/env python3
# To the extent possible under law, the libtcod maintainers have waived all
# copyright and related or neighboring rights for this example.  This work is
# published from: United States.
# https://creativecommons.org/publicdomain/zero/1.0/
"""An demonstration of event handling using the tcod.event module.
"""
from typing import List

import tcod


WIDTH, HEIGHT = 720, 480
FLAGS = tcod.context.SDL_WINDOW_RESIZABLE | tcod.context.SDL_WINDOW_MAXIMIZED


def main() -> None:
    """Example program for tcod.event"""

    event_log: List[str] = []
    motion_desc = ""

    with tcod.context.new(
        width=WIDTH, height=HEIGHT, sdl_window_flags=FLAGS
    ) as context:
        console = tcod.Console(*context.recommended_console_size())
        while True:
            # Display all event items.
            console.clear()
            console.print(0, console.height - 1, motion_desc)
            for i, item in enumerate(event_log[::-1]):
                y = console.height - 3 - i
                if y < 0:
                    break
                console.print(0, y, item)
            context.present(console, integer_scaling=True)

            # Handle events.
            for event in tcod.event.wait():
                context.convert_event(event)  # Set tile coordinates for event.
                print(repr(event))
                if event.type == "QUIT":
                    raise SystemExit()
                if event.type == "WINDOWRESIZED":
                    console = tcod.Console(*context.recommended_console_size())
                if event.type == "MOUSEMOTION":
                    motion_desc = str(event)
                else:
                    event_log.append(str(event))


if __name__ == "__main__":
    main()
