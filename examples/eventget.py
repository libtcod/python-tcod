#!/usr/bin/env python3
# To the extent possible under law, the libtcod maintainers have waived all
# copyright and related or neighboring rights for this example.  This work is
# published from: United States.
# https://creativecommons.org/publicdomain/zero/1.0/
"""An demonstration of event handling using the tcod.event module.
"""
from typing import List

import tcod
import tcod.sdl.joystick
import tcod.sdl.sys

WIDTH, HEIGHT = 720, 480


def main() -> None:
    """Example program for tcod.event"""

    event_log: List[str] = []
    motion_desc = ""
    joysticks = tcod.sdl.joystick.get_joysticks()

    with tcod.context.new(width=WIDTH, height=HEIGHT) as context:
        console = context.new_console()
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
                if isinstance(event, tcod.event.Quit):
                    raise SystemExit()
                if isinstance(event, tcod.event.WindowResized) and event.type == "WINDOWRESIZED":
                    console = context.new_console()
                if isinstance(event, tcod.event.MouseMotion):
                    motion_desc = str(event)
                else:
                    event_log.append(str(event))


if __name__ == "__main__":
    main()
