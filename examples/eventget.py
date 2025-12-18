#!/usr/bin/env python
# To the extent possible under law, the libtcod maintainers have waived all
# copyright and related or neighboring rights for this example.  This work is
# published from: United States.
# https://creativecommons.org/publicdomain/zero/1.0/
"""An demonstration of event handling using the tcod.event module."""

import tcod.context
import tcod.event
import tcod.sdl.joystick

WIDTH, HEIGHT = 1280, 720


def main() -> None:  # noqa: C901, PLR0912
    """Example program for tcod.event."""
    event_log: list[str] = []
    motion_desc = ""
    tcod.sdl.joystick.init()
    controllers: set[tcod.sdl.joystick.GameController] = set()
    joysticks: set[tcod.sdl.joystick.Joystick] = set()

    with tcod.context.new(width=WIDTH, height=HEIGHT) as context:
        if context.sdl_window:
            context.sdl_window.start_text_input()
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
                match event:
                    case tcod.event.Quit():
                        raise SystemExit
                    case tcod.event.WindowResized(type="WindowResized"):
                        console = context.new_console()
                    case tcod.event.ControllerDevice(type="CONTROLLERDEVICEADDED", controller=controller):
                        controllers.add(controller)
                    case tcod.event.ControllerDevice(type="CONTROLLERDEVICEREMOVED", controller=controller):
                        controllers.remove(controller)
                    case tcod.event.JoystickDevice(type="JOYDEVICEADDED", joystick=joystick):
                        joysticks.add(joystick)
                    case tcod.event.JoystickDevice(type="JOYDEVICEREMOVED", joystick=joystick):
                        joysticks.remove(joystick)
                    case tcod.event.MouseMotion():
                        motion_desc = str(event)
                    case _:  # Log all events other than MouseMotion.
                        event_log.append(repr(event))


if __name__ == "__main__":
    main()
