#!/usr/bin/env python3

import tcod
import tcod.event


def main():
    """Example program for tcod.event"""
    WIDTH, HEIGHT = 120, 60
    TITLE = None

    with tcod.console_init_root(WIDTH, HEIGHT, TITLE, order="F") as console:
        tcod.sys_set_fps(24)
        while True:
            tcod.console_flush()
            for event in tcod.event.wait():
                print(event)
                if event.type == "QUIT":
                    raise SystemExit()
                elif event.type == "MOUSEMOTION":
                    console.ch[:, -1] = 0
                    console.print_(0, HEIGHT - 1, repr(event))
                else:
                    console.blit(console, 0, 0, 0, 1, WIDTH, HEIGHT - 2)
                    console.ch[:, -3] = 0
                    console.print_(0, HEIGHT - 3, repr(event))


if __name__ == "__main__":
    main()
