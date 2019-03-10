#!/usr/bin/env python3

import tcod

WIDTH, HEIGHT = 80, 60

key = tcod.Key()
mouse = tcod.Mouse()

with tcod.console_init_root(WIDTH, HEIGHT, 'tcod events example',
                            renderer=tcod.RENDERER_SDL) as console:
    tcod.sys_set_fps(24)
    while not tcod.console_is_window_closed():
        ev = tcod.sys_wait_for_event(tcod.EVENT_ANY, key, mouse, False)
        if ev & tcod.EVENT_KEY:
            console.blit(console, 0, 0, 0, 1, WIDTH, HEIGHT - 2)
            console.print_(0, HEIGHT - 3, repr(key))
            print(key)
        if ev & tcod.EVENT_MOUSE_MOVE:
            console.rect(0, HEIGHT - 1, WIDTH, 1, True)
            console.print_(0, HEIGHT - 1, repr(mouse))
            print(mouse)
        elif ev & tcod.EVENT_MOUSE:
            console.blit(console, 0, 0, 0, 1, WIDTH, HEIGHT - 2)
            console.print_(0, HEIGHT - 3, repr(mouse))
            print(mouse)
        tcod.console_flush()
