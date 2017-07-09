#!/usr/bin/env python
"""
    first draft example of libtcod's SDL hook

    draws an animated checkerboard pattern
"""

import tcod
import tdl

# generate a callback for libtcod
@tcod.ffi.callback('SDL_renderer_t')
def sdl_hook(surface):
    # cast (void *) to (SDL_Surface *)
    surface = tcod.ffi.cast('SDL_Surface *', surface)

    # assume pixels are 32bit, mostly unsafe, but this is what libtcod uses
    pixels = tcod.ffi.cast('uint32*', surface.pixels)

    for y in range(surface.h):
        for x in range(surface.w):
            index = y * surface.w + x
            # make a simple animatied pattern
            bit = ((tick + x) % 100 < 50) ^ ((tick + y) % 100 < 50)
            pixels[index] = 0xffffffff * bit

if __name__ == '__main__':
    # hook callback to libtcod
    tcod.sys_register_SDL_renderer(sdl_hook)

    con = tdl.init(32, 32, renderer='SDL') # MUST BE SDL RENDERER

    tick = 0
    while(True):
        tick += 1
        for event in tdl.event.get():
            if event.type == 'QUIT':
                raise SystemExit()
        tdl.flush() # will call sdl_hook
