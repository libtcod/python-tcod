#!/usr/bin/env python
"""
    example of libtcod's SDL hook

    draws a simple white square.
"""

import tcod
import tdl

import numpy as np

# generate a callback for libtcod
@tcod.ffi.callback('SDL_renderer_t')
def sdl_hook(surface):
    tcod.lib.SDL_UpperBlit(my_surface, tcod.ffi.NULL, surface, [{'x':0, 'y':0}])

pixels = np.zeros((100, 150, 4), dtype=np.uint8)
my_surface = tcod.lib.SDL_CreateRGBSurfaceWithFormatFrom(
    tcod.ffi.cast('void*', pixels.ctypes.data),
    pixels.shape[1], pixels.shape[0], 32,
    pixels.strides[0],
    tcod.lib.SDL_PIXELFORMAT_RGBA32,
)


if __name__ == '__main__':
    # hook callback to libtcod
    tcod.sys_register_SDL_renderer(sdl_hook)

    con = tdl.init(32, 32, renderer='SDL') # MUST BE SDL RENDERER

    pixels[:] = 255

    tick = 0
    while(True):
        tick += 1
        for event in tdl.event.get():
            if event.type == 'QUIT':
                raise SystemExit()
        tdl.flush() # will call sdl_hook
