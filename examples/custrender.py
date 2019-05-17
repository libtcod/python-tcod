#!/usr/bin/env python3
# To the extent possible under law, the libtcod maintainers have waived all
# copyright and related or neighboring rights for this script.  This work is
# published from: United States.
# https://creativecommons.org/publicdomain/zero/1.0/
"""A custom rendering engine for python-tcod.

This module bypasses tcod.console_flush to manually adjust where the console
will be rendered.  It uses the FFI to call libtcod and SDL2 functions directly.

It can be extend to allow arbitrary rendering on top of the SDL renderer.

It is also designed to be copied into your own project and imported as a
module.
"""
from typing import Optional, Tuple

import tcod
import tcod.tileset
import tcod.event

assert tcod.__version__ > "10.0.3", tcod.__version__


def get_renderer_size() -> Tuple[int, int]:
    """Return the renderer output size as a (width, height) tuple."""
    sdl_renderer = tcod.lib.TCOD_sys_get_sdl_renderer()
    assert sdl_renderer
    renderer_size = tcod.ffi.new("int[2]")
    tcod.lib.SDL_GetRendererOutputSize(
        sdl_renderer, renderer_size, renderer_size + 1
    )
    return renderer_size[0], renderer_size[1]


def get_viewport(
    console: tcod.console.Console,
    correct_aspect: bool = False,
    integer_scale: bool = False,
) -> Tuple[int, int, int, int]:
    """Return a viewport which follows the given constants.

    `console` is a Console object, it is used as reference for what the correct
    aspect should be.  The default tileset from `tcod.tileset` is also used as
    a reference for the current font size.

    If `correct_aspect` is True then the viewport will be letter-boxed to fit
    the screen instead of stretched.

    If `integer_scale` is True then the viewport to be scaled in integer
    proportions, this is ignored when the screen is too small.
    """
    assert tcod.sys_get_renderer() == tcod.RENDERER_SDL2
    sdl_renderer = tcod.lib.TCOD_sys_get_sdl_renderer()
    assert sdl_renderer
    tileset = tcod.tileset.get_default()
    aspect = (console.width * tileset.tile_width,
              console.height * tileset.tile_height)
    renderer_size = get_renderer_size()
    scale = renderer_size[0] / aspect[0], renderer_size[1] / aspect[1]
    if correct_aspect:
        scale = min(scale), min(scale)
    if integer_scale:
        scale = (int(scale[0]) if scale[0] >= 1 else scale[0],
                 int(scale[1]) if scale[1] >= 1 else scale[1])
    view_size = aspect[0] * scale[0], aspect[1] * scale[1]
    view_offset = ((renderer_size[0] - view_size[0]) // 2,
                   (renderer_size[1] - view_size[1]) // 2)
    return tuple(int(x) for x in (*view_offset, *view_size))  # type: ignore
    # https://github.com/python/mypy/issues/224


def clear(color: Tuple[int, int, int]) -> None:
    """Clear the SDL renderer held by libtcod with a clear color."""
    sdl_renderer = tcod.lib.TCOD_sys_get_sdl_renderer()
    assert sdl_renderer
    tcod.lib.SDL_SetRenderDrawColor(sdl_renderer, *color, 255)
    tcod.lib.SDL_RenderClear(sdl_renderer)


def present() -> None:
    """Present the SDL renderer held by libtcod to the screen."""
    sdl_renderer = tcod.lib.TCOD_sys_get_sdl_renderer()
    assert sdl_renderer
    tcod.lib.SDL_RenderPresent(sdl_renderer)


def accumulate(
    console: tcod.console.Console,
    viewport: Optional[Tuple[int, int, int, int]] = None,
) -> None:
    """Render a console to SDL's renderer.

    `console` is the console to renderer.  Background alpha is supported and
    well defined.  Foregound alpha is also supported, but not as well-defined.
    The `default tileset` will be used for graphics.

    `viewport` is where to draw the console on the screen.  If it is None then
    the console will be stretched over the full screen.  You can use
    `get_viewport` to make a viewport with specific constraints.

    You will need to call `present` yourself to show the rendered console, if
    the viewport does not cover the full screen then you'll need to call
    `clear` beforehand to clear the pixels outside of the viewport.

    This function can be called multiple times, but the current implementation
    is optimized to handle only one console.  Keep this in mind when rendering
    multiple different consoles.

    This function depends on a provisional function of the libtcod API.  You
    may want to pin your exact version of python-tcod to prevent a break.
    """
    assert tcod.sys_get_renderer() \
        in (tcod.RENDERER_SDL2, tcod.RENDERER_OPENGL2)
    if viewport is None:
        viewport = tcod.ffi.NULL
    else:
        viewport = tcod.ffi.new("struct SDL_Rect*", viewport)
    tcod.lib.TCOD_sys_accumulate_console_(console.console_c, viewport)


def main() -> None:
    """An example of of the use of this module."""
    with tcod.console_init_root(20, 4, renderer=tcod.RENDERER_SDL2) as console:
        TEXT = "Console with a fixed aspect ratio and integer scaling."
        console.print_box(0, 0, 0, 0, TEXT)
        while True:
            # Clear background with white.
            clear((255, 255, 255))
            # Draw the console to SDL's buffer.
            accumulate(console, get_viewport(console, True, True))
            # If you want you can use the FFI to do additional drawing here:
            ...
            # Present the SDL2 renderer to the display.
            present()

            for event in tcod.event.wait():
                if event.type == "QUIT":
                    raise SystemExit()
                elif event.type == "WINDOWRESIZED":
                    # You can change to a console of a different size in
                    # response to a WINDOWRESIZED event if you want.
                    ...


if __name__ == "__main__":
    main()
