#!/usr/bin/env python3
"""Render a console with a fixed aspect ratio.

This script bypasses tcod.console_flush to manually adjust where the console
will be rendered.  It uses an FFI to call libtcod and SDL2 functions directly.
"""
from typing import Any, Tuple

import tcod
import tcod.event


def get_renderer_size(sdl_renderer: Any) -> Tuple[int, int]:
    """Return the renderer size as a (width, height) tuple."""
    renderer_size = tcod.ffi.new("int[2]")
    tcod.lib.SDL_GetRendererOutputSize(
        sdl_renderer, renderer_size, renderer_size + 1
    )
    return renderer_size[0], renderer_size[1]


def get_viewport(sdl_renderer: Any, aspect: Tuple[int, int]) -> Any:
    """Return an SDL_Rect object that will fit this renderer with this aspect
    ratio."""
    current_size = get_renderer_size(sdl_renderer)
    scale = min(x / y for x, y in zip(current_size, aspect))
    view_size = [round(x * scale) for x in aspect]
    view_offset = [(x - y) // 2 for x, y in zip(current_size, view_size)]
    return tcod.ffi.new("SDL_Rect*", (*view_offset, *view_size))


def main() -> None:
    with tcod.console_init_root(20, 4, renderer=tcod.RENDERER_SDL2) as console:
        # Get the SDL2 objects setup by libtcod.
        sdl_window = tcod.lib.TCOD_sys_get_sdl_window()
        sdl_renderer = tcod.lib.TCOD_sys_get_sdl_renderer()
        # Aspect is generally console_size * tile_size.
        aspect: Tuple[int, int] = get_renderer_size(sdl_renderer)
        console.print_box(0, 0, 0, 0, "Console with a fixed aspect ratio.")
        while True:
            # Clear background with white.
            tcod.lib.SDL_SetRenderDrawColor(sdl_renderer, 255, 255, 255, 255)
            tcod.lib.SDL_RenderClear(sdl_renderer)
            # Accumulate console graphics.
            # This next function is provisional, the API is not stable.
            tcod.lib.TCOD_sys_accumulate_console_(
                console.console_c, get_viewport(sdl_renderer, aspect)
            )
            # Present the SDL2 renderer to the display.
            tcod.lib.SDL_RenderPresent(sdl_renderer)

            for event in tcod.event.wait():
                if event.type == "QUIT":
                    raise SystemExit()


if __name__ == "__main__":
    main()
