"""This module is used to create and handle libtcod contexts.

See :ref:`getting-started` for beginner examples on how to use this module.

:any:`Context`'s are intended to replace several libtcod functions such as
:any:`tcod.console_init_root`, :any:`tcod.console_flush`,
:any:`tcod.console.recommended_size`, and many other functions which rely on
hidden global objects within libtcod.  If you begin using contexts then
most of these functions will no longer work properly.

Instead of calling :any:`tcod.console_init_root` you can call
:any:`tcod.context.new` with different keywords depending on how you plan
to setup the size of the console.  You should use
:any:`tcod.tileset` to load the font for a context.

.. note::
    If you use contexts then you should expect the following functions to no
    longer be available, because these functions rely on a global console,
    tileset, or some other kind of global state:

    :any:`tcod.console_init_root`,
    :any:`tcod.console_set_custom_font`,
    :any:`tcod.console_flush`,
    :any:`tcod.console_is_fullscreen`,
    :any:`tcod.console_is_window_closed`,
    :any:`tcod.console_set_fade`,
    :any:`tcod.console_set_fullscreen`,
    :any:`tcod.console_set_window_title`,
    :any:`tcod.sys_set_fps`,
    :any:`tcod.sys_get_last_frame_length`,
    :any:`tcod.sys_set_renderer`,
    :any:`tcod.sys_save_screenshot`,
    :any:`tcod.sys_force_fullscreen_resolution`,
    :any:`tcod.sys_get_current_resolution`,
    :any:`tcod.sys_register_SDL_renderer`,
    :any:`tcod.console_map_ascii_code_to_font`,
    :any:`tcod.sys_get_char_size`,
    :any:`tcod.sys_update_char`,
    :any:`tcod.console.recommended_size`,
    :any:`tcod.tileset.get_default`,
    :any:`tcod.tileset.set_default`.

    Some event functions can no longer return tile coordinates for the mouse:
    :any:`tcod.sys_check_for_event`,
    :any:`tcod.sys_wait_for_event`,
    :any:`tcod.mouse_get_status`.

.. versionadded:: 11.12
"""  # noqa: E501
import os
import sys
from typing import Any, Iterable, List, Optional, Tuple, Union

from typing_extensions import Literal

import tcod
import tcod.event
import tcod.tileset
from tcod._internal import _check, _check_warn, deprecate, pending_deprecate
from tcod.loader import ffi, lib

__all__ = (
    "Context",
    "new",
    "new_window",
    "new_terminal",
    "SDL_WINDOW_FULLSCREEN",
    "SDL_WINDOW_FULLSCREEN_DESKTOP",
    "SDL_WINDOW_HIDDEN",
    "SDL_WINDOW_BORDERLESS",
    "SDL_WINDOW_RESIZABLE",
    "SDL_WINDOW_MINIMIZED",
    "SDL_WINDOW_MAXIMIZED",
    "SDL_WINDOW_INPUT_GRABBED",
    "SDL_WINDOW_ALLOW_HIGHDPI",
    "RENDERER_OPENGL",
    "RENDERER_OPENGL2",
    "RENDERER_SDL",
    "RENDERER_SDL2",
)

SDL_WINDOW_FULLSCREEN = lib.SDL_WINDOW_FULLSCREEN
"""Exclusive fullscreen mode.

It's generally not recommended to use this flag unless you know what you're
doing.
`SDL_WINDOW_FULLSCREEN_DESKTOP` should be used instead whenever possible.
"""
SDL_WINDOW_FULLSCREEN_DESKTOP = lib.SDL_WINDOW_FULLSCREEN_DESKTOP
"""A borderless fullscreen window at the desktop resolution."""
SDL_WINDOW_HIDDEN = lib.SDL_WINDOW_HIDDEN
"""Window is hidden."""
SDL_WINDOW_BORDERLESS = lib.SDL_WINDOW_BORDERLESS
"""Window has no decorative border."""
SDL_WINDOW_RESIZABLE = lib.SDL_WINDOW_RESIZABLE
"""Window can be resized."""
SDL_WINDOW_MINIMIZED = lib.SDL_WINDOW_MINIMIZED
"""Window is minimized."""
SDL_WINDOW_MAXIMIZED = lib.SDL_WINDOW_MAXIMIZED
"""Window is maximized."""
SDL_WINDOW_INPUT_GRABBED = lib.SDL_WINDOW_INPUT_GRABBED
"""Window has grabbed the input."""
SDL_WINDOW_ALLOW_HIGHDPI = lib.SDL_WINDOW_ALLOW_HIGHDPI
"""High DPI mode, see the SDL documentation."""

RENDERER_OPENGL = lib.TCOD_RENDERER_OPENGL
"""A renderer for older versions of OpenGL.

Should support OpenGL 1 and GLES 1
"""
RENDERER_OPENGL2 = lib.TCOD_RENDERER_OPENGL2
"""An SDL2/OPENGL2 renderer.  Usually faster than regular SDL2.

Recommended if you need a high performance renderer.

Should support OpenGL 2.0 and GLES 2.0.
"""
RENDERER_SDL = lib.TCOD_RENDERER_SDL
"""Same as RENDERER_SDL2, but forces SDL2 into software mode."""
RENDERER_SDL2 = lib.TCOD_RENDERER_SDL2
"""The main SDL2 renderer.

Rendering is decided by SDL2 and can be changed by using an SDL2 hint:
https://wiki.libsdl.org/SDL_HINT_RENDER_DRIVER
"""


def _handle_tileset(tileset: Optional[tcod.tileset.Tileset]) -> Any:
    """Get the TCOD_Tileset pointer from a Tileset or return a NULL pointer."""
    return tileset._tileset_p if tileset else ffi.NULL


def _handle_title(title: Optional[str]) -> Any:
    """Return title as a CFFI string.

    If title is None then return a decent default title is returned.
    """
    if title is None:
        title = os.path.basename(sys.argv[0])
    return ffi.new("char[]", title.encode("utf-8"))


class Context:
    """Context manager for libtcod context objects.

    You should use :any:`tcod.context.new_terminal` or
    :any:`tcod.context.new_window` to create a new context.
    """

    def __init__(self, context_p: Any):
        """Creates a context from a cffi pointer."""
        self._context_p = context_p

    @classmethod
    def _claim(cls, context_p: Any) -> "Context":
        return cls(ffi.gc(context_p, lib.TCOD_context_delete))

    def __enter__(self) -> "Context":
        """This context can be used as a context manager."""
        return self

    def close(self) -> None:
        """Delete the context, closing any windows opened by this context.

        This instance is invalid after this call."""
        if hasattr(self, "_context_p"):
            ffi.release(self._context_p)
            del self._context_p

    def __exit__(self, *args: Any) -> None:
        """The libtcod context is closed as this context manager exits."""
        self.close()

    def present(
        self,
        console: tcod.console.Console,
        *,
        keep_aspect: bool = False,
        integer_scaling: bool = False,
        clear_color: Tuple[int, int, int] = (0, 0, 0),
        align: Tuple[float, float] = (0.5, 0.5)
    ) -> None:
        """Present a console to this context's display.

        `console` is the console you want to present.

        If `keep_aspect` is True then the console aspect will be preserved with
        a letterbox.  Otherwise the console will be stretched to fill the
        screen.

        If `integer_scaling` is True then the console will be scaled in integer
        increments.  This will have no effect if the console must be shrunk.
        You can use :any:`tcod.console.recommended_size` to create a console
        which will fit the window without needing to be scaled.

        `clear_color` is an RGB tuple used to clear the screen before the
        console is presented, this will affect the border/letterbox color.

        `align` is an (x, y) tuple determining where the console will be placed
        when letter-boxing exists.  Values of 0 will put the console at the
        upper-left corner.  Values of 0.5 will center the console.
        """
        clear_rgba = (clear_color[0], clear_color[1], clear_color[2], 255)
        viewport_args = ffi.new(
            "TCOD_ViewportOptions*",
            {
                "tcod_version": lib.TCOD_COMPILEDVERSION,
                "keep_aspect": keep_aspect,
                "integer_scaling": integer_scaling,
                "clear_color": clear_rgba,
                "align_x": align[0],
                "align_y": align[1],
            },
        )
        _check(
            lib.TCOD_context_present(
                self._context_p, console.console_c, viewport_args
            )
        )

    def pixel_to_tile(self, x: int, y: int) -> Tuple[int, int]:
        """Convert window pixel coordinates to tile coordinates."""
        with ffi.new("int[2]", (x, y)) as xy:
            _check(
                lib.TCOD_context_screen_pixel_to_tile_i(
                    self._context_p, xy, xy + 1
                )
            )
            return xy[0], xy[1]

    def pixel_to_subtile(self, x: int, y: int) -> Tuple[float, float]:
        """Convert window pixel coordinates to sub-tile coordinates."""
        with ffi.new("double[2]", (x, y)) as xy:
            _check(
                lib.TCOD_context_screen_pixel_to_tile_d(
                    self._context_p, xy, xy + 1
                )
            )
            return xy[0], xy[1]

    def convert_event(self, event: tcod.event.Event) -> None:
        """Fill in the tile coordinates of a mouse event using this context."""
        if isinstance(event, (tcod.event.MouseState, tcod.event.MouseMotion)):
            event.tile = tcod.event.Point(*self.pixel_to_tile(*event.pixel))
        if isinstance(event, tcod.event.MouseMotion):
            prev_tile = self.pixel_to_tile(
                event.pixel[0] - event.pixel_motion[0],
                event.pixel[1] - event.pixel_motion[1],
            )
            event.tile_motion = tcod.event.Point(
                event.tile[0] - prev_tile[0], event.tile[1] - prev_tile[1]
            )

    def save_screenshot(self, path: Optional[str] = None) -> None:
        """Save a screen-shot to the given file path."""
        c_path = path.encode("utf-8") if path is not None else ffi.NULL
        _check(lib.TCOD_context_save_screenshot(self._context_p, c_path))

    def change_tileset(self, tileset: Optional[tcod.tileset.Tileset]) -> None:
        """Change the active tileset used by this context."""
        _check(
            lib.TCOD_context_change_tileset(
                self._context_p, _handle_tileset(tileset)
            )
        )

    def new_console(
        self,
        min_columns: int = 1,
        min_rows: int = 1,
        magnification: float = 1.0,
        order: Union[Literal["C"], Literal["F"]] = "C",
    ) -> tcod.console.Console:
        """Return a new console sized for this context.

        `min_columns` and `min_rows` are the minimum size to use for the new
        console.

        `magnification` determines the apparent size of the tiles on the output
        display.  A `magnification` larger then 1.0 will output smaller
        consoles, which will show as larger tiles when presented.
        `magnification` must be greater than zero.

        `order` is passed to :any:`tcod.console.Console` to determine the
        memory order of its NumPy attributes.

        The times where it is the most useful to call this method are:

        * After the context is created, even if the console was given a
          specific size.
        * After the :any:`change_tileset` method is called.
        * After any window resized event, or any manual resizing of the window.

        .. versionadded:: 11.18

        .. versionchanged:: 11.19
            Added `order` parameter.

        .. seealso::
            :any:`tcod.console.Console`
        """
        if magnification < 0:
            raise ValueError(
                "Magnification must be greater than zero. (Got %f)"
                % magnification
            )
        size = ffi.new("int[2]")
        _check(
            lib.TCOD_context_recommended_console_size(
                self._context_p, magnification, size, size + 1
            )
        )
        width, height = max(min_columns, size[0]), max(min_rows, size[1])
        return tcod.console.Console(width, height, order=order)

    def recommended_console_size(
        self, min_columns: int = 1, min_rows: int = 1
    ) -> Tuple[int, int]:
        """Return the recommended (columns, rows) of a console for this
        context.

        `min_columns`, `min_rows` are the lowest values which will be returned.

        If result is only used to create a new console then you may want to
        call :any:`Context.new_console` instead.
        """
        with ffi.new("int[2]") as size:
            _check(
                lib.TCOD_context_recommended_console_size(
                    self._context_p, 1.0, size, size + 1
                )
            )
            return max(min_columns, size[0]), max(min_rows, size[1])

    @property
    def renderer_type(self) -> int:
        """Return the libtcod renderer type used by this context."""
        return _check(lib.TCOD_context_get_renderer_type(self._context_p))

    @property
    def sdl_window_p(self) -> Any:
        '''A cffi `SDL_Window*` pointer.  This pointer might be NULL.

        This pointer will become invalid if the context is closed or goes out
        of scope.

        Python-tcod's FFI provides most SDL functions.  So it's possible for
        anyone with the SDL2 documentation to work directly with SDL's
        pointers.

        Example::

            import tcod

            def toggle_fullscreen(context: tcod.context.Context) -> None:
                """Toggle a context window between fullscreen and windowed modes."""
                if not context.sdl_window_p:
                    return
                fullscreen = tcod.lib.SDL_GetWindowFlags(context.sdl_window_p) & (
                    tcod.lib.SDL_WINDOW_FULLSCREEN | tcod.lib.SDL_WINDOW_FULLSCREEN_DESKTOP
                )
                tcod.lib.SDL_SetWindowFullscreen(
                    context.sdl_window_p,
                    0 if fullscreen else tcod.lib.SDL_WINDOW_FULLSCREEN_DESKTOP,
                )
        '''  # noqa: E501
        return lib.TCOD_context_get_sdl_window(self._context_p)


@ffi.def_extern()  # type: ignore
def _pycall_cli_output(catch_reference: Any, output: Any) -> None:
    """Callback for the libtcod context CLI.  Catches the CLI output."""
    catch = ffi.from_handle(catch_reference)  # type: List[str]
    catch.append(ffi.string(output).decode("utf-8"))


def new(
    *,
    x: Optional[int] = None,
    y: Optional[int] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    columns: Optional[int] = None,
    rows: Optional[int] = None,
    renderer: Optional[int] = None,
    tileset: Optional[tcod.tileset.Tileset] = None,
    vsync: bool = True,
    sdl_window_flags: Optional[int] = None,
    title: Optional[str] = None,
    argv: Optional[Iterable[str]] = None
) -> Context:
    """Create a new context with the desired pixel size.

    `x`, `y`, `width`, and `height` are the desired position and size of the
    window.  If these are None then they will be derived from `columns` and
    `rows`.  So if you plan on having a console of a fixed size then you should
    set `columns` and `rows` instead of the window keywords.

    `columns` and `rows` is the desired size of the console.  Can be left as
    `None` when you're setting a context by a window size instead of a console.

    Providing no size information at all is also acceptable.

    `renderer` is the desired libtcod renderer to use.
    Typical options are :any:`tcod.context.RENDERER_OPENGL2` for a faster
    renderer or :any:`tcod.context.RENDERER_SDL2` for a reliable renderer.

    `tileset` is the font/tileset for the new context to render with.
    The fall-back tileset available from passing None is useful for
    prototyping, but will be unreliable across platforms.

    `vsync` is the Vertical Sync option for the window.  The default of True
    is recommended but you may want to use False for benchmarking purposes.

    `sdl_window_flags` is a bit-field of SDL window flags, if None is given
    then a default of :any:`tcod.context.SDL_WINDOW_RESIZABLE` is used.
    There's more info on the SDL documentation:
    https://wiki.libsdl.org/SDL_CreateWindow#Remarks

    `title` is the desired title of the window.

    `argv` these arguments are passed to libtcod and allow an end-user to make
    last second changes such as forcing fullscreen or windowed mode, or
    changing the libtcod renderer.
    By default this will pass in `sys.argv` but you can disable this feature
    by providing an empty list instead.
    Certain commands such as ``-help`` will raise a SystemExit exception from
    this function with the output message.

    When a window size is given instead of a console size then you can use
    :any:`Context.recommended_console_size` to automatically find the size of
    the console which should be used.

    .. versionadded:: 11.16
    """
    if renderer is None:
        renderer = RENDERER_OPENGL2
    if sdl_window_flags is None:
        sdl_window_flags = SDL_WINDOW_RESIZABLE
    if argv is None:
        argv = sys.argv
    argv_encoded = [  # Needs to be kept alive for argv_c.
        ffi.new("char[]", arg.encode("utf-8")) for arg in argv
    ]
    argv_c = ffi.new("char*[]", argv_encoded)

    catch_msg = []  # type: List[str]
    catch_handle = ffi.new_handle(catch_msg)  # Keep alive.

    title_p = _handle_title(title)  # Keep alive.

    params = ffi.new(
        "struct TCOD_ContextParams*",
        {
            "tcod_version": lib.TCOD_COMPILEDVERSION,
            "window_x": x if x is not None else lib.SDL_WINDOWPOS_UNDEFINED,
            "window_y": y if y is not None else lib.SDL_WINDOWPOS_UNDEFINED,
            "pixel_width": width or 0,
            "pixel_height": height or 0,
            "columns": columns or 0,
            "rows": rows or 0,
            "renderer_type": renderer,
            "tileset": _handle_tileset(tileset),
            "vsync": vsync,
            "sdl_window_flags": sdl_window_flags,
            "window_title": title_p,
            "argc": len(argv_c),
            "argv": argv_c,
            "cli_output": ffi.addressof(lib, "_pycall_cli_output"),
            "cli_userdata": catch_handle,
            "window_xy_defined": True,
        },
    )
    context_pp = ffi.new("TCOD_Context**")
    error = lib.TCOD_context_new(params, context_pp)
    if error == lib.TCOD_E_REQUIRES_ATTENTION:
        raise SystemExit(catch_msg[0])
    _check_warn(error)
    return Context._claim(context_pp[0])


@pending_deprecate(
    "Call tcod.context.new with width and height as keyword parameters."
)
def new_window(
    width: int,
    height: int,
    *,
    renderer: Optional[int] = None,
    tileset: Optional[tcod.tileset.Tileset] = None,
    vsync: bool = True,
    sdl_window_flags: Optional[int] = None,
    title: Optional[str] = None
) -> Context:
    """Create a new context with the desired pixel size.

    .. deprecated:: 11.16
        :any:`tcod.context.new` provides more options, such as window position.
    """
    return new(
        width=width,
        height=height,
        renderer=renderer,
        tileset=tileset,
        vsync=vsync,
        sdl_window_flags=sdl_window_flags,
        title=title,
    )


@pending_deprecate(
    "Call tcod.context.new with columns and rows as keyword parameters."
)
def new_terminal(
    columns: int,
    rows: int,
    *,
    renderer: Optional[int] = None,
    tileset: Optional[tcod.tileset.Tileset] = None,
    vsync: bool = True,
    sdl_window_flags: Optional[int] = None,
    title: Optional[str] = None
) -> Context:
    """Create a new context with the desired console size.

    .. deprecated:: 11.16
        :any:`tcod.context.new` provides more options.
    """
    return new(
        columns=columns,
        rows=rows,
        renderer=renderer,
        tileset=tileset,
        vsync=vsync,
        sdl_window_flags=sdl_window_flags,
        title=title,
    )
