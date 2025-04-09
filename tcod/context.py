"""This module is used to create and handle libtcod contexts.

See :ref:`getting-started` for beginner examples on how to use this module.

:any:`Context`'s are intended to replace several libtcod functions such as
:any:`libtcodpy.console_init_root`, :any:`libtcodpy.console_flush`,
:any:`tcod.console.recommended_size`, and many other functions which rely on
hidden global objects within libtcod.  If you begin using contexts then
most of these functions will no longer work properly.

Instead of calling :any:`libtcodpy.console_init_root` you can call
:any:`tcod.context.new` with different keywords depending on how you plan
to setup the size of the console.  You should use
:any:`tcod.tileset` to load the font for a context.

.. note::
    If you use contexts then expect deprecated functions from ``libtcodpy`` to no longer work correctly.
    Those functions rely on a global console or tileset which doesn't exists with contexts.
    Also ``libtcodpy`` event functions will no longer return tile coordinates for the mouse.

    New programs not using ``libtcodpy`` can ignore this warning.

.. versionadded:: 11.12
"""

from __future__ import annotations

import copy
import pickle
import sys
import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal, NoReturn, TypeVar

from typing_extensions import Self, deprecated

import tcod.console
import tcod.event
import tcod.render
import tcod.sdl.render
import tcod.sdl.video
import tcod.tileset
from tcod._internal import _check, _check_warn
from tcod.cffi import ffi, lib

__all__ = (
    "RENDERER_OPENGL",
    "RENDERER_OPENGL2",
    "RENDERER_SDL",
    "RENDERER_SDL2",
    "RENDERER_XTERM",
    "SDL_WINDOW_ALLOW_HIGHDPI",
    "SDL_WINDOW_BORDERLESS",
    "SDL_WINDOW_FULLSCREEN",
    "SDL_WINDOW_FULLSCREEN_DESKTOP",
    "SDL_WINDOW_HIDDEN",
    "SDL_WINDOW_INPUT_GRABBED",
    "SDL_WINDOW_MAXIMIZED",
    "SDL_WINDOW_MINIMIZED",
    "SDL_WINDOW_RESIZABLE",
    "Context",
    "new",
    "new_terminal",
    "new_window",
)

_Event = TypeVar("_Event", bound=tcod.event.Event)

SDL_WINDOW_FULLSCREEN = lib.SDL_WINDOW_FULLSCREEN
"""Fullscreen mode."""
# SDL_WINDOW_FULLSCREEN_DESKTOP = lib.SDL_WINDOW_FULLSCREEN_DESKTOP
# """A borderless fullscreen window at the desktop resolution."""
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
SDL_WINDOW_INPUT_GRABBED = lib.SDL_WINDOW_MOUSE_GRABBED
"""Window has grabbed the input."""
SDL_WINDOW_ALLOW_HIGHDPI = lib.SDL_WINDOW_HIGH_PIXEL_DENSITY
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
RENDERER_XTERM = lib.TCOD_RENDERER_XTERM
"""A renderer targeting modern terminals with 24-bit color support.

This is an experimental renderer with partial support for XTerm and SSH.
This will work best on those terminals.

Terminal inputs and events will be passed to SDL's event system.

There is poor support for ANSI escapes on Windows 10.
It is not recommended to use this renderer on Windows.

.. versionadded:: 13.3
"""


def _handle_tileset(tileset: tcod.tileset.Tileset | None) -> Any:  # noqa: ANN401
    """Get the TCOD_Tileset pointer from a Tileset or return a NULL pointer."""
    return tileset._tileset_p if tileset else ffi.NULL


def _handle_title(title: str | None) -> Any:  # noqa: ANN401
    """Return title as a CFFI string.

    If title is None then return a decent default title is returned.
    """
    if title is None:
        title = Path(sys.argv[0]).name
    return ffi.new("char[]", title.encode("utf-8"))


class Context:
    """Context manager for libtcod context objects.

    Use :any:`tcod.context.new` to create a new context.
    """

    def __init__(self, context_p: Any) -> None:  # noqa: ANN401
        """Create a context from a cffi pointer."""
        self._context_p = context_p

    @classmethod
    def _claim(cls, context_p: Any) -> Context:  # noqa: ANN401
        """Return a new instance wrapping a context pointer."""
        return cls(ffi.gc(context_p, lib.TCOD_context_delete))

    @property
    def _p(self) -> Any:  # noqa: ANN401
        """Return the context pointer or raise if it is missing."""
        try:
            return self._context_p
        except AttributeError:
            msg = "This context has been closed can no longer be used."
            raise RuntimeError(msg) from None

    def __enter__(self) -> Self:
        """Enter this context which will close on exiting."""
        return self

    def close(self) -> None:
        """Close this context, closing any windows opened by this context.

        Afterwards doing anything with this instance other than closing it again is invalid.
        """
        if hasattr(self, "_context_p"):
            ffi.release(self._context_p)
            del self._context_p

    def __exit__(self, *_: object) -> None:
        """Automatically close on the context on exit."""
        self.close()

    def present(
        self,
        console: tcod.console.Console,
        *,
        keep_aspect: bool = False,
        integer_scaling: bool = False,
        clear_color: tuple[int, int, int] = (0, 0, 0),
        align: tuple[float, float] = (0.5, 0.5),
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
        _check(lib.TCOD_context_present(self._p, console.console_c, viewport_args))

    def pixel_to_tile(self, x: float, y: float) -> tuple[float, float]:
        """Convert window pixel coordinates to tile coordinates."""
        with ffi.new("double[2]", (x, y)) as xy:
            _check(lib.TCOD_context_screen_pixel_to_tile_d(self._p, xy, xy + 1))
            return xy[0], xy[1]

    @deprecated("Use pixel_to_tile method instead.")
    def pixel_to_subtile(self, x: float, y: float) -> tuple[float, float]:
        """Convert window pixel coordinates to sub-tile coordinates."""
        with ffi.new("double[2]", (x, y)) as xy:
            _check(lib.TCOD_context_screen_pixel_to_tile_d(self._p, xy, xy + 1))
            return xy[0], xy[1]

    def convert_event(self, event: _Event) -> _Event:
        """Return an event with mouse pixel coordinates converted into tile coordinates.

        Example::

            context: tcod.context.Context
            for event in tcod.event.get():
                event_tile = context.convert_event(event)
                if isinstance(event, tcod.event.MouseMotion):
                    # Events start with pixel coordinates and motion.
                    print(f"Pixels: {event.position=}, {event.motion=}")
                if isinstance(event_tile, tcod.event.MouseMotion):
                    # Tile coordinates are used in the returned event.
                    print(f"Tiles: {event_tile.position=}, {event_tile.motion=}")

        .. versionchanged:: 15.0
            Now returns a new event with the coordinates converted into tiles.
        """
        event_copy = copy.copy(event)
        if isinstance(event, (tcod.event.MouseState, tcod.event.MouseMotion)):
            assert isinstance(event_copy, (tcod.event.MouseState, tcod.event.MouseMotion))
            event_copy.position = event._tile = tcod.event.Point(*self.pixel_to_tile(*event.position))
        if isinstance(event, tcod.event.MouseMotion):
            assert isinstance(event_copy, tcod.event.MouseMotion)
            assert event._tile is not None
            prev_tile = self.pixel_to_tile(
                event.position[0] - event.motion[0],
                event.position[1] - event.motion[1],
            )
            event_copy.motion = event._tile_motion = tcod.event.Point(
                int(event._tile[0]) - int(prev_tile[0]), int(event._tile[1]) - int(prev_tile[1])
            )
        return event_copy

    def save_screenshot(self, path: str | None = None) -> None:
        """Save a screen-shot to the given file path."""
        c_path = path.encode("utf-8") if path is not None else ffi.NULL
        _check(lib.TCOD_context_save_screenshot(self._p, c_path))

    def change_tileset(self, tileset: tcod.tileset.Tileset | None) -> None:
        """Change the active tileset used by this context.

        The new tileset will take effect on the next call to :any:`present`.
        Contexts not using a renderer with an emulated terminal will be unaffected by this method.

        This does not do anything to resize the window, keep this in mind if the tileset as a differing tile size.
        Access the window with :any:`sdl_window` to resize it manually, if needed.

        Using this method only one tileset is active per-frame.
        See :any:`tcod.render` if you want to renderer with multiple tilesets in a single frame.
        """
        _check(lib.TCOD_context_change_tileset(self._p, _handle_tileset(tileset)))

    def new_console(
        self,
        min_columns: int = 1,
        min_rows: int = 1,
        magnification: float = 1.0,
        order: Literal["C", "F"] = "C",
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

        Example::

            scale = 1  # Tile size scale.  This example uses integers but floating point numbers are also valid.
            context = tcod.context.new()
            while True:
                # Create a cleared, dynamically-sized console for each frame.
                console = context.new_console(magnification=scale)
                # This printed output will wrap if the window is shrunk.
                console.print_box(0, 0, console.width, console.height, "Hello world")
                # Use integer_scaling to prevent subpixel distortion.
                # This may add padding around the rendered console.
                context.present(console, integer_scaling=True)
                for event in tcod.event.wait():
                    if isinstance(event, tcod.event.Quit):
                        raise SystemExit()
                    elif isinstance(event, tcod.event.MouseWheel):
                        # Use the mouse wheel to change the rendered tile size.
                        scale = max(1, scale + event.y)
        """
        if magnification < 0:
            msg = f"Magnification must be greater than zero. (Got {magnification:f})"
            raise ValueError(msg)
        size = ffi.new("int[2]")
        _check(lib.TCOD_context_recommended_console_size(self._p, magnification, size, size + 1))
        width, height = max(min_columns, size[0]), max(min_rows, size[1])
        return tcod.console.Console(width, height, order=order)

    def recommended_console_size(self, min_columns: int = 1, min_rows: int = 1) -> tuple[int, int]:
        """Return the recommended (columns, rows) of a console for this context.

        `min_columns`, `min_rows` are the lowest values which will be returned.

        If result is only used to create a new console then you may want to call :any:`Context.new_console` instead.
        """
        with ffi.new("int[2]") as size:
            _check(lib.TCOD_context_recommended_console_size(self._p, 1.0, size, size + 1))
            return max(min_columns, size[0]), max(min_rows, size[1])

    @property
    def renderer_type(self) -> int:
        """Return the libtcod renderer type used by this context."""
        return _check(lib.TCOD_context_get_renderer_type(self._p))

    @property
    def sdl_window_p(self) -> Any:  # noqa: ANN401
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

        '''
        return lib.TCOD_context_get_sdl_window(self._p)

    @property
    def sdl_window(self) -> tcod.sdl.video.Window | None:
        '''Return a :any:`tcod.sdl.video.Window` referencing this contexts SDL window if it exists.

        Example::

            import tcod
            import tcod.sdl.video

            def toggle_fullscreen(context: tcod.context.Context) -> None:
                """Toggle a context window between fullscreen and windowed modes."""
                window = context.sdl_window
                if not window:
                    return
                if window.fullscreen:
                    window.fullscreen = False
                else:
                    window.fullscreen = tcod.sdl.video.WindowFlags.FULLSCREEN_DESKTOP

        .. versionadded:: 13.4
        '''
        p = self.sdl_window_p
        return tcod.sdl.video.Window(p) if p else None

    @property
    def sdl_renderer(self) -> tcod.sdl.render.Renderer | None:
        """Return a :any:`tcod.sdl.render.Renderer` referencing this contexts SDL renderer if it exists.

        .. versionadded:: 13.4
        """
        p = lib.TCOD_context_get_sdl_renderer(self._p)
        return tcod.sdl.render.Renderer(p) if p else None

    @property
    def sdl_atlas(self) -> tcod.render.SDLTilesetAtlas | None:
        """Return a :any:`tcod.render.SDLTilesetAtlas` referencing libtcod's SDL texture atlas if it exists.

        .. versionadded:: 13.5
        """
        if self._p.type not in (lib.TCOD_RENDERER_SDL, lib.TCOD_RENDERER_SDL2):
            return None
        context_data = ffi.cast("struct TCOD_RendererSDL2*", self._context_p.contextdata_)
        return tcod.render.SDLTilesetAtlas._from_ref(context_data.renderer, context_data.atlas)

    def __reduce__(self) -> NoReturn:
        """Contexts can not be pickled, so this class will raise :class:`pickle.PicklingError`."""
        msg = "Python-tcod contexts can not be pickled."
        raise pickle.PicklingError(msg)


@ffi.def_extern()  # type: ignore[misc]
def _pycall_cli_output(catch_reference: Any, output: Any) -> None:  # noqa: ANN401
    """Callback for the libtcod context CLI.

    Catches the CLI output.
    """
    catch: list[str] = ffi.from_handle(catch_reference)
    catch.append(ffi.string(output).decode("utf-8"))


def new(  # noqa: PLR0913
    *,
    x: int | None = None,
    y: int | None = None,
    width: int | None = None,
    height: int | None = None,
    columns: int | None = None,
    rows: int | None = None,
    renderer: int | None = None,
    tileset: tcod.tileset.Tileset | None = None,
    vsync: bool = True,
    sdl_window_flags: int | None = None,
    title: str | None = None,
    argv: Iterable[str] | None = None,
    console: tcod.console.Console | None = None,
) -> Context:
    """Create a new context with the desired pixel size.

    `x`, `y`, `width`, and `height` are the desired position and size of the
    window.  If these are None then they will be derived from `columns` and
    `rows`.  So if you plan on having a console of a fixed size then you should
    set `columns` and `rows` instead of the window keywords.

    `columns` and `rows` is the desired size of the console.  Can be left as
    `None` when you're setting a context by a window size instead of a console.

    `console` automatically fills in the `columns` and `rows` parameters from an existing :any:`tcod.console.Console`
    instance.

    Providing no size information at all is also acceptable.

    `renderer` now does nothing and should not be set.  It may be removed in the future.

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

    .. versionchanged:: 13.2
        Added the `console` parameter.
    """
    if renderer is not None:
        warnings.warn(
            "The renderer parameter was deprecated and will likely be removed in a future version of libtcod.  "
            "Remove the renderer parameter to fix this warning.",
            FutureWarning,
            stacklevel=2,
        )
    renderer = RENDERER_SDL2
    if sdl_window_flags is None:
        sdl_window_flags = SDL_WINDOW_RESIZABLE
    if argv is None:
        argv = sys.argv
    if console is not None:
        columns = columns or console.width
        rows = rows or console.height
    argv_encoded = [ffi.new("char[]", arg.encode("utf-8")) for arg in argv]  # Needs to be kept alive for argv_c.
    argv_c = ffi.new("char*[]", argv_encoded)

    catch_msg: list[str] = []
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


@deprecated("Call tcod.context.new with width and height as keyword parameters.")
def new_window(  # noqa: PLR0913
    width: int,
    height: int,
    *,
    renderer: int | None = None,
    tileset: tcod.tileset.Tileset | None = None,
    vsync: bool = True,
    sdl_window_flags: int | None = None,
    title: str | None = None,
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


@deprecated("Call tcod.context.new with columns and rows as keyword parameters.")
def new_terminal(  # noqa: PLR0913
    columns: int,
    rows: int,
    *,
    renderer: int | None = None,
    tileset: tcod.tileset.Tileset | None = None,
    vsync: bool = True,
    sdl_window_flags: int | None = None,
    title: str | None = None,
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
