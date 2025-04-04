.. _getting-started:

Getting Started
===============

Python 3 and python-tcod must be installed, see :ref:`installation`.

Fixed-size console
------------------

This example is a hello world script which handles font loading,
fixed-sized consoles, window contexts, and event handling.
This example requires the
`dejavu10x10_gs_tc.png <https://github.com/libtcod/python-tcod/blob/11.13.5/fonts/libtcod/dejavu10x10_gs_tc.png>`_
font to be in the same directory as the script.

By default this will create a window which can be resized and the fixed-size
console will be stretched to fit the window.  You can add arguments to
:any:`Context.present` to fix the aspect ratio or only scale the console by
integer increments.

.. code-block:: python

    #!/usr/bin/env python
    # Make sure 'dejavu10x10_gs_tc.png' is in the same directory as this script.
    import tcod.console
    import tcod.context
    import tcod.event
    import tcod.tileset

    WIDTH, HEIGHT = 80, 60  # Console width and height in tiles.


    def main() -> None:
        """Script entry point."""
        # Load the font, a 32 by 8 tile font with libtcod's old character layout.
        tileset = tcod.tileset.load_tilesheet(
            "dejavu10x10_gs_tc.png", 32, 8, tcod.tileset.CHARMAP_TCOD,
        )
        # Create the main console.
        console = tcod.console.Console(WIDTH, HEIGHT)
        # Create a window based on this console and tileset.
        with tcod.context.new(  # New window for a console of size columns×rows.
            columns=console.width, rows=console.height, tileset=tileset,
        ) as context:
            while True:  # Main loop, runs until SystemExit is raised.
                console.clear()
                console.print(x=0, y=0, text="Hello World!")
                context.present(console)  # Show the console.

                # This event loop will wait until at least one event is processed before exiting.
                # For a non-blocking event loop replace `tcod.event.wait` with `tcod.event.get`.
                for event in tcod.event.wait():
                    context.convert_event(event)  # Sets tile coordinates for mouse events.
                    print(event)  # Print event names and attributes.
                    match event:
                        case tcod.event.Quit():
                            raise SystemExit
            # The window will be closed after the above with-block exits.


    if __name__ == "__main__":
        main()

Dynamically-sized console
-------------------------

The next example shows a more advanced setup.  A maximized window is created
and the console is dynamically scaled to fit within it.  If the window is
resized then the console will be resized to match it.

Because a tileset wasn't manually loaded in this example an OS dependent
fallback font will be used.  This is useful for prototyping but it's not
recommended to release with this font since it can fail to load on some
platforms.

The `integer_scaling` parameter to :any:`Context.present` prevents the console
from being slightly stretched, since the console will rarely be the prefect
size a small border will exist.  This border is black by default but can be
changed to another color.

You'll need to consider things like the console being too small for your code
to handle or the tiles being small compared to an extra large monitor
resolution.  :any:`Context.new_console` can be given a minimum size that it
will never go below.

You can call :any:`Context.new_console` every frame or only when the window
is resized.  This example creates a new console every frame instead of
clearing the console every frame and replacing it only on resizing the window.

.. code-block:: python

    #!/usr/bin/env python
    import tcod.context
    import tcod.event

    WIDTH, HEIGHT = 720, 480  # Window pixel resolution (when not maximized.)
    FLAGS = tcod.context.SDL_WINDOW_RESIZABLE | tcod.context.SDL_WINDOW_MAXIMIZED


    def main() -> None:
        """Script entry point."""
        with tcod.context.new(  # New window with pixel resolution of width×height.
            width=WIDTH, height=HEIGHT, sdl_window_flags=FLAGS
        ) as context:
            while True:
                console = context.new_console()  # Console size based on window resolution and tile size.
                console.print(0, 0, "Hello World")
                context.present(console, integer_scaling=True)

                for event in tcod.event.wait():
                    event = context.convert_event(event)  # Sets tile coordinates for mouse events.
                    print(event)  # Print event names and attributes.
                    match event:
                        case tcod.event.Quit():
                            raise SystemExit
                        case tcod.event.WindowResized(type="WindowSizeChanged"):
                            pass  # The next call to context.new_console may return a different size.


    if __name__ == "__main__":
        main()
