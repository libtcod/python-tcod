.. _part-1:

Part 1 - Moving a player around the screen
##############################################################################

.. include:: notice.rst

In part 1 you will become familiar with the initialization, rendering, and event system of tcod.
This will be done as a series of small implementations.
It is recommend to save your progress after each section is finished and tested.

Initial script
==============================================================================

First start with a modern top-level script.
You should have ``main.py`` script from :ref:`part-0`:

.. code-block:: python

    from __future__ import annotations


    def main() -> None:
        ...


    if __name__ == "__main__":
        main()

You will replace body of the ``main`` function in the following section.

Loading a tileset and opening a window
==============================================================================

From here it is time to setup a ``tcod`` program.
Download `Alloy_curses_12x12.png <https://raw.githubusercontent.com/HexDecimal/python-tcod-tutorial-2023/6b69bf9b5531963a0e5f09f9d8fe72a4001d4881/data/Alloy_curses_12x12.png>`_ [#tileset]_ and place this file in your projects ``data/`` directory.
This tileset is from the `Dwarf Fortress tileset repository <https://dwarffortresswiki.org/index.php/DF2014:Tileset_repository>`_.
These kinds of tilesets are always loaded with :python:`columns=16, rows=16, charmap=tcod.tileset.CHARMAP_CP437`.
Use the string :python:`"data/Alloy_curses_12x12.png"` to refer to the path of the tileset. [#why_not_pathlib]_

Load the tileset with :any:`tcod.tileset.load_tilesheet`.
Pass the tileset to :any:`tcod.tileset.procedural_block_elements` which will fill in most `Block Elements <https://en.wikipedia.org/wiki/Block_Elements>`_ missing from `Code Page 437 <https://en.wikipedia.org/wiki/Code_page_437>`_.
Then pass the tileset to :any:`tcod.context.new`, you only need to provide the ``tileset`` parameter.

:any:`tcod.context.new` returns a :any:`Context` which will be used with Python's :python:`with` statement.
We want to keep the name of the context, so use the syntax: :python:`with tcod.context.new(tileset=tileset) as context:`.
The new block can't be empty, so add :python:`pass` to the with statement body.

These functions are part of modules which have not been imported yet, so new imports for ``tcod.context`` and ``tcod.tileset`` must be added to the top of the script.

.. code-block:: python
    :emphasize-lines: 3,4,8-14

    from __future__ import annotations

    import tcod.context  # Add these imports
    import tcod.tileset


    def main() -> None:
        """Load a tileset and open a window using it, this window will immediately close."""
        tileset = tcod.tileset.load_tilesheet(
            "data/Alloy_curses_12x12.png", columns=16, rows=16, charmap=tcod.tileset.CHARMAP_CP437
        )
        tcod.tileset.procedural_block_elements(tileset=tileset)
        with tcod.context.new(tileset=tileset) as context:
            pass  # The window will stay open for the duration of this block


    if __name__ == "__main__":
        main()

If an import fails that means you do not have ``tcod`` installed on the Python environment you just used to run the script.
If you use an IDE then make sure the Python environment it is using is correct and then run :shell:`pip install tcod` from the shell terminal within that IDE.

There is no game loop, so if you run this script now then a window will open and then immediately close.
If that happens without seeing a traceback in your terminal then the script is correct.

Configuring an event loop
==============================================================================

The next step is to keep the window open until the user closes it.

Since nothing is displayed yet a :any:`Console` should be created with :python:`"Hello World"` printed to it.
The size of the console can be used as a reference to create the context by adding the console to :any:`tcod.context.new`. [#init_context]_

Begin the main game loop with a :python:`while True:` statement.

To actually display the console to the window the :any:`Context.present` method must be called with the console as a parameter.
Do this first in the game loop before handing events.

Events are checked by iterating over all pending events with :any:`tcod.event.wait`.
Use the code :python:`for event in tcod.event.wait():` to begin handing events.

In the event loop start with the line :python:`print(event)` so that all events can be viewed from the program output.
Then test if an event is for closing the window with :python:`if isinstance(event, tcod.event.Quit):`.
If this is True then you should exit the function with :python:`raise SystemExit()`. [#why_raise]_

.. code-block:: python
    :emphasize-lines: 3,5,15-23

    from __future__ import annotations

    import tcod.console
    import tcod.context
    import tcod.event
    import tcod.tileset


    def main() -> None:
        """Show "Hello World" until the window is closed."""
        tileset = tcod.tileset.load_tilesheet(
            "data/Alloy_curses_12x12.png", columns=16, rows=16, charmap=tcod.tileset.CHARMAP_CP437
        )
        tcod.tileset.procedural_block_elements(tileset=tileset)
        console = tcod.console.Console(80, 50)
        console.print(0, 0, "Hello World")  # Test text by printing "Hello World" to the console
        with tcod.context.new(console=console, tileset=tileset) as context:
            while True:  # Main loop
                context.present(console)  # Render the console to the window and show it
                for event in tcod.event.wait():  # Event loop, blocks until pending events exist
                    print(event)
                    if isinstance(event, tcod.event.Quit):
                        raise SystemExit()


    if __name__ == "__main__":
        main()

If you run this then you get a window saying :python:`"Hello World"`.
The window can be resized and the console will be stretched to fit the new resolution.
When you do anything such as press a key or interact with the window the event for that action will be printed to the program output.

An example game state
==============================================================================

What exists now is not very interactive.
The next step is to change state based on user input.

Like ``tcod`` you'll need to install ``attrs`` with Pip, such as with :shell:`pip install attrs`.

Start by adding an ``attrs`` class called ``ExampleState``.
This a normal class with the :python:`@attrs.define(eq=False)` decorator added.

This class should hold coordinates for the player.
It should also have a ``on_draw`` method which takes :any:`tcod.console.Console` as a parameter and marks the player position on it.
The parameters for ``on_draw`` are ``self`` because this is an instance method and :python:`console: tcod.console.Console`.
``on_draw`` returns nothing, so be sure to add :python:`-> None`.

:any:`Console.print` is the simplest way to draw the player because other options would require bounds-checking.
Call this method using the players current coordinates and the :python:`"@"` character.

.. code-block:: python
    :emphasize-lines: 3,10-21

    from __future__ import annotations

    import attrs
    import tcod.console
    import tcod.context
    import tcod.event
    import tcod.tileset


    @attrs.define(eq=False)
    class ExampleState:
        """Example state with a hard-coded player position."""

        player_x: int
        """Player X position, left-most position is zero."""
        player_y: int
        """Player Y position, top-most position is zero."""

        def on_draw(self, console: tcod.console.Console) -> None:
            """Draw the player glyph."""
            console.print(self.player_x, self.player_y, "@")

    ...

Now remove the :python:`console.print(0, 0, "Hello World")` line from ``main``.

Before the context is made create a new ``ExampleState`` with player coordinates on the screen.
Each :any:`Console` has :python:`.width` and :python:`.height` attributes which you can divide by 2 to get a centered coordinate for the player.
Use Python's floor division operator :python:`//` so that the resulting type is :python:`int`.

Modify the drawing routine so that the console is cleared, then passed to :python:`ExampleState.on_draw`, then passed to :any:`Context.present`.

.. code-block:: python
    :emphasize-lines: 9,12-14

    ...
    def main() -> None:
        """Run ExampleState."""
        tileset = tcod.tileset.load_tilesheet(
            "data/Alloy_curses_12x12.png", columns=16, rows=16, charmap=tcod.tileset.CHARMAP_CP437
        )
        tcod.tileset.procedural_block_elements(tileset=tileset)
        console = tcod.console.Console(80, 50)
        state = ExampleState(player_x=console.width // 2, player_y=console.height // 2)
        with tcod.context.new(console=console, tileset=tileset) as context:
            while True:
                console.clear()  # Clear the console before any drawing
                state.on_draw(console)  # Draw the current state
                context.present(console)  # Display the console on the window
                for event in tcod.event.wait():
                    print(event)
                    if isinstance(event, tcod.event.Quit):
                        raise SystemExit()


    if __name__ == "__main__":
        main()

Now if you run the script you'll see ``@``.

The next step is to move the player on events.
A new method will be added to the ``ExampleState`` for this called ``on_event``.
``on_event`` takes a ``self`` and a :any:`tcod.event.Event` parameter and returns nothing.

Events are best handled using Python's `Structural Pattern Matching <https://peps.python.org/pep-0622/>`_.
Consider reading `Python's Structural Pattern Matching Tutorial <https://peps.python.org/pep-0636/>`_.

Begin matching with :python:`match event:`.
The equivalent to :python:`if isinstance(event, tcod.event.Quit):` is :python:`case tcod.event.Quit():`.
Keyboard keys can be checked with :python:`case tcod.event.KeyDown(sym=tcod.event.KeySym.LEFT):`.
Make a case for each arrow key: ``LEFT`` ``RIGHT`` ``UP`` ``DOWN`` and move the player in the direction of that key.
Since events are printed you can check the :any:`KeySym` of a key by pressing that key and looking at the printed output.
See :any:`KeySym` for a list of all keys.

Finally replace the event handling code in ``main`` to defer to the states ``on_event`` method.
The full script so far is:

.. code-block:: python
    :emphasize-lines: 23-35,53

    from __future__ import annotations

    import attrs
    import tcod.console
    import tcod.context
    import tcod.event
    import tcod.tileset


    @attrs.define(eq=False)
    class ExampleState:
        """Example state with a hard-coded player position."""

        player_x: int
        """Player X position, left-most position is zero."""
        player_y: int
        """Player Y position, top-most position is zero."""

        def on_draw(self, console: tcod.console.Console) -> None:
            """Draw the player glyph."""
            console.print(self.player_x, self.player_y, "@")

        def on_event(self, event: tcod.event.Event) -> None:
            """Move the player on events and handle exiting. Movement is hard-coded."""
            match event:
                case tcod.event.Quit():
                    raise SystemExit()
                case tcod.event.KeyDown(sym=tcod.event.KeySym.LEFT):
                    self.player_x -= 1
                case tcod.event.KeyDown(sym=tcod.event.KeySym.RIGHT):
                    self.player_x += 1
                case tcod.event.KeyDown(sym=tcod.event.KeySym.UP):
                    self.player_y -= 1
                case tcod.event.KeyDown(sym=tcod.event.KeySym.DOWN):
                    self.player_y += 1


    def main() -> None:
        """Run ExampleState."""
        tileset = tcod.tileset.load_tilesheet(
            "data/Alloy_curses_12x12.png", columns=16, rows=16, charmap=tcod.tileset.CHARMAP_CP437
        )
        tcod.tileset.procedural_block_elements(tileset=tileset)
        console = tcod.console.Console(80, 50)
        state = ExampleState(player_x=console.width // 2, player_y=console.height // 2)
        with tcod.context.new(console=console, tileset=tileset) as context:
            while True:
                console.clear()
                state.on_draw(console)
                context.present(console)
                for event in tcod.event.wait():
                    print(event)
                    state.on_event(event)  # Pass events to the state


    if __name__ == "__main__":
        main()

Now when you run this script you have a player character you can move around with the arrow keys before closing the window.

You can review the part-1 source code `here <https://github.com/HexDecimal/python-tcod-tutorial-2023/tree/part-1>`_.

.. rubric:: Footnotes

.. [#tileset] The choice of tileset came down to what looked nice while also being square.
              Other options such as using a BDF font were considered, but in the end this tutorial won't go too much into Unicode.

.. [#why_not_pathlib]
    :any:`pathlib` is not used because this example is too simple for that.
    The working directory will always be the project root folder for the entire tutorial, including distributions.
    :any:`pathlib` will be used later for saved games and configuration directories, and not for static data.

.. [#init_context] This tutorial follows the setup for a fixed-size console.
                   The alternatives shown in :ref:`getting-started` are outside the scope of this tutorial.

.. [#why_raise] You could use :python:`return` here to exit the ``main`` function and end the program, but :python:`raise SystemExit()` is used because it will close the program from anywhere.
                :python:`raise SystemExit()` is also more useful to teach than :any:`sys.exit`.
