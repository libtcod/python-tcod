Part 1 - Moving a player around the screen
##############################################################################

Initial script
==============================================================================

First start with a modern top-level script.

.. code-block:: python

    def main() -> None:
        ...


    if __name__ == "__main__":
        main()

You will replace body of the ``main`` function in the following section.

Loading a tileset and opening a window
==============================================================================

From here it is time to setup a ``tcod`` program.
Download `Alloy_curses_12x12.png <https://raw.githubusercontent.com/HexDecimal/python-tcod-tutorial-2023/6b69bf9b5531963a0e5f09f9d8fe72a4001d4881/data/Alloy_curses_12x12.png>`_ and place this file in your projects ``data/`` directory.
This tileset is from the `Dwarf Fortress tileset repository <https://dwarffortresswiki.org/index.php/DF2014:Tileset_repository>`_.
These kinds of tilesets are always loaded with ``columns=16, rows=16, charmap=tcod.tileset.CHARMAP_CP437``.

Load the tileset with :any:`tcod.tileset.load_tilesheet` and then pass it to :any:`tcod.context.new`.
These functions are part of modules which have not been imported yet, so new imports need to be added.
:any:`tcod.context.new` returns a :any:`Context` which is used with the ``with`` statement.

.. code-block:: python
    :emphasize-lines: 2,3,8-12

    ...
    import tcod.context  # Add these imports
    import tcod.tileset


    def main() -> None:
        """Load a tileset and open a window using it, this window will immediately close."""
        tileset = tcod.tileset.load_tilesheet(
            "data/Alloy_curses_12x12.png", columns=16, rows=16, charmap=tcod.tileset.CHARMAP_CP437
        )
        with tcod.context.new(tileset=tileset) as context:
            pass  # The window will stay open for the duration of this block
    ...

If an import fails that means you do not have ``tcod`` installed on the Python environment you just used to run the script.
If you use an IDE then make sure the Python environment it is using is correct and then run ``pip install tcod`` from the shell terminal within that IDE.

If you run this script now then a window will open and then immediately close.
If that happens without seeing a traceback in your terminal then the script is correct.

Configuring an event loop
==============================================================================

The next step is to keep the window open until the user closes it.

Since nothing is displayed yet a :any:`Console` should be created with ``"Hello World"`` printed to it.
The size of the console can be used as a reference to create the context by adding the console to :any:`tcod.context.new`.

Begin the main game loop with a ``while True:`` statement.

To actually display the console to the window the :any:`Context.present` method must be called with the console as a parameter.
Do this first in the game loop before handing events.

Events are checked by iterating over all pending events with :any:`tcod.event.wait`.
Use the code ``for event in tcod.event.wait():`` to begin handing events.
Test if an event is for closing the window with ``if isinstance(event, tcod.event.Quit):``.
If this is True then you should exit the function with ``raise SystemExit``.

.. code-block:: python
    :emphasize-lines: 2,3,11-18

    ...
    import tcod.console
    import tcod.event


    def main() -> None:
        """Show "Hello World" until the window is closed."""
        tileset = tcod.tileset.load_tilesheet(
            "data/Alloy_curses_12x12.png", columns=16, rows=16, charmap=tcod.tileset.CHARMAP_CP437
        )
        console = tcod.console.Console(80, 50)
        console.print(0, 0, "Hello World")  # Test text by printing "Hello World" to the console
        with tcod.context.new(console=console, tileset=tileset) as context:
            while True:  # Main loop
                context.present(console)  # Render the console to the window and show it
                for event in tcod.event.wait():  # Event loop, blocks until pending events exist
                    if isinstance(event, tcod.event.Quit):
                        raise SystemExit()
    ...

If you run this then you get a window saying ``"Hello World"``.
The window can be resized and the console will be stretched to fit the new resolution.

An example game state
==============================================================================

What exists now is not very interactive.
The next step is to change state based on user input.

Like ``tcod`` you'll need to install ``attrs`` with Pip, such as with ``pip install attrs``.

Start by adding an ``attrs`` class called ``ExampleState``.
This a normal class with the ``@attrs.define(eq=False)`` decorator added.

This class should hold coordinates for the player.
It should also have a ``on_draw`` method which takes :any:`tcod.console.Console` as a parameter and marks the player position on it.
The parameters for ``on_draw`` are ``self`` because this is an instance method and ``console: tcod.console.Console``.
``on_draw`` returns nothing, so be sure to add ``-> None``.

:any:`Console.print` is the simplest way to draw the player because other options would require bounds-checking.
Call this method using the players current coordinates and the ``"@"`` character.

.. code-block:: python

    ...
    import attrs


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

Now remove the ``console.print(0, 0, "Hello World")`` line from ``main``.

Before the context is made create a new ``ExampleState`` with player coordinates on the screen.
Each :any:`Console` has ``.width`` and ``.height`` attributes which you can divide by 2 to get a centered coordinate for the player.
Use Python's floor division operator ``//`` so that the resulting type is ``int``.

Modify the drawing routine so that the console is cleared, then passed to ``ExampleState.on_draw``, then passed to :any:`Context.present`.

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
                    if isinstance(event, tcod.event.Quit):
                        raise SystemExit()
    ...

Now if you run the script you'll see ``@``.

The next step is to move the player on events.
A new method will be added to the ``ExampleState`` for this called ``on_event``.
``on_event`` takes a ``self`` and a :any:`tcod.event.Event` parameter and returns nothing.

Events are best handled using Python's `Structural Pattern Matching <https://peps.python.org/pep-0622/>`_.
Consider reading `Python's Structural Pattern Matching Tutorial <https://peps.python.org/pep-0636/>`_.

Begin matching with ``match event:``.
The equivalent to ``if isinstance(event, tcod.event.Quit):`` is ``case tcod.event.Quit():``.
Keyboard keys can be checked with ``case tcod.event.KeyDown(sym=tcod.event.KeySym.LEFT):``.
Make a case for each arrow key: ``LEFT`` ``RIGHT`` ``UP`` ``DOWN`` and move the player in the direction of that key.
See :any:`KeySym` for a list of all keys.

.. code-block:: python

    ...
    @attrs.define(eq=False)
    class ExampleState:
        ...

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
    ...

Now replace the event handling code in ``main`` to defer to the states ``on_event`` method.

.. code-block:: python
    :emphasize-lines: 11

    ...
    def main() -> None:
        ...
        state = ExampleState(player_x=console.width // 2, player_y=console.height // 2)
        with tcod.context.new(console=console, tileset=tileset) as context:
            while True:
                console.clear()
                state.on_draw(console)
                context.present(console)
                for event in tcod.event.wait():
                    state.on_event(event)  # Pass events to the state
    ...

Now when you run this script you have a player character you can move around with the arrow keys before closing the window.

You can review the part-1 source code `here <https://github.com/HexDecimal/python-tcod-tutorial-2023/tree/part-1>`_.
