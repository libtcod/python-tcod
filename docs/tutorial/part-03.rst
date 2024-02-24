.. _part-3:

Part 3 - UI State
##############################################################################

.. include:: notice.rst

.. warning::

    **This part is still a draft and is being worked on.
    Sections here will be incorrect as these examples were hastily moved from an earlier part.**


State protocol
==============================================================================

To have more states than ``ExampleState`` one must use an abstract type which can be used to refer to any state.
In this case a `Protocol`_ will be used, called ``State``.

Create a new module: ``game/state.py``.
In this module add the class :python:`class State(Protocol):`.
``Protocol`` is from Python's ``typing`` module.
``State`` should have the ``on_event`` and ``on_draw`` methods from ``ExampleState`` but these methods will be empty other than the docstrings describing what they are for.
These methods refer to types from ``tcod`` and those types will need to be imported.
``State`` should also have :python:`__slots__ = ()` [#slots]_ in case the class is used for a subclass.

``game/state.py`` should look like this:

.. code-block:: python

    """Base classes for states."""
    from __future__ import annotations

    from typing import Protocol

    import tcod.console
    import tcod.event


    class State(Protocol):
        """An abstract game state."""

        __slots__ = ()

        def on_event(self, event: tcod.event.Event) -> None:
            """Called on events."""

        def on_draw(self, console: tcod.console.Console) -> None:
            """Called when the state is being drawn."""

The ``InGame`` class does not need to be updated since it is already a structural subtype of ``State``.
Note that subclasses of ``State`` will never be in same module as ``State``, this will be the same for all abstract classes.

State globals
==============================================================================

A new global will be added: :python:`states: list[game.state.State] = []`.
States are implemented as a list/stack to support `pushdown automata <https://gameprogrammingpatterns.com/state.html#pushdown-automata>`_.
Representing states as a stack makes it easier to implement popup windows, menus, and other "history aware" states.

State functions
==============================================================================

Create a new module: ``game/state_tools.py``.
This module will handle events and rendering of the global state.

In this module add the function :python:`def main_draw() -> None:`.
This will hold the "clear, draw, present" logic from the ``main`` function which will be moved to this function.
Render the active state with :python:`g.states[-1].on_draw(g.console)`.
If ``g.states`` is empty then this function should immediately :python:`return` instead of doing anything.
Empty containers in Python are :python:`False` when checked for truthiness.

Next the function :python:`def main_loop() -> None:` is created.
The :python:`while` loop from ``main`` will be moved to this function.
The while loop will be replaced by :python:`while g.states:` so that this function will exit if no state exists.
Drawing will be replaced by a call to ``main_draw``.
Events in the for-loop will be passed to the active state :python:`g.states[-1].on_event(event)`.
Any states ``on_event`` method could potentially change the state so ``g.states`` must be checked to be non-empty for every handled event.

.. code-block:: python

    """State handling functions."""
    from __future__ import annotations

    import tcod.console

    import g


    def main_draw() -> None:
        """Render and present the active state."""
        if not g.states:
            return
        g.console.clear()
        g.states[-1].on_draw(g.console)
        g.context.present(g.console)


    def main_loop() -> None:
        """Run the active state forever."""
        while g.states:
            main_draw()
            for event in tcod.event.wait():
                if g.states:
                    g.states[-1].on_event(event)

Now ``main.py`` can be edited to use the global variables and the new game loop.

Add :python:`import g` and :python:`import game.state_tools`.
Replace references to ``console`` with ``g.console``.
Replace references to ``context`` with ``g.context``.

States are initialed by assigning a list with the initial state to ``g.states``.
The previous game loop is replaced by a call to :python:`game.state_tools.main_loop()`.

.. code-block:: python
    :emphasize-lines: 3-4,12-15

    ...

    import g
    import game.state_tools

    def main() -> None:
        """Entry point function."""
        tileset = tcod.tileset.load_tilesheet(
            "data/Alloy_curses_12x12.png", columns=16, rows=16, charmap=tcod.tileset.CHARMAP_CP437
        )
        tcod.tileset.procedural_block_elements(tileset=tileset)
        g.console = tcod.console.Console(80, 50)
        g.states = [ExampleState(player_x=console.width // 2, player_y=console.height // 2)]
        with tcod.context.new(console=g.console, tileset=tileset) as g.context:
            game.state_tools.main_loop()
    ...

After this you can test the game.
There should be no visible differences from before.


.. rubric:: Footnotes

.. [#slots] This is done to prevent subclasses from requiring a ``__dict__`` attribute.
                If you are still wondering what ``__slots__`` is then `the Python docs have a detailed explanation <https://docs.python.org/3/reference/datamodel.html#slots>`_.

.. _Protocol: https://mypy.readthedocs.io/en/stable/protocols.html
