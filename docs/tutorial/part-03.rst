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

Now add a few small classes using :python:`@attrs.define()`:
A ``Push`` class with a :python:`state: State` attribute.
A ``Pop`` class with no attributes.
A ``Reset`` class with a :python:`state: State` attribute.

Then add a :python:`StateResult: TypeAlias = "Push | Pop | Reset | None"`.
This is a type which combines all of the previous classes.

Edit ``State``'s ``on_event`` method to return ``StateResult``.

``game/state.py`` should look like this:

.. code-block:: python

    """Base classes for states."""

    from __future__ import annotations

    from typing import Protocol, TypeAlias

    import attrs
    import tcod.console
    import tcod.event


    class State(Protocol):
        """An abstract game state."""

        __slots__ = ()

        def on_event(self, event: tcod.event.Event) -> StateResult:
            """Called on events."""

        def on_draw(self, console: tcod.console.Console) -> None:
            """Called when the state is being drawn."""


    @attrs.define()
    class Push:
        """Push a new state on top of the stack."""

        state: State


    @attrs.define()
    class Pop:
        """Remove the current state from the stack."""


    @attrs.define()
    class Reset:
        """Replace the entire stack with a new state."""

        state: State


    StateResult: TypeAlias = "Push | Pop | Reset | None"
    """Union of state results."""

The ``InGame`` class does not need to be updated since it is already a structural subtype of ``State``.
Note that subclasses of ``State`` will never be in same module as ``State``, this will be the same for all abstract classes.

New globals
==============================================================================

A new global will be added: :python:`states: list[game.state.State] = []`.
States are implemented as a list/stack to support `pushdown automata <https://gameprogrammingpatterns.com/state.html#pushdown-automata>`_.
Representing states as a stack makes it easier to implement popup windows, sub-menus, and other prompts.

The ``console`` variable from ``main.py`` will be moved to ``g.py``.
Add :python:`console: tcod.console.Console` and replace all references to ``console`` in ``main.py`` with ``g.console``.

.. code-block:: python
    :emphasize-lines: 9,17-21

    """This module stores globally mutable variables used by this program."""

    from __future__ import annotations

    import tcod.console
    import tcod.context
    import tcod.ecs

    import game.state

    context: tcod.context.Context
    """The window managed by tcod."""

    world: tcod.ecs.Registry
    """The active ECS registry and current session."""

    states: list[game.state.State] = []
    """A stack of states with the last item being the active state."""

    console: tcod.console.Console
    """The current main console."""


State functions
==============================================================================

Create a new module: ``game/state_tools.py``.
This module will handle events and rendering of the global state.

In this module add the function :python:`def main_draw() -> None:`.
This will hold the "clear, draw, present" logic from the ``main`` function which will be moved to this function.
Render the active state with :python:`g.states[-1].on_draw(g.console)`.
If ``g.states`` is empty then this function should immediately :python:`return` instead of doing anything.
Empty containers in Python are :python:`False` when checked for truthiness.

Next is to handle the ``StateResult`` type.
Start by adding the :python:`def apply_state_result(result: StateResult) -> None:` function.
This function will :python:`match result:` to decide on what to do.

:python:`case Push(state=state):` should append ``state`` to ``g.states``.

:python:`case Pop():` should simply call :python:`g.states.pop()`.

:python:`case Reset(state=state):` should call :python:`apply_state_result(Pop())` until ``g.state`` is empty then call :python:`apply_state_result(Push(state))`.

:python:`case None:` should be handled by explicitly ignoring it.

:python:`case _:` handles anything else and should invoke :python:`raise TypeError(result)` since no other types are expected.

Now the function :python:`def main_loop() -> None:` is created.
The :python:`while` loop from ``main`` will be moved to this function.
The while loop will be replaced by :python:`while g.states:` so that this function will exit if no state exists.
Drawing will be replaced by a call to ``main_draw``.
Events with mouse coordinates should be converted to tiles using :python:`tile_event = g.context.convert_event(event)` before being passed to a state.
:python:`apply_state_result(g.states[-1].on_event(tile_event))` will pass the event and handle the return result at the same time.
``g.states`` must be checked to be non-empty inside the event handing for-loop because ``apply_state_result`` could cause ``g.states`` to become empty.

Next is the utility function :python:`def get_previous_state(state: State) -> State | None:`.
Get the index of ``state`` in ``g.states`` by identity [#identity]_ using :python:`current_index = next(index for index, value in enumerate(g.states) if value is state)`.
Return the previous state if :python:`current_index > 0` or else return None using :python:`return g.states[current_index - 1] if current_index > 0 else None`.

Next is :python:`def draw_previous_state(state: State, console: tcod.console.Console, dim: bool = True) -> None:`.
Call ``get_previous_state`` to get the previous state and return early if the result is :python:`None`.
Then call the previous states :python:`State.on_draw` method as normal.
Afterwards test :python:`dim and state is g.states[-1]` to see if the console should be dimmed.
If it should be dimmed then reduce the color values of the console with :python:`console.rgb["fg"] //= 4` and :python:`console.rgb["bg"] //= 4`.
This is used to indicate that any graphics behind the active state are non-interactable.


.. code-block:: python

    """State handling functions."""

    from __future__ import annotations

    import tcod.console

    import g
    from game.state import Pop, Push, Reset, StateResult


    def main_draw() -> None:
        """Render and present the active state."""
        if not g.states:
            return
        g.console.clear()
        g.states[-1].on_draw(g.console)
        g.context.present(g.console)


    def apply_state_result(result: StateResult) -> None:
        """Apply a StateResult to `g.states`."""
        match result:
            case Push(state=state):
                g.states.append(state)
            case Pop():
                g.states.pop()
            case Reset(state=state):
                while g.states:
                    apply_state_result(Pop())
                apply_state_result(Push(state))
            case None:
                pass
            case _:
                raise TypeError(result)


    def main_loop() -> None:
        """Run the active state forever."""
        while g.states:
            main_draw()
            for event in tcod.event.wait():
                tile_event = g.context.convert_event(event)
                if g.states:
                    apply_state_result(g.states[-1].on_event(tile_event))


    def get_previous_state(state: State) -> State | None:
        """Return the state before `state` in the stack if it exists."""
        current_index = next(index for index, value in enumerate(g.states) if value is state)
        return g.states[current_index - 1] if current_index > 0 else None


    def draw_previous_state(state: State, console: tcod.console.Console, dim: bool = True) -> None:
        """Draw previous states, optionally dimming all but the active state."""
        prev_state = get_previous_state(state)
        if prev_state is None:
            return
        prev_state.on_draw(console)
        if dim and state is g.states[-1]:
            console.rgb["fg"] //= 4
            console.rgb["bg"] //= 4

Menus
==============================================================================

.. code-block:: python

    """Menu UI classes."""

    from __future__ import annotations

    from collections.abc import Callable
    from typing import Protocol

    import attrs
    import tcod.console
    import tcod.event
    from tcod.event import KeySym

    import game.state_tools
    from game.constants import DIRECTION_KEYS
    from game.state import Pop, State, StateResult


    class MenuItem(Protocol):
        """Menu item protocol."""

        __slots__ = ()

        def on_event(self, event: tcod.event.Event) -> StateResult:
            """Handle events passed to the menu item."""

        def on_draw(self, console: tcod.console.Console, x: int, y: int, highlight: bool) -> None:
            """Draw is item at the given position."""


    @attrs.define()
    class SelectItem(MenuItem):
        """Clickable menu item."""

        label: str
        callback: Callable[[], StateResult]

        def on_event(self, event: tcod.event.Event) -> StateResult:
            """Handle events selecting this item."""
            match event:
                case tcod.event.KeyDown(sym=sym) if sym in {KeySym.RETURN, KeySym.RETURN2, KeySym.KP_ENTER}:
                    return self.callback()
                case tcod.event.MouseButtonUp(button=tcod.event.MouseButton.LEFT):
                    return self.callback()
                case _:
                    return None

        def on_draw(self, console: tcod.console.Console, x: int, y: int, highlight: bool) -> None:
            """Render this items label."""
            console.print(x, y, self.label, fg=(255, 255, 255), bg=(64, 64, 64) if highlight else (0, 0, 0))


    @attrs.define()
    class ListMenu(State):
        """Simple list menu state."""

        items: tuple[MenuItem, ...]
        selected: int | None = 0
        x: int = 0
        y: int = 0

        def on_event(self, event: tcod.event.Event) -> StateResult:
            """Handle events for menus."""
            match event:
                case tcod.event.Quit():
                    raise SystemExit()
                case tcod.event.KeyDown(sym=sym) if sym in DIRECTION_KEYS:
                    dx, dy = DIRECTION_KEYS[sym]
                    if dx != 0 or dy == 0:
                        return self.activate_selected(event)
                    if self.selected is not None:
                        self.selected += dy
                        self.selected %= len(self.items)
                    else:
                        self.selected = 0 if dy == 1 else len(self.items) - 1
                    return None
                case tcod.event.MouseMotion(position=(_, y)):
                    y -= self.y
                    self.selected = y if 0 <= y < len(self.items) else None
                    return None
                case tcod.event.KeyDown(sym=KeySym.ESCAPE):
                    return self.on_cancel()
                case tcod.event.MouseButtonUp(button=tcod.event.MouseButton.RIGHT):
                    return self.on_cancel()
                case _:
                    return self.activate_selected(event)

        def activate_selected(self, event: tcod.event.Event) -> StateResult:
            """Call the selected menu items callback."""
            if self.selected is not None:
                return self.items[self.selected].on_event(event)
            return None

        def on_cancel(self) -> StateResult:
            """Handle escape or right click being pressed on menus."""
            return Pop()

        def on_draw(self, console: tcod.console.Console) -> None:
            """Render the menu."""
            game.state_tools.draw_previous_state(self, console)
            for i, item in enumerate(self.items):
                item.on_draw(console, x=self.x, y=self.y + i, highlight=i == self.selected)

Update states
==============================================================================

.. code-block:: python

    class MainMenu(game.menus.ListMenu):
        """Main/escape menu."""

        __slots__ = ()

        def __init__(self) -> None:
            """Initialize the main menu."""
            items = [
                game.menus.SelectItem("New game", self.new_game),
                game.menus.SelectItem("Quit", self.quit),
            ]
            if hasattr(g, "world"):
                items.insert(0, game.menus.SelectItem("Continue", self.continue_))

            super().__init__(
                items=tuple(items),
                selected=0,
                x=5,
                y=5,
            )

        @staticmethod
        def continue_() -> StateResult:
            """Return to the game."""
            return Reset(InGame())

        @staticmethod
        def new_game() -> StateResult:
            """Begin a new game."""
            g.world = game.world_tools.new_world()
            return Reset(InGame())

        @staticmethod
        def quit() -> StateResult:
            """Close the program."""
            raise SystemExit()

.. code-block:: python
    :emphasize-lines: 2,5,19-23

    @attrs.define()
    class InGame(State):
        """Primary in-game state."""

        def on_event(self, event: tcod.event.Event) -> StateResult:
            """Handle events for the in-game state."""
            (player,) = g.world.Q.all_of(tags=[IsPlayer])
            match event:
                case tcod.event.Quit():
                    raise SystemExit()
                case tcod.event.KeyDown(sym=sym) if sym in DIRECTION_KEYS:
                    player.components[Position] += DIRECTION_KEYS[sym]
                    # Auto pickup gold
                    for gold in g.world.Q.all_of(components=[Gold], tags=[player.components[Position], IsItem]):
                        player.components[Gold] += gold.components[Gold]
                        text = f"Picked up {gold.components[Gold]}g, total: {player.components[Gold]}g"
                        g.world[None].components[("Text", str)] = text
                        gold.clear()
                    return None
                case tcod.event.KeyDown(sym=KeySym.ESCAPE):
                    return Push(MainMenu())
                case _:
                    return None

        ...

Update main.py
==============================================================================

Now ``main.py`` can be edited to use the global variables and the new game loop.

Add :python:`import g` and :python:`import game.state_tools`.
Replace references to ``console`` with ``g.console``.

States are initialed by assigning a list with the initial state to ``g.states``.
The previous game loop is replaced by a call to :python:`game.state_tools.main_loop()`.

.. code-block:: python
    :emphasize-lines: 3-4,13-16

    ...

    import g
    import game.state_tools
    import game.states

    def main() -> None:
        """Entry point function."""
        tileset = tcod.tileset.load_tilesheet(
            "data/Alloy_curses_12x12.png", columns=16, rows=16, charmap=tcod.tileset.CHARMAP_CP437
        )
        tcod.tileset.procedural_block_elements(tileset=tileset)
        g.console = tcod.console.Console(80, 50)
        g.states = [game.states.MainMenu()]
        with tcod.context.new(console=g.console, tileset=tileset) as g.context:
            game.state_tools.main_loop()
    ...

After this you can test the game.
There should be no visible differences from before.

You can review the part-3 source code `here <https://github.com/HexDecimal/python-tcod-tutorial-2023/tree/part-3>`_.

.. rubric:: Footnotes

.. [#slots] This is done to prevent subclasses from requiring a ``__dict__`` attribute.
    See :any:`slots` for a detailed explanation of what they are.

.. [#identity] See :any:`is`.
    Since ``State`` classes use ``attrs`` they might compare equal when they're not the same object.
    This means :python:`list.index` won't work for this case.

.. _Protocol: https://mypy.readthedocs.io/en/stable/protocols.html
