.. _part-2:

Part 2 - Entities
##############################################################################

.. include:: notice.rst

In part 2 entities will be added and the state system will be refactored to be more generic.
This part will also begin to split logic into multiple Python modules using a namespace called ``game``.

Entities will be handled with an ECS implementation, in this case: `tcod-ecs`_.
``tcod-ecs`` is a standalone package and is installed separately from ``tcod``.
Use :shell:`pip install tcod-ecs` to install this package.

Namespace package
==============================================================================

Create a new folder called ``game`` and inside the folder create a new python file named ``__init__.py``.
``game/__init__.py`` only needs a docstring describing that it is a namespace package:

.. code-block:: python

    """Game namespace package."""

This package will be used to organize new modules.

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

The ``ExampleState`` class does not need to be updated since it is already a structural subtype of ``State``.
Note that subclasses of ``State`` will never be in same module as ``State``, this will be the same for all abstract classes.

Organizing globals
==============================================================================

There are a few variables which will need to be accessible from multiple modules.
Any global variables which might be assigned from other modules will need to a tracked and handled with care.

Create a new module: ``g.py`` [#g]_.
This module is exceptional and will be placed at the top-level instead of in the ``game`` folder.

``console`` and ``context`` from ``main.py`` will now be annotated in ``g.py``.
These will not be assigned here, only annotated with a type-hint.

A new global will be added: :python:`states: list[game.state.State] = []`.
States are implemented as a list/stack to support `pushdown automata <https://gameprogrammingpatterns.com/state.html#pushdown-automata>`_.
Representing states as a stack makes it easier to implement popup windows, menus, and other "history aware" states.

Finally :python:`world: tcod.ecs.Registry` will be added to hold the ECS scope.

It is important to document all variables placed in this module with docstrings.

.. code-block:: python

    """This module stores globally mutable variables used by this program."""
    from __future__ import annotations

    import tcod.console
    import tcod.context
    import tcod.ecs

    import game.state

    console: tcod.console.Console
    """The main console."""

    context: tcod.context.Context
    """The window managed by tcod."""

    states: list[game.state.State] = []
    """A stack of states with the last item being the active state."""

    world: tcod.ecs.Registry
    """The active ECS registry and current session."""

Now other modules can :python:`import g` to access global variables.

Ideally you should not overuse this module for too many things.
When a variables can either be taken as a function parameter or accessed as a global then passing as a parameter is always preferable.

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


ECS tags
==============================================================================

Create ``game/tags.py``.
This will hold some sentinel values to be used as tags for ``tcod-ecs``.
These tags can be anything that's both unique and unchanging, in this case Python strings are used.

For example :python:`IsPlayer: Final = "IsPlayer"` will tag an object as being controlled by the player.
The name is ``IsPlayer`` and string is the same as the name.
The ``Final`` annotation clarifies that this a constant.
Sentinel values for ``tcod-ecs`` are named like classes, similar to names like :python:`None` or :python:`False`.

Repeat this for ``IsActor`` and ``IsItem`` tags.
The ``game/tags.py`` module should look like this:

.. code-block:: python

    """Collection of common tags."""
    from __future__ import annotations

    from typing import Final

    IsPlayer: Final = "IsPlayer"
    """Entity is the player."""

    IsActor: Final = "IsActor"
    """Entity is an actor."""

    IsItem: Final = "IsItem"
    """Entity is an item."""

ECS components
==============================================================================

Next is a new ``game/components.py`` module.
This will hold the components for the graphics and position of entities.

Start by adding an import for ``attrs``.
The ability to easily design small classes which are frozen/immutable is important for working with ``tcod-ecs``.

The first component will be a ``Position`` class.
This class will be decorated with :python:`@attrs.define(frozen=True)`.
For attributes this class will have :python:`x: int` and :python:`y: int`.

It will be common to add vectors to a ``Position`` with code such as :python:`new_pos: Position = Position(0, 0) + (0, 1)`.
Create the dunder method :python:`def __add__(self, direction: tuple[int, int]) -> Self:` to allow this syntax.
Unpack the input with :python:`x, y = direction`.
:python:`self.__class__` is the current class so :python:`self.__class__(self.x + x, self.y + y)` will create a new instance with the direction added to the previous values.

The new class will look like this:

.. code-block:: python

    @attrs.define(frozen=True)
    class Position:
        """An entities position."""

        x: int
        y: int

        def __add__(self, direction: tuple[int, int]) -> Self:
            """Add a vector to this position."""
            x, y = direction
            return self.__class__(self.x + x, self.y + y)

Because ``Position`` is immutable, ``tcod-ecs`` is able to reliably track changes to this component.
Normally you can only query entities by which components they have.
A callback can be registered with ``tcod-ecs`` to mirror component values as tags.
This allows querying an entity by its exact position.

Add :python:`import tcod.ecs.callbacks` and :python:`from tcod.ecs import Entity`.
Then create the new function :python:`def on_position_changed(entity: Entity, old: Position | None, new: Position | None) -> None:` decorated with :python:`@tcod.ecs.callbacks.register_component_changed(component=Position)`.
This function is called when the ``Position`` component is either added, removed, or modified by assignment.
The goal of this function is to mirror the current position to the :class:`set`-like attribute ``entity.tags``.

:python:`if old == new:` then a position was assigned its own value or an equivalent value.
The cost of discarding and adding the same value can sometimes be high so this case should be guarded and ignored.
:python:`if old is not None:` then the value tracked by ``entity.tags`` is outdated and must be removed.
:python:`if new is not None:` then ``new`` is the up-to-date value to be tracked by ``entity.tags``.

The function should look like this:

.. code-block:: python

    @tcod.ecs.callbacks.register_component_changed(component=Position)
    def on_position_changed(entity: Entity, old: Position | None, new: Position | None) -> None:
        """Mirror position components as a tag."""
        if old == new:  # New position is equivalent to its previous value
            return  # Ignore and return
        if old is not None:  # Position component removed or changed
            entity.tags.discard(old)  # Remove old position from tags
        if new is not None:  # Position component added or changed
            entity.tags.add(new)  # Add new position to tags

Next is the ``Graphic`` component.
This will have the attributes :python:`ch: int = ord("!")` and :python:`fg: tuple[int, int, int] = (255, 255, 255)`.
By default all new components should be marked as frozen.

.. code-block:: python

    @attrs.define(frozen=True)
    class Graphic:
        """An entities icon and color."""

        ch: int = ord("!")
        fg: tuple[int, int, int] = (255, 255, 255)

One last component: ``Gold``.
Define this as :python:`Gold: Final = ("Gold", int)`.
``(name, type)`` is tcod-ecs specific syntax to handle multiple components sharing the same type.

.. code-block:: python

    Gold: Final = ("Gold", int)
    """Amount of gold."""

That was the last component.
The ``game/components.py`` module should look like this:

.. code-block:: python

    """Collection of common components."""
    from __future__ import annotations

    from typing import Final, Self

    import attrs
    import tcod.ecs.callbacks
    from tcod.ecs import Entity


    @attrs.define(frozen=True)
    class Position:
        """An entities position."""

        x: int
        y: int

        def __add__(self, direction: tuple[int, int]) -> Self:
            """Add a vector to this position."""
            x, y = direction
            return self.__class__(self.x + x, self.y + y)


    @tcod.ecs.callbacks.register_component_changed(component=Position)
    def on_position_changed(entity: Entity, old: Position | None, new: Position | None) -> None:
        """Mirror position components as a tag."""
        if old == new:
            return
        if old is not None:
            entity.tags.discard(old)
        if new is not None:
            entity.tags.add(new)


    @attrs.define(frozen=True)
    class Graphic:
        """An entities icon and color."""

        ch: int = ord("!")
        fg: tuple[int, int, int] = (255, 255, 255)


    Gold: Final = ("Gold", int)
    """Amount of gold."""

ECS entities and registry
==============================================================================

Now it is time to create entities.
To do that you need to create the ECS registry.

Make a new script called ``game/world_tools.py``.
This module will be used to create the ECS registry.

Random numbers from :mod:`random` will be used.
In this case we want to use ``Random`` as a component so add :python:`from random import Random`.
Get the registry with :python:`from tcod.ecs import Registry`.
Collect all our components and tags with :python:`from game.components import Gold, Graphic, Position` and :python:`from game.tags import IsActor, IsItem, IsPlayer`.

This module will have one function: :python:`def new_world() -> Registry:`.
Think of the ECS registry as containing the world since this is how it will be used.
Start this function with :python:`world = Registry()`.

Entities are referenced with the syntax :python:`world[unique_id]`.
If the same ``unique_id`` is used then you will access the same entity.
:python:`new_entity = world[object()]` is the syntax to spawn new entities because ``object()`` is always unique.
Whenever a global entity is needed then :python:`world[None]` will be used.

Create an instance of :python:`Random()` and assign it to both :python:`world[None].components[Random]` and ``rng``.
This can done on one line with :python:`rng = world[None].components[Random] = Random()`.

Next create the player entity with :python:`player = world[object()]`.
Assign the following components to the new player entity: :python:`player.components[Position] = Position(5, 5)`, :python:`player.components[Graphic] = Graphic(ord("@"))`, and :python:`player.components[Gold] = 0`.
Then update the players tags with :python:`player.tags |= {IsPlayer, IsActor}`.

To add some variety we will scatter gold randomly across the world.
Start a for-loop with :python:`for _ in range(10):` then create a ``gold`` entity in this loop.

The ``Random`` instance ``rng`` has access to functions from Python's random module such as :any:`random.randint`.
Set ``Position`` to :python:`Position(rng.randint(0, 20), rng.randint(0, 20))`.
Set ``Graphic`` to :python:`Graphic(ord("$"), fg=(255, 255, 0))`.
Set ``Gold`` to :python:`rng.randint(1, 10)`.
Then add ``IsItem`` as a tag.

Once the for-loop exits then :python:`return world`.
Make sure :python:`return` has the correct indentation and is not part of the for-loop or else you will only spawn one gold.

``game/world_tools.py`` should look like this:

.. code-block:: python

    """Functions for working with worlds."""
    from __future__ import annotations

    from random import Random

    from tcod.ecs import Registry

    from game.components import Gold, Graphic, Position
    from game.tags import IsActor, IsItem, IsPlayer


    def new_world() -> Registry:
        """Return a freshly generated world."""
        world = Registry()

        rng = world[None].components[Random] = Random()

        player = world[object()]
        player.components[Position] = Position(5, 5)
        player.components[Graphic] = Graphic(ord("@"))
        player.components[Gold] = 0
        player.tags |= {IsPlayer, IsActor}

        for _ in range(10):
            gold = world[object()]
            gold.components[Position] = Position(rng.randint(0, 20), rng.randint(0, 20))
            gold.components[Graphic] = Graphic(ord("$"), fg=(255, 255, 0))
            gold.components[Gold] = rng.randint(1, 10)
            gold.tags |= {IsItem}

        return world

New in-game state
==============================================================================

Now there is a new ECS world but the example state does not know how to render it.
A new state needs to be made which is aware of the new entities.

Create a new script called ``game/states.py``.
``states`` is for derived classes, ``state`` is for the abstract class.
New states will be created in this module and this module will be allowed to import many first party modules without issues.

Before adding a new state it is time to add a more complete set of directional keys.
These will be added as a dictionary and can be reused anytime we want to know how a key translates to a direction.
Use :python:`from tcod.event import KeySym` to make ``KeySym`` enums easier to write.
Then add the following:

.. code-block:: python

    DIRECTION_KEYS: Final = {
        # Arrow keys
        KeySym.LEFT: (-1, 0),
        KeySym.RIGHT: (1, 0),
        KeySym.UP: (0, -1),
        KeySym.DOWN: (0, 1),
        # Arrow key diagonals
        KeySym.HOME: (-1, -1),
        KeySym.END: (-1, 1),
        KeySym.PAGEUP: (1, -1),
        KeySym.PAGEDOWN: (1, 1),
        # Keypad
        KeySym.KP_4: (-1, 0),
        KeySym.KP_6: (1, 0),
        KeySym.KP_8: (0, -1),
        KeySym.KP_2: (0, 1),
        KeySym.KP_7: (-1, -1),
        KeySym.KP_1: (-1, 1),
        KeySym.KP_9: (1, -1),
        KeySym.KP_3: (1, 1),
        # VI keys
        KeySym.h: (-1, 0),
        KeySym.l: (1, 0),
        KeySym.k: (0, -1),
        KeySym.j: (0, 1),
        KeySym.y: (-1, -1),
        KeySym.b: (-1, 1),
        KeySym.u: (1, -1),
        KeySym.n: (1, 1),
    }

Create a new :python:`class InGame:` decorated with :python:`@attrs.define(eq=False)`.
States will always use ``g.world`` to access the ECS registry.
States prefer ``console`` as a parameter over the global ``g.console`` so always use ``console`` when it exists.

.. code-block:: python

    @attrs.define(eq=False)
    class InGame:
        """Primary in-game state."""
        ...

Create an ``on_event`` method matching the ``State`` protocol.
Copying these methods from ``State`` or ``ExampleState`` should be enough.

Now to do an tcod-ecs query to fetch the player entity.
In tcod-ecs queries most often start with :python:`g.world.Q.all_of(components=[], tags=[])`.
Which components and tags are asked for will narrow down the returned set of entities to only those matching the requirements.
The query to fetch player entities is :python:`g.world.Q.all_of(tags=[IsPlayer])`.
We expect only one player so the result will be unpacked into a single name: :python:`(player,) = g.world.Q.all_of(tags=[IsPlayer])`.

Next is to handle the event.
Handling :python:`case tcod.event.Quit():` is the same as before: :python:`raise SystemExit()`.

The case for direction keys will now be done in a single case: :python:`case tcod.event.KeyDown(sym=sym) if sym in DIRECTION_KEYS:`.
``sym=sym`` assigns from the event attribute to a local name.
The left side is the ``event.sym`` attribute and right side is the local name ``sym`` being assigned to.
The case also has a condition which must pass for this branch to be taken and in this case we ensure that only keys from the ``DIRECTION_KEYS`` dictionary are valid ``sym``'s.

Inside this branch moving the player is simple.
Access the ``(x, y)`` vector with :python:`DIRECTION_KEYS[sym]` and use ``+=`` to add it to the players current ``Position`` component.
This triggers the earlier written ``__add__`` dunder method and ``on_position_changed`` callback.

Now that the player has moved it would be a good time to interact with the gold entities.
The query to see if the player has stepped on gold is to check for whichever entities have a ``Gold`` component, an ``IsItem`` tag, and the players current position as a tag.
The query for this is :python:`g.world.Q.all_of(components=[Gold], tags=[player.components[Position], IsItem]):`.

We will iterate over whatever matches this query using a :python:`for gold in ...:` loop.
Add the entities ``Gold`` component to the player.
Keep in mind that ``Gold`` is treated like an ``int`` so its usage is predictable.
Now print the current amount of gold using :python:`print(f"Picked up {gold.components[Gold]}g, total: {player.components[Gold]}g")`.
Then use :python:`gold.clear()` at the end to remove all components and tags from the gold entity which will effectively delete it.

.. code-block:: python

        ...
        def on_event(self, event: tcod.event.Event) -> None:
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
                        print(f"Picked up {gold.components[Gold]}g, total: {player.components[Gold]}g")
                        gold.clear()
        ...

Now start with the ``on_draw`` method.
Any entity with both a ``Position`` and a ``Graphic`` is drawable.
Iterate over these entities with :python:`for entity in g.world.Q.all_of(components=[Position, Graphic]):`.
Accessing components can be slow in a loop, so assign components to local names before using them (:python:`pos = entity.components[Position]` and :python:`graphic = entity.components[Graphic]`).

Check if a components position is in the bounds of the console.
:python:`0 <= pos.x < console.width and 0 <= pos.y < console.height` tells if the position is in bounds.
Instead of nesting this method further, this check should be a guard using :python:`if not (...):` and :python:`continue`.

Draw the graphic by assigning it to the consoles Numpy array directly with :python:`console.rgb[["ch", "fg"]][pos.y, pos.x] = graphic.ch, graphic.fg`.
``console.rgb`` is a ``ch,fg,bg`` array and :python:`[["ch", "fg"]]` narrows it down to only ``ch,fg``.
The array is in C row-major memory order so you access it with yx (or ij) ordering.

.. code-block:: python

        ...
        def on_draw(self, console: tcod.console.Console) -> None:
            """Draw the standard screen."""
            for entity in g.world.Q.all_of(components=[Position, Graphic]):
                pos = entity.components[Position]
                if not (0 <= pos.x < console.width and 0 <= pos.y < console.height):
                    continue
                graphic = entity.components[Graphic]
                console.rgb[["ch", "fg"]][pos.y, pos.x] = graphic.ch, graphic.fg

``game/states.py`` should now look like this:

.. code-block:: python

    """A collection of game states."""
    from __future__ import annotations

    from typing import Final

    import attrs
    import tcod.console
    import tcod.event
    from tcod.event import KeySym

    import g
    from game.components import Gold, Graphic, Position
    from game.tags import IsItem, IsPlayer

    DIRECTION_KEYS: Final = {
        # Arrow keys
        KeySym.LEFT: (-1, 0),
        KeySym.RIGHT: (1, 0),
        KeySym.UP: (0, -1),
        KeySym.DOWN: (0, 1),
        # Arrow key diagonals
        KeySym.HOME: (-1, -1),
        KeySym.END: (-1, 1),
        KeySym.PAGEUP: (1, -1),
        KeySym.PAGEDOWN: (1, 1),
        # Keypad
        KeySym.KP_4: (-1, 0),
        KeySym.KP_6: (1, 0),
        KeySym.KP_8: (0, -1),
        KeySym.KP_2: (0, 1),
        KeySym.KP_7: (-1, -1),
        KeySym.KP_1: (-1, 1),
        KeySym.KP_9: (1, -1),
        KeySym.KP_3: (1, 1),
        # VI keys
        KeySym.h: (-1, 0),
        KeySym.l: (1, 0),
        KeySym.k: (0, -1),
        KeySym.j: (0, 1),
        KeySym.y: (-1, -1),
        KeySym.b: (-1, 1),
        KeySym.u: (1, -1),
        KeySym.n: (1, 1),
    }


    @attrs.define(eq=False)
    class InGame:
        """Primary in-game state."""

        def on_event(self, event: tcod.event.Event) -> None:
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
                        print(f"Picked up ${gold.components[Gold]}, total: ${player.components[Gold]}")
                        gold.clear()

        def on_draw(self, console: tcod.console.Console) -> None:
            """Draw the standard screen."""
            for entity in g.world.Q.all_of(components=[Position, Graphic]):
                pos = entity.components[Position]
                if not (0 <= pos.x < console.width and 0 <= pos.y < console.height):
                    continue
                graphic = entity.components[Graphic]
                console.rgb[["ch", "fg"]][pos.y, pos.x] = graphic.ch, graphic.fg

Back to ``main.py``.
At this point you should know which imports to add and which are no longed needed.
``ExampleState`` should be removed.
``g.state`` will be initialized with :python:`[game.states.InGame()]` instead.
Add :python:`g.world = game.world_tools.new_world()`.

``main.py`` will look like this:

.. code-block:: python
    :emphasize-lines: 5-12,22-23

    #!/usr/bin/env python3
    """Main entry-point module. This script is used to start the program."""
    from __future__ import annotations

    import tcod.console
    import tcod.context
    import tcod.tileset

    import g
    import game.state_tools
    import game.states
    import game.world_tools


    def main() -> None:
        """Entry point function."""
        tileset = tcod.tileset.load_tilesheet(
            "data/Alloy_curses_12x12.png", columns=16, rows=16, charmap=tcod.tileset.CHARMAP_CP437
        )
        tcod.tileset.procedural_block_elements(tileset=tileset)
        g.console = tcod.console.Console(80, 50)
        g.states = [game.states.InGame()]
        g.world = game.world_tools.new_world()
        with tcod.context.new(console=g.console, tileset=tileset) as g.context:
            game.state_tools.main_loop()


    if __name__ == "__main__":
        main()

Now you can play a simple game where you wander around collecting gold.

You can review the part-2 source code `here <https://github.com/HexDecimal/python-tcod-tutorial-2023/tree/part-2>`_.

.. rubric:: Footnotes

.. [#slots] This is done to prevent subclasses from requiring a ``__dict__`` attribute.
                If you are still wondering what ``__slots__`` is then `the Python docs have a detailed explanation <https://docs.python.org/3/reference/datamodel.html#slots>`_.

.. [#g] ``global``, ``globals``, and ``glob`` were already taken by keywords, built-ins, and the standard library.
        The alternatives are to either put this in the ``game`` namespace or to add an underscore such as ``globals_.py``.

.. _Protocol: https://mypy.readthedocs.io/en/stable/protocols.html
