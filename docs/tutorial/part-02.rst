.. _part-2:

Part 2 - Entities
##############################################################################

.. include:: notice.rst

In part 2 entities will be added and a new state will be created to handle them.
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

Organizing globals
==============================================================================

There are a few variables which will need to be accessible from multiple modules.
Any global variables which might be assigned from other modules will need to a tracked and handled with care.

Create a new module: ``g.py`` [#g]_.
This module is exceptional and will be placed at the top-level instead of in the ``game`` folder.

In ``g.py`` import ``tcod.context`` and ``tcod.ecs``.

``context`` from ``main.py`` will now be annotated in ``g.py`` by adding the line :python:`context: tcod.context.Context` by itself.
Notice that is this only a type-hinted name and nothing is assigned to it.
This means that type-checking will assume the variable always exists but using it before it is assigned will crash at run-time.

``main.py`` should add :python:`import g` and replace the variables named ``context`` with ``g.context``.

Then add the :python:`world: tcod.ecs.Registry` global to hold the ECS scope.

It is important to document all variables placed in this module with docstrings.

.. code-block:: python

    """This module stores globally mutable variables used by this program."""

    from __future__ import annotations

    import tcod.context
    import tcod.ecs

    context: tcod.context.Context
    """The window managed by tcod."""

    world: tcod.ecs.Registry
    """The active ECS registry and current session."""

Ideally you should not overuse this module for too many things.
When a variable can either be taken as a function parameter or accessed as a global then passing as a parameter is always preferable.

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

New InGame state
==============================================================================

Now there is a new ECS world but the example state does not know how to render it.
A new state needs to be made which is aware of the new entities.

Before adding a new state it is time to add a more complete set of directional keys.
Create a new module called ``game/constants.py``.
Keys will be mapped to direction using a dictionary which can be reused anytime we want to know how a key translates to a direction.
Use :python:`from tcod.event import KeySym` to make ``KeySym`` enums easier to write.

``game/constants.py`` should look like this:

.. code-block:: python

    """Global constants are stored here."""

    from __future__ import annotations

    from typing import Final

    from tcod.event import KeySym

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

Create a new module called ``game/states.py``.
``states`` is for derived classes, ``state`` is for the abstract class.
New states will be created in this module and this module will be allowed to import many first party modules without issues.

Create a new :python:`class InGame:` decorated with :python:`@attrs.define()`.
States will always use ``g.world`` to access the ECS registry.

.. code-block:: python

    @attrs.define()
    class InGame:
        """Primary in-game state."""
        ...

Create an ``on_event`` and ``on_draw`` method matching the ``ExampleState`` class.
Copying ``ExampleState`` and modifying it should be enough since this wil replace ``ExampleState``.

Now to do an tcod-ecs query to fetch the player entity.
In tcod-ecs queries most often start with :python:`g.world.Q.all_of(components=[], tags=[])`.
Which components and tags are asked for will narrow down the returned set of entities to only those matching the requirements.
The query to fetch player entities is :python:`g.world.Q.all_of(tags=[IsPlayer])`.
We expect only one player so the result will be unpacked into a single name: :python:`(player,) = g.world.Q.all_of(tags=[IsPlayer])`.

Next is to handle the event.
Handling :python:`case tcod.event.Quit():` is the same as before: :python:`raise SystemExit`.

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
Add the entities ``Gold`` component to the players similar component.
Keep in mind that ``Gold`` is treated like an ``int`` so its usage is predictable.

Format the added and total of gold using a Python f-string_: :python:`text = f"Picked up {gold.components[Gold]}g, total: {player.components[Gold]}g"`.
Store ``text`` globally in the ECS registry with :python:`g.world[None].components[("Text", str)] = text`.
This is done as two lines to avoid creating a line with an excessive length.

Then use :python:`gold.clear()` at the end to remove all components and tags from the gold entity which will effectively delete it.

.. code-block:: python

        ...
        def on_event(self, event: tcod.event.Event) -> None:
            """Handle events for the in-game state."""
            (player,) = g.world.Q.all_of(tags=[IsPlayer])
            match event:
                case tcod.event.Quit():
                    raise SystemExit
                case tcod.event.KeyDown(sym=sym) if sym in DIRECTION_KEYS:
                    player.components[Position] += DIRECTION_KEYS[sym]
                    # Auto pickup gold
                    for gold in g.world.Q.all_of(components=[Gold], tags=[player.components[Position], IsItem]):
                        player.components[Gold] += gold.components[Gold]
                        text = f"Picked up {gold.components[Gold]}g, total: {player.components[Gold]}g"
                        g.world[None].components[str] = text
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

That ends the entity rendering loop.
Next is to print the ``("Text", str)`` component if it exists.
A normal access will raise ``KeyError`` if the component is accessed before being assigned.
This case will be handled by the ``.get`` method of the ``Entity.components`` attribute.
:python:`g.world[None].components.get(("Text", str))` will return :python:`None` instead of raising ``KeyError``.
Assigning this result to ``text`` and then checking :python:`if text:` will ensure that ``text`` within the branch is not None and that the string is not empty.
We will not use ``text`` outside of the branch, so an assignment expression can be used here to check and assign the name at the same time with :python:`if text := g.world[None].components.get(("Text", str)):`.

In this branch you will print ``text`` to the bottom of the console with a white foreground and black background.
The call to do this is :python:`console.print(x=0, y=console.height - 1, string=text, fg=(255, 255, 255), bg=(0, 0, 0))`.

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

            if text := g.world[None].components.get(("Text", str)):
                console.print(x=0, y=console.height - 1, string=text, fg=(255, 255, 255), bg=(0, 0, 0))

Verify the indentation of the ``if`` branch is correct.
It should be at the same level as the ``for`` loop and not inside of it.

``game/states.py`` should now look like this:

.. code-block:: python

    """A collection of game states."""

    from __future__ import annotations

    import attrs
    import tcod.console
    import tcod.event

    import g
    from game.components import Gold, Graphic, Position
    from game.constants import DIRECTION_KEYS
    from game.tags import IsItem, IsPlayer


    @attrs.define()
    class InGame:
        """Primary in-game state."""

        def on_event(self, event: tcod.event.Event) -> None:
            """Handle events for the in-game state."""
            (player,) = g.world.Q.all_of(tags=[IsPlayer])
            match event:
                case tcod.event.Quit():
                    raise SystemExit
                case tcod.event.KeyDown(sym=sym) if sym in DIRECTION_KEYS:
                    player.components[Position] += DIRECTION_KEYS[sym]
                    # Auto pickup gold
                    for gold in g.world.Q.all_of(components=[Gold], tags=[player.components[Position], IsItem]):
                        player.components[Gold] += gold.components[Gold]
                        text = f"Picked up {gold.components[Gold]}g, total: {player.components[Gold]}g"
                        g.world[None].components[("Text", str)] = text
                        gold.clear()

        def on_draw(self, console: tcod.console.Console) -> None:
            """Draw the standard screen."""
            for entity in g.world.Q.all_of(components=[Position, Graphic]):
                pos = entity.components[Position]
                if not (0 <= pos.x < console.width and 0 <= pos.y < console.height):
                    continue
                graphic = entity.components[Graphic]
                console.rgb[["ch", "fg"]][pos.y, pos.x] = graphic.ch, graphic.fg

            if text := g.world[None].components.get(("Text", str)):
                console.print(x=0, y=console.height - 1, string=text, fg=(255, 255, 255), bg=(0, 0, 0))

Main script update
==============================================================================

Back to ``main.py``.
At this point you should know to import the modules needed.

The ``ExampleState`` class is obsolete and will be removed.
``state`` will be created with :python:`game.states.InGame()` instead.

If you have not replaced ``context`` with ``g.context`` yet then do it now.

Add :python:`g.world = game.world_tools.new_world()` before the main loop.

``main.py`` will look like this:

.. code-block:: python
    :emphasize-lines: 10-12,22-24,28

    #!/usr/bin/env python3
    """Main entry-point module. This script is used to start the program."""

    from __future__ import annotations

    import tcod.console
    import tcod.context
    import tcod.event
    import tcod.tileset

    import g
    import game.states
    import game.world_tools


    def main() -> None:
        """Entry point function."""
        tileset = tcod.tileset.load_tilesheet(
            "data/Alloy_curses_12x12.png", columns=16, rows=16, charmap=tcod.tileset.CHARMAP_CP437
        )
        tcod.tileset.procedural_block_elements(tileset=tileset)
        console = tcod.console.Console(80, 50)
        state = game.states.InGame()
        g.world = game.world_tools.new_world()
        with tcod.context.new(console=console, tileset=tileset) as g.context:
            while True:  # Main loop
                console.clear()  # Clear the console before any drawing
                state.on_draw(console)  # Draw the current state
                g.context.present(console)  # Render the console to the window and show it
                for event in tcod.event.wait():  # Event loop, blocks until pending events exist
                    print(event)
                    state.on_event(event)  # Dispatch events to the state


    if __name__ == "__main__":
        main()

Now you can play a simple game where you wander around collecting gold.

You can review the part-2 source code `here <https://github.com/HexDecimal/python-tcod-tutorial-2023/tree/part-2>`_.

.. rubric:: Footnotes

.. [#g] ``global``, ``globals``, and ``glob`` were already taken by keywords, built-ins, and the standard library.
        The alternatives are to either put this in the ``game`` namespace or to add an underscore such as ``globals_.py``.

.. _f-string: https://docs.python.org/3/tutorial/inputoutput.html#formatted-string-literals
