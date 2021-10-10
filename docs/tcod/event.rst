tcod.event - SDL2 Event Handling
================================

.. automodule:: tcod.event
    :members:
    :member-order: bysource
    :exclude-members:
        KeySym, Scancode, Modifier, get, wait


Getting events
--------------

The primary way to capture events is with the :any:`tcod.event.get` and :any:`tcod.event.wait` functions.
These functions return events in a loop until the internal event queue is empty.
Use :func:`isinstance`, :any:`tcod.event.EventDispatch`, or `match statements <https://docs.python.org/3/tutorial/controlflow.html#match-statements>`_
(introduced in Python 3.10) to determine which event was returned.

.. autofunction:: tcod.event.get
.. autofunction:: tcod.event.wait

Keyboard Enums
--------------

- :class:`KeySym`: Keys based on their glyph.
- :class:`Scancode`: Keys based on their physical location.
- :class:`Modifier`: Keyboard modifier keys.

.. autoclass:: KeySym
    :members:

.. autoclass:: Scancode
    :members:

.. autoclass:: Modifier
    :members:
    :member-order: bysource
    :undoc-members:
