
Glossary
========

.. glossary::

    console defaults
        The default values implied by any Console print or put functions which
        don't explicitly ask for them as parameters.

    libtcod-cffi
        This is the `cffi` implementation of libtcodpy, the original was
        made using `ctypes` which was more difficult to maintain.

        `libtcod-cffi` is now part of :term:`python-tcod`.

    python-tcod
        `python-tcod` is a superset of the :term:`libtcodpy` API.  The major
        additions include class functionality in returned objects, no manual
        memory management, pickle-able objects, and `numpy` array attributes
        in most objects.

        The `numpy` attributes in particular can be used to dramatically speed
        up the performance of your program compared to using :term:`libtcodpy`.

        `python-tcod` is installed as part of :term:`python-tdl`

    python-tdl
        `tdl` is a high-level wrapper over :term:`libtcodpy` although it now
        uses :term:`python-tcod`, it doesn't do anything that you couldn't do
        yourself with just :term:`libtcodpy` and Python.

        Currently no new features are planned for `tdl`, instead new features
        are added to `libtcod` itself and then ported to :term:`python-tcod`.

        :term:`python-tcod` and :term:`libtcodpy` are included in installations
        of `python-tdl`.

    libtcodpy
        `libtcodpy` is more or less a direct port of `libtcod`'s C API to
        Python.  This caused a handful of issues including instances needing
        to be freed manually or a memory leak will occur and some functions
        performing badly in Python due to the need to call them frequently.

        These issues are fixed in :term:`python-tcod` which implements the full
        `libtcodpy` API.  If :term:`python-tcod` is installed then imports
        of `libtcodpy` are aliased to the `tcod` module.
        So if you come across a project using the original `libtcodpy` you can
        delete the `libtcodpy/` folder and then :term:`python-tcod` will load
        instead.
