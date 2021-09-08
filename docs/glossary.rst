
Glossary
========

.. glossary::

    console defaults
        The default values implied by any Console print or put functions which
        don't explicitly ask for them as parameters.

        These have been deprecated since version `8.5`.

    libtcod-cffi
        This is the `cffi` implementation of libtcodpy, the original was
        made using `ctypes` which was more difficult to maintain.

        `libtcod-cffi` has since been part of :term:`python-tcod` providing
        all of the :term:`libtcodpy` API until the newer features could be
        implemented.

    python-tcod
        `python-tcod` is a superset of the :term:`libtcodpy` API.  The major
        additions include class functionality in returned objects, no manual
        memory management, pickle-able objects, and `numpy` array attributes
        in most objects.

        The `numpy` functions in particular can be used to dramatically speed
        up the performance of a program compared to using :term:`libtcodpy`.

    python-tdl
        `tdl` is a high-level wrapper over :term:`libtcodpy` although it now
        uses :term:`python-tcod`, it doesn't do anything that you couldn't do
        yourself with just :term:`libtcodpy` and Python.

        It included a lot of core functions written in Python that most
        definitely shouldn't have been.  `tdl` was very to use, but the cost
        was severe performance issues throughout the entire module.
        This left it impractical for any real use as a roguelike library.

        Currently no new features are planned for `tdl`, instead new features
        are added to `libtcod` itself and then ported to :term:`python-tcod`.

        :term:`python-tdl` and :term:`libtcodpy` are included in installations
        of `python-tcod`.

    libtcodpy
        `libtcodpy` is more or less a direct port of `libtcod`'s C API to
        Python.
        This caused a handful of issues including instances needing to be
        freed manually or else a memory leak would occur, and many functions
        performing badly in Python due to the need to call them frequently.

        These issues are fixed in :term:`python-tcod` which implements the full
        `libtcodpy` API.  If :term:`python-tcod` is installed then imports
        of `libtcodpy` are aliased to the `tcod` module.
        So if you come across a project using the original `libtcodpy` you can
        delete the `libtcodpy/` folder and then :term:`python-tcod` will load
        instead.

    color control
    color controls
        Libtcod's old system which assigns colors to specific codepoints.
        See :any:`tcod.COLCTRL_STOP`, :any:`tcod.COLCTRL_FORE_RGB`, and :any:`tcod.COLCTRL_BACK_RGB` for examples.
