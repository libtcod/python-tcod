
Glossary
========

.. glossary::

    color control
    color controls
        .. data:: tcod.COLCTRL_1
        .. data:: tcod.COLCTRL_2
        .. data:: tcod.COLCTRL_3
        .. data:: tcod.COLCTRL_4
        .. data:: tcod.COLCTRL_5

            Configurable color control constant which can be set up with
            :any:`tcod.console_set_color_control`.

        .. data:: tcod.COLCTRL_STOP
        .. data:: tcod.COLCTRL_FORE_RGB
        .. data:: tcod.COLCTRL_BACK_RGB

    console defaults
        The default values implied by any Console print or put functions which
        don't explicitly ask for them as parameters.

    libtcod-cffi
    python-tcod
        `python-tcod` is a superset of the :term:`libtcodpy` API.  The major
        additions include class functionality in returned objects, no manual
        memory management, pickle-able objects, and `numpy` array attributes
        in most objects.

        The `numpy` attributes in particular can be used to dynamically speed
        up the performance of your program compared to using :term:`libtcodpy`.

    tdl
    python-tdl
        `tdl` is a high-level wrapper over :term:`libtcodpy` and now
        :term:`python-tcod`, it usually doesn't do anything that you couldn't
        do yourself with just :term:`libtcodpy` and Python.

        Currently no new features are planned for `tdl`, instead new features
        are added to `libtcod` itself and then ported to
        :term:`python-tcod`.

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
