===========
 Changelog
===========
2.0a1
 * Realising how dumb it'd be to remote it,
   the userData parameter has been added back.
   Functions which use it are marked depreciated.
 * Python exceptions will now propagate out of libtcod callbacks.
 * Some libtcod object oriented functions now have Python class methods
   associated with them (only BSP for now, more will be added later.)
 * Regression tests were added.
   Focusing on backwards compatibilty with libtcodpy.
   Several neglected functions were fixed during this.
 * All libtcod allocations are handled by the Python garbage collector.
   You'll no longer have to call the delete functions on each object.
 * Now generates documentation for Read the Docs.
   You can find the latest documentation for libtcod-cffi
   `here <https://libtcod-cffi.readthedocs.io/en/latest/>`_.

2.0a0
 * updated to compile with libtcod-1.6.2 and SDL-2.0.4

1.0
 * sub packages have been removed to follow the libtcodpy API more closely
 * bsp and pathfinding functions which take a callback no longer have the
   userdata parameter, if you need to pass data then you should use functools,
   methods, or enclosing scope rules
 * numpy buffer alignment issues on some 64-bit OS's fixed

0.3
 * switched to using pycparser to compile libtcod headers, this may have
   included many more functions in tcod's namespace than before
 * parser custom listener fixed again, likely for good

0.2.12
 * version increment due to how extremely broken the non-Windows builds were
   (false alarm, this module is just really hard to run integrated tests on)

0.2.11
 * SDL is now bundled correctly in all Python wheels

0.2.10
 * now using GitHub integrations, gaps in platform support have been filled,
   there should now be wheels for Mac OSX and 64-bit Python on Windows
 * the building process was simplified from a linking standpoint, most
   libraries are now statically linked
 * parser module is broken again

0.2.9
 * Fixed crashes in list and parser modules

0.2.8
 * Fixed off by one error in fov buffer

0.2.7
 * Re-factored some code to reduce compiler warnings
 * Instructions on how to solve pip/cffi issues added to the readme
 * Official support for Python 3.5

0.2.6
 * Added requirements.txt to fix a common pip/cffi issue.
 * Provided SDL headers are now for Windows only.

0.2.5
 * Added /usr/include/SDL to include path

0.2.4
 * Compiler will now use distribution specific SDL header files before falling
   back on the included header files.

0.2.3
 * better Color performance
 * parser now works when using a custom listener class
 * SDL renderer callback now receives a accessible SDL_Surface cdata object.

0.2.2
 * This module can now compile and link properly on Linux

0.2.1
 * console_check_for_keypress and console_wait_for_keypress will work now
 * console_fill_foreground was fixed
 * console_init_root can now accept a regular string on Python 3

0.2.0
 * The library is now backwards compatible with the original libtcod.py module.
   Everything except libtcod's cfg parser is supported.

0.1.0
 * First version released
