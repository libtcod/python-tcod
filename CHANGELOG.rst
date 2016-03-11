===========
 Changelog
===========
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
