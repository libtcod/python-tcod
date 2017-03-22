===========
 Changelog
===========
3.0.1 - 2017-03-22
------------------
Changed
 * `KeyEvent`'s with `text` now have all their modifier keys set to False.
Fixed
 * Undefined behaviour in text events caused crashes on 32-bit builds.

3.0.0 - 2017-03-21
------------------
Added
 * `KeyEvent` supports libtcod text and meta keys.
Changed
 * `KeyEvent` parameters have been moved.
 * This version requires `libtcod-cffi>=2.3.0`.
Deprecated
 * `KeyEvent` camel capped attribute names are deprecated.
Fixed
 * Crashes with key-codes undefined by libtcod.
 * `tdl.map` typedef issues with libtcod-cffi.


2.0.1 - 2017-02-22
------------------
Fixed
 * `tdl.init` renderer was defaulted to OpenGL which is not supported in the
   current version of libtcod.

2.0.0 - 2017-02-15
------------------
Changed
 * Dependencies updated, tdl now requires libtcod-cffi 2.x.x
 * Some event behaviours have changed with SDL2, event keys might be different
   than what you expect.
Removed
 * Key repeat functions were removed from SDL2.
   `set_key_repeat` is now stubbed, and does nothing.

1.6.0
 * Console.blit methods can now take fg_alpha and bg_alpha parameters.

1.5.3
 * set_font no longer crashes when loading a file without the implied font size in its name

1.5.2
 * fixed non-square Map instances

1.5.1
 * fixed errors with Unicode and non-Unicode literals on Python 2
 * fixed attribute error in compute_fov

1.5.0
 * python-tdl distributions are now universal builds
 * new Map class
 * map.bresenham now returns a list
 * this release will require libtcod-cffi v0.2.3 or later

1.4.0
 * The DLL's have been moved into another library which you can find at https://github.com/HexDecimal/libtcod-cffi
   You can use this library to have some raw access to libtcod if you want.
   Plus it can be used alongside TDL.
 * The libtocd console objects in Console instances have been made public.
 * Added tdl.event.wait function.  This function can called with a timeout and
   can automatically call tdl.flush.

1.3.1
 * Fixed pathfinding regressions.

1.3.0
 * Updated backend to use python-cffi instead of ctypes.  This gives decent
   boost to speed in CPython and a drastic to boost in speed in PyPy.

1.2.0
 * The set_colors method now changes the default colors used by the draw_*
   methods.  You can use Python's Ellipsis to explicitly select default colors
   this way.
 * Functions and Methods renamed to match Python's style-guide PEP 8, the old
   function names still exist and are depreciated.
 * The fgcolor and bgcolor parameters have been shortened to fg and bg

1.1.7
 * Noise generator now seeds properly
 * The OS event queue will now be handled during a call to tdl.flush. This
   prevents a common newbie programmer hang where events are handled
   infrequently during long animations, simulations, or early development
 * Fixed a major bug that would cause a crash in later versions of Python 3

1.1.6
 * Fixed a race condition when importing on some platforms
 * Fixed a type issue with quickFOV on Linux
 * Added a bresenham function to the tdl.map module

1.1.5
 * a for loop can iterate over all coordinates of a Console
 * drawStr can be configured to scroll or raise an error
 * You can now configure or disable key repeating with tdl.event.setKeyRepeat
 * Typewriter class removed, use a Window instance for the same functionality
 * setColors method fixed

1.1.4
 * Merged the Typewriter and MetaConsole classes,
   You now have a virtual cursor with Console and Window objects
 * Fixed the clear method on the Window class
 * Fixed screenshot function
 * Fixed some drawing operations with unchanging backgrounds
 * Instances of Console and Noise can be pickled and copied
 * Added KeyEvent.keychar
 * Fixed event.keyWait, and now converts window closed events into Alt+F4

1.1.3
 * Some of the setFont parameters were incorrectly labeled and documented
 * setFont can auto-detect tilesets if the font sizes are in the filenames
 * Added some X11 unicode tilesets, including unifont.

1.1.2
 * Window title now defaults to the running scripts filename
 * Fixed incorrect deltaTime for App.update
 * App will no longer call tdl.flush on its own, you'll need to call this yourself
 * tdl.noise module added
 * clear method now defaults to black on black

1.1.1
 * map submodule added with AStar class and quickFOV function
 * new Typewriter class
 * most console functions can use Python-style negative indexes now
 * new App.runOnce method
 * rectangle geometry is less strict

1.1.0
 * KeyEvent.keyname is now KeyEvent.key
 * MouseButtonEvent.button now behaves like KeyEvent.keyname does
 * event.App class added
 * drawing methods no longer have a default for the character parameter
 * KeyEvent.ctrl is now KeyEvent.control
