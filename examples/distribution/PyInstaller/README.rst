PyInstaller Example
===================

First, install the packages: ``tdl`` and ``PyInstaller``.

On Windows you must also install the ``pywin32`` package
(named ``pypiwin32`` if you're using pip install.)

Next, download the `hook-tcod.py` and `hook-tdl.py` files from this repository.
Give PyInstaller the location of these files with the `--additional-hooks-dir`
argument.

`hook-tcod.py` is always needed.  `hook-tdl.py` only installs the default
font used by the tdl module and is optional if a custom font is used.

Then run the PyInstaller script with this command::

    PyInstaller hello_world.py --additional-hooks-dir=.

The finished build will be placed at ``dist/hello_world``. You should see references to the `hook-tdl` in the output.

You can also build to one file with the command::

    PyInstaller hello_world.py --additional-hooks-dir=. -F

The PyInstaller manual can be found at: https://pythonhosted.org/PyInstaller/
