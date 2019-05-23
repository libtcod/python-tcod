PyInstaller Example
===================

First, install the packages: ``tcod`` and ``PyInstaller``.

On Windows you must also install the ``pywin32`` package
(named ``pypiwin32`` if you're using pip install.)

Then run the PyInstaller script with this command::

    PyInstaller hello_world.py --add-data "terminal8x8_gs_ro.png;."

The finished build will be placed in the ``dist/`` directory.

You can also build to one executable file using the following command::

    PyInstaller hello_world.py --add-data "terminal8x8_gs_ro.png;." --onefile

The PyInstaller manual can be found at: https://pythonhosted.org/PyInstaller/
