PyInstaller Example
===================

First, install the packages: ``tdl``, ``libtcod-cffi``, and ``PyInstaller``.

On Windows you must also install the ``pywin32`` package
(named ``pypiwin32`` if you're using pip install.)

Then run the PyInstaller script with this command::

    PyInstaller hello_world.py --additional-hooks-dir=.

The finished build will be placed at ``dist/hello_world``.

You can also build to one file with the command::

    PyInstaller hello_world.py --additional-hooks-dir=. -F

The PyInstaller manual can be found at: https://pythonhosted.org/PyInstaller/