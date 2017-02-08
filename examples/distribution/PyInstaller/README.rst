PyInstaller Example
===================

First, install the packages: ``tdl``, ``libtcod-cffi``, ``PyInstaller``.

On Windows you must also install the ``pywin32`` package
(named ``pypiwin32`` if you're using pip install.)

Then run the PyInstaller script with this command::

    PyInstaller --additional-hooks-dir=. hello_world.py

The finished build will be placed at ``dist/hello_world``.
    
libtcod-cffi/tdl does not support one-file mode at this time.
