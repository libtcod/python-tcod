PyInstaller Example
===================

It's recommended to use a virtual environment to package Python executables.
Use the following guide on how to set one up:
https://docs.python.org/3/tutorial/venv.html

Once the virtual environment is active you should install ``tcod``, ``PyInstaller``, and ``pypiwin32`` if on Windows from the ``requirements.txt`` file:

    pip install -r requirements.txt

Then run PyInstaller on the included Spec file::

    PyInstaller main.spec

The finished build will be placed in the ``dist/`` directory.

You can also build to one file using the following command::

    PyInstaller main.spec --onefile

Single file distributions have performance downsides so it is preferred to distribute a larger program this way.

For `tcod` it is recommended to set the ``PYTHONOPTIMIZE=1`` environment variable before running PyInstaller.  This disables warnings for `tcod`'s deprecated functions and will improve the performance of those functions if you were using them.

The PyInstaller documentation can be found at: https://pythonhosted.org/PyInstaller/
