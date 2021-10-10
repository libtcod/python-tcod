.. _installation:

Installation
============
Once python-tcod is installed, you'll be able to import the `tcod` module.

The latest version of Python 3 is recommended for a normal install.
Python 2 can not be used with the latest versions of python-tcod.
These instructions include installing Python if you don't have it yet.

Windows
-------
`First install the latest recent version of Python 3.
<https://www.python.org/downloads/>`_

.. important::
    Make sure ``Add Python to environment variables`` is checked in the installer.
    Otherwise Python will not be added to the Windows ``PATH``.
    If you forgot to do this then you can reopen the installer and *modify* your installation.

If you don't already have it, then install the latest
`Microsoft Visual C++ Redistributable
<https://support.microsoft.com/en-ca/help/2977003/the-latest-supported-visual-c-downloads>`_.
**vc_redist.x86.exe** for a 32-bit install of Python, or **vc_redist.x64.exe**
for a 64-bit install.  You'll need to keep this in mind when distributing any
libtcod program to end-users.

You should verify your Python install with your terminal.
The terminal you use can be the Windows Command Prompt, PowerShell, GitBash, or similar.
It can not be the Python interpreter (indicated with a ``>>>`` prompt.)
Run the following commands (excluding the ``>``) to verify your Python installation::

    >python -V
    Python 3.10.0

    >pip -V
    pip 21.2.4 from ...\Python310\lib\site-packages\pip (python 3.10)

The above outputs would be the result of Python 3.10 being installed.
**Make sure the mentioned Python versions you get are not different than the latest version you just installed.**

To install python-tcod run the following from a Windows command line::

    >pip install tcod

If Python was installed for all users then you may need to add the ``--user``
flag to pip.

You can then verify that ``tcod`` is importable from the Python interpreter::

    >python

    >>> import tcod

If ``import tcod`` doesn't throw an ``ImportError`` then ``tcod`` has been installed correctly to your system libraries.

Some IDE's such as PyCharm will create a virtual environment which will ignore your system libraries and require tcod to be installed again in that new environment.

MacOS
-----
The latest version of python-tcod only supports MacOS 10.9 (Mavericks) or
later.

`First install a recent version of Python 3.
<https://www.python.org/downloads/>`_

Then to install using pip in a user environment, use the following command::

    python3 -m pip install --user tcod

Linux (Debian-based)
--------------------
On Linux python-tcod will need to be built from source.
You can run this command to download python-tcod's dependencies with apt::

    sudo apt install build-essential python3-dev python3-pip python3-numpy libsdl2-dev libffi-dev

If your GCC version is less than 6.1, or your SDL version is less than 2.0.5,
then you will need to perform a distribution upgrade before continuing.

Once dependencies are resolved you can build and install python-tcod using pip
in a user environment::

    python3 -m pip install --user tcod

PyCharm
-------
PyCharm will often run your project in a virtual environment, hiding any modules
you installed system-wide.  You must install python-tcod inside of the virtual
environment in order for it to be importable in your projects scripts.

By default the bottom bar of PyCharm will have a tab labeled `terminal`.
Open this tab and you should see a prompt with ``(venv)`` on it.
This means your commands will run in the virtual environment of your project.

From this terminal you can install ``tcod`` to the virtual environment with the following command::

    pip install tcod

You can now use ``import tcod``.

If you are working with multiple people or computers or are using a Git repository then it is recommend to pin
the tcod version in a `requirements.txt file <https://pip.pypa.io/en/stable/user_guide/#requirements-files>`_.
PyCharm will automatically update the virtual environment from these files.

Upgrading python-tcod
---------------------
python-tcod is updated often, you can re-run pip with the ``--upgrade`` flag
to ensure you have the latest version, for example::

    python3 -m pip install --upgrade tcod

Upgrading from libtcodpy to python-tcod
---------------------------------------
`libtcodpy` is no longer maintained and using it can make it difficult to
collaborate with developers across multiple operating systems, or to distribute
to those platforms.
New API features are only available on `python-tcod`.

You can recognize a libtcodpy program because it includes this file structure::

    libtcodpy/ (or libtcodpy.py)
    libtcod.dll (or libtcod-mingw.dll)
    SDL2.dll (or SDL.dll)

First make sure your libtcodpy project works in Python 3.  libtcodpy
already supports both 2 and 3 so you don't need to worry about updating it,
but you will need to worry about bit-size.  If you're using a
32-bit version of Python 2 then you'll need to upgrade to a 32-bit version of
Python 3 until libtcodpy can be completely removed.

For Python 3 you'll want the latest version of `tcod`, for Python 2 you'll need
to install ``tcod==6.0.7`` instead, see the Python 2.7 instructions below.

Once you've installed python-tcod you can safely delete the ``libtcodpy/``
folder, the ``libtcodpy.py`` script, and all the DLL files of a libtcodpy
program, python-tcod will seamlessly and immediately take the place of
libtcodpy's API.

From then on anyone can follow the instructions in this guide to install
python-tcod and your project will work for them regardless of their platform.

Distributing
------------
Once your project is finished, it can be distributed using
`PyInstaller <https://www.pyinstaller.org/>`_.

Python 2.7
----------
While it's not recommended, you can still install `python-tcod` on
`Python 2.7`.

`Keep in mind that Python 2's end-of-life has already passed.  You should not be
starting any new projects in Python 2!
<https://www.python.org/doc/sunset-python-2/>`_

Follow the instructions for your platform normally.  When it comes to
install with pip, tell it to get python-tcod version 6::

    python2 -m pip install tcod==6.0.7
