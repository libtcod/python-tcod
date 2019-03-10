Installation
============
Python 3.5 or above is required.

The recommended way to install is by using pip.  Older versions of pip will
have issues installing tcod, so make sure it's up-to-date.

Windows
-------
`First install a recent version of Python 3.
<https://www.python.org/downloads/>`_
Make sure Python is added to the Windows `PATH`.

If you don't already have it, then install the latest
`Microsoft Visual C++ Redistributable
<https://support.microsoft.com/en-ca/help/2977003/the-latest-supported-visual-c-downloads>`_.
`vc_redist.x86.exe` for a 32-bit install of Python, or `vc_redist.x64.exe` for
a 64-bit install.  You'll need to keep this in mind when distributing any
libtcod program to end-users.

Then to install python-tcod run the following from a Windows command line::

    py -m pip install tcod

MacOS
-----
The latest version of python-tcod only supports MacOS 10.9 (Mavericks) or
later.

`First install a recent version of Python 3.
<https://www.python.org/downloads/>`_

Then to install using pip, use the following command::

    python3 -m pip install tcod

Linux (Debian-based)
--------------------
On Linux python-tcod will need to be built from source.
You can run this command to download python-tcod's dependencies with `apt`::

    sudo apt install g++ python3-dev python3-pip python3-numpy libsdl2-dev libffi-dev libomp5

Then you can build and install python-tcod using `pip` in a user environment::

    python3 -m pip install --user tcod

Upgrading python-tcod
---------------------
`python-tcod` is updated often, you can re-run pip with the ``--upgrade`` flag
to ensure you have the latest version, for example::

    python3 -m pip install --upgrade tcod

Upgrading from libtcodpy to python-tcod
---------------------------------------
`libtcodpy` is no longer maintained and using it can make it difficult to
collaborate with developers across multiple operating systems, or to distribute
to those platforms.
New API features are only available on `python-tcod`.

You can recognise a libtcodpy program because it includes this file structure::

    libtcodpy/
    libtcod.dll
    SDL2.dll

First make sure your `libtcodpy` project works in Python 3.  `libtcodpy`
supports both 2 and 3 so you don't need to worry about having to update
libtcodpy itself, but you will need to worry about bit-size.  If you're using a
32-bit version of Python 2 then you'll need to upgrade to a 32-bit version of
Python 3 until libtcodpy is removed.

Once you've installed python-tcod you can safely delete the ``libtcodpy/``
folder and all DLL files of a libtcodpy program, python-tcod will seamlessly
take the place of `libtcodpy`'s API.

From then on anyone can follow the instructions to install python-tcod and your
project will work for them regardless of their platform or bit-size.

`python-tcod` projects can be distributed using
`PyInstaller <https://www.pyinstaller.org/>`._
