.. contents::
   :backlinks: top

=======
 About
=======
This is a Python CFFI port of libtcod, this is an offshoot of the python-tdl project and has been made into it's own package.
Both projects are still developed together, and this module provides functions specifically to be used by python-tdl.

This library is hosted on GitHub: https://github.com/HexDecimal/libtcod-cffi

Any issues you have with this module can be reported at the GitHub issue tracker: https://github.com/HexDecimal/libtcod-cffi/issues

python-tdl is a port of libtcod made to be "Pythonic", you can find it here: https://github.com/HexDecimal/python-tdl

And libtcod is here: http://roguecentral.org/doryen/libtcod/

=================================
 Installation (Windows, Mac OSX)
=================================
The recommended way to install is by using pip, make sure pip is up-to-date
otherwise it won't find the wheel and will attempt to build from source.

With Python installed, run the following commands to install libtcod-cffi::

    python -m pip install --upgrade pip
    python -m pip install libtcod-cffi

======================
 Installation (Linux)
======================
There are no libtcod-cffi wheels for Linux, you can still use pip to install
libtcod-cffi but will need to have the proper build enviroment set up first.

Install the needed dev packages, update the cffi module, then install via pip.
Assuming you are using a debian like distribution you can use the following
commands to do this::

    apt-get install libsdl1.2 libsdl1.2-dev zlib-dev libffi-dev python-dev mesa-common-dev
    python -m pip install --upgrade cffi<2 cffi>=1.1
    python -m pip install libtcod-cffi

The Python cffi module must be 1.1 or later, otherwise you will recieve the
following error:: "ImportError: No module named 'cffi.setuptools_ext'"

=======
 Usage
=======
This module was designed to be backwards compatible with the libtcod.py script that was distributed with libtcod.
If you had code that runs on the original module you can use this library as a drop-in replacement like this::

    import tcod as libtcod

Guides and Tutorials for the original library should also work with this one.
When possible, using PyPy will give the best performance, and is highly reccomended.

==============
 Requirements
==============
* Python 2.7+, Python 3.3+, or PyPy
* Windows, Linux, or Mac OSX
* Supports all 32-bit and 64-bit platforms
* Running on Linux requires the following packages: libsdl1.2
* Installing on Linux or form source will require the following packages:
  python-cffi 1.1+
* An up-to-date version of python-cffi: https://pypi.python.org/pypi/cffi
* Linux will require libsdl1.2, libsdl1.2-dev, libpng, zlib, libffi-dev, and python-dev packages

=========
 License
=========
libtcod-cffi is distributed under the Simplified 2-clause FreeBSD license.
Read LICENSE.txt, and the tcod/lib/README's for more details.
