.. contents::
   :backlinks: top

==================================
 Installation (Windows, Mac OS X)
==================================
The recommended way to install is by using pip, make sure pip is up-to-date,
old versions of pip may have issues installing libtcod-cffi.

To install using pip, use the following commands::

    python -m pip install --upgrade pip
    python -m pip install tdl

======================
 Installation (Linux)
======================
Installing on Linux is similar to on Windows or Mac OS X, but you will have to
meet the dependancies of libtcod-cffi first.

* libtcod-cffi: https://pypi.python.org/pypi/libtcod-cffi

=======
 About
=======
TDL is a port of the C library libtcod in an attempt to make it more "Pythonic"

The library can be used for displaying tilesets (ANSI, Unicode, or graphical) in true color.

It also provides functionality to compute path-finding and field of view.

python-tdl is hosted on GitHub: https://github.com/HexDecimal/python-tdl

Online Documentation: http://pythonhosted.org/tdl/

Issue Tracker: https://github.com/HexDecimal/python-tdl/issues

python-tdl is a cffi port of "libtcod".  You can find more about libtcod at
http://roguecentral.org/doryen/libtcod/

==============
 Requirements
==============
* Python 2.7+, 3.3+, or PyPy
* Windows, Linux, or Mac OS X
* Supports all 32-bit and 64-bit platforms
* libtcod-cffi:  found at https://pypi.python.org/pypi/libtcod-cffi

=========
 License
=========
python-tdl is distributed under the Simplified 2-clause FreeBSD license.
Read LICENSE.txt for more details.
