.. contents::
   :backlinks: top

==============
 Installation
==============
The recommended way to install is by using pip, make sure pip is up-to-date,
old versions of pip may have issues installing libtcod-cffi.

To install using pip, use the following command::

    python -m pip install tdl

Wheels are missing for PyPy on Mac OS X and Linux, so you'll have to meet the
additional dependencies of libtcod-cffi before running pip.

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
* libtcod-cffi (automatically installed with pip)

=========
 License
=========
python-tdl is distributed under the Simplified 2-clause FreeBSD license.
Read LICENSE.txt for more details.
