.. contents::
   :backlinks: top

==============
 Installation
==============
The latest Windows installers can be found on PyPI, you'll need to install the
latest version of these two packages:

* libtcod-cffi: https://pypi.python.org/pypi/libtcod-cffi
* python-tdl: https://pypi.python.org/pypi/tdl

You might get errors during the installation of libtcod-cffi such as
"ImportError: No module named 'cffi.setuptools_ext'"
This will happen if your cffi module is out of date.

The recommended way to install is by using pip, but be sure to update your cffi
module first.  Use the following commands:

    pip install -U cffi

    pip install tdl

You could install manually, but you'd also have to install libtcod-cffi as well
which is a binary module.  This requires your Python installation to be set up
with a compiler.

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
* Python 2.7+ or 3.x
* 32 bit Windows, 32/64 bit Linux, or Mac OS/X (64 bit architecture)
* libtcod-cffi:  found at https://pypi.python.org/pypi/libtcod-cffi

=========
 License
=========
python-tdl is distributed under the Simplified 2-clause FreeBSD license.
Read LICENSE.txt for more details.
