==============
 Installation
==============
The latest Windows installer can be found on PyPI: https://pypi.python.org/pypi/tdl

If it's available you can use pip instead by running the command:

    pip install tdl

This module can also be manually installed by going into the "setup.py" directory and running the command:

    python setup.py install

This will require that your Python installation can compile binaries.

=======
 About
=======
TDL is a port of the C library libtcod in an attempt to make it more "Pythonic"

The library can be used for displaying tilesets (ANSI, Unicode, or graphical) in true color.

It also provides functionality to compute path-finding and field of view.

python-tdl is hosted on GitHub: https://github.com/HexDecimal/python-tdl

Online Documentation: http://pythonhosted.org/tdl/

Issue Tracker: https://github.com/HexDecimal/python-tdl/issues

python-tdl is a cffi port of "libtcod".  You can find more about libtcod at http://roguecentral.org/doryen/libtcod/

==============
 Requirements
==============
* Python 2.6+ or 3.x
* 32 bit Windows, 32/64 bit Linux, or Mac OS/X (64 bit architecture)
* An up-to-date version of the Python module: cffi

=========
 License
=========
python-tdl is distributed under the FreeBSD license, same as libtcod.  Read LICENSE.txt for more details.
