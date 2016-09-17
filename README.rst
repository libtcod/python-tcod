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

==============
 Installation
==============
The recommended way to install is by using pip.

With Python installed, run the following command to install libtcod-cffi::

    python -m pip install libtcod-cffi

=======
 Usage
=======
This module was designed to be backwards compatible with the libtcod.py script that was distributed with libtcod.
If you had code that runs on the original module you can use this library as a drop-in replacement like this::

    import tcod as libtcod

Guides and Tutorials for the original library should also work with this one.
When possible, using PyPy will give the best performance, and is highly recommended.

==============
 Requirements
==============
* Python 2.7+, Python 3.3+, or PyPy
* Windows, Linux, or Mac OS X
* Running on Linux requires the following packages: libsdl1.2
* Installing on Linux or form source will require the following packages:
  python-cffi 1.1+
* An up-to-date version of python-cffi: https://pypi.python.org/pypi/cffi
* Linux will require the package libsdl1.2 to run, and the packages
  libsdl1.2-dev, libffi-dev, python-dev, and mesa-common-dev to install

=========
 License
=========
libtcod-cffi is distributed under the Simplified 2-clause FreeBSD license.
Read LICENSE.txt, and the tcod/lib/README's for more details.
