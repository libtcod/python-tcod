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
The latest Windows installer can be found on PyPI: https://pypi.python.org/pypi/libtcod-cffi

If it's available you can use pip instead by running the command::

    pip install libtcod-cffi

This module can also be manually installed by going into the "setup.py" directory and running the command::

    python setup.py install

This will require setuptools which you can find here: https://pypi.python.org/pypi/setuptools
It also requires that your Python installation is set up to compile binaries.

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
* Python 2.7+ or 3.2+
* 32 bit Windows, 32/64 bit Linux, or Mac OS/X (64 bit architecture)
* An up-to-date version of python-cffi: https://pypi.python.org/pypi/cffi
* Linux will require libsdl, libpng, zlib, and python-dev packages

=========
 License
=========
libtcod-cffi is distributed under the Simplified 2-clause FreeBSD license.
Read LICENSE.txt, and the tcod/lib/README's for more details.
