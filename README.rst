=======
 About
=======

This is a direct Python CFFI port of libtcod, this is an offshoot of the python-tdl project and has been made into it's own package.
Both projects are still developed together, and this module provide special functions specifically to be used by python-tdl.

You can find python-tdl here: https://github.com/HexDecimal/python-tdl

And libtcod is here: http://roguecentral.org/doryen/libtcod/

==============
 Installation
==============
The latest Windows installer can be found on PyPI: https://pypi.python.org/pypi/libtcod-cffi

If it's available you can use pip instead by running the command:

    pip install libtcod-cffi

This module can also be manually installed by going into the "setup.py" directory and running the command:

    python setup.py install

This will require setuptools which you can find here: https://pypi.python.org/pypi/setuptools
It also requires that your Python installation is set up to compile binaries.

==============
 Requirements
==============
* Python 2.6+ or 3.x
* 32 bit Windows, 32/64 bit Linux, or Mac OS/X (64 bit architecture)
* An up-to-date version of the Python module: cffi

=========
 License
=========
python-tdl is distributed under the Simplified 2-clause FreeBSD license.
Read LICENSE.txt, and the libtcod-cffi/lib README's for more details.
