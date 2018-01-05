.. contents::
   :backlinks: top

========
 Status
========
|VersionsBadge| |ImplementationBadge| |LicenseBadge|

|PyPI| |RTD| |Appveyor| |Travis| |Coveralls| |Codecov| |Codacy| |Scrutinizer| |Landscape|

|Requires| |Pyup|

=======
 About
=======
This is a Python cffi_ port of libtcod_.

This library is `hosted on GitHub <https://github.com/HexDecimal/python-tdl>`_.

Any issues you have with this module can be reported at the
`GitHub issue tracker <https://github.com/HexDecimal/python-tdl/issues>`_.

=======
 Usage
=======
This module was designed to be backward compatible with the original libtcodpy
module distributed with libtcod.
If you had code that runs on libtcodpy then you can use this library as a
drop-in replacement::

    import tcod as libtcod

Guides and Tutorials for libtcodpy should work with the tcod module.

The latest documentation can be found
`here <https://python-tdl.readthedocs.io/en/latest/>`_.

==============
 Installation
==============
The recommended way to install is by using pip.  Older versions of pip will
have issues installing tdl, so make sure it's up-to-date.

Windows / MacOS
---------------
To install using pip, use the following command::

    > python -m pip install tdl

If you get the error "ImportError: DLL load failed: The specified module could
not be found." when trying to import tcod/tdl then you may need the latest
`Microsoft Visual C runtime
<https://support.microsoft.com/en-ca/help/2977003/the-latest-supported-visual-c-downloads>`_.

Linux
-----
The easiest method to install tdl on Linux would be from the PPA,
this method will work for the Zesty, Artful, and Bionic versions of Ubuntu::

    $ sudo add-apt-repository ppa:4b796c65/ppa
    $ sudo apt-get update
    $ sudo apt-get install python-tdl python3-tdl

Otherwise tdl will need to be built from source.  Assuming you have Python,
pip, and apt-get, then you'll run these commands to install tdl and its
dependencies to your user environment::

    $ sudo apt-get install gcc python-dev python3-dev libsdl2-dev libffi-dev
    $ pip2 install tdl
    $ pip3 install tdl

==============
 Requirements
==============
* Python 2.7+, Python 3.4+, or PyPy 5.4+
* Windows, Linux, or MacOS.
* On Linux, requires libsdl2 to run.

=========
 License
=========
python-tdl is distributed under the `Simplified 2-clause FreeBSD license
<https://github.com/HexDecimal/python-tdl/blob/master/LICENSE.txt>`_.

.. _LICENSE.txt: https://github.com/HexDecimal/python-tdl/blob/master/LICENSE.txt

.. _python-tdl: https://github.com/HexDecimal/python-tdl/

.. _cffi: https://cffi.readthedocs.io/en/latest/

.. _numpy: https://docs.scipy.org/doc/numpy/user/index.html

.. _libtcod: https://bitbucket.org/libtcod/libtcod/

.. _pip: https://pip.pypa.io/en/stable/installing/

.. |VersionsBadge| image:: https://img.shields.io/pypi/pyversions/tdl.svg?maxAge=2592000
    :target: https://pypi.python.org/pypi/tdl

.. |ImplementationBadge| image:: https://img.shields.io/pypi/implementation/tdl.svg?maxAge=2592000
    :target: https://pypi.python.org/pypi/tdl

.. |LicenseBadge| image:: https://img.shields.io/pypi/l/tdl.svg?maxAge=2592000
    :target: https://github.com/HexDecimal/tdl/blob/master/LICENSE.txt

.. |PyPI| image:: https://img.shields.io/pypi/v/tdl.svg?maxAge=10800
    :target: https://pypi.python.org/pypi/tdl

.. |RTD| image:: https://readthedocs.org/projects/python-tdl/badge/?version=latest
    :target: http://python-tdl.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. |Appveyor| image:: https://ci.appveyor.com/api/projects/status/bb04bpankj0h1cpa/branch/master?svg=true
    :target: https://ci.appveyor.com/project/HexDecimal/python-tdl/branch/master

.. |Travis| image:: https://travis-ci.org/HexDecimal/python-tdl.svg?branch=master
    :target: https://travis-ci.org/HexDecimal/python-tdl

.. |Coveralls| image:: https://coveralls.io/repos/github/HexDecimal/python-tdl/badge.svg?branch=master
    :target: https://coveralls.io/github/HexDecimal/python-tdl?branch=master

.. |Codecov| image:: https://codecov.io/gh/HexDecimal/python-tdl/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/HexDecimal/python-tdl

.. |Issues| image:: https://img.shields.io/github/issues/HexDecimal/python-tdl.svg?maxAge=3600
    :target: https://github.com/HexDecimal/python-tdl/issues

.. |Codacy| image:: https://img.shields.io/codacy/grade/6f3d153f1ccc435ca592633e4c35d9f5.svg?maxAge=10800
    :target: https://www.codacy.com/app/4b796c65-github/python-tdl

.. |Scrutinizer| image:: https://scrutinizer-ci.com/g/HexDecimal/python-tdl/badges/quality-score.png?b=master
    :target: https://scrutinizer-ci.com/g/HexDecimal/python-tdl/

.. |Landscape| image:: https://landscape.io/github/HexDecimal/python-tdl/dev/landscape.svg?style=flat
    :target: https://landscape.io/github/HexDecimal/python-tdl/dev
    :alt: Code Health

.. |Requires| image:: https://requires.io/github/HexDecimal/python-tdl/requirements.svg?branch=master
    :target: https://requires.io/github/HexDecimal/python-tdl/requirements/?branch=master
    :alt: Requirements Status

.. |Pyup| image:: https://pyup.io/repos/github/hexdecimal/python-tdl/shield.svg
     :target: https://pyup.io/repos/github/hexdecimal/python-tdl/
