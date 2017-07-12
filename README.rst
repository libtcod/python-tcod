.. contents::
   :backlinks: top

========
 Status
========
|VersionsBadge| |ImplementationBadge| |LicenseBadge|

|PyPI| |RTD| |Appveyor| |Travis| |Coveralls| |Codecov| |Codacy| |Scrutinizer|

|Requires| |Pyup|

==============
 Installation
==============
The recommended way to install is by using pip.  Older versions of pip will
have issues installing tdl, so make sure it's up-to-date.

Windows / MacOS
---------------
To install using pip, use the following command::

    > python -m pip install tdl

Linux
-----
On Linux, tdl will need to be built from source.  Assuming you have Python and
pip, you run these commands to install tdl::

    $ sudo apt install gcc libsdl2-dev libffi-dev python-dev libomp-dev
    $ pip install tdl

=======
 About
=======
This is a Python cffi_ port of libtcod_.

This library is `hosted on GitHub <https://github.com/HexDecimal/python-tdl>`_.

Any issues you have with this module can be reported at the
`GitHub issue tracker <https://github.com/HexDecimal/python-tdl/issues>`_.

python-tdl is distributed under the `Simplified 2-clause FreeBSD license
<https://github.com/HexDecimal/python-tdl/blob/master/LICENSE.txt>`_.

=======
 Usage
=======
This module was designed to be backward compatible with the original libtcod
module that is distributed with libtcod.
If you had code that runs on the original module you can use this library as a
drop-in replacement like this::

    import tcod as libtcod

Guides and Tutorials for libtcodpy should work with the tcod module.

The latest documentation can be found
`here <https://python-tdl.readthedocs.io/en/latest/>`_.

==============
 Requirements
==============
* Python 2.7+, Python 3.4+, or PyPy 5.4+
* Windows, Linux, or MacOS.
* Linux requires the libsdl2 package and must be installed from source.

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

.. |Requires| image:: https://requires.io/github/HexDecimal/python-tdl/requirements.svg?branch=master
    :target: https://requires.io/github/HexDecimal/python-tdl/requirements/?branch=master
    :alt: Requirements Status

.. |Pyup| image:: https://pyup.io/repos/github/hexdecimal/python-tdl/shield.svg
     :target: https://pyup.io/repos/github/hexdecimal/python-tdl/
