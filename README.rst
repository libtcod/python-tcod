.. contents::
   :backlinks: top

========
 Status
========
|VersionsBadge| |ImplementationBadge| |LicenseBadge|

|PyPI| |RTD| |Codecov| |Pyup| |CommitsSinceLastRelease|

=======
 About
=======
This is a Python cffi_ port of libtcod_.

This library is `hosted on GitHub <https://github.com/libtcod/python-tcod>`_.

Any issues you have with this module can be reported at the
`GitHub issue tracker <https://github.com/libtcod/python-tcod/issues>`_.

=======
 Usage
=======
This module was designed to be backward compatible with the original libtcodpy
module distributed with libtcod.
If you had code that runs on libtcodpy then you can use this library as a
drop-in replacement.  This installs a libtcodpy module so you'll only need to
delete the libtcodpy/ folder that's usually bundled in an older libtcodpy
project.

Guides and Tutorials for libtcodpy should work with the tcod module.

The latest documentation can be found here:
https://python-tcod.readthedocs.io/en/latest/

==============
 Installation
==============
Detailed installation instructions are here:
https://python-tcod.readthedocs.io/en/latest/installation.html

For the most part it's just::

    pip3 install tcod

==============
 Requirements
==============
* Python 3.10+
* Windows, Linux, or MacOS X 10.9+.
* On Linux, requires libsdl3

===========
 Changelog
===========

You can find the most recent changelog
`here <https://github.com/libtcod/python-tcod/blob/main/CHANGELOG.md>`_.

=========
 License
=========
python-tcod is distributed under the `Simplified 2-clause FreeBSD license
<https://github.com/HexDecimal/python-tdl/blob/master/LICENSE.txt>`_.

.. _LICENSE.txt: https://github.com/libtcod/python-tcod/blob/master/LICENSE.txt

.. _python-tdl: https://github.com/libtcod/python-tcod/

.. _cffi: https://cffi.readthedocs.io/en/latest/

.. _numpy: https://docs.scipy.org/doc/numpy/user/index.html

.. _libtcod: https://github.com/libtcod/libtcod

.. _pip: https://pip.pypa.io/en/stable/installing/

.. |VersionsBadge| image:: https://img.shields.io/pypi/pyversions/tcod.svg?maxAge=2592000
    :target: https://pypi.python.org/pypi/tcod

.. |ImplementationBadge| image:: https://img.shields.io/pypi/implementation/tcod.svg?maxAge=2592000
    :target: https://pypi.python.org/pypi/tcod

.. |LicenseBadge| image:: https://img.shields.io/pypi/l/tcod.svg?maxAge=2592000
    :target: https://github.com/HexDecimal/tcod/blob/master/LICENSE.txt

.. |PyPI| image:: https://img.shields.io/pypi/v/tcod.svg?maxAge=10800
    :target: https://pypi.python.org/pypi/tcod

.. |RTD| image:: https://readthedocs.org/projects/python-tcod/badge/?version=latest
    :target: http://python-tcod.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. |Codecov| image:: https://codecov.io/gh/libtcod/python-tcod/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/libtcod/python-tcod

.. |Issues| image:: https://img.shields.io/github/issues/libtcod/python-tcod.svg?maxAge=3600
    :target: https://github.com/libtcod/python-tcod/issues

.. |Pyup| image:: https://pyup.io/repos/github/libtcod/python-tcod/shield.svg
    :target: https://pyup.io/repos/github/libtcod/python-tcod/
    :alt: Updates

.. |CommitsSinceLastRelease| image:: https://img.shields.io/github/commits-since/libtcod/python-tcod/latest
    :target: https://github.com/libtcod/python-tcod/blob/main/CHANGELOG.md
