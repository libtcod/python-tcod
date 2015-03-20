#!/usr/bin/env python2

try:
    # use setuptools or distribute if available
    from setuptools import setup
except ImportError:
    from distutils.core import setup

from tdl import __version__

setup(name='tdl',
      version=__version__,
      author='Kyle Stewart',
      author_email='4B796C65+pythonTDL@gmail.com',
      description='Pythonic port of rogue-like library libtcod.',
      long_description=open('README.txt', 'r').read() + open('CHANGELOG.txt', 'r').read(),
      url='https://github.com/HexDecimal/python-tdl',
      download_url='https://pypi.python.org/pypi/tdl',
      packages=['tdl'],
      package_data={'tdl': ['*.txt', 'lib/*.txt', '*.bmp', '*.png', 'lib/win32/*',
                            'lib/darwin/*.dylib', 'lib/linux*/*']},
      #install_requires=['ctypes'], # ctypes requirement causes issues sometimes
      classifiers=['Development Status :: 5 - Production/Stable',
                   'Environment :: Win32 (MS Windows)',
                   'Environment :: MacOS X',
                   'Environment :: X11 Applications',
                   'Intended Audience :: Developers',
                   'License :: OSI Approved :: BSD License',
                   'Natural Language :: English',
                   'Operating System :: POSIX',
                   'Operating System :: MacOS',
                   'Operating System :: Microsoft :: Windows',
                   'Programming Language :: Python :: 2.6',
                   'Programming Language :: Python :: 2.7',
                   'Programming Language :: Python :: 3',
                   'Programming Language :: Python :: 3.0',
                   'Programming Language :: Python :: 3.1',
                   'Programming Language :: Python :: 3.2',
                   'Programming Language :: Python :: 3.3',
                   'Programming Language :: Python :: 3.4',
                   'Programming Language :: Python :: Implementation :: CPython',
                   'Programming Language :: Python :: Implementation :: PyPy',
                   'Topic :: Games/Entertainment',
                   'Topic :: Multimedia :: Graphics',
                   'Topic :: Software Development :: Libraries :: Python Modules',
                   ],
      keywords = 'portable rogue-like rogue-likes text ctypes ASCII ANSI Unicode libtcod fov',
      platforms = ['Windows', 'Mac OS X', 'Linux'],
      license = 'New BSD License'
      )
