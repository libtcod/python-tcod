#!/usr/bin/env python

REVISION = int(''.join(c for c in "$Revision$" if c.isdigit()))
VERSION = '1.0r%i' % REVISION

try:
    from setuptools import setup
except ImportError:
    print('This module will use setuptools if available.')
    from distutils.core import setup

setup(name='tdl',
      version=VERSION,
      author='Kyle Stewart',
      author_email='4B796C65+tdl@gmail.com',
      description='Graphical and utility library for making a roguelike or other tile-based video games',
      long_description="""
tdl is a ctypes port of The Doryen Library.

The library is used for displaying tilesets (ascii or graphical) in true color.
""",
      url='http://4b796c65.googlepages.com/tdl',
      download_url='https://launchpad.net/rlu/+download',
      packages=['tdl'],
      #package_dir={'tdl': 'tdl'},
      package_data={'tdl': ['lib/*.txt', '*.bmp', '*.png', 'lib/linux*/*.so', 'lib/win32/*.dll', 'lib/darwin/*']},
      include_package_data=True,
      install_requires=['setuptools'],
      classifiers=['Development Status :: 4 - Beta',
                   'Programming Language :: Python',
                   'Environment :: Win32 (MS Windows)',
                   'Environment :: MacOS X',
                   'Environment :: X11 Applications',
                   'Natural Language :: English',
                   'Intended Audience :: Developers',
                   'Topic :: Games/Entertainment',
                   'Topic :: Multimedia :: Graphics',
                   'License :: OSI Approved :: zlib/libpng License',
                   'Operating System :: OS Independent',
                   'Operating System :: POSIX',
                   'Operating System :: MacOS',
                   'Operating System :: Microsoft :: Windows',
                   'Topic :: Software Development :: Libraries :: Python Modules',
                   'Programming Language :: Python :: 2.6',
                   'Programming Language :: Python :: 2.7',
                   'Programming Language :: Python :: 3',
                   'Programming Language :: Python :: 3.0',
                   'Programming Language :: Python :: 3.1',
                   ],
      keywords = 'roguelike roguelikes console text curses doryen ascii',
      zip_safe=True,
      )
