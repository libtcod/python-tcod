#!/usr/bin/env python

import os.path
import subprocess
from distutils.core import setup

setup(name='tdl',
      version='1.1.0',
      author='Kyle Stewart',
      author_email='4B796C65+pythonTDL@gmail.com',
      description='Simple graphical library for making a roguelike or other tile-based video game.',
      long_description="""python-tdl is a ctypes port of "libtcod".

      The library is used for displaying tilesets (ascii or graphical) in true color.
      """,
      url='http://code.google.com/p/python-tdl/',
      download_url='http://code.google.com/p/python-tdl/downloads/list',
      packages=['tdl'],
      package_data={'tdl': ['lib/*.txt', '*.bmp', '*.png', 'lib/win32/*',
                            'lib/darwin/*.dylib', 'lib/linux*/*']},
      classifiers=['Development Status :: 5 - Production/Stable',
                   'Programming Language :: Python',
                   'Environment :: Win32 (MS Windows)',
                   'Environment :: MacOS X',
                   'Environment :: X11 Applications',
                   'Natural Language :: English',
                   'Intended Audience :: Developers',
                   'Topic :: Games/Entertainment',
                   'Topic :: Multimedia :: Graphics :: Presentation',
                   'Topic :: Software Development :: Libraries :: Python Modules',
                   'License :: OSI Approved :: BSD License',
                   'Natural Language :: English',
                   'Operating System :: POSIX',
                   'Operating System :: MacOS',
                   'Operating System :: Microsoft :: Windows',
                   'Programming Language :: Python :: 2.6',
                   'Programming Language :: Python :: 2.7',
                   'Programming Language :: Python :: 3',
                   'Programming Language :: Python :: Implementation :: PyPy',
                   ],
      keywords = 'roguelike roguelikes console text curses doryen ascii libtcod',
      platforms = ['Windows', 'Mac OS X', 'Linux'],
      license = 'New BSD License'
      )
