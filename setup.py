#!/usr/bin/env python

try:
    # use setuptools or distribute if available
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(name='tdl',
      version='1.1.3',
      author='Kyle Stewart',
      author_email='4B796C65+pythonTDL@gmail.com',
      description='Simple graphical library for making a rogue-like or other tile-based video game.',
      long_description="""python-tdl is a ctypes port of "libtcod".

      The library is used for displaying tilesets (ansi, unicode, or graphical) in true color.
      """,
      url='http://code.google.com/p/python-tdl/',
      download_url='http://code.google.com/p/python-tdl/downloads/list',
      packages=['tdl'],
      package_data={'tdl': ['lib/*.txt', '*.bmp', '*.png', 'lib/win32/*',
                            'lib/darwin/*.dylib', 'lib/linux*/*']},
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
                   'Programming Language :: Python :: Implementation :: PyPy',
                   'Topic :: Games/Entertainment',
                   'Topic :: Multimedia :: Graphics',
                   'Topic :: Software Development :: Libraries :: Python Modules',
                   ],
      keywords = 'portable rogue-like rogue-likes text ctypes ASCII ANSI Unicode libtcod',
      platforms = ['Windows', 'Mac OS X', 'Linux'],
      license = 'New BSD License'
      )
