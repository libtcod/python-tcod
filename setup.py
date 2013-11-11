#!/usr/bin/env python

try:
    # use setuptools or distribute if available
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(name='tdl',
      version='1.1.5',
      author='Kyle Stewart',
      author_email='4B796C65+pythonTDL@gmail.com',
      description='Pythonic port of rogue-like library libtcod.',
      long_description="""python-tdl is a ctypes port of "libtcod".

      The library is used for displaying tilesets (ANSI, Unicode, or graphical) in true color.
      
      It also provides functionality to compute path-finding and field of view.
      
      python-tdl on GoogleCode: http://code.google.com/p/python-tdl/
      Online Documentation: http://pythonhosted.org/tdl/
      Issue Tracker: http://code.google.com/p/python-tdl/issues/list
      
      libtcod: http://doryen.eptalys.net/libtcod/
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
      keywords = 'portable rogue-like rogue-likes text ctypes ASCII ANSI Unicode libtcod fov pathfinsing',
      platforms = ['Windows', 'Mac OS X', 'Linux'],
      license = 'New BSD License'
      )
