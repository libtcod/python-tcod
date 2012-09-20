#!/usr/bin/env python

import os.path
import subprocess
from distutils.core import setup

def getVersion():
    """getting the latest revision number from svn is a pain
    when setup.py is run this function is called and sets up the tdl/VERSION file
    when run from an sdist build the svn data isn't found and uses the stored version instead
    """
    REVISION = None
    if os.path.exists('.svn'): # if .svn/ doesn't even exist, don't bother running svnversion
        svnversion = subprocess.Popen('svnversion -n', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        REVISION = svnversion.communicate()[0] # get stdout
        
    if not REVISION or not any([c.isdigit() for c in REVISION]):
        # no numbers so assume an error
        # but likely a real user install, so get version from file
        VERSION = open('tdl/VERSION', 'r').read()
        return VERSION
    # building on the svn, save version in a file for user installs
    if b':' in REVISION:
        REVISION = REVISION.split(b':')[-1] # take "latest" revision, I think
    REVISION = b''.join((c for c in REVISION if c.isdigit())) # remove letters
    VERSION = '1.0r%s' % REVISION
    open('tdl/VERSION', 'w').write(VERSION)
    return VERSION

VERSION = getVersion()
print('TDL version is %s' % VERSION)

setup(name='tdl',
      version=VERSION,
      author='Kyle Stewart',
      author_email='4B796C65+tdl@gmail.com',
      description='Graphical and utility library for making a roguelike or other tile-based video games',
      long_description="""
tdl is a ctypes port of "libtcod".

The library is used for displaying tilesets (ascii or graphical) in true color.
""",
      url='http://4b796c65.googlepages.com/tdl',
      download_url='https://launchpad.net/rlu/+download',
      packages=['tdl'],
      package_data={'tdl': ['lib/*.txt', '*.bmp', '*.png', 'lib/win32/*',
                            'lib/darwin/*.dylib', 'lib/linux*/*',
                            'lib/darwin/SDL.framework/Versions/A/SDL']},
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
      keywords = 'roguelike roguelikes console text curses doryen ascii libtcod',
      platforms = ['Windows', 'Mac OS X', 'Linux'],
      license = 'New BSD License'
      )
