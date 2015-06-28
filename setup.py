#!/usr/bin/env python

import sys

import platform
from setuptools import setup


def _get_lib_path_crossplatform():
    '''Locate the right DLL path for this OS'''
    bits, linkage = platform.architecture()
    if 'win32' in sys.platform:
        return 'lib/win32/*.dll'
    elif 'linux' in sys.platform:
        if bits == '32bit':
            return 'lib/linux32/*.so'
        elif bits == '64bit':
            return 'lib/linux64/*.so'
    elif 'darwin' in sys.platform:
        return 'lib/darwin/*.dylib'
    raise ImportError('Operating system "%s" has no supported dynamic link libarary. (%s, %s)' % (sys.platform, bits, linkage))


setup(
    name='libtcod-cffi',
    version=open('tcod/version.txt', 'r').read(),
    author='Kyle Stewart',
    author_email='4B796C65+pythonTDL@gmail.com',
    description='A Python cffi port of libtcod.',
    long_description='\n'.join([open('README.rst', 'r').read(),
                                open('CHANGELOG.rst', 'r').read()]),
    url='https://github.com/HexDecimal/libtcod-cffi',
    download_url='https://pypi.python.org/pypi/libtcod-cffi',
    packages=['tcod'],
    package_data={'tcod': ['*.txt', '*.rst', 'lib/*.txt',
    # only add the libraries for the current build platform
                           _get_lib_path_crossplatform()]},
    setup_requires=["cffi>=1.1.0"],
    cffi_modules=["build_libtcod.py:ffi"],
    install_requires=["cffi>=1.1.0",
                      "setuptools>=17.1.0",
                      "distribute>=0.7.3"], # seems to be needed for Python 2.7
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
               'Programming Language :: Python :: 2.7',
               'Programming Language :: Python :: 3',
               'Programming Language :: Python :: 3.2',
               'Programming Language :: Python :: 3.3',
               'Programming Language :: Python :: 3.4',
               'Programming Language :: Python :: Implementation :: CPython',
               'Programming Language :: Python :: Implementation :: PyPy',
               'Topic :: Games/Entertainment',
               'Topic :: Multimedia :: Graphics',
               'Topic :: Software Development :: Libraries :: Python Modules',
               ],
    keywords = 'roguelike roguelikes cffi ASCII ANSI Unicode libtcod noise fov heightmap namegen',
    platforms = ['Windows', 'Mac OS X', 'Linux'],
    license = 'Simplified BSD License'
    )
