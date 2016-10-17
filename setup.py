#!/usr/bin/env python

import sys
import os

import platform
import subprocess

from setuptools import setup, Command

with open('tcod/version.txt', 'r') as f:
    version = f.read()

def get_package_data():
    '''get data files which will be included in the main tcod/ directory'''
    BITSIZE, LINKAGE = platform.architecture()
    files = ['version.txt',
             'lib/LIBTCOD-CREDITS.txt',
             'lib/LIBTCOD-LICENSE.txt',
             'lib/README-SDL.txt']
    if 'win32' in sys.platform:
        if BITSIZE == '32bit':
            files += ['x86/SDL2.dll']
        else:
            files += ['x64/SDL2.dll']
    return {'tcod': files}

setup(
    name='libtcod-cffi',
    version=version,
    author='Kyle Stewart',
    author_email='4B796C65+tcod@gmail.com',
    description='A Python cffi port of libtcod.',
    long_description='\n'.join([open('README.rst', 'r').read(),
                                open('CHANGELOG.rst', 'r').read()]),
    url='https://github.com/HexDecimal/libtcod-cffi',
    download_url='https://pypi.python.org/pypi/libtcod-cffi',
    packages=['tcod'],
    package_data=get_package_data(),
    setup_requires=["cffi>=1.8.1,<2", "pycparser>=2.14,<3"],
    cffi_modules=["build_libtcod.py:ffi"],
    install_requires=["cffi>=1.8.1,<2"],
    classifiers=['Development Status :: 5 - Production/Stable',
               'Environment :: Win32 (MS Windows)',
               'Environment :: MacOS X',
               'Environment :: X11 Applications',
               'Intended Audience :: Developers',
               'License :: OSI Approved :: BSD License',
               'Natural Language :: English',
               'Operating System :: POSIX',
               'Operating System :: MacOS :: MacOS X',
               'Operating System :: Microsoft :: Windows',
               'Programming Language :: Python :: 2',
               'Programming Language :: Python :: 2.7',
               'Programming Language :: Python :: 3',
               'Programming Language :: Python :: 3.3',
               'Programming Language :: Python :: 3.4',
               'Programming Language :: Python :: 3.5',
               'Programming Language :: Python :: Implementation :: CPython',
               'Programming Language :: Python :: Implementation :: PyPy',
               'Topic :: Games/Entertainment',
               'Topic :: Multimedia :: Graphics',
               'Topic :: Software Development :: Libraries :: Python Modules',
               ],
    keywords = 'roguelike roguelikes cffi ASCII ANSI Unicode libtcod noise fov heightmap namegen',
    platforms = ['Windows', 'Mac OS X', 'Linux'],
    license = 'Simplified BSD License',
    test_suite='nose2.collector.collector',
    )
