#!/usr/bin/env python

import sys
import os

import platform
import subprocess

from setuptools import setup

VERSION_PATH = 'src/version.txt'

def update_and_get_version():
    '''Update tcod/version.txt to the version number provided by git, then
    return the result, regardless if the update was successful.

    tcod/version.txt is placed in sdist, but is not in the master branch
    '''
    try:
        version = subprocess.check_output(['git', 'describe'], stderr=sys.stderr)
        version = version[:-1] # remove newline
        version = version.replace(b'v', b'') # remove the v from old tags
        tag, commit, obj = version.split(b'-')
        # using anything other than the tag with the current setup is not
        # useful at this moment
        version  = tag
        with open(VERSION_PATH, 'wb') as f:
            f.write(version)
    except subprocess.CalledProcessError:
        # when run from an sdist version.txt is already up-to-date
        if not os.path.exists(VERSION_PATH):
            raise
    with open(VERSION_PATH, 'r') as f:
        return f.read()

def _get_lib_path_crossplatform():
    '''Locate the right DLL path for this OS'''
    bits, linkage = platform.architecture()
    if 'win32' in sys.platform:
        return ['lib/win32/SDL.dll',
                'lib/win32/zlib1.dll',
                'lib/win32/libtcod-VS.dll']
    elif 'linux' in sys.platform:
        if bits == '32bit':
            return ['lib/linux32/*.so']
        elif bits == '64bit':
            return ['lib/linux64/*.so']
    elif 'darwin' in sys.platform:
        return ['lib/darwin/*.dylib']
    raise ImportError('Operating system "%s" has no supported dynamic link libarary. (%s, %s)' % (sys.platform, bits, linkage))


setup(
    name='libtcod-cffi',
    version=update_and_get_version(),
    author='Kyle Stewart',
    author_email='4B796C65+pythonTDL@gmail.com',
    description='A Python cffi port of libtcod.',
    long_description='\n'.join([open('README.rst', 'r').read(),
                                open('CHANGELOG.rst', 'r').read()]),
    url='https://github.com/HexDecimal/libtcod-cffi',
    download_url='https://pypi.python.org/pypi/libtcod-cffi',
    package_dir={'tcod': 'src'},
    package_data={'tcod':
    # only add the libraries needed for the current build platform
        ['*.txt', '*.rst', 'lib/*.txt'] + _get_lib_path_crossplatform()},
    setup_requires=["cffi>=1.1.0,<2"],
    cffi_modules=["build_libtcod.py:ffi"],
    install_requires=["cffi>=1.1.0,<2",
                      "setuptools>=17.1.0"],
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
