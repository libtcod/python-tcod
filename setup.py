#!/usr/bin/env python

import sys

from setuptools import setup

import platform

exec(open('tdl/version.py').read()) # get __version__

is_pypy = platform.python_implementation() == 'PyPy'

def get_package_data():
    '''get data files which will be included in the main tcod/ directory'''
    BITSIZE, LINKAGE = platform.architecture()
    files = [
        'lib/LIBTCOD-CREDITS.txt',
        'lib/LIBTCOD-LICENSE.txt',
        'lib/README-SDL.txt'
        ]
    if 'win32' in sys.platform:
        if BITSIZE == '32bit':
            files += ['x86/SDL2.dll']
        else:
            files += ['x64/SDL2.dll']
    if sys.platform == 'darwin':
        files += ['SDL2.framework/Versions/A/SDL2']
    return files

needs_pytest = {'pytest', 'test', 'ptr'}.intersection(sys.argv)
pytest_runner = ['pytest-runner'] if needs_pytest else []

setup(
    name='tdl',
    version=__version__,
    author='Kyle Stewart',
    author_email='4B796C65+tdl@gmail.com',
    description='Pythonic cffi port of libtcod.',
    long_description='\n'.join([open('README.rst', 'r').read(),
                                open('CHANGELOG.rst', 'r').read()]),
    url='https://github.com/HexDecimal/python-tdl',
    download_url='https://pypi.python.org/pypi/tdl',
    packages=['tdl', 'tcod'],
    package_data={
        'tdl': ['*.png'],
        'tcod': get_package_data(),
        },
    install_requires=[
        'cffi>=1.8.1,<2',
        'numpy>=1.10,<2' if not is_pypy else '',
        ],
    cffi_modules=['build_libtcod.py:ffi'],
    setup_requires=[
        'cffi>=1.8.1,<2',
        'pycparser>=2.14,<3',
        ] + pytest_runner,
    tests_require=[
        'pytest',
        'pytest-cov',
        'pytest-benchmark',
        ],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
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
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy',
        'Topic :: Games/Entertainment',
        'Topic :: Multimedia :: Graphics',
        'Topic :: Software Development :: Libraries :: Python Modules',
        ],
    keywords='roguelike cffi Unicode libtcod fov heightmap namegen',
    platforms=[
        'Windows',
        'MacOS',
        'Linux',
        ],
    license='Simplified BSD License',
    )
