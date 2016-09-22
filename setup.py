#!/usr/bin/env python2

import sys
import os

build_docs = None
try:
    # use setuptools or distribute if available
    from setuptools import setup, Command

    import subprocess

    class build_docs(Command):
        description = "update the documentation using epydoc"
        user_options = []
        def initialize_options(self):
            pass
        def finalize_options(self):
            pass
        def run(self):
            'run a command using a local epydoc script'
            command = [sys.executable,
                       os.path.join(sys.prefix, 'Scripts\epydoc.py'),
                       '--config=docs/epydoc.config']

            subprocess.check_call(command)

except ImportError:
    from distutils.core import setup

setup(
    name='tdl',
    version=open('tdl/version.txt', 'r').read(),
    author='Kyle Stewart',
    author_email='4B796C65+pythonTDL@gmail.com',
    description='Pythonic cffi port of libtcod.',
    long_description='\n'.join([open('README.rst', 'r').read(),
                                open('CHANGELOG.rst', 'r').read()]),
    url='https://github.com/HexDecimal/python-tdl',
    download_url='https://pypi.python.org/pypi/tdl',
    packages=['tdl'],
    package_data={'tdl': ['*.txt', '*.rst', '*.bmp', '*.png']},
    install_requires=["libtcod-cffi>=0.2.8,<2"],
    cmdclass={'build_docs': build_docs},
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
               'Programming Language :: Python :: 3.3',
               'Programming Language :: Python :: 3.4',
               'Programming Language :: Python :: Implementation :: CPython',
               'Programming Language :: Python :: Implementation :: PyPy',
               'Topic :: Games/Entertainment',
               'Topic :: Multimedia :: Graphics',
               'Topic :: Software Development :: Libraries :: Python Modules',
               ],
    keywords = 'rogue-like rogue-likes text cffi ASCII ANSI Unicode libtcod fov',
    platforms = ['Windows', 'Mac OS X', 'Linux'],
    license = 'Simplified BSD License',
    tests_require = ['nose2', 'cov-core'],
    test_suite='nose2.collector.collector',
    )
