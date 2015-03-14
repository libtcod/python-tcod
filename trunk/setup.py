#!/usr/bin/env python2

import subprocess

try:
    # use setuptools or distribute if available
    from setuptools import setup
except ImportError:
    from distutils.core import setup

base_version = '1.1.6' # base version
def get_version():
    """Probably a much more elegant way to do this.
    
    Update tdl/VERSION.txt with the current svn version, then return the result"""
    try:
        revision = subprocess.check_output('svnversion', universal_newlines=True)[:-1]
    except:
        revision = None

    if revision:
        if ':' in revision:
            revision = revision.split(':')[-1]
        if 'M' in revision:
            revision = str(int(revision[:-1]) + 1)
        file = open('tdl/VERSION.txt', 'w')
        file.write(base_version + 'r' + revision)
        file.close()
        print('revision %s' % revision)
        
    file = open('tdl/VERSION.txt', 'r')
    version = file.read()
    file.close()
    return version
    
setup(name='tdl',
      version=get_version(),
      author='Kyle Stewart',
      author_email='4B796C65+pythonTDL@gmail.com',
      description='Pythonic port of rogue-like library libtcod.',
      long_description=open('README.txt', 'r').read() + open('CHANGELOG.txt', 'r').read(),
      url='http://code.google.com/p/python-tdl/',
      download_url='https://pypi.python.org/pypi/tdl',
      packages=['tdl'],
      package_data={'tdl': ['*.txt', 'lib/*.txt', '*.bmp', '*.png', 'lib/win32/*',
                            'lib/darwin/*.dylib', 'lib/linux*/*']},
      #install_requires=['ctypes'], # ctypes requirement causes issues sometimes
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
                   'Programming Language :: Python :: 3.0',
                   'Programming Language :: Python :: 3.1',
                   'Programming Language :: Python :: 3.2',
                   'Programming Language :: Python :: 3.3',
                   'Programming Language :: Python :: Implementation :: CPython',
                   'Programming Language :: Python :: Implementation :: PyPy',
                   'Topic :: Games/Entertainment',
                   'Topic :: Multimedia :: Graphics',
                   'Topic :: Software Development :: Libraries :: Python Modules',
                   ],
      keywords = 'portable rogue-like rogue-likes text ctypes ASCII ANSI Unicode libtcod fov',
      platforms = ['Windows', 'Mac OS X', 'Linux'],
      license = 'New BSD License'
      )
