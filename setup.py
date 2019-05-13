#!/usr/bin/env python3

import sys

from setuptools import setup

from subprocess import check_output
import platform
import warnings

from distutils.unixccompiler import UnixCCompiler

C_STANDARD = "-std=c99"
CPP_STANDARD = "-std=c++14"

old_compile = UnixCCompiler._compile


def new_compile(self, obj, src, ext, cc_args, extra_postargs, pp_opts):
    cc_args = list(cc_args)
    if UnixCCompiler.language_map[ext] == "c":
        cc_args.append(C_STANDARD)
    elif UnixCCompiler.language_map[ext] == "c++":
        cc_args.append(CPP_STANDARD)
    return old_compile(self, obj, src, ext, cc_args, extra_postargs, pp_opts)


UnixCCompiler._compile = new_compile


def get_version():
    """Get the current version from a git tag, or by reading tcod/version.py"""
    try:
        tag = check_output(
            ["git", "describe", "--abbrev=0"], universal_newlines=True
        ).strip()
        assert not tag.startswith("v")
        version = tag

        # add .devNN if needed
        log = check_output(
            ["git", "log", "%s..HEAD" % tag, "--oneline"],
            universal_newlines=True,
        )
        commits_since_tag = log.count("\n")
        if commits_since_tag:
            version += ".dev%i" % commits_since_tag

        # update tcod/version.py
        open("tcod/version.py", "w").write('__version__ = "%s"\n' % version)
        return version
    except:
        try:
            exec(open("tcod/version.py").read(), globals())
            return __version__
        except FileNotFoundError:
            warnings.warn(
                "Unknown version: "
                "Not in a Git repository and not from a sdist bundle or wheel."
            )
            return "0.0.0"


is_pypy = platform.python_implementation() == "PyPy"


def get_package_data():
    """get data files which will be included in the main tcod/ directory"""
    BITSIZE, LINKAGE = platform.architecture()
    files = [
        "py.typed",
        "lib/LIBTCOD-CREDITS.txt",
        "lib/LIBTCOD-LICENSE.txt",
        "lib/README-SDL.txt",
    ]
    if "win32" in sys.platform:
        if BITSIZE == "32bit":
            files += ["x86/SDL2.dll"]
        else:
            files += ["x64/SDL2.dll"]
    if sys.platform == "darwin":
        files += ["SDL2.framework/Versions/A/SDL2"]
    return files


def get_long_description():
    """Return this projects description."""
    with open("README.rst", "r") as f:
        readme = f.read()
    with open("CHANGELOG.rst", "r") as f:
        changelog = f.read()
        changelog = changelog.replace("\nUnreleased\n------------------", "")
    return "\n".join([readme, changelog])


if sys.version_info < (3, 5):
    error = """
    This version of python-tcod only supports Python 3.5 and above.
    The last version supporting Python 2.7/3.4 was 'tcod==6.0.7'.

    The end-of-life for Python 2 is the year 2020.
    https://pythonclock.org/

    Python {py} detected.
    """.format(
        py=".".join([str(v) for v in sys.version_info[:3]])
    )

    print(error)
    sys.exit(1)

needs_pytest = {"pytest", "test", "ptr"}.intersection(sys.argv)
pytest_runner = ["pytest-runner"] if needs_pytest else []

setup(
    name="tcod",
    version=get_version(),
    author="Kyle Stewart",
    author_email="4B796C65+tdl@gmail.com",
    description="Pythonic cffi port of libtcod.",
    long_description=get_long_description(),
    url="https://github.com/libtcod/python-tcod",
    py_modules=["libtcodpy"],
    packages=["tdl", "tcod"],
    package_data={"tdl": ["*.png"], "tcod": get_package_data()},
    python_requires=">=3.5",
    install_requires=[
        "cffi>=1.12.0,<2",
        "numpy>=1.10,<2" if not is_pypy else "",
    ],
    cffi_modules=["build_libtcod.py:ffi"],
    setup_requires=["cffi>=1.8.1,<2", "pycparser>=2.14,<3"] + pytest_runner,
    tests_require=["pytest", "pytest-cov", "pytest-benchmark"],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Environment :: Win32 (MS Windows)",
        "Environment :: MacOS X",
        "Environment :: X11 Applications",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Operating System :: POSIX",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
        "Topic :: Games/Entertainment",
        "Topic :: Multimedia :: Graphics",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="roguelike cffi Unicode libtcod fov heightmap namegen",
    platforms=["Windows", "MacOS", "Linux"],
    license="Simplified BSD License",
)
