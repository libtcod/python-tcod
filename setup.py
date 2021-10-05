#!/usr/bin/env python3

import os
import pathlib
import platform
import re
import subprocess
import sys
import warnings
from typing import List

from setuptools import setup  # type: ignore

SDL_VERSION_NEEDED = (2, 0, 5)

PATH = pathlib.Path(__file__).parent  # setup.py current directory


def get_version() -> str:
    """Get the current version from a git tag, or by reading tcod/version.py"""
    if (PATH / ".git").exists():
        tag = subprocess.check_output(["git", "describe", "--abbrev=0"], universal_newlines=True).strip()
        assert not tag.startswith("v")
        version = tag

        # add .devNN if needed
        log = subprocess.check_output(["git", "log", f"{tag}..HEAD", "--oneline"], universal_newlines=True)
        commits_since_tag = log.count("\n")
        if commits_since_tag:
            version += ".dev%i" % commits_since_tag

        # update tcod/version.py
        with open(PATH / "tcod/version.py", "w", encoding="utf-8") as version_file:
            version_file.write(f'__version__ = "{version}"\n')
        return version
    else:  # Not a Git respotitory.
        try:
            with open(PATH / "tcod/version.py", encoding="utf-8") as version_file:
                match = re.match(r'__version__ = "(\S+)"', version_file.read())
            assert match
            return match.groups()[0]
        except FileNotFoundError:
            warnings.warn("Unknown version: Not in a Git repository and not from a sdist bundle or wheel.")
        return "0.0.0"


is_pypy = platform.python_implementation() == "PyPy"


def get_package_data() -> List[str]:
    """get data files which will be included in the main tcod/ directory"""
    BITSIZE, _ = platform.architecture()
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


def get_long_description() -> str:
    """Return this projects description."""
    with open(PATH / "README.rst", "r", encoding="utf-8") as readme_file:
        return readme_file.read()


def check_sdl_version() -> None:
    """Check the local SDL version on Linux distributions."""
    if not sys.platform.startswith("linux"):
        return
    needed_version = "%i.%i.%i" % SDL_VERSION_NEEDED
    try:
        sdl_version_str = subprocess.check_output(["sdl2-config", "--version"], universal_newlines=True).strip()
    except FileNotFoundError:
        raise RuntimeError(
            "libsdl2-dev or equivalent must be installed on your system"
            " and must be at least version %s."
            "\nsdl2-config must be on PATH." % (needed_version,)
        )
    print("Found SDL %s." % (sdl_version_str,))
    sdl_version = tuple(int(s) for s in sdl_version_str.split("."))
    if sdl_version < SDL_VERSION_NEEDED:
        raise RuntimeError("SDL version must be at least %s, (found %s)" % (needed_version, sdl_version_str))


if not os.path.exists(PATH / "libtcod/src"):
    print("Libtcod submodule is uninitialized.")
    print("Did you forget to run 'git submodule update --init'?")
    sys.exit(1)

check_sdl_version()

needs_pytest = {"pytest", "test", "ptr"}.intersection(sys.argv)
pytest_runner = ["pytest-runner"] if needs_pytest else []

setup(
    name="tcod",
    version=get_version(),
    author="Kyle Benesch",
    author_email="4b796c65+tcod@gmail.com",
    description="The official Python port of libtcod.",
    long_description=get_long_description(),
    url="https://github.com/libtcod/python-tcod",
    project_urls={
        "Documentation": "https://python-tcod.readthedocs.io",
        "Changelog": "https://github.com/libtcod/python-tcod/blob/develop/CHANGELOG.rst",
        "Source": "https://github.com/libtcod/python-tcod",
        "Tracker": "https://github.com/libtcod/python-tcod/issues",
        "Forum": "https://github.com/libtcod/python-tcod/discussions",
    },
    py_modules=["libtcodpy"],
    packages=["tcod", "tcod.__pyinstaller"],
    package_data={"tcod": get_package_data()},
    python_requires=">=3.7",
    setup_requires=[
        *pytest_runner,
        "cffi~=1.13",
        "pycparser>=2.14",
    ],
    install_requires=[
        "cffi~=1.13",  # Also required by pyproject.toml.
        "numpy>=1.20.3" if not is_pypy else "",
        "typing_extensions",
    ],
    cffi_modules=["build_libtcod.py:ffi"],
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
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
        "Topic :: Games/Entertainment",
        "Topic :: Multimedia :: Graphics",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="roguelike cffi Unicode libtcod field-of-view pathfinding",
    platforms=["Windows", "MacOS", "Linux"],
    license="Simplified BSD License",
)
