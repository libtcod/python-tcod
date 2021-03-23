#!/usr/bin/env python3
import sys

from cx_Freeze import Executable, setup

# cx_Freeze options, see documentation:
# https://cx-freeze.readthedocs.io/en/latest/distutils.html#build-exe
build_exe_options = {
    "packages": [],
    "excludes": [
        "numpy.core.tests",
        "numpy.distutils",
        "numpy.doc",
        "numpy.f2py.tests",
        "numpy.lib.tests",
        "numpy.ma.tests",
        "numpy.ma.testutils",
        "numpy.matrixlib.tests",
        "numpy.polynomial.tests",
        "numpy.random.tests",
        "numpy.testing",
        "numpy.tests",
        "numpy.typing.tests",
        "distutils",
        "setuptools",
        "msilib",
        "test",
        "tkinter",
        "unittest",
    ],
    "include_files": ["data"],  # Bundle the data directory.
    "optimize": 1,  # Enable release mode.
    "include_msvcr": False,
}

# Hide the terminal on Windows apps.
base = None
if sys.platform == "win32":
    base = "Win32GUI"

setup(
    name="tcod cx_Freeze example",
    options={"build_exe": build_exe_options},
    executables=[
        # cx_Freeze Executable options:
        # https://cx-freeze.readthedocs.io/en/latest/distutils.html#cx-freeze-executable
        Executable(
            script="main.py",
            base=base,
            target_name="start",  # Name of the target executable.
        )
    ],
)
