
import sys

from cx_Freeze import setup, Executable

# cx_Freeze options, see documentation.
build_exe_options = {
    'packages': ['cffi'],
    'excludes': [],
    'include_files': ['data'],
    }

# Hide the terminal on Windows apps.
base = None
if sys.platform == "win32":
    base = "Win32GUI"

setup(
    name='tdl cxfreeze example',
    options = {'build_exe': build_exe_options},
    executables = [Executable('hello_world.py', base=base)],
    )
