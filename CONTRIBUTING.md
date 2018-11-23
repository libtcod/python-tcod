
# Code style

New and refactored Python code should follow the
[PEP 8](https://www.python.org/dev/peps/pep-0008/) guidelines.

It's recommended to use an editor supporting
[EditorConfig](https://editorconfig.org/).

# Building python-tcod

To work with the tcod source, your environment must be set up to build
Python C extensions.  You'll also need `cpp` installed for
use with pycparser.

## Windows

- Install [Microsoft Visual Studio](https://www.visualstudio.com/vs/community/)
-- When asked, choose to install the Python development tools.
- Install [MinGW](http://www.mingw.org/) or [MSYS2](https://www.msys2.org/).
-- The MinGW installer is [here](https://sourceforge.net/projects/mingw/files/latest/download).
-- Add the binary folder (default MinGW folder is `C:\MinGW\bin`) to your user
   environment PATH variable.
- Open a command prompt in the cloned git directory.
- Make sure the libtcod submodule is downloaded with this command:
  `git submodule update --init`
- Install an editable version of tcod with this command:
  `py -m pip install --editable . --verbose`

## MacOS

- Open a command prompt in the cloned git directory.
- Install the Xcode command line tools with this command:
  `xcode-select --install`
- Make sure the libtcod submodule is downloaded with this command:
  `git submodule update --init`
- Install an editable version of tcod with this command:
  `pip install --editable . --verbose`

## Linux

- Open a command prompt in the cloned git directory.
- Assuming a Debian based distribution of Linux.
  Install tcod's dependancies with this command:
  `sudo apt install gcc python-dev libsdl2-dev libffi-dev libomp-dev`
- Make sure the libtcod submodule is downloaded with this command:
  `git submodule update --init`
- Install an editable version of tdl with this command:
  `pip install --editable . --verbose`
