


# Building tdl

To work with the tdl source, your environment must be set up to build
Python C extensions.  You'll also need `cpp` installed for
use with pycparser.

## Windows

- Install [Microsoft Visual Studio](https://www.visualstudio.com/vs/community/)
-- When asked, choose to install the Python development tools.
- Install [MinGW](http://www.mingw.org/).
-- Installer is [here](https://sourceforge.net/projects/mingw/files/latest/download).
-- Add the binary folder (default folder is `C:\MinGW\bin`) to your user
   environment PATH variable.
- Open a command prompt in the cloned git directory.
- Make sure the libtcod submodule is downloaded with this command:
  `git submodule update --init`
- Install an editable version of tdl with this command:
  `py -m pip install --editable . --verbose`

## MacOS

- Open a command prompt in the cloned git directory.
- Install the Xcode command line tools with this command:
  `xcode-select --install`
- Make sure the libtcod submodule is downloaded with this command:
  `git submodule update --init`
- Install an editable version of tdl with this command:
  `pip install --editable . --verbose`

## Linux

- Open a command prompt in the cloned git directory.
- Assuming a Debian based distribution of Linux.
  Install tdl's dependancies with this command:
  `sudo apt install gcc python-dev libsdl2-dev libffi-dev libomp-dev`
- Make sure the libtcod submodule is downloaded with this command:
  `git submodule update --init`
- Install an editable version of tdl with this command:
  `pip install --editable . --verbose`
