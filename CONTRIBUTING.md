## Code style

Code styles are enforced using black and linters.
These are best enabled with a pre-commit which you can setup with:

```sh
pip install pre-commit
pre-commit install
```

## Building python-tcod

To work with the tcod source, your environment must be set up to build
Python C extensions. You'll also need `cpp` installed for
use with pycparser.

### Windows

- Install [Microsoft Visual Studio](https://www.visualstudio.com/vs/community/)
  - When asked, choose to install the Python development tools.
- Open a command prompt in the cloned git directory.
- Make sure the libtcod submodule is downloaded with this command:
  `git submodule update --init`
- Install an editable version of tcod with this command:
  `py -m pip install --editable . --verbose`

### MacOS

- Open a command prompt in the cloned git directory.
- Install the Xcode command line tools with this command:
  `xcode-select --install`
- Make sure the libtcod submodule is downloaded with this command:
  `git submodule update --init`
- Install an editable version of tcod with this command:
  `pip install --editable . --verbose`

### Linux

- Open a command prompt in the cloned git directory.
- Assuming a Debian based distribution of Linux.
  Install tcod's dependencies with this command:
  `sudo apt install gcc python-dev libsdl2-dev libffi-dev`
- Make sure the libtcod submodule is downloaded with this command:
  `git submodule update --init`
- Install an editable version of tdl with this command:
  `pip install --editable . --verbose`
