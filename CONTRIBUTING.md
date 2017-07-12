


# Building tdl



## Windows

Building Python binaries on Windows requires
[Microsoft Visual Studio](https://www.visualstudio.com/vs/community/)
with the Python development tools installed.

Due to how tdl handles C parsing, you also need to install
[MinGW](http://www.mingw.org/).
You must make sure that the folder conatining `gcc.exe` is on your
Windows %PATH%.

Now open a command prompt and navigate to `setup.py`, then run the
following commands:

    > git submodule update --init
    > py -m pip install --editable . --verbose

This will download dependencies and build tdl in-place.

## MacOS

To build Python binaries you'll need the Xcode command line tools.
You can install them with this command:

    $ xcode-select --install

Now navigate to `setup.py` and run the following:

    $ git submodule update --init
    $ pip install --editable . --verbose

## Linux

The dependancies for Linux can be installed using apt:

    $ sudo apt install gcc python-dev libsdl2-dev libffi-dev libomp-dev

Now navigate to `setup.py` and run the following:

    $ git submodule update --init
    $ pip install --editable . --verbose
