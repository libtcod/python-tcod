cx_Freeze Example
=================

It's recommended to use a virtual environment to package Python executables.
Use the following guide on how to set one up:
https://docs.python.org/3/tutorial/venv.html

Once the virtual environment is active you should install `tcod` and `cx_Freeze` from the `requirements.txt` file, then build using `setup.py`:

    pip install -r requirements.txt
    python setup.py build

An executable package will be placed in ``build/<platform>/``

The cx_Freeze documentation can be found at: https://cx-freeze.readthedocs.io
