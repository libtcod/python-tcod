Frequently Asked Questions
==========================

How do you set a frames-per-second while using contexts?
--------------------------------------------------------

You'll need to use an external tool to manage the framerate.
This can either be your own custom tool or you can copy the Clock class from the
`framerate.py <https://github.com/libtcod/python-tcod/blob/develop/examples/framerate.py>`_
example.

I get ``No module named 'tcod'`` when I try to ``import tcod`` in PyCharm.
--------------------------------------------------------------------------

`PyCharm`_ will automatically setup a `Python virtual environment <https://docs.python.org/3/tutorial/venv.html>`_ for new or added projects.
By default this virtual environment is isolated and will ignore global Python packages installed from the standard terminal. **In this case you MUST install tcod inside of your per-project virtual environment.**

The recommended way to work with PyCharm is to add a ``requirements.txt`` file to the root of your PyCharm project with a `requirement specifier <https://pip.pypa.io/en/stable/cli/pip_install/#requirement-specifiers>`_ for `tcod`.
This file should have the following:

.. code-block:: python

    # requirements.txt
    # https://pip.pypa.io/en/stable/cli/pip_install/#requirements-file-format
    tcod

Once this file is saved to your projects root directory then PyCharm will detect it and ask if you want these requirements installed.  Say yes and `tcod` will be installed to the `virtual environment`.  Be sure to add more specifiers for any modules you're using other than `tcod`, such as `numpy`.

Alternatively you can open the `Terminal` tab in PyCharm and run ``pip install tcod`` there.  This will install `tcod` to the currently open project.


.. _PyCharm: https://www.jetbrains.com/pycharm/
