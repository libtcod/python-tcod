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


How do I add custom tiles?
--------------------------

Libtcod uses Unicode to identify tiles.
To prevent conflicts with real glyphs you should decide on codepoints from a `Private Use Area <https://en.wikipedia.org/wiki/Private_Use_Areas>`_ before continuing.
If you're unsure, then use the codepoints from ``0x100000`` to ``0x10FFFD`` for your custom tiles.

Normally you load a font with :func:`tcod.tileset.load_tilesheet` which will return a :any:`Tileset` that gets passed to :func:`tcod.context.new`'s `tileset` parameter.
:func:`tcod.tileset.load_tilesheet` assigns the codepoints from `charmap` to the tilesheet in row-major order.

There are two ways to extend a tileset like the above:

- Increase the tilesheet size vertically and update the `rows` parameter in :func:`tcod.tileset.load_tilesheet` to match the new image size, then modify the `charmap` parameter to map the new tiles to codepoints.
  If you edited a CP437 tileset this way then you'd add your new codepoints to the end of :any:`tcod.tileset.CHARMAP_CP437` before using the result as the `charmap` parameter.
  You can also use :any:`Tileset.remap` if you want to reassign tiles based on their position rather than editing `charmap`.
- Or do not modify the original tilesheet.
  Load the tileset normally, then add new tiles with :any:`Tileset.set_tile` with manually loaded images.


.. _PyCharm: https://www.jetbrains.com/pycharm/
