.. _part-0:

Part 0 - Setting up a project
##############################################################################

.. include:: notice.rst

Starting tools
==============================================================================

The IDE used for this tutorial is `Visual Studio Code <https://code.visualstudio.com/>`_ [#vscode]_ (not to be mistaken for Visual Studio).

Git will be used for version control.
`Follow the instructions here <https://git-scm.com/downloads>`_.

Python 3.11 was used to make this tutorial.
`Get the latest version of Python here <https://www.python.org/downloads/>`_.
If there exists a version of Python later then 3.11 then install that version instead.


First script
==============================================================================

First start with a modern top-level script.
Create a script in the project root folder called ``main.py`` which checks :python:`if __name__ == "__main__":` and calls a ``main`` function.
Any modern script using type-hinting will also have :python:`from __future__ import annotations` near the top.

.. code-block:: python

    from __future__ import annotations


    def main() -> None:
        print("Hello World!")


    if __name__ == "__main__":
        main()

In VSCode on the left sidebar is a **Run and Debug** tab.
On this tab select **create a launch.json** file.
This will prompt about what kind of program to launch.
Pick ``Python``, then ``Module``, then when asked for the module name type ``main``.
From now on the :kbd:`F5` key will launch ``main.py`` in debug mode.

Run the script now and ``Hello World!`` should be visible in the terminal output.

.. rubric:: Footnotes

.. [#vscode] Alternatives like `PyCharm <https://www.jetbrains.com/pycharm/>`_ were considered,
             but VSCode works the best with Git projects since workspace settings are portable and can be committed without issues.
