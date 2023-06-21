Part 0 - Setting up a project
##############################################################################

Starting tools
==============================================================================

The IDE used for this tutorial is `Visual Studio Code <https://code.visualstudio.com/>`_ (not to be mistaken for Visual Studio).

Git will be used for version control.
`Follow the instructions here <https://git-scm.com/downloads>`_.

Python 3.11 was used to make this tutorial.
`Get the latest version of Python here <https://www.python.org/downloads/>`_.
If there exists a version of Python later then 3.11 then install that version instead.


First script
==============================================================================

First start with a modern top-level script.
Create a script in the project root folder called ``main.py`` which checks ``if __name__ == "__main__":`` and calls a ``main`` function.

.. code-block:: python

    def main() -> None:
        print("Hello World!")


    if __name__ == "__main__":
        main()

In VSCode on the left sidebar is a **Run and Debug** tab.
On this tab select **create a launch.json** file.
This will prompt about what kind of program to launch.
Pick ``Python``, then ``Module``, then when asked for the module name type ``main``.
From now on the ``F5`` key will launch ``main.py`` in debug mode.

Run the script now and ``"Hello World!"`` should be visible in the terminal output.
