"""The fast Python port of libtcod.

This module can be used as a drop in replacement for the official libtcodpy module.

Bring any issues or feature requests to GitHub: https://github.com/libtcod/python-tcod

Read the documentation online: https://python-tcod.readthedocs.io/en/latest/
"""

from __future__ import annotations

import warnings
from typing import Any

from tcod import (
    bsp,
    color,
    console,
    constants,
    context,
    event,
    image,
    libtcodpy,
    los,
    map,
    noise,
    path,
    random,
    tileset,
)
from tcod.version import __version__


def __getattr__(name: str, stacklevel: int = 1) -> Any:  # noqa: ANN401
    """Mark access to color constants as deprecated."""
    if name == "Console":
        warnings.warn(
            "tcod.Console is deprecated.\nReplace 'tcod.Console' with 'tcod.console.Console'",
            FutureWarning,
            stacklevel=stacklevel + 1,
        )
        return console.Console
    value: Any = getattr(constants, name, None)
    if isinstance(value, color.Color):
        warnings.warn(
            f"Color constants will be removed from future releases.\nReplace 'tcod.{name}' with '{tuple(value)}'",
            FutureWarning,
            stacklevel=stacklevel + 1,
        )
        return value
    if value is not None:
        warnings.warn(
            "Soon the 'tcod' module will no longer hold constants directly."
            "\nAdd 'from tcod import libtcodpy' if you haven't already."
            f"\nReplace 'tcod.{name}' with 'libtcodpy.{name}'",
            FutureWarning,
            stacklevel=stacklevel + 1,
        )
        return value
    value = getattr(libtcodpy, name, None)
    if value is not None:
        warnings.warn(
            "Soon the 'tcod' module will no longer be an implicit reference to 'libtcodpy'."
            "\nAdd 'from tcod import libtcodpy' if you haven't already."
            f"\nReplace 'tcod.{name}' with 'libtcodpy.{name}'",
            FutureWarning,
            stacklevel=stacklevel + 1,
        )
        return value

    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


__all__ = [
    "__version__",
    "bsp",
    "color",
    "console",
    "context",
    "event",
    "tileset",
    "image",
    "los",
    "map",
    "noise",
    "path",
    "random",
    "tileset",
]
