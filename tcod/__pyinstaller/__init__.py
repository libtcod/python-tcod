"""PyInstaller entry point for tcod."""
import os
from typing import List


def get_hook_dirs() -> List[str]:
    """Return the current directory."""
    return [os.path.dirname(__file__)]
