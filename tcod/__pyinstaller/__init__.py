"""PyInstaller entry point for tcod."""

from __future__ import annotations

from pathlib import Path


def get_hook_dirs() -> list[str]:
    """Return the current directory."""
    return [str(Path(__file__).parent)]
