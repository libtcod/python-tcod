"""Test for tcod.tileset module."""

import tcod.tileset

# ruff: noqa: D103


def test_proc_block_elements() -> None:
    tileset = tcod.tileset.Tileset(8, 8)
    tcod.tileset.procedural_block_elements(tileset=tileset)
    tileset = tcod.tileset.Tileset(0, 0)
    tcod.tileset.procedural_block_elements(tileset=tileset)
