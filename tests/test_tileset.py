"""Test for tcod.tileset module."""

from pathlib import Path

import pytest

import tcod.console
import tcod.tileset

PROJECT_DIR = Path(__file__).parent / ".."

TERMINAL_FONT = PROJECT_DIR / "fonts/libtcod/terminal8x8_aa_ro.png"
BDF_FONT = PROJECT_DIR / "libtcod/data/fonts/Tamzen5x9r.bdf"

BAD_FILE = PROJECT_DIR / "CHANGELOG.md"  # Any existing non-font file


def test_proc_block_elements() -> None:
    tileset = tcod.tileset.Tileset(0, 0)
    with pytest.deprecated_call():
        tcod.tileset.procedural_block_elements(tileset=tileset)
    tileset += tcod.tileset.procedural_block_elements(shape=tileset.tile_shape)

    tileset = tcod.tileset.Tileset(8, 8)
    with pytest.deprecated_call():
        tcod.tileset.procedural_block_elements(tileset=tileset)
    tileset += tcod.tileset.procedural_block_elements(shape=tileset.tile_shape)


def test_tileset_mix() -> None:
    tileset1 = tcod.tileset.Tileset(1, 1)
    tileset1[ord("a")] = [[0]]

    tileset2 = tcod.tileset.Tileset(1, 1)
    tileset2[ord("a")] = [[1]]
    tileset2[ord("b")] = [[1]]

    assert (tileset1 + tileset2)[ord("a")].tolist() == [[[255, 255, 255, 1]]]  # Replaces tile
    assert (tileset1 | tileset2)[ord("a")].tolist() == [[[255, 255, 255, 0]]]  # Skips existing tile


def test_tileset_contains() -> None:
    tileset = tcod.tileset.Tileset(1, 1)

    # Missing keys
    assert None not in tileset
    assert ord("x") not in tileset
    assert -1 not in tileset
    with pytest.raises(KeyError, match=rf"{ord('x')}"):
        tileset[ord("x")]
    with pytest.raises(KeyError, match=rf"{ord('x')}"):
        del tileset[ord("x")]
    assert len(tileset) == 0

    # Assigned tile is found
    tileset[ord("x")] = [[255]]
    assert ord("x") in tileset
    assert len(tileset) == 1

    # Can be deleted and reassigned
    del tileset[ord("x")]
    assert ord("x") not in tileset
    assert len(tileset) == 0
    tileset[ord("x")] = [[255]]
    assert ord("x") in tileset
    assert len(tileset) == 1


def test_tileset_assignment() -> None:
    tileset = tcod.tileset.Tileset(1, 2)
    tileset[ord("a")] = [[1], [1]]
    tileset[ord("b")] = [[[255, 255, 255, 2]], [[255, 255, 255, 2]]]

    with pytest.raises(ValueError, match=r".*must be \(2, 1, 4\) or \(2, 1\), got \(2, 1, 3\)"):
        tileset[ord("c")] = [[[255, 255, 255]], [[255, 255, 255]]]

    assert tileset.get_tile(ord("d")).shape == (2, 1, 4)


def test_tileset_render() -> None:
    tileset = tcod.tileset.Tileset(1, 2)
    tileset[ord("x")] = [[255], [0]]
    console = tcod.console.Console(3, 2)
    console.rgb[0, 0] = (ord("x"), (255, 0, 0), (0, 255, 0))
    output = tileset.render(console)
    assert output.shape == (4, 3, 4)
    assert output[0:2, 0].tolist() == [[255, 0, 0, 255], [0, 255, 0, 255]]


def test_tileset_tilesheet() -> None:
    tileset = tcod.tileset.load_tilesheet(TERMINAL_FONT, 16, 16, tcod.tileset.CHARMAP_CP437)
    assert tileset.tile_shape == (8, 8)

    with pytest.raises(RuntimeError):
        tcod.tileset.load_tilesheet(BAD_FILE, 16, 16, tcod.tileset.CHARMAP_CP437)


def test_tileset_bdf() -> None:
    tileset = tcod.tileset.load_bdf(BDF_FONT)
    assert tileset.tile_shape == (9, 5)

    with pytest.raises(RuntimeError):
        tileset = tcod.tileset.load_bdf(BAD_FILE)
