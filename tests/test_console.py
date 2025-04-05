"""Tests for tcod.console."""

import pickle
from pathlib import Path

import numpy as np
import pytest

import tcod
import tcod.console

# ruff: noqa: D103


def test_array_read_write() -> None:
    console = tcod.console.Console(width=12, height=10)
    FG = (255, 254, 253)
    BG = (1, 2, 3)
    CH = ord("&")
    with pytest.warns():
        tcod.console_put_char_ex(console, 0, 0, CH, FG, BG)
    assert console.ch[0, 0] == CH
    assert tuple(console.fg[0, 0]) == FG
    assert tuple(console.bg[0, 0]) == BG

    with pytest.warns():
        tcod.console_put_char_ex(console, 1, 2, CH, FG, BG)
    assert console.ch[2, 1] == CH
    assert tuple(console.fg[2, 1]) == FG
    assert tuple(console.bg[2, 1]) == BG

    console.clear()
    assert console.ch[1, 1] == ord(" ")
    assert tuple(console.fg[1, 1]) == (255, 255, 255)
    assert tuple(console.bg[1, 1]) == (0, 0, 0)

    ch_slice = console.ch[1, :]
    ch_slice[2] = CH
    console.fg[1, ::2] = FG
    console.bg[...] = BG

    with pytest.warns():
        assert tcod.console_get_char(console, 2, 1) == CH
    with pytest.warns():
        assert tuple(tcod.console_get_char_foreground(console, 2, 1)) == FG
    with pytest.warns():
        assert tuple(tcod.console_get_char_background(console, 2, 1)) == BG


@pytest.mark.filterwarnings("ignore")
def test_console_defaults() -> None:
    console = tcod.console.Console(width=12, height=10)

    console.default_bg = [2, 3, 4]  # type: ignore[assignment]
    assert console.default_bg == (2, 3, 4)

    console.default_fg = (4, 5, 6)
    assert console.default_fg == (4, 5, 6)

    console.default_bg_blend = tcod.BKGND_ADD
    assert console.default_bg_blend == tcod.BKGND_ADD

    console.default_alignment = tcod.RIGHT
    assert console.default_alignment == tcod.RIGHT


@pytest.mark.filterwarnings("ignore:Parameter names have been moved around,")
@pytest.mark.filterwarnings("ignore:Pass the key color to Console.blit instead")
@pytest.mark.filterwarnings("ignore:.*default values have been deprecated")
def test_console_methods() -> None:
    console = tcod.console.Console(width=12, height=10)
    console.put_char(0, 0, ord("@"))
    with pytest.deprecated_call():
        console.print_(0, 0, "Test")
    with pytest.deprecated_call():
        console.print_rect(0, 0, 2, 8, "a b c d e f")
    console.get_height_rect(0, 0, 2, 8, "a b c d e f")
    with pytest.deprecated_call():
        console.rect(0, 0, 2, 2, True)
    with pytest.deprecated_call():
        console.hline(0, 1, 10)
    with pytest.deprecated_call():
        console.vline(1, 0, 10)
    with pytest.deprecated_call():
        console.print_frame(0, 0, 8, 8, "Frame")
    console.blit(0, 0, 0, 0, console, 0, 0)  # type: ignore[arg-type]
    console.blit(0, 0, 0, 0, console, 0, 0, key_color=(0, 0, 0))  # type: ignore[arg-type]
    with pytest.deprecated_call():
        console.set_key_color((254, 0, 254))


def test_console_pickle() -> None:
    console = tcod.console.Console(width=12, height=10)
    console.ch[...] = ord(".")
    console.fg[...] = (10, 20, 30)
    console.bg[...] = (1, 2, 3)
    console2 = pickle.loads(pickle.dumps(console))
    assert (console.ch == console2.ch).all()
    assert (console.fg == console2.fg).all()
    assert (console.bg == console2.bg).all()


def test_console_pickle_fortran() -> None:
    console = tcod.console.Console(2, 3, order="F")
    console2 = pickle.loads(pickle.dumps(console))
    assert console.ch.strides == console2.ch.strides
    assert console.fg.strides == console2.fg.strides
    assert console.bg.strides == console2.bg.strides


def test_console_repr() -> None:
    from numpy import array  # noqa: F401  # Used for eval

    eval(repr(tcod.console.Console(10, 2)))  # noqa: S307


def test_console_str() -> None:
    console = tcod.console.Console(10, 2)
    console.ch[:] = ord(".")
    with pytest.warns():
        console.print_(0, 0, "Test")
    assert str(console) == ("<Test......\n ..........>")


def test_console_fortran_buffer() -> None:
    tcod.console.Console(
        width=1,
        height=2,
        order="F",
        buffer=np.zeros((1, 2), order="F", dtype=tcod.console.Console.DTYPE),
    )


def test_console_clear() -> None:
    console = tcod.console.Console(1, 1)
    assert console.fg[0, 0].tolist() == [255, 255, 255]
    assert console.bg[0, 0].tolist() == [0, 0, 0]
    console.clear(fg=(7, 8, 9), bg=(10, 11, 12))
    assert console.fg[0, 0].tolist() == [7, 8, 9]
    assert console.bg[0, 0].tolist() == [10, 11, 12]


def test_console_semigraphics() -> None:
    console = tcod.console.Console(1, 1)
    console.draw_semigraphics(
        [[[255, 255, 255], [255, 255, 255]], [[255, 255, 255], [0, 0, 0]]],
    )


def test_rexpaint(tmp_path: Path) -> None:
    xp_path = tmp_path / "test.xp"
    consoles = tcod.console.Console(80, 24, order="F"), tcod.console.Console(8, 8, order="F")
    tcod.console.save_xp(xp_path, consoles, compress_level=0)
    loaded = tcod.console.load_xp(xp_path, order="F")
    assert len(consoles) == len(loaded)
    assert loaded[0].rgba.flags["F_CONTIGUOUS"]
    assert consoles[0].rgb.shape == loaded[0].rgb.shape
    assert consoles[1].rgb.shape == loaded[1].rgb.shape
    with pytest.raises(FileNotFoundError):
        tcod.console.load_xp(tmp_path / "non_existent")


def test_draw_frame() -> None:
    console = tcod.console.Console(3, 3, order="C")
    with pytest.raises(TypeError):
        console.draw_frame(0, 0, 3, 3, title="test", decoration="123456789")
    with pytest.raises(TypeError):
        console.draw_frame(0, 0, 3, 3, decoration="0123456789")

    console.draw_frame(0, 0, 3, 3, decoration=(49, 50, 51, 52, 53, 54, 55, 56, 57))
    assert console.ch.tolist() == [[49, 50, 51], [52, 53, 54], [55, 56, 57]]
    with pytest.warns():
        console.draw_frame(0, 0, 3, 3, title="T")
