"""Test SDL specific features."""

import sys

import numpy as np
import pytest

import tcod.sdl.render
import tcod.sdl.sys
import tcod.sdl.video

# ruff: noqa: D103


def test_sdl_window(uses_window: None) -> None:
    assert tcod.sdl.video.get_grabbed_window() is None
    window = tcod.sdl.video.new_window(1, 1)
    window.raise_window()
    window.maximize()
    window.restore()
    window.minimize()
    window.hide()
    window.show()
    assert window.title == sys.argv[0]
    window.title = "Title"
    assert window.title == "Title"
    assert window.opacity == 1.0
    window.position = window.position
    window.fullscreen = window.fullscreen
    window.resizable = window.resizable
    window.size = window.size
    window.min_size = window.min_size
    window.max_size = window.max_size
    window.border_size  # noqa: B018
    window.set_icon(np.zeros((32, 32, 3), dtype=np.uint8))
    with pytest.raises(TypeError):
        window.set_icon(np.zeros((32, 32, 5), dtype=np.uint8))
    with pytest.raises(TypeError):
        window.set_icon(np.zeros((32, 32), dtype=np.uint8))
    window.opacity = window.opacity
    window.grab = window.grab


def test_sdl_window_bad_types() -> None:
    with pytest.raises(TypeError):
        tcod.sdl.video.Window(tcod.ffi.cast("SDL_Window*", tcod.ffi.NULL))
    with pytest.raises(TypeError):
        tcod.sdl.video.Window(tcod.ffi.new("SDL_Rect*"))


def test_sdl_screen_saver(uses_window: None) -> None:
    tcod.sdl.sys.init()
    assert tcod.sdl.video.screen_saver_allowed(False) is False
    assert tcod.sdl.video.screen_saver_allowed(True) is True
    assert tcod.sdl.video.screen_saver_allowed() is True


def test_sdl_render(uses_window: None) -> None:
    window = tcod.sdl.video.new_window(1, 1)
    render = tcod.sdl.render.new_renderer(window, software=True, vsync=False, target_textures=True)
    render.clear()
    render.present()
    render.clear()
    rgb = render.upload_texture(np.zeros((8, 8, 3), np.uint8))
    assert (rgb.width, rgb.height) == (8, 8)
    assert rgb.access == tcod.sdl.render.TextureAccess.STATIC
    assert rgb.format == tcod.lib.SDL_PIXELFORMAT_RGB24
    rgb.alpha_mod = rgb.alpha_mod
    rgb.blend_mode = rgb.blend_mode
    rgb.color_mod = rgb.color_mod
    rgba = render.upload_texture(np.zeros((8, 8, 4), np.uint8), access=tcod.sdl.render.TextureAccess.TARGET)
    with render.set_render_target(rgba):
        render.copy(rgb)
    with pytest.raises(TypeError):
        render.upload_texture(np.zeros((8, 8, 5), np.uint8))

    assert (render.read_pixels() == (0, 0, 0, 255)).all()
    assert (render.read_pixels(format="RGB") == (0, 0, 0)).all()
    assert render.read_pixels(rect=(1, 2, 3, 4)).shape == (4, 3, 4)

    render.draw_point((0, 0))
    render.draw_line((0, 0), (1, 1))
    render.draw_rect((0, 0, 1, 1))
    render.fill_rect((0, 0, 1, 1))

    render.draw_points([(0, 0)])
    render.draw_lines([(0, 0), (1, 1)])
    render.draw_rects([(0, 0, 1, 1)])
    render.fill_rects([(0, 0, 1, 1)])

    for dtype in (np.intc, np.int8, np.uint8, np.float32, np.float16):
        render.draw_points(np.ones((3, 2), dtype=dtype))
        render.draw_lines(np.ones((3, 2), dtype=dtype))
        render.draw_rects(np.ones((3, 4), dtype=dtype))
        render.fill_rects(np.ones((3, 4), dtype=dtype))

    with pytest.raises(TypeError, match=r"shape\[1\] must be 2"):
        render.draw_points(np.ones((3, 1), dtype=dtype))
    with pytest.raises(TypeError, match=r"must have 2 axes"):
        render.draw_points(np.ones((3,), dtype=dtype))

    render.geometry(
        None,
        np.zeros((1, 2), np.float32),
        np.zeros((1, 4), np.uint8),
        np.zeros((1, 2), np.float32),
        np.zeros((3,), np.uint8),
    )


def test_sdl_render_bad_types() -> None:
    with pytest.raises(TypeError):
        tcod.sdl.render.Renderer(tcod.ffi.cast("SDL_Renderer*", tcod.ffi.NULL))
    with pytest.raises(TypeError):
        tcod.sdl.render.Renderer(tcod.ffi.new("SDL_Rect*"))
