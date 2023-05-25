from typing import Any, Callable, Iterator, List, Optional, Tuple, Union

import numpy
import numpy as np
import pytest
from numpy.typing import NDArray

import tcod
import tcod as libtcodpy

pytestmark = [
    pytest.mark.filterwarnings("ignore::DeprecationWarning"),
    pytest.mark.filterwarnings("ignore::PendingDeprecationWarning"),
]


def test_console_behavior(console: tcod.Console) -> None:
    assert not console


@pytest.mark.skip("takes too long")
@pytest.mark.filterwarnings("ignore")
def test_credits_long(console: tcod.console.Console) -> None:
    libtcodpy.console_credits()


def test_credits(console: tcod.console.Console) -> None:
    libtcodpy.console_credits_render(0, 0, True)
    libtcodpy.console_credits_reset()


def assert_char(
    console: tcod.console.Console,
    x: int,
    y: int,
    ch: Optional[Union[str, int]] = None,
    fg: Optional[Tuple[int, int, int]] = None,
    bg: Optional[Tuple[int, int, int]] = None,
) -> None:
    if ch is not None:
        if isinstance(ch, str):
            ch = ord(ch)
        assert console.ch[y, x] == ch
    if fg is not None:
        assert (console.fg[y, x] == fg).all()
    if bg is not None:
        assert (console.bg[y, x] == bg).all()


@pytest.mark.filterwarnings("ignore")
def test_console_defaults(console: tcod.console.Console, fg: Tuple[int, int, int], bg: Tuple[int, int, int]) -> None:
    libtcodpy.console_set_default_foreground(console, fg)
    libtcodpy.console_set_default_background(console, bg)
    libtcodpy.console_clear(console)
    assert_char(console, 0, 0, None, fg, bg)


@pytest.mark.filterwarnings("ignore")
def test_console_set_char_background(console: tcod.console.Console, bg: Tuple[int, int, int]) -> None:
    libtcodpy.console_set_char_background(console, 0, 0, bg, libtcodpy.BKGND_SET)
    assert_char(console, 0, 0, bg=bg)


@pytest.mark.filterwarnings("ignore")
def test_console_set_char_foreground(console: tcod.console.Console, fg: Tuple[int, int, int]) -> None:
    libtcodpy.console_set_char_foreground(console, 0, 0, fg)
    assert_char(console, 0, 0, fg=fg)


@pytest.mark.filterwarnings("ignore")
def test_console_set_char(console: tcod.console.Console, ch: int) -> None:
    libtcodpy.console_set_char(console, 0, 0, ch)
    assert_char(console, 0, 0, ch=ch)


@pytest.mark.filterwarnings("ignore")
def test_console_put_char(console: tcod.console.Console, ch: int) -> None:
    libtcodpy.console_put_char(console, 0, 0, ch, libtcodpy.BKGND_SET)
    assert_char(console, 0, 0, ch=ch)


@pytest.mark.filterwarnings("ignore")
def console_put_char_ex(
    console: tcod.console.Console, ch: int, fg: Tuple[int, int, int], bg: Tuple[int, int, int]
) -> None:
    libtcodpy.console_put_char_ex(console, 0, 0, ch, fg, bg)
    assert_char(console, 0, 0, ch=ch, fg=fg, bg=bg)


@pytest.mark.filterwarnings("ignore")
def test_console_printing(console: tcod.console.Console, fg: Tuple[int, int, int], bg: Tuple[int, int, int]) -> None:
    libtcodpy.console_set_background_flag(console, libtcodpy.BKGND_SET)
    assert libtcodpy.console_get_background_flag(console) == libtcodpy.BKGND_SET

    libtcodpy.console_set_alignment(console, libtcodpy.LEFT)
    assert libtcodpy.console_get_alignment(console) == libtcodpy.LEFT

    libtcodpy.console_print(console, 0, 0, "print")
    libtcodpy.console_print_ex(console, 0, 0, libtcodpy.BKGND_SET, libtcodpy.LEFT, "print ex")

    assert libtcodpy.console_print_rect(console, 0, 0, 8, 8, "print rect") > 0
    assert (
        libtcodpy.console_print_rect_ex(console, 0, 0, 8, 8, libtcodpy.BKGND_SET, libtcodpy.LEFT, "print rect ex") > 0
    )
    assert libtcodpy.console_get_height_rect(console, 0, 0, 8, 8, "get height") > 0

    libtcodpy.console_set_color_control(libtcodpy.COLCTRL_1, fg, bg)


@pytest.mark.filterwarnings("ignore")
def test_console_rect(console: tcod.console.Console) -> None:
    libtcodpy.console_rect(console, 0, 0, 4, 4, False, libtcodpy.BKGND_SET)


@pytest.mark.filterwarnings("ignore")
def test_console_lines(console: tcod.console.Console) -> None:
    libtcodpy.console_hline(console, 0, 0, 4)
    libtcodpy.console_vline(console, 0, 0, 4)


@pytest.mark.filterwarnings("ignore")
def test_console_print_frame(console: tcod.console.Console) -> None:
    libtcodpy.console_print_frame(console, 0, 0, 9, 9)


@pytest.mark.filterwarnings("ignore")
def test_console_fade(console: tcod.console.Console) -> None:
    libtcodpy.console_set_fade(0, (0, 0, 0))
    libtcodpy.console_get_fade()
    libtcodpy.console_get_fading_color()


def assertConsolesEqual(a: tcod.console.Console, b: tcod.console.Console) -> bool:
    return bool((a.fg[:] == b.fg[:]).all() and (a.bg[:] == b.bg[:]).all() and (a.ch[:] == b.ch[:]).all())


@pytest.mark.filterwarnings("ignore")
def test_console_blit(console: tcod.console.Console, offscreen: tcod.console.Console) -> None:
    libtcodpy.console_print(offscreen, 0, 0, "test")
    libtcodpy.console_blit(offscreen, 0, 0, 0, 0, console, 0, 0, 1, 1)
    assertConsolesEqual(console, offscreen)
    libtcodpy.console_set_key_color(offscreen, (0, 0, 0))


@pytest.mark.filterwarnings("ignore")
def test_console_asc_read_write(console: tcod.console.Console, offscreen: tcod.console.Console, tmpdir: Any) -> None:
    libtcodpy.console_print(console, 0, 0, "test")

    asc_file = tmpdir.join("test.asc").strpath
    assert libtcodpy.console_save_asc(console, asc_file)
    assert libtcodpy.console_load_asc(offscreen, asc_file)
    assertConsolesEqual(console, offscreen)


@pytest.mark.filterwarnings("ignore")
def test_console_apf_read_write(console: tcod.console.Console, offscreen: tcod.console.Console, tmpdir: Any) -> None:
    libtcodpy.console_print(console, 0, 0, "test")

    apf_file = tmpdir.join("test.apf").strpath
    assert libtcodpy.console_save_apf(console, apf_file)
    assert libtcodpy.console_load_apf(offscreen, apf_file)
    assertConsolesEqual(console, offscreen)


@pytest.mark.filterwarnings("ignore")
def test_console_rexpaint_load_test_file(console: tcod.console.Console) -> None:
    xp_console = libtcodpy.console_from_xp("libtcod/data/rexpaint/test.xp")
    assert xp_console
    assert libtcodpy.console_get_char(xp_console, 0, 0) == ord("T")
    assert libtcodpy.console_get_char(xp_console, 1, 0) == ord("e")
    assert libtcodpy.console_get_char_background(xp_console, 0, 1) == libtcodpy.Color(255, 0, 0)
    assert libtcodpy.console_get_char_background(xp_console, 1, 1) == libtcodpy.Color(0, 255, 0)
    assert libtcodpy.console_get_char_background(xp_console, 2, 1) == libtcodpy.Color(0, 0, 255)


@pytest.mark.filterwarnings("ignore")
def test_console_rexpaint_save_load(
    console: tcod.console.Console,
    tmpdir: Any,
    ch: int,
    fg: Tuple[int, int, int],
    bg: Tuple[int, int, int],
) -> None:
    libtcodpy.console_print(console, 0, 0, "test")
    libtcodpy.console_put_char_ex(console, 1, 1, ch, fg, bg)
    xp_file = tmpdir.join("test.xp").strpath
    assert libtcodpy.console_save_xp(console, xp_file, 1)
    xp_console = libtcodpy.console_from_xp(xp_file)
    assert xp_console
    assertConsolesEqual(console, xp_console)
    assert libtcodpy.console_load_xp(None, xp_file)  # type: ignore
    assertConsolesEqual(console, xp_console)


@pytest.mark.filterwarnings("ignore")
def test_console_rexpaint_list_save_load(console: tcod.console.Console, tmpdir: Any) -> None:
    con1 = libtcodpy.console_new(8, 2)
    con2 = libtcodpy.console_new(8, 2)
    libtcodpy.console_print(con1, 0, 0, "hello")
    libtcodpy.console_print(con2, 0, 0, "world")
    xp_file = tmpdir.join("test.xp").strpath
    assert libtcodpy.console_list_save_xp([con1, con2], xp_file, 1)
    loaded_consoles = libtcodpy.console_list_load_xp(xp_file)
    assert loaded_consoles
    for a, b in zip([con1, con2], loaded_consoles):
        assertConsolesEqual(a, b)
        libtcodpy.console_delete(a)
        libtcodpy.console_delete(b)


@pytest.mark.filterwarnings("ignore")
def test_console_fullscreen(console: tcod.console.Console) -> None:
    libtcodpy.console_set_fullscreen(False)


@pytest.mark.filterwarnings("ignore")
def test_console_key_input(console: tcod.console.Console) -> None:
    libtcodpy.console_check_for_keypress()
    libtcodpy.console_is_key_pressed(libtcodpy.KEY_ENTER)


@pytest.mark.filterwarnings("ignore")
def test_console_fill_errors(console: tcod.console.Console) -> None:
    with pytest.raises(TypeError):
        libtcodpy.console_fill_background(console, [0], [], [])
    with pytest.raises(TypeError):
        libtcodpy.console_fill_foreground(console, [0], [], [])


@pytest.mark.filterwarnings("ignore")
def test_console_fill(console: tcod.console.Console) -> None:
    width = libtcodpy.console_get_width(console)
    height = libtcodpy.console_get_height(console)
    fill = [i % 256 for i in range(width * height)]
    libtcodpy.console_fill_background(console, fill, fill, fill)
    libtcodpy.console_fill_foreground(console, fill, fill, fill)
    libtcodpy.console_fill_char(console, fill)

    # verify fill
    bg, fg, ch = [], [], []
    for y in range(height):
        for x in range(width):
            bg.append(libtcodpy.console_get_char_background(console, x, y)[0])
            fg.append(libtcodpy.console_get_char_foreground(console, x, y)[0])
            ch.append(libtcodpy.console_get_char(console, x, y))
    assert fill == bg
    assert fill == fg
    assert fill == ch


@pytest.mark.filterwarnings("ignore")
def test_console_fill_numpy(console: tcod.console.Console) -> None:
    width = libtcodpy.console_get_width(console)
    height = libtcodpy.console_get_height(console)
    fill: NDArray[np.intc] = numpy.zeros((height, width), dtype=np.intc)
    for y in range(height):
        fill[y, :] = y % 256

    libtcodpy.console_fill_background(console, fill, fill, fill)  # type: ignore
    libtcodpy.console_fill_foreground(console, fill, fill, fill)  # type: ignore
    libtcodpy.console_fill_char(console, fill)  # type: ignore

    # verify fill
    bg: NDArray[np.intc] = numpy.zeros((height, width), dtype=numpy.intc)
    fg: NDArray[np.intc] = numpy.zeros((height, width), dtype=numpy.intc)
    ch: NDArray[np.intc] = numpy.zeros((height, width), dtype=numpy.intc)
    for y in range(height):
        for x in range(width):
            bg[y, x] = libtcodpy.console_get_char_background(console, x, y)[0]
            fg[y, x] = libtcodpy.console_get_char_foreground(console, x, y)[0]
            ch[y, x] = libtcodpy.console_get_char(console, x, y)
    fill = fill.tolist()
    assert fill == bg.tolist()
    assert fill == fg.tolist()
    assert fill == ch.tolist()


@pytest.mark.filterwarnings("ignore")
def test_console_buffer(console: tcod.console.Console) -> None:
    buffer = libtcodpy.ConsoleBuffer(
        libtcodpy.console_get_width(console),
        libtcodpy.console_get_height(console),
    )
    buffer = buffer.copy()
    buffer.set_fore(0, 0, 0, 0, 0, "@")
    buffer.set_back(0, 0, 0, 0, 0)
    buffer.set(0, 0, 0, 0, 0, 0, 0, 0, "@")
    buffer.blit(console)


@pytest.mark.filterwarnings("ignore:Console array attributes perform better")
def test_console_buffer_error(console: tcod.console.Console) -> None:
    buffer = libtcodpy.ConsoleBuffer(0, 0)
    with pytest.raises(ValueError):
        buffer.blit(console)


@pytest.mark.filterwarnings("ignore")
def test_console_font_mapping(console: tcod.console.Console) -> None:
    libtcodpy.console_map_ascii_code_to_font(ord("@"), 1, 1)
    libtcodpy.console_map_ascii_codes_to_font(ord("@"), 1, 0, 0)
    libtcodpy.console_map_string_to_font("@", 0, 0)


@pytest.mark.filterwarnings("ignore")
def test_mouse(console: tcod.console.Console) -> None:
    libtcodpy.mouse_show_cursor(True)
    libtcodpy.mouse_is_cursor_visible()
    mouse = libtcodpy.mouse_get_status()
    repr(mouse)
    libtcodpy.mouse_move(0, 0)


@pytest.mark.filterwarnings("ignore")
def test_sys_time(console: tcod.console.Console) -> None:
    libtcodpy.sys_set_fps(0)
    libtcodpy.sys_get_fps()
    libtcodpy.sys_get_last_frame_length()
    libtcodpy.sys_sleep_milli(0)
    libtcodpy.sys_elapsed_milli()
    libtcodpy.sys_elapsed_seconds()


@pytest.mark.filterwarnings("ignore")
def test_sys_screenshot(console: tcod.console.Console, tmpdir: Any) -> None:
    libtcodpy.sys_save_screenshot(tmpdir.join("test.png").strpath)


@pytest.mark.filterwarnings("ignore")
def test_sys_custom_render(console: tcod.console.Console) -> None:
    if libtcodpy.sys_get_renderer() != libtcodpy.RENDERER_SDL:
        pytest.xfail(reason="Only supports SDL")

    escape = []

    def sdl_callback(sdl_surface: Any) -> None:
        escape.append(True)

    libtcodpy.sys_register_SDL_renderer(sdl_callback)
    libtcodpy.console_flush()
    assert escape, "proof that sdl_callback was called"


@pytest.mark.filterwarnings("ignore")
def test_image(console: tcod.console.Console, tmpdir: Any) -> None:
    img = libtcodpy.image_new(16, 16)
    libtcodpy.image_clear(img, (0, 0, 0))
    libtcodpy.image_invert(img)
    libtcodpy.image_hflip(img)
    libtcodpy.image_rotate90(img)
    libtcodpy.image_vflip(img)
    libtcodpy.image_scale(img, 24, 24)
    libtcodpy.image_set_key_color(img, (255, 255, 255))
    libtcodpy.image_get_alpha(img, 0, 0)
    libtcodpy.image_is_pixel_transparent(img, 0, 0)
    libtcodpy.image_get_size(img)
    libtcodpy.image_get_pixel(img, 0, 0)
    libtcodpy.image_get_mipmap_pixel(img, 0, 0, 1, 1)
    libtcodpy.image_put_pixel(img, 0, 0, (255, 255, 255))
    libtcodpy.image_blit(img, console, 0, 0, libtcodpy.BKGND_SET, 1, 1, 0)
    libtcodpy.image_blit_rect(img, console, 0, 0, 16, 16, libtcodpy.BKGND_SET)
    libtcodpy.image_blit_2x(img, console, 0, 0)
    libtcodpy.image_save(img, tmpdir.join("test.png").strpath)
    libtcodpy.image_delete(img)

    img = libtcodpy.image_from_console(console)
    libtcodpy.image_refresh_console(img, console)
    libtcodpy.image_delete(img)

    libtcodpy.image_delete(libtcodpy.image_load("libtcod/data/img/circle.png"))


@pytest.mark.parametrize("sample", ["@", "\u2603"])  # Unicode snowman
@pytest.mark.xfail(reason="Unreliable")
@pytest.mark.filterwarnings("ignore")
def test_clipboard(console: tcod.console.Console, sample: str) -> None:
    saved = libtcodpy.sys_clipboard_get()
    try:
        libtcodpy.sys_clipboard_set(sample)
        assert libtcodpy.sys_clipboard_get() == sample
    finally:
        libtcodpy.sys_clipboard_set(saved)


# arguments to test with and the results expected from these arguments
LINE_ARGS = (-5, 0, 5, 10)
EXCLUSIVE_RESULTS = [(-4, 1), (-3, 2), (-2, 3), (-1, 4), (0, 5), (1, 6), (2, 7), (3, 8), (4, 9), (5, 10)]
INCLUSIVE_RESULTS = [(-5, 0)] + EXCLUSIVE_RESULTS


@pytest.mark.filterwarnings("ignore")
def test_line_step() -> None:
    """
    libtcodpy.line_init and libtcodpy.line_step
    """
    libtcodpy.line_init(*LINE_ARGS)
    for expected_xy in EXCLUSIVE_RESULTS:
        assert libtcodpy.line_step() == expected_xy
    assert libtcodpy.line_step() == (None, None)


@pytest.mark.filterwarnings("ignore")
def test_line() -> None:
    """
    tests normal use, lazy evaluation, and error propagation
    """
    # test normal results
    test_result: List[Tuple[int, int]] = []

    def line_test(x: int, y: int) -> bool:
        test_result.append((x, y))
        return True

    assert libtcodpy.line(*LINE_ARGS, py_callback=line_test) == 1
    assert test_result == INCLUSIVE_RESULTS

    # test lazy evaluation
    test_result = []

    def return_false(x: int, y: int) -> bool:
        test_result.append((x, y))
        return False

    assert libtcodpy.line(*LINE_ARGS, py_callback=return_false) == 0
    assert test_result == INCLUSIVE_RESULTS[:1]


@pytest.mark.filterwarnings("ignore")
def test_line_iter() -> None:
    """
    libtcodpy.line_iter
    """
    assert list(libtcodpy.line_iter(*LINE_ARGS)) == INCLUSIVE_RESULTS


@pytest.mark.filterwarnings("ignore")
def test_bsp() -> None:
    """
    commented out statements work in libtcod-cffi
    """
    bsp = libtcodpy.bsp_new_with_size(0, 0, 64, 64)
    repr(bsp)  # test __repr__ on leaf
    libtcodpy.bsp_resize(bsp, 0, 0, 32, 32)
    assert bsp is not None

    # test getter/setters
    bsp.x = bsp.x
    bsp.y = bsp.y
    bsp.w = bsp.w
    bsp.h = bsp.h
    bsp.position = bsp.position
    bsp.horizontal = bsp.horizontal
    bsp.level = bsp.level

    # cover functions on leaf
    # self.assertFalse(libtcodpy.bsp_left(bsp))
    # self.assertFalse(libtcodpy.bsp_right(bsp))
    # self.assertFalse(libtcodpy.bsp_father(bsp))
    assert libtcodpy.bsp_is_leaf(bsp)

    assert libtcodpy.bsp_contains(bsp, 1, 1)
    # self.assertFalse(libtcodpy.bsp_contains(bsp, -1, -1))
    # self.assertEqual(libtcodpy.bsp_find_node(bsp, 1, 1), bsp)
    # self.assertFalse(libtcodpy.bsp_find_node(bsp, -1, -1))

    libtcodpy.bsp_split_once(bsp, False, 4)
    repr(bsp)  # test __repr__ with parent
    libtcodpy.bsp_split_once(bsp, True, 4)
    repr(bsp)

    # cover functions on parent
    assert libtcodpy.bsp_left(bsp)
    assert libtcodpy.bsp_right(bsp)
    # self.assertFalse(libtcodpy.bsp_father(bsp))
    assert not libtcodpy.bsp_is_leaf(bsp)
    # self.assertEqual(libtcodpy.bsp_father(libtcodpy.bsp_left(bsp)), bsp)
    # self.assertEqual(libtcodpy.bsp_father(libtcodpy.bsp_right(bsp)), bsp)

    libtcodpy.bsp_split_recursive(bsp, None, 4, 2, 2, 1.0, 1.0)

    # cover bsp_traverse
    def traverse(node: tcod.bsp.BSP, user_data: Any) -> None:
        return None

    libtcodpy.bsp_traverse_pre_order(bsp, traverse)
    libtcodpy.bsp_traverse_in_order(bsp, traverse)
    libtcodpy.bsp_traverse_post_order(bsp, traverse)
    libtcodpy.bsp_traverse_level_order(bsp, traverse)
    libtcodpy.bsp_traverse_inverted_level_order(bsp, traverse)

    # test __repr__ on deleted node
    son = libtcodpy.bsp_left(bsp)
    libtcodpy.bsp_remove_sons(bsp)
    repr(son)

    libtcodpy.bsp_delete(bsp)


@pytest.mark.filterwarnings("ignore")
def test_map() -> None:
    map = libtcodpy.map_new(16, 16)
    assert libtcodpy.map_get_width(map) == 16
    assert libtcodpy.map_get_height(map) == 16
    libtcodpy.map_copy(map, map)
    libtcodpy.map_clear(map)
    libtcodpy.map_set_properties(map, 0, 0, True, True)
    assert libtcodpy.map_is_transparent(map, 0, 0)
    assert libtcodpy.map_is_walkable(map, 0, 0)
    libtcodpy.map_is_in_fov(map, 0, 0)
    libtcodpy.map_delete(map)


@pytest.mark.filterwarnings("ignore")
def test_color() -> None:
    color_a = libtcodpy.Color(0, 1, 2)
    assert list(color_a) == [0, 1, 2]
    assert color_a[0] == color_a.r
    assert color_a[1] == color_a.g
    assert color_a[2] == color_a.b

    color_a[1] = 3
    color_a["b"] = color_a["b"]
    assert list(color_a) == [0, 3, 2]

    assert color_a == color_a

    color_b = libtcodpy.Color(255, 255, 255)
    assert color_a != color_b

    color = libtcodpy.color_lerp(color_a, color_b, 0.5)  # type: ignore
    libtcodpy.color_set_hsv(color, 0, 0, 0)
    libtcodpy.color_get_hsv(color)  # type: ignore
    libtcodpy.color_scale_HSV(color, 0, 0)


def test_color_repr() -> None:
    Color = libtcodpy.Color
    col = Color(0, 1, 2)
    assert eval(repr(col)) == col


@pytest.mark.filterwarnings("ignore")
def test_color_math() -> None:
    color_a = libtcodpy.Color(0, 1, 2)
    color_b = libtcodpy.Color(0, 10, 20)

    assert color_a + color_b == libtcodpy.Color(0, 11, 22)
    assert color_b - color_a == libtcodpy.Color(0, 9, 18)
    assert libtcodpy.Color(255, 255, 255) * color_a == color_a
    assert color_a * 100 == libtcodpy.Color(0, 100, 200)


@pytest.mark.filterwarnings("ignore")
def test_color_gen_map() -> None:
    colors = libtcodpy.color_gen_map([(0, 0, 0), (255, 255, 255)], [0, 8])
    assert colors[0] == libtcodpy.Color(0, 0, 0)
    assert colors[-1] == libtcodpy.Color(255, 255, 255)


@pytest.mark.filterwarnings("ignore")
def test_namegen_parse() -> None:
    libtcodpy.namegen_parse("libtcod/data/namegen/jice_celtic.cfg")
    assert libtcodpy.namegen_generate("Celtic female")
    assert libtcodpy.namegen_get_sets()
    libtcodpy.namegen_destroy()


@pytest.mark.filterwarnings("ignore")
def test_noise() -> None:
    noise = libtcodpy.noise_new(1)
    libtcodpy.noise_set_type(noise, libtcodpy.NOISE_SIMPLEX)
    libtcodpy.noise_get(noise, [0])
    libtcodpy.noise_get_fbm(noise, [0], 4)
    libtcodpy.noise_get_turbulence(noise, [0], 4)
    libtcodpy.noise_delete(noise)


@pytest.mark.filterwarnings("ignore")
def test_random() -> None:
    rand = libtcodpy.random_get_instance()
    rand = libtcodpy.random_new()
    libtcodpy.random_delete(rand)
    rand = libtcodpy.random_new_from_seed(42)
    libtcodpy.random_set_distribution(rand, libtcodpy.DISTRIBUTION_LINEAR)
    libtcodpy.random_get_int(rand, 0, 1)
    libtcodpy.random_get_int_mean(rand, 0, 1, 0)
    libtcodpy.random_get_float(rand, 0, 1)
    libtcodpy.random_get_double(rand, 0, 1)
    libtcodpy.random_get_float_mean(rand, 0, 1, 0)
    libtcodpy.random_get_double_mean(rand, 0, 1, 0)

    backup = libtcodpy.random_save(rand)
    libtcodpy.random_restore(rand, backup)

    libtcodpy.random_delete(rand)
    libtcodpy.random_delete(backup)


@pytest.mark.filterwarnings("ignore")
def test_heightmap() -> None:
    h_map = libtcodpy.heightmap_new(16, 16)
    repr(h_map)
    noise = libtcodpy.noise_new(2)

    # basic operations
    libtcodpy.heightmap_set_value(h_map, 0, 0, 1)
    libtcodpy.heightmap_add(h_map, 1)
    libtcodpy.heightmap_scale(h_map, 1)
    libtcodpy.heightmap_clear(h_map)
    libtcodpy.heightmap_clamp(h_map, 0, 0)
    libtcodpy.heightmap_copy(h_map, h_map)
    libtcodpy.heightmap_normalize(h_map)
    libtcodpy.heightmap_lerp_hm(h_map, h_map, h_map, 0)
    libtcodpy.heightmap_add_hm(h_map, h_map, h_map)
    libtcodpy.heightmap_multiply_hm(h_map, h_map, h_map)

    # modifying the heightmap
    libtcodpy.heightmap_add_hill(h_map, 0, 0, 4, 1)
    libtcodpy.heightmap_dig_hill(h_map, 0, 0, 4, 1)
    libtcodpy.heightmap_rain_erosion(h_map, 1, 1, 1)
    libtcodpy.heightmap_kernel_transform(h_map, 3, [-1, 1, 0], [0, 0, 0], [0.33, 0.33, 0.33], 0, 1)
    libtcodpy.heightmap_add_voronoi(h_map, 10, 3, [1, 3, 5])
    libtcodpy.heightmap_add_fbm(h_map, noise, 1, 1, 1, 1, 4, 1, 1)
    libtcodpy.heightmap_scale_fbm(h_map, noise, 1, 1, 1, 1, 4, 1, 1)
    libtcodpy.heightmap_dig_bezier(h_map, (0, 16, 16, 0), (0, 0, 16, 16), 1, 1, 1, 1)

    # read data
    libtcodpy.heightmap_get_value(h_map, 0, 0)
    libtcodpy.heightmap_get_interpolated_value(h_map, 0, 0)

    libtcodpy.heightmap_get_slope(h_map, 0, 0)
    libtcodpy.heightmap_get_normal(h_map, 0, 0, 0)
    libtcodpy.heightmap_count_cells(h_map, 0, 0)
    libtcodpy.heightmap_has_land_on_border(h_map, 0)
    libtcodpy.heightmap_get_minmax(h_map)

    libtcodpy.noise_delete(noise)
    libtcodpy.heightmap_delete(h_map)


MAP: NDArray[Any] = np.array(
    [
        list(line)
        for line in (
            "############",
            "#   ###    #",
            "#   ###    #",
            "#   ### ####",
            "## #### # ##",
            "##      ####",
            "############",
        )
    ]
)

MAP_HEIGHT, MAP_WIDTH = MAP.shape

POINT_A = (2, 2)
POINT_B = (9, 2)
POINT_C = (9, 4)

POINTS_AB = POINT_A + POINT_B  # valid path
POINTS_AC = POINT_A + POINT_C  # invalid path


@pytest.fixture()
def map_() -> Iterator[tcod.map.Map]:
    map_ = tcod.map.Map(MAP_WIDTH, MAP_HEIGHT)
    map_.walkable[...] = map_.transparent[...] = MAP[...] == " "
    yield map_
    libtcodpy.map_delete(map_)


@pytest.fixture()
def path_callback(map_: tcod.map.Map) -> Callable[[int, int, int, int, None], bool]:
    def callback(ox: int, oy: int, dx: int, dy: int, user_data: None) -> bool:
        if map_.walkable[dy, dx]:
            return True
        return False

    return callback


@pytest.mark.filterwarnings("ignore")
def test_map_fov(map_: tcod.map.Map) -> None:
    libtcodpy.map_compute_fov(map_, *POINT_A)


@pytest.mark.filterwarnings("ignore")
def test_astar(map_: tcod.map.Map) -> None:
    astar = libtcodpy.path_new_using_map(map_)

    assert not libtcodpy.path_compute(astar, *POINTS_AC)
    assert libtcodpy.path_size(astar) == 0
    assert libtcodpy.path_compute(astar, *POINTS_AB)
    assert libtcodpy.path_get_origin(astar) == POINT_A
    assert libtcodpy.path_get_destination(astar) == POINT_B
    libtcodpy.path_reverse(astar)
    assert libtcodpy.path_get_origin(astar) == POINT_B
    assert libtcodpy.path_get_destination(astar) == POINT_A

    assert libtcodpy.path_size(astar) != 0
    assert libtcodpy.path_size(astar) > 0
    assert not libtcodpy.path_is_empty(astar)

    x: Optional[int]
    y: Optional[int]

    for i in range(libtcodpy.path_size(astar)):
        x, y = libtcodpy.path_get(astar, i)

    while (x, y) != (None, None):
        x, y = libtcodpy.path_walk(astar, False)

    libtcodpy.path_delete(astar)


@pytest.mark.filterwarnings("ignore")
def test_astar_callback(map_: tcod.map.Map, path_callback: Callable[[int, int, int, int, Any], bool]) -> None:
    astar = libtcodpy.path_new_using_function(
        libtcodpy.map_get_width(map_),
        libtcodpy.map_get_height(map_),
        path_callback,
    )
    libtcodpy.path_compute(astar, *POINTS_AB)
    libtcodpy.path_delete(astar)


@pytest.mark.filterwarnings("ignore")
def test_dijkstra(map_: tcod.map.Map) -> None:
    path = libtcodpy.dijkstra_new(map_)

    libtcodpy.dijkstra_compute(path, *POINT_A)

    assert not libtcodpy.dijkstra_path_set(path, *POINT_C)
    assert libtcodpy.dijkstra_get_distance(path, *POINT_C) == -1

    assert libtcodpy.dijkstra_path_set(path, *POINT_B)
    assert libtcodpy.dijkstra_size(path)
    assert not libtcodpy.dijkstra_is_empty(path)

    libtcodpy.dijkstra_reverse(path)

    x: Optional[int]
    y: Optional[int]

    for i in range(libtcodpy.dijkstra_size(path)):
        x, y = libtcodpy.dijkstra_get(path, i)

    while (x, y) != (None, None):
        x, y = libtcodpy.dijkstra_path_walk(path)

    libtcodpy.dijkstra_delete(path)


@pytest.mark.filterwarnings("ignore")
def test_dijkstra_callback(map_: tcod.map.Map, path_callback: Callable[[int, int, int, int, Any], bool]) -> None:
    path = libtcodpy.dijkstra_new_using_function(
        libtcodpy.map_get_width(map_),
        libtcodpy.map_get_height(map_),
        path_callback,
    )
    libtcodpy.dijkstra_compute(path, *POINT_A)
    libtcodpy.dijkstra_delete(path)


@pytest.mark.filterwarnings("ignore")
def test_alpha_blend(console: tcod.console.Console) -> None:
    for i in range(256):
        libtcodpy.console_put_char(console, 0, 0, "x", libtcodpy.BKGND_ALPHA(i))
        libtcodpy.console_put_char(console, 0, 0, "x", libtcodpy.BKGND_ADDALPHA(i))


if __name__ == "__main__":
    pytest.main()
