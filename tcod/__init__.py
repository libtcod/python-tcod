"""
    This module provides a simple CFFI API to libtcod.

    This port has large partial support for libtcod's C functions.
    Use tcod/libtcod_cdef.h in the source distribution to see specially what
    functions were exported and what new functions have been added by TDL.

    The ffi and lib variables should be familiar to anyone that has used CFFI
    before, otherwise it's time to read up on how they work:
    https://cffi.readthedocs.org/en/latest/using.html

    Otherwise this module can be used as a drop in replacement for the official
    libtcod.py module.

    Bring any issues or requests to GitHub:
    https://github.com/HexDecimal/libtcod-cffi
"""
from tcod.loader import lib, ffi, __sdl_version__  # noqa: F4
from tcod.libtcodpy import *  # noqa: F4

try:
    from tcod.version import __version__
except ImportError:  # Gets imported without version.py by ReadTheDocs
    __version__ = ""

__all__ = [  # noqa: F405
    "__version__",
    # --- From libtcodpy.py ---
    "Color",
    "Bsp",
    "NB_FOV_ALGORITHMS",
    "NOISE_DEFAULT_HURST",
    "NOISE_DEFAULT_LACUNARITY",
    "ConsoleBuffer",
    "Dice",
    "Key",
    "Mouse",
    "FOV_PERMISSIVE",
    "BKGND_ALPHA",
    "BKGND_ADDALPHA",
    "bsp_new_with_size",
    "bsp_split_once",
    "bsp_split_recursive",
    "bsp_resize",
    "bsp_left",
    "bsp_right",
    "bsp_father",
    "bsp_is_leaf",
    "bsp_contains",
    "bsp_find_node",
    "bsp_traverse_pre_order",
    "bsp_traverse_in_order",
    "bsp_traverse_post_order",
    "bsp_traverse_level_order",
    "bsp_traverse_inverted_level_order",
    "bsp_remove_sons",
    "bsp_delete",
    "color_lerp",
    "color_set_hsv",
    "color_get_hsv",
    "color_scale_HSV",
    "color_gen_map",
    "console_init_root",
    "console_set_custom_font",
    "console_get_width",
    "console_get_height",
    "console_map_ascii_code_to_font",
    "console_map_ascii_codes_to_font",
    "console_map_string_to_font",
    "console_is_fullscreen",
    "console_set_fullscreen",
    "console_is_window_closed",
    "console_has_mouse_focus",
    "console_is_active",
    "console_set_window_title",
    "console_credits",
    "console_credits_reset",
    "console_credits_render",
    "console_flush",
    "console_set_default_background",
    "console_set_default_foreground",
    "console_clear",
    "console_put_char",
    "console_put_char_ex",
    "console_set_char_background",
    "console_set_char_foreground",
    "console_set_char",
    "console_set_background_flag",
    "console_get_background_flag",
    "console_set_alignment",
    "console_get_alignment",
    "console_print",
    "console_print_ex",
    "console_print_rect",
    "console_print_rect_ex",
    "console_get_height_rect",
    "console_rect",
    "console_hline",
    "console_vline",
    "console_print_frame",
    "console_set_color_control",
    "console_get_default_background",
    "console_get_default_foreground",
    "console_get_char_background",
    "console_get_char_foreground",
    "console_get_char",
    "console_set_fade",
    "console_get_fade",
    "console_get_fading_color",
    "console_wait_for_keypress",
    "console_check_for_keypress",
    "console_is_key_pressed",
    "console_new",
    "console_from_file",
    "console_blit",
    "console_set_key_color",
    "console_delete",
    "console_fill_foreground",
    "console_fill_background",
    "console_fill_char",
    "console_load_asc",
    "console_save_asc",
    "console_load_apf",
    "console_save_apf",
    "console_load_xp",
    "console_save_xp",
    "console_from_xp",
    "console_list_load_xp",
    "console_list_save_xp",
    "path_new_using_map",
    "path_new_using_function",
    "path_compute",
    "path_get_origin",
    "path_get_destination",
    "path_size",
    "path_reverse",
    "path_get",
    "path_is_empty",
    "path_walk",
    "path_delete",
    "dijkstra_new",
    "dijkstra_new_using_function",
    "dijkstra_compute",
    "dijkstra_path_set",
    "dijkstra_get_distance",
    "dijkstra_size",
    "dijkstra_reverse",
    "dijkstra_get",
    "dijkstra_is_empty",
    "dijkstra_path_walk",
    "dijkstra_delete",
    "heightmap_new",
    "heightmap_set_value",
    "heightmap_add",
    "heightmap_scale",
    "heightmap_clear",
    "heightmap_clamp",
    "heightmap_copy",
    "heightmap_normalize",
    "heightmap_lerp_hm",
    "heightmap_add_hm",
    "heightmap_multiply_hm",
    "heightmap_add_hill",
    "heightmap_dig_hill",
    "heightmap_rain_erosion",
    "heightmap_kernel_transform",
    "heightmap_add_voronoi",
    "heightmap_add_fbm",
    "heightmap_scale_fbm",
    "heightmap_dig_bezier",
    "heightmap_get_value",
    "heightmap_get_interpolated_value",
    "heightmap_get_slope",
    "heightmap_get_normal",
    "heightmap_count_cells",
    "heightmap_has_land_on_border",
    "heightmap_get_minmax",
    "heightmap_delete",
    "image_new",
    "image_clear",
    "image_invert",
    "image_hflip",
    "image_rotate90",
    "image_vflip",
    "image_scale",
    "image_set_key_color",
    "image_get_alpha",
    "image_is_pixel_transparent",
    "image_load",
    "image_from_console",
    "image_refresh_console",
    "image_get_size",
    "image_get_pixel",
    "image_get_mipmap_pixel",
    "image_put_pixel",
    "image_blit",
    "image_blit_rect",
    "image_blit_2x",
    "image_save",
    "image_delete",
    "line_init",
    "line_step",
    "line",
    "line_iter",
    "line_where",
    "map_new",
    "map_copy",
    "map_set_properties",
    "map_clear",
    "map_compute_fov",
    "map_is_in_fov",
    "map_is_transparent",
    "map_is_walkable",
    "map_delete",
    "map_get_width",
    "map_get_height",
    "mouse_show_cursor",
    "mouse_is_cursor_visible",
    "mouse_move",
    "mouse_get_status",
    "namegen_parse",
    "namegen_generate",
    "namegen_generate_custom",
    "namegen_get_sets",
    "namegen_destroy",
    "noise_new",
    "noise_set_type",
    "noise_get",
    "noise_get_fbm",
    "noise_get_turbulence",
    "noise_delete",
    "parser_new",
    "parser_new_struct",
    "parser_run",
    "parser_delete",
    "parser_get_bool_property",
    "parser_get_int_property",
    "parser_get_char_property",
    "parser_get_float_property",
    "parser_get_string_property",
    "parser_get_color_property",
    "parser_get_dice_property",
    "parser_get_list_property",
    "random_get_instance",
    "random_new",
    "random_new_from_seed",
    "random_set_distribution",
    "random_get_int",
    "random_get_float",
    "random_get_double",
    "random_get_int_mean",
    "random_get_float_mean",
    "random_get_double_mean",
    "random_save",
    "random_restore",
    "random_delete",
    "struct_add_flag",
    "struct_add_property",
    "struct_add_value_list",
    "struct_add_list_property",
    "struct_add_structure",
    "struct_get_name",
    "struct_is_mandatory",
    "struct_get_type",
    "sys_set_fps",
    "sys_get_fps",
    "sys_get_last_frame_length",
    "sys_sleep_milli",
    "sys_elapsed_milli",
    "sys_elapsed_seconds",
    "sys_set_renderer",
    "sys_get_renderer",
    "sys_save_screenshot",
    "sys_force_fullscreen_resolution",
    "sys_get_current_resolution",
    "sys_get_char_size",
    "sys_update_char",
    "sys_register_SDL_renderer",
    "sys_check_for_event",
    "sys_wait_for_event",
    "sys_clipboard_set",
    "sys_clipboard_get",
    # --- From constants.py ---
    "FOV_BASIC",
    "FOV_DIAMOND",
    "FOV_PERMISSIVE_0",
    "FOV_PERMISSIVE_1",
    "FOV_PERMISSIVE_2",
    "FOV_PERMISSIVE_3",
    "FOV_PERMISSIVE_4",
    "FOV_PERMISSIVE_5",
    "FOV_PERMISSIVE_6",
    "FOV_PERMISSIVE_7",
    "FOV_PERMISSIVE_8",
    "FOV_RESTRICTIVE",
    "FOV_SHADOW",
    "KEY_0",
    "KEY_1",
    "KEY_2",
    "KEY_3",
    "KEY_4",
    "KEY_5",
    "KEY_6",
    "KEY_7",
    "KEY_8",
    "KEY_9",
    "KEY_ALT",
    "KEY_APPS",
    "KEY_BACKSPACE",
    "KEY_CAPSLOCK",
    "KEY_CHAR",
    "KEY_CONTROL",
    "KEY_DELETE",
    "KEY_DOWN",
    "KEY_END",
    "KEY_ENTER",
    "KEY_ESCAPE",
    "KEY_F1",
    "KEY_F10",
    "KEY_F11",
    "KEY_F12",
    "KEY_F2",
    "KEY_F3",
    "KEY_F4",
    "KEY_F5",
    "KEY_F6",
    "KEY_F7",
    "KEY_F8",
    "KEY_F9",
    "KEY_HOME",
    "KEY_INSERT",
    "KEY_KP0",
    "KEY_KP1",
    "KEY_KP2",
    "KEY_KP3",
    "KEY_KP4",
    "KEY_KP5",
    "KEY_KP6",
    "KEY_KP7",
    "KEY_KP8",
    "KEY_KP9",
    "KEY_KPADD",
    "KEY_KPDEC",
    "KEY_KPDIV",
    "KEY_KPENTER",
    "KEY_KPMUL",
    "KEY_KPSUB",
    "KEY_LEFT",
    "KEY_LWIN",
    "KEY_NONE",
    "KEY_NUMLOCK",
    "KEY_PAGEDOWN",
    "KEY_PAGEUP",
    "KEY_PAUSE",
    "KEY_PRINTSCREEN",
    "KEY_RIGHT",
    "KEY_RWIN",
    "KEY_SCROLLLOCK",
    "KEY_SHIFT",
    "KEY_SPACE",
    "KEY_TAB",
    "KEY_TEXT",
    "KEY_UP",
    "BKGND_ADD",
    "BKGND_ADDA",
    "BKGND_ALPH",
    "BKGND_BURN",
    "BKGND_COLOR_BURN",
    "BKGND_COLOR_DODGE",
    "BKGND_DARKEN",
    "BKGND_DEFAULT",
    "BKGND_LIGHTEN",
    "BKGND_MULTIPLY",
    "BKGND_NONE",
    "BKGND_OVERLAY",
    "BKGND_SCREEN",
    "BKGND_SET",
    "CENTER",
    "CHAR_ARROW2_E",
    "CHAR_ARROW2_N",
    "CHAR_ARROW2_S",
    "CHAR_ARROW2_W",
    "CHAR_ARROW_E",
    "CHAR_ARROW_N",
    "CHAR_ARROW_S",
    "CHAR_ARROW_W",
    "CHAR_BLOCK1",
    "CHAR_BLOCK2",
    "CHAR_BLOCK3",
    "CHAR_BULLET",
    "CHAR_BULLET_INV",
    "CHAR_BULLET_SQUARE",
    "CHAR_CENT",
    "CHAR_CHECKBOX_SET",
    "CHAR_CHECKBOX_UNSET",
    "CHAR_CLUB",
    "CHAR_COPYRIGHT",
    "CHAR_CROSS",
    "CHAR_CURRENCY",
    "CHAR_DARROW_H",
    "CHAR_DARROW_V",
    "CHAR_DCROSS",
    "CHAR_DHLINE",
    "CHAR_DIAMOND",
    "CHAR_DIVISION",
    "CHAR_DNE",
    "CHAR_DNW",
    "CHAR_DSE",
    "CHAR_DSW",
    "CHAR_DTEEE",
    "CHAR_DTEEN",
    "CHAR_DTEES",
    "CHAR_DTEEW",
    "CHAR_DVLINE",
    "CHAR_EXCLAM_DOUBLE",
    "CHAR_FEMALE",
    "CHAR_FUNCTION",
    "CHAR_GRADE",
    "CHAR_HALF",
    "CHAR_HEART",
    "CHAR_HLINE",
    "CHAR_LIGHT",
    "CHAR_MALE",
    "CHAR_MULTIPLICATION",
    "CHAR_NE",
    "CHAR_NOTE",
    "CHAR_NOTE_DOUBLE",
    "CHAR_NW",
    "CHAR_ONE_QUARTER",
    "CHAR_PILCROW",
    "CHAR_POUND",
    "CHAR_POW1",
    "CHAR_POW2",
    "CHAR_POW3",
    "CHAR_RADIO_SET",
    "CHAR_RADIO_UNSET",
    "CHAR_RESERVED",
    "CHAR_SE",
    "CHAR_SECTION",
    "CHAR_SMILIE",
    "CHAR_SMILIE_INV",
    "CHAR_SPADE",
    "CHAR_SUBP_DIAG",
    "CHAR_SUBP_E",
    "CHAR_SUBP_N",
    "CHAR_SUBP_NE",
    "CHAR_SUBP_NW",
    "CHAR_SUBP_SE",
    "CHAR_SUBP_SW",
    "CHAR_SW",
    "CHAR_TEEE",
    "CHAR_TEEN",
    "CHAR_TEES",
    "CHAR_TEEW",
    "CHAR_THREE_QUARTERS",
    "CHAR_UMLAUT",
    "CHAR_VLINE",
    "CHAR_YEN",
    "COLCTRL_1",
    "COLCTRL_2",
    "COLCTRL_3",
    "COLCTRL_4",
    "COLCTRL_5",
    "COLCTRL_BACK_RGB",
    "COLCTRL_FORE_RGB",
    "COLCTRL_NUMBER",
    "COLCTRL_STOP",
    "COLOR_AMBER",
    "COLOR_AZURE",
    "COLOR_BLUE",
    "COLOR_CHARTREUSE",
    "COLOR_CRIMSON",
    "COLOR_CYAN",
    "COLOR_DARK",
    "COLOR_DARKER",
    "COLOR_DARKEST",
    "COLOR_DESATURATED",
    "COLOR_FLAME",
    "COLOR_FUCHSIA",
    "COLOR_GREEN",
    "COLOR_HAN",
    "COLOR_LEVELS",
    "COLOR_LIGHT",
    "COLOR_LIGHTER",
    "COLOR_LIGHTEST",
    "COLOR_LIME",
    "COLOR_MAGENTA",
    "COLOR_NB",
    "COLOR_NORMAL",
    "COLOR_ORANGE",
    "COLOR_PINK",
    "COLOR_PURPLE",
    "COLOR_RED",
    "COLOR_SEA",
    "COLOR_SKY",
    "COLOR_TURQUOISE",
    "COLOR_VIOLET",
    "COLOR_YELLOW",
    "DISTRIBUTION_GAUSSIAN",
    "DISTRIBUTION_GAUSSIAN_INVERSE",
    "DISTRIBUTION_GAUSSIAN_RANGE",
    "DISTRIBUTION_GAUSSIAN_RANGE_INVERSE",
    "DISTRIBUTION_LINEAR",
    "EVENT_ANY",
    "EVENT_FINGER",
    "EVENT_FINGER_MOVE",
    "EVENT_FINGER_PRESS",
    "EVENT_FINGER_RELEASE",
    "EVENT_KEY",
    "EVENT_KEY_PRESS",
    "EVENT_KEY_RELEASE",
    "EVENT_MOUSE",
    "EVENT_MOUSE_MOVE",
    "EVENT_MOUSE_PRESS",
    "EVENT_MOUSE_RELEASE",
    "EVENT_NONE",
    "FONT_LAYOUT_ASCII_INCOL",
    "FONT_LAYOUT_ASCII_INROW",
    "FONT_LAYOUT_CP437",
    "FONT_LAYOUT_TCOD",
    "FONT_TYPE_GRAYSCALE",
    "FONT_TYPE_GREYSCALE",
    "KEY_PRESSED",
    "KEY_RELEASED",
    "LEFT",
    "NB_RENDERERS",
    "NOISE_DEFAULT",
    "NOISE_PERLIN",
    "NOISE_SIMPLEX",
    "NOISE_WAVELET",
    "RENDERER_GLSL",
    "RENDERER_OPENGL",
    "RENDERER_OPENGL2",
    "RENDERER_SDL",
    "RENDERER_SDL2",
    "RIGHT",
    "RNG_CMWC",
    "RNG_MT",
    "TYPE_BOOL",
    "TYPE_CHAR",
    "TYPE_COLOR",
    "TYPE_CUSTOM00",
    "TYPE_CUSTOM01",
    "TYPE_CUSTOM02",
    "TYPE_CUSTOM03",
    "TYPE_CUSTOM04",
    "TYPE_CUSTOM05",
    "TYPE_CUSTOM06",
    "TYPE_CUSTOM07",
    "TYPE_CUSTOM08",
    "TYPE_CUSTOM09",
    "TYPE_CUSTOM10",
    "TYPE_CUSTOM11",
    "TYPE_CUSTOM12",
    "TYPE_CUSTOM13",
    "TYPE_CUSTOM14",
    "TYPE_CUSTOM15",
    "TYPE_DICE",
    "TYPE_FLOAT",
    "TYPE_INT",
    "TYPE_LIST",
    "TYPE_NONE",
    "TYPE_STRING",
    "TYPE_VALUELIST00",
    "TYPE_VALUELIST01",
    "TYPE_VALUELIST02",
    "TYPE_VALUELIST03",
    "TYPE_VALUELIST04",
    "TYPE_VALUELIST05",
    "TYPE_VALUELIST06",
    "TYPE_VALUELIST07",
    "TYPE_VALUELIST08",
    "TYPE_VALUELIST09",
    "TYPE_VALUELIST10",
    "TYPE_VALUELIST11",
    "TYPE_VALUELIST12",
    "TYPE_VALUELIST13",
    "TYPE_VALUELIST14",
    "TYPE_VALUELIST15",
    "amber",
    "azure",
    "black",
    "blue",
    "brass",
    "celadon",
    "chartreuse",
    "copper",
    "crimson",
    "cyan",
    "dark_amber",
    "dark_azure",
    "dark_blue",
    "dark_chartreuse",
    "dark_crimson",
    "dark_cyan",
    "dark_flame",
    "dark_fuchsia",
    "dark_gray",
    "dark_green",
    "dark_grey",
    "dark_han",
    "dark_lime",
    "dark_magenta",
    "dark_orange",
    "dark_pink",
    "dark_purple",
    "dark_red",
    "dark_sea",
    "dark_sepia",
    "dark_sky",
    "dark_turquoise",
    "dark_violet",
    "dark_yellow",
    "darker_amber",
    "darker_azure",
    "darker_blue",
    "darker_chartreuse",
    "darker_crimson",
    "darker_cyan",
    "darker_flame",
    "darker_fuchsia",
    "darker_gray",
    "darker_green",
    "darker_grey",
    "darker_han",
    "darker_lime",
    "darker_magenta",
    "darker_orange",
    "darker_pink",
    "darker_purple",
    "darker_red",
    "darker_sea",
    "darker_sepia",
    "darker_sky",
    "darker_turquoise",
    "darker_violet",
    "darker_yellow",
    "darkest_amber",
    "darkest_azure",
    "darkest_blue",
    "darkest_chartreuse",
    "darkest_crimson",
    "darkest_cyan",
    "darkest_flame",
    "darkest_fuchsia",
    "darkest_gray",
    "darkest_green",
    "darkest_grey",
    "darkest_han",
    "darkest_lime",
    "darkest_magenta",
    "darkest_orange",
    "darkest_pink",
    "darkest_purple",
    "darkest_red",
    "darkest_sea",
    "darkest_sepia",
    "darkest_sky",
    "darkest_turquoise",
    "darkest_violet",
    "darkest_yellow",
    "desaturated_amber",
    "desaturated_azure",
    "desaturated_blue",
    "desaturated_chartreuse",
    "desaturated_crimson",
    "desaturated_cyan",
    "desaturated_flame",
    "desaturated_fuchsia",
    "desaturated_green",
    "desaturated_han",
    "desaturated_lime",
    "desaturated_magenta",
    "desaturated_orange",
    "desaturated_pink",
    "desaturated_purple",
    "desaturated_red",
    "desaturated_sea",
    "desaturated_sky",
    "desaturated_turquoise",
    "desaturated_violet",
    "desaturated_yellow",
    "flame",
    "fuchsia",
    "gold",
    "gray",
    "green",
    "grey",
    "han",
    "light_amber",
    "light_azure",
    "light_blue",
    "light_chartreuse",
    "light_crimson",
    "light_cyan",
    "light_flame",
    "light_fuchsia",
    "light_gray",
    "light_green",
    "light_grey",
    "light_han",
    "light_lime",
    "light_magenta",
    "light_orange",
    "light_pink",
    "light_purple",
    "light_red",
    "light_sea",
    "light_sepia",
    "light_sky",
    "light_turquoise",
    "light_violet",
    "light_yellow",
    "lighter_amber",
    "lighter_azure",
    "lighter_blue",
    "lighter_chartreuse",
    "lighter_crimson",
    "lighter_cyan",
    "lighter_flame",
    "lighter_fuchsia",
    "lighter_gray",
    "lighter_green",
    "lighter_grey",
    "lighter_han",
    "lighter_lime",
    "lighter_magenta",
    "lighter_orange",
    "lighter_pink",
    "lighter_purple",
    "lighter_red",
    "lighter_sea",
    "lighter_sepia",
    "lighter_sky",
    "lighter_turquoise",
    "lighter_violet",
    "lighter_yellow",
    "lightest_amber",
    "lightest_azure",
    "lightest_blue",
    "lightest_chartreuse",
    "lightest_crimson",
    "lightest_cyan",
    "lightest_flame",
    "lightest_fuchsia",
    "lightest_gray",
    "lightest_green",
    "lightest_grey",
    "lightest_han",
    "lightest_lime",
    "lightest_magenta",
    "lightest_orange",
    "lightest_pink",
    "lightest_purple",
    "lightest_red",
    "lightest_sea",
    "lightest_sepia",
    "lightest_sky",
    "lightest_turquoise",
    "lightest_violet",
    "lightest_yellow",
    "lime",
    "magenta",
    "orange",
    "peach",
    "pink",
    "purple",
    "red",
    "sea",
    "sepia",
    "silver",
    "sky",
    "turquoise",
    "violet",
    "white",
    "yellow",
]
