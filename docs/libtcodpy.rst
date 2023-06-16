Old API Functions ``libtcodpy``
===============================

This is all the functions included since the start of the Python port.
This collection is often called :term:`libtcodpy`, the name of the original
Python port.  These functions are reproduced by python-tcod in their entirely.

Use ``from tcod import libtcodpy`` to access this module.

**A large majority of these functions are deprecated and will be removed in
the future.
In general this entire section should be avoided whenever possible.**
See :ref:`getting-started` for how to make a new python-tcod project with its
modern API.

bsp
---

.. autofunction:: libtcodpy.bsp_new_with_size
.. autofunction:: libtcodpy.bsp_split_once
.. autofunction:: libtcodpy.bsp_split_recursive
.. autofunction:: libtcodpy.bsp_resize
.. autofunction:: libtcodpy.bsp_left
.. autofunction:: libtcodpy.bsp_right
.. autofunction:: libtcodpy.bsp_father
.. autofunction:: libtcodpy.bsp_is_leaf
.. autofunction:: libtcodpy.bsp_contains
.. autofunction:: libtcodpy.bsp_find_node
.. autofunction:: libtcodpy.bsp_traverse_pre_order
.. autofunction:: libtcodpy.bsp_traverse_in_order
.. autofunction:: libtcodpy.bsp_traverse_post_order
.. autofunction:: libtcodpy.bsp_traverse_level_order
.. autofunction:: libtcodpy.bsp_traverse_inverted_level_order
.. autofunction:: libtcodpy.bsp_remove_sons
.. autofunction:: libtcodpy.bsp_delete

color
-----

.. autoclass:: libtcodpy.Color
   :member-order: bysource
   :members:

.. autofunction:: libtcodpy.color_lerp
.. autofunction:: libtcodpy.color_set_hsv
.. autofunction:: libtcodpy.color_get_hsv
.. autofunction:: libtcodpy.color_scale_HSV
.. autofunction:: libtcodpy.color_gen_map

color controls
~~~~~~~~~~~~~~
Libtcod color control constants.
These can be inserted into Python strings with the ``%c`` format specifier as shown below.

.. data:: libtcodpy.COLCTRL_1

    These can be configured with :any:`libtcodpy.console_set_color_control`.
    However, it is recommended to use :any:`libtcodpy.COLCTRL_FORE_RGB` and :any:`libtcodpy.COLCTRL_BACK_RGB` instead.

.. data:: libtcodpy.COLCTRL_2
.. data:: libtcodpy.COLCTRL_3
.. data:: libtcodpy.COLCTRL_4
.. data:: libtcodpy.COLCTRL_5

.. data:: libtcodpy.COLCTRL_STOP

    When this control character is inserted into a string the foreground and background colors will be reset for the
    remaining characters of the string.

    >>> import tcod
    >>> reset_color = f"{libtcodpy.COLCTRL_STOP:c}"

.. data:: libtcodpy.COLCTRL_FORE_RGB

    Sets the foreground color to the next 3 Unicode characters for the remaining characters.

    >>> fg = (255, 255, 255)
    >>> change_fg = f"{libtcodpy.COLCTRL_FORE_RGB:c}{fg[0]:c}{fg[1]:c}{fg[2]:c}"
    >>> string = f"Old color {change_fg}new color{libtcodpy.COLCTRL_STOP:c} old color."

.. data:: libtcodpy.COLCTRL_BACK_RGB

    Sets the background color to the next 3 Unicode characters for the remaining characters.

    >>> from typing import Tuple
    >>> def change_colors(fg: Tuple[int, int, int], bg: Tuple[int, int, int]) -> str:
    ...     """Return the control codes to change the foreground and background colors."""
    ...     return "%c%c%c%c%c%c%c%c" % (libtcodpy.COLCTRL_FORE_RGB, *fg, libtcodpy.COLCTRL_BACK_RGB, *bg)
    >>> string = f"Old {change_colors(fg=(255, 255, 255), bg=(0, 0, 255))}new"

console
-------

.. autofunction:: libtcodpy.console_set_custom_font
.. autofunction:: libtcodpy.console_init_root
.. autofunction:: libtcodpy.console_flush

.. autofunction:: libtcodpy.console_blit
.. autofunction:: libtcodpy.console_check_for_keypress
.. autofunction:: libtcodpy.console_clear
.. autofunction:: libtcodpy.console_credits
.. autofunction:: libtcodpy.console_credits_render
.. autofunction:: libtcodpy.console_credits_reset
.. autofunction:: libtcodpy.console_delete
.. autofunction:: libtcodpy.console_fill_background
.. autofunction:: libtcodpy.console_fill_char
.. autofunction:: libtcodpy.console_fill_foreground
.. autofunction:: libtcodpy.console_from_file
.. autofunction:: libtcodpy.console_from_xp
.. autofunction:: libtcodpy.console_get_alignment
.. autofunction:: libtcodpy.console_get_background_flag
.. autofunction:: libtcodpy.console_get_char
.. autofunction:: libtcodpy.console_get_char_background
.. autofunction:: libtcodpy.console_get_char_foreground
.. autofunction:: libtcodpy.console_get_default_background
.. autofunction:: libtcodpy.console_get_default_foreground
.. autofunction:: libtcodpy.console_get_fade
.. autofunction:: libtcodpy.console_get_fading_color
.. autofunction:: libtcodpy.console_get_height
.. autofunction:: libtcodpy.console_get_height_rect
.. autofunction:: libtcodpy.console_get_width
.. autofunction:: libtcodpy.console_hline
.. autofunction:: libtcodpy.console_is_fullscreen
.. autofunction:: libtcodpy.console_is_key_pressed
.. autofunction:: libtcodpy.console_is_window_closed
.. autofunction:: libtcodpy.console_load_apf
.. autofunction:: libtcodpy.console_load_asc
.. autofunction:: libtcodpy.console_load_xp
.. autofunction:: libtcodpy.console_list_load_xp
.. autofunction:: libtcodpy.console_list_save_xp
.. autofunction:: libtcodpy.console_map_ascii_code_to_font
.. autofunction:: libtcodpy.console_map_ascii_codes_to_font
.. autofunction:: libtcodpy.console_map_string_to_font
.. autofunction:: libtcodpy.console_new
.. autofunction:: libtcodpy.console_print
.. autofunction:: libtcodpy.console_print_ex
.. autofunction:: libtcodpy.console_print_frame
.. autofunction:: libtcodpy.console_print_rect
.. autofunction:: libtcodpy.console_print_rect_ex
.. autofunction:: libtcodpy.console_put_char
.. autofunction:: libtcodpy.console_put_char_ex
.. autofunction:: libtcodpy.console_rect
.. autofunction:: libtcodpy.console_save_apf
.. autofunction:: libtcodpy.console_save_asc
.. autofunction:: libtcodpy.console_save_xp
.. autofunction:: libtcodpy.console_set_alignment
.. autofunction:: libtcodpy.console_set_background_flag
.. autofunction:: libtcodpy.console_set_char
.. autofunction:: libtcodpy.console_set_char_background
.. autofunction:: libtcodpy.console_set_char_foreground
.. autofunction:: libtcodpy.console_set_color_control
.. autofunction:: libtcodpy.console_set_default_background
.. autofunction:: libtcodpy.console_set_default_foreground
.. autofunction:: libtcodpy.console_set_fade
.. autofunction:: libtcodpy.console_set_fullscreen
.. autofunction:: libtcodpy.console_set_key_color
.. autofunction:: libtcodpy.console_set_window_title
.. autofunction:: libtcodpy.console_vline
.. autofunction:: libtcodpy.console_wait_for_keypress

.. autoclass: libtcodpy.ConsoleBuffer
   :members:

Event
-----

.. autoclass:: libtcodpy.Key()
   :members:

.. autoclass:: libtcodpy.Mouse()
   :members:

.. _event types:

Event Types
~~~~~~~~~~~

.. data:: libtcodpy.EVENT_NONE
.. data:: libtcodpy.EVENT_KEY_PRESS
.. data:: libtcodpy.EVENT_KEY_RELEASE
.. data:: libtcodpy.EVENT_KEY

    Same as ``libtcodpy.EVENT_KEY_PRESS | libtcodpy.EVENT_KEY_RELEASE``

.. data:: libtcodpy.EVENT_MOUSE_MOVE
.. data:: libtcodpy.EVENT_MOUSE_PRESS
.. data:: libtcodpy.EVENT_MOUSE_RELEASE
.. data:: libtcodpy.EVENT_MOUSE

    Same as ``libtcodpy.EVENT_MOUSE_MOVE | libtcodpy.EVENT_MOUSE_PRESS | libtcodpy.EVENT_MOUSE_RELEASE``

.. data:: libtcodpy.EVENT_FINGER_MOVE
.. data:: libtcodpy.EVENT_FINGER_PRESS
.. data:: libtcodpy.EVENT_FINGER_RELEASE
.. data:: libtcodpy.EVENT_FINGER

    Same as ``libtcodpy.EVENT_FINGER_MOVE | libtcodpy.EVENT_FINGER_PRESS | libtcodpy.EVENT_FINGER_RELEASE``

.. data:: libtcodpy.EVENT_ANY

    Same as ``libtcodpy.EVENT_KEY | libtcodpy.EVENT_MOUSE | libtcodpy.EVENT_FINGER``

sys
---

.. autofunction:: libtcodpy.sys_set_fps
.. autofunction:: libtcodpy.sys_get_fps
.. autofunction:: libtcodpy.sys_get_last_frame_length
.. autofunction:: libtcodpy.sys_sleep_milli
.. autofunction:: libtcodpy.sys_elapsed_milli
.. autofunction:: libtcodpy.sys_elapsed_seconds
.. autofunction:: libtcodpy.sys_set_renderer
.. autofunction:: libtcodpy.sys_get_renderer
.. autofunction:: libtcodpy.sys_save_screenshot
.. autofunction:: libtcodpy.sys_force_fullscreen_resolution
.. autofunction:: libtcodpy.sys_get_current_resolution
.. autofunction:: libtcodpy.sys_get_char_size
.. autofunction:: libtcodpy.sys_update_char
.. autofunction:: libtcodpy.sys_register_SDL_renderer
.. autofunction:: libtcodpy.sys_check_for_event
.. autofunction:: libtcodpy.sys_wait_for_event

pathfinding
-----------

.. autofunction:: libtcodpy.dijkstra_compute
.. autofunction:: libtcodpy.dijkstra_delete
.. autofunction:: libtcodpy.dijkstra_get
.. autofunction:: libtcodpy.dijkstra_get_distance
.. autofunction:: libtcodpy.dijkstra_is_empty
.. autofunction:: libtcodpy.dijkstra_new
.. autofunction:: libtcodpy.dijkstra_new_using_function
.. autofunction:: libtcodpy.dijkstra_path_set
.. autofunction:: libtcodpy.dijkstra_path_walk
.. autofunction:: libtcodpy.dijkstra_reverse
.. autofunction:: libtcodpy.dijkstra_size

.. autofunction:: libtcodpy.path_compute
.. autofunction:: libtcodpy.path_delete
.. autofunction:: libtcodpy.path_get
.. autofunction:: libtcodpy.path_get_destination
.. autofunction:: libtcodpy.path_get_origin
.. autofunction:: libtcodpy.path_is_empty
.. autofunction:: libtcodpy.path_new_using_function
.. autofunction:: libtcodpy.path_new_using_map
.. autofunction:: libtcodpy.path_reverse
.. autofunction:: libtcodpy.path_size
.. autofunction:: libtcodpy.path_walk

heightmap
---------

.. autofunction:: libtcodpy.heightmap_add
.. autofunction:: libtcodpy.heightmap_add_fbm
.. autofunction:: libtcodpy.heightmap_add_hill
.. autofunction:: libtcodpy.heightmap_add_hm
.. autofunction:: libtcodpy.heightmap_add_voronoi
.. autofunction:: libtcodpy.heightmap_clamp
.. autofunction:: libtcodpy.heightmap_clear
.. autofunction:: libtcodpy.heightmap_copy
.. autofunction:: libtcodpy.heightmap_count_cells
.. autofunction:: libtcodpy.heightmap_delete
.. autofunction:: libtcodpy.heightmap_dig_bezier
.. autofunction:: libtcodpy.heightmap_dig_hill
.. autofunction:: libtcodpy.heightmap_get_interpolated_value
.. autofunction:: libtcodpy.heightmap_get_minmax
.. autofunction:: libtcodpy.heightmap_get_normal
.. autofunction:: libtcodpy.heightmap_get_slope
.. autofunction:: libtcodpy.heightmap_get_value
.. autofunction:: libtcodpy.heightmap_has_land_on_border
.. autofunction:: libtcodpy.heightmap_kernel_transform
.. autofunction:: libtcodpy.heightmap_lerp_hm
.. autofunction:: libtcodpy.heightmap_multiply_hm
.. autofunction:: libtcodpy.heightmap_new
.. autofunction:: libtcodpy.heightmap_normalize
.. autofunction:: libtcodpy.heightmap_rain_erosion
.. autofunction:: libtcodpy.heightmap_scale
.. autofunction:: libtcodpy.heightmap_scale_fbm
.. autofunction:: libtcodpy.heightmap_set_value

image
-----

.. autofunction:: libtcodpy.image_load
.. autofunction:: libtcodpy.image_from_console

.. autofunction:: libtcodpy.image_blit
.. autofunction:: libtcodpy.image_blit_2x
.. autofunction:: libtcodpy.image_blit_rect
.. autofunction:: libtcodpy.image_clear
.. autofunction:: libtcodpy.image_delete
.. autofunction:: libtcodpy.image_get_alpha
.. autofunction:: libtcodpy.image_get_mipmap_pixel
.. autofunction:: libtcodpy.image_get_pixel
.. autofunction:: libtcodpy.image_get_size
.. autofunction:: libtcodpy.image_hflip
.. autofunction:: libtcodpy.image_invert
.. autofunction:: libtcodpy.image_is_pixel_transparent
.. autofunction:: libtcodpy.image_new
.. autofunction:: libtcodpy.image_put_pixel
.. autofunction:: libtcodpy.image_refresh_console
.. autofunction:: libtcodpy.image_rotate90
.. autofunction:: libtcodpy.image_save
.. autofunction:: libtcodpy.image_scale
.. autofunction:: libtcodpy.image_set_key_color
.. autofunction:: libtcodpy.image_vflip

line
----

.. autofunction:: libtcodpy.line_init
.. autofunction:: libtcodpy.line_step
.. autofunction:: libtcodpy.line
.. autofunction:: libtcodpy.line_iter
.. autofunction:: libtcodpy.line_where

map
---

.. autofunction:: libtcodpy.map_clear
.. autofunction:: libtcodpy.map_compute_fov
.. autofunction:: libtcodpy.map_copy
.. autofunction:: libtcodpy.map_delete
.. autofunction:: libtcodpy.map_get_height
.. autofunction:: libtcodpy.map_get_width
.. autofunction:: libtcodpy.map_is_in_fov
.. autofunction:: libtcodpy.map_is_transparent
.. autofunction:: libtcodpy.map_is_walkable
.. autofunction:: libtcodpy.map_new
.. autofunction:: libtcodpy.map_set_properties

mouse
-----

.. autofunction:: libtcodpy.mouse_get_status
.. autofunction:: libtcodpy.mouse_is_cursor_visible
.. autofunction:: libtcodpy.mouse_move
.. autofunction:: libtcodpy.mouse_show_cursor

namegen
-------

.. autofunction:: libtcodpy.namegen_destroy
.. autofunction:: libtcodpy.namegen_generate
.. autofunction:: libtcodpy.namegen_generate_custom
.. autofunction:: libtcodpy.namegen_get_sets
.. autofunction:: libtcodpy.namegen_parse

noise
-----

.. autofunction:: libtcodpy.noise_delete
.. autofunction:: libtcodpy.noise_get
.. autofunction:: libtcodpy.noise_get_fbm
.. autofunction:: libtcodpy.noise_get_turbulence
.. autofunction:: libtcodpy.noise_new
.. autofunction:: libtcodpy.noise_set_type

parser
------

.. autofunction:: libtcodpy.parser_delete
.. autofunction:: libtcodpy.parser_get_bool_property
.. autofunction:: libtcodpy.parser_get_char_property
.. autofunction:: libtcodpy.parser_get_color_property
.. autofunction:: libtcodpy.parser_get_dice_property
.. autofunction:: libtcodpy.parser_get_float_property
.. autofunction:: libtcodpy.parser_get_int_property
.. autofunction:: libtcodpy.parser_get_list_property
.. autofunction:: libtcodpy.parser_get_string_property
.. autofunction:: libtcodpy.parser_new
.. autofunction:: libtcodpy.parser_new_struct
.. autofunction:: libtcodpy.parser_run

random
------

.. autofunction:: libtcodpy.random_delete
.. autofunction:: libtcodpy.random_get_double
.. autofunction:: libtcodpy.random_get_double_mean
.. autofunction:: libtcodpy.random_get_float
.. autofunction:: libtcodpy.random_get_float_mean
.. autofunction:: libtcodpy.random_get_instance
.. autofunction:: libtcodpy.random_get_int
.. autofunction:: libtcodpy.random_get_int_mean
.. autofunction:: libtcodpy.random_new
.. autofunction:: libtcodpy.random_new_from_seed
.. autofunction:: libtcodpy.random_restore
.. autofunction:: libtcodpy.random_save
.. autofunction:: libtcodpy.random_set_distribution

struct
------
.. autofunction:: libtcodpy.struct_add_flag
.. autofunction:: libtcodpy.struct_add_list_property
.. autofunction:: libtcodpy.struct_add_property
.. autofunction:: libtcodpy.struct_add_structure
.. autofunction:: libtcodpy.struct_add_value_list
.. autofunction:: libtcodpy.struct_get_name
.. autofunction:: libtcodpy.struct_get_type
.. autofunction:: libtcodpy.struct_is_mandatory

other
-----

.. autoclass:: libtcodpy.ConsoleBuffer
    :members:

.. autoclass:: libtcodpy.Dice(nb_dices=0, nb_faces=0, multiplier=0, addsub=0)
   :members:
