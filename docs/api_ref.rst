
API Reference
=============

Console
-------

.. autofunction:: tcod.console_set_custom_font
.. autofunction:: tcod.console_init_root
.. autofunction:: tcod.console_flush

.. autoclass:: tcod.Console(width, height)
    :members:


BSP
---

.. autoclass:: tcod.BSP(x, y, w, h)
   :members:

HeightMap
---------

.. autoclass:: tcod.HeightMap(x, y)
   :members:

Image
-----

.. autoclass:: tcod.Image(width, height)
    :members:

.. autofunction:: tcod.image_load
.. autofunction:: tcod.image_from_console

Event
-----

.. autoclass:: tcod.Key()
   :members:

.. autoclass:: tcod.Mouse()
   :members:

.. autofunction:: tcod.clipboard_set
.. autofunction:: tcod.clipboard_get

.. _event types:

Event Types
~~~~~~~~~~~

.. data:: tcod.EVENT_NONE
.. data:: tcod.EVENT_KEY_PRESS
.. data:: tcod.EVENT_KEY_RELEASE
.. data:: tcod.EVENT_KEY

    Same as ``tcod.EVENT_KEY_PRESS | tcod.EVENT_KEY_RELEASE``

.. data:: tcod.EVENT_MOUSE_MOVE
.. data:: tcod.EVENT_MOUSE_PRESS
.. data:: tcod.EVENT_MOUSE_RELEASE
.. data:: tcod.EVENT_MOUSE

    Same as ``tcod.EVENT_MOUSE_MOVE | tcod.EVENT_MOUSE_PRESS | tcod.EVENT_MOUSE_RELEASE``

.. data:: tcod.EVENT_FINGER_MOVE
.. data:: tcod.EVENT_FINGER_PRESS
.. data:: tcod.EVENT_FINGER_RELEASE
.. data:: tcod.EVENT_FINGER

    Same as ``tcod.EVENT_FINGER_MOVE | tcod.EVENT_FINGER_PRESS | tcod.EVENT_FINGER_RELEASE``

.. data:: tcod.EVENT_ANY

    Same as ``tcod.EVENT_KEY | tcod.EVENT_MOUSE | tcod.EVENT_FINGER``

bsp
---

.. autofunction:: tcod.bsp_new_with_size
.. autofunction:: tcod.bsp_split_once
.. autofunction:: tcod.bsp_split_recursive
.. autofunction:: tcod.bsp_resize
.. autofunction:: tcod.bsp_left
.. autofunction:: tcod.bsp_right
.. autofunction:: tcod.bsp_father
.. autofunction:: tcod.bsp_is_leaf
.. autofunction:: tcod.bsp_contains
.. autofunction:: tcod.bsp_find_node
.. autofunction:: tcod.bsp_traverse_pre_order
.. autofunction:: tcod.bsp_traverse_in_order
.. autofunction:: tcod.bsp_traverse_post_order
.. autofunction:: tcod.bsp_traverse_level_order
.. autofunction:: tcod.bsp_traverse_inverted_level_order
.. autofunction:: tcod.bsp_remove_sons
.. autofunction:: tcod.bsp_delete

color
-----

.. autoclass:: tcod.Color
   :members:

.. autofunction:: tcod.color_lerp
.. autofunction:: tcod.color_set_hsv
.. autofunction:: tcod.color_get_hsv
.. autofunction:: tcod.color_scale_HSV
.. autofunction:: tcod.color_gen_map

console
-------

.. autoclass: tcod.ConsoleBuffer
   :members:

.. autofunction:: tcod.console_blit
.. autofunction:: tcod.console_check_for_keypress
.. autofunction:: tcod.console_clear
.. autofunction:: tcod.console_credits
.. autofunction:: tcod.console_credits_render
.. autofunction:: tcod.console_credits_reset
.. autofunction:: tcod.console_delete
.. autofunction:: tcod.console_disable_keyboard_repeat
.. autofunction:: tcod.console_fill_background
.. autofunction:: tcod.console_fill_char
.. autofunction:: tcod.console_fill_foreground
.. autofunction:: tcod.console_from_file
.. autofunction:: tcod.console_get_alignment
.. autofunction:: tcod.console_get_background_flag
.. autofunction:: tcod.console_get_char
.. autofunction:: tcod.console_get_char_background
.. autofunction:: tcod.console_get_char_foreground
.. autofunction:: tcod.console_get_default_background
.. autofunction:: tcod.console_get_default_foreground
.. autofunction:: tcod.console_get_fade
.. autofunction:: tcod.console_get_fading_color
.. autofunction:: tcod.console_get_height
.. autofunction:: tcod.console_get_height_rect
.. autofunction:: tcod.console_get_width
.. autofunction:: tcod.console_hline
.. autofunction:: tcod.console_is_fullscreen
.. autofunction:: tcod.console_is_key_pressed
.. autofunction:: tcod.console_is_window_closed
.. autofunction:: tcod.console_load_apf
.. autofunction:: tcod.console_load_asc
.. autofunction:: tcod.console_map_ascii_code_to_font
.. autofunction:: tcod.console_map_ascii_codes_to_font
.. autofunction:: tcod.console_map_string_to_font
.. autofunction:: tcod.console_new
.. autofunction:: tcod.console_print
.. autofunction:: tcod.console_print_ex
.. autofunction:: tcod.console_print_frame
.. autofunction:: tcod.console_print_rect
.. autofunction:: tcod.console_print_rect_ex
.. autofunction:: tcod.console_put_char
.. autofunction:: tcod.console_put_char_ex
.. autofunction:: tcod.console_rect
.. autofunction:: tcod.console_save_apf
.. autofunction:: tcod.console_save_asc
.. autofunction:: tcod.console_set_alignment
.. autofunction:: tcod.console_set_background_flag
.. autofunction:: tcod.console_set_char
.. autofunction:: tcod.console_set_char_background
.. autofunction:: tcod.console_set_char_foreground
.. autofunction:: tcod.console_set_color_control
.. autofunction:: tcod.console_set_default_background
.. autofunction:: tcod.console_set_default_foreground
.. autofunction:: tcod.console_set_fade
.. autofunction:: tcod.console_set_fullscreen
.. autofunction:: tcod.console_set_key_color
.. autofunction:: tcod.console_set_keyboard_repeat
.. autofunction:: tcod.console_set_window_title
.. autofunction:: tcod.console_vline
.. autofunction:: tcod.console_wait_for_keypress

sys
---

.. autofunction:: tcod.sys_set_fps
.. autofunction:: tcod.sys_get_fps
.. autofunction:: tcod.sys_get_last_frame_length
.. autofunction:: tcod.sys_sleep_milli
.. autofunction:: tcod.sys_elapsed_milli
.. autofunction:: tcod.sys_elapsed_seconds
.. autofunction:: tcod.sys_set_renderer
.. autofunction:: tcod.sys_get_renderer
.. autofunction:: tcod.sys_save_screenshot
.. autofunction:: tcod.sys_force_fullscreen_resolution
.. autofunction:: tcod.sys_get_current_resolution
.. autofunction:: tcod.sys_get_char_size
.. autofunction:: tcod.sys_update_char
.. autofunction:: tcod.sys_register_SDL_renderer
.. autofunction:: tcod.sys_check_for_event
.. autofunction:: tcod.sys_wait_for_event

pathfinding
-----------

.. autofunction:: tcod.dijkstra_compute
.. autofunction:: tcod.dijkstra_delete
.. autofunction:: tcod.dijkstra_get
.. autofunction:: tcod.dijkstra_get_distance
.. autofunction:: tcod.dijkstra_is_empty
.. autofunction:: tcod.dijkstra_new
.. autofunction:: tcod.dijkstra_new_using_function
.. autofunction:: tcod.dijkstra_path_set
.. autofunction:: tcod.dijkstra_path_walk
.. autofunction:: tcod.dijkstra_reverse
.. autofunction:: tcod.dijkstra_size

.. autofunction:: tcod.path_compute
.. autofunction:: tcod.path_delete
.. autofunction:: tcod.path_get
.. autofunction:: tcod.path_get_destination
.. autofunction:: tcod.path_get_origin
.. autofunction:: tcod.path_is_empty
.. autofunction:: tcod.path_new_using_function
.. autofunction:: tcod.path_new_using_map
.. autofunction:: tcod.path_reverse
.. autofunction:: tcod.path_size
.. autofunction:: tcod.path_walk

heightmap
---------

.. autofunction:: tcod.heightmap_add
.. autofunction:: tcod.heightmap_add_fbm
.. autofunction:: tcod.heightmap_add_hill
.. autofunction:: tcod.heightmap_add_hm
.. autofunction:: tcod.heightmap_add_voronoi
.. autofunction:: tcod.heightmap_clamp
.. autofunction:: tcod.heightmap_clear
.. autofunction:: tcod.heightmap_copy
.. autofunction:: tcod.heightmap_count_cells
.. autofunction:: tcod.heightmap_delete
.. autofunction:: tcod.heightmap_dig_bezier
.. autofunction:: tcod.heightmap_dig_hill
.. autofunction:: tcod.heightmap_get_interpolated_value
.. autofunction:: tcod.heightmap_get_minmax
.. autofunction:: tcod.heightmap_get_normal
.. autofunction:: tcod.heightmap_get_slope
.. autofunction:: tcod.heightmap_get_value
.. autofunction:: tcod.heightmap_has_land_on_border
.. autofunction:: tcod.heightmap_kernel_transform
.. autofunction:: tcod.heightmap_lerp_hm
.. autofunction:: tcod.heightmap_multiply_hm
.. autofunction:: tcod.heightmap_new
.. autofunction:: tcod.heightmap_normalize
.. autofunction:: tcod.heightmap_rain_erosion
.. autofunction:: tcod.heightmap_scale
.. autofunction:: tcod.heightmap_scale_fbm
.. autofunction:: tcod.heightmap_set_value

image
-----

.. autofunction:: tcod.image_blit
.. autofunction:: tcod.image_blit_2x
.. autofunction:: tcod.image_blit_rect
.. autofunction:: tcod.image_clear
.. autofunction:: tcod.image_delete
.. autofunction:: tcod.image_get_alpha
.. autofunction:: tcod.image_get_mipmap_pixel
.. autofunction:: tcod.image_get_pixel
.. autofunction:: tcod.image_get_size
.. autofunction:: tcod.image_hflip
.. autofunction:: tcod.image_invert
.. autofunction:: tcod.image_is_pixel_transparent
.. autofunction:: tcod.image_new
.. autofunction:: tcod.image_put_pixel
.. autofunction:: tcod.image_refresh_console
.. autofunction:: tcod.image_rotate90
.. autofunction:: tcod.image_save
.. autofunction:: tcod.image_scale
.. autofunction:: tcod.image_set_key_color
.. autofunction:: tcod.image_vflip

map
---

.. autofunction:: tcod.map_clear
.. autofunction:: tcod.map_compute_fov
.. autofunction:: tcod.map_copy
.. autofunction:: tcod.map_delete
.. autofunction:: tcod.map_get_height
.. autofunction:: tcod.map_get_width
.. autofunction:: tcod.map_is_in_fov
.. autofunction:: tcod.map_is_transparent
.. autofunction:: tcod.map_is_walkable
.. autofunction:: tcod.map_new
.. autofunction:: tcod.map_set_properties

mouse
-----

.. autofunction:: tcod.mouse_get_status
.. autofunction:: tcod.mouse_is_cursor_visible
.. autofunction:: tcod.mouse_move
.. autofunction:: tcod.mouse_show_cursor

namegen
-------

.. autofunction:: tcod.namegen_destroy
.. autofunction:: tcod.namegen_generate
.. autofunction:: tcod.namegen_generate_custom
.. autofunction:: tcod.namegen_get_sets
.. autofunction:: tcod.namegen_parse

noise
-----

.. autofunction:: tcod.noise_delete
.. autofunction:: tcod.noise_get
.. autofunction:: tcod.noise_get_fbm
.. autofunction:: tcod.noise_get_turbulence
.. autofunction:: tcod.noise_new
.. autofunction:: tcod.noise_set_type

parser
------

.. autofunction:: tcod.parser_delete
.. autofunction:: tcod.parser_get_bool_property
.. autofunction:: tcod.parser_get_char_property
.. autofunction:: tcod.parser_get_color_property
.. autofunction:: tcod.parser_get_dice_property
.. autofunction:: tcod.parser_get_float_property
.. autofunction:: tcod.parser_get_int_property
.. autofunction:: tcod.parser_get_list_property
.. autofunction:: tcod.parser_get_string_property
.. autofunction:: tcod.parser_new
.. autofunction:: tcod.parser_new_struct
.. autofunction:: tcod.parser_run

random
------

.. autofunction:: tcod.random_delete
.. autofunction:: tcod.random_get_double
.. autofunction:: tcod.random_get_double_mean
.. autofunction:: tcod.random_get_float
.. autofunction:: tcod.random_get_float_mean
.. autofunction:: tcod.random_get_instance
.. autofunction:: tcod.random_get_int
.. autofunction:: tcod.random_get_int_mean
.. autofunction:: tcod.random_new
.. autofunction:: tcod.random_new_from_seed
.. autofunction:: tcod.random_restore
.. autofunction:: tcod.random_save
.. autofunction:: tcod.random_set_distribution

struct
------
.. autofunction:: tcod.struct_add_flag
.. autofunction:: tcod.struct_add_list_property
.. autofunction:: tcod.struct_add_property
.. autofunction:: tcod.struct_add_structure
.. autofunction:: tcod.struct_add_value_list
.. autofunction:: tcod.struct_get_name
.. autofunction:: tcod.struct_get_type
.. autofunction:: tcod.struct_is_mandatory

other
-----

.. autoclass:: tcod.ConsoleBuffer
    :members:

.. autoclass:: tcod.Dice(nb_dices=0, nb_faces=0, multiplier=0, addsub=0)
   :members:
