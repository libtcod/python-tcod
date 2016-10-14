
/* TDL FUNCTONS ----------------------------------------------------------- */

TCOD_value_t TDL_list_get_union(TCOD_list_t l,int idx);
bool TDL_list_get_bool(TCOD_list_t l,int idx);
char TDL_list_get_char(TCOD_list_t l,int idx);
int TDL_list_get_int(TCOD_list_t l,int idx);
float TDL_list_get_float(TCOD_list_t l,int idx);
char* TDL_list_get_string(TCOD_list_t l,int idx);
TCOD_color_t TDL_list_get_color(TCOD_list_t l,int idx);
TCOD_dice_t TDL_list_get_dice(TCOD_list_t l,int idx);
/*bool (*TDL_parser_new_property_func)(const char *propname, TCOD_value_type_t type, TCOD_value_t *value);*/

/* color functions modified to use integers instead of structs */
TCOD_color_t TDL_color_from_int(int color);
int TDL_color_to_int(TCOD_color_t *color);
int* TDL_color_int_to_array(int color);
int TDL_color_RGB(int r, int g, int b);
int TDL_color_HSV(float h, float s, float v);
bool TDL_color_equals(int c1, int c2);
int TDL_color_add(int c1, int c2);
int TDL_color_subtract(int c1, int c2);
int TDL_color_multiply(int c1, int c2);
int TDL_color_multiply_scalar(int c, float value);
int TDL_color_lerp(int c1, int c2, float coef);
float TDL_color_get_hue(int color);
float TDL_color_get_saturation(int color);
float TDL_color_get_value(int color);
int TDL_color_set_hue(int color, float h);
int TDL_color_set_saturation(int color, float h);
int TDL_color_set_value(int color, float h);
int TDL_color_shift_hue(int color, float hshift);
int TDL_color_scale_HSV(int color, float scoef, float vcoef);

/* map data functions using a bitmap of:
 * 1 = is_transparant
 * 2 = is_walkable
 * 4 = in_fov
 */
void TDL_map_data_from_buffer(TCOD_map_t map, uint8 *buffer);
void TDL_map_fov_to_buffer(TCOD_map_t map, uint8 *buffer, bool cumulative);

int TDL_console_put_char_ex(TCOD_console_t console, int x, int y,
                            int ch, int fg, int bg, TCOD_bkgnd_flag_t flag);
int TDL_console_get_bg(TCOD_console_t console, int x, int y);
int TDL_console_get_fg(TCOD_console_t console, int x, int y);
void TDL_console_set_bg(TCOD_console_t console, int x, int y, int color,
                        TCOD_bkgnd_flag_t flag);
void TDL_console_set_fg(TCOD_console_t console, int x, int y, int color);
