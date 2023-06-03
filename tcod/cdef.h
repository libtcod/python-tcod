/* Python specific cdefs which are loaded directly into cffi. */
#include "../libtcod/src/libtcod/libtcod.h"
extern "Python" {
bool _pycall_parser_new_struct(TCOD_parser_struct_t str, const char* name);
bool _pycall_parser_new_flag(const char* name);
bool _pycall_parser_new_property(const char* propname, TCOD_value_type_t type, TCOD_value_t value);
bool _pycall_parser_end_struct(TCOD_parser_struct_t str, const char* name);
void _pycall_parser_error(const char* msg);

bool _pycall_bsp_callback(TCOD_bsp_t* node, void* userData);

float _pycall_path_old(int x, int y, int xDest, int yDest, void* user_data);
float _pycall_path_simple(int x, int y, int xDest, int yDest, void* user_data);
float _pycall_path_swap_src_dest(int x1, int y1, int x2, int y2, void* user_data);
float _pycall_path_dest_only(int x1, int y1, int x2, int y2, void* user_data);

void _pycall_sdl_hook(struct SDL_Surface*);

void _pycall_cli_output(void* userdata, const char* output);

// Libtcod log watch function.
void _libtcod_log_watcher(const TCOD_LogMessage* message, void* userdata);
}
