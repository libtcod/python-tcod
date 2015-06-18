#!/usr/bin/env python3

from cffi import FFI

ffi = FFI()
ffi.cdef("""

/* base types */
typedef unsigned char uint8;
typedef char int8;
typedef unsigned short uint16;
typedef short int16;
typedef unsigned int uint32;
typedef int int32;
/* int with the same size as a pointer (32 or 64 depending on OS) */
typedef long intptr;
typedef unsigned long uintptr;

typedef uint8 bool;



typedef struct {
	uint8 r,g,b;
} TCOD_color_t;

typedef enum {
	TCODK_NONE,
	TCODK_ESCAPE,
	TCODK_BACKSPACE,
	TCODK_TAB,
	TCODK_ENTER,
	TCODK_SHIFT,
	TCODK_CONTROL,
	TCODK_ALT,
	TCODK_PAUSE,
	TCODK_CAPSLOCK,
	TCODK_PAGEUP,
	TCODK_PAGEDOWN,
	TCODK_END,
	TCODK_HOME,
	TCODK_UP,
	TCODK_LEFT,
	TCODK_RIGHT,
	TCODK_DOWN,
	TCODK_PRINTSCREEN,
	TCODK_INSERT,
	TCODK_DELETE,
	TCODK_LWIN,
	TCODK_RWIN,
	TCODK_APPS,
	TCODK_0,
	TCODK_1,
	TCODK_2,
	TCODK_3,
	TCODK_4,
	TCODK_5,
	TCODK_6,
	TCODK_7,
	TCODK_8,
	TCODK_9,
	TCODK_KP0,
	TCODK_KP1,
	TCODK_KP2,
	TCODK_KP3,
	TCODK_KP4,
	TCODK_KP5,
	TCODK_KP6,
	TCODK_KP7,
	TCODK_KP8,
	TCODK_KP9,
	TCODK_KPADD,
	TCODK_KPSUB,
	TCODK_KPDIV,
	TCODK_KPMUL,
	TCODK_KPDEC,
	TCODK_KPENTER,
	TCODK_F1,
	TCODK_F2,
	TCODK_F3,
	TCODK_F4,
	TCODK_F5,
	TCODK_F6,
	TCODK_F7,
	TCODK_F8,
	TCODK_F9,
	TCODK_F10,
	TCODK_F11,
	TCODK_F12,
	TCODK_NUMLOCK,
	TCODK_SCROLLLOCK,
	TCODK_SPACE,
	TCODK_CHAR
} TCOD_keycode_t;

/* key data : special code or character */
typedef struct {
	TCOD_keycode_t vk; /*  key code */
	char c; /* character if vk == TCODK_CHAR else 0 */
	bool pressed ; /* does this correspond to a key press or key release event ? */
	bool lalt ;
	bool lctrl ;
	bool ralt ;
	bool rctrl ;
	bool shift ;
} TCOD_key_t;

typedef enum {
	TCOD_BKGND_NONE,
	TCOD_BKGND_SET,
	TCOD_BKGND_MULTIPLY,
	TCOD_BKGND_LIGHTEN,
	TCOD_BKGND_DARKEN,
	TCOD_BKGND_SCREEN,
	TCOD_BKGND_COLOR_DODGE,
	TCOD_BKGND_COLOR_BURN,
	TCOD_BKGND_ADD,
	TCOD_BKGND_ADDA,
	TCOD_BKGND_BURN,
	TCOD_BKGND_OVERLAY,
	TCOD_BKGND_ALPH,
	TCOD_BKGND_DEFAULT
} TCOD_bkgnd_flag_t;


// MOUSE

/* mouse data */
typedef struct {
  int x,y; /* absolute position */
  int dx,dy; /* movement since last update in pixels */
  int cx,cy; /* cell coordinates in the root console */
  int dcx,dcy; /* movement since last update in console cells */
  bool lbutton ; /* left button status */
  bool rbutton ; /* right button status */
  bool mbutton ; /* middle button status */
  bool lbutton_pressed ; /* left button pressed event */ 
  bool rbutton_pressed ; /* right button pressed event */ 
  bool mbutton_pressed ; /* middle button pressed event */ 
  bool wheel_up ; /* wheel up event */
  bool wheel_down ; /* wheel down event */
} TCOD_mouse_t;

void TCOD_mouse_show_cursor(bool visible);
TCOD_mouse_t TCOD_mouse_get_status();
bool TCOD_mouse_is_cursor_visible();
void TCOD_mouse_move(int x, int y);
//void TCOD_mouse_includes_touch(bool enable);

// CONSOLE TYPES --------------------------------------------------------

typedef enum {
	TCOD_KEY_PRESSED=1,
	TCOD_KEY_RELEASED=2,
} TCOD_key_status_t;

/* custom font flags */
typedef enum {
	TCOD_FONT_LAYOUT_ASCII_INCOL=1,
	TCOD_FONT_LAYOUT_ASCII_INROW=2,
	TCOD_FONT_TYPE_GREYSCALE=4,
	TCOD_FONT_TYPE_GRAYSCALE=4,
	TCOD_FONT_LAYOUT_TCOD=8,
} TCOD_font_flags_t;

typedef enum {
	TCOD_RENDERER_GLSL,
	TCOD_RENDERER_OPENGL,
	TCOD_RENDERER_SDL,
	TCOD_NB_RENDERERS,
} TCOD_renderer_t;

typedef enum {
	TCOD_LEFT, 
	TCOD_RIGHT, 
	TCOD_CENTER 
} TCOD_alignment_t;

typedef enum {
	TCOD_COLCTRL_1 = 1,
	TCOD_COLCTRL_2,
	TCOD_COLCTRL_3,
	TCOD_COLCTRL_4,
	TCOD_COLCTRL_5,
	TCOD_COLCTRL_NUMBER=5,
	TCOD_COLCTRL_FORE_RGB,
	TCOD_COLCTRL_BACK_RGB,
	TCOD_COLCTRL_STOP
} TCOD_colctrl_t;

// CONSOLE --------------------------------------------------------------

typedef void * TCOD_console_t;

void TCOD_console_init_root(int w, int h, const char * title, bool fullscreen, TCOD_renderer_t renderer);
void TCOD_console_set_window_title(const char *title);
void TCOD_console_set_fullscreen(bool fullscreen);
bool TCOD_console_is_fullscreen();
bool TCOD_console_is_window_closed();

void TCOD_console_set_custom_font(const char *fontFile, int flags,int nb_char_horiz, int nb_char_vertic);
void TCOD_console_map_ascii_code_to_font(int asciiCode, int fontCharX, int fontCharY);
void TCOD_console_map_ascii_codes_to_font(int asciiCode, int nbCodes, int fontCharX, int fontCharY);
void TCOD_console_map_string_to_font(const char *s, int fontCharX, int fontCharY);

void TCOD_console_set_dirty(int x, int y, int w, int h);
void TCOD_console_set_default_background(TCOD_console_t con,TCOD_color_t col);
void TCOD_console_set_default_foreground(TCOD_console_t con,TCOD_color_t col);
void TCOD_console_clear(TCOD_console_t con);
void TCOD_console_set_char_background(TCOD_console_t con,int x, int y, TCOD_color_t col, TCOD_bkgnd_flag_t flag);
void TCOD_console_set_char_foreground(TCOD_console_t con,int x, int y, TCOD_color_t col);
void TCOD_console_set_char(TCOD_console_t con,int x, int y, int c);
void TCOD_console_put_char(TCOD_console_t con,int x, int y, int c, TCOD_bkgnd_flag_t flag);
void TCOD_console_put_char_ex(TCOD_console_t con,int x, int y, int c, TCOD_color_t fore, TCOD_color_t back);

void TCOD_console_set_background_flag(TCOD_console_t con,TCOD_bkgnd_flag_t flag);
TCOD_bkgnd_flag_t TCOD_console_get_background_flag(TCOD_console_t con);
void TCOD_console_set_alignment(TCOD_console_t con,TCOD_alignment_t alignment);
TCOD_alignment_t TCOD_console_get_alignment(TCOD_console_t con);
void TCOD_console_print(TCOD_console_t con,int x, int y, const char *fmt, ...);
void TCOD_console_print_ex(TCOD_console_t con,int x, int y, TCOD_bkgnd_flag_t flag, TCOD_alignment_t alignment, const char *fmt, ...);
int TCOD_console_print_rect(TCOD_console_t con,int x, int y, int w, int h, const char *fmt, ...);
int TCOD_console_print_rect_ex(TCOD_console_t con,int x, int y, int w, int h, TCOD_bkgnd_flag_t flag, TCOD_alignment_t alignment, const char *fmt, ...);
int TCOD_console_get_height_rect(TCOD_console_t con,int x, int y, int w, int h, const char *fmt, ...);

void TCOD_console_rect(TCOD_console_t con,int x, int y, int w, int h, bool clear, TCOD_bkgnd_flag_t flag);
void TCOD_console_hline(TCOD_console_t con,int x,int y, int l, TCOD_bkgnd_flag_t flag);
void TCOD_console_vline(TCOD_console_t con,int x,int y, int l, TCOD_bkgnd_flag_t flag);
void TCOD_console_print_frame(TCOD_console_t con,int x,int y,int w,int h, bool empty, TCOD_bkgnd_flag_t flag, const char *fmt, ...);

/* unicode support */
void TCOD_console_map_string_to_font_utf(const wchar_t *s, int fontCharX, int fontCharY);
void TCOD_console_print_utf(TCOD_console_t con,int x, int y, const wchar_t *fmt, ...);
void TCOD_console_print_ex_utf(TCOD_console_t con,int x, int y, TCOD_bkgnd_flag_t flag, TCOD_alignment_t alignment, const wchar_t *fmt, ...);
int TCOD_console_print_rect_utf(TCOD_console_t con,int x, int y, int w, int h, const wchar_t *fmt, ...);
int TCOD_console_print_rect_ex_utf(TCOD_console_t con,int x, int y, int w, int h, TCOD_bkgnd_flag_t flag, TCOD_alignment_t alignment, const wchar_t *fmt, ...);
int TCOD_console_get_height_rect_utf(TCOD_console_t con,int x, int y, int w, int h, const wchar_t *fmt, ...);


TCOD_color_t TCOD_console_get_default_background(TCOD_console_t con);
TCOD_color_t TCOD_console_get_default_foreground(TCOD_console_t con);
TCOD_color_t TCOD_console_get_char_background(TCOD_console_t con,int x, int y);
TCOD_color_t TCOD_console_get_char_foreground(TCOD_console_t con,int x, int y);
int TCOD_console_get_char(TCOD_console_t con,int x, int y);

void TCOD_console_set_fade(uint8 val, TCOD_color_t fade);
uint8 TCOD_console_get_fade();
TCOD_color_t TCOD_console_get_fading_color();

void TCOD_console_flush();

void TCOD_console_set_color_control(TCOD_colctrl_t con, TCOD_color_t fore, TCOD_color_t back);

TCOD_key_t TCOD_console_check_for_keypress(int flags);
TCOD_key_t TCOD_console_wait_for_keypress(bool flush);
void TCOD_console_set_keyboard_repeat(int initial_delay, int interval);
void TCOD_console_disable_keyboard_repeat();
bool TCOD_console_is_key_pressed(TCOD_keycode_t key);

/* ASCII paint file support */
TCOD_console_t TCOD_console_from_file(const char *filename);
bool TCOD_console_load_asc(TCOD_console_t con, const char *filename);
bool TCOD_console_load_apf(TCOD_console_t con, const char *filename);
bool TCOD_console_save_asc(TCOD_console_t con, const char *filename);
bool TCOD_console_save_apf(TCOD_console_t con, const char *filename);

TCOD_console_t TCOD_console_new(int w, int h);
int TCOD_console_get_width(TCOD_console_t con);
int TCOD_console_get_height(TCOD_console_t con);
void TCOD_console_set_key_color(TCOD_console_t con,TCOD_color_t col);
void TCOD_console_blit(TCOD_console_t src,int xSrc, int ySrc, int wSrc, int hSrc, TCOD_console_t dst, int xDst, int yDst, float foreground_alpha, float background_alpha);
void TCOD_console_delete(TCOD_console_t console);

void TCOD_console_credits();
void TCOD_console_credits_reset();
bool TCOD_console_credits_render(int x, int y, bool alpha);

// IMAGE -------------------------------------------------


typedef void *TCOD_image_t;

TCOD_image_t TCOD_image_new(int width, int height);
TCOD_image_t TCOD_image_from_console(TCOD_console_t console);
void TCOD_image_refresh_console(TCOD_image_t image, TCOD_console_t console);
TCOD_image_t TCOD_image_load(const char *filename);
void TCOD_image_clear(TCOD_image_t image, TCOD_color_t color);
void TCOD_image_invert(TCOD_image_t image);
void TCOD_image_hflip(TCOD_image_t image);
void TCOD_image_rotate90(TCOD_image_t image, int numRotations);
void TCOD_image_vflip(TCOD_image_t image);
void TCOD_image_scale(TCOD_image_t image, int neww, int newh);
void TCOD_image_save(TCOD_image_t image, const char *filename);
void TCOD_image_get_size(TCOD_image_t image, int *w,int *h);
TCOD_color_t TCOD_image_get_pixel(TCOD_image_t image,int x, int y);
int TCOD_image_get_alpha(TCOD_image_t image,int x, int y);
TCOD_color_t TCOD_image_get_mipmap_pixel(TCOD_image_t image,float x0,float y0, float x1, float y1);
void TCOD_image_put_pixel(TCOD_image_t image,int x, int y,TCOD_color_t col);
void TCOD_image_blit(TCOD_image_t image, TCOD_console_t console, float x, float y, 
	TCOD_bkgnd_flag_t bkgnd_flag, float scalex, float scaley, float angle);
void TCOD_image_blit_rect(TCOD_image_t image, TCOD_console_t console, int x, int y, int w, int h, 
	TCOD_bkgnd_flag_t bkgnd_flag);
void TCOD_image_blit_2x(TCOD_image_t image, TCOD_console_t dest, int dx, int dy, int sx, int sy, int w, int h);
void TCOD_image_delete(TCOD_image_t image);
void TCOD_image_set_key_color(TCOD_image_t image, TCOD_color_t key_color);
bool TCOD_image_is_pixel_transparent(TCOD_image_t image, int x, int y);



// SYS -----------------------------------------------------------


//uint32 TCOD_sys_elapsed_milli();
//float TCOD_sys_elapsed_seconds();
//void TCOD_sys_sleep_milli(uint32 val);
void TCOD_sys_save_screenshot(const char *filename);
void TCOD_sys_force_fullscreen_resolution(int width, int height);
void TCOD_sys_set_renderer(TCOD_renderer_t renderer);
TCOD_renderer_t TCOD_sys_get_renderer();
void TCOD_sys_set_fps(int val);
int TCOD_sys_get_fps();
float TCOD_sys_get_last_frame_length();
void TCOD_sys_get_current_resolution(int *w, int *h);
void TCOD_sys_get_fullscreen_offsets(int *offx, int *offy);
void TCOD_sys_update_char(int asciiCode, int fontx, int fonty, TCOD_image_t img, int x, int y);
void TCOD_sys_get_char_size(int *w, int *h);
//void *TCOD_sys_get_sdl_window();

typedef enum {
  TCOD_EVENT_KEY_PRESS=1,
  TCOD_EVENT_KEY_RELEASE=2,
  TCOD_EVENT_KEY=...,
  TCOD_EVENT_MOUSE_MOVE=4,
  TCOD_EVENT_MOUSE_PRESS=8,
  TCOD_EVENT_MOUSE_RELEASE=16,
  TCOD_EVENT_MOUSE=...,
  TCOD_EVENT_ANY=...,
} TCOD_event_t;
TCOD_event_t TCOD_sys_wait_for_event(int eventMask, TCOD_key_t *key, TCOD_mouse_t *mouse, bool flush);
TCOD_event_t TCOD_sys_check_for_event(int eventMask, TCOD_key_t *key, TCOD_mouse_t *mouse);

/* filesystem stuff */
//bool TCOD_sys_create_directory(const char *path);
//bool TCOD_sys_delete_file(const char *path);
//bool TCOD_sys_delete_directory(const char *path);
//bool TCOD_sys_is_directory(const char *path);
//TCOD_list_t TCOD_sys_get_directory_content(const char *path, const char *pattern);
//bool TCOD_sys_file_exists(const char * filename, ...);
//bool TCOD_sys_read_file(const char *filename, unsigned char **buf, uint32 *size);
//bool TCOD_sys_write_file(const char *filename, unsigned char *buf, uint32 size);

/* clipboard */
void TCOD_sys_clipboard_set(const char *value);
char *TCOD_sys_clipboard_get();

/* thread stuff */
typedef void *TCOD_thread_t;
typedef void *TCOD_semaphore_t;
typedef void *TCOD_mutex_t;
typedef void *TCOD_cond_t;
/* threads */
TCOD_thread_t TCOD_thread_new(int (*func)(void *), void *data);
void TCOD_thread_delete(TCOD_thread_t th);
int TCOD_sys_get_num_cores();
void TCOD_thread_wait(TCOD_thread_t th);
/* mutex */
TCOD_mutex_t TCOD_mutex_new();
void TCOD_mutex_in(TCOD_mutex_t mut);
void TCOD_mutex_out(TCOD_mutex_t mut);
void TCOD_mutex_delete(TCOD_mutex_t mut);
/* semaphore */
TCOD_semaphore_t TCOD_semaphore_new(int initVal);
void TCOD_semaphore_lock(TCOD_semaphore_t sem);
void TCOD_semaphore_unlock(TCOD_semaphore_t sem);
void TCOD_semaphore_delete( TCOD_semaphore_t sem);
/* condition */
TCOD_cond_t TCOD_condition_new();
void TCOD_condition_signal(TCOD_cond_t sem);
void TCOD_condition_broadcast(TCOD_cond_t sem);
void TCOD_condition_wait(TCOD_cond_t sem, TCOD_mutex_t mut);
void TCOD_condition_delete( TCOD_cond_t sem);
/* dynamic library */
typedef void *TCOD_library_t;
TCOD_library_t TCOD_load_library(const char *path);
void * TCOD_get_function_address(TCOD_library_t library, const char *function_name);
void TCOD_close_library(TCOD_library_t);
/* SDL renderer callback */
typedef void (*SDL_renderer_t) (void *sdl_surface);
void TCOD_sys_register_SDL_renderer(SDL_renderer_t renderer);


// MERSENNE

/* dice roll */
typedef struct {
	int nb_rolls;
	int nb_faces;
	float multiplier;
	float addsub;
} TCOD_dice_t;

/* PRNG algorithms */
typedef enum {
    TCOD_RNG_MT,
    TCOD_RNG_CMWC
} TCOD_random_algo_t;

typedef enum {
	TCOD_DISTRIBUTION_LINEAR,
	TCOD_DISTRIBUTION_GAUSSIAN,
	TCOD_DISTRIBUTION_GAUSSIAN_RANGE,
	TCOD_DISTRIBUTION_GAUSSIAN_INVERSE,
	TCOD_DISTRIBUTION_GAUSSIAN_RANGE_INVERSE
} TCOD_distribution_t;

typedef void *TCOD_random_t;

TCOD_random_t TCOD_random_get_instance(void);
TCOD_random_t TCOD_random_new(TCOD_random_algo_t algo);
TCOD_random_t TCOD_random_save(TCOD_random_t mersenne);
void TCOD_random_restore(TCOD_random_t mersenne, TCOD_random_t backup);
TCOD_random_t TCOD_random_new_from_seed(TCOD_random_algo_t algo, uint32 seed);
void TCOD_random_delete(TCOD_random_t mersenne);

void TCOD_random_set_distribution (TCOD_random_t mersenne, TCOD_distribution_t distribution);

int TCOD_random_get_int (TCOD_random_t mersenne, int min, int max);
float TCOD_random_get_float (TCOD_random_t mersenne, float min, float max);
double TCOD_random_get_double (TCOD_random_t mersenne, double min, double max);

int TCOD_random_get_int_mean (TCOD_random_t mersenne, int min, int max, int mean);
float TCOD_random_get_float_mean (TCOD_random_t mersenne, float min, float max, float mean);
double TCOD_random_get_double_mean (TCOD_random_t mersenne, double min, double max, double mean);

TCOD_dice_t TCOD_random_dice_new (const char * s);
int TCOD_random_dice_roll (TCOD_random_t mersenne, TCOD_dice_t dice);
int TCOD_random_dice_roll_s (TCOD_random_t mersenne, const char * s);



// NOISE


typedef void *TCOD_noise_t;

typedef enum {
	TCOD_NOISE_PERLIN = 1,
	TCOD_NOISE_SIMPLEX = 2,
	TCOD_NOISE_WAVELET = 4,
	TCOD_NOISE_DEFAULT = 0
} TCOD_noise_type_t;

/* create a new noise object */
TCOD_noise_t TCOD_noise_new(int dimensions, float hurst, float lacunarity, TCOD_random_t random);

/* simplified API */
void TCOD_noise_set_type (TCOD_noise_t noise, TCOD_noise_type_t type);
float TCOD_noise_get_ex (TCOD_noise_t noise, float *f, TCOD_noise_type_t type);
float TCOD_noise_get_fbm_ex (TCOD_noise_t noise, float *f, float octaves, TCOD_noise_type_t type);
float TCOD_noise_get_turbulence_ex (TCOD_noise_t noise, float *f, float octaves, TCOD_noise_type_t type);
float TCOD_noise_get (TCOD_noise_t noise, float *f);
float TCOD_noise_get_fbm (TCOD_noise_t noise, float *f, float octaves);
float TCOD_noise_get_turbulence (TCOD_noise_t noise, float *f, float octaves);
/* delete the noise object */
void TCOD_noise_delete(TCOD_noise_t noise);


// CUSTOM FUNCTONS

void set_char(TCOD_console_t console, int x, int y,
                     int ch, int fg, int bg);

""")

ffi.set_source('tdl._libtcod', """
#include <libtcod.h>

void set_char(TCOD_console_t console, int x, int y,
                     int ch, int fg, int bg){
    // normalize x, y
    int width=TCOD_console_get_width(console);
    int height=TCOD_console_get_height(console);
    TCOD_color_t color;
    
    x = x % width;
    if(x<0){x += width;};
    y = y % height;
    if(y<0){y += height;};
    
    if(ch != -1){
        TCOD_console_set_char(console, x, y, ch);
    }
    if(fg != -1){
        color.r = fg >> 16 & 0xff;
        color.g = fg >> 8 & 0xff;
        color.b = fg & 0xff;
        TCOD_console_set_char_foreground(console, x, y, color);
    }
    if(bg != -1){
        color.r = bg >> 16 & 0xff;
        color.g = bg >> 8 & 0xff;
        color.b = bg & 0xff;
        TCOD_console_set_char_background(console, x, y, color, 1);
    }
    
}

""",
include_dirs=['include/', 'tdl/include/', 'tdl/include/Release/'],
library_dirs=['lib/win32/', 'tdl/lib/win32/'],
libraries=['libtcod-VS'])


if __name__ == "__main__":
    ffi.compile()