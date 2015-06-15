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
	/* single walls */
	TCOD_CHAR_HLINE=196,
	TCOD_CHAR_VLINE=179,
	TCOD_CHAR_NE=191,
	TCOD_CHAR_NW=218,
	TCOD_CHAR_SE=217,
	TCOD_CHAR_SW=192,
	TCOD_CHAR_TEEW=180,
	TCOD_CHAR_TEEE=195,
	TCOD_CHAR_TEEN=193,
	TCOD_CHAR_TEES=194,
	TCOD_CHAR_CROSS=197,
	/* double walls */
	TCOD_CHAR_DHLINE=205,
	TCOD_CHAR_DVLINE=186,
	TCOD_CHAR_DNE=187,
	TCOD_CHAR_DNW=201,
	TCOD_CHAR_DSE=188,
	TCOD_CHAR_DSW=200,
	TCOD_CHAR_DTEEW=185,
	TCOD_CHAR_DTEEE=204,
	TCOD_CHAR_DTEEN=202,
	TCOD_CHAR_DTEES=203,
	TCOD_CHAR_DCROSS=206,
	/* blocks */
	TCOD_CHAR_BLOCK1=176,
	TCOD_CHAR_BLOCK2=177,
	TCOD_CHAR_BLOCK3=178,
	/* arrows */
	TCOD_CHAR_ARROW_N=24,
	TCOD_CHAR_ARROW_S=25,
	TCOD_CHAR_ARROW_E=26,
	TCOD_CHAR_ARROW_W=27,
	/* arrows without tail */
	TCOD_CHAR_ARROW2_N=30,
	TCOD_CHAR_ARROW2_S=31,
	TCOD_CHAR_ARROW2_E=16,
	TCOD_CHAR_ARROW2_W=17,
	/* double arrows */
	TCOD_CHAR_DARROW_H=29,
	TCOD_CHAR_DARROW_V=18,
	/* GUI stuff */
	TCOD_CHAR_CHECKBOX_UNSET=224,
	TCOD_CHAR_CHECKBOX_SET=225,
	TCOD_CHAR_RADIO_UNSET=9,
	TCOD_CHAR_RADIO_SET=10,
	/* sub-pixel resolution kit */
	TCOD_CHAR_SUBP_NW=226,
	TCOD_CHAR_SUBP_NE=227,
	TCOD_CHAR_SUBP_N=228,
	TCOD_CHAR_SUBP_SE=229,
	TCOD_CHAR_SUBP_DIAG=230,
	TCOD_CHAR_SUBP_E=231,
	TCOD_CHAR_SUBP_SW=232,
	/* miscellaneous */
	TCOD_CHAR_SMILIE = 1,
	TCOD_CHAR_SMILIE_INV = 2,
	TCOD_CHAR_HEART = 3,
	TCOD_CHAR_DIAMOND = 4,
	TCOD_CHAR_CLUB = 5,
	TCOD_CHAR_SPADE = 6,
	TCOD_CHAR_BULLET = 7,
	TCOD_CHAR_BULLET_INV = 8,
	TCOD_CHAR_MALE = 11,
	TCOD_CHAR_FEMALE = 12,
	TCOD_CHAR_NOTE = 13,
	TCOD_CHAR_NOTE_DOUBLE = 14,
	TCOD_CHAR_LIGHT = 15,
	TCOD_CHAR_EXCLAM_DOUBLE = 19,
	TCOD_CHAR_PILCROW = 20,
	TCOD_CHAR_SECTION = 21,
	TCOD_CHAR_POUND = 156,
	TCOD_CHAR_MULTIPLICATION = 158,
	TCOD_CHAR_FUNCTION = 159,
	TCOD_CHAR_RESERVED = 169,
	TCOD_CHAR_HALF = 171,
	TCOD_CHAR_ONE_QUARTER = 172,
	TCOD_CHAR_COPYRIGHT = 184,
	TCOD_CHAR_CENT = 189,
	TCOD_CHAR_YEN = 190,
	TCOD_CHAR_CURRENCY = 207,
	TCOD_CHAR_THREE_QUARTERS = 243,
	TCOD_CHAR_DIVISION = 246,
	TCOD_CHAR_GRADE = 248,
	TCOD_CHAR_UMLAUT = 249,
	TCOD_CHAR_POW1 = 251,
	TCOD_CHAR_POW3 = 252,
	TCOD_CHAR_POW2 = 253,
	TCOD_CHAR_BULLET_SQUARE = 254,
	/* diacritics */
} TCOD_chars_t;

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

/* color names */
enum {
	TCOD_COLOR_RED,
	TCOD_COLOR_FLAME,
	TCOD_COLOR_ORANGE,
	TCOD_COLOR_AMBER,
	TCOD_COLOR_YELLOW,
	TCOD_COLOR_LIME,
	TCOD_COLOR_CHARTREUSE,
	TCOD_COLOR_GREEN,
	TCOD_COLOR_SEA,
	TCOD_COLOR_TURQUOISE,
	TCOD_COLOR_CYAN,
	TCOD_COLOR_SKY,
	TCOD_COLOR_AZURE,
	TCOD_COLOR_BLUE,
	TCOD_COLOR_HAN,
	TCOD_COLOR_VIOLET,
	TCOD_COLOR_PURPLE,
	TCOD_COLOR_FUCHSIA,
	TCOD_COLOR_MAGENTA,
	TCOD_COLOR_PINK,
	TCOD_COLOR_CRIMSON,
	TCOD_COLOR_NB
};

/* color levels */
enum {
	TCOD_COLOR_DESATURATED,
	TCOD_COLOR_LIGHTEST,
	TCOD_COLOR_LIGHTER,
	TCOD_COLOR_LIGHT,
	TCOD_COLOR_NORMAL,
	TCOD_COLOR_DARK,
	TCOD_COLOR_DARKER,
	TCOD_COLOR_DARKEST,
	TCOD_COLOR_LEVELS
};


typedef struct {
	uint8 r,g,b;
} TCOD_color_t;


/* constructors */
TCOD_color_t TCOD_color_RGB(uint8 r, uint8 g, uint8 b);
TCOD_color_t TCOD_color_HSV(float h, float s, float v);
/* basic operations */
bool TCOD_color_equals (TCOD_color_t c1, TCOD_color_t c2);
TCOD_color_t TCOD_color_add (TCOD_color_t c1, TCOD_color_t c2);
TCOD_color_t TCOD_color_subtract (TCOD_color_t c1, TCOD_color_t c2);
TCOD_color_t TCOD_color_multiply (TCOD_color_t c1, TCOD_color_t c2);
TCOD_color_t TCOD_color_multiply_scalar (TCOD_color_t c1, float value);
TCOD_color_t TCOD_color_lerp (TCOD_color_t c1, TCOD_color_t c2, float coef);
/* HSV transformations */
void TCOD_color_set_HSV (TCOD_color_t *c,float h, float s, float v);
void TCOD_color_get_HSV (TCOD_color_t c,float * h, float * s, float * v);
float TCOD_color_get_hue (TCOD_color_t c);
void TCOD_color_set_hue (TCOD_color_t *c, float h);
float TCOD_color_get_saturation (TCOD_color_t c);
void TCOD_color_set_saturation (TCOD_color_t *c, float s);
float TCOD_color_get_value (TCOD_color_t c);
void TCOD_color_set_value (TCOD_color_t *c, float v);
void TCOD_color_shift_hue (TCOD_color_t *c, float hshift);
void TCOD_color_scale_HSV (TCOD_color_t *c, float scoef, float vcoef);
/* color map */
void TCOD_color_gen_map(TCOD_color_t *map, int nb_key, TCOD_color_t const *key_color, int const *key_index);

/* a cell in the console */
typedef struct {
	int c;		/* character ascii code */
	int cf;		/* character number in font */
	TCOD_color_t fore;	/* foreground color */
	TCOD_color_t back;	/* background color */
	uint8 dirt;	/* cell modified since last flush ? */
} char_t;

/* TCODConsole non public data */
typedef struct {
	char_t *buf; /* current console */
	char_t *oldbuf; /* console for last frame */
	/* console width and height (in characters,not pixels) */
	int w,h;
	/* default background operator for print & print_rect functions */
	TCOD_bkgnd_flag_t bkgnd_flag;
	/* default alignment for print & print_rect functions */
	TCOD_alignment_t alignment;
	/* foreground (text), background and key colors */
	TCOD_color_t fore,back,key;
	uint8 fade;
	bool haskey; /* a key color has been defined */
} TCOD_console_data_t;

/* fov internal stuff */
typedef struct {
	bool transparent:1;
	bool walkable:1;
	bool fov:1;
} cell_t;
typedef struct {
	int width;
	int height;
	int nbcells;
	cell_t *cells;
} map_t;

typedef struct {
	/* number of characters in the bitmap font */
	int fontNbCharHoriz;
	int fontNbCharVertic;
	/* font type and layout */
	bool font_tcod_layout;
	bool font_in_row;
	bool font_greyscale;
	/* character size in font */
	int font_width;
	int font_height;
	char font_file[512];
	char window_title[512];
	/* ascii code to tcod layout converter */
	int *ascii_to_tcod;
	/* whether each character in the font is a colored tile */
	bool *colored;
	/* the root console */
	TCOD_console_data_t *root;
	/* nb chars in the font */
	int max_font_chars;
	/* fullscreen data */
	bool fullscreen;
	int fullscreen_offsetx;
	int fullscreen_offsety;
	/* asked by the user */
	int fullscreen_width;
	int fullscreen_height;
	/* actual resolution */
	int actual_fullscreen_width;
	int actual_fullscreen_height;
	/* renderer to use */
	TCOD_renderer_t renderer;
	/* user post-processing callback */
	void* sdl_cbk;
	/* fading data */
	TCOD_color_t fading_color;
	uint8 fade;
} TCOD_internal_context_t;

extern TCOD_internal_context_t TCOD_ctx;





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


// IMAGE
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
void TCOD_mouse_includes_touch(bool enable);



// SYS
uint32 TCOD_sys_elapsed_milli();
float TCOD_sys_elapsed_seconds();
void TCOD_sys_sleep_milli(uint32 val);
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
void *TCOD_sys_get_sdl_window();

typedef enum {
  TCOD_EVENT_KEY_PRESS=1,
  TCOD_EVENT_KEY_RELEASE=2,
  //TCOD_EVENT_KEY=TCOD_EVENT_KEY_PRESS|TCOD_EVENT_KEY_RELEASE,
  TCOD_EVENT_MOUSE_MOVE=4,
  TCOD_EVENT_MOUSE_PRESS=8,
  TCOD_EVENT_MOUSE_RELEASE=16,
  //TCOD_EVENT_MOUSE=TCOD_EVENT_MOUSE_MOVE|TCOD_EVENT_MOUSE_PRESS|TCOD_EVENT_MOUSE_RELEASE,
  //TCOD_EVENT_ANY=TCOD_EVENT_KEY|TCOD_EVENT_MOUSE,
} TCOD_event_t;
TCOD_event_t TCOD_sys_wait_for_event(int eventMask, TCOD_key_t *key, TCOD_mouse_t *mouse, bool flush);
TCOD_event_t TCOD_sys_check_for_event(int eventMask, TCOD_key_t *key, TCOD_mouse_t *mouse);

/* filesystem stuff */
bool TCOD_sys_create_directory(const char *path);
bool TCOD_sys_delete_file(const char *path);
bool TCOD_sys_delete_directory(const char *path);
bool TCOD_sys_is_directory(const char *path);
//TCOD_list_t TCOD_sys_get_directory_content(const char *path, const char *pattern);
bool TCOD_sys_file_exists(const char * filename, ...);
bool TCOD_sys_read_file(const char *filename, unsigned char **buf, uint32 *size);
bool TCOD_sys_write_file(const char *filename, unsigned char *buf, uint32 size);

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
""")
#lib = ffi.dlopen('lib/win32/libtcod-mingw.dll')
ffi.set_source('_tcod_cffi', None)
ffi.dlopen('lib/win32/zlib1.dll')
ffi.dlopen('lib/win32/SDL.dll')
ffi.dlopen('lib/win32/libtcod-mingw.dll')

if __name__ == "__main__":
    ffi.compile()