#!/usr/bin/env python
#
# libtcod python samples
# This code demonstrates various usages of libtcod modules
# It's in the public domain.
#
from __future__ import division

import os

import math

import numpy as np
import tcod as libtcod

SAMPLE_SCREEN_WIDTH = 46
SAMPLE_SCREEN_HEIGHT = 20
SAMPLE_SCREEN_X = 20
SAMPLE_SCREEN_Y = 10
font = os.path.join('data/fonts/consolas10x10_gs_tc.png')
libtcod.console_set_custom_font(font, libtcod.FONT_TYPE_GREYSCALE | libtcod.FONT_LAYOUT_TCOD)
root_console = libtcod.console_init_root(80, 50, "tcod python sample", False)
sample_console = libtcod.console_new(SAMPLE_SCREEN_WIDTH, SAMPLE_SCREEN_HEIGHT)

class Sample():
    def __init__(self, name='', func=None):
        self.name = name
        self.func = func

    def on_enter(self):
        pass

    def on_draw(self, delta_time):
        pass

    def on_key(self, key):
        pass

    def on_mouse(self, mouse):
        pass

#############################################
# true color sample
#############################################
class TrueColorSample(Sample):

    def __init__(self):
        self.name = "True colors"
        # corner colors
        self.colors =  np.array([(50, 40, 150), (240, 85, 5),
                                (50, 35, 240), (10, 200, 130)], dtype=np.int16)
        # color shift direction
        self.slide_dir = np.array([[1, 1, 1], [-1, -1, 1],
                                   [1, -1, 1], [1, 1, -1]], dtype=np.int16)
        # corner indexes
        self.corners = np.array([0, 1, 2, 3])
        # sample screen mesh-grid
        self.mgrid = np.mgrid[0:1:SAMPLE_SCREEN_HEIGHT * 1j,
                              0:1:SAMPLE_SCREEN_WIDTH * 1j]

    def on_enter(self):
        libtcod.sys_set_fps(0)
        sample_console.clear()

    def on_draw(self, delta_time):
        self.slide_corner_colors()
        self.interpolate_corner_colors()
        self.darken_background_characters()
        self.randomize_sample_conole()
        self.print_banner()

    def slide_corner_colors(self):
        # pick random RGB channels for each corner
        rand_channels = np.random.randint(low=0, high=3, size=4)

        # shift picked color channels in the direction of slide_dir
        self.colors[self.corners, rand_channels] += \
            self.slide_dir[self.corners, rand_channels] * 5

        # reverse slide_dir values when limits are reached
        self.slide_dir[self.colors[:] == 255] = -1
        self.slide_dir[self.colors[:] == 0] = 1

    def interpolate_corner_colors(self):
        # interpolate corner colors across the sample console
        for i in range(3): # for each color channel
            left = ((self.colors[2,i] - self.colors[0,i]) * self.mgrid[0] +
                    self.colors[0,i])
            right = ((self.colors[3,i] - self.colors[1,i]) * self.mgrid[0] +
                     self.colors[1,i])
            sample_console.bg[:,:,i] = (right - left) * self.mgrid[1] + left

    def darken_background_characters(self):
        # darken background characters
        sample_console.fg[:] = sample_console.bg[:]
        sample_console.fg[:] //= 2

    def randomize_sample_conole(self):
        # randomize sample console characters
        sample_console.ch[:] = np.random.randint(
            low=ord('a'), high=ord('z') + 1,
            size=sample_console.ch.size,
            dtype=np.intc,
            ).reshape(sample_console.ch.shape)

    def print_banner(self):
        # print text on top of samples
        sample_console.default_bg = libtcod.grey
        sample_console.print_rect(
            x=sample_console.width // 2, y=5,
            width=sample_console.width - 2, height=sample_console.height - 1,
            string="The Doryen library uses 24 bits colors, for both "
                   "background and foreground.",
            bg_blend=libtcod.BKGND_MULTIPLY, alignment=libtcod.CENTER,
            )

#############################################
# offscreen console sample
#############################################

class OffscreenConsoleSample(Sample):

    def __init__(self):
        self.name = 'Offscreen console'
        self.secondary = libtcod.console.Console(sample_console.width // 2,
                                                 sample_console.height // 2)
        self.screenshot = libtcod.console.Console(sample_console.width,
                                                  sample_console.height)
        self.counter = 0
        self.x = 0
        self.y = 0
        self.xdir = 1
        self.ydir = 1

        self.secondary.print_frame(
            0, 0, sample_console.width // 2, sample_console.height // 2,
            "Offscreen console",
            False,
            libtcod.BKGND_NONE
            )

        self.secondary.print_rect(
            sample_console.width // 4, 2,
            sample_console.width // 2 - 2, sample_console.height // 2,
            "You can render to an offscreen console and blit in on another "
            "one, simulating alpha transparency.",
            libtcod.BKGND_NONE, libtcod.CENTER
            )

    def on_enter(self):
        libtcod.sys_set_fps(0)
        # get a "screenshot" of the current sample screen
        sample_console.blit(0, 0, sample_console.width, sample_console.height,
                            self.screenshot, 0, 0)

    def on_draw(self, delta_time):
        self.counter += delta_time * 1.5
        if self.counter >= 1:
            self.counter -= 1
            self.x += self.xdir
            self.y += self.ydir
            if self.x == sample_console.width / 2 + 5:
                self.xdir = -1
            elif self.x == -5:
                self.xdir = 1
            if self.y == sample_console.height / 2 + 5:
                self.ydir = -1
            elif self.y == -5:
                self.ydir = 1
        self.screenshot.blit(0, 0, sample_console.width, sample_console.height,
                             sample_console, 0, 0)
        self.secondary.blit(
            0, 0, sample_console.width // 2, sample_console.height // 2,
            sample_console, self.x, self.y, 1.0, 0.75
            )

#############################################
# line drawing sample
#############################################

class LineDrawingSample(Sample):

    FLAG_NAMES=['BKGND_NONE',
                'BKGND_SET',
                'BKGND_MULTIPLY',
                'BKGND_LIGHTEN',
                'BKGND_DARKEN',
                'BKGND_SCREEN',
                'BKGND_COLOR_DODGE',
                'BKGND_COLOR_BURN',
                'BKGND_ADD',
                'BKGND_ADDALPHA',
                'BKGND_BURN',
                'BKGND_OVERLAY',
                'BKGND_ALPHA',
                ]

    def __init__(self):
        self.name = 'Line drawing'
        self.mk_flag = libtcod.BKGND_SET
        self.bk_flag = libtcod.BKGND_SET

        self.bk = libtcod.console_new(sample_console.width, sample_console.height)
        # initialize the colored background
        for x in range(sample_console.width):
            for y in range(sample_console.height):
                col = libtcod.Color(x * 255 // (sample_console.width - 1),
                                    (x + y) * 255 // (sample_console.width - 1 +
                                    sample_console.height - 1),
                                    y * 255 // (sample_console.height-1))
                self.bk.bg[y, x] = col
                self.bk.ch[:] = ord(' ')


    def on_key(self, key):
        if key.vk in (libtcod.KEY_ENTER, libtcod.KEY_KPENTER):
            self.bk_flag += 1
            if (self.bk_flag & 0xff) > libtcod.BKGND_ALPH:
                self.bk_flag=libtcod.BKGND_NONE

    def on_enter(self):
        libtcod.sys_set_fps(0)
        libtcod.console_set_default_foreground(sample_console, libtcod.white)

    def on_draw(self, delta_time):
        alpha = 0.0
        if (self.bk_flag & 0xff) == libtcod.BKGND_ALPH:
            # for the alpha mode, update alpha every frame
            alpha = (1.0 + math.cos(libtcod.sys_elapsed_seconds() * 2)) / 2.0
            self.bk_flag = libtcod.BKGND_ALPHA(alpha)
        elif (self.bk_flag & 0xff) == libtcod.BKGND_ADDA:
            # for the add alpha mode, update alpha every frame
            alpha = (1.0 + math.cos(libtcod.sys_elapsed_seconds() * 2)) / 2.0
            self.bk_flag = libtcod.BKGND_ADDALPHA(alpha)

        self.bk.blit(0, 0, sample_console.width,
                     sample_console.height, sample_console, 0, 0)
        recty = int((sample_console.height - 2) * ((1.0 +
                    math.cos(libtcod.sys_elapsed_seconds())) / 2.0))
        for x in range(sample_console.width):
            col = libtcod.Color(x * 255 // sample_console.width,
                                x * 255 // sample_console.width,
                                x * 255 // sample_console.width)
            libtcod.console_set_char_background(
                sample_console, x, recty, col, self.bk_flag)
            libtcod.console_set_char_background(
                sample_console, x, recty + 1, col, self.bk_flag)
            libtcod.console_set_char_background(
                sample_console, x, recty + 2, col, self.bk_flag)
        angle = libtcod.sys_elapsed_seconds() * 2.0
        cos_angle=math.cos(angle)
        sin_angle=math.sin(angle)
        xo = int(sample_console.width // 2 * (1 + cos_angle))
        yo = int(sample_console.height // 2 + sin_angle * sample_console.width // 2)
        xd = int(sample_console.width // 2 * (1 - cos_angle))
        yd = int(sample_console.height // 2 - sin_angle * sample_console.width // 2)
        # draw the line
        # in python the easiest way is to use the line iterator
        for x, y in libtcod.line_iter(xo, yo, xd, yd):
            if 0 <= x < sample_console.width and \
               0 <= y < sample_console.height:
                libtcod.console_set_char_background(
                    sample_console, x, y, libtcod.light_blue, self.bk_flag)
        sample_console.print_(
            2, 2,
            '%s (ENTER to change)' % self.FLAG_NAMES[self.bk_flag & 0xff]
            )

#############################################
# noise sample
#############################################

NOISE_OPTIONS = [ # [name, algorithm, implementation],
    ['perlin noise', libtcod.NOISE_PERLIN, libtcod.noise.SIMPLE],
    ['simplex noise', libtcod.NOISE_SIMPLEX, libtcod.noise.SIMPLE],
    ['wavelet noise', libtcod.NOISE_WAVELET, libtcod.noise.SIMPLE],
    ['perlin fbm', libtcod.NOISE_PERLIN, libtcod.noise.FBM],
    ['perlin turbulence', libtcod.NOISE_PERLIN, libtcod.noise.TURBULENCE],
    ['simplex fbm', libtcod.NOISE_SIMPLEX, libtcod.noise.FBM],
    ['simplex turbulence',
     libtcod.NOISE_SIMPLEX, libtcod.noise.TURBULENCE],
    ['wavelet fbm', libtcod.NOISE_WAVELET, libtcod.noise.FBM],
    ['wavelet turbulence',
     libtcod.NOISE_WAVELET, libtcod.noise.TURBULENCE],
    ]

class NoiseSample(Sample):

    def __init__(self):
        self.name = 'Noise'
        self.func = 0
        self.dx = 0.0
        self.dy = 0.0
        self.octaves = 4.0
        self.zoom = 3.0
        self.hurst = libtcod.NOISE_DEFAULT_HURST
        self.lacunarity = libtcod.NOISE_DEFAULT_LACUNARITY
        self.noise = self.get_noise()
        self.img=libtcod.image_new(SAMPLE_SCREEN_WIDTH*2,SAMPLE_SCREEN_HEIGHT*2)

    @property
    def algorithm(self):
        return NOISE_OPTIONS[self.func][1]

    @property
    def implementation(self):
        return NOISE_OPTIONS[self.func][2]

    def get_noise(self):
        return libtcod.noise.Noise(
            2,
            self.algorithm,
            self.implementation,
            self.hurst,
            self.lacunarity,
            self.octaves,
            seed=None,
            )

    def on_enter(self):
        libtcod.sys_set_fps(0)

    def on_draw(self, delta_time):
        sample_console.clear()
        self.dx += delta_time * 0.25
        self.dy += delta_time * 0.25
        for y in range(2 * sample_console.height):
            for x in range(2 * sample_console.width):
                f = [self.zoom * x / (2 * sample_console.width) + self.dx,
                     self.zoom * y / (2 * sample_console.height) + self.dy]
                value = self.noise.get_point(*f)
                c = int((value + 1.0) / 2.0 * 255)
                c = max(0, min(c, 255))
                self.img.put_pixel(x, y, (c // 2, c // 2, c))
        sample_console.default_bg = libtcod.grey
        rectw = 24
        recth = 13
        if self.implementation == libtcod.noise.SIMPLE:
            recth = 10
        self.img.blit_2x(sample_console, 0, 0)
        sample_console.default_bg = libtcod.grey
        sample_console.rect(2, 2, rectw, recth, False, libtcod.BKGND_MULTIPLY)
        sample_console.fg[2:2+recth,2:2+rectw] = \
            (sample_console.fg[2:2+recth,2:2+rectw] *
             sample_console.default_bg / 255)

        for curfunc in range(len(NOISE_OPTIONS)):
            text = '%i : %s' % (curfunc + 1, NOISE_OPTIONS[curfunc][0])
            if curfunc == self.func:
                sample_console.default_fg = libtcod.white
                sample_console.default_bg = libtcod.light_blue
                sample_console.print_(2, 2 + curfunc, text,
                                      libtcod.BKGND_SET, libtcod.LEFT)
            else:
                sample_console.default_fg = libtcod.grey
                sample_console.print_(2, 2 + curfunc, text)
        sample_console.default_fg = libtcod.white
        sample_console.print_(2, 11, 'Y/H : zoom (%2.1f)' % self.zoom)
        if self.implementation != libtcod.noise.SIMPLE:
            sample_console.print_(2, 12, 'E/D : hurst (%2.1f)' % self.hurst)
            sample_console.print_(2, 13,
                                  'R/F : lacunarity (%2.1f)' %
                                  self.lacunarity)
            sample_console.print_(2, 14,
                                  'T/G : octaves (%2.1f)' % self.octaves)

    def on_key(self, key):
        if key.vk == libtcod.KEY_NONE:
            return
        if ord('9') >= key.c >= ord('1'):
            self.func = key.c - ord('1')
            self.noise = self.get_noise()
        elif key.c in (ord('E'), ord('e')):
            self.hurst += 0.1
            self.noise = self.get_noise()
        elif key.c in (ord('D'), ord('d')):
            self.hurst -= 0.1
            self.noise = self.get_noise()
        elif key.c in (ord('R'), ord('r')):
            self.lacunarity += 0.5
            self.noise = self.get_noise()
        elif key.c in (ord('F'), ord('f')):
            self.lacunarity -= 0.5
            self.noise = self.get_noise()
        elif key.c in (ord('T'), ord('t')):
            self.octaves += 0.5
            self.noise.octaves = self.octaves
        elif key.c in (ord('G'), ord('g')):
            self.octaves -= 0.5
            self.noise.octaves = self.octaves
        elif key.c in (ord('Y'), ord('y')):
            self.zoom += 0.2
        elif key.c in (ord('H'), ord('h')):
            self.zoom -= 0.2

#############################################
# field of view sample
#############################################
DARK_WALL = libtcod.Color(0, 0, 100)
LIGHT_WALL = libtcod.Color(130, 110, 50)
DARK_GROUND = libtcod.Color(50, 50, 150)
LIGHT_GROUND = libtcod.Color(200, 180, 50)

SAMPLE_MAP = [
    '##############################################',
    '#######################      #################',
    '#####################    #     ###############',
    '######################  ###        ###########',
    '##################      #####             ####',
    '################       ########    ###### ####',
    '###############      #################### ####',
    '################    ######                  ##',
    '########   #######  ######   #     #     #  ##',
    '########   ######      ###                  ##',
    '########                                    ##',
    '####       ######      ###   #     #     #  ##',
    '#### ###   ########## ####                  ##',
    '#### ###   ##########   ###########=##########',
    '#### ##################   #####          #####',
    '#### ###             #### #####          #####',
    '####           #     ####                #####',
    '########       #     #### #####          #####',
    '########       #####      ####################',
    '##############################################',
    ]

SAMPLE_MAP = np.array([list(line) for line in SAMPLE_MAP])

FOV_ALGO_NAMES = [
    'BASIC      ',
    'DIAMOND    ',
    'SHADOW     ',
    'PERMISSIVE0',
    'PERMISSIVE1',
    'PERMISSIVE2',
    'PERMISSIVE3',
    'PERMISSIVE4',
    'PERMISSIVE5',
    'PERMISSIVE6',
    'PERMISSIVE7',
    'PERMISSIVE8',
    'RESTRICTIVE',
    ]

TORCH_RADIUS = 10
SQUARED_TORCH_RADIUS = TORCH_RADIUS * TORCH_RADIUS

class FOVSample(Sample):

    def __init__(self):
        self.name = 'Field of view'

        self.px = 20
        self.py = 10
        self.recompute = True
        self.torch = False
        self.map = None
        self.noise = None
        self.torchx = 0.0
        self.light_walls = True
        self.algo_num = 0
        # 1d noise for the torch flickering
        self.noise = libtcod.noise_new(1, 1.0, 1.0)

        self.map = libtcod.map_new(SAMPLE_SCREEN_WIDTH, SAMPLE_SCREEN_HEIGHT)
        self.map.walkable[:] = SAMPLE_MAP[:] == ' '
        self.map.transparent[:] = self.map.walkable[:] | (SAMPLE_MAP == '=')

        self.light_map_bg = np.full(SAMPLE_MAP.shape + (3,), LIGHT_GROUND,
                                    dtype=np.uint8)
        self.light_map_bg[SAMPLE_MAP[:] == '#'] = LIGHT_WALL
        self.dark_map_bg = np.full(SAMPLE_MAP.shape + (3,), DARK_GROUND,
                                   dtype=np.uint8)
        self.dark_map_bg[SAMPLE_MAP[:] == '#'] = DARK_WALL

    def draw_ui(self):
        libtcod.console_set_default_foreground(sample_console, libtcod.white)
        libtcod.console_print(
            sample_console, 1, 1,
            "IJKL : move around\n"
            "T : torch fx %s\n"
            "W : light walls %s\n"
            "+-: algo %s" %
                ('on ' if self.torch else 'off',
                 'on ' if self.light_walls else 'off',
                 FOV_ALGO_NAMES[self.algo_num],
                 ),
             )
        libtcod.console_set_default_foreground(sample_console, libtcod.black)

    def on_enter(self):
        libtcod.sys_set_fps(60)
        # we draw the foreground only the first time.
        #  during the player movement, only the @ is redrawn.
        #  the rest impacts only the background color
        # draw the help text & player @
        libtcod.console_clear(sample_console)
        self.draw_ui()
        libtcod.console_put_char(sample_console, self.px, self.py, '@',
                                 libtcod.BKGND_NONE)
        # draw windows
        sample_console.ch[np.where(SAMPLE_MAP == '=')] = libtcod.CHAR_DHLINE
        sample_console.fg[np.where(SAMPLE_MAP == '=')] = libtcod.black

    def on_draw(self, delta_time):
        dx = 0.0
        dy = 0.0
        di = 0.0
        if self.recompute:
            self.recompute = False
            self.map.compute_fov(
                self.px,
                self.py,
                TORCH_RADIUS if self.torch else 0,
                self.light_walls,
                self.algo_num
                )
        sample_console.bg[:] = self.dark_map_bg[:]
        if self.torch:
            # slightly change the perlin noise parameter
            self.torchx += 0.1
            # randomize the light position between -1.5 and 1.5
            tdx = [self.torchx + 20.0]
            dx =  libtcod.noise_get(self.noise, tdx, libtcod.NOISE_SIMPLEX) * 1.5
            tdx[0] += 30.0
            dy =  libtcod.noise_get(self.noise, tdx, libtcod.NOISE_SIMPLEX) * 1.5
            di = 0.2 * libtcod.noise_get(self.noise, [self.torchx], libtcod.NOISE_SIMPLEX)
            #where_fov = np.where(self.map.fov[:])
            mgrid = np.mgrid[:SAMPLE_SCREEN_HEIGHT,:SAMPLE_SCREEN_WIDTH]
            # get squared distance
            light = ((mgrid[0] - self.py + dy) ** 2 +
                     (mgrid[1] - self.px + dx) ** 2)
            light = light.astype(np.float16)
            where_visible = np.where((light < SQUARED_TORCH_RADIUS) &
                                     self.map.fov[:])
            light[where_visible] = SQUARED_TORCH_RADIUS - light[where_visible]
            light[where_visible] /= SQUARED_TORCH_RADIUS
            light[where_visible] += di
            light[where_visible] = light[where_visible].clip(0, 1)

            for yx in zip(*where_visible):
                sample_console.bg[yx] = libtcod.color_lerp(
                    tuple(self.dark_map_bg[yx]),
                    tuple(self.light_map_bg[yx]),
                    light[yx],
                    )
        else:
            where_fov = np.where(self.map.fov[:])
            sample_console.bg[where_fov] = self.light_map_bg[where_fov]


    def on_key(self, key):
        MOVE_KEYS = {
            ord('i'): (0, -1),
            ord('j'): (-1, 0),
            ord('k'): (0, 1),
            ord('l'): (1, 0),
        }
        FOV_SELECT_KEYS = {ord('-'): -1, ord('='): 1}
        if key.c in MOVE_KEYS:
            x, y = MOVE_KEYS[key.c]
            if SAMPLE_MAP[self.py + y][self.px + x] == ' ':
                libtcod.console_put_char(sample_console, self.px, self.py, ' ',
                                         libtcod.BKGND_NONE)
                self.px += x
                self.py += y
                libtcod.console_put_char(sample_console, self.px, self.py, '@',
                                         libtcod.BKGND_NONE)
                self.recompute = True
        elif key.c == ord('t'):
            self.torch = not self.torch
            self.draw_ui()
            self.recompute = True
        elif key.c == ord('w'):
            self.light_walls = not self.light_walls
            self.draw_ui()
            self.recompute = True
        elif key.c in FOV_SELECT_KEYS:
            self.algo_num += FOV_SELECT_KEYS[key.c]
            self.algo_num %= libtcod.NB_FOV_ALGORITHMS
            self.draw_ui()
            self.recompute = True

#############################################
# pathfinding sample
#############################################

class PathfindingSample(Sample):
    def __init__(self):
        self.name = 'Path finding'

        self.px = 20
        self.py = 10
        self.dx = 24
        self.dy = 1
        self.map = None
        self.path = None
        self.dijk_dist = 0.0
        self.using_astar = True
        self.dijk = None
        self.recalculate = False
        self.busy = 0.0
        self.oldchar = ' '

        self.map = libtcod.map_new(SAMPLE_SCREEN_WIDTH, SAMPLE_SCREEN_HEIGHT)
        for y in range(SAMPLE_SCREEN_HEIGHT):
            for x in range(SAMPLE_SCREEN_WIDTH):
                if SAMPLE_MAP[y][x] == ' ':
                    # ground
                    libtcod.map_set_properties(self.map, x, y, True, True)
                elif SAMPLE_MAP[y][x] == '=':
                    # window
                    libtcod.map_set_properties(self.map, x, y, True, False)
        self.path = libtcod.path_new_using_map(self.map)
        self.dijk = libtcod.dijkstra_new(self.map)

    def on_enter(self):
        libtcod.sys_set_fps(60)
        # we draw the foreground only the first time.
        #  during the player movement, only the @ is redrawn.
        #  the rest impacts only the background color
        # draw the help text & player @
        libtcod.console_clear(sample_console)
        libtcod.console_set_default_foreground(sample_console, libtcod.white)
        libtcod.console_put_char(sample_console, self.dx, self.dy, '+',
                                 libtcod.BKGND_NONE)
        libtcod.console_put_char(sample_console, self.px, self.py, '@',
                                 libtcod.BKGND_NONE)
        libtcod.console_print(sample_console, 1, 1,
                                   "IJKL / mouse :\nmove destination\nTAB : A*/dijkstra")
        libtcod.console_print(sample_console, 1, 4,
                                    "Using : A*")
        # draw windows
        for y in range(SAMPLE_SCREEN_HEIGHT):
            for x in range(SAMPLE_SCREEN_WIDTH):
                if SAMPLE_MAP[y][x] == '=':
                    libtcod.console_put_char(sample_console, x, y,
                                             libtcod.CHAR_DHLINE,
                                             libtcod.BKGND_NONE)
        self.recalculate = True

    def on_draw(self, delta_time):
        if self.recalculate:
            if self.using_astar :
                libtcod.path_compute(self.path, self.px, self.py, self.dx, self.dy)
            else:
                self.dijk_dist = 0.0
                # compute dijkstra grid (distance from px,py)
                libtcod.dijkstra_compute(self.dijk,self.px,self.py)
                # get the maximum distance (needed for rendering)
                for y in range(SAMPLE_SCREEN_HEIGHT):
                    for x in range(SAMPLE_SCREEN_WIDTH):
                        d=libtcod.dijkstra_get_distance(self.dijk,x,y)
                        if d > self.dijk_dist:
                            self.dijk_dist=d
                # compute path from px,py to dx,dy
                libtcod.dijkstra_path_set(self.dijk,self.dx,self.dy)
            self.recalculate = False
            self.busy = 0.2
        # draw the dungeon
        for y in range(SAMPLE_SCREEN_HEIGHT):
            for x in range(SAMPLE_SCREEN_WIDTH):
                if SAMPLE_MAP[y][x] == '#':
                    libtcod.console_set_char_background(sample_console, x, y, DARK_WALL,
                                             libtcod.BKGND_SET)
                else:
                    libtcod.console_set_char_background(sample_console, x, y, DARK_GROUND,
                                             libtcod.BKGND_SET)
        # draw the path
        if self.using_astar :
            for i in range(libtcod.path_size(self.path)):
                x,y = libtcod.path_get(self.path, i)
                libtcod.console_set_char_background(sample_console, x, y,
                                     LIGHT_GROUND, libtcod.BKGND_SET)
        else:
            for y in range(SAMPLE_SCREEN_HEIGHT):
                for x in range(SAMPLE_SCREEN_WIDTH):
                    if SAMPLE_MAP[y][x] != '#':
                        libtcod.console_set_char_background(sample_console, x, y, libtcod.color_lerp(LIGHT_GROUND,DARK_GROUND,
                            0.9 * libtcod.dijkstra_get_distance(self.dijk,x,y) / self.dijk_dist), libtcod.BKGND_SET)
            for i in range(libtcod.dijkstra_size(self.dijk)):
                x,y=libtcod.dijkstra_get(self.dijk,i)
                libtcod.console_set_char_background(sample_console,x,y,LIGHT_GROUND, libtcod.BKGND_SET )

        # move the creature
        self.busy -= libtcod.sys_get_last_frame_length()
        if self.busy <= 0.0:
            self.busy = 0.2
            if self.using_astar :
                if not libtcod.path_is_empty(self.path):
                    libtcod.console_put_char(sample_console, self.px, self.py, ' ',
                                             libtcod.BKGND_NONE)
                    self.px, self.py = libtcod.path_walk(self.path, True)
                    libtcod.console_put_char(sample_console, self.px, self.py, '@',
                                             libtcod.BKGND_NONE)
            else:
                if not libtcod.dijkstra_is_empty(self.dijk):
                    libtcod.console_put_char(sample_console, self.px, self.py, ' ',
                                             libtcod.BKGND_NONE)
                    self.px, self.py = libtcod.dijkstra_path_walk(self.dijk)
                    libtcod.console_put_char(sample_console, self.px, self.py, '@',
                                             libtcod.BKGND_NONE)
                    self.recalculate = True

    def on_key(self, key):
        if key.c in (ord('I'), ord('i')) and self.dy > 0:
            # destination move north
            libtcod.console_put_char(sample_console, self.dx, self.dy, self.oldchar,
                                     libtcod.BKGND_NONE)
            self.dy -= 1
            self.oldchar = libtcod.console_get_char(sample_console, self.dx,
                                                    self.dy)
            libtcod.console_put_char(sample_console, self.dx, self.dy, '+',
                                     libtcod.BKGND_NONE)
            if SAMPLE_MAP[self.dy][self.dx] == ' ':
                self.recalculate = True
        elif key.c in (ord('K'), ord('k')) and self.dy < SAMPLE_SCREEN_HEIGHT - 1:
            # destination move south
            libtcod.console_put_char(sample_console, self.dx, self.dy, self.oldchar,
                                     libtcod.BKGND_NONE)
            self.dy += 1
            self.oldchar = libtcod.console_get_char(sample_console, self.dx,
                                                    self.dy)
            libtcod.console_put_char(sample_console, self.dx, self.dy, '+',
                                     libtcod.BKGND_NONE)
            if SAMPLE_MAP[self.dy][self.dx] == ' ':
                self.recalculate = True
        elif key.c in (ord('J'), ord('j')) and self.dx > 0:
            # destination move west
            libtcod.console_put_char(sample_console, self.dx, self.dy, self.oldchar,
                                     libtcod.BKGND_NONE)
            self.dx -= 1
            self.oldchar = libtcod.console_get_char(sample_console, self.dx,
                                                    self.dy)
            libtcod.console_put_char(sample_console, self.dx, self.dy, '+',
                                     libtcod.BKGND_NONE)
            if SAMPLE_MAP[self.dy][self.dx] == ' ':
                self.recalculate = True
        elif key.c in (ord('L'), ord('l')) and self.dx < SAMPLE_SCREEN_WIDTH - 1:
            # destination move east
            libtcod.console_put_char(sample_console, self.dx, self.dy, self.oldchar,
                                     libtcod.BKGND_NONE)
            self.dx += 1
            self.oldchar = libtcod.console_get_char(sample_console, self.dx,
                                                    self.dy)
            libtcod.console_put_char(sample_console, self.dx, self.dy, '+',
                                     libtcod.BKGND_NONE)
            if SAMPLE_MAP[self.dy][self.dx] == ' ':
                self.recalculate = True
        elif key.vk == libtcod.KEY_TAB:
            self.using_astar = not self.using_astar
            if self.using_astar :
                libtcod.console_print(sample_console, 1, 4,
                                        "Using : A*      ")
            else:
                libtcod.console_print(sample_console, 1, 4,
                                        "Using : Dijkstra")
            self.recalculate=True

    def on_mouse(self, mouse):
        mx = mouse.cx - SAMPLE_SCREEN_X
        my = mouse.cy - SAMPLE_SCREEN_Y
        if 0 <= mx < SAMPLE_SCREEN_WIDTH and 0 <= my < SAMPLE_SCREEN_HEIGHT  and \
            (self.dx != mx or self.dy != my):
            libtcod.console_put_char(sample_console, self.dx, self.dy, self.oldchar,
                                     libtcod.BKGND_NONE)
            self.dx = mx
            self.dy = my
            self.oldchar = libtcod.console_get_char(sample_console, self.dx,
                                                    self.dy)
            libtcod.console_put_char(sample_console, self.dx, self.dy, '+',
                                     libtcod.BKGND_NONE)
            if SAMPLE_MAP[self.dy][self.dx] == ' ':
                self.recalculate = True

#############################################
# bsp sample
#############################################
bsp_depth = 8
bsp_min_room_size = 4
# a room fills a random part of the node or the maximum available space ?
bsp_random_room = False
# if true, there is always a wall on north & west side of a room
bsp_room_walls = True
bsp_map = None
# draw a vertical line
def vline(m, x, y1, y2):
    if y1 > y2:
        y1,y2 = y2,y1
    for y in range(y1,y2+1):
        m[x][y] = True

# draw a vertical line up until we reach an empty space
def vline_up(m, x, y):
    while y >= 0 and not m[x][y]:
        m[x][y] = True
        y -= 1

# draw a vertical line down until we reach an empty space
def vline_down(m, x, y):
    while y < SAMPLE_SCREEN_HEIGHT and not m[x][y]:
        m[x][y] = True
        y += 1

# draw a horizontal line
def hline(m, x1, y, x2):
    if x1 > x2:
        x1,x2 = x2,x1
    for x in range(x1,x2+1):
        m[x][y] = True

# draw a horizontal line left until we reach an empty space
def hline_left(m, x, y):
    while x >= 0 and not m[x][y]:
        m[x][y] = True
        x -= 1

# draw a horizontal line right until we reach an empty space
def hline_right(m, x, y):
    while x < SAMPLE_SCREEN_WIDTH and not m[x][y]:
        m[x][y]=True
        x += 1

# the class building the dungeon from the bsp nodes
def traverse_node(node, *dat):
    global bsp_map
    if libtcod.bsp_is_leaf(node):
        # calculate the room size
        minx = node.x + 1
        maxx = node.x + node.w - 1
        miny = node.y + 1
        maxy = node.y + node.h - 1
        if not bsp_room_walls:
            if minx > 1:
                minx -= 1
            if miny > 1:
                miny -=1
        if maxx == SAMPLE_SCREEN_WIDTH - 1:
            maxx -= 1
        if maxy == SAMPLE_SCREEN_HEIGHT - 1:
            maxy -= 1
        if bsp_random_room:
            minx = libtcod.random_get_int(None, minx, maxx - bsp_min_room_size + 1)
            miny = libtcod.random_get_int(None, miny, maxy - bsp_min_room_size + 1)
            maxx = libtcod.random_get_int(None, minx + bsp_min_room_size - 1, maxx)
            maxy = libtcod.random_get_int(None, miny + bsp_min_room_size - 1, maxy)
        # resize the node to fit the room
        node.x = minx
        node.y = miny
        node.w = maxx-minx + 1
        node.h = maxy-miny + 1
        # dig the room
        for x in range(minx, maxx + 1):
            for y in range(miny, maxy + 1):
                bsp_map[x][y] = True
    else:
        # resize the node to fit its sons
        left = libtcod.bsp_left(node)
        right = libtcod.bsp_right(node)
        node.x = min(left.x, right.x)
        node.y = min(left.y, right.y)
        node.w = max(left.x + left.w, right.x + right.w) - node.x
        node.h = max(left.y + left.h, right.y + right.h) - node.y
        # create a corridor between the two lower nodes
        if node.horizontal:
            # vertical corridor
            if left.x + left.w - 1 < right.x or right.x + right.w - 1 < left.x:
                # no overlapping zone. we need a Z shaped corridor
                x1 = libtcod.random_get_int(None, left.x, left.x + left.w - 1)
                x2 = libtcod.random_get_int(None, right.x, right.x + right.w - 1)
                y = libtcod.random_get_int(None, left.y + left.h, right.y)
                vline_up(bsp_map, x1, y - 1)
                hline(bsp_map, x1, y, x2)
                vline_down(bsp_map, x2, y + 1)
            else:
                # straight vertical corridor
                minx = max(left.x, right.x)
                maxx = min(left.x + left.w - 1, right.x + right.w - 1)
                x = libtcod.random_get_int(None, minx, maxx)
                vline_down(bsp_map, x, right.y)
                vline_up(bsp_map, x, right.y - 1)
        else:
            # horizontal corridor
            if left.y + left.h - 1 < right.y or right.y + right.h - 1 < left.y:
                # no overlapping zone. we need a Z shaped corridor
                y1 = libtcod.random_get_int(None, left.y, left.y + left.h - 1)
                y2 = libtcod.random_get_int(None, right.y, right.y + right.h - 1)
                x = libtcod.random_get_int(None, left.x + left.w, right.x)
                hline_left(bsp_map, x - 1, y1)
                vline(bsp_map, x, y1, y2)
                hline_right(bsp_map, x + 1, y2)
            else:
                # straight horizontal corridor
                miny = max(left.y, right.y)
                maxy = min(left.y + left.h - 1, right.y + right.h - 1)
                y = libtcod.random_get_int(None, miny, maxy)
                hline_left(bsp_map, right.x - 1, y)
                hline_right(bsp_map, right.x, y)
    return True

bsp = None
bsp_generate = True
bsp_refresh = False
class BSPSample(Sample):
    def __init__(self):
        self.name = 'Bsp toolkit'

    def on_draw(self, delta_time):
        global bsp, bsp_generate, bsp_refresh, bsp_map
        global bsp_random_room, bsp_room_walls, bsp_depth, bsp_min_room_size
        if bsp_generate or bsp_refresh:
            # dungeon generation
            if bsp is None:
                # create the bsp
                bsp = libtcod.bsp_new_with_size(0, 0, SAMPLE_SCREEN_WIDTH,
                                                SAMPLE_SCREEN_HEIGHT)
            else:
                # restore the nodes size
                libtcod.bsp_resize(bsp, 0, 0, SAMPLE_SCREEN_WIDTH,
                                   SAMPLE_SCREEN_HEIGHT)
            bsp_map = list()
            for x in range(SAMPLE_SCREEN_WIDTH):
                bsp_map.append([False] * SAMPLE_SCREEN_HEIGHT)
            if bsp_generate:
                # build a new random bsp tree
                libtcod.bsp_remove_sons(bsp)
                if bsp_room_walls:
                    libtcod.bsp_split_recursive(bsp, 0, bsp_depth,
                                                bsp_min_room_size + 1,
                                                bsp_min_room_size + 1, 1.5, 1.5)
                else:
                    libtcod.bsp_split_recursive(bsp, 0, bsp_depth,
                                                bsp_min_room_size,
                                                bsp_min_room_size, 1.5, 1.5)
            # create the dungeon from the bsp
            libtcod.bsp_traverse_inverted_level_order(bsp, traverse_node)
            bsp_generate = False
            bsp_refresh = False
        libtcod.console_clear(sample_console)
        libtcod.console_set_default_foreground(sample_console, libtcod.white)
        rooms = 'OFF'
        if bsp_random_room:
            rooms = 'ON'
        libtcod.console_print(sample_console, 1, 1,
                                   "ENTER : rebuild bsp\n"
                                   "SPACE : rebuild dungeon\n"
                                   "+-: bsp depth %d\n"
                                   "*/: room size %d\n"
                                   "1 : random room size %s" % (bsp_depth,
                                   bsp_min_room_size, rooms))
        if bsp_random_room:
            walls = 'OFF'
            if bsp_room_walls:
                walls ='ON'
            libtcod.console_print(sample_console, 1, 6,
                                       '2 : room walls %s' % walls)
        # render the level
        for y in range(SAMPLE_SCREEN_HEIGHT):
            for x in range(SAMPLE_SCREEN_WIDTH):
                if not bsp_map[x][y]:
                    libtcod.console_set_char_background(sample_console, x, y, DARK_WALL,
                                             libtcod.BKGND_SET)
                else:
                    libtcod.console_set_char_background(sample_console, x, y, DARK_GROUND,
                                             libtcod.BKGND_SET)

    def on_key(self, key):
        global bsp, bsp_generate, bsp_refresh, bsp_map
        global bsp_random_room, bsp_room_walls, bsp_depth, bsp_min_room_size
        if key.vk in (libtcod.KEY_ENTER ,libtcod.KEY_KPENTER):
            bsp_generate = True
        elif key.c==ord(' '):
            bsp_refresh = True
        elif key.c == ord('='):
            bsp_depth += 1
            bsp_generate = True
        elif key.c == ord('-') and bsp_depth > 1:
            bsp_depth -= 1
            bsp_generate = True
        elif key.c==ord('*'):
            bsp_min_room_size += 1
            bsp_generate = True
        elif key.c == ord('/') and bsp_min_room_size > 2:
            bsp_min_room_size -= 1
            bsp_generate = True
        elif key.c == ord('1') or key.vk in (libtcod.KEY_1, libtcod.KEY_KP1):
            bsp_random_room = not bsp_random_room
            if not bsp_random_room:
                bsp_room_walls = True
            bsp_refresh = True
        elif key.c == ord('2') or key.vk in (libtcod.KEY_2, libtcod.KEY_KP2):
            bsp_room_walls = not bsp_room_walls
            bsp_refresh = True

#############################################
# image sample
#############################################
img_blue = libtcod.Color(0, 0, 255)
img_green = libtcod.Color(0, 255, 0)
class ImageSample(Sample):
    def __init__(self):
        self.name = 'Image toolkit'

        self.img = libtcod.image_load('data/img/skull.png')
        self.img.set_key_color(libtcod.black)
        self.circle = libtcod.image_load('data/img/circle.png')

    def on_enter(self):
        libtcod.sys_set_fps(0)

    def on_draw(self, delta_time):
        sample_console.default_bg = libtcod.black
        sample_console.clear()
        x = sample_console.width / 2 + math.cos(libtcod.sys_elapsed_seconds()) * 10.0
        y = float(sample_console.height / 2)
        scalex= 0.2 + 1.8 * (1.0 + math.cos(libtcod.sys_elapsed_seconds() / 2)) / 2.0
        scaley = scalex
        angle = libtcod.sys_elapsed_seconds()
        elapsed = libtcod.sys_elapsed_milli() // 2000
        if elapsed & 1 != 0:
            # split the color channels of circle.png
            # the red channel
            sample_console.default_bg = libtcod.red

            sample_console.rect(0, 3, 15, 15, False, libtcod.BKGND_SET)
            self.circle.blit_rect(sample_console, 0, 3, -1, -1,
                                    libtcod.BKGND_MULTIPLY)
            # the green channel
            sample_console.default_bg = img_green
            sample_console.rect(15, 3, 15, 15, False, libtcod.BKGND_SET)
            self.circle.blit_rect(sample_console,
                                  15, 3, -1, -1, libtcod.BKGND_MULTIPLY)
            # the blue channel
            sample_console.default_bg = img_blue
            sample_console.rect(30, 3, 15, 15, False, libtcod.BKGND_SET)
            self.circle.blit_rect(sample_console,
                                  30, 3, -1, -1, libtcod.BKGND_MULTIPLY)
        else:
            # render circle.png with normal blitting
            self.circle.blit_rect(sample_console,
                                  0, 3, -1, -1, libtcod.BKGND_SET)
            self.circle.blit_rect(sample_console,
                                  15, 3, -1, -1, libtcod.BKGND_SET)
            self.circle.blit_rect(sample_console,
                                  30, 3, -1, -1, libtcod.BKGND_SET)
        self.img.blit(sample_console, x, y,
                      libtcod.BKGND_SET, scalex, scaley, angle)

#############################################
# mouse sample
#############################################
butstatus=('OFF', 'ON')

class MouseSample(Sample):
    def __init__(self):
        self.name = 'Mouse support'

        self.lbut = self.mbut = self.rbut = 0

    def on_enter(self):
        libtcod.console_set_default_background(sample_console, libtcod.grey)
        libtcod.console_set_default_foreground(sample_console,
                                             libtcod.light_yellow)
        libtcod.mouse_move(320, 200)
        libtcod.mouse_show_cursor(True)
        libtcod.sys_set_fps(60)

    def on_mouse(self, mouse):
        libtcod.console_clear(sample_console)
        if mouse.lbutton_pressed:
            self.lbut = not self.lbut
        if mouse.rbutton_pressed:
            self.rbut = not self.rbut
        if mouse.mbutton_pressed:
            self.mbut = not self.mbut
        wheel=""
        if mouse.wheel_up :
            wheel="UP"
        elif mouse.wheel_down :
            wheel="DOWN"
        sample_console.print_(
            1, 1,
           "Mouse position : %4dx%4d\n"
           "Mouse cell     : %4dx%4d\n"
           "Mouse movement : %4dx%4d\n"
           "Left button    : %s (toggle %s)\n"
           "Right button   : %s (toggle %s)\n"
           "Middle button  : %s (toggle %s)\n"
           "Wheel          : %s" %
               (mouse.x, mouse.y,
                mouse.cx, mouse.cy,
                mouse.dx, mouse.dy,
                butstatus[mouse.lbutton], butstatus[self.lbut],
                butstatus[mouse.rbutton], butstatus[self.rbut],
                butstatus[mouse.mbutton], butstatus[self.mbut],
                wheel,
                )
           )
        sample_console.print_(1, 10, "1 : Hide cursor\n2 : Show cursor")

    def on_key(self, key):
        if key.c == ord('1'):
            libtcod.mouse_show_cursor(False)
        elif key.c == ord('2'):
            libtcod.mouse_show_cursor(True)

#############################################
# name generator sample
#############################################

class NameGeneratorSample(Sample):
    def __init__(self):
        self.name = 'Name generator'

        self.curset = 0
        self.nbsets = 0
        self.delay = 0.0
        self.names = []
        self.sets = None

    def on_enter(self):
        libtcod.sys_set_fps(60)

    def on_draw(self, delta_time):
        if self.nbsets == 0:
            # parse all *.cfg files in data/namegen
            for file in os.listdir(b'data/namegen') :
                if file.find(b'.cfg') > 0 :
                    libtcod.namegen_parse(os.path.join(b'data',b'namegen',file))
            # get the sets list
            self.sets=libtcod.namegen_get_sets()
            print (self.sets)
            self.nbsets=len(self.sets)
        while len(self.names)> 15:
            self.names.pop(0)
        libtcod.console_clear(sample_console)
        libtcod.console_set_default_foreground(sample_console,libtcod.white)
        libtcod.console_print(sample_console,1,1,"%s\n\n+ : next generator\n- : prev generator" %
            self.sets[self.curset])
        for i in range(len(self.names)) :
            libtcod.console_print_ex(sample_console,SAMPLE_SCREEN_WIDTH-2,2+i,
            libtcod.BKGND_NONE,libtcod.RIGHT,self.names[i])
        self.delay += libtcod.sys_get_last_frame_length()
        if self.delay > 0.5 :
            self.delay -= 0.5
            self.names.append(libtcod.namegen_generate(self.sets[self.curset]))

    def on_key(self, key):
        if key.c == ord('='):
            self.curset += 1
            if self.curset == self.nbsets :
                self.curset=0
            self.names.append("======")
        elif key.c == ord('-'):
            self.curset -= 1
            if self.curset < 0 :
                self.curset=self.nbsets-1
            self.names.append("======")

#############################################
# python fast render sample
#############################################
numpy_available = True

use_numpy = numpy_available  #default option
SCREEN_W = SAMPLE_SCREEN_WIDTH
SCREEN_H = SAMPLE_SCREEN_HEIGHT
HALF_W = SCREEN_W // 2
HALF_H = SCREEN_H // 2
RES_U = 80  #texture resolution
RES_V = 80
TEX_STRETCH = 5  #texture stretching with tunnel depth
SPEED = 15
LIGHT_BRIGHTNESS = 3.5  #brightness multiplier for all lights (changes their radius)
LIGHTS_CHANCE = 0.07  #chance of a light appearing
MAX_LIGHTS = 6
MIN_LIGHT_STRENGTH = 0.2
LIGHT_UPDATE = 0.05  #how much the ambient light changes to reflect current light sources
AMBIENT_LIGHT = 0.8  #brightness of tunnel texture

#the coordinates of all tiles in the screen, as numpy arrays. example: (4x3 pixels screen)
#xc = [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]
#yc = [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]]
if numpy_available:
    (xc, yc) = np.meshgrid(range(SCREEN_W), range(SCREEN_H))
    #translate coordinates of all pixels to center
    xc = xc - HALF_W
    yc = yc - HALF_H

noise2d = libtcod.noise_new(2, 0.5, 2.0)
if numpy_available:  #the texture starts empty
    texture = np.zeros((RES_U, RES_V))

class Light:
    def __init__(self, x, y, z, r, g, b, strength):
        self.x, self.y, self.z = x, y, z  #pos.
        self.r, self.g, self.b = r, g, b  #color
        self.strength = strength  #between 0 and 1, defines brightness

class FastRenderSample(Sample):
    def __init__(self):
        self.name = 'Python fast render'

    def on_enter(self):
        global frac_t, abs_t, lights, tex_r, tex_g, tex_b
        libtcod.sys_set_fps(0)
        libtcod.console_clear(sample_console)  #render status message
        libtcod.console_set_default_foreground(sample_console,libtcod.white)
        libtcod.console_print(sample_console, 1, SCREEN_H - 3, "Renderer: NumPy")

        frac_t = RES_V - 1  #time is represented in number of pixels of the texture, start later in time to initialize texture
        abs_t = RES_V - 1
        lights = []  #lights list, and current color of the tunnel texture
        tex_r, tex_g, tex_b = 0, 0, 0

    def on_draw(self, delta_time):
        global use_numpy, frac_t, abs_t, lights, tex_r, tex_g, tex_b, xc, yc, texture, texture2, brightness2, R2, G2, B2

        time_delta = libtcod.sys_get_last_frame_length() * SPEED  #advance time
        frac_t += time_delta  #increase fractional (always < 1.0) time
        abs_t += time_delta  #increase absolute elapsed time
        int_t = int(frac_t)  #integer time units that passed this frame (number of texture pixels to advance)
        frac_t -= int_t  #keep this < 1.0

        #change texture color according to presence of lights (basically, sum them
        #to get ambient light and smoothly change the current color into that)
        ambient_r = AMBIENT_LIGHT * sum([light.r * light.strength for light in lights])
        ambient_g = AMBIENT_LIGHT * sum([light.g * light.strength for light in lights])
        ambient_b = AMBIENT_LIGHT * sum([light.b * light.strength for light in lights])
        alpha = LIGHT_UPDATE * time_delta
        tex_r = tex_r * (1 - alpha) + ambient_r * alpha
        tex_g = tex_g * (1 - alpha) + ambient_g * alpha
        tex_b = tex_b * (1 - alpha) + ambient_b * alpha

        if int_t >= 1:  #roll texture (ie, advance in tunnel) according to int_t
            int_t = int_t % RES_V  #can't roll more than the texture's size (can happen when time_delta is large)
            int_abs_t = int(abs_t)  #new pixels are based on absolute elapsed time

            texture = np.roll(texture, -int_t, 1)
            #replace new stretch of texture with new values
            for v in range(RES_V - int_t, RES_V):
                for u in range(0, RES_U):
                    tex_v = (v + int_abs_t) / float(RES_V)
                    texture[u,v] = (libtcod.noise_get_fbm(noise2d, [u/float(RES_U), tex_v], 32.0) +
                                    libtcod.noise_get_fbm(noise2d, [1 - u/float(RES_U), tex_v], 32.0))

        #squared distance from center, clipped to sensible minimum and maximum values
        sqr_dist = xc**2 + yc**2
        sqr_dist = sqr_dist.clip(1.0 / RES_V, RES_V**2)

        #one coordinate into the texture, represents depth in the tunnel
        v = TEX_STRETCH * float(RES_V) / sqr_dist + frac_t
        v = v.clip(0, RES_V - 1)

        #another coordinate, represents rotation around the tunnel
        u = np.mod(RES_U * (np.arctan2(yc, xc) / (2 * np.pi) + 0.5), RES_U)

        #retrieve corresponding pixels from texture
        brightness = texture[u.astype(int), v.astype(int)] / 4.0 + 0.5

        #use the brightness map to compose the final color of the tunnel
        R = brightness * tex_r
        G = brightness * tex_g
        B = brightness * tex_b

        #create new light source
        if libtcod.random_get_float(0, 0, 1) <= time_delta * LIGHTS_CHANCE and len(lights) < MAX_LIGHTS:
            x = libtcod.random_get_float(0, -0.5, 0.5)
            y = libtcod.random_get_float(0, -0.5, 0.5)
            strength = libtcod.random_get_float(0, MIN_LIGHT_STRENGTH, 1.0)

            color = libtcod.Color(0, 0, 0)  #create bright colors with random hue
            hue = libtcod.random_get_float(0, 0, 360)
            libtcod.color_set_hsv(color, hue, 0.5, strength)
            lights.append(Light(x, y, TEX_STRETCH, color.r, color.g, color.b, strength))

        #eliminate lights that are going to be out of view
        lights = [light for light in lights if light.z - time_delta > 1.0 / RES_V]

        for light in lights:  #render lights
            #move light's Z coordinate with time, then project its XYZ coordinates to screen-space
            light.z -= float(time_delta) / TEX_STRETCH
            xl = light.x / light.z * SCREEN_H
            yl = light.y / light.z * SCREEN_H

            #calculate brightness of light according to distance from viewer and strength,
            #then calculate brightness of each pixel with inverse square distance law
            light_brightness = LIGHT_BRIGHTNESS * light.strength * (1.0 - light.z / TEX_STRETCH)
            brightness = light_brightness / ((xc - xl)**2 + (yc - yl)**2)

            #make all pixels shine around this light
            R += brightness * light.r
            G += brightness * light.g
            B += brightness * light.b

        #truncate values
        R = R.clip(0, 255)
        G = G.clip(0, 255)
        B = B.clip(0, 255)

        #fill the screen with these background colors
        sample_console.bg.transpose()[:] = [R.T, G.T, B.T]

#############################################
# main loop
#############################################

RENDERER_KEYS = {
    libtcod.KEY_F1: libtcod.RENDERER_GLSL,
    libtcod.KEY_F2: libtcod.RENDERER_OPENGL,
    libtcod.KEY_F3: libtcod.RENDERER_SDL,
    }

RENDERER_NAMES = ('F1 GLSL   ','F2 OPENGL ','F3 SDL    ')

SAMPLES = (
    TrueColorSample(),
    OffscreenConsoleSample(),
    LineDrawingSample(),
    NoiseSample(),
    FOVSample(),
    PathfindingSample(),
    BSPSample(),
    ImageSample(),
    MouseSample(),
    NameGeneratorSample(),
    FastRenderSample()
    )

cur_sample = 0

def main():
    global cur_sample
    credits_end = False
    SAMPLES[cur_sample].on_enter()
    draw_samples_menu()
    draw_renderer_menu()

    while not libtcod.console_is_window_closed():
        root_console.default_fg = (255, 255, 255)
        root_console.default_bg = (0, 0, 0)
        # render credits
        if not credits_end:
            root_console.clear()
            draw_samples_menu()
            draw_renderer_menu()
            credits_end = libtcod.console_credits_render(60, 43, 0)

        # render the sample
        SAMPLES[cur_sample].on_draw(libtcod.sys_get_last_frame_length())
        sample_console.blit(0, 0, sample_console.width, sample_console.height,
                            root_console, SAMPLE_SCREEN_X, SAMPLE_SCREEN_Y)
        draw_stats()
        handle_events()
        libtcod.console_flush()

def handle_events():
    global cur_sample
    key = libtcod.Key()
    mouse = libtcod.Mouse()
    EVENT_MASK = libtcod.EVENT_MOUSE | libtcod.EVENT_KEY_PRESS
    while(libtcod.sys_check_for_event(EVENT_MASK, key, mouse)):
        SAMPLES[cur_sample].on_mouse(mouse)
        SAMPLES[cur_sample].on_key(key)
        # key handler
        if key.vk == libtcod.KEY_DOWN:
            cur_sample = (cur_sample + 1) % len(SAMPLES)
            SAMPLES[cur_sample].on_enter()
            draw_samples_menu()
        elif key.vk == libtcod.KEY_UP:
            cur_sample = (cur_sample - 1) % len(SAMPLES)
            SAMPLES[cur_sample].on_enter()
            draw_samples_menu()
        elif key.vk == libtcod.KEY_ENTER and key.lalt:
            libtcod.console_set_fullscreen(not libtcod.console_is_fullscreen())
        elif key.vk == libtcod.KEY_PRINTSCREEN or key.c == 'p':
            print("screenshot")
            if key.lalt :
                libtcod.console_save_apf(None, "samples.apf")
                print("apf")
            else :
                libtcod.sys_save_screenshot()
                print("png")
        elif key.vk == libtcod.KEY_ESCAPE:
            raise SystemExit()
        elif key.vk in RENDERER_KEYS:
            libtcod.sys_set_renderer(RENDERER_KEYS[key.vk])
            draw_renderer_menu()

def draw_samples_menu():
    for i, sample in enumerate(SAMPLES):
        if i == cur_sample:
            root_console.default_fg = libtcod.white
            root_console.default_bg = libtcod.light_blue
        else:
            root_console.default_fg = libtcod.grey
            root_console.default_bg = libtcod.black
        root_console.print_(2, 46 - (len(SAMPLES) - i),
                            '  %s' % sample.name.ljust(19),
                            libtcod.BKGND_SET, libtcod.LEFT)

def draw_stats():
    root_console.default_fg = libtcod.grey
    root_console.print_(
        79, 46,
        ' last frame : %3d ms (%3d fps)' % (
            libtcod.sys_get_last_frame_length() * 1000.0,
            libtcod.sys_get_fps(),
            ),
        libtcod.BKGND_NONE, libtcod.RIGHT
        )
    root_console.print_(
        79, 47,
        'elapsed : %8d ms %4.2fs' % (libtcod.sys_elapsed_milli(),
                                     libtcod.sys_elapsed_seconds()),
        libtcod.BKGND_NONE, libtcod.RIGHT,
        )

def draw_renderer_menu():
    current_renderer = libtcod.sys_get_renderer()
    root_console.default_fg = libtcod.grey
    root_console.default_bg = libtcod.black
    root_console.print_(42, 46 - (libtcod.NB_RENDERERS + 1),
                        "Renderer :", libtcod.BKGND_SET, libtcod.LEFT)
    for i, name in enumerate(RENDERER_NAMES):
        if i == current_renderer:
            root_console.default_fg = libtcod.white
            root_console.default_bg = libtcod.light_blue
        else:
            root_console.default_fg = libtcod.grey
            root_console.default_bg = libtcod.black
        root_console.print_(
            42, 46 - (libtcod.NB_RENDERERS - i),
            name, libtcod.BKGND_SET, libtcod.LEFT
            )

if __name__ == '__main__':
    main()
