#!/usr/bin/env python3
"""
This code demonstrates various usages of python-tcod.
"""
# To the extent possible under law, the libtcod maintainers have waived all
# copyright and related or neighboring rights to these samples.
# https://creativecommons.org/publicdomain/zero/1.0/

import os

import copy
import math
import random
import time

import numpy as np
import tcod
import tcod.event

SAMPLE_SCREEN_WIDTH = 46
SAMPLE_SCREEN_HEIGHT = 20
SAMPLE_SCREEN_X = 20
SAMPLE_SCREEN_Y = 10
FONT = "data/fonts/consolas10x10_gs_tc.png"

root_console = None
sample_console = tcod.console.Console(
    SAMPLE_SCREEN_WIDTH, SAMPLE_SCREEN_HEIGHT, order="F"
)


class Sample(tcod.event.EventDispatch):
    def __init__(self, name: str = ""):
        self.name = name

    def on_enter(self):
        pass

    def on_draw(self):
        pass

    def ev_keydown(self, event: tcod.event.KeyDown):
        global cur_sample
        if event.sym == tcod.event.K_DOWN:
            cur_sample = (cur_sample + 1) % len(SAMPLES)
            SAMPLES[cur_sample].on_enter()
            draw_samples_menu()
        elif event.sym == tcod.event.K_UP:
            cur_sample = (cur_sample - 1) % len(SAMPLES)
            SAMPLES[cur_sample].on_enter()
            draw_samples_menu()
        elif (
            event.sym == tcod.event.K_RETURN
            and event.mod & tcod.event.KMOD_LALT
        ):
            tcod.console_set_fullscreen(not tcod.console_is_fullscreen())
        elif event.sym == tcod.event.K_PRINTSCREEN or event.sym == ord("p"):
            print("screenshot")
            if event.mod & tcod.event.KMOD_LALT:
                tcod.console_save_apf(None, "samples.apf")
                print("apf")
            else:
                tcod.sys_save_screenshot()
                print("png")
        elif event.sym == tcod.event.K_ESCAPE:
            raise SystemExit()
        elif event.sym in RENDERER_KEYS:
            tcod.sys_set_renderer(RENDERER_KEYS[event.sym])
            draw_renderer_menu()

    def ev_quit(self, event: tcod.event.Quit):
        raise SystemExit()


class TrueColorSample(Sample):
    def __init__(self):
        self.name = "True colors"
        # corner colors
        self.colors = np.array(
            [(50, 40, 150), (240, 85, 5), (50, 35, 240), (10, 200, 130)],
            dtype=np.int16,
        )
        # color shift direction
        self.slide_dir = np.array(
            [[1, 1, 1], [-1, -1, 1], [1, -1, 1], [1, 1, -1]], dtype=np.int16
        )
        # corner indexes
        self.corners = np.array([0, 1, 2, 3])
        # sample screen interpolation mesh-grid
        self.interp_x, self.interp_y = np.mgrid[
            0 : 1 : SAMPLE_SCREEN_WIDTH * 1j, 0 : 1 : SAMPLE_SCREEN_HEIGHT * 1j
        ]

    def on_enter(self):
        tcod.sys_set_fps(0)

    def on_draw(self):
        self.slide_corner_colors()
        self.interpolate_corner_colors()
        self.darken_background_characters()
        self.randomize_sample_conole()
        self.print_banner()

    def slide_corner_colors(self):
        # pick random RGB channels for each corner
        rand_channels = np.random.randint(low=0, high=3, size=4)

        # shift picked color channels in the direction of slide_dir
        self.colors[self.corners, rand_channels] += (
            self.slide_dir[self.corners, rand_channels] * 5
        )

        # reverse slide_dir values when limits are reached
        self.slide_dir[self.colors[:] == 255] = -1
        self.slide_dir[self.colors[:] == 0] = 1

    def interpolate_corner_colors(self):
        # interpolate corner colors across the sample console
        for i in range(3):  # for each color channel
            left = (
                self.colors[2, i] - self.colors[0, i]
            ) * self.interp_y + self.colors[0, i]
            right = (
                self.colors[3, i] - self.colors[1, i]
            ) * self.interp_y + self.colors[1, i]
            sample_console.bg[:, :, i] = (right - left) * self.interp_x + left

    def darken_background_characters(self):
        # darken background characters
        sample_console.fg[:] = sample_console.bg[:]
        sample_console.fg[:] //= 2

    def randomize_sample_conole(self):
        # randomize sample console characters
        sample_console.ch[:] = np.random.randint(
            low=ord("a"),
            high=ord("z") + 1,
            size=sample_console.ch.size,
            dtype=np.intc,
        ).reshape(sample_console.ch.shape)

    def print_banner(self):
        # print text on top of samples
        sample_console.print_box(
            x=1,
            y=5,
            width=sample_console.width - 2,
            height=sample_console.height - 1,
            string="The Doryen library uses 24 bits colors, for both "
            "background and foreground.",
            fg=tcod.white,
            bg=tcod.grey,
            bg_blend=tcod.BKGND_MULTIPLY,
            alignment=tcod.CENTER,
        )


class OffscreenConsoleSample(Sample):
    def __init__(self):
        self.name = "Offscreen console"
        self.secondary = tcod.console.Console(
            sample_console.width // 2, sample_console.height // 2
        )
        self.screenshot = tcod.console.Console(
            sample_console.width, sample_console.height
        )
        self.counter = 0
        self.x = 0
        self.y = 0
        self.xdir = 1
        self.ydir = 1

        self.secondary.draw_frame(
            0,
            0,
            sample_console.width // 2,
            sample_console.height // 2,
            "Offscreen console",
            False,
            fg=tcod.white,
            bg=tcod.black,
        )

        self.secondary.print_box(
            1,
            2,
            sample_console.width // 2 - 2,
            sample_console.height // 2,
            "You can render to an offscreen console and blit in on another "
            "one, simulating alpha transparency.",
            fg=tcod.white,
            bg=None,
            alignment=tcod.CENTER,
        )

    def on_enter(self):
        self.counter = time.perf_counter()
        tcod.sys_set_fps(0)
        # get a "screenshot" of the current sample screen
        sample_console.blit(dest=self.screenshot)

    def on_draw(self):
        if time.perf_counter() - self.counter >= 1:
            self.counter = time.perf_counter()
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
        self.screenshot.blit(sample_console)
        self.secondary.blit(
            sample_console,
            self.x,
            self.y,
            0,
            0,
            sample_console.width // 2,
            sample_console.height // 2,
            1.0,
            0.75,
        )


class LineDrawingSample(Sample):

    FLAG_NAMES = [
        "BKGND_NONE",
        "BKGND_SET",
        "BKGND_MULTIPLY",
        "BKGND_LIGHTEN",
        "BKGND_DARKEN",
        "BKGND_SCREEN",
        "BKGND_COLOR_DODGE",
        "BKGND_COLOR_BURN",
        "BKGND_ADD",
        "BKGND_ADDALPHA",
        "BKGND_BURN",
        "BKGND_OVERLAY",
        "BKGND_ALPHA",
    ]

    def __init__(self):
        self.name = "Line drawing"
        self.mk_flag = tcod.BKGND_SET
        self.bk_flag = tcod.BKGND_SET

        self.bk = tcod.console.Console(
            sample_console.width, sample_console.height, order="F"
        )
        # initialize the colored background
        self.bk.bg[:, :, 0] = np.linspace(0, 255, self.bk.width)[:, np.newaxis]
        self.bk.bg[:, :, 2] = np.linspace(0, 255, self.bk.height)
        self.bk.bg[:, :, 1] = (
            self.bk.bg[:, :, 0].astype(int) + self.bk.bg[:, :, 2]
        ) / 2
        self.bk.ch[:] = ord(" ")

    def ev_keydown(self, event: tcod.event.KeyDown):
        if event.sym in (tcod.event.K_RETURN, tcod.event.K_KP_ENTER):
            self.bk_flag += 1
            if (self.bk_flag & 0xFF) > tcod.BKGND_ALPH:
                self.bk_flag = tcod.BKGND_NONE
        else:
            super().ev_keydown(event)

    def on_enter(self):
        tcod.sys_set_fps(0)

    def on_draw(self):
        alpha = 0.0
        if (self.bk_flag & 0xFF) == tcod.BKGND_ALPH:
            # for the alpha mode, update alpha every frame
            alpha = (1.0 + math.cos(time.time() * 2)) / 2.0
            self.bk_flag = tcod.BKGND_ALPHA(alpha)
        elif (self.bk_flag & 0xFF) == tcod.BKGND_ADDA:
            # for the add alpha mode, update alpha every frame
            alpha = (1.0 + math.cos(time.time() * 2)) / 2.0
            self.bk_flag = tcod.BKGND_ADDALPHA(alpha)

        self.bk.blit(sample_console)
        recty = int(
            (sample_console.height - 2) * ((1.0 + math.cos(time.time())) / 2.0)
        )
        for x in range(sample_console.width):
            col = [x * 255 // sample_console.width] * 3
            tcod.console_set_char_background(
                sample_console, x, recty, col, self.bk_flag
            )
            tcod.console_set_char_background(
                sample_console, x, recty + 1, col, self.bk_flag
            )
            tcod.console_set_char_background(
                sample_console, x, recty + 2, col, self.bk_flag
            )
        angle = time.time() * 2.0
        cos_angle = math.cos(angle)
        sin_angle = math.sin(angle)
        xo = int(sample_console.width // 2 * (1 + cos_angle))
        yo = int(
            sample_console.height // 2 + sin_angle * sample_console.width // 2
        )
        xd = int(sample_console.width // 2 * (1 - cos_angle))
        yd = int(
            sample_console.height // 2 - sin_angle * sample_console.width // 2
        )
        # draw the line
        # in python the easiest way is to use the line iterator
        for x, y in tcod.line_iter(xo, yo, xd, yd):
            if (
                0 <= x < sample_console.width
                and 0 <= y < sample_console.height
            ):
                tcod.console_set_char_background(
                    sample_console, x, y, tcod.light_blue, self.bk_flag
                )
        sample_console.print(
            2,
            2,
            "%s (ENTER to change)" % self.FLAG_NAMES[self.bk_flag & 0xFF],
            fg=tcod.white,
            bg=None,
        )


class NoiseSample(Sample):
    NOISE_OPTIONS = [  # [name, algorithm, implementation],
        ["perlin noise", tcod.NOISE_PERLIN, tcod.noise.SIMPLE],
        ["simplex noise", tcod.NOISE_SIMPLEX, tcod.noise.SIMPLE],
        ["wavelet noise", tcod.NOISE_WAVELET, tcod.noise.SIMPLE],
        ["perlin fbm", tcod.NOISE_PERLIN, tcod.noise.FBM],
        ["perlin turbulence", tcod.NOISE_PERLIN, tcod.noise.TURBULENCE],
        ["simplex fbm", tcod.NOISE_SIMPLEX, tcod.noise.FBM],
        ["simplex turbulence", tcod.NOISE_SIMPLEX, tcod.noise.TURBULENCE],
        ["wavelet fbm", tcod.NOISE_WAVELET, tcod.noise.FBM],
        ["wavelet turbulence", tcod.NOISE_WAVELET, tcod.noise.TURBULENCE],
    ]

    def __init__(self):
        self.name = "Noise"
        self.func = 0
        self.dx = 0.0
        self.dy = 0.0
        self.octaves = 4.0
        self.zoom = 3.0
        self.hurst = tcod.NOISE_DEFAULT_HURST
        self.lacunarity = tcod.NOISE_DEFAULT_LACUNARITY
        self.noise = self.get_noise()
        self.img = tcod.image_new(
            SAMPLE_SCREEN_WIDTH * 2, SAMPLE_SCREEN_HEIGHT * 2
        )

    @property
    def algorithm(self):
        return self.NOISE_OPTIONS[self.func][1]

    @property
    def implementation(self):
        return self.NOISE_OPTIONS[self.func][2]

    def get_noise(self):
        return tcod.noise.Noise(
            2,
            self.algorithm,
            self.implementation,
            self.hurst,
            self.lacunarity,
            self.octaves,
            seed=None,
        )

    def on_enter(self):
        tcod.sys_set_fps(0)

    def on_draw(self):
        self.dx = time.perf_counter() * 0.25
        self.dy = time.perf_counter() * 0.25
        for y in range(2 * sample_console.height):
            for x in range(2 * sample_console.width):
                f = [
                    self.zoom * x / (2 * sample_console.width) + self.dx,
                    self.zoom * y / (2 * sample_console.height) + self.dy,
                ]
                value = self.noise.get_point(*f)
                c = int((value + 1.0) / 2.0 * 255)
                c = max(0, min(c, 255))
                self.img.put_pixel(x, y, (c // 2, c // 2, c))
        rectw = 24
        recth = 13
        if self.implementation == tcod.noise.SIMPLE:
            recth = 10
        self.img.blit_2x(sample_console, 0, 0)
        sample_console.draw_rect(
            2,
            2,
            rectw,
            recth,
            ch=0,
            fg=None,
            bg=tcod.grey,
            bg_blend=tcod.BKGND_MULTIPLY,
        )
        sample_console.fg[2 : 2 + rectw, 2 : 2 + recth] = (
            sample_console.fg[2 : 2 + rectw, 2 : 2 + recth] * tcod.grey / 255
        )

        for curfunc in range(len(self.NOISE_OPTIONS)):
            text = "%i : %s" % (curfunc + 1, self.NOISE_OPTIONS[curfunc][0])
            if curfunc == self.func:
                sample_console.print(
                    2, 2 + curfunc, text, fg=tcod.white, bg=tcod.light_blue
                )
            else:
                sample_console.print(
                    2, 2 + curfunc, text, fg=tcod.grey, bg=None
                )
        sample_console.print(
            2, 11, "Y/H : zoom (%2.1f)" % self.zoom, fg=tcod.white, bg=None
        )
        if self.implementation != tcod.noise.SIMPLE:
            sample_console.print(
                2,
                12,
                "E/D : hurst (%2.1f)" % self.hurst,
                fg=tcod.white,
                bg=None,
            )
            sample_console.print(
                2,
                13,
                "R/F : lacunarity (%2.1f)" % self.lacunarity,
                fg=tcod.white,
                bg=None,
            )
            sample_console.print(
                2,
                14,
                "T/G : octaves (%2.1f)" % self.octaves,
                fg=tcod.white,
                bg=None,
            )

    def ev_keydown(self, event: tcod.event.KeyDown):
        if ord("9") >= event.sym >= ord("1"):
            self.func = event.sym - ord("1")
            self.noise = self.get_noise()
        elif event.sym == ord("e"):
            self.hurst += 0.1
            self.noise = self.get_noise()
        elif event.sym == ord("d"):
            self.hurst -= 0.1
            self.noise = self.get_noise()
        elif event.sym == ord("r"):
            self.lacunarity += 0.5
            self.noise = self.get_noise()
        elif event.sym == ord("f"):
            self.lacunarity -= 0.5
            self.noise = self.get_noise()
        elif event.sym == ord("t"):
            self.octaves += 0.5
            self.noise.octaves = self.octaves
        elif event.sym == ord("g"):
            self.octaves -= 0.5
            self.noise.octaves = self.octaves
        elif event.sym == ord("y"):
            self.zoom += 0.2
        elif event.sym == ord("h"):
            self.zoom -= 0.2
        else:
            super().ev_keydown(event)


#############################################
# field of view sample
#############################################
DARK_WALL = (0, 0, 100)
LIGHT_WALL = (130, 110, 50)
DARK_GROUND = (50, 50, 150)
LIGHT_GROUND = (200, 180, 50)

SAMPLE_MAP = [
    "##############################################",
    "#######################      #################",
    "#####################    #     ###############",
    "######################  ###        ###########",
    "##################      #####             ####",
    "################       ########    ###### ####",
    "###############      #################### ####",
    "################    ######                  ##",
    "########   #######  ######   #     #     #  ##",
    "########   ######      ###                  ##",
    "########                                    ##",
    "####       ######      ###   #     #     #  ##",
    "#### ###   ########## ####                  ##",
    "#### ###   ##########   ###########=##########",
    "#### ##################   #####          #####",
    "#### ###             #### #####          #####",
    "####           #     ####                #####",
    "########       #     #### #####          #####",
    "########       #####      ####################",
    "##############################################",
]

SAMPLE_MAP = np.array([list(line) for line in SAMPLE_MAP]).transpose()

FOV_ALGO_NAMES = [
    "BASIC      ",
    "DIAMOND    ",
    "SHADOW     ",
    "PERMISSIVE0",
    "PERMISSIVE1",
    "PERMISSIVE2",
    "PERMISSIVE3",
    "PERMISSIVE4",
    "PERMISSIVE5",
    "PERMISSIVE6",
    "PERMISSIVE7",
    "PERMISSIVE8",
    "RESTRICTIVE",
]

TORCH_RADIUS = 10
SQUARED_TORCH_RADIUS = TORCH_RADIUS * TORCH_RADIUS


class FOVSample(Sample):
    def __init__(self):
        self.name = "Field of view"

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
        self.noise = tcod.noise_new(1, 1.0, 1.0)

        self.map = tcod.map.Map(
            SAMPLE_SCREEN_WIDTH, SAMPLE_SCREEN_HEIGHT, order="F"
        )
        self.map.walkable[:] = SAMPLE_MAP[:] == " "
        self.map.transparent[:] = self.map.walkable[:] | (SAMPLE_MAP == "=")

        self.light_map_bg = np.full(
            SAMPLE_MAP.shape + (3,), LIGHT_GROUND, dtype=np.uint8
        )
        self.light_map_bg[SAMPLE_MAP[:] == "#"] = LIGHT_WALL
        self.dark_map_bg = np.full(
            SAMPLE_MAP.shape + (3,), DARK_GROUND, dtype=np.uint8
        )
        self.dark_map_bg[SAMPLE_MAP[:] == "#"] = DARK_WALL

    def draw_ui(self):
        sample_console.print(
            1,
            1,
            "IJKL : move around\n"
            "T : torch fx %s\n"
            "W : light walls %s\n"
            "+-: algo %s"
            % (
                "on " if self.torch else "off",
                "on " if self.light_walls else "off",
                FOV_ALGO_NAMES[self.algo_num],
            ),
            fg=tcod.white,
            bg=None,
        )

    def on_enter(self):
        tcod.sys_set_fps(60)
        # we draw the foreground only the first time.
        #  during the player movement, only the @ is redrawn.
        #  the rest impacts only the background color
        # draw the help text & player @
        sample_console.clear()
        self.draw_ui()
        tcod.console_put_char(
            sample_console, self.px, self.py, "@", tcod.BKGND_NONE
        )
        # draw windows
        sample_console.ch[np.where(SAMPLE_MAP == "=")] = tcod.CHAR_DHLINE
        sample_console.fg[np.where(SAMPLE_MAP == "=")] = tcod.black

    def on_draw(self):
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
                self.algo_num,
            )
        sample_console.bg[:] = self.dark_map_bg[:]
        if self.torch:
            # slightly change the perlin noise parameter
            self.torchx += 0.1
            # randomize the light position between -1.5 and 1.5
            tdx = [self.torchx + 20.0]
            dx = tcod.noise_get(self.noise, tdx, tcod.NOISE_SIMPLEX) * 1.5
            tdx[0] += 30.0
            dy = tcod.noise_get(self.noise, tdx, tcod.NOISE_SIMPLEX) * 1.5
            di = 0.2 * tcod.noise_get(
                self.noise, [self.torchx], tcod.NOISE_SIMPLEX
            )
            # where_fov = np.where(self.map.fov[:])
            mgrid = np.mgrid[:SAMPLE_SCREEN_WIDTH, :SAMPLE_SCREEN_HEIGHT]
            # get squared distance
            light = (mgrid[0] - self.px + dx) ** 2 + (
                mgrid[1] - self.py + dy
            ) ** 2
            light = light.astype(np.float16)
            visible = (light < SQUARED_TORCH_RADIUS) & self.map.fov[:]
            light[...] = SQUARED_TORCH_RADIUS - light
            light[...] /= SQUARED_TORCH_RADIUS
            light[...] += di
            light[...] = light.clip(0, 1)
            light[~visible] = 0

            sample_console.bg[...] = (
                self.light_map_bg.astype(np.float16) - self.dark_map_bg
            ) * light[..., np.newaxis] + self.dark_map_bg
        else:
            where_fov = np.where(self.map.fov[:])
            sample_console.bg[where_fov] = self.light_map_bg[where_fov]

    def ev_keydown(self, event: tcod.event.KeyDown):
        MOVE_KEYS = {
            ord("i"): (0, -1),
            ord("j"): (-1, 0),
            ord("k"): (0, 1),
            ord("l"): (1, 0),
        }
        FOV_SELECT_KEYS = {ord("-"): -1, ord("="): 1}
        if event.sym in MOVE_KEYS:
            x, y = MOVE_KEYS[event.sym]
            if SAMPLE_MAP[self.px + x, self.py + y] == " ":
                tcod.console_put_char(
                    sample_console, self.px, self.py, " ", tcod.BKGND_NONE
                )
                self.px += x
                self.py += y
                tcod.console_put_char(
                    sample_console, self.px, self.py, "@", tcod.BKGND_NONE
                )
                self.recompute = True
        elif event.sym == ord("t"):
            self.torch = not self.torch
            self.draw_ui()
            self.recompute = True
        elif event.sym == ord("w"):
            self.light_walls = not self.light_walls
            self.draw_ui()
            self.recompute = True
        elif event.sym in FOV_SELECT_KEYS:
            self.algo_num += FOV_SELECT_KEYS[event.sym]
            self.algo_num %= tcod.NB_FOV_ALGORITHMS
            self.draw_ui()
            self.recompute = True
        else:
            super().ev_keydown(event)


class PathfindingSample(Sample):
    def __init__(self):
        self.name = "Path finding"

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
        self.oldchar = " "

        self.map = tcod.map_new(SAMPLE_SCREEN_WIDTH, SAMPLE_SCREEN_HEIGHT)
        for y in range(SAMPLE_SCREEN_HEIGHT):
            for x in range(SAMPLE_SCREEN_WIDTH):
                if SAMPLE_MAP[x, y] == " ":
                    # ground
                    tcod.map_set_properties(self.map, x, y, True, True)
                elif SAMPLE_MAP[x, y] == "=":
                    # window
                    tcod.map_set_properties(self.map, x, y, True, False)
        self.path = tcod.path_new_using_map(self.map)
        self.dijk = tcod.dijkstra_new(self.map)

    def on_enter(self):
        tcod.sys_set_fps(60)
        # we draw the foreground only the first time.
        #  during the player movement, only the @ is redrawn.
        #  the rest impacts only the background color
        # draw the help text & player @
        sample_console.clear()
        sample_console.ch[self.dx, self.dy] = ord("+")
        sample_console.fg[self.dx, self.dy] = tcod.white
        sample_console.ch[self.px, self.py] = ord("@")
        sample_console.fg[self.px, self.py] = tcod.white
        sample_console.print(
            1,
            1,
            "IJKL / mouse :\nmove destination\nTAB : A*/dijkstra",
            fg=tcod.white,
            bg=None,
        )
        sample_console.print(1, 4, "Using : A*", fg=tcod.white, bg=None)
        # draw windows
        for y in range(SAMPLE_SCREEN_HEIGHT):
            for x in range(SAMPLE_SCREEN_WIDTH):
                if SAMPLE_MAP[x, y] == "=":
                    tcod.console_put_char(
                        sample_console, x, y, tcod.CHAR_DHLINE, tcod.BKGND_NONE
                    )
        self.recalculate = True

    def on_draw(self):
        if self.recalculate:
            if self.using_astar:
                tcod.path_compute(
                    self.path, self.px, self.py, self.dx, self.dy
                )
            else:
                self.dijk_dist = 0.0
                # compute dijkstra grid (distance from px,py)
                tcod.dijkstra_compute(self.dijk, self.px, self.py)
                # get the maximum distance (needed for rendering)
                for y in range(SAMPLE_SCREEN_HEIGHT):
                    for x in range(SAMPLE_SCREEN_WIDTH):
                        d = tcod.dijkstra_get_distance(self.dijk, x, y)
                        if d > self.dijk_dist:
                            self.dijk_dist = d
                # compute path from px,py to dx,dy
                tcod.dijkstra_path_set(self.dijk, self.dx, self.dy)
            self.recalculate = False
            self.busy = 0.2
        # draw the dungeon
        for y in range(SAMPLE_SCREEN_HEIGHT):
            for x in range(SAMPLE_SCREEN_WIDTH):
                if SAMPLE_MAP[x, y] == "#":
                    tcod.console_set_char_background(
                        sample_console, x, y, DARK_WALL, tcod.BKGND_SET
                    )
                else:
                    tcod.console_set_char_background(
                        sample_console, x, y, DARK_GROUND, tcod.BKGND_SET
                    )
        # draw the path
        if self.using_astar:
            for i in range(tcod.path_size(self.path)):
                x, y = tcod.path_get(self.path, i)
                tcod.console_set_char_background(
                    sample_console, x, y, LIGHT_GROUND, tcod.BKGND_SET
                )
        else:
            for y in range(SAMPLE_SCREEN_HEIGHT):
                for x in range(SAMPLE_SCREEN_WIDTH):
                    if SAMPLE_MAP[x, y] != "#":
                        tcod.console_set_char_background(
                            sample_console,
                            x,
                            y,
                            tcod.color_lerp(
                                LIGHT_GROUND,
                                DARK_GROUND,
                                0.9
                                * tcod.dijkstra_get_distance(self.dijk, x, y)
                                / self.dijk_dist,
                            ),
                            tcod.BKGND_SET,
                        )
            for i in range(tcod.dijkstra_size(self.dijk)):
                x, y = tcod.dijkstra_get(self.dijk, i)
                tcod.console_set_char_background(
                    sample_console, x, y, LIGHT_GROUND, tcod.BKGND_SET
                )

        # move the creature
        self.busy -= tcod.sys_get_last_frame_length()
        if self.busy <= 0.0:
            self.busy = 0.2
            if self.using_astar:
                if not tcod.path_is_empty(self.path):
                    tcod.console_put_char(
                        sample_console, self.px, self.py, " ", tcod.BKGND_NONE
                    )
                    self.px, self.py = tcod.path_walk(self.path, True)
                    tcod.console_put_char(
                        sample_console, self.px, self.py, "@", tcod.BKGND_NONE
                    )
            else:
                if not tcod.dijkstra_is_empty(self.dijk):
                    tcod.console_put_char(
                        sample_console, self.px, self.py, " ", tcod.BKGND_NONE
                    )
                    self.px, self.py = tcod.dijkstra_path_walk(self.dijk)
                    tcod.console_put_char(
                        sample_console, self.px, self.py, "@", tcod.BKGND_NONE
                    )
                    self.recalculate = True

    def ev_keydown(self, event: tcod.event.KeyDown):
        if event.sym == ord("i") and self.dy > 0:
            # destination move north
            tcod.console_put_char(
                sample_console, self.dx, self.dy, self.oldchar, tcod.BKGND_NONE
            )
            self.dy -= 1
            self.oldchar = sample_console.ch[self.dx, self.dy]
            tcod.console_put_char(
                sample_console, self.dx, self.dy, "+", tcod.BKGND_NONE
            )
            if SAMPLE_MAP[self.dx, self.dy] == " ":
                self.recalculate = True
        elif event.sym == ord("k") and self.dy < SAMPLE_SCREEN_HEIGHT - 1:
            # destination move south
            tcod.console_put_char(
                sample_console, self.dx, self.dy, self.oldchar, tcod.BKGND_NONE
            )
            self.dy += 1
            self.oldchar = sample_console.ch[self.dx, self.dy]
            tcod.console_put_char(
                sample_console, self.dx, self.dy, "+", tcod.BKGND_NONE
            )
            if SAMPLE_MAP[self.dx, self.dy] == " ":
                self.recalculate = True
        elif event.sym == ord("j") and self.dx > 0:
            # destination move west
            tcod.console_put_char(
                sample_console, self.dx, self.dy, self.oldchar, tcod.BKGND_NONE
            )
            self.dx -= 1
            self.oldchar = sample_console.ch[self.dx, self.dy]
            tcod.console_put_char(
                sample_console, self.dx, self.dy, "+", tcod.BKGND_NONE
            )
            if SAMPLE_MAP[self.dx, self.dy] == " ":
                self.recalculate = True
        elif event.sym == ord("l") and self.dx < SAMPLE_SCREEN_WIDTH - 1:
            # destination move east
            tcod.console_put_char(
                sample_console, self.dx, self.dy, self.oldchar, tcod.BKGND_NONE
            )
            self.dx += 1
            self.oldchar = sample_console.ch[self.dx, self.dy]
            tcod.console_put_char(
                sample_console, self.dx, self.dy, "+", tcod.BKGND_NONE
            )
            if SAMPLE_MAP[self.dx, self.dy] == " ":
                self.recalculate = True
        elif event.sym == tcod.event.K_TAB:
            self.using_astar = not self.using_astar
            if self.using_astar:
                tcod.console_print(sample_console, 1, 4, "Using : A*      ")
            else:
                tcod.console_print(sample_console, 1, 4, "Using : Dijkstra")
            self.recalculate = True
        else:
            super().ev_keydown(event)

    def ev_mousemotion(self, event: tcod.event.MouseMotion):
        mx = event.tile.x - SAMPLE_SCREEN_X
        my = event.tile.y - SAMPLE_SCREEN_Y
        if (
            0 <= mx < SAMPLE_SCREEN_WIDTH
            and 0 <= my < SAMPLE_SCREEN_HEIGHT
            and (self.dx != mx or self.dy != my)
        ):
            tcod.console_put_char(
                sample_console, self.dx, self.dy, self.oldchar, tcod.BKGND_NONE
            )
            self.dx = mx
            self.dy = my
            self.oldchar = sample_console.ch[self.dx, self.dy]
            tcod.console_put_char(
                sample_console, self.dx, self.dy, "+", tcod.BKGND_NONE
            )
            if SAMPLE_MAP[self.dx, self.dy] == " ":
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


# draw a vertical line
def vline(m, x, y1, y2):
    if y1 > y2:
        y1, y2 = y2, y1
    for y in range(y1, y2 + 1):
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
        x1, x2 = x2, x1
    for x in range(x1, x2 + 1):
        m[x][y] = True


# draw a horizontal line left until we reach an empty space
def hline_left(m, x, y):
    while x >= 0 and not m[x][y]:
        m[x][y] = True
        x -= 1


# draw a horizontal line right until we reach an empty space
def hline_right(m, x, y):
    while x < SAMPLE_SCREEN_WIDTH and not m[x][y]:
        m[x][y] = True
        x += 1


# the class building the dungeon from the bsp nodes
def traverse_node(bsp_map, node):
    if not node.children:
        # calculate the room size
        if bsp_room_walls:
            node.width -= 1
            node.height -= 1
        if bsp_random_room:
            new_width = random.randint(
                min(node.width, bsp_min_room_size), node.width
            )
            new_height = random.randint(
                min(node.height, bsp_min_room_size), node.height
            )
            node.x += random.randint(0, node.width - new_width)
            node.y += random.randint(0, node.height - new_height)
            node.width, node.height = new_width, new_height
        # dig the room
        for x in range(node.x, node.x + node.width):
            for y in range(node.y, node.y + node.height):
                bsp_map[x][y] = True
    else:
        # resize the node to fit its sons
        left, right = node.children
        node.x = min(left.x, right.x)
        node.y = min(left.y, right.y)
        node.w = max(left.x + left.w, right.x + right.w) - node.x
        node.h = max(left.y + left.h, right.y + right.h) - node.y
        # create a corridor between the two lower nodes
        if node.horizontal:
            # vertical corridor
            if left.x + left.w - 1 < right.x or right.x + right.w - 1 < left.x:
                # no overlapping zone. we need a Z shaped corridor
                x1 = random.randint(left.x, left.x + left.w - 1)
                x2 = random.randint(right.x, right.x + right.w - 1)
                y = random.randint(left.y + left.h, right.y)
                vline_up(bsp_map, x1, y - 1)
                hline(bsp_map, x1, y, x2)
                vline_down(bsp_map, x2, y + 1)
            else:
                # straight vertical corridor
                minx = max(left.x, right.x)
                maxx = min(left.x + left.w - 1, right.x + right.w - 1)
                x = random.randint(minx, maxx)
                vline_down(bsp_map, x, right.y)
                vline_up(bsp_map, x, right.y - 1)
        else:
            # horizontal corridor
            if left.y + left.h - 1 < right.y or right.y + right.h - 1 < left.y:
                # no overlapping zone. we need a Z shaped corridor
                y1 = random.randint(left.y, left.y + left.h - 1)
                y2 = random.randint(right.y, right.y + right.h - 1)
                x = random.randint(left.x + left.w, right.x)
                hline_left(bsp_map, x - 1, y1)
                vline(bsp_map, x, y1, y2)
                hline_right(bsp_map, x + 1, y2)
            else:
                # straight horizontal corridor
                miny = max(left.y, right.y)
                maxy = min(left.y + left.h - 1, right.y + right.h - 1)
                y = random.randint(miny, maxy)
                hline_left(bsp_map, right.x - 1, y)
                hline_right(bsp_map, right.x, y)
    return True


class BSPSample(Sample):
    def __init__(self):
        self.name = "Bsp toolkit"
        self.bsp = tcod.bsp.BSP(
            1, 1, SAMPLE_SCREEN_WIDTH - 1, SAMPLE_SCREEN_HEIGHT - 1
        )
        self.bsp_map = np.zeros(
            (SAMPLE_SCREEN_WIDTH, SAMPLE_SCREEN_HEIGHT), dtype=bool, order="F"
        )
        self.bsp_generate()

    def bsp_generate(self):
        self.bsp.children = ()
        if bsp_room_walls:
            self.bsp.split_recursive(
                bsp_depth,
                bsp_min_room_size + 1,
                bsp_min_room_size + 1,
                1.5,
                1.5,
            )
        else:
            self.bsp.split_recursive(
                bsp_depth, bsp_min_room_size, bsp_min_room_size, 1.5, 1.5
            )
        self.bsp_refresh()

    def bsp_refresh(self):
        self.bsp_map[...] = False
        for node in copy.deepcopy(self.bsp).inverted_level_order():
            traverse_node(self.bsp_map, node)

    def on_draw(self):
        sample_console.clear()
        rooms = "OFF"
        if bsp_random_room:
            rooms = "ON"
        sample_console.print(
            1,
            1,
            "ENTER : rebuild bsp\n"
            "SPACE : rebuild dungeon\n"
            "+-: bsp depth %d\n"
            "*/: room size %d\n"
            "1 : random room size %s" % (bsp_depth, bsp_min_room_size, rooms),
            fg=tcod.white,
            bg=None,
        )
        if bsp_random_room:
            walls = "OFF"
            if bsp_room_walls:
                walls = "ON"
            sample_console.print(
                1, 6, "2 : room walls %s" % walls, fg=tcod.white, bg=None
            )
        # render the level
        for y in range(SAMPLE_SCREEN_HEIGHT):
            for x in range(SAMPLE_SCREEN_WIDTH):
                color = DARK_GROUND if self.bsp_map[x][y] else DARK_WALL
                tcod.console_set_char_background(
                    sample_console, x, y, color, tcod.BKGND_SET
                )

    def ev_keydown(self, event: tcod.event.KeyDown):
        global bsp_random_room, bsp_room_walls, bsp_depth, bsp_min_room_size
        if event.sym in (tcod.event.K_RETURN, tcod.event.K_KP_ENTER):
            self.bsp_generate()
        elif event.sym == ord(" "):
            self.bsp_refresh()
        elif event.sym in (tcod.event.K_EQUALS, tcod.event.K_KP_PLUS):
            bsp_depth += 1
            self.bsp_generate()
        elif event.sym in (tcod.event.K_MINUS, tcod.event.K_KP_MINUS):
            bsp_depth = max(1, bsp_depth - 1)
            self.bsp_generate()
        elif event.sym in (tcod.event.K_8, tcod.event.K_KP_MULTIPLY):
            bsp_min_room_size += 1
            self.bsp_generate()
        elif event.sym in (tcod.event.K_SLASH, tcod.event.K_KP_DIVIDE):
            bsp_min_room_size = max(2, bsp_min_room_size - 1)
            self.bsp_generate()
        elif event.sym in (tcod.event.K_1, tcod.event.K_KP_1):
            bsp_random_room = not bsp_random_room
            if not bsp_random_room:
                bsp_room_walls = True
            self.bsp_refresh()
        elif event.sym in (tcod.event.K_2, tcod.event.K_KP_2):
            bsp_room_walls = not bsp_room_walls
            self.bsp_refresh()
        else:
            super().ev_keydown(event)


class ImageSample(Sample):
    def __init__(self):
        self.name = "Image toolkit"

        self.img = tcod.image_load("data/img/skull.png")
        self.img.set_key_color(tcod.black)
        self.circle = tcod.image_load("data/img/circle.png")

    def on_enter(self):
        tcod.sys_set_fps(0)

    def on_draw(self):
        sample_console.clear()
        x = sample_console.width / 2 + math.cos(time.time()) * 10.0
        y = sample_console.height / 2
        scalex = 0.2 + 1.8 * (1.0 + math.cos(time.time() / 2)) / 2.0
        scaley = scalex
        angle = time.perf_counter()
        if int(time.time()) % 2:
            # split the color channels of circle.png
            # the red channel
            sample_console.draw_rect(0, 3, 15, 15, 0, None, (255, 0, 0))
            self.circle.blit_rect(
                sample_console, 0, 3, -1, -1, tcod.BKGND_MULTIPLY
            )
            # the green channel
            sample_console.draw_rect(15, 3, 15, 15, 0, None, (0, 255, 0))
            self.circle.blit_rect(
                sample_console, 15, 3, -1, -1, tcod.BKGND_MULTIPLY
            )
            # the blue channel
            sample_console.draw_rect(30, 3, 15, 15, 0, None, (0, 0, 255))
            self.circle.blit_rect(
                sample_console, 30, 3, -1, -1, tcod.BKGND_MULTIPLY
            )
        else:
            # render circle.png with normal blitting
            self.circle.blit_rect(sample_console, 0, 3, -1, -1, tcod.BKGND_SET)
            self.circle.blit_rect(
                sample_console, 15, 3, -1, -1, tcod.BKGND_SET
            )
            self.circle.blit_rect(
                sample_console, 30, 3, -1, -1, tcod.BKGND_SET
            )
        self.img.blit(
            sample_console, x, y, tcod.BKGND_SET, scalex, scaley, angle
        )


class MouseSample(Sample):
    def __init__(self):
        self.name = "Mouse support"

        self.motion = tcod.event.MouseMotion()
        self.lbut = self.mbut = self.rbut = 0
        self.log = []

    def on_enter(self):
        tcod.mouse_move(320, 200)
        tcod.mouse_show_cursor(True)
        tcod.sys_set_fps(60)

    def ev_mousemotion(self, event: tcod.event.MouseMotion):
        self.motion = event

    def ev_mousebuttondown(self, event: tcod.event.MouseButtonDown):
        if event.button == tcod.event.BUTTON_LEFT:
            self.lbut = True
        elif event.button == tcod.event.BUTTON_MIDDLE:
            self.mbut = True
        elif event.button == tcod.event.BUTTON_RIGHT:
            self.rbut = True

    def ev_mousebuttonup(self, event: tcod.event.MouseButtonUp):
        if event.button == tcod.event.BUTTON_LEFT:
            self.lbut = False
        elif event.button == tcod.event.BUTTON_MIDDLE:
            self.mbut = False
        elif event.button == tcod.event.BUTTON_RIGHT:
            self.rbut = False

    def on_draw(self):
        sample_console.clear(bg=tcod.grey)
        sample_console.print(
            1,
            1,
            "Mouse position : %4dx%4d\n"
            "Mouse cell     : %4dx%4d\n"
            "Mouse movement : %4dx%4d\n"
            "Left button    : %s\n"
            "Right button   : %s\n"
            "Middle button  : %s\n"
            % (
                self.motion.pixel.x,
                self.motion.pixel.y,
                self.motion.tile.x,
                self.motion.tile.y,
                self.motion.tile_motion.x,
                self.motion.tile_motion.y,
                ("OFF", "ON")[self.lbut],
                ("OFF", "ON")[self.rbut],
                ("OFF", "ON")[self.mbut],
            ),
            fg=tcod.light_yellow,
            bg=None,
        )
        sample_console.print(
            1,
            10,
            "1 : Hide cursor\n2 : Show cursor",
            fg=tcod.light_yellow,
            bg=None,
        )

    def ev_keydown(self, event: tcod.event.KeyDown):
        if event.sym == ord("1"):
            tcod.mouse_show_cursor(False)
        elif event.sym == ord("2"):
            tcod.mouse_show_cursor(True)
        else:
            super().ev_keydown(event)


class NameGeneratorSample(Sample):
    def __init__(self):
        self.name = "Name generator"

        self.curset = 0
        self.nbsets = 0
        self.delay = 0.0
        self.names = []
        self.sets = None

    def on_enter(self):
        tcod.sys_set_fps(60)

    def on_draw(self):
        if self.nbsets == 0:
            # parse all *.cfg files in data/namegen
            for file in os.listdir("data/namegen"):
                if file.find(".cfg") > 0:
                    tcod.namegen_parse(os.path.join("data/namegen", file))
            # get the sets list
            self.sets = tcod.namegen_get_sets()
            print(self.sets)
            self.nbsets = len(self.sets)
        while len(self.names) > 15:
            self.names.pop(0)
        sample_console.clear(bg=tcod.grey)
        sample_console.print(
            1,
            1,
            "%s\n\n+ : next generator\n- : prev generator"
            % self.sets[self.curset],
            fg=tcod.white,
            bg=None,
        )
        for i in range(len(self.names)):
            sample_console.print(
                SAMPLE_SCREEN_WIDTH - 2,
                2 + i,
                self.names[i],
                fg=tcod.white,
                bg=None,
                alignment=tcod.RIGHT,
            )
        self.delay += tcod.sys_get_last_frame_length()
        if self.delay > 0.5:
            self.delay -= 0.5
            self.names.append(tcod.namegen_generate(self.sets[self.curset]))

    def ev_keydown(self, event: tcod.event.KeyDown):
        if event.sym == ord("="):
            self.curset += 1
            if self.curset == self.nbsets:
                self.curset = 0
            self.names.append("======")
        elif event.sym == ord("-"):
            self.curset -= 1
            if self.curset < 0:
                self.curset = self.nbsets - 1
            self.names.append("======")
        else:
            super().ev_keydown(event)


#############################################
# python fast render sample
#############################################
numpy_available = True

use_numpy = numpy_available  # default option
SCREEN_W = SAMPLE_SCREEN_WIDTH
SCREEN_H = SAMPLE_SCREEN_HEIGHT
HALF_W = SCREEN_W // 2
HALF_H = SCREEN_H // 2
RES_U = 80  # texture resolution
RES_V = 80
TEX_STRETCH = 5  # texture stretching with tunnel depth
SPEED = 15
LIGHT_BRIGHTNESS = (
    3.5
)  # brightness multiplier for all lights (changes their radius)
LIGHTS_CHANCE = 0.07  # chance of a light appearing
MAX_LIGHTS = 6
MIN_LIGHT_STRENGTH = 0.2
LIGHT_UPDATE = (
    0.05
)  # how much the ambient light changes to reflect current light sources
AMBIENT_LIGHT = 0.8  # brightness of tunnel texture

# the coordinates of all tiles in the screen, as numpy arrays.
# example: (4x3 pixels screen)
# xc = [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]
# yc = [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]]
if numpy_available:
    (xc, yc) = np.meshgrid(range(SCREEN_W), range(SCREEN_H))
    # translate coordinates of all pixels to center
    xc = xc - HALF_W
    yc = yc - HALF_H

noise2d = tcod.noise_new(2, 0.5, 2.0)
if numpy_available:  # the texture starts empty
    texture = np.zeros((RES_U, RES_V))


class Light:
    def __init__(self, x, y, z, r, g, b, strength):
        self.x, self.y, self.z = x, y, z  # pos.
        self.r, self.g, self.b = r, g, b  # color
        self.strength = strength  # between 0 and 1, defines brightness


class FastRenderSample(Sample):
    def __init__(self):
        self.name = "Python fast render"

    def on_enter(self):
        global frac_t, abs_t, lights, tex_r, tex_g, tex_b
        tcod.sys_set_fps(0)
        sample_console.clear()  # render status message
        sample_console.print(
            1, SCREEN_H - 3, "Renderer: NumPy", fg=tcod.white, bg=None
        )

        # time is represented in number of pixels of the texture, start later
        # in time to initialize texture
        frac_t = RES_V - 1
        abs_t = RES_V - 1
        lights = []  # lights list, and current color of the tunnel texture
        tex_r, tex_g, tex_b = 0, 0, 0

    def on_draw(self):
        global use_numpy, frac_t, abs_t, lights, tex_r, tex_g, tex_b, xc, yc
        global texture, texture2, brightness2, R2, G2, B2

        time_delta = tcod.sys_get_last_frame_length() * SPEED  # advance time
        frac_t += time_delta  # increase fractional (always < 1.0) time
        abs_t += time_delta  # increase absolute elapsed time
        # integer time units that passed this frame (number of texture pixels
        # to advance)
        int_t = int(frac_t)
        frac_t -= int_t  # keep this < 1.0

        # change texture color according to presence of lights (basically, sum
        # them to get ambient light and smoothly change the current color into
        # that)
        ambient_r = AMBIENT_LIGHT * sum(
            light.r * light.strength for light in lights
        )
        ambient_g = AMBIENT_LIGHT * sum(
            light.g * light.strength for light in lights
        )
        ambient_b = AMBIENT_LIGHT * sum(
            light.b * light.strength for light in lights
        )
        alpha = LIGHT_UPDATE * time_delta
        tex_r = tex_r * (1 - alpha) + ambient_r * alpha
        tex_g = tex_g * (1 - alpha) + ambient_g * alpha
        tex_b = tex_b * (1 - alpha) + ambient_b * alpha

        if int_t >= 1:
            # roll texture (ie, advance in tunnel) according to int_t
            # can't roll more than the texture's size (can happen when
            # time_delta is large)
            int_t = int_t % RES_V
            # new pixels are based on absolute elapsed time
            int_abs_t = int(abs_t)

            texture = np.roll(texture, -int_t, 1)
            # replace new stretch of texture with new values
            for v in range(RES_V - int_t, RES_V):
                for u in range(0, RES_U):
                    tex_v = (v + int_abs_t) / float(RES_V)
                    texture[u, v] = tcod.noise_get_fbm(
                        noise2d, [u / float(RES_U), tex_v], 32.0
                    ) + tcod.noise_get_fbm(
                        noise2d, [1 - u / float(RES_U), tex_v], 32.0
                    )

        # squared distance from center,
        # clipped to sensible minimum and maximum values
        sqr_dist = xc ** 2 + yc ** 2
        sqr_dist = sqr_dist.clip(1.0 / RES_V, RES_V ** 2)

        # one coordinate into the texture, represents depth in the tunnel
        v = TEX_STRETCH * float(RES_V) / sqr_dist + frac_t
        v = v.clip(0, RES_V - 1)

        # another coordinate, represents rotation around the tunnel
        u = np.mod(RES_U * (np.arctan2(yc, xc) / (2 * np.pi) + 0.5), RES_U)

        # retrieve corresponding pixels from texture
        brightness = texture[u.astype(int), v.astype(int)] / 4.0 + 0.5

        # use the brightness map to compose the final color of the tunnel
        R = brightness * tex_r
        G = brightness * tex_g
        B = brightness * tex_b

        # create new light source
        if (
            random.random() <= time_delta * LIGHTS_CHANCE
            and len(lights) < MAX_LIGHTS
        ):
            x = random.uniform(-0.5, 0.5)
            y = random.uniform(-0.5, 0.5)
            strength = random.uniform(MIN_LIGHT_STRENGTH, 1.0)

            color = tcod.Color(0, 0, 0)  # create bright colors with random hue
            hue = random.uniform(0, 360)
            tcod.color_set_hsv(color, hue, 0.5, strength)
            lights.append(
                Light(x, y, TEX_STRETCH, color.r, color.g, color.b, strength)
            )

        # eliminate lights that are going to be out of view
        lights = [
            light for light in lights if light.z - time_delta > 1.0 / RES_V
        ]

        for light in lights:  # render lights
            # move light's Z coordinate with time, then project its XYZ
            # coordinates to screen-space
            light.z -= float(time_delta) / TEX_STRETCH
            xl = light.x / light.z * SCREEN_H
            yl = light.y / light.z * SCREEN_H

            # calculate brightness of light according to distance from viewer
            # and strength, then calculate brightness of each pixel with
            # inverse square distance law
            light_brightness = (
                LIGHT_BRIGHTNESS
                * light.strength
                * (1.0 - light.z / TEX_STRETCH)
            )
            brightness = light_brightness / ((xc - xl) ** 2 + (yc - yl) ** 2)

            # make all pixels shine around this light
            R += brightness * light.r
            G += brightness * light.g
            B += brightness * light.b

        # truncate values
        R = R.clip(0, 255)
        G = G.clip(0, 255)
        B = B.clip(0, 255)

        # fill the screen with these background colors
        sample_console.bg.transpose(2, 1, 0)[...] = (R, G, B)


#############################################
# main loop
#############################################

RENDERER_KEYS = {
    tcod.event.K_F1: tcod.RENDERER_GLSL,
    tcod.event.K_F2: tcod.RENDERER_OPENGL,
    tcod.event.K_F3: tcod.RENDERER_SDL,
    tcod.event.K_F4: tcod.RENDERER_SDL2,
    tcod.event.K_F5: tcod.RENDERER_OPENGL2,
}

RENDERER_NAMES = (
    "F1 GLSL   ",
    "F2 OPENGL ",
    "F3 SDL    ",
    "F4 SDL2   ",
    "F5 OPENGL2",
)

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
    FastRenderSample(),
)

cur_sample = 0


def main():
    global cur_sample, root_console
    tcod.console_set_custom_font(
        FONT, tcod.FONT_TYPE_GREYSCALE | tcod.FONT_LAYOUT_TCOD
    )
    root_console = tcod.console_init_root(
        80, 50, "python-tcod samples", False, tcod.RENDERER_SDL2, order="F"
    )
    credits_end = False
    SAMPLES[cur_sample].on_enter()
    draw_samples_menu()
    draw_renderer_menu()

    while not tcod.console_is_window_closed():
        root_console.clear()
        draw_samples_menu()
        draw_renderer_menu()
        # render credits
        if not credits_end:
            credits_end = tcod.console_credits_render(60, 43, 0)

        # render the sample
        SAMPLES[cur_sample].on_draw()
        sample_console.blit(root_console, SAMPLE_SCREEN_X, SAMPLE_SCREEN_Y)
        draw_stats()
        tcod.console_flush()
        handle_events()


def handle_events():
    for event in tcod.event.get():
        SAMPLES[cur_sample].dispatch(event)


def draw_samples_menu():
    for i, sample in enumerate(SAMPLES):
        if i == cur_sample:
            fg = tcod.white
            bg = tcod.light_blue
        else:
            fg = tcod.grey
            bg = tcod.black
        root_console.print(
            2,
            46 - (len(SAMPLES) - i),
            "  %s" % sample.name.ljust(19),
            fg,
            bg,
            alignment=tcod.LEFT,
        )


def draw_stats():
    root_console.print(
        79,
        46,
        " last frame : %3d ms (%3d fps)"
        % (tcod.sys_get_last_frame_length() * 1000.0, tcod.sys_get_fps()),
        fg=tcod.grey,
        bg=None,
        alignment=tcod.RIGHT,
    )
    root_console.print(
        79,
        47,
        "elapsed : %8d ms %4.2fs"
        % (time.perf_counter() * 1000, time.perf_counter()),
        fg=tcod.grey,
        bg=None,
        alignment=tcod.RIGHT,
    )


def draw_renderer_menu():
    current_renderer = tcod.sys_get_renderer()
    root_console.print(
        42,
        46 - (tcod.NB_RENDERERS + 1),
        "Renderer :",
        fg=tcod.grey,
        bg=tcod.black,
    )
    for i, name in enumerate(RENDERER_NAMES):
        if i == current_renderer:
            fg = tcod.white
            bg = tcod.light_blue
        else:
            fg = tcod.grey
            bg = tcod.black
        root_console.print(42, 46 - tcod.NB_RENDERERS + i, name, fg, bg)


if __name__ == "__main__":
    main()
