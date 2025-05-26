#!/usr/bin/env python
"""This code demonstrates various usages of python-tcod."""

# To the extent possible under law, the libtcod maintainers have waived all
# copyright and related or neighboring rights to these samples.
# https://creativecommons.org/publicdomain/zero/1.0/
from __future__ import annotations

import copy
import importlib.metadata
import math
import random
import sys
import time
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

import tcod.bsp
import tcod.cffi
import tcod.console
import tcod.context
import tcod.event
import tcod.image
import tcod.los
import tcod.map
import tcod.noise
import tcod.path
import tcod.render
import tcod.sdl.mouse
import tcod.sdl.render
import tcod.tileset
from tcod import libtcodpy
from tcod.sdl.video import WindowFlags

if TYPE_CHECKING:
    from numpy.typing import NDArray

# ruff: noqa: S311

if not sys.warnoptions:
    warnings.simplefilter("default")  # Show all warnings.

DATA_DIR = Path(__file__).parent / "../libtcod/data"
"""Path of the samples data directory."""

assert DATA_DIR.exists(), "Data directory is missing, did you forget to run `git submodule update --init`?"

WHITE = (255, 255, 255)
GREY = (127, 127, 127)
BLACK = (0, 0, 0)
LIGHT_BLUE = (63, 63, 255)
LIGHT_YELLOW = (255, 255, 63)

SAMPLE_SCREEN_WIDTH = 46
SAMPLE_SCREEN_HEIGHT = 20
SAMPLE_SCREEN_X = 20
SAMPLE_SCREEN_Y = 10
FONT = DATA_DIR / "fonts/dejavu10x10_gs_tc.png"

# Mutable global names.
context: tcod.context.Context
tileset: tcod.tileset.Tileset
console_render: tcod.render.SDLConsoleRender  # Optional SDL renderer.
sample_minimap: tcod.sdl.render.Texture  # Optional minimap texture.
root_console = tcod.console.Console(80, 50, order="F")
sample_console = tcod.console.Console(SAMPLE_SCREEN_WIDTH, SAMPLE_SCREEN_HEIGHT, order="F")
cur_sample = 0  # Current selected sample.
frame_times = [time.perf_counter()]
frame_length = [0.0]
START_TIME = time.perf_counter()


def _get_elapsed_time() -> float:
    """Return time passed since the start of the program."""
    return time.perf_counter() - START_TIME


class Sample(tcod.event.EventDispatch[None]):
    def __init__(self, name: str = "") -> None:
        self.name = name

    def on_enter(self) -> None:
        pass

    def on_draw(self) -> None:
        pass

    def ev_keydown(self, event: tcod.event.KeyDown) -> None:
        global cur_sample
        if event.sym == tcod.event.KeySym.DOWN:
            cur_sample = (cur_sample + 1) % len(SAMPLES)
            SAMPLES[cur_sample].on_enter()
            draw_samples_menu()
        elif event.sym == tcod.event.KeySym.UP:
            cur_sample = (cur_sample - 1) % len(SAMPLES)
            SAMPLES[cur_sample].on_enter()
            draw_samples_menu()
        elif event.sym == tcod.event.KeySym.RETURN and event.mod & tcod.event.Modifier.ALT:
            sdl_window = context.sdl_window
            if sdl_window:
                sdl_window.fullscreen = False if sdl_window.fullscreen else WindowFlags.FULLSCREEN_DESKTOP
        elif event.sym in (tcod.event.KeySym.PRINTSCREEN, tcod.event.KeySym.p):
            print("screenshot")
            if event.mod & tcod.event.Modifier.ALT:
                libtcodpy.console_save_apf(root_console, "samples.apf")
                print("apf")
            else:
                libtcodpy.sys_save_screenshot()
                print("png")
        elif event.sym in RENDERER_KEYS:
            # Swap the active context for one with a different renderer.
            init_context(RENDERER_KEYS[event.sym])


class TrueColorSample(Sample):
    def __init__(self) -> None:
        self.name = "True colors"
        # corner colors
        self.colors: NDArray[np.int16] = np.array(
            [(50, 40, 150), (240, 85, 5), (50, 35, 240), (10, 200, 130)],
            dtype=np.int16,
        )
        # color shift direction
        self.slide_dir: NDArray[np.int16] = np.array([[1, 1, 1], [-1, -1, 1], [1, -1, 1], [1, 1, -1]], dtype=np.int16)
        # corner indexes
        self.corners: NDArray[np.int16] = np.array([0, 1, 2, 3], dtype=np.int16)

    def on_draw(self) -> None:
        self.slide_corner_colors()
        self.interpolate_corner_colors()
        self.darken_background_characters()
        self.randomize_sample_console()
        self.print_banner()

    def slide_corner_colors(self) -> None:
        # pick random RGB channels for each corner
        rand_channels = np.random.randint(low=0, high=3, size=4)

        # shift picked color channels in the direction of slide_dir
        self.colors[self.corners, rand_channels] += self.slide_dir[self.corners, rand_channels] * 5

        # reverse slide_dir values when limits are reached
        self.slide_dir[self.colors[:] == 255] = -1
        self.slide_dir[self.colors[:] == 0] = 1

    def interpolate_corner_colors(self) -> None:
        # interpolate corner colors across the sample console
        left = np.linspace(self.colors[0], self.colors[2], SAMPLE_SCREEN_HEIGHT)
        right = np.linspace(self.colors[1], self.colors[3], SAMPLE_SCREEN_HEIGHT)
        sample_console.bg[:] = np.linspace(left, right, SAMPLE_SCREEN_WIDTH)

    def darken_background_characters(self) -> None:
        # darken background characters
        sample_console.fg[:] = sample_console.bg[:]
        sample_console.fg[:] //= 2

    def randomize_sample_console(self) -> None:
        # randomize sample console characters
        sample_console.ch[:] = np.random.randint(
            low=ord("a"),
            high=ord("z") + 1,
            size=sample_console.ch.size,
            dtype=np.intc,
        ).reshape(sample_console.ch.shape)

    def print_banner(self) -> None:
        # print text on top of samples
        sample_console.print_box(
            x=1,
            y=5,
            width=sample_console.width - 2,
            height=sample_console.height - 1,
            string="The Doryen library uses 24 bits colors, for both background and foreground.",
            fg=WHITE,
            bg=GREY,
            bg_blend=libtcodpy.BKGND_MULTIPLY,
            alignment=libtcodpy.CENTER,
        )


class OffscreenConsoleSample(Sample):
    def __init__(self) -> None:
        self.name = "Offscreen console"
        self.secondary = tcod.console.Console(sample_console.width // 2, sample_console.height // 2)
        self.screenshot = tcod.console.Console(sample_console.width, sample_console.height)
        self.counter = 0.0
        self.x = 0
        self.y = 0
        self.x_dir = 1
        self.y_dir = 1

        self.secondary.draw_frame(
            0,
            0,
            sample_console.width // 2,
            sample_console.height // 2,
            "Offscreen console",
            clear=False,
            fg=WHITE,
            bg=BLACK,
        )

        self.secondary.print_box(
            1,
            2,
            sample_console.width // 2 - 2,
            sample_console.height // 2,
            "You can render to an offscreen console and blit in on another one, simulating alpha transparency.",
            fg=WHITE,
            bg=None,
            alignment=libtcodpy.CENTER,
        )

    def on_enter(self) -> None:
        self.counter = _get_elapsed_time()
        # get a "screenshot" of the current sample screen
        sample_console.blit(dest=self.screenshot)

    def on_draw(self) -> None:
        if _get_elapsed_time() - self.counter >= 1:
            self.counter = _get_elapsed_time()
            self.x += self.x_dir
            self.y += self.y_dir
            if self.x == sample_console.width / 2 + 5:
                self.x_dir = -1
            elif self.x == -5:
                self.x_dir = 1
            if self.y == sample_console.height / 2 + 5:
                self.y_dir = -1
            elif self.y == -5:
                self.y_dir = 1
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
    FLAG_NAMES = (
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
    )

    def __init__(self) -> None:
        self.name = "Line drawing"
        self.mk_flag = libtcodpy.BKGND_SET
        self.bk_flag = libtcodpy.BKGND_SET

        self.bk = tcod.console.Console(sample_console.width, sample_console.height, order="F")
        # initialize the colored background
        self.bk.bg[:, :, 0] = np.linspace(0, 255, self.bk.width)[:, np.newaxis]
        self.bk.bg[:, :, 2] = np.linspace(0, 255, self.bk.height)
        self.bk.bg[:, :, 1] = (self.bk.bg[:, :, 0].astype(int) + self.bk.bg[:, :, 2]) / 2
        self.bk.ch[:] = ord(" ")

    def ev_keydown(self, event: tcod.event.KeyDown) -> None:
        if event.sym in (tcod.event.KeySym.RETURN, tcod.event.KeySym.KP_ENTER):
            self.bk_flag += 1
            if (self.bk_flag & 0xFF) > libtcodpy.BKGND_ALPH:
                self.bk_flag = libtcodpy.BKGND_NONE
        else:
            super().ev_keydown(event)

    def on_draw(self) -> None:
        alpha = 0.0
        if (self.bk_flag & 0xFF) == libtcodpy.BKGND_ALPH:
            # for the alpha mode, update alpha every frame
            alpha = (1.0 + math.cos(time.time() * 2)) / 2.0
            self.bk_flag = libtcodpy.BKGND_ALPHA(int(alpha))
        elif (self.bk_flag & 0xFF) == libtcodpy.BKGND_ADDA:
            # for the add alpha mode, update alpha every frame
            alpha = (1.0 + math.cos(time.time() * 2)) / 2.0
            self.bk_flag = libtcodpy.BKGND_ADDALPHA(int(alpha))

        self.bk.blit(sample_console)
        rect_y = int((sample_console.height - 2) * ((1.0 + math.cos(time.time())) / 2.0))
        for x in range(sample_console.width):
            value = x * 255 // sample_console.width
            col = (value, value, value)
            libtcodpy.console_set_char_background(sample_console, x, rect_y, col, self.bk_flag)
            libtcodpy.console_set_char_background(sample_console, x, rect_y + 1, col, self.bk_flag)
            libtcodpy.console_set_char_background(sample_console, x, rect_y + 2, col, self.bk_flag)
        angle = time.time() * 2.0
        cos_angle = math.cos(angle)
        sin_angle = math.sin(angle)
        xo = int(sample_console.width // 2 * (1 + cos_angle))
        yo = int(sample_console.height // 2 + sin_angle * sample_console.width // 2)
        xd = int(sample_console.width // 2 * (1 - cos_angle))
        yd = int(sample_console.height // 2 - sin_angle * sample_console.width // 2)
        # draw the line
        # in python the easiest way is to use the line iterator
        for x, y in tcod.los.bresenham((xo, yo), (xd, yd)).tolist():
            if 0 <= x < sample_console.width and 0 <= y < sample_console.height:
                libtcodpy.console_set_char_background(sample_console, x, y, LIGHT_BLUE, self.bk_flag)
        sample_console.print(
            2,
            2,
            f"{self.FLAG_NAMES[self.bk_flag & 0xFF]} (ENTER to change)",
            fg=WHITE,
            bg=None,
        )


class NoiseSample(Sample):
    NOISE_OPTIONS = (  # (name, algorithm, implementation)
        (
            "perlin noise",
            tcod.noise.Algorithm.PERLIN,
            tcod.noise.Implementation.SIMPLE,
        ),
        (
            "simplex noise",
            tcod.noise.Algorithm.SIMPLEX,
            tcod.noise.Implementation.SIMPLE,
        ),
        (
            "wavelet noise",
            tcod.noise.Algorithm.WAVELET,
            tcod.noise.Implementation.SIMPLE,
        ),
        (
            "perlin fbm",
            tcod.noise.Algorithm.PERLIN,
            tcod.noise.Implementation.FBM,
        ),
        (
            "perlin turbulence",
            tcod.noise.Algorithm.PERLIN,
            tcod.noise.Implementation.TURBULENCE,
        ),
        (
            "simplex fbm",
            tcod.noise.Algorithm.SIMPLEX,
            tcod.noise.Implementation.FBM,
        ),
        (
            "simplex turbulence",
            tcod.noise.Algorithm.SIMPLEX,
            tcod.noise.Implementation.TURBULENCE,
        ),
        (
            "wavelet fbm",
            tcod.noise.Algorithm.WAVELET,
            tcod.noise.Implementation.FBM,
        ),
        (
            "wavelet turbulence",
            tcod.noise.Algorithm.WAVELET,
            tcod.noise.Implementation.TURBULENCE,
        ),
    )

    def __init__(self) -> None:
        self.name = "Noise"
        self.func = 0
        self.dx = 0.0
        self.dy = 0.0
        self.octaves = 4.0
        self.zoom = 3.0
        self.hurst = libtcodpy.NOISE_DEFAULT_HURST
        self.lacunarity = libtcodpy.NOISE_DEFAULT_LACUNARITY
        self.noise = self.get_noise()
        self.img = tcod.image.Image(SAMPLE_SCREEN_WIDTH * 2, SAMPLE_SCREEN_HEIGHT * 2)

    @property
    def algorithm(self) -> int:
        return self.NOISE_OPTIONS[self.func][1]

    @property
    def implementation(self) -> int:
        return self.NOISE_OPTIONS[self.func][2]

    def get_noise(self) -> tcod.noise.Noise:
        return tcod.noise.Noise(
            2,
            self.algorithm,
            self.implementation,
            self.hurst,
            self.lacunarity,
            self.octaves,
            seed=None,
        )

    def on_draw(self) -> None:
        self.dx = _get_elapsed_time() * 0.25
        self.dy = _get_elapsed_time() * 0.25
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
        rect_w = 24
        rect_h = 13
        if self.implementation == tcod.noise.Implementation.SIMPLE:
            rect_h = 10
        sample_console.draw_semigraphics(np.asarray(self.img))
        sample_console.draw_rect(
            2,
            2,
            rect_w,
            rect_h,
            ch=0,
            fg=None,
            bg=GREY,
            bg_blend=libtcodpy.BKGND_MULTIPLY,
        )
        sample_console.fg[2 : 2 + rect_w, 2 : 2 + rect_h] = (
            sample_console.fg[2 : 2 + rect_w, 2 : 2 + rect_h] * GREY / 255
        )

        for cur_func in range(len(self.NOISE_OPTIONS)):
            text = f"{cur_func + 1} : {self.NOISE_OPTIONS[cur_func][0]}"
            if cur_func == self.func:
                sample_console.print(2, 2 + cur_func, text, fg=WHITE, bg=LIGHT_BLUE)
            else:
                sample_console.print(2, 2 + cur_func, text, fg=GREY, bg=None)
        sample_console.print(2, 11, f"Y/H : zoom ({self.zoom:2.1f})", fg=WHITE, bg=None)
        if self.implementation != tcod.noise.Implementation.SIMPLE:
            sample_console.print(
                2,
                12,
                f"E/D : hurst ({self.hurst:2.1f})",
                fg=WHITE,
                bg=None,
            )
            sample_console.print(
                2,
                13,
                f"R/F : lacunarity ({self.lacunarity:2.1f})",
                fg=WHITE,
                bg=None,
            )
            sample_console.print(
                2,
                14,
                f"T/G : octaves ({self.octaves:2.1f})",
                fg=WHITE,
                bg=None,
            )

    def ev_keydown(self, event: tcod.event.KeyDown) -> None:
        if tcod.event.KeySym.N9 >= event.sym >= tcod.event.KeySym.N1:
            self.func = event.sym - tcod.event.KeySym.N1
            self.noise = self.get_noise()
        elif event.sym == tcod.event.KeySym.e:
            self.hurst += 0.1
            self.noise = self.get_noise()
        elif event.sym == tcod.event.KeySym.d:
            self.hurst -= 0.1
            self.noise = self.get_noise()
        elif event.sym == tcod.event.KeySym.r:
            self.lacunarity += 0.5
            self.noise = self.get_noise()
        elif event.sym == tcod.event.KeySym.f:
            self.lacunarity -= 0.5
            self.noise = self.get_noise()
        elif event.sym == tcod.event.KeySym.t:
            self.octaves += 0.5
            self.noise.octaves = self.octaves
        elif event.sym == tcod.event.KeySym.g:
            self.octaves -= 0.5
            self.noise.octaves = self.octaves
        elif event.sym == tcod.event.KeySym.y:
            self.zoom += 0.2
        elif event.sym == tcod.event.KeySym.h:
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

SAMPLE_MAP_ = (
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
)

SAMPLE_MAP: NDArray[Any] = np.array([[ord(c) for c in line] for line in SAMPLE_MAP_]).transpose()

FOV_ALGO_NAMES = (
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
    "SYMMETRIC_SHADOWCAST",
)

TORCH_RADIUS = 10
SQUARED_TORCH_RADIUS = TORCH_RADIUS * TORCH_RADIUS


class FOVSample(Sample):
    def __init__(self) -> None:
        self.name = "Field of view"

        self.player_x = 20
        self.player_y = 10
        self.torch = False
        self.light_walls = True
        self.algo_num = libtcodpy.FOV_SYMMETRIC_SHADOWCAST
        self.noise = tcod.noise.Noise(1)  # 1D noise for the torch flickering.

        map_shape = (SAMPLE_SCREEN_WIDTH, SAMPLE_SCREEN_HEIGHT)

        self.walkable: NDArray[np.bool_] = np.zeros(map_shape, dtype=bool, order="F")
        self.walkable[:] = SAMPLE_MAP[:] == ord(" ")

        self.transparent: NDArray[np.bool_] = np.zeros(map_shape, dtype=bool, order="F")
        self.transparent[:] = self.walkable[:] | (SAMPLE_MAP[:] == ord("="))

        # Lit background colors for the map.
        self.light_map_bg: NDArray[np.uint8] = np.full(SAMPLE_MAP.shape, LIGHT_GROUND, dtype="3B")
        self.light_map_bg[SAMPLE_MAP[:] == ord("#")] = LIGHT_WALL
        # Dark background colors for the map.
        self.dark_map_bg: NDArray[np.uint8] = np.full(SAMPLE_MAP.shape, DARK_GROUND, dtype="3B")
        self.dark_map_bg[SAMPLE_MAP[:] == ord("#")] = DARK_WALL

    def draw_ui(self) -> None:
        sample_console.print(
            1,
            1,
            "IJKL : move around\nT : torch fx {}\nW : light walls {}\n+-: algo {}".format(
                "on " if self.torch else "off",
                "on " if self.light_walls else "off",
                FOV_ALGO_NAMES[self.algo_num],
            ),
            fg=WHITE,
            bg=None,
        )

    def on_draw(self) -> None:
        sample_console.clear()
        # Draw the help text & player @.
        self.draw_ui()
        sample_console.print(self.player_x, self.player_y, "@")
        # Draw windows.
        sample_console.rgb["ch"][SAMPLE_MAP[:] == ord("=")] = 0x2550  # BOX DRAWINGS DOUBLE HORIZONTAL
        sample_console.rgb["fg"][SAMPLE_MAP[:] == ord("=")] = BLACK

        # Get a 2D boolean array of visible cells.
        fov = tcod.map.compute_fov(
            transparency=self.transparent,
            pov=(self.player_x, self.player_y),
            radius=TORCH_RADIUS if self.torch else 0,
            light_walls=self.light_walls,
            algorithm=self.algo_num,
        )

        if self.torch:
            # Derive the touch from noise based on the current time.
            torch_t = _get_elapsed_time() * 5
            # Randomize the light position between -1.5 and 1.5
            torch_x = self.player_x + self.noise.get_point(torch_t) * 1.5
            torch_y = self.player_y + self.noise.get_point(torch_t + 11) * 1.5
            # Extra light brightness.
            brightness = 0.2 * self.noise.get_point(torch_t + 17)

            # Get the squared distance using a mesh grid.
            x, y = np.mgrid[:SAMPLE_SCREEN_WIDTH, :SAMPLE_SCREEN_HEIGHT]
            # Center the mesh grid on the torch position.
            x = x.astype(np.float32) - torch_x
            y = y.astype(np.float32) - torch_y

            distance_squared = x**2 + y**2  # 2D squared distance array.

            # Get the currently visible cells.
            visible = (distance_squared < SQUARED_TORCH_RADIUS) & fov

            # Invert the values, so that the center is the 'brightest' point.
            light = SQUARED_TORCH_RADIUS - distance_squared
            light /= SQUARED_TORCH_RADIUS  # Convert into non-squared distance.
            light += brightness  # Add random brightness.
            light.clip(0, 1, out=light)  # Clamp values in-place.
            light[~visible] = 0  # Set non-visible areas to darkness.

            # Setup background colors for floating point math.
            light_bg: NDArray[np.float16] = self.light_map_bg.astype(np.float16)
            dark_bg: NDArray[np.float16] = self.dark_map_bg.astype(np.float16)

            # Linear interpolation between colors.
            sample_console.rgb["bg"] = dark_bg + (light_bg - dark_bg) * light[..., np.newaxis]
        else:
            sample_console.bg[...] = np.select(
                condlist=[fov[:, :, np.newaxis]],
                choicelist=[self.light_map_bg],
                default=self.dark_map_bg,
            )

    def ev_keydown(self, event: tcod.event.KeyDown) -> None:
        MOVE_KEYS = {  # noqa: N806
            tcod.event.KeySym.i: (0, -1),
            tcod.event.KeySym.j: (-1, 0),
            tcod.event.KeySym.k: (0, 1),
            tcod.event.KeySym.l: (1, 0),
        }
        FOV_SELECT_KEYS = {  # noqa: N806
            tcod.event.KeySym.MINUS: -1,
            tcod.event.KeySym.EQUALS: 1,
            tcod.event.KeySym.KP_MINUS: -1,
            tcod.event.KeySym.KP_PLUS: 1,
        }
        if event.sym in MOVE_KEYS:
            x, y = MOVE_KEYS[event.sym]
            if self.walkable[self.player_x + x, self.player_y + y]:
                self.player_x += x
                self.player_y += y
        elif event.sym == tcod.event.KeySym.t:
            self.torch = not self.torch
        elif event.sym == tcod.event.KeySym.w:
            self.light_walls = not self.light_walls
        elif event.sym in FOV_SELECT_KEYS:
            self.algo_num += FOV_SELECT_KEYS[event.sym]
            self.algo_num %= len(FOV_ALGO_NAMES)
        else:
            super().ev_keydown(event)


class PathfindingSample(Sample):
    def __init__(self) -> None:
        """Initialize this sample."""
        self.name = "Path finding"

        self.player_x = 20
        self.player_y = 10
        self.dest_x = 24
        self.dest_y = 1
        self.using_astar = True
        self.busy = 0.0
        self.cost = SAMPLE_MAP.T[:] == ord(" ")
        self.graph = tcod.path.SimpleGraph(cost=self.cost, cardinal=70, diagonal=99)
        self.pathfinder = tcod.path.Pathfinder(graph=self.graph)

        self.background_console = tcod.console.Console(SAMPLE_SCREEN_WIDTH, SAMPLE_SCREEN_HEIGHT)

        # draw the dungeon
        self.background_console.rgb["fg"] = BLACK
        self.background_console.rgb["bg"] = DARK_GROUND
        self.background_console.rgb["bg"][SAMPLE_MAP.T[:] == ord("#")] = DARK_WALL
        self.background_console.rgb["ch"][SAMPLE_MAP.T[:] == ord("=")] = ord("═")

    def on_enter(self) -> None:
        """Do nothing."""

    def on_draw(self) -> None:
        """Recompute and render pathfinding."""
        self.pathfinder = tcod.path.Pathfinder(graph=self.graph)
        # self.pathfinder.clear() # Known issues, needs fixing  # noqa: ERA001
        self.pathfinder.add_root((self.player_y, self.player_x))

        # draw the dungeon
        self.background_console.blit(dest=sample_console)

        sample_console.print(self.dest_x, self.dest_y, "+", fg=WHITE)
        sample_console.print(self.player_x, self.player_y, "@", fg=WHITE)
        sample_console.print(1, 1, "IJKL / mouse :\nmove destination\nTAB : A*/dijkstra", fg=WHITE, bg=None)
        sample_console.print(1, 4, "Using : A*", fg=WHITE, bg=None)

        if not self.using_astar:
            self.pathfinder.resolve(goal=None)
            reachable = self.pathfinder.distance != np.iinfo(self.pathfinder.distance.dtype).max

            # draw distance from player
            dijkstra_max_dist = float(self.pathfinder.distance[reachable].max())
            np.array(self.pathfinder.distance, copy=True, dtype=np.float32)
            interpolate = self.pathfinder.distance[reachable] * 0.9 / dijkstra_max_dist
            color_delta = (np.array(DARK_GROUND) - np.array(LIGHT_GROUND)).astype(np.float32)
            sample_console.rgb.T["bg"][reachable] = np.array(LIGHT_GROUND) + interpolate[:, np.newaxis] * color_delta

        # draw the path
        path = self.pathfinder.path_to((self.dest_y, self.dest_x))[1:, ::-1]
        sample_console.rgb["bg"][tuple(path.T)] = LIGHT_GROUND

        # move the creature
        self.busy -= frame_length[-1]
        if self.busy <= 0.0:
            self.busy = 0.2
            if len(path):
                self.player_x = int(path.item(0, 0))
                self.player_y = int(path.item(0, 1))

    def ev_keydown(self, event: tcod.event.KeyDown) -> None:
        """Handle movement and UI."""
        if event.sym == tcod.event.KeySym.i and self.dest_y > 0:  # destination move north
            self.dest_y -= 1
        elif event.sym == tcod.event.KeySym.k and self.dest_y < SAMPLE_SCREEN_HEIGHT - 1:  # destination move south
            self.dest_y += 1
        elif event.sym == tcod.event.KeySym.j and self.dest_x > 0:  # destination move west
            self.dest_x -= 1
        elif event.sym == tcod.event.KeySym.l and self.dest_x < SAMPLE_SCREEN_WIDTH - 1:  # destination move east
            self.dest_x += 1
        elif event.sym == tcod.event.KeySym.TAB:
            self.using_astar = not self.using_astar
        else:
            super().ev_keydown(event)

    def ev_mousemotion(self, event: tcod.event.MouseMotion) -> None:
        """Move destination via mouseover."""
        mx = event.tile.x - SAMPLE_SCREEN_X
        my = event.tile.y - SAMPLE_SCREEN_Y
        if 0 <= mx < SAMPLE_SCREEN_WIDTH and 0 <= my < SAMPLE_SCREEN_HEIGHT:
            self.dest_x = mx
            self.dest_y = my


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
def vline(m: NDArray[np.bool_], x: int, y1: int, y2: int) -> None:
    if y1 > y2:
        y1, y2 = y2, y1
    for y in range(y1, y2 + 1):
        m[x, y] = True


# draw a vertical line up until we reach an empty space
def vline_up(m: NDArray[np.bool_], x: int, y: int) -> None:
    while y >= 0 and not m[x, y]:
        m[x, y] = True
        y -= 1


# draw a vertical line down until we reach an empty space
def vline_down(m: NDArray[np.bool_], x: int, y: int) -> None:
    while y < SAMPLE_SCREEN_HEIGHT and not m[x, y]:
        m[x, y] = True
        y += 1


# draw a horizontal line
def hline(m: NDArray[np.bool_], x1: int, y: int, x2: int) -> None:
    if x1 > x2:
        x1, x2 = x2, x1
    for x in range(x1, x2 + 1):
        m[x, y] = True


# draw a horizontal line left until we reach an empty space
def hline_left(m: NDArray[np.bool_], x: int, y: int) -> None:
    while x >= 0 and not m[x, y]:
        m[x, y] = True
        x -= 1


# draw a horizontal line right until we reach an empty space
def hline_right(m: NDArray[np.bool_], x: int, y: int) -> None:
    while x < SAMPLE_SCREEN_WIDTH and not m[x, y]:
        m[x, y] = True
        x += 1


# the class building the dungeon from the bsp nodes
def traverse_node(bsp_map: NDArray[np.bool_], node: tcod.bsp.BSP) -> None:
    if not node.children:
        # calculate the room size
        if bsp_room_walls:
            node.width -= 1
            node.height -= 1
        if bsp_random_room:
            new_width = random.randint(min(node.width, bsp_min_room_size), node.width)
            new_height = random.randint(min(node.height, bsp_min_room_size), node.height)
            node.x += random.randint(0, node.width - new_width)
            node.y += random.randint(0, node.height - new_height)
            node.width, node.height = new_width, new_height
        # dig the room
        bsp_map[node.x : node.x + node.width, node.y : node.y + node.height] = True
    else:
        # resize the node to fit its sons
        left, right = node.children
        node.x = min(left.x, right.x)
        node.y = min(left.y, right.y)
        node.width = max(left.x + left.width, right.x + right.width) - node.x
        node.height = max(left.y + left.height, right.y + right.height) - node.y
        # create a corridor between the two lower nodes
        if node.horizontal:
            # vertical corridor
            if left.x + left.width - 1 < right.x or right.x + right.width - 1 < left.x:
                # no overlapping zone. we need a Z shaped corridor
                x1 = random.randint(left.x, left.x + left.width - 1)
                x2 = random.randint(right.x, right.x + right.width - 1)
                y = random.randint(left.y + left.height, right.y)
                vline_up(bsp_map, x1, y - 1)
                hline(bsp_map, x1, y, x2)
                vline_down(bsp_map, x2, y + 1)
            else:
                # straight vertical corridor
                min_x = max(left.x, right.x)
                max_x = min(left.x + left.width - 1, right.x + right.width - 1)
                x = random.randint(min_x, max_x)
                vline_down(bsp_map, x, right.y)
                vline_up(bsp_map, x, right.y - 1)
        elif left.y + left.height - 1 < right.y or right.y + right.height - 1 < left.y:  # horizontal corridor
            # no overlapping zone. we need a Z shaped corridor
            y1 = random.randint(left.y, left.y + left.height - 1)
            y2 = random.randint(right.y, right.y + right.height - 1)
            x = random.randint(left.x + left.width, right.x)
            hline_left(bsp_map, x - 1, y1)
            vline(bsp_map, x, y1, y2)
            hline_right(bsp_map, x + 1, y2)
        else:
            # straight horizontal corridor
            min_y = max(left.y, right.y)
            max_y = min(left.y + left.height - 1, right.y + right.height - 1)
            y = random.randint(min_y, max_y)
            hline_left(bsp_map, right.x - 1, y)
            hline_right(bsp_map, right.x, y)


class BSPSample(Sample):
    def __init__(self) -> None:
        self.name = "Bsp toolkit"
        self.bsp = tcod.bsp.BSP(1, 1, SAMPLE_SCREEN_WIDTH - 1, SAMPLE_SCREEN_HEIGHT - 1)
        self.bsp_map: NDArray[np.bool_] = np.zeros((SAMPLE_SCREEN_WIDTH, SAMPLE_SCREEN_HEIGHT), dtype=bool, order="F")
        self.bsp_generate()

    def bsp_generate(self) -> None:
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
            self.bsp.split_recursive(bsp_depth, bsp_min_room_size, bsp_min_room_size, 1.5, 1.5)
        self.bsp_refresh()

    def bsp_refresh(self) -> None:
        self.bsp_map[...] = False
        for node in copy.deepcopy(self.bsp).inverted_level_order():
            traverse_node(self.bsp_map, node)

    def on_draw(self) -> None:
        sample_console.clear()
        rooms = "OFF"
        if bsp_random_room:
            rooms = "ON"
        sample_console.print(
            1,
            1,
            "ENTER : rebuild bsp\n"
            "SPACE : rebuild dungeon\n"
            f"+-: bsp depth {bsp_depth}\n"
            f"*/: room size {bsp_min_room_size}\n"
            f"1 : random room size {rooms}",
            fg=WHITE,
            bg=None,
        )
        if bsp_random_room:
            walls = "OFF"
            if bsp_room_walls:
                walls = "ON"
            sample_console.print(1, 6, f"2 : room walls {walls}", fg=WHITE, bg=None)
        # render the level
        for y in range(SAMPLE_SCREEN_HEIGHT):
            for x in range(SAMPLE_SCREEN_WIDTH):
                color = DARK_GROUND if self.bsp_map[x][y] else DARK_WALL
                libtcodpy.console_set_char_background(sample_console, x, y, color, libtcodpy.BKGND_SET)

    def ev_keydown(self, event: tcod.event.KeyDown) -> None:
        global bsp_random_room, bsp_room_walls, bsp_depth, bsp_min_room_size
        if event.sym in (tcod.event.KeySym.RETURN, tcod.event.KeySym.KP_ENTER):
            self.bsp_generate()
        elif event.sym == tcod.event.KeySym.SPACE:
            self.bsp_refresh()
        elif event.sym in (tcod.event.KeySym.EQUALS, tcod.event.KeySym.KP_PLUS):
            bsp_depth += 1
            self.bsp_generate()
        elif event.sym in (tcod.event.KeySym.MINUS, tcod.event.KeySym.KP_MINUS):
            bsp_depth = max(1, bsp_depth - 1)
            self.bsp_generate()
        elif event.sym in (tcod.event.KeySym.N8, tcod.event.KeySym.KP_MULTIPLY):
            bsp_min_room_size += 1
            self.bsp_generate()
        elif event.sym in (tcod.event.KeySym.SLASH, tcod.event.KeySym.KP_DIVIDE):
            bsp_min_room_size = max(2, bsp_min_room_size - 1)
            self.bsp_generate()
        elif event.sym in (tcod.event.KeySym.N1, tcod.event.KeySym.KP_1):
            bsp_random_room = not bsp_random_room
            if not bsp_random_room:
                bsp_room_walls = True
            self.bsp_refresh()
        elif event.sym in (tcod.event.KeySym.N2, tcod.event.KeySym.KP_2):
            bsp_room_walls = not bsp_room_walls
            self.bsp_refresh()
        else:
            super().ev_keydown(event)


class ImageSample(Sample):
    def __init__(self) -> None:
        self.name = "Image toolkit"

        self.img = tcod.image.Image.from_file(DATA_DIR / "img/skull.png")
        self.img.set_key_color(BLACK)
        self.circle = tcod.image.Image.from_file(DATA_DIR / "img/circle.png")

    def on_draw(self) -> None:
        sample_console.clear()
        x = sample_console.width / 2 + math.cos(time.time()) * 10.0
        y = sample_console.height / 2
        scalex = 0.2 + 1.8 * (1.0 + math.cos(time.time() / 2)) / 2.0
        scaley = scalex
        angle = _get_elapsed_time()
        if int(time.time()) % 2:
            # split the color channels of circle.png
            # the red channel
            sample_console.draw_rect(0, 3, 15, 15, 0, None, (255, 0, 0))
            self.circle.blit_rect(sample_console, 0, 3, -1, -1, libtcodpy.BKGND_MULTIPLY)
            # the green channel
            sample_console.draw_rect(15, 3, 15, 15, 0, None, (0, 255, 0))
            self.circle.blit_rect(sample_console, 15, 3, -1, -1, libtcodpy.BKGND_MULTIPLY)
            # the blue channel
            sample_console.draw_rect(30, 3, 15, 15, 0, None, (0, 0, 255))
            self.circle.blit_rect(sample_console, 30, 3, -1, -1, libtcodpy.BKGND_MULTIPLY)
        else:
            # render circle.png with normal blitting
            self.circle.blit_rect(sample_console, 0, 3, -1, -1, libtcodpy.BKGND_SET)
            self.circle.blit_rect(sample_console, 15, 3, -1, -1, libtcodpy.BKGND_SET)
            self.circle.blit_rect(sample_console, 30, 3, -1, -1, libtcodpy.BKGND_SET)
        self.img.blit(sample_console, x, y, libtcodpy.BKGND_SET, scalex, scaley, angle)


class MouseSample(Sample):
    def __init__(self) -> None:
        self.name = "Mouse support"

        self.motion = tcod.event.MouseMotion()
        self.mouse_left = self.mouse_middle = self.mouse_right = 0
        self.log: list[str] = []

    def on_enter(self) -> None:
        sdl_window = context.sdl_window
        if sdl_window:
            tcod.sdl.mouse.warp_in_window(sdl_window, 320, 200)
        tcod.sdl.mouse.show(True)

    def ev_mousemotion(self, event: tcod.event.MouseMotion) -> None:
        self.motion = event

    def ev_mousebuttondown(self, event: tcod.event.MouseButtonDown) -> None:
        if event.button == tcod.event.BUTTON_LEFT:
            self.mouse_left = True
        elif event.button == tcod.event.BUTTON_MIDDLE:
            self.mouse_middle = True
        elif event.button == tcod.event.BUTTON_RIGHT:
            self.mouse_right = True

    def ev_mousebuttonup(self, event: tcod.event.MouseButtonUp) -> None:
        if event.button == tcod.event.BUTTON_LEFT:
            self.mouse_left = False
        elif event.button == tcod.event.BUTTON_MIDDLE:
            self.mouse_middle = False
        elif event.button == tcod.event.BUTTON_RIGHT:
            self.mouse_right = False

    def on_draw(self) -> None:
        sample_console.clear(bg=GREY)
        sample_console.print(
            1,
            1,
            f"Pixel position : {self.motion.position.x:4d}x{self.motion.position.y:4d}\n"
            f"Tile position  : {self.motion.tile.x:4d}x{self.motion.tile.y:4d}\n"
            f"Tile movement  : {self.motion.tile_motion.x:4d}x{self.motion.tile_motion.y:4d}\n"
            f"Left button    : {'ON' if self.mouse_left else 'OFF'}\n"
            f"Right button   : {'ON' if self.mouse_right else 'OFF'}\n"
            f"Middle button  : {'ON' if self.mouse_middle else 'OFF'}\n",
            fg=LIGHT_YELLOW,
            bg=None,
        )
        sample_console.print(
            1,
            10,
            "1 : Hide cursor\n2 : Show cursor",
            fg=LIGHT_YELLOW,
            bg=None,
        )

    def ev_keydown(self, event: tcod.event.KeyDown) -> None:
        if event.sym == tcod.event.KeySym.N1:
            tcod.sdl.mouse.show(False)
        elif event.sym == tcod.event.KeySym.N2:
            tcod.sdl.mouse.show(True)
        else:
            super().ev_keydown(event)


class NameGeneratorSample(Sample):
    def __init__(self) -> None:
        self.name = "Name generator"

        self.current_set = 0
        self.delay = 0.0
        self.names: list[str] = []
        self.sets: list[str] = []

    def on_draw(self) -> None:
        if not self.sets:
            # parse all *.cfg files in data/namegen
            for file in (DATA_DIR / "namegen").iterdir():
                if file.suffix == ".cfg":
                    libtcodpy.namegen_parse(file)
            # get the sets list
            self.sets = libtcodpy.namegen_get_sets()
            print(self.sets)
        while len(self.names) > 15:
            self.names.pop(0)
        sample_console.clear(bg=GREY)
        sample_console.print(
            1,
            1,
            f"{self.sets[self.current_set]}\n\n+ : next generator\n- : prev generator",
            fg=WHITE,
            bg=None,
        )
        for i in range(len(self.names)):
            sample_console.print(
                SAMPLE_SCREEN_WIDTH - 1,
                2 + i,
                self.names[i],
                fg=WHITE,
                bg=None,
                alignment=libtcodpy.RIGHT,
            )
        self.delay += frame_length[-1]
        if self.delay > 0.5:
            self.delay -= 0.5
            self.names.append(libtcodpy.namegen_generate(self.sets[self.current_set]))

    def ev_keydown(self, event: tcod.event.KeyDown) -> None:
        if event.sym == tcod.event.KeySym.EQUALS:
            self.current_set += 1
            self.names.append("======")
        elif event.sym == tcod.event.KeySym.MINUS:
            self.current_set -= 1
            self.names.append("======")
        else:
            super().ev_keydown(event)
        self.current_set %= len(self.sets)


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
LIGHT_BRIGHTNESS = 3.5  # brightness multiplier for all lights (changes their radius)
LIGHTS_CHANCE = 0.07  # chance of a light appearing
MAX_LIGHTS = 6
MIN_LIGHT_STRENGTH = 0.2
LIGHT_UPDATE = 0.05  # how much the ambient light changes to reflect current light sources
AMBIENT_LIGHT = 0.8  # brightness of tunnel texture

# the coordinates of all tiles in the screen, as numpy arrays.
# example: (4x3 pixels screen)
# xc = [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]  # noqa: ERA001
# yc = [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]]  # noqa: ERA001
if numpy_available:
    (xc, yc) = np.meshgrid(range(SCREEN_W), range(SCREEN_H))
    # translate coordinates of all pixels to center
    xc = xc - HALF_W
    yc = yc - HALF_H

noise2d = tcod.noise.Noise(2, hurst=0.5, lacunarity=2.0)
if numpy_available:  # the texture starts empty
    texture = np.zeros((RES_U, RES_V))


class Light:
    def __init__(
        self,
        x: float,
        y: float,
        z: float,
        r: int,
        g: int,
        b: int,
        strength: float,
    ) -> None:
        self.x, self.y, self.z = x, y, z  # pos.
        self.r, self.g, self.b = r, g, b  # color
        self.strength = strength  # between 0 and 1, defines brightness


class FastRenderSample(Sample):
    def __init__(self) -> None:
        self.name = "Python fast render"

    def on_enter(self) -> None:
        sample_console.clear()  # render status message
        sample_console.print(1, SCREEN_H - 3, "Renderer: NumPy", fg=WHITE, bg=None)

        # time is represented in number of pixels of the texture, start later
        # in time to initialize texture
        self.frac_t: float = RES_V - 1
        self.abs_t: float = RES_V - 1
        # light and current color of the tunnel texture
        self.lights: list[Light] = []
        self.tex_r = 0.0
        self.tex_g = 0.0
        self.tex_b = 0.0

    def on_draw(self) -> None:
        global texture

        time_delta = frame_length[-1] * SPEED  # advance time
        self.frac_t += time_delta  # increase fractional (always < 1.0) time
        self.abs_t += time_delta  # increase absolute elapsed time
        # integer time units that passed this frame (number of texture pixels
        # to advance)
        int_t = int(self.frac_t)
        self.frac_t %= 1.0  # keep this < 1.0

        # change texture color according to presence of lights (basically, sum
        # them to get ambient light and smoothly change the current color into
        # that)
        ambient_r = AMBIENT_LIGHT * sum(light.r * light.strength for light in self.lights)
        ambient_g = AMBIENT_LIGHT * sum(light.g * light.strength for light in self.lights)
        ambient_b = AMBIENT_LIGHT * sum(light.b * light.strength for light in self.lights)
        alpha = LIGHT_UPDATE * time_delta
        self.tex_r = self.tex_r * (1 - alpha) + ambient_r * alpha
        self.tex_g = self.tex_g * (1 - alpha) + ambient_g * alpha
        self.tex_b = self.tex_b * (1 - alpha) + ambient_b * alpha

        if int_t >= 1:
            # roll texture (ie, advance in tunnel) according to int_t
            # can't roll more than the texture's size (can happen when
            # time_delta is large)
            int_t = int_t % RES_V
            # new pixels are based on absolute elapsed time
            int_abs_t = int(self.abs_t)

            texture = np.roll(texture, -int_t, 1)  # type: ignore[assignment]
            # replace new stretch of texture with new values
            for v in range(RES_V - int_t, RES_V):
                for u in range(RES_U):
                    tex_v = (v + int_abs_t) / float(RES_V)
                    texture[u, v] = libtcodpy.noise_get_fbm(
                        noise2d, [u / float(RES_U), tex_v], 32.0
                    ) + libtcodpy.noise_get_fbm(noise2d, [1 - u / float(RES_U), tex_v], 32.0)

        # squared distance from center,
        # clipped to sensible minimum and maximum values
        sqr_dist = xc**2 + yc**2
        sqr_dist = sqr_dist.clip(1.0 / RES_V, RES_V**2)

        # one coordinate into the texture, represents depth in the tunnel
        vv = TEX_STRETCH * float(RES_V) / sqr_dist + self.frac_t
        vv = vv.clip(0, RES_V - 1)

        # another coordinate, represents rotation around the tunnel
        uu = np.mod(RES_U * (np.arctan2(yc, xc) / (2 * np.pi) + 0.5), RES_U)

        # retrieve corresponding pixels from texture
        brightness = texture[uu.astype(int), vv.astype(int)] / 4.0 + 0.5

        # use the brightness map to compose the final color of the tunnel
        rr = brightness * self.tex_r
        gg = brightness * self.tex_g
        bb = brightness * self.tex_b

        # create new light source
        if random.random() <= time_delta * LIGHTS_CHANCE and len(self.lights) < MAX_LIGHTS:
            x = random.uniform(-0.5, 0.5)
            y = random.uniform(-0.5, 0.5)
            strength = random.uniform(MIN_LIGHT_STRENGTH, 1.0)

            color = libtcodpy.Color(0, 0, 0)  # create bright colors with random hue
            hue = random.uniform(0, 360)
            libtcodpy.color_set_hsv(color, hue, 0.5, strength)
            self.lights.append(Light(x, y, TEX_STRETCH, color.r, color.g, color.b, strength))

        # eliminate lights that are going to be out of view
        self.lights = [light for light in self.lights if light.z - time_delta > 1.0 / RES_V]

        for light in self.lights:  # render lights
            # move light's Z coordinate with time, then project its XYZ
            # coordinates to screen-space
            light.z -= float(time_delta) / TEX_STRETCH
            xl = light.x / light.z * SCREEN_H
            yl = light.y / light.z * SCREEN_H

            # calculate brightness of light according to distance from viewer
            # and strength, then calculate brightness of each pixel with
            # inverse square distance law
            light_brightness = LIGHT_BRIGHTNESS * light.strength * (1.0 - light.z / TEX_STRETCH)
            brightness = light_brightness / ((xc - xl) ** 2 + (yc - yl) ** 2)

            # make all pixels shine around this light
            rr += brightness * light.r
            gg += brightness * light.g
            bb += brightness * light.b

        # truncate values
        rr = rr.clip(0, 255)
        gg = gg.clip(0, 255)
        bb = bb.clip(0, 255)

        # fill the screen with these background colors
        sample_console.bg.transpose(2, 1, 0)[...] = (rr, gg, bb)


#############################################
# main loop
#############################################

RENDERER_KEYS = {
    tcod.event.KeySym.F1: libtcodpy.RENDERER_GLSL,
    tcod.event.KeySym.F2: libtcodpy.RENDERER_OPENGL,
    tcod.event.KeySym.F3: libtcodpy.RENDERER_SDL,
    tcod.event.KeySym.F4: libtcodpy.RENDERER_SDL2,
    tcod.event.KeySym.F5: libtcodpy.RENDERER_OPENGL2,
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


def init_context(renderer: int) -> None:
    """Setup or reset a global context with common parameters set.

    This function exists to more easily switch between renderers.
    """
    global context, console_render, sample_minimap
    if "context" in globals():
        context.close()
    libtcod_version = (
        f"{tcod.cffi.lib.TCOD_MAJOR_VERSION}.{tcod.cffi.lib.TCOD_MINOR_VERSION}.{tcod.cffi.lib.TCOD_PATCHLEVEL}"
    )
    context = tcod.context.new(
        columns=root_console.width,
        rows=root_console.height,
        title=f"""python-tcod samples (python-tcod {importlib.metadata.version("tcod")}, libtcod {libtcod_version})""",
        vsync=False,  # VSync turned off since this is for benchmarking.
        tileset=tileset,
    )
    if context.sdl_renderer:  # If this context supports SDL rendering.
        # Start by setting the logical size so that window resizing doesn't break anything.
        context.sdl_renderer.logical_size = (
            tileset.tile_width * root_console.width,
            tileset.tile_height * root_console.height,
        )
        assert context.sdl_atlas
        # Generate the console renderer and minimap.
        console_render = tcod.render.SDLConsoleRender(context.sdl_atlas)
        sample_minimap = context.sdl_renderer.new_texture(
            SAMPLE_SCREEN_WIDTH,
            SAMPLE_SCREEN_HEIGHT,
            format=tcod.cffi.lib.SDL_PIXELFORMAT_RGB24,
            access=tcod.sdl.render.TextureAccess.STREAMING,  # Updated every frame.
        )


def main() -> None:
    global context, tileset
    tileset = tcod.tileset.load_tilesheet(FONT, 32, 8, tcod.tileset.CHARMAP_TCOD)
    init_context(libtcodpy.RENDERER_SDL2)
    try:
        SAMPLES[cur_sample].on_enter()

        while True:
            root_console.clear()
            draw_samples_menu()
            draw_renderer_menu()

            # render the sample
            SAMPLES[cur_sample].on_draw()
            sample_console.blit(root_console, SAMPLE_SCREEN_X, SAMPLE_SCREEN_Y)
            draw_stats()
            if context.sdl_renderer:
                # Clear the screen to ensure no garbage data outside of the logical area is displayed
                context.sdl_renderer.draw_color = (0, 0, 0, 255)
                context.sdl_renderer.clear()
                # SDL renderer support, upload the sample console background to a minimap texture.
                sample_minimap.update(sample_console.rgb.T["bg"])
                # Render the root_console normally, this is the drawing step of context.present without presenting.
                context.sdl_renderer.copy(console_render.render(root_console))
                # Render the minimap to the screen.
                context.sdl_renderer.copy(
                    sample_minimap,
                    dest=(
                        tileset.tile_width * 24,
                        tileset.tile_height * 36,
                        SAMPLE_SCREEN_WIDTH * 3,
                        SAMPLE_SCREEN_HEIGHT * 3,
                    ),
                )
                context.sdl_renderer.present()
            else:  # No SDL renderer, just use plain context rendering.
                context.present(root_console)

            handle_time()
            handle_events()
    finally:
        # Normally context would be used in a with block and closed
        # automatically. but since this context might be switched to one with a
        # different renderer it is closed manually here.
        context.close()


def handle_time() -> None:
    if len(frame_times) > 100:
        frame_times.pop(0)
        frame_length.pop(0)
    frame_times.append(time.perf_counter())
    frame_length.append(frame_times[-1] - frame_times[-2])


def handle_events() -> None:
    for event in tcod.event.get():
        if context.sdl_renderer:
            # Manual handing of tile coordinates since context.present is skipped.
            if isinstance(event, (tcod.event.MouseState, tcod.event.MouseMotion)):
                event.tile = tcod.event.Point(
                    event.position.x // tileset.tile_width, event.position.y // tileset.tile_height
                )
            if isinstance(event, tcod.event.MouseMotion):
                prev_tile = (
                    (event.position[0] - event.motion[0]) // tileset.tile_width,
                    (event.position[1] - event.motion[1]) // tileset.tile_height,
                )
                event.tile_motion = tcod.event.Point(event.tile[0] - prev_tile[0], event.tile[1] - prev_tile[1])
        else:
            context.convert_event(event)

        SAMPLES[cur_sample].dispatch(event)
        if isinstance(event, tcod.event.Quit):
            raise SystemExit
        if isinstance(event, tcod.event.KeyDown) and event.sym == tcod.event.KeySym.ESCAPE:
            raise SystemExit


def draw_samples_menu() -> None:
    for i, sample in enumerate(SAMPLES):
        if i == cur_sample:
            fg = WHITE
            bg = LIGHT_BLUE
        else:
            fg = GREY
            bg = BLACK
        root_console.print(
            2,
            46 - (len(SAMPLES) - i),
            f"  {sample.name.ljust(19)}",
            fg,
            bg,
            alignment=libtcodpy.LEFT,
        )


def draw_stats() -> None:
    try:
        fps = 1 / (sum(frame_length) / len(frame_length))
    except ZeroDivisionError:
        fps = 0
    root_console.print(
        root_console.width,
        46,
        f"last frame :{frame_length[-1] * 1000.0:5.1f} ms ({int(fps):4d} fps)",
        fg=GREY,
        alignment=libtcodpy.RIGHT,
    )
    root_console.print(
        root_console.width,
        47,
        f"elapsed : {int(_get_elapsed_time() * 1000):8d} ms {_get_elapsed_time():5.2f}s",
        fg=GREY,
        alignment=libtcodpy.RIGHT,
    )


def draw_renderer_menu() -> None:
    root_console.print(
        42,
        46 - (libtcodpy.NB_RENDERERS + 1),
        "Renderer :",
        fg=GREY,
        bg=BLACK,
    )
    for i, name in enumerate(RENDERER_NAMES):
        if i == context.renderer_type:
            fg = WHITE
            bg = LIGHT_BLUE
        else:
            fg = GREY
            bg = BLACK
        root_console.print(42, 46 - libtcodpy.NB_RENDERERS + i, name, fg, bg)


if __name__ == "__main__":
    main()
