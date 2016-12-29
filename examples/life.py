#!/usr/bin/env python

import random
import time

import tdl

WIDTH = 80
HEIGHT = 40

class LifeBoard():

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.live_cells = set()
        self.wrap = True

    def set(self, x, y, value):
        if value:
            self.live_cells.add((x, y))
        else:
            self.live_cells.discard((x, y))

    def set_batch(self, x, y, batch):
        for y_, line in enumerate(batch):
            for x_, char in enumerate(line):
                self.set(x + x_, y + y_, char != ' ')

    def get(self, x, y):
        if(self.wrap is False
           and not (0 <= x < self.width and 0 <= y < self.height)):
            return False
        return (x % self.width, y % self.height) in self.live_cells

    def clear(self):
        self.live_cells.clear()

    def toggle(self, x, y):
        self.live_cells.symmetric_difference_update([(x, y)])

    def wrap_edges(self):
        for x in range(-1, self.width + 1):
            self.set(x, -1, self.get(x, -1))
            self.set(x, self.height, self.get(x, self.height))
        for y in range(self.height):
            self.set(-1, y, self.get(-1, y))
            self.set(self.width, y, self.get(self.width, y))


    def get_neighbours(self, x, y):
        return len(self.live_cells & {(x - 1, y - 1), (x, y - 1),
                                      (x + 1,y - 1), (x + 1, y),
                                      (x + 1, y + 1), (x, y + 1),
                                      (x - 1, y + 1), (x - 1, y)})

    def rule(self, is_alive, neighbours):
        """
        1. Any live cell with fewer than two live neighbours dies, as if caused
           by under-population.
        2. Any live cell with two or three live neighbours lives on to the next
           generation.
        3. Any live cell with more than three live neighbours dies, as if by
           overcrowding.
        4. Any dead cell with exactly three live neighbours becomes a live
           cell, as if by reproduction.
        """
        if is_alive:
            return 2 <= neighbours <= 3
        else:
            return neighbours == 3

    def step(self):
        self.wrap_edges()
        next_generation = set()
        for x in range(self.width):
            for y in range(self.height):
                if self.rule(self.get(x, y), self.get_neighbours(x, y)):
                    next_generation.add((x, y))
        self.live_cells = next_generation

def main():
    console = tdl.init(WIDTH, HEIGHT)
    board = LifeBoard(WIDTH, HEIGHT - 1)
    # The R-pentomino
    #board.set_batch(WIDTH // 2 - 2,HEIGHT // 2 - 2,
    #                [' **',
    #                 '** ',
    #                 ' * '])

    # Diehard
    #board.set_batch(WIDTH // 2 - 5,HEIGHT // 2 - 2,
    #                ['      * ',
    #                 '**      ',
    #                 ' *   ***'])

    # Gosper glider gun
    board.set_batch(1, 1,
                    ['                                    ',
                     '                        *           ',
                     '                      * *           ',
                     '            **      **            **',
                     '           *   *    **            **',
                     '**        *     *   **              ',
                     '**        *   * **    * *           ',
                     '          *     *       *           ',
                     '           *   *                    ',
                     '            **                      '])

    play = False
    redraw = True
    mouse_drawing = None
    mouse_x = -1
    mouse_y = -1
    while True:
        for event in tdl.event.get():
            if event.type == 'QUIT':
                return
            elif event.type == 'KEYDOWN':
                if event.key == 'SPACE':
                    play = not play
                    redraw = True
                elif event.char.upper() == 'S':
                    board.step()
                    redraw = True
                elif event.char.upper() == 'C':
                    board.clear()
                    redraw = True
                elif event.char.upper() == 'W':
                    board.wrap = not board.wrap
                    redraw = True
            elif event.type == 'MOUSEDOWN':
                x, y, = event.cell
                board.toggle(x, y)
                mouse_drawing = event.cell
                redraw = True
            elif event.type == 'MOUSEUP':
                mouse_drawing = None
            elif event.type == 'MOUSEMOTION':
                if(mouse_drawing and mouse_drawing != event.cell):
                    x, y = mouse_drawing = event.cell
                    board.toggle(x, y)
                mouse_x, mouse_y = event.cell
                redraw = True
        if play and mouse_drawing is None:
            board.step()
            redraw = True
        if redraw:
            redraw = False
            console.clear()
            for x, y in board.live_cells:
                console.draw_char(x, y, '*')
            #console.draw_rect(0, -1, None, None, None, bg=(64, 64, 80))
            console.draw_rect(0, -1, None, None, None, bg=(64, 64, 80))
            console.draw_str(0, -1, "Mouse:Toggle Cells, Space:%5s, [S]tep, [C]lear, [W]rap Turn %s" % (['Play', 'Pause'][play], ['On', 'Off'][board.wrap]), None, None)
            if (mouse_x, mouse_y) in console:
                console.draw_char(mouse_x, mouse_y,
                                  None, (0, 0, 0), (255, 255, 255))
        else:
            time.sleep(0.01)
        tdl.flush()
        tdl.set_title("Conway's Game of Life - %i FPS" % tdl.get_fps())


if __name__ == '__main__':
    main()

