#!/usr/bin/env python
# To the extent possible under law, the libtcod maintainers have waived all
# copyright and related or neighboring rights for this example.  This work is
# published from: United States.
# https://creativecommons.org/publicdomain/zero/1.0/
"""A system to control time since the original libtcod tools are deprecated."""

import statistics
import time
from collections import deque
from typing import Deque, Optional

import tcod

WIDTH, HEIGHT = 720, 480


class Clock:
    """Measure framerate performance and sync to a given framerate.

    Everything important is handled by `Clock.sync`.  You can use the fps
    properties to track the performance of an application.
    """

    def __init__(self) -> None:
        """Initialize this object with empty data."""
        self.last_time = time.perf_counter()  # Last time this was synced.
        self.time_samples: Deque[float] = deque()  # Delta time samples.
        self.max_samples = 64  # Number of fps samples to log.  Can be changed.
        self.drift_time = 0.0  # Tracks how much the last frame was overshot.

    def sync(self, fps: Optional[float] = None) -> float:
        """Sync to a given framerate and return the delta time.

        `fps` is the desired framerate in frames-per-second.  If None is given
        then this function will track the time and framerate without waiting.

        `fps` must be above zero when given.
        """
        if fps is not None:
            # Wait until a target time based on the last time and framerate.
            desired_frame_time = 1 / fps
            target_time = self.last_time + desired_frame_time - self.drift_time
            # Sleep might take slightly longer than asked.
            sleep_time = max(0, target_time - self.last_time - 0.001)
            if sleep_time:
                time.sleep(sleep_time)
            # Busy wait until the target_time is reached.
            while (drift_time := time.perf_counter() - target_time) < 0:
                pass
            self.drift_time = min(drift_time, desired_frame_time)

        # Get the delta time.
        current_time = time.perf_counter()
        delta_time = max(0, current_time - self.last_time)
        self.last_time = current_time

        # Record the performance of the current frame.
        self.time_samples.append(delta_time)
        while len(self.time_samples) > self.max_samples:
            self.time_samples.popleft()

        return delta_time

    @property
    def min_fps(self) -> float:
        """The FPS of the slowest frame."""
        try:
            return 1 / max(self.time_samples)
        except (ValueError, ZeroDivisionError):
            return 0

    @property
    def max_fps(self) -> float:
        """The FPS of the fastest frame."""
        try:
            return 1 / min(self.time_samples)
        except (ValueError, ZeroDivisionError):
            return 0

    @property
    def mean_fps(self) -> float:
        """The FPS of the sampled frames overall."""
        if not self.time_samples:
            return 0
        try:
            return 1 / statistics.fmean(self.time_samples)
        except ZeroDivisionError:
            return 0

    @property
    def median_fps(self) -> float:
        """The FPS of the median frame."""
        if not self.time_samples:
            return 0
        try:
            return 1 / statistics.median(self.time_samples)
        except ZeroDivisionError:
            return 0

    @property
    def last_fps(self) -> float:
        """The FPS of the most recent frame."""
        if not self.time_samples or self.time_samples[-1] == 0:
            return 0
        return 1 / self.time_samples[-1]


def main() -> None:
    """Example program for Clock."""
    # vsync is False in this example, but you'll want it to be True unless you
    # need to benchmark or set framerates above 60 FPS.
    with tcod.context.new(width=WIDTH, height=HEIGHT, vsync=False) as context:
        line_x = 0  # Highlight a line, helpful to measure frames visually.
        clock = Clock()
        delta_time = 0.0  # The time passed between frames.
        desired_fps = 50
        while True:
            console = context.new_console(order="F")
            console.tiles_rgb["bg"][line_x % console.width, :] = (255, 0, 0)
            console.print(
                1,
                1,
                f"Current time:{time.perf_counter() * 1000:8.2f}ms"
                f"\nDelta time:{delta_time * 1000:8.2f}ms"
                f"\nDesired FPS:{desired_fps:3d} (use scroll wheel to adjust)"
                f"\n  last:{clock.last_fps:.2f} fps"
                f"\n  mean:{clock.mean_fps:.2f} fps"
                f"\nmedian:{clock.median_fps:.2f} fps"
                f"\n   min:{clock.min_fps:.2f} fps"
                f"\n   max:{clock.max_fps:.2f} fps",
            )
            context.present(console, integer_scaling=True)
            delta_time = clock.sync(fps=desired_fps)
            line_x += 1

            # Handle events.
            for event in tcod.event.get():
                context.convert_event(event)  # Set tile coordinates for event.
                if isinstance(event, tcod.event.Quit):
                    raise SystemExit
                if isinstance(event, tcod.event.MouseWheel):
                    desired_fps = max(1, desired_fps + event.y)


if __name__ == "__main__":
    main()
