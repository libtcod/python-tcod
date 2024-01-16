#!/usr/bin/env python
"""Shows how to use tcod.sdl.audio to play a custom-made audio stream.

Opens an audio device using SDL and plays a square wave for 1 second.
"""
import math
import time
from typing import Any

import attrs
import numpy as np
from numpy.typing import NDArray
from scipy import signal  # type: ignore

import tcod.sdl.audio

VOLUME = 10 ** (-12 / 10)  # -12dB, square waves can be loud


@attrs.define
class PullWave:
    """Square wave stream generator for an SDL audio device in pull mode."""

    time: float = 0.0

    def __call__(self, device: tcod.sdl.audio.AudioDevice, stream: NDArray[Any]) -> None:
        """Stream a square wave to SDL on demand.

        This function must run faster than the stream duration.
        Numpy is used to keep performance within these limits.
        """
        sample_rate = device.frequency
        n_samples = device.buffer_samples
        duration = n_samples / sample_rate
        print(f"{duration=} {self.time=}")

        t = np.linspace(self.time, self.time + duration, n_samples, endpoint=False)
        self.time += duration
        wave = signal.square(t * (math.tau * 440)).astype(np.float32)
        wave *= VOLUME

        stream[:] = device.convert(wave)


if __name__ == "__main__":
    with tcod.sdl.audio.open(callback=PullWave()) as device:
        print(device)
        time.sleep(1)
