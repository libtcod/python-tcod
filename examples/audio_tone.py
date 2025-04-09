#!/usr/bin/env python
"""Shows how to use tcod.sdl.audio to play audio.

Opens an audio device using SDL then plays tones using various methods.
"""

import math
import time

import attrs
import numpy as np
from scipy import signal  # type: ignore[import-untyped]

import tcod.sdl.audio

VOLUME = 10 ** (-12 / 10)  # -12dB, square waves can be loud


@attrs.define
class PullWave:
    """Square wave stream generator for an SDL audio device in pull mode."""

    frequency: float
    time: float = 0.0

    def __call__(self, stream: tcod.sdl.audio.AudioStream, request: tcod.sdl.audio.AudioStreamCallbackData) -> None:
        """Stream a square wave to SDL on demand.

        This function must run faster than the stream duration.
        Numpy is used to keep performance within these limits.
        """
        duration = request.additional_samples / self.frequency

        t = np.linspace(self.time, self.time + duration, request.additional_samples, endpoint=False)
        self.time += duration
        wave = signal.square(t * (math.tau * 440)).astype(np.float32)
        stream.queue_audio(wave)


if __name__ == "__main__":
    device = tcod.sdl.audio.get_default_playback().open(channels=1, frequency=44100)
    print(f"{device.name=}")
    device.gain = VOLUME
    print(device)

    print("Sawtooth wave queued with AudioStream.queue_audio")
    stream = device.new_stream(format=np.float32, channels=1, frequency=44100)
    t = np.linspace(0, 1.0, 44100, endpoint=False)
    wave = signal.sawtooth(t * (math.tau * 440)).astype(np.float32)
    stream.queue_audio(wave)
    stream.flush()
    while stream.queued_samples:
        time.sleep(0.01)

    print("---")
    time.sleep(0.5)

    print("Square wave attached to AudioStream.getter_callback")
    stream = device.new_stream(format=np.float32, channels=1, frequency=44100)
    stream.getter_callback = PullWave(device.frequency)

    time.sleep(1)
    stream.getter_callback = None

    print("---")
    time.sleep(0.5)

    print("Sawtooth wave played with BasicMixer.play")
    mixer = tcod.sdl.audio.BasicMixer(device, frequency=44100, channels=2)
    channel = mixer.play(wave)
    while channel.busy:
        time.sleep(0.01)

    print("---")
    device.close()
