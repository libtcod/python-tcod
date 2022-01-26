from __future__ import annotations

import sys
import threading
import time
import weakref
from typing import Any, Iterator, List, Optional

import numpy as np
from numpy.typing import ArrayLike, DTypeLike, NDArray

import tcod.sdl.sys
from tcod.loader import ffi, lib


def _get_format(format: DTypeLike) -> int:
    """Return a SDL_AudioFormat bitfield from a NumPy dtype."""
    dt: Any = np.dtype(format)
    assert dt.fields is None
    bitsize = dt.itemsize * 8
    assert 0 < bitsize <= lib.SDL_AUDIO_MASK_BITSIZE
    assert dt.str[1] in "uif"
    is_signed = dt.str[1] != "u"
    is_float = dt.str[1] == "f"
    byteorder = dt.byteorder
    if byteorder == "=":
        byteorder = "<" if sys.byteorder == "little" else ">"

    return (  # type: ignore
        bitsize
        | (lib.SDL_AUDIO_MASK_DATATYPE * is_float)
        | (lib.SDL_AUDIO_MASK_ENDIAN * (byteorder == ">"))
        | (lib.SDL_AUDIO_MASK_SIGNED * is_signed)
    )


def _dtype_from_format(format: int) -> np.dtype[Any]:
    """Return a dtype from a SDL_AudioFormat."""
    bitsize = format & lib.SDL_AUDIO_MASK_BITSIZE
    assert bitsize % 8 == 0
    bytesize = bitsize // 8
    byteorder = ">" if format & lib.SDL_AUDIO_MASK_ENDIAN else "<"
    if format & lib.SDL_AUDIO_MASK_DATATYPE:
        kind = "f"
    elif format & lib.SDL_AUDIO_MASK_SIGNED:
        kind = "i"
    else:
        kind = "u"
    return np.dtype(f"{byteorder}{kind}{bytesize}")


class AudioDevice:
    def __init__(
        self,
        device: Optional[str] = None,
        capture: bool = False,
        *,
        frequency: int = 44100,
        format: DTypeLike = np.float32,
        channels: int = 2,
        samples: int = 0,
        allowed_changes: int = 0,
    ):
        self.__sdl_subsystems = tcod.sdl.sys._ScopeInit(tcod.sdl.sys.Subsystem.AUDIO)
        self.__handle = ffi.new_handle(weakref.ref(self))
        desired = ffi.new(
            "SDL_AudioSpec*",
            {
                "freq": frequency,
                "format": _get_format(format),
                "channels": channels,
                "samples": samples,
                "callback": ffi.NULL,
                "userdata": self.__handle,
            },
        )
        obtained = ffi.new("SDL_AudioSpec*")
        self.device_id = lib.SDL_OpenAudioDevice(
            ffi.NULL if device is None else device.encode("utf-8"),
            capture,
            desired,
            obtained,
            allowed_changes,
        )
        assert self.device_id != 0, tcod.sdl.sys._get_error()
        self.frequency = obtained.freq
        self.is_capture = capture
        self.format = _dtype_from_format(obtained.format)
        self.channels = int(obtained.channels)
        self.silence = int(obtained.silence)
        self.samples = int(obtained.samples)
        self.buffer_size = int(obtained.size)
        self.unpause()

    @property
    def _sample_size(self) -> int:
        return self.format.itemsize * self.channels

    def pause(self) -> None:
        lib.SDL_PauseAudioDevice(self.device_id, True)

    def unpause(self) -> None:
        lib.SDL_PauseAudioDevice(self.device_id, False)

    def _verify_array_format(self, samples: NDArray[Any]) -> NDArray[Any]:
        if samples.dtype != self.format:
            raise TypeError(f"Expected an array of dtype {self.format}, got {samples.dtype} instead.")
        return samples

    def _convert_array(self, samples_: ArrayLike) -> NDArray[Any]:
        if isinstance(samples_, np.ndarray):
            samples_ = self._verify_array_format(samples_)
        samples: NDArray[Any] = np.asarray(samples_, dtype=self.format)
        if len(samples.shape) < 2:
            samples = samples[:, np.newaxis]
        return np.ascontiguousarray(np.broadcast_to(samples, (samples.shape[0], self.channels)), dtype=self.format)

    @property
    def queued_audio_bytes(self) -> int:
        return int(lib.SDL_GetQueuedAudioSize(self.device_id))

    def queue_audio(self, samples: ArrayLike) -> None:
        assert not self.is_capture
        samples = self._convert_array(samples)
        buffer = ffi.from_buffer(samples)
        lib.SDL_QueueAudio(self.device_id, buffer, len(buffer))

    def dequeue_audio(self) -> NDArray[Any]:
        assert self.is_capture
        out_samples = self.queued_audio_bytes // self._sample_size
        out = np.empty((out_samples, self.channels), self.format)
        buffer = ffi.from_buffer(out)
        bytes_returned = lib.SDL_DequeueAudio(self.device_id, buffer, len(buffer))
        samples_returned = bytes_returned // self._sample_size
        assert samples_returned == out_samples
        return out

    def __del__(self) -> None:
        self.close()

    def close(self) -> None:
        if not self.device_id:
            return
        lib.SDL_CloseAudioDevice(self.device_id)
        self.device_id = 0

    @staticmethod
    def __default_callback(stream: NDArray[Any], silence: int) -> None:
        stream[...] = silence


class Mixer(threading.Thread):
    def __init__(self, device: AudioDevice):
        super().__init__(daemon=True)
        self.device = device
        self.device.unpause()
        self.start()

    def run(self) -> None:
        buffer = np.full((self.device.samples, self.device.channels), self.device.silence, dtype=self.device.format)
        while True:
            time.sleep(0.001)
            if self.device.queued_audio_bytes == 0:
                self.on_stream(buffer)
                self.device.queue_audio(buffer)
                buffer[:] = self.device.silence

    def on_stream(self, stream: NDArray[Any]) -> None:
        pass


class BasicMixer(Mixer):
    def __init__(self, device: AudioDevice):
        super().__init__(device)
        self.play_buffers: List[List[NDArray[Any]]] = []

    def play(self, sound: ArrayLike) -> None:
        array = np.asarray(sound, dtype=self.device.format)
        assert array.size
        if len(array.shape) == 1:
            array = array[:, np.newaxis]
        chunks: List[NDArray[Any]] = np.split(array, range(0, len(array), self.device.samples)[1:])[::-1]
        self.play_buffers.append(chunks)

    def on_stream(self, stream: NDArray[Any]) -> None:
        super().on_stream(stream)
        for chunks in self.play_buffers:
            chunk = chunks.pop()
            stream[: len(chunk)] += chunk

        self.play_buffers = [chunks for chunks in self.play_buffers if chunks]


@ffi.def_extern()  # type: ignore
def _sdl_audio_callback(userdata: Any, stream: Any, length: int) -> None:
    """Handle audio device callbacks."""
    device: Optional[AudioDevice] = ffi.from_handle(userdata)()
    assert device is not None
    _ = np.frombuffer(ffi.buffer(stream, length), dtype=device.format).reshape(-1, device.channels)


def _get_devices(capture: bool) -> Iterator[str]:
    """Get audio devices from SDL_GetAudioDeviceName."""
    with tcod.sdl.sys._ScopeInit(tcod.sdl.sys.Subsystem.AUDIO):
        device_count = lib.SDL_GetNumAudioDevices(capture)
        for i in range(device_count):
            yield str(ffi.string(lib.SDL_GetAudioDeviceName(i, capture)), encoding="utf-8")


def get_devices() -> Iterator[str]:
    """Iterate over the available audio output devices."""
    yield from _get_devices(capture=False)


def get_capture_devices() -> Iterator[str]:
    """Iterate over the available audio capture devices."""
    yield from _get_devices(capture=True)
