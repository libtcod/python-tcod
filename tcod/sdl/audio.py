"""SDL2 audio playback and recording tools.

.. versionadded:: 13.5
"""
from __future__ import annotations

import enum
import sys
import threading
import time
from typing import Any, Callable, Dict, Hashable, Iterator, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike, DTypeLike, NDArray
from typing_extensions import Final, Literal

import tcod.sdl.sys
from tcod.loader import ffi, lib
from tcod.sdl import _get_error


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

    return int(
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
    """An SDL audio device.

    Open new audio devices using :any:`tcod.sdl.audio.open`.
    """

    def __init__(
        self,
        device_id: int,
        capture: bool,
        spec: Any,  # SDL_AudioSpec*
    ):
        assert device_id >= 0
        assert ffi.typeof(spec) is ffi.typeof("SDL_AudioSpec*")
        assert spec
        self.device_id: Final[int] = device_id
        """The SDL device identifier used for SDL C functions."""
        self.spec: Final[Any] = spec
        """The SDL_AudioSpec as a CFFI object."""
        self.frequency: Final[int] = spec.freq
        """The audio device sound frequency."""
        self.is_capture: Final[bool] = capture
        """True if this is a recording device instead of an output device."""
        self.format: Final[np.dtype[Any]] = _dtype_from_format(spec.format)
        """The format used for audio samples with this device."""
        self.channels: Final[int] = int(spec.channels)
        """The number of audio channels for this device."""
        self.silence: float = int(spec.silence)
        """The value of silence, according to SDL."""
        self.buffer_samples: Final[int] = int(spec.samples)
        """The size of the audio buffer in samples."""
        self.buffer_bytes: Final[int] = int(spec.size)
        """The size of the audio buffer in bytes."""
        self._handle: Optional[Any] = None
        self._callback: Callable[[AudioDevice, NDArray[Any]], None] = self.__default_callback

    @property
    def callback(self) -> Callable[[AudioDevice, NDArray[Any]], None]:
        """If the device was opened with a callback enabled, then you may get or set the callback with this attribute."""
        if self._handle is None:
            raise TypeError("This AudioDevice was opened without a callback.")
        return self._callback

    @callback.setter
    def callback(self, new_callback: Callable[[AudioDevice, NDArray[Any]], None]) -> None:
        if self._handle is None:
            raise TypeError("This AudioDevice was opened without a callback.")
        self._callback = new_callback

    @property
    def _sample_size(self) -> int:
        """The size of a sample in bytes."""
        return self.format.itemsize * self.channels

    @property
    def stopped(self) -> bool:
        """Is True if the device has failed or was closed."""
        return bool(lib.SDL_GetAudioDeviceStatus(self.device_id) != lib.SDL_AUDIO_STOPPED)

    @property
    def paused(self) -> bool:
        """Get or set the device paused state."""
        return bool(lib.SDL_GetAudioDeviceStatus(self.device_id) != lib.SDL_AUDIO_PLAYING)

    @paused.setter
    def paused(self, value: bool) -> None:
        lib.SDL_PauseAudioDevice(self.device_id, value)

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
    def _queued_bytes(self) -> int:
        """The current amount of bytes remaining in the audio queue."""
        return int(lib.SDL_GetQueuedAudioSize(self.device_id))

    @property
    def queued_samples(self) -> int:
        """The current amount of samples remaining in the audio queue."""
        return self._queued_bytes // self._sample_size

    def queue_audio(self, samples: ArrayLike) -> None:
        """Append audio samples to the audio data queue."""
        assert not self.is_capture
        samples = self._convert_array(samples)
        buffer = ffi.from_buffer(samples)
        lib.SDL_QueueAudio(self.device_id, buffer, len(buffer))

    def dequeue_audio(self) -> NDArray[Any]:
        """Return the audio buffer from a capture stream."""
        assert self.is_capture
        out_samples = self._queued_bytes // self._sample_size
        out = np.empty((out_samples, self.channels), self.format)
        buffer = ffi.from_buffer(out)
        bytes_returned = lib.SDL_DequeueAudio(self.device_id, buffer, len(buffer))
        samples_returned = bytes_returned // self._sample_size
        assert samples_returned == out_samples
        return out

    def __del__(self) -> None:
        self.close()

    def close(self) -> None:
        """Close this audio device.  Using this object after it has been closed is invalid."""
        if not hasattr(self, "device_id"):
            return
        lib.SDL_CloseAudioDevice(self.device_id)
        del self.device_id

    @staticmethod
    def __default_callback(device: AudioDevice, stream: NDArray[Any]) -> None:
        stream[...] = device.silence


class Channel:
    mixer: Mixer

    def __init__(self) -> None:
        self.volume: Union[float, Tuple[float, ...]] = 1.0
        self.sound_queue: List[NDArray[Any]] = []
        self.on_end_callback: Optional[Callable[[Channel], None]] = None

    @property
    def busy(self) -> bool:
        return bool(self.sound_queue)

    def play(
        self,
        sound: ArrayLike,
        *,
        on_end: Optional[Callable[[Channel], None]] = None,
    ) -> None:
        self.sound_queue[:] = [self._verify_audio_sample(sound)]
        self.on_end_callback = on_end

    def _verify_audio_sample(self, sample: ArrayLike) -> NDArray[Any]:
        """Verify an audio sample is valid and return it as a Numpy array."""
        array: NDArray[Any] = np.asarray(sample)
        assert array.dtype == self.mixer.device.format
        if len(array.shape) == 1:
            array = array[:, np.newaxis]
        return array

    def _on_mix(self, stream: NDArray[Any]) -> None:
        while self.sound_queue and stream.size:
            buffer = self.sound_queue[0]
            if buffer.shape[0] > stream.shape[0]:
                # Mix part of the buffer into the stream.
                stream[:] += buffer[: stream.shape[0]] * self.volume
                self.sound_queue[0] = buffer[stream.shape[0] :]
                break  # Stream was filled.
            # Remaining buffer fits the stream array.
            stream[: buffer.shape[0]] += buffer * self.volume
            stream = stream[buffer.shape[0] :]
            self.sound_queue.pop(0)
            if not self.sound_queue and self.on_end_callback is not None:
                self.on_end_callback(self)

    def fadeout(self, time: float) -> None:
        assert time >= 0
        time_samples = round(time * self.mixer.device.frequency) + 1
        buffer: NDArray[np.float32] = np.zeros((time_samples, self.mixer.device.channels), np.float32)
        self._on_mix(buffer)
        buffer *= np.linspace(1.0, 0.0, time_samples + 1, endpoint=False)[1:]
        self.sound_queue[:] = [buffer]

    def stop(self) -> None:
        self.fadeout(0.0005)


class Mixer(threading.Thread):
    def __init__(self, device: AudioDevice):
        assert device.format == np.float32
        super().__init__(daemon=True)
        self.device = device

    def run(self) -> None:
        buffer = np.full(
            (self.device.buffer_samples, self.device.channels), self.device.silence, dtype=self.device.format
        )
        while True:
            if self.device._queued_bytes > 0:
                time.sleep(0.001)
                continue
            self.on_stream(buffer)
            self.device.queue_audio(buffer)
            buffer[:] = self.device.silence

    def on_stream(self, stream: NDArray[Any]) -> None:
        pass


class BasicMixer(Mixer):
    def __init__(self, device: AudioDevice):
        super().__init__(device)
        self.channels: Dict[Hashable, Channel] = {}

    def get_channel(self, key: Hashable) -> Channel:
        if key not in self.channels:
            self.channels[key] = Channel()
            self.channels[key].mixer = self
        return self.channels[key]

    def get_free_channel(self) -> Channel:
        i = 0
        while True:
            if not self.get_channel(i).busy:
                return self.channels[i]
            i += 1

    def play(
        self,
        sound: ArrayLike,
        *,
        on_end: Optional[Callable[[Channel], None]] = None,
    ) -> Channel:
        channel = self.get_free_channel()
        channel.play(sound, on_end=on_end)
        return channel

    def on_stream(self, stream: NDArray[Any]) -> None:
        super().on_stream(stream)
        for channel in list(self.channels.values()):
            channel._on_mix(stream)


class _AudioCallbackUserdata:
    device: AudioDevice


@ffi.def_extern()  # type: ignore
def _sdl_audio_callback(userdata: Any, stream: Any, length: int) -> None:
    """Handle audio device callbacks."""
    data: _AudioCallbackUserdata = ffi.from_handle(userdata)()
    device = data.device
    buffer = np.frombuffer(ffi.buffer(stream, length), dtype=device.format).reshape(-1, device.channels)
    device._callback(device, buffer)


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


class AllowedChanges(enum.IntFlag):
    """Which parameters are allowed to be changed when the values given are not supported."""

    NONE = 0
    """"""
    FREQUENCY = 0x01
    """"""
    FORMAT = 0x02
    """"""
    CHANNELS = 0x04
    """"""
    SAMPLES = 0x08
    """"""
    ANY = FREQUENCY | FORMAT | CHANNELS | SAMPLES
    """"""


def open(
    name: Optional[str] = None,
    capture: bool = False,
    *,
    frequency: int = 44100,
    format: DTypeLike = np.float32,
    channels: int = 2,
    samples: int = 0,
    allowed_changes: AllowedChanges = AllowedChanges.NONE,
    paused: bool = False,
    callback: Union[None, Literal[True], Callable[[AudioDevice, NDArray[Any]], None]] = None,
) -> AudioDevice:
    """Open an audio device for playback or capture and return it.

    Args:
        name: The name of the device to open, or None for the most reasonable default.
        capture: True if this is a recording device, or False if this is an output device.
        frequency: The desired sample rate to open the device with.
        format: The data format to use for samples as a NumPy dtype.
        channels: The number of speakers for the device. 1, 2, 4, or 6 are typical options.
        samples:  The desired size of the audio buffer, must be a power of two.
        allowed_changes:
            By default if the hardware does not support the desired format than SDL will transparently convert between
            formats for you.
            Otherwise you can specify which parameters are allowed to be changed to fit the hardware better.
        paused:
            If True then the device will begin in a paused state.
            It can then be unpaused by assigning False to :any:`AudioDevice.paused`.
        callback:
            If None then this device will be opened in push mode and you'll have to use :any:`AudioDevice.queue_audio`
            to send audio data or :any:`AudioDevice.dequeue_audio` to receive it.
            If a callback is given then you can change it later, but you can not enable or disable the callback on an
            opened device.
            If True then a default callback which plays silence will be used, this is useful if you need the audio
            device before your callback is ready.

    If a callback is given then it will be called with the `AudioDevice` and a Numpy buffer of the data stream.
    This callback will be run on a separate thread.
    Exceptions not handled by the callback become unraiseable and will be handled by :any:`sys.unraisablehook`.

    .. seealso::
        https://wiki.libsdl.org/SDL_AudioSpec
        https://wiki.libsdl.org/SDL_OpenAudioDevice

    """
    tcod.sdl.sys.init(tcod.sdl.sys.Subsystem.AUDIO)
    desired = ffi.new(
        "SDL_AudioSpec*",
        {
            "freq": frequency,
            "format": _get_format(format),
            "channels": channels,
            "samples": samples,
            "callback": ffi.NULL,
            "userdata": ffi.NULL,
        },
    )
    callback_data = _AudioCallbackUserdata()
    if callback is not None:
        handle = ffi.new_handle(callback_data)
        desired.callback = lib._sdl_audio_callback
        desired.userdata = handle
    else:
        handle = None

    obtained = ffi.new("SDL_AudioSpec*")
    device_id: int = lib.SDL_OpenAudioDevice(
        ffi.NULL if name is None else name.encode("utf-8"),
        capture,
        desired,
        obtained,
        allowed_changes,
    )
    assert device_id >= 0, _get_error()
    device = AudioDevice(device_id, capture, obtained)
    if callback is not None:
        callback_data.device = device
        device._handle = handle
        if callback is not True:
            device._callback = callback
    device.paused = paused
    return device
