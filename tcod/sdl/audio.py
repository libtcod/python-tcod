"""SDL2 audio playback and recording tools.

This module includes SDL's low-level audio API and a naive implementation of an SDL mixer.
If you have experience with audio mixing then you might be better off writing your own mixer or
modifying the existing one which was written using Python/Numpy.

This module is designed to integrate with the wider Python ecosystem.
It leaves the loading to sound samples to other libraries like
`SoundFile <https://pysoundfile.readthedocs.io/en/latest/>`_.

Example::

    # Synchronous audio example using SDL's low-level API.
    import time

    import soundfile  # pip install soundfile
    import tcod.sdl.audio

    device = tcod.sdl.audio.open()  # Open the default output device.
    sound, sample_rate = soundfile.read("example_sound.wav", dtype="float32")  # Load an audio sample using SoundFile.
    converted = device.convert(sound, sample_rate)  # Convert this sample to the format expected by the device.
    device.queue_audio(converted)  # Play audio synchronously by appending it to the device buffer.

    while device.queued_samples:  # Wait until device is done playing.
        time.sleep(0.001)

Example::

    # Asynchronous audio example using BasicMixer.
    import time

    import soundfile  # pip install soundfile
    import tcod.sdl.audio

    mixer = tcod.sdl.audio.BasicMixer(tcod.sdl.audio.open())  # Setup BasicMixer with the default audio output.
    sound, sample_rate = soundfile.read("example_sound.wav")  # Load an audio sample using SoundFile.
    sound = mixer.device.convert(sound, sample_rate)  # Convert this sample to the format expected by the device.
    channel = mixer.play(sound)  # Start asynchronous playback, audio is mixed on a separate Python thread.
    while channel.busy:  # Wait until the sample is done playing.
        time.sleep(0.001)

.. versionadded:: 13.5
"""

from __future__ import annotations

import enum
import sys
import threading
import time
from types import TracebackType
from typing import Any, Callable, Final, Hashable, Iterator

import numpy as np
from numpy.typing import ArrayLike, DTypeLike, NDArray
from typing_extensions import Literal, Self

import tcod.sdl.sys
from tcod.cffi import ffi, lib
from tcod.sdl._internal import _check, _get_error, _ProtectedContext


def _get_format(format: DTypeLike) -> int:
    """Return a SDL_AudioFormat bit-field from a NumPy dtype."""
    dt: Any = np.dtype(format)
    assert dt.fields is None
    bitsize = dt.itemsize * 8
    assert 0 < bitsize <= lib.SDL_AUDIO_MASK_BITSIZE
    if dt.str[1] not in "uif":
        msg = f"Unexpected dtype: {dt}"
        raise TypeError(msg)
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
    """Return a dtype from a SDL_AudioFormat.

    >>> _dtype_from_format(tcod.lib.AUDIO_F32LSB)
    dtype('float32')
    >>> _dtype_from_format(tcod.lib.AUDIO_F32MSB)
    dtype('>f4')
    >>> _dtype_from_format(tcod.lib.AUDIO_S16LSB)
    dtype('int16')
    >>> _dtype_from_format(tcod.lib.AUDIO_S16MSB)
    dtype('>i2')
    >>> _dtype_from_format(tcod.lib.AUDIO_U16LSB)
    dtype('uint16')
    >>> _dtype_from_format(tcod.lib.AUDIO_U16MSB)
    dtype('>u2')
    """
    bitsize = format & lib.SDL_AUDIO_MASK_BITSIZE
    assert bitsize % 8 == 0
    byte_size = bitsize // 8
    byteorder = ">" if format & lib.SDL_AUDIO_MASK_ENDIAN else "<"
    if format & lib.SDL_AUDIO_MASK_DATATYPE:
        kind = "f"
    elif format & lib.SDL_AUDIO_MASK_SIGNED:
        kind = "i"
    else:
        kind = "u"
    return np.dtype(f"{byteorder}{kind}{byte_size}")


def convert_audio(
    in_sound: ArrayLike, in_rate: int, *, out_rate: int, out_format: DTypeLike, out_channels: int
) -> NDArray[Any]:
    """Convert an audio sample into a format supported by this device.

    Returns the converted array.  This might be a reference to the input array if no conversion was needed.

    Args:
        in_sound: The input ArrayLike sound sample.  Input format and channels are derived from the array.
        in_rate: The sample-rate of the input array.
        out_rate: The sample-rate of the output array.
        out_format: The output format of the converted array.
        out_channels: The number of audio channels of the output array.

    .. versionadded:: 13.6

    .. versionchanged:: 16.0
        Now converts floating types to `np.float32` when SDL doesn't support the specific format.

    .. seealso::
        :any:`AudioDevice.convert`
    """
    in_array: NDArray[Any] = np.asarray(in_sound)
    if len(in_array.shape) == 1:
        in_array = in_array[:, np.newaxis]
    if len(in_array.shape) != 2:  # noqa: PLR2004
        msg = f"Expected a 1 or 2 ndim input, got {in_array.shape} instead."
        raise TypeError(msg)
    cvt = ffi.new("SDL_AudioCVT*")
    in_channels = in_array.shape[1]
    in_format = _get_format(in_array.dtype)
    out_sdl_format = _get_format(out_format)
    try:
        if (
            _check(lib.SDL_BuildAudioCVT(cvt, in_format, in_channels, in_rate, out_sdl_format, out_channels, out_rate))
            == 0
        ):
            return in_array  # No conversion needed.
    except RuntimeError as exc:
        if (  # SDL now only supports float32, but later versions may add more support for more formats.
            exc.args[0] == "Invalid source format"
            and np.issubdtype(in_array.dtype, np.floating)
            and in_array.dtype != np.float32
        ):
            return convert_audio(  # Try again with float32
                in_array.astype(np.float32),
                in_rate,
                out_rate=out_rate,
                out_format=out_format,
                out_channels=out_channels,
            )
        raise
    # Upload to the SDL_AudioCVT buffer.
    cvt.len = in_array.itemsize * in_array.size
    out_buffer = cvt.buf = ffi.new("uint8_t[]", cvt.len * cvt.len_mult)
    np.frombuffer(ffi.buffer(out_buffer[0 : cvt.len]), dtype=in_array.dtype).reshape(in_array.shape)[:] = in_array

    _check(lib.SDL_ConvertAudio(cvt))
    out_array: NDArray[Any] = (
        np.frombuffer(ffi.buffer(out_buffer[0 : cvt.len_cvt]), dtype=out_format).reshape(-1, out_channels).copy()
    )
    return out_array


class AudioDevice:
    """An SDL audio device.

    Open new audio devices using :any:`tcod.sdl.audio.open`.

    When you use this object directly the audio passed to :any:`queue_audio` is always played synchronously.
    For more typical asynchronous audio you should pass an AudioDevice to :any:`BasicMixer`.

    .. versionchanged:: 16.0
        Can now be used as a context which will close the device on exit.
    """

    def __init__(
        self,
        device_id: int,
        capture: bool,
        spec: Any,  # SDL_AudioSpec*  # noqa: ANN401
    ) -> None:
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
        self._handle: Any | None = None
        self._callback: Callable[[AudioDevice, NDArray[Any]], None] = self.__default_callback

    def __repr__(self) -> str:
        """Return a representation of this device."""
        if self.stopped:
            return f"<{self.__class__.__name__}() stopped=True>"
        items = [
            f"{self.__class__.__name__}(device_id={self.device_id})",
            f"frequency={self.frequency}",
            f"is_capture={self.is_capture}",
            f"format={self.format}",
            f"channels={self.channels}",
            f"buffer_samples={self.buffer_samples}",
            f"buffer_bytes={self.buffer_bytes}",
            f"paused={self.paused}",
        ]

        if self.silence:
            items.append(f"silence={self.silence}")
        if self._handle is not None:
            items.append(f"callback={self._callback}")
        return f"""<{" ".join(items)}>"""

    @property
    def callback(self) -> Callable[[AudioDevice, NDArray[Any]], None]:
        """If the device was opened with a callback enabled, then you may get or set the callback with this attribute."""
        if self._handle is None:
            msg = "This AudioDevice was opened without a callback."
            raise TypeError(msg)
        return self._callback

    @callback.setter
    def callback(self, new_callback: Callable[[AudioDevice, NDArray[Any]], None]) -> None:
        if self._handle is None:
            msg = "This AudioDevice was opened without a callback."
            raise TypeError(msg)
        self._callback = new_callback

    @property
    def _sample_size(self) -> int:
        """The size of a sample in bytes."""
        return self.format.itemsize * self.channels

    @property
    def stopped(self) -> bool:
        """Is True if the device has failed or was closed."""
        if not hasattr(self, "device_id"):
            return True
        return bool(lib.SDL_GetAudioDeviceStatus(self.device_id) == lib.SDL_AUDIO_STOPPED)

    @property
    def paused(self) -> bool:
        """Get or set the device paused state."""
        return bool(lib.SDL_GetAudioDeviceStatus(self.device_id) != lib.SDL_AUDIO_PLAYING)

    @paused.setter
    def paused(self, value: bool) -> None:
        lib.SDL_PauseAudioDevice(self.device_id, value)

    def _verify_array_format(self, samples: NDArray[Any]) -> NDArray[Any]:
        if samples.dtype != self.format:
            msg = f"Expected an array of dtype {self.format}, got {samples.dtype} instead."
            raise TypeError(msg)
        return samples

    def _convert_array(self, samples_: ArrayLike) -> NDArray[Any]:
        if isinstance(samples_, np.ndarray):
            samples_ = self._verify_array_format(samples_)
        samples: NDArray[Any] = np.asarray(samples_, dtype=self.format)
        if len(samples.shape) < 2:  # noqa: PLR2004
            samples = samples[:, np.newaxis]
        return np.ascontiguousarray(np.broadcast_to(samples, (samples.shape[0], self.channels)), dtype=self.format)

    def convert(self, sound: ArrayLike, rate: int | None = None) -> NDArray[Any]:
        """Convert an audio sample into a format supported by this device.

        Returns the converted array.  This might be a reference to the input array if no conversion was needed.

        Args:
            sound: An ArrayLike sound sample.
            rate: The sample-rate of the input array.
                  If None is given then it's assumed to be the same as the device.

        .. versionadded:: 13.6

        .. seealso::
            :any:`convert_audio`
        """
        in_array: NDArray[Any] = np.asarray(sound)
        if len(in_array.shape) == 1:
            in_array = in_array[:, np.newaxis]
        return convert_audio(
            in_sound=sound,
            in_rate=rate if rate is not None else self.frequency,
            out_channels=self.channels if in_array.shape[1] > 1 else 1,
            out_format=self.format,
            out_rate=self.frequency,
        )

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

    def __enter__(self) -> Self:
        """Return self and enter a managed context."""
        return self

    def __exit__(
        self, type: type[BaseException] | None, value: BaseException | None, traceback: TracebackType | None
    ) -> None:
        """Close the device when exiting the context."""
        self.close()

    @staticmethod
    def __default_callback(device: AudioDevice, stream: NDArray[Any]) -> None:
        stream[...] = device.silence


class _LoopSoundFunc:
    def __init__(self, sound: NDArray[Any], loops: int, on_end: Callable[[Channel], None] | None) -> None:
        self.sound = sound
        self.loops = loops
        self.on_end = on_end

    def __call__(self, channel: Channel) -> None:
        if not self.loops:
            if self.on_end is not None:
                self.on_end(channel)
            return
        channel.play(self.sound, volume=channel.volume, on_end=self)
        if self.loops > 0:
            self.loops -= 1


class Channel:
    """An audio channel for :any:`BasicMixer`.  Use :any:`BasicMixer.get_channel` to initialize this object.

    .. versionadded:: 13.6
    """

    mixer: BasicMixer
    """The :any:`BasicMixer` is channel belongs to."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.volume: float | tuple[float, ...] = 1.0
        self.sound_queue: list[NDArray[Any]] = []
        self.on_end_callback: Callable[[Channel], None] | None = None

    @property
    def busy(self) -> bool:
        """Is True when this channel is playing audio."""
        return bool(self.sound_queue)

    def play(
        self,
        sound: ArrayLike,
        *,
        volume: float | tuple[float, ...] = 1.0,
        loops: int = 0,
        on_end: Callable[[Channel], None] | None = None,
    ) -> None:
        """Play an audio sample, stopping any audio currently playing on this channel.

        Parameters are the same as :any:`BasicMixer.play`.
        """
        sound = self._verify_audio_sample(sound)
        with self._lock:
            self.volume = volume
            self.sound_queue[:] = [sound]
            self.on_end_callback = on_end
            if loops:
                self.on_end_callback = _LoopSoundFunc(sound, loops, on_end)

    def _verify_audio_sample(self, sample: ArrayLike) -> NDArray[Any]:
        """Verify an audio sample is valid and return it as a Numpy array."""
        array: NDArray[Any] = np.asarray(sample)
        if array.dtype != self.mixer.device.format:
            msg = f"Audio sample must be dtype={self.mixer.device.format}, input was dtype={array.dtype}"
            raise TypeError(msg)
        if len(array.shape) == 1:
            array = array[:, np.newaxis]
        return array

    def _on_mix(self, stream: NDArray[Any]) -> None:
        """Mix the next part of this channels audio into an active audio stream."""
        with self._lock:
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
        """Fadeout this channel then stop playing."""
        with self._lock:
            if not self.sound_queue:
                return
            time_samples = round(time * self.mixer.device.frequency) + 1
            buffer: NDArray[np.float32] = np.zeros((time_samples, self.mixer.device.channels), np.float32)
            self._on_mix(buffer)
            buffer *= np.linspace(1.0, 0.0, time_samples + 1, endpoint=False)[1:, np.newaxis]
            self.sound_queue[:] = [buffer]

    def stop(self) -> None:
        """Stop audio on this channel."""
        self.fadeout(0.0005)


class BasicMixer(threading.Thread):
    """An SDL sound mixer implemented in Python and Numpy.

    .. versionadded:: 13.6
    """

    def __init__(self, device: AudioDevice) -> None:
        self.channels: dict[Hashable, Channel] = {}
        assert device.format == np.float32
        super().__init__(daemon=True)
        self.device = device
        """The :any:`AudioDevice`"""
        self._lock = threading.RLock()
        self._running = True
        self.start()

    def run(self) -> None:
        buffer = np.full(
            (self.device.buffer_samples, self.device.channels), self.device.silence, dtype=self.device.format
        )
        while self._running:
            if self.device._queued_bytes > 0:
                time.sleep(0.001)
                continue
            self._on_stream(buffer)
            self.device.queue_audio(buffer)
            buffer[:] = self.device.silence

    def close(self) -> None:
        """Shutdown this mixer, all playing audio will be abruptly stopped."""
        self._running = False

    def get_channel(self, key: Hashable) -> Channel:
        """Return a channel tied to with the given key.

        Channels are initialized as you access them with this function.
        :any:`int` channels starting from zero are used internally.

        This can be used to generate a ``"music"`` channel for example.
        """
        with self._lock:
            if key not in self.channels:
                self.channels[key] = Channel()
                self.channels[key].mixer = self
            return self.channels[key]

    def _get_next_channel(self) -> Channel:
        """Return the next available channel for the play method."""
        with self._lock:
            i = 0
            while True:
                if not self.get_channel(i).busy:
                    return self.channels[i]
                i += 1

    def play(
        self,
        sound: ArrayLike,
        *,
        volume: float | tuple[float, ...] = 1.0,
        loops: int = 0,
        on_end: Callable[[Channel], None] | None = None,
    ) -> Channel:
        """Play a sound, return the channel the sound is playing on.

        Args:
            sound: The sound to play.  This a Numpy array matching the format of the loaded audio device.
            volume: The volume to play the sound at.
                    You can also pass a tuple of floats to set the volume for each channel/speaker.
            loops: How many times to play the sound, `-1` can be used to loop the sound forever.
            on_end: A function to call when this sound has ended.
                    This is called with the :any:`Channel` which was playing the sound.
        """
        channel = self._get_next_channel()
        channel.play(sound, volume=volume, loops=loops, on_end=on_end)
        return channel

    def stop(self) -> None:
        """Stop playback on all channels from this mixer."""
        with self._lock:
            for channel in self.channels.values():
                channel.stop()

    def _on_stream(self, stream: NDArray[Any]) -> None:
        """Called to fill the audio buffer."""
        with self._lock:
            for channel in list(self.channels.values()):
                channel._on_mix(stream)


class _AudioCallbackUserdata:
    device: AudioDevice


@ffi.def_extern()  # type: ignore
def _sdl_audio_callback(userdata: Any, stream: Any, length: int) -> None:  # noqa: ANN401
    """Handle audio device callbacks."""
    data: _AudioCallbackUserdata = ffi.from_handle(userdata)
    device = data.device
    buffer = np.frombuffer(ffi.buffer(stream, length), dtype=device.format).reshape(-1, device.channels)
    with _ProtectedContext(device):
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


def open(  # noqa: PLR0913
    name: str | None = None,
    capture: bool = False,
    *,
    frequency: int = 44100,
    format: DTypeLike = np.float32,
    channels: int = 2,
    samples: int = 0,
    allowed_changes: AllowedChanges = AllowedChanges.NONE,
    paused: bool = False,
    callback: None | Literal[True] | Callable[[AudioDevice, NDArray[Any]], None] = None,
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
