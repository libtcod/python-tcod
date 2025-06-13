"""SDL2 audio playback and recording tools.

This module includes SDL's low-level audio API and a naive implementation of an SDL mixer.
If you have experience with audio mixing then you might be better off writing your own mixer or
modifying the existing one which was written using Python/Numpy.

This module is designed to integrate with the wider Python ecosystem.
It leaves the loading to sound samples to other libraries like
`SoundFile <https://pysoundfile.readthedocs.io/en/latest/>`_.

Example:
    # Synchronous audio example
    import time

    import soundfile  # pip install soundfile
    import tcod.sdl.audio

    device = tcod.sdl.get_default_playback().open()  # Open the default output device

    # AudioDevice's can be opened again to form a hierarchy
    # This can be used to give music and sound effects their own configuration
    device_music = device.open()
    device_music.gain = 0  # Mute music
    device_effects = device.open()
    device_effects.gain = 10 ** (-6 / 10)  # -6dB

    sound, sample_rate = soundfile.read("example_sound.wav", dtype="float32")  # Load an audio sample using SoundFile
    stream = device_effects.new_stream(format=sound.dtype, frequency=sample_rate, channels=sound.shape[1])
    stream.queue_audio(sound)  # Play audio by appending it to the audio stream
    stream.flush()

    while stream.queued_samples:  # Wait until stream is finished
        time.sleep(0.001)

.. versionadded:: 13.5
"""

from __future__ import annotations

import contextlib
import enum
import sys
import threading
import weakref
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, Literal, NamedTuple

import numpy as np
from typing_extensions import Self, deprecated

import tcod.sdl.sys
from tcod.cffi import ffi, lib
from tcod.sdl._internal import _check, _check_float, _check_int, _check_p

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Iterable, Iterator
    from types import TracebackType

    from numpy.typing import ArrayLike, DTypeLike, NDArray


def _get_format(format: DTypeLike, /) -> int:  # noqa: A002
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
        | (lib.SDL_AUDIO_MASK_FLOAT * is_float)
        | (lib.SDL_AUDIO_MASK_BIG_ENDIAN * (byteorder == ">"))
        | (lib.SDL_AUDIO_MASK_SIGNED * is_signed)
    )


def _dtype_from_format(format: int, /) -> np.dtype[Any]:  # noqa: A002
    """Return a dtype from a SDL_AudioFormat.

    >>> _dtype_from_format(tcod.lib.SDL_AUDIO_F32LE)
    dtype('float32')
    >>> _dtype_from_format(tcod.lib.SDL_AUDIO_F32BE)
    dtype('>f4')
    >>> _dtype_from_format(tcod.lib.SDL_AUDIO_S16LE)
    dtype('int16')
    >>> _dtype_from_format(tcod.lib.SDL_AUDIO_S16BE)
    dtype('>i2')
    >>> _dtype_from_format(tcod.lib.SDL_AUDIO_S8)
    dtype('int8')
    >>> _dtype_from_format(tcod.lib.SDL_AUDIO_U8)
    dtype('uint8')
    """
    bitsize = format & lib.SDL_AUDIO_MASK_BITSIZE
    assert bitsize % 8 == 0
    byte_size = bitsize // 8
    byteorder = ">" if format & lib.SDL_AUDIO_MASK_BIG_ENDIAN else "<"
    if format & lib.SDL_AUDIO_MASK_FLOAT:
        kind = "f"
    elif format & lib.SDL_AUDIO_MASK_SIGNED:
        kind = "i"
    else:
        kind = "u"
    return np.dtype(f"{byteorder}{kind}{byte_size}")


def _silence_value_for_format(dtype: DTypeLike, /) -> int:
    """Return the silence value for the given dtype format."""
    return int(lib.SDL_GetSilenceValueForFormat(_get_format(dtype)))


class _AudioSpec(NamedTuple):
    """Named tuple for `SDL_AudioSpec`."""

    format: int
    channels: int
    frequency: int

    @classmethod
    def from_c(cls, c_spec_p: Any) -> Self:  # noqa: ANN401
        return cls(int(c_spec_p.format), int(c_spec_p.channels), int(c_spec_p.freq))

    @property
    def _dtype(self) -> np.dtype[Any]:
        return _dtype_from_format(self.format)


def convert_audio(
    in_sound: ArrayLike, in_rate: int, *, out_rate: int, out_format: DTypeLike, out_channels: int
) -> NDArray[np.number]:
    """Convert an audio sample into a format supported by this device.

    Returns the converted array in the shape `(sample, channel)`.
    This will reference the input array data if no conversion was needed.

    Args:
        in_sound: The input ArrayLike sound sample.  Input format and channels are derived from the array.
        in_rate: The sample-rate of the input array.
        out_rate: The sample-rate of the output array.
        out_format: The output format of the converted array.
        out_channels: The number of audio channels of the output array.

    Examples::

        >>> tcod.sdl.audio.convert_audio(np.zeros(5), 44100, out_rate=44100, out_format=np.uint8, out_channels=1).T
        array([[128, 128, 128, 128, 128]], dtype=uint8)
        >>> tcod.sdl.audio.convert_audio(np.zeros(3), 22050, out_rate=44100, out_format=np.int8, out_channels=2).T
        array([[0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0]], dtype=int8)

    .. versionadded:: 13.6

    .. versionchanged:: 16.0
        Now converts floating types to `np.float32` when SDL doesn't support the specific format.

    .. seealso::
        :any:`AudioDevice.convert`
    """
    in_array: NDArray[Any] = np.asarray(in_sound)
    if len(in_array.shape) == 1:
        in_array = in_array[:, np.newaxis]
    elif len(in_array.shape) != 2:  # noqa: PLR2004
        msg = f"Expected a 1 or 2 ndim input, got {in_array.shape} instead."
        raise TypeError(msg)
    in_spec = _AudioSpec(format=_get_format(in_array.dtype), channels=in_array.shape[1], frequency=in_rate)
    out_spec = _AudioSpec(format=_get_format(out_format), channels=out_channels, frequency=out_rate)
    if in_spec == out_spec:
        return in_array  # No conversion needed

    out_buffer = ffi.new("uint8_t**")
    out_length = ffi.new("int*")
    try:
        _check(
            lib.SDL_ConvertAudioSamples(
                [in_spec],
                ffi.from_buffer("const uint8_t*", in_array),
                len(in_array) * in_array.itemsize,
                [out_spec],
                out_buffer,
                out_length,
            )
        )
        return (  # type: ignore[no-any-return]
            np.frombuffer(ffi.buffer(out_buffer[0], out_length[0]), dtype=out_format).reshape(-1, out_channels).copy()
        )
    except RuntimeError as exc:
        if (  # SDL now only supports float32, but later versions may add more support for more formats.
            exc.args[0] == "Parameter 'src_spec->format' is invalid"
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
    finally:
        lib.SDL_free(out_buffer[0])


class AudioDevice:
    """An SDL audio device.

    Example:
        device = tcod.sdl.audio.get_default_playback().open()  # Open a common audio device

    .. versionchanged:: 16.0
        Can now be used as a context which will close the device on exit.

    .. versionchanged:: 19.0
        Removed `spec` and `callback` attribute.

        `queued_samples`, `queue_audio`, and `dequeue_audio` moved to :any:`AudioStream` class.

    """

    __slots__ = (
        "__weakref__",
        "_device_id",
        "buffer_bytes",
        "buffer_samples",
        "channels",
        "device_id",
        "format",
        "frequency",
        "is_capture",
        "is_physical",
        "silence",
    )

    def __init__(
        self,
        device_id: Any,  # noqa: ANN401
        /,
    ) -> None:
        """Initialize the class from a raw `SDL_AudioDeviceID`."""
        assert device_id >= 0
        assert ffi.typeof(device_id) is ffi.typeof("SDL_AudioDeviceID"), ffi.typeof(device_id)
        spec = ffi.new("SDL_AudioSpec*")
        samples = ffi.new("int*")
        _check(lib.SDL_GetAudioDeviceFormat(device_id, spec, samples))
        self._device_id: object = device_id
        self.device_id: Final[int] = int(device_id)
        """The SDL device identifier used for SDL C functions."""
        self.frequency: Final[int] = spec.freq
        """The audio device sound frequency."""
        self.is_capture: Final[bool] = bool(not lib.SDL_IsAudioDevicePlayback(device_id))
        """True if this is a recording device instead of an output device."""
        self.format: Final[np.dtype[Any]] = _dtype_from_format(spec.format)
        """The format used for audio samples with this device."""
        self.channels: Final[int] = int(spec.channels)
        """The number of audio channels for this device."""
        self.silence: float = int(lib.SDL_GetSilenceValueForFormat(spec.format))
        """The value of silence, according to SDL."""
        self.buffer_samples: Final[int] = int(samples[0])
        """The size of the audio buffer in samples."""
        self.buffer_bytes: Final[int] = int(self.format.itemsize * self.channels * self.buffer_samples)
        """The size of the audio buffer in bytes."""
        self.is_physical: Final[bool] = bool(lib.SDL_IsAudioDevicePhysical(device_id))
        """True of this is a physical device, or False if this is a logical device.

        .. versionadded:: 19.0
        """

    def __repr__(self) -> str:
        """Return a representation of this device."""
        items = [
            f"{self.__class__.__name__}(device_id={self.device_id})",
            f"frequency={self.frequency}",
            f"is_capture={self.is_capture}",
            f"is_physical={self.is_physical}",
            f"format={self.format}",
            f"channels={self.channels}",
            f"buffer_samples={self.buffer_samples}",
            f"buffer_bytes={self.buffer_bytes}",
            f"paused={self.paused}",
        ]

        if self.silence:
            items.append(f"silence={self.silence}")
        return f"""<{" ".join(items)}>"""

    @property
    def name(self) -> str:
        """Name of the device.

        .. versionadded:: 19.0
        """
        return str(ffi.string(_check_p(lib.SDL_GetAudioDeviceName(self.device_id))), encoding="utf-8")

    @property
    def gain(self) -> float:
        """Get or set the logical audio device gain.

        Default is 1.0 but can be set higher or zero.

        .. versionadded:: 19.0
        """
        return _check_float(lib.SDL_GetAudioDeviceGain(self.device_id), failure=-1.0)

    @gain.setter
    def gain(self, value: float, /) -> None:
        _check(lib.SDL_SetAudioDeviceGain(self.device_id, value))

    def open(
        self,
        format: DTypeLike | None = None,  # noqa: A002
        channels: int | None = None,
        frequency: int | None = None,
    ) -> Self:
        """Open a new logical audio device for this device.

        .. versionadded:: 19.0

        .. seealso::
            https://wiki.libsdl.org/SDL3/SDL_OpenAudioDevice
        """
        new_spec = _AudioSpec(
            format=_get_format(format if format is not None else self.format),
            channels=channels if channels is not None else self.channels,
            frequency=frequency if frequency is not None else self.frequency,
        )
        return self.__class__(
            ffi.gc(
                ffi.cast(
                    "SDL_AudioDeviceID", _check_int(lib.SDL_OpenAudioDevice(self.device_id, (new_spec,)), failure=0)
                ),
                lib.SDL_CloseAudioDevice,
            )
        )

    @property
    def _sample_size(self) -> int:
        """The size of a sample in bytes."""
        return self.format.itemsize * self.channels

    @property
    @deprecated("This is no longer used by the SDL3 API")
    def stopped(self) -> bool:
        """Is True if the device has failed or was closed.

        .. deprecated:: 19.0
            No longer used by the SDL3 API.
        """
        return bool(not hasattr(self, "device_id"))

    @property
    def paused(self) -> bool:
        """Get or set the device paused state."""
        return bool(lib.SDL_AudioDevicePaused(self.device_id))

    @paused.setter
    def paused(self, value: bool) -> None:
        if value:
            _check(lib.SDL_PauseAudioDevice(self.device_id))
        else:
            _check(lib.SDL_ResumeAudioDevice(self.device_id))

    def _verify_array_format(self, samples: NDArray[Any]) -> NDArray[Any]:
        if samples.dtype != self.format:
            msg = f"Expected an array of dtype {self.format}, got {samples.dtype} instead."
            raise TypeError(msg)
        return samples

    def _convert_array(self, samples_: ArrayLike) -> NDArray[np.number]:
        if isinstance(samples_, np.ndarray):
            samples_ = self._verify_array_format(samples_)
        samples: NDArray[np.number] = np.asarray(samples_, dtype=self.format)
        if len(samples.shape) < 2:  # noqa: PLR2004
            samples = samples[:, np.newaxis]
        return np.ascontiguousarray(np.broadcast_to(samples, (samples.shape[0], self.channels)), dtype=self.format)

    def convert(self, sound: ArrayLike, rate: int | None = None) -> NDArray[np.number]:
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

    def close(self) -> None:
        """Close this audio device.  Using this object after it has been closed is invalid."""
        if not hasattr(self, "device_id"):
            return
        ffi.release(self._device_id)
        del self._device_id

    @deprecated("Use contextlib.closing if you want to close this device after a context.")
    def __enter__(self) -> Self:
        """Return self and enter a managed context.

        .. deprecated:: 19.0
            Use :func:`contextlib.closing` if you want to close this device after a context.
        """
        return self

    def __exit__(
        self,
        type: type[BaseException] | None,  # noqa: A002
        value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the device when exiting the context."""
        self.close()

    @staticmethod
    def __default_callback(device: AudioDevice, stream: NDArray[Any]) -> None:
        stream[...] = device.silence

    def new_stream(
        self,
        format: DTypeLike,  # noqa: A002
        channels: int,
        frequency: int,
    ) -> AudioStream:
        """Create, bind, and return a new :any:`AudioStream` for this device.

        .. versionadded:: 19.0
        """
        new_stream = AudioStream.new(format=format, channels=channels, frequency=frequency)
        self.bind((new_stream,))
        return new_stream

    def bind(self, streams: Iterable[AudioStream], /) -> None:
        """Bind one or more :any:`AudioStream`'s to this device.

        .. seealso::
            https://wiki.libsdl.org/SDL3/SDL_BindAudioStreams
        """
        streams = list(streams)
        _check(lib.SDL_BindAudioStreams(self.device_id, [s._stream_p for s in streams], len(streams)))


@dataclass(frozen=True)
class AudioStreamCallbackData:
    """Data provided to AudioStream callbacks.

    .. versionadded:: 19.0
    """

    additional_bytes: int
    """Amount of bytes needed to fulfill the request of the caller. Can be zero."""
    additional_samples: int
    """Amount of samples needed to fulfill the request of the caller. Can be zero."""
    total_bytes: int
    """Amount of bytes requested or provided by the caller."""
    total_samples: int
    """Amount of samples requested or provided by the caller."""


_audio_stream_get_callbacks: dict[AudioStream, Callable[[AudioStream, AudioStreamCallbackData], Any]] = {}
_audio_stream_put_callbacks: dict[AudioStream, Callable[[AudioStream, AudioStreamCallbackData], Any]] = {}

_audio_stream_registry: weakref.WeakValueDictionary[int, AudioStream] = weakref.WeakValueDictionary()


class AudioStream:
    """An SDL audio stream.

    This class is commonly created with :any:`AudioDevice.new_stream` which creates a new stream bound to the device.

    ..versionadded:: 19.0
    """

    __slots__ = ("__weakref__", "_stream_p")

    _stream_p: Any

    def __new__(  # noqa: PYI034
        cls,
        stream_p: Any,  # noqa: ANN401
        /,
    ) -> AudioStream:
        """Return an AudioStream for the provided `SDL_AudioStream*` C pointer."""
        assert ffi.typeof(stream_p) is ffi.typeof("SDL_AudioStream*"), ffi.typeof(stream_p)
        stream_int = int(ffi.cast("intptr_t", stream_p))
        self = super().__new__(cls)
        self._stream_p = stream_p
        return _audio_stream_registry.setdefault(stream_int, self)

    @classmethod
    def new(  # noqa: PLR0913
        cls,
        format: DTypeLike,  # noqa: A002
        channels: int,
        frequency: int,
        out_format: DTypeLike | None = None,
        out_channels: int | None = None,
        out_frequency: int | None = None,
    ) -> Self:
        """Create a new unbound AudioStream."""
        in_spec = _AudioSpec(format=_get_format(format), channels=channels, frequency=frequency)
        out_spec = _AudioSpec(
            format=_get_format(out_format) if out_format is not None else in_spec.format,
            channels=out_channels if out_channels is not None else channels,
            frequency=out_frequency if out_frequency is not None else frequency,
        )
        return cls(ffi.gc(_check_p(lib.SDL_CreateAudioStream((in_spec,), (out_spec,))), lib.SDL_DestroyAudioStream))

    def close(self) -> None:
        """Close this AudioStream and release its resources."""
        if not hasattr(self, "_stream_p"):
            return
        self.getter_callback = None
        self.putter_callback = None
        ffi.release(self._stream_p)

    def unbind(self) -> None:
        """Unbind this stream from its currently bound device."""
        lib.SDL_UnbindAudioStream(self._stream_p)

    @property
    @contextlib.contextmanager
    def _lock(self) -> Iterator[None]:
        """Lock context for this stream."""
        try:
            lib.SDL_LockAudioStream(self._stream_p)
            yield
        finally:
            lib.SDL_UnlockAudioStream(self._stream_p)

    @property
    def _src_spec(self) -> _AudioSpec:
        c_spec = ffi.new("SDL_AudioSpec*")
        _check(lib.SDL_GetAudioStreamFormat(self._stream_p, c_spec, ffi.NULL))
        return _AudioSpec.from_c(c_spec)

    @property
    def _src_sample_size(self) -> int:
        spec = self._src_spec
        return spec._dtype.itemsize * spec.channels

    @property
    def _dst_sample_size(self) -> int:
        spec = self._dst_spec
        return spec._dtype.itemsize * spec.channels

    @property
    def _dst_spec(self) -> _AudioSpec:
        c_spec = ffi.new("SDL_AudioSpec*")
        _check(lib.SDL_GetAudioStreamFormat(self._stream_p, ffi.NULL, c_spec))
        return _AudioSpec.from_c(c_spec)

    @property
    def queued_bytes(self) -> int:
        """The current amount of bytes remaining in the audio queue."""
        return _check_int(lib.SDL_GetAudioStreamQueued(self._stream_p), failure=-1)

    @property
    def queued_samples(self) -> int:
        """The estimated amount of samples remaining in the audio queue."""
        return self.queued_bytes // self._src_sample_size

    @property
    def available_bytes(self) -> int:
        """The current amount of converted data in this audio stream."""
        return _check_int(lib.SDL_GetAudioStreamAvailable(self._stream_p), failure=-1)

    @property
    def available_samples(self) -> int:
        """The current amount of converted samples in this audio stream."""
        return self.available_bytes // self._dst_sample_size

    def queue_audio(self, samples: ArrayLike) -> None:
        """Append audio samples to the audio data queue."""
        with self._lock:
            src_spec = self._src_spec
            src_format = _dtype_from_format(src_spec.format)
            if isinstance(samples, np.ndarray) and samples.dtype != src_format:
                msg = f"Expected an array of dtype {src_format}, got {samples.dtype} instead."
                raise TypeError(msg)
            samples = np.asarray(samples, dtype=src_format)
            if len(samples.shape) < 2:  # noqa: PLR2004
                samples = samples[:, np.newaxis]
            samples = np.ascontiguousarray(
                np.broadcast_to(samples, (samples.shape[0], src_spec.channels)), dtype=src_format
            )
            buffer = ffi.from_buffer(samples)
            _check(lib.SDL_PutAudioStreamData(self._stream_p, buffer, len(buffer)))

    def flush(self) -> None:
        """Ensure all queued data is available.

        This may queue silence to the end of the stream.

        .. seealso::
            https://wiki.libsdl.org/SDL3/SDL_FlushAudioStream
        """
        _check(lib.SDL_FlushAudioStream(self._stream_p))

    def dequeue_audio(self) -> NDArray[Any]:
        """Return the converted output audio from this stream."""
        with self._lock:
            dst_spec = self._dst_spec
            out_samples = self.available_samples
            out = np.empty((out_samples, dst_spec.channels), _dtype_from_format(dst_spec.format))
            buffer = ffi.from_buffer(out)
            bytes_returned = _check_int(lib.SDL_GetAudioStreamData(self._stream_p, buffer, len(buffer)), failure=-1)
            samples_returned = bytes_returned // self._dst_sample_size
            return out[:samples_returned]

    @property
    def gain(self) -> float:
        """Get or set the audio stream gain.

        Default is 1.0 but can be set higher or zero.
        """
        return _check_float(lib.SDL_GetAudioStreamGain(self._stream_p), failure=-1.0)

    @gain.setter
    def gain(self, value: float, /) -> None:
        _check(lib.SDL_SetAudioStreamGain(self._stream_p, value))

    @property
    def frequency_ratio(self) -> float:
        """Get or set the frequency ratio, affecting the speed and pitch of the stream.

        Higher values play the audio faster.

        Default is 1.0.
        """
        return _check_float(lib.SDL_GetAudioStreamFrequencyRatio(self._stream_p), failure=-1.0)

    @frequency_ratio.setter
    def frequency_ratio(self, value: float, /) -> None:
        _check(lib.SDL_SetAudioStreamFrequencyRatio(self._stream_p, value))

    @property
    def getter_callback(self) -> Callable[[AudioStream, AudioStreamCallbackData], Any] | None:
        """Get or assign the stream get-callback for this stream.

        .. seealso::
            https://wiki.libsdl.org/SDL3/SDL_SetAudioStreamGetCallback
        """
        return _audio_stream_get_callbacks.get(self)

    @getter_callback.setter
    def getter_callback(self, callback: Callable[[AudioStream, AudioStreamCallbackData], Any] | None, /) -> None:
        if callback is None:
            _check(lib.SDL_SetAudioStreamGetCallback(self._stream_p, ffi.NULL, ffi.NULL))
            _audio_stream_get_callbacks.pop(self, None)
        else:
            _audio_stream_get_callbacks[self] = callback
            _check(
                lib.SDL_SetAudioStreamGetCallback(self._stream_p, lib._sdl_audio_stream_callback, ffi.cast("void*", 0))
            )

    @property
    def putter_callback(self) -> Callable[[AudioStream, AudioStreamCallbackData], Any] | None:
        """Get or assign the stream put-callback for this stream.

        .. seealso::
            https://wiki.libsdl.org/SDL3/SDL_SetAudioStreamPutCallback
        """
        return _audio_stream_put_callbacks.get(self)

    @putter_callback.setter
    def putter_callback(self, callback: Callable[[AudioStream, AudioStreamCallbackData], Any] | None, /) -> None:
        if callback is None:
            _check(lib.SDL_SetAudioStreamPutCallback(self._stream_p, ffi.NULL, ffi.NULL))
            _audio_stream_put_callbacks.pop(self, None)
        else:
            _audio_stream_put_callbacks[self] = callback
            _check(
                lib.SDL_SetAudioStreamPutCallback(self._stream_p, lib._sdl_audio_stream_callback, ffi.cast("void*", 1))
            )


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
        """Initialize this channel with generic attributes."""
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


@deprecated(
    "Changes in the SDL3 API have made this classes usefulness questionable."
    "\nThis class should be replaced with custom streams."
)
class BasicMixer:
    """An SDL sound mixer implemented in Python and Numpy.

    Example::

        import time

        import soundfile  # pip install soundfile
        import tcod.sdl.audio

        device = tcod.sdl.audio.get_default_playback().open()
        mixer = tcod.sdl.audio.BasicMixer(device)  # Setup BasicMixer with the default audio output
        sound, sample_rate = soundfile.read("example_sound.wav")  # Load an audio sample using SoundFile
        sound = mixer.device.convert(sound, sample_rate)  # Convert this sample to the format expected by the device
        channel = mixer.play(sound)  # Start asynchronous playback, audio is mixed on a separate Python thread
        while channel.busy:  # Wait until the sample is done playing
            time.sleep(0.001)


    .. versionadded:: 13.6

    .. versionchanged:: 19.0
        Added `frequency` and `channels` parameters.

    .. deprecated:: 19.0
        Changes in the SDL3 API have made this classes usefulness questionable.
        This class should be replaced with custom streams.
    """

    def __init__(self, device: AudioDevice, *, frequency: int | None = None, channels: int | None = None) -> None:
        """Initialize this mixer using the provided device."""
        self.channels: dict[Hashable, Channel] = {}
        self.device = device
        """The :any:`AudioDevice`"""
        self._frequency = frequency if frequency is not None else device.frequency
        self._channels = channels if channels is not None else device.channels
        self._lock = threading.RLock()
        self._stream = device.new_stream(format=np.float32, frequency=self._frequency, channels=self._channels)
        self._stream.getter_callback = self._on_stream

    def close(self) -> None:
        """Shutdown this mixer, all playing audio will be abruptly stopped."""
        self._stream.close()

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

    def _on_stream(self, audio_stream: AudioStream, data: AudioStreamCallbackData) -> None:
        """Called to fill the audio buffer."""
        if data.additional_samples <= 0:
            return
        stream: NDArray[np.float32] = np.zeros((data.additional_samples, self._channels), dtype=np.float32)
        with self._lock:
            for channel in list(self.channels.values()):
                channel._on_mix(stream)
            audio_stream.queue_audio(stream)


@ffi.def_extern()  # type: ignore[misc]
def _sdl_audio_stream_callback(userdata: Any, stream_p: Any, additional_amount: int, total_amount: int, /) -> None:  # noqa: ANN401
    """Handle audio device callbacks."""
    stream = AudioStream(stream_p)
    is_put_callback = bool(userdata)
    callback = (_audio_stream_put_callbacks if is_put_callback else _audio_stream_get_callbacks).get(stream)
    if callback is None:
        return
    sample_size = stream._dst_sample_size if is_put_callback else stream._src_sample_size
    callback(
        stream,
        AudioStreamCallbackData(
            additional_bytes=additional_amount,
            additional_samples=additional_amount // sample_size,
            total_bytes=total_amount,
            total_samples=total_amount // sample_size,
        ),
    )


def get_devices() -> dict[str, AudioDevice]:
    """Iterate over the available audio output devices.

    .. versionchanged:: 19.0
        Now returns a dictionary of :any:`AudioDevice`.
    """
    tcod.sdl.sys.init(tcod.sdl.sys.Subsystem.AUDIO)
    count = ffi.new("int[1]")
    devices_array = ffi.gc(lib.SDL_GetAudioPlaybackDevices(count), lib.SDL_free)
    return {
        device.name: device
        for device in (AudioDevice(ffi.cast("SDL_AudioDeviceID", p)) for p in devices_array[0 : count[0]])
    }


def get_capture_devices() -> dict[str, AudioDevice]:
    """Iterate over the available audio capture devices.

    .. versionchanged:: 19.0
        Now returns a dictionary of :any:`AudioDevice`.
    """
    tcod.sdl.sys.init(tcod.sdl.sys.Subsystem.AUDIO)
    count = ffi.new("int[1]")
    devices_array = ffi.gc(lib.SDL_GetAudioRecordingDevices(count), lib.SDL_free)
    return {
        device.name: device
        for device in (AudioDevice(ffi.cast("SDL_AudioDeviceID", p)) for p in devices_array[0 : count[0]])
    }


def get_default_playback() -> AudioDevice:
    """Return the default playback device.

    Example:
        playback_device = tcod.sdl.audio.get_default_playback().open()

    .. versionadded:: 19.0
    """
    tcod.sdl.sys.init(tcod.sdl.sys.Subsystem.AUDIO)
    return AudioDevice(ffi.cast("SDL_AudioDeviceID", lib.SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK))


def get_default_recording() -> AudioDevice:
    """Return the default recording device.

    Example:
        recording_device = tcod.sdl.audio.get_default_recording().open()

    .. versionadded:: 19.0
    """
    tcod.sdl.sys.init(tcod.sdl.sys.Subsystem.AUDIO)
    return AudioDevice(ffi.cast("SDL_AudioDeviceID", lib.SDL_AUDIO_DEVICE_DEFAULT_RECORDING))


@deprecated("This is no longer used", category=FutureWarning)
class AllowedChanges(enum.IntFlag):
    """Which parameters are allowed to be changed when the values given are not supported.

    .. deprecated:: 19.0
        This is no longer used.
    """

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


@deprecated(
    "This is an outdated method.\nUse 'tcod.sdl.audio.get_default_playback().open()' instead.", category=FutureWarning
)
def open(  # noqa: A001, PLR0913
    name: str | None = None,
    capture: bool = False,  # noqa: FBT001, FBT002
    *,
    frequency: int = 44100,
    format: DTypeLike = np.float32,  # noqa: A002
    channels: int = 2,
    samples: int = 0,  # noqa: ARG001
    allowed_changes: AllowedChanges = AllowedChanges.NONE,  # noqa: ARG001
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
        samples: This parameter is ignored.
        allowed_changes: This parameter is ignored.
        paused:
            If True then the device will begin in a paused state.
            It can then be unpaused by assigning False to :any:`AudioDevice.paused`.
        callback: An optional callback to use, this is deprecated.

    If a callback is given then it will be called with the `AudioDevice` and a Numpy buffer of the data stream.
    This callback will be run on a separate thread.

    .. versionchanged:: 19.0
        SDL3 returns audio devices differently, exact formatting is set with :any:`AudioDevice.new_stream` instead.

        `samples` and `allowed_changes` are ignored.

    .. deprecated:: 19.0
        This is an outdated method.
        Use :any:`AudioDevice.open` instead, for example:
        ``tcod.sdl.audio.get_default_playback().open()``
    """
    tcod.sdl.sys.init(tcod.sdl.sys.Subsystem.AUDIO)
    if name is None:
        device = get_default_playback() if not capture else get_default_recording()
    else:
        device = (get_devices() if not capture else get_capture_devices())[name]
    assert device.is_capture is capture
    device = device.open(frequency=frequency, format=format, channels=channels)
    device.paused = paused

    if callback is not None and callback is not True:
        stream = device.new_stream(format=format, channels=channels, frequency=frequency)

        def _get_callback(stream: AudioStream, data: AudioStreamCallbackData) -> None:
            if data.additional_samples <= 0:
                return
            buffer = np.full(
                (data.additional_samples, channels), fill_value=_silence_value_for_format(format), dtype=format
            )
            callback(device, buffer)
            stream.queue_audio(buffer)

        stream.getter_callback = _get_callback

    return device
