"""Test tcod.sdl.audio module."""

import contextlib
import sys
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

import tcod.sdl.audio


def device_works(device: Callable[[], tcod.sdl.audio.AudioDevice]) -> bool:
    try:
        device().open().close()
    except RuntimeError:
        return False
    return True


needs_audio_device = pytest.mark.xfail(
    not device_works(tcod.sdl.audio.get_default_playback), reason="This test requires an audio device"
)
needs_audio_capture = pytest.mark.xfail(
    not device_works(tcod.sdl.audio.get_default_recording), reason="This test requires an audio capture device"
)


def test_devices() -> None:
    list(tcod.sdl.audio.get_devices())
    list(tcod.sdl.audio.get_capture_devices())


@needs_audio_device
def test_audio_device() -> None:
    with tcod.sdl.audio.open(frequency=44100, format=np.float32, channels=2, paused=True) as device:
        assert not device.stopped
        device.convert(np.zeros(4, dtype=np.float32), 22050)
        assert device.convert(np.zeros((4, 4), dtype=np.float32)).shape[1] == device.channels
        device.convert(np.zeros(4, dtype=np.int8)).shape[0]
        assert device.paused is True
        device.paused = False
        assert device.paused is False
        device.paused = True
        with contextlib.closing(tcod.sdl.audio.BasicMixer(device, frequency=44100, channels=2)) as mixer:
            assert mixer.play(np.zeros(4, np.float32)).busy
            mixer.play(np.zeros(0, np.float32))
            mixer.play(np.full(1, 0.01, np.float32), on_end=lambda _: None)
            mixer.play(np.full(1, 0.01, np.float32), loops=2, on_end=lambda _: None)
            mixer.play(np.full(4, 0.01, np.float32), loops=2).stop()
            mixer.play(np.full(100000, 0.01, np.float32))
            with pytest.raises(TypeError, match=r".*must be dtype=float32.*was dtype=int32"):
                mixer.play(np.zeros(1, np.int32))
            time.sleep(0.001)
            mixer.stop()


@needs_audio_capture
def test_audio_capture() -> None:
    with contextlib.closing(tcod.sdl.audio.get_default_recording().open()) as device:
        device.new_stream(np.float32, 1, 11025).dequeue_audio()


@needs_audio_device
def test_audio_device_repr() -> None:
    with contextlib.closing(tcod.sdl.audio.get_default_playback().open()) as device:
        assert not device.stopped
        assert "paused=False" in repr(device)


def test_convert_bad_shape() -> None:
    with pytest.raises(TypeError):
        tcod.sdl.audio.convert_audio(
            np.zeros((1, 1, 1), np.float32), 8000, out_rate=8000, out_format=np.float32, out_channels=1
        )


def test_convert_bad_type() -> None:
    with pytest.raises(TypeError, match=r".*bool"):
        tcod.sdl.audio.convert_audio(np.zeros(8, bool), 8000, out_rate=8000, out_format=np.float32, out_channels=1)
    with pytest.raises(RuntimeError, match=r"Parameter 'src_spec->format' is invalid"):
        tcod.sdl.audio.convert_audio(np.zeros(8, np.int64), 8000, out_rate=8000, out_format=np.float32, out_channels=1)


def test_convert_float64() -> None:
    np.testing.assert_array_equal(
        tcod.sdl.audio.convert_audio(
            np.ones(8, np.float64), 8000, out_rate=8000, out_format=np.float32, out_channels=1
        ),
        np.ones((8, 1), np.float32),
    )


@needs_audio_device
def test_audio_callback() -> None:
    class CheckCalled:
        was_called: bool = False

        def __call__(self, device: tcod.sdl.audio.AudioDevice, stream: NDArray[Any]) -> None:
            self.was_called = True
            assert isinstance(device, tcod.sdl.audio.AudioDevice)
            assert isinstance(stream, np.ndarray)
            assert len(stream.shape) == 2  # noqa: PLR2004

    check_called = CheckCalled()
    with tcod.sdl.audio.open(callback=check_called, paused=False) as device:
        assert not device.stopped
        while not check_called.was_called:
            time.sleep(0.001)


@pytest.mark.skipif(sys.version_info < (3, 8), reason="Needs sys.unraisablehook support")
@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
@needs_audio_device
def test_audio_callback_unraisable() -> None:
    """Test unraisable error in audio callback.

    This can't be checked with pytest very well, so at least make sure this doesn't crash.
    """

    class CheckCalled:
        was_called: bool = False

        def __call__(self, device: tcod.sdl.audio.AudioDevice, stream: NDArray[Any]) -> None:
            self.was_called = True
            raise Exception("Test unraisable error")  # noqa

    check_called = CheckCalled()
    with tcod.sdl.audio.open(callback=check_called, paused=False) as device:
        assert not device.stopped
        while not check_called.was_called:
            time.sleep(0.001)
