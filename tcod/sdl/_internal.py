"""tcod.sdl private functions."""

from __future__ import annotations

import logging
import sys as _sys
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Callable, NoReturn, TypeVar

from tcod.cffi import ffi, lib

T = TypeVar("T")

logger = logging.getLogger("tcod.sdl")

_LOG_PRIORITY = {
    1: logging.DEBUG,  # SDL_LOG_PRIORITY_VERBOSE
    2: logging.DEBUG,  # SDL_LOG_PRIORITY_DEBUG
    3: logging.INFO,  # SDL_LOG_PRIORITY_INFO
    4: logging.WARNING,  # SDL_LOG_PRIORITY_WARN
    5: logging.ERROR,  # SDL_LOG_PRIORITY_ERROR
    6: logging.CRITICAL,  # SDL_LOG_PRIORITY_CRITICAL
}

_LOG_CATEGORY = {
    int(lib.SDL_LOG_CATEGORY_APPLICATION): "APPLICATION",
    int(lib.SDL_LOG_CATEGORY_ERROR): "ERROR",
    int(lib.SDL_LOG_CATEGORY_ASSERT): "ASSERT",
    int(lib.SDL_LOG_CATEGORY_SYSTEM): "SYSTEM",
    int(lib.SDL_LOG_CATEGORY_AUDIO): "AUDIO",
    int(lib.SDL_LOG_CATEGORY_VIDEO): "VIDEO",
    int(lib.SDL_LOG_CATEGORY_RENDER): "RENDER",
    int(lib.SDL_LOG_CATEGORY_INPUT): "INPUT",
    int(lib.SDL_LOG_CATEGORY_TEST): "TEST",
    int(lib.SDL_LOG_CATEGORY_CUSTOM): "",
}


@dataclass
class _UnraisableHookArgs:
    exc_type: type[BaseException]
    exc_value: BaseException | None
    exc_traceback: TracebackType | None
    err_msg: str | None
    object: object


class _ProtectedContext:
    def __init__(self, obj: object = None) -> None:
        self.obj = obj

    def __enter__(self) -> None:
        pass

    def __exit__(
        self, exc_type: type[BaseException] | None, value: BaseException | None, traceback: TracebackType | None
    ) -> bool:
        if exc_type is None:
            return False
        _sys.unraisablehook(_UnraisableHookArgs(exc_type, value, traceback, None, self.obj))
        return True


@ffi.def_extern()  # type: ignore
def _sdl_log_output_function(_userdata: None, category: int, priority: int, message_p: Any) -> None:  # noqa: ANN401
    """Pass logs sent by SDL to Python's logging system."""
    message = str(ffi.string(message_p), encoding="utf-8")
    logger.log(_LOG_PRIORITY.get(priority, 0), "%s:%s", _LOG_CATEGORY.get(category, ""), message)


def _get_error() -> str:
    """Return a message from SDL_GetError as a Unicode string."""
    return str(ffi.string(lib.SDL_GetError()), encoding="utf-8")


def _check(result: int) -> int:
    """Check if an SDL function returned without errors, and raise an exception if it did."""
    if result < 0:
        raise RuntimeError(_get_error())
    return result


def _check_p(result: Any) -> Any:  # noqa: ANN401
    """Check if an SDL function returned NULL, and raise an exception if it did."""
    if not result:
        raise RuntimeError(_get_error())
    return result


def _compiled_version() -> tuple[int, int, int]:
    return int(lib.SDL_MAJOR_VERSION), int(lib.SDL_MINOR_VERSION), int(lib.SDL_PATCHLEVEL)


def _linked_version() -> tuple[int, int, int]:
    sdl_version = ffi.new("SDL_version*")
    lib.SDL_GetVersion(sdl_version)
    return int(sdl_version.major), int(sdl_version.minor), int(sdl_version.patch)


def _version_at_least(required: tuple[int, int, int]) -> None:
    """Raise an error if the compiled version is less than required.  Used to guard recently defined SDL functions."""
    if required <= _compiled_version():
        return
    msg = f"This feature requires SDL version {required}, but tcod was compiled with version {_compiled_version()}"
    raise RuntimeError(msg)


def _required_version(required: tuple[int, int, int]) -> Callable[[T], T]:
    if not lib:  # Read the docs mock object.
        return lambda x: x
    if required <= _compiled_version():
        return lambda x: x

    def replacement(*_args: object, **_kwargs: object) -> NoReturn:
        msg = f"This feature requires SDL version {required}, but tcod was compiled with version {_compiled_version()}"
        raise RuntimeError(msg)

    return lambda x: replacement  # type: ignore[return-value]


lib.SDL_LogSetOutputFunction(lib._sdl_log_output_function, ffi.NULL)
if __debug__:
    lib.SDL_LogSetAllPriority(lib.SDL_LOG_PRIORITY_VERBOSE)
