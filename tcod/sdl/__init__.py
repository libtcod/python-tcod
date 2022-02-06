from __future__ import annotations

import logging
from typing import Any, Callable, Tuple, TypeVar

from tcod.loader import ffi, lib

T = TypeVar("T")

logger = logging.getLogger(__name__)

_LOG_PRIORITY = {
    1: logging.DEBUG,  # SDL_LOG_PRIORITY_VERBOSE
    2: logging.DEBUG,  # SDL_LOG_PRIORITY_DEBUG
    3: logging.INFO,  # SDL_LOG_PRIORITY_INFO
    4: logging.WARNING,  # SDL_LOG_PRIORITY_WARN
    5: logging.ERROR,  # SDL_LOG_PRIORITY_ERROR
    6: logging.CRITICAL,  # SDL_LOG_PRIORITY_CRITICAL
}


@ffi.def_extern()  # type: ignore
def _sdl_log_output_function(_userdata: Any, category: int, priority: int, message: Any) -> None:
    """Pass logs sent by SDL to Python's logging system."""
    logger.log(_LOG_PRIORITY.get(priority, 0), "%i:%s", category, ffi.string(message).decode("utf-8"))


def _get_error() -> str:
    """Return a message from SDL_GetError as a Unicode string."""
    return str(ffi.string(lib.SDL_GetError()), encoding="utf-8")


def _check(result: int) -> int:
    """Check if an SDL function returned without errors, and raise an exception if it did."""
    if result < 0:
        raise RuntimeError(_get_error())
    return result


def _check_p(result: Any) -> Any:
    """Check if an SDL function returned NULL, and raise an exception if it did."""
    if not result:
        raise RuntimeError(_get_error())
    return result


if lib._sdl_log_output_function:
    lib.SDL_LogSetOutputFunction(lib._sdl_log_output_function, ffi.NULL)


def _compiled_version() -> Tuple[int, int, int]:
    return int(lib.SDL_MAJOR_VERSION), int(lib.SDL_MINOR_VERSION), int(lib.SDL_PATCHLEVEL)


def _linked_version() -> Tuple[int, int, int]:
    sdl_version = ffi.new("SDL_version*")
    lib.SDL_GetVersion(sdl_version)
    return int(sdl_version.major), int(sdl_version.minor), int(sdl_version.patch)


def _version_at_least(required: Tuple[int, int, int]) -> None:
    """Raise an error if the compiled version is less than required.  Used to guard recentally defined SDL functions."""
    if required <= _compiled_version():
        return
    raise RuntimeError(
        f"This feature requires SDL version {required}, but tcod was compiled with version {_compiled_version()}"
    )


def _required_version(required: Tuple[int, int, int]) -> Callable[[T], T]:
    if not lib:  # Read the docs mock object.
        return lambda x: x
    if required <= _compiled_version():
        return lambda x: x

    def replacement(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError(
            f"This feature requires SDL version {required}, but tcod was compiled with version {_compiled_version()}"
        )

    return lambda x: replacement  # type: ignore[return-value]
