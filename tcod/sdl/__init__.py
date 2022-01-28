from __future__ import annotations

import logging
from typing import Any

from tcod.loader import ffi, lib

logger = logging.getLogger(__name__)

_LOG_PRIORITY = {
    int(lib.SDL_LOG_PRIORITY_VERBOSE): logging.DEBUG,
    int(lib.SDL_LOG_PRIORITY_DEBUG): logging.DEBUG,
    int(lib.SDL_LOG_PRIORITY_INFO): logging.INFO,
    int(lib.SDL_LOG_PRIORITY_WARN): logging.WARNING,
    int(lib.SDL_LOG_PRIORITY_ERROR): logging.ERROR,
    int(lib.SDL_LOG_PRIORITY_CRITICAL): logging.CRITICAL,
}


@ffi.def_extern()  # type: ignore
def _sdl_log_output_function(_userdata: Any, category: int, priority: int, message: Any) -> None:
    """Pass logs sent by SDL to Python's logging system."""
    logger.log(_LOG_PRIORITY.get(priority, 0), "%i:%s", category, ffi.string(message).decode("utf-8"))


lib.SDL_LogSetOutputFunction(lib._sdl_log_output_function, ffi.NULL)
