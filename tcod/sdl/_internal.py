"""tcod.sdl private functions."""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NoReturn, Protocol, TypeVar, overload, runtime_checkable

from typing_extensions import Self

from tcod.cffi import ffi, lib

if TYPE_CHECKING:
    from types import TracebackType

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
        sys.unraisablehook(_UnraisableHookArgs(exc_type, value, traceback, None, self.obj))
        return True


@runtime_checkable
class PropertyPointer(Protocol):
    """Methods for classes which support pointers being set to properties."""

    @classmethod
    def _from_property_pointer(cls, raw_cffi_pointer: Any, /) -> Self:  # noqa: ANN401
        """Convert a raw pointer to this class."""
        ...

    def _as_property_pointer(self) -> Any:  # noqa: ANN401
        """Return a CFFI pointer for this object."""
        ...


class Properties:
    """SDL properties interface."""

    def __init__(self, p: Any | None = None) -> None:  # noqa: ANN401
        """Create new properties or use an existing pointer."""
        if p is None:
            self.p = ffi.gc(
                ffi.cast("SDL_PropertiesID", _check_int(lib.SDL_CreateProperties(), failure=0)),
                lib.SDL_DestroyProperties,
            )
        else:
            self.p = p

    @overload
    def __getitem__(self, key: tuple[str, type[bool]], /) -> bool: ...
    @overload
    def __getitem__(self, key: tuple[str, type[int]], /) -> int: ...
    @overload
    def __getitem__(self, key: tuple[str, type[float]], /) -> float: ...
    @overload
    def __getitem__(self, key: tuple[str, type[str]], /) -> str: ...

    def __getitem__(self, key: tuple[str, type[Any]], /) -> Any:
        """Get a typed value from this property."""
        key_, type_ = key
        name = key_.encode("utf-8")
        match lib.SDL_GetPropertyType(self.p, name):
            case lib.SDL_PROPERTY_TYPE_STRING:
                assert type_ is str
                return str(ffi.string(lib.SDL_GetStringProperty(self.p, name, ffi.NULL)), encoding="utf-8")
            case lib.SDL_PROPERTY_TYPE_NUMBER:
                assert type_ is int
                return int(lib.SDL_GetNumberProperty(self.p, name, 0))
            case lib.SDL_PROPERTY_TYPE_FLOAT:
                assert type_ is float
                return float(lib.SDL_GetFloatProperty(self.p, name, 0.0))
            case lib.SDL_PROPERTY_TYPE_BOOLEAN:
                assert type_ is bool
                return bool(lib.SDL_GetBooleanProperty(self.p, name, False))  # noqa: FBT003
            case lib.SDL_PROPERTY_TYPE_POINTER:
                assert isinstance(type_, PropertyPointer)
                return type_._from_property_pointer(lib.SDL_GetPointerProperty(self.p, name, ffi.NULL))
            case lib.SDL_PROPERTY_TYPE_INVALID:
                raise KeyError("Invalid type.")  # noqa: EM101, TRY003
            case _:
                raise AssertionError

    def __setitem__(self, key: tuple[str, type[T]], value: T, /) -> None:
        """Assign a property."""
        key_, type_ = key
        name = key_.encode("utf-8")
        if type_ is str:
            assert isinstance(value, str)
            lib.SDL_SetStringProperty(self.p, name, value.encode("utf-8"))
        elif type_ is int:
            assert isinstance(value, int)
            lib.SDL_SetNumberProperty(self.p, name, value)
        elif type_ is float:
            assert isinstance(value, (int, float))
            lib.SDL_SetFloatProperty(self.p, name, value)
        elif type_ is bool:
            lib.SDL_SetFloatProperty(self.p, name, bool(value))
        else:
            assert isinstance(type_, PropertyPointer)
            assert isinstance(value, PropertyPointer)
            lib.SDL_SetPointerProperty(self.p, name, value._as_property_pointer())


@ffi.def_extern()  # type: ignore[misc]
def _sdl_log_output_function(_userdata: None, category: int, priority: int, message_p: Any) -> None:  # noqa: ANN401
    """Pass logs sent by SDL to Python's logging system."""
    message = str(ffi.string(message_p), encoding="utf-8")
    logger.log(_LOG_PRIORITY.get(priority, 0), "%s:%s", _LOG_CATEGORY.get(category, ""), message)


def _get_error() -> str:
    """Return a message from SDL_GetError as a Unicode string."""
    return str(ffi.string(lib.SDL_GetError()), encoding="utf-8")


def _check(result: bool, /) -> bool:
    """Check if an SDL function returned without errors, and raise an exception if it did."""
    if not result:
        raise RuntimeError(_get_error())
    return result


def _check_int(result: int, /, failure: int) -> int:
    """Check if an SDL function returned without errors, and raise an exception if it did."""
    if result == failure:
        raise RuntimeError(_get_error())
    return result


def _check_float(result: float, /, failure: float) -> float:
    """Check if an SDL function returned without errors, and raise an exception if it did."""
    if result == failure:
        raise RuntimeError(_get_error())
    return result


def _check_p(result: Any) -> Any:  # noqa: ANN401
    """Check if an SDL function returned NULL, and raise an exception if it did."""
    if not result:
        raise RuntimeError(_get_error())
    return result


def _compiled_version() -> tuple[int, int, int]:
    return int(lib.SDL_MAJOR_VERSION), int(lib.SDL_MINOR_VERSION), int(lib.SDL_MICRO_VERSION)


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

    return lambda _: replacement  # type: ignore[return-value]


lib.SDL_SetLogOutputFunction(lib._sdl_log_output_function, ffi.NULL)
if __debug__:
    lib.SDL_SetLogPriorities(lib.SDL_LOG_PRIORITY_VERBOSE)
