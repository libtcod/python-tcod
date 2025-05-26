"""Internal helper functions used by the rest of the library."""

from __future__ import annotations

import locale
import sys
import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, AnyStr, Literal, NoReturn, SupportsInt, TypeVar

from typing_extensions import LiteralString, deprecated

from tcod.cffi import ffi, lib

if TYPE_CHECKING:
    from pathlib import Path
    from types import TracebackType


FuncType = Callable[..., Any]
F = TypeVar("F", bound=FuncType)
T = TypeVar("T")


def _deprecate_passthrough(
    message: str,  # noqa: ARG001
    /,
    *,
    category: type[Warning] = DeprecationWarning,  # noqa: ARG001
    stacklevel: int = 0,  # noqa: ARG001
) -> Callable[[F], F]:
    """Return a decorator which skips wrapping a warning onto functions. This is used for non-debug runs."""

    def decorator(func: F) -> F:
        return func

    return decorator


deprecate = deprecated if __debug__ or TYPE_CHECKING else _deprecate_passthrough


def pending_deprecate(
    message: LiteralString = "This function may be deprecated in the future."
    " Consider raising an issue on GitHub if you need this feature.",
    category: type[Warning] = PendingDeprecationWarning,
    stacklevel: int = 0,
) -> Callable[[F], F]:
    """Like deprecate, but the default parameters are filled out for a generic pending deprecation warning."""
    return deprecate(message, category=category, stacklevel=stacklevel)


def verify_order(order: Literal["C", "F"]) -> Literal["C", "F"]:
    """Verify and return a Numpy order string."""
    order = order.upper()  # type: ignore[assignment]
    if order not in ("C", "F"):
        msg = f"order must be 'C' or 'F', not {order!r}"
        raise TypeError(msg)
    return order


def _raise_tcod_error() -> NoReturn:
    """Raise an error from libtcod, this function assumes an error exists."""
    raise RuntimeError(ffi.string(lib.TCOD_get_error()).decode("utf-8"))


def _check(error: int) -> int:
    """Detect and convert a libtcod error code into an exception."""
    if error < 0:
        _raise_tcod_error()
    return error


def _check_p(pointer: T) -> T:
    """Treats NULL pointers as errors and raises a libtcod exception."""
    if not pointer:
        _raise_tcod_error()
    return pointer


def _check_warn(error: int, stacklevel: int = 2) -> int:
    """Like _check, but raises a warning on positive error codes."""
    if _check(error) > 0:
        warnings.warn(
            ffi.string(lib.TCOD_get_error()).decode(),
            RuntimeWarning,
            stacklevel=stacklevel + 1,
        )
    return error


def _unpack_char_p(char_p: Any) -> str:  # noqa: ANN401
    if char_p == ffi.NULL:
        return ""
    return str(ffi.string(char_p), encoding="utf-8")


def _int(int_or_str: SupportsInt | str | bytes) -> int:
    """Return an integer where a single character string may be expected."""
    if isinstance(int_or_str, str):
        return ord(int_or_str)
    if isinstance(int_or_str, bytes):
        return int_or_str[0]
    return int(int_or_str)


def _bytes(string: AnyStr) -> bytes:
    if isinstance(string, str):
        return string.encode("utf-8")
    return string


def _unicode(string: AnyStr, stacklevel: int = 2) -> str:
    if isinstance(string, bytes):
        warnings.warn(
            "Passing byte strings as parameters to Unicode functions is deprecated.",
            FutureWarning,
            stacklevel=stacklevel + 1,
        )
        return string.decode("latin-1")
    return str(string)


def _fmt(string: str, stacklevel: int = 2) -> bytes:
    if isinstance(string, bytes):
        warnings.warn(
            "Passing byte strings as parameters to Unicode functions is deprecated.",
            FutureWarning,
            stacklevel=stacklevel + 1,
        )
        string = string.decode("latin-1")
    return string.encode("utf-8").replace(b"%", b"%%")


def _path_encode(path: Path) -> bytes:
    """Return a bytes file path for the current locale when on Windows, uses fsdecode for other platforms."""
    if sys.platform != "win32":
        return bytes(path)  # Sane and expected behavior for converting Path into bytes
    try:
        return str(path).encode(locale.getlocale()[1] or "utf-8")  # Stay classy, Windows
    except UnicodeEncodeError as exc:
        if sys.version_info >= (3, 11):
            exc.add_note("""Consider calling 'locale.setlocale(locale.LC_CTYPES, ".UTF8")' to support Unicode paths.""")
        raise


class _PropagateException:
    """Context manager designed to propagate exceptions outside of a cffi callback context.

    Normally cffi suppresses the exception.

    When propagate is called this class will hold onto the error until the
    control flow leaves the context, then the error will be raised.

    with _PropagateException as propagate:
    # give propagate as onerror parameter for ffi.def_extern
    """

    def __init__(self) -> None:
        self.caught: BaseException | None = None

    def propagate(self, *exc_info: Any) -> None:  # noqa: ANN401
        """Set an exception to be raised once this context exits.

        If multiple errors are caught, only keep the first exception raised.
        """
        if self.caught is None:
            self.caught = exc_info[1]

    def __enter__(self) -> Callable[[Any], None]:
        """Once in context, only the propagate call is needed to use this class effectively."""
        return self.propagate

    def __exit__(
        self, _type: type[BaseException] | None, value: BaseException | None, traceback: TracebackType | None
    ) -> None:
        """If we're holding on to an exception, raise it now.

        self.caught is reset now in case of nested manager shenanigans.
        """
        to_raise, self.caught = self.caught, None
        if to_raise is not None:
            raise to_raise from value


class _CDataWrapper:
    """A generally deprecated CData wrapper class used by libtcodpy."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        self.cdata = self._get_cdata_from_args(*args, **kwargs)
        if self.cdata is None:
            self.cdata = ffi.NULL
        super().__init__()

    @staticmethod
    def _get_cdata_from_args(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        if len(args) == 1 and isinstance(args[0], ffi.CData) and not kwargs:
            return args[0]
        return None

    def __hash__(self) -> int:
        return hash(self.cdata)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _CDataWrapper):
            return NotImplemented
        return bool(self.cdata == other.cdata)

    def __getattr__(self, attr: str) -> Any:  # noqa: ANN401
        if "cdata" in self.__dict__:
            return getattr(self.__dict__["cdata"], attr)
        raise AttributeError(attr)

    def __setattr__(self, attr: str, value: Any) -> None:  # noqa: ANN401
        if hasattr(self, "cdata") and hasattr(self.cdata, attr):
            setattr(self.cdata, attr, value)
        else:
            super().__setattr__(attr, value)


def _console(console: Any) -> Any:  # noqa: ANN401
    """Return a cffi console pointer."""
    try:
        return console.console_c
    except AttributeError:
        warnings.warn(
            (
                "Falsy console parameters are deprecated, "
                "always use the root console instance returned by "
                "console_init_root."
            ),
            DeprecationWarning,
            stacklevel=3,
        )
        return ffi.NULL
