"""This module internal helper functions used by the rest of the library.
"""
import functools
import warnings
from typing import Any, AnyStr, Callable, TypeVar, Union, cast

import numpy as np
from typing_extensions import Literal, NoReturn

from tcod.loader import ffi, lib

FuncType = Callable[..., Any]
F = TypeVar("F", bound=FuncType)


def deprecate(
    message: str, category: Any = DeprecationWarning, stacklevel: int = 0
) -> Callable[[F], F]:
    """Return a decorator which adds a warning to functions."""

    def decorator(func: F) -> F:
        if not __debug__:
            return func

        @functools.wraps(func)
        def wrapper(*args, **kargs):  # type: ignore
            warnings.warn(message, category, stacklevel=stacklevel + 2)
            return func(*args, **kargs)

        return cast(F, wrapper)

    return decorator


def pending_deprecate(
    message: str = "This function may be deprecated in the future."
    " Consider raising an issue on GitHub if you need this feature.",
    category: Any = PendingDeprecationWarning,
    stacklevel: int = 0,
) -> Callable[[F], F]:
    """Like deprecate, but the default parameters are filled out for a generic
    pending deprecation warning."""
    return deprecate(message, category, stacklevel)


def verify_order(
    order: Union[Literal["C"], Literal["F"]]
) -> Union[Literal["C"], Literal["F"]]:
    order = order.upper()  # type: ignore
    if order not in ("C", "F"):
        raise TypeError("order must be 'C' or 'F', not %r" % (order,))
    return order


def _raise_tcod_error() -> NoReturn:
    """Raise an error from libtcod, this function assumes an error exists."""
    raise RuntimeError(ffi.string(lib.TCOD_get_error()).decode("utf-8"))


def _check(error: int) -> int:
    """Detect and convert a libtcod error code into an exception."""
    if error < 0:
        _raise_tcod_error()
    return error


def _check_p(pointer: Any) -> Any:
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


def _unpack_char_p(char_p: Any) -> str:
    if char_p == ffi.NULL:
        return ""
    return ffi.string(char_p).decode()  # type: ignore


def _int(int_or_str: Any) -> int:
    "return an integer where a single character string may be expected"
    if isinstance(int_or_str, str):
        return ord(int_or_str)
    if isinstance(int_or_str, bytes):
        return int_or_str[0]
    return int(int_or_str)  # check for __count__


def _bytes(string: AnyStr) -> bytes:
    if isinstance(string, str):
        return string.encode("utf-8")
    return string


def _unicode(string: AnyStr, stacklevel: int = 2) -> str:
    if isinstance(string, bytes):
        warnings.warn(
            (
                "Passing byte strings as parameters to Unicode functions is "
                "deprecated."
            ),
            DeprecationWarning,
            stacklevel=stacklevel + 1,
        )
        return string.decode("latin-1")
    return string


def _fmt(string: str, stacklevel: int = 2) -> bytes:
    if isinstance(string, bytes):
        warnings.warn(
            (
                "Passing byte strings as parameters to Unicode functions is "
                "deprecated."
            ),
            DeprecationWarning,
            stacklevel=stacklevel + 1,
        )
        string = string.decode("latin-1")
    return string.encode("utf-8").replace(b"%", b"%%")


class _PropagateException:
    """Context manager designed to propagate exceptions outside of a cffi
    callback context.  Normally cffi suppresses the exception.

    When propagate is called this class will hold onto the error until the
    control flow leaves the context, then the error will be raised.

    with _PropagateException as propagate:
    # give propagate as onerror parameter for ffi.def_extern
    """

    def __init__(self) -> None:
        # (exception, exc_value, traceback)
        self.exc_info = None  # type: Any

    def propagate(self, *exc_info: Any) -> None:
        """Set an exception to be raised once this context exits.

        If multiple errors are caught, only keep the first exception raised.
        """
        if not self.exc_info:
            self.exc_info = exc_info

    def __enter__(self) -> Callable[[Any], None]:
        """Once in context, only the propagate call is needed to use this
        class effectively.
        """
        return self.propagate

    def __exit__(self, type: Any, value: Any, traceback: Any) -> None:
        """If we're holding on to an exception, raise it now.

        Prefers our held exception over any current raising error.

        self.exc_info is reset now in case of nested manager shenanigans.
        """
        if self.exc_info:
            type, value, traceback = self.exc_info
            self.exc_info = None
        if type:
            # Python 2/3 compatible throw
            exception = type(value)
            exception.__traceback__ = traceback
            raise exception


class _CDataWrapper(object):
    def __init__(self, *args: Any, **kargs: Any):
        self.cdata = self._get_cdata_from_args(*args, **kargs)
        if self.cdata is None:
            self.cdata = ffi.NULL
        super(_CDataWrapper, self).__init__()

    @staticmethod
    def _get_cdata_from_args(*args: Any, **kargs: Any) -> Any:
        if len(args) == 1 and isinstance(args[0], ffi.CData) and not kargs:
            return args[0]
        else:
            return None

    def __hash__(self) -> int:
        return hash(self.cdata)

    def __eq__(self, other: Any) -> Any:
        try:
            return self.cdata == other.cdata
        except AttributeError:
            return NotImplemented

    def __getattr__(self, attr: str) -> Any:
        if "cdata" in self.__dict__:
            return getattr(self.__dict__["cdata"], attr)
        raise AttributeError(attr)

    def __setattr__(self, attr: str, value: Any) -> None:
        if hasattr(self, "cdata") and hasattr(self.cdata, attr):
            setattr(self.cdata, attr, value)
        else:
            super(_CDataWrapper, self).__setattr__(attr, value)


def _console(console: Any) -> Any:
    """Return a cffi console."""
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


class TempImage(object):
    """An Image-like container for NumPy arrays."""

    def __init__(self, array: Any):
        self._array = np.ascontiguousarray(array, dtype=np.uint8)
        height, width, depth = self._array.shape
        if depth != 3:
            raise TypeError(
                "Array must have RGB channels.  Shape is: %r"
                % (self._array.shape,)
            )
        self._buffer = ffi.from_buffer("TCOD_color_t[]", self._array)
        self._mipmaps = ffi.new(
            "struct TCOD_mipmap_*",
            {
                "width": width,
                "height": height,
                "fwidth": width,
                "fheight": height,
                "buf": self._buffer,
                "dirty": True,
            },
        )
        self.image_c = ffi.new(
            "TCOD_Image*",
            {
                "nb_mipmaps": 1,
                "mipmaps": self._mipmaps,
                "has_key_color": False,
            },
        )


def _asimage(image: Any) -> TempImage:
    """Convert this input into an Image-like object."""
    if hasattr(image, "image_c"):
        return image  # type: ignore
    return TempImage(image)
