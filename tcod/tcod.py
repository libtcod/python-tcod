"""This module focuses on improvements to the Python libtcod API.
"""
from typing import Any, AnyStr, Callable
import warnings

from tcod.libtcod import ffi


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
    """ context manager designed to propagate exceptions outside of a cffi
    callback context.  normally cffi suppresses the exception

    when propagate is called this class will hold onto the error until the
    control flow leaves the context, then the error will be raised

    with _PropagateException as propagate:
    # give propagate as onerror parameter for ffi.def_extern
    """

    def __init__(self) -> None:
        # (exception, exc_value, traceback)
        self.exc_info = None  # type: Any

    def propagate(self, *exc_info: Any) -> None:
        """ set an exception to be raised once this context exits

        if multiple errors are caught, only keep the first exception raised
        """
        if not self.exc_info:
            self.exc_info = exc_info

    def __enter__(self) -> Callable[[Any], None]:
        """ once in context, only the propagate call is needed to use this
        class effectively
        """
        return self.propagate

    def __exit__(self, type: Any, value: Any, traceback: Any) -> None:
        """ if we're holding on to an exception, raise it now

        prefers our held exception over any current raising error

        self.exc_info is reset now in case of nested manager shenanigans
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
