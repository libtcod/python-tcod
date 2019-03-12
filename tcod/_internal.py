import functools
from typing import Any, Callable, TypeVar, cast
import warnings

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


def verify_order(order: str) -> str:
    order = order.upper()
    if order not in ("C", "F"):
        raise TypeError("order must be 'C' or 'F', not %r" % (order,))
    return order


def handle_order(shape: Any, order: str) -> Any:
    order = verify_order(order)
    if order == "C":
        return shape
    else:
        return tuple(reversed(shape))
