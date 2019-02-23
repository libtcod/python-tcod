import functools
from typing import Any, Callable, TypeVar, cast
import warnings

FuncType = Callable[..., Any]
F = TypeVar("F", bound=FuncType)


def deprecate(
    message: str, category: Any = DeprecationWarning, stacklevel: int = 0
) -> Callable[[F], F]:
    def decorator(func: F) -> F:
        if not __debug__:
            return func

        @functools.wraps(func)
        def wrapper(*args, **kargs):  # type: ignore
            warnings.warn(message, category, stacklevel=stacklevel + 2)
            return func(*args, **kargs)

        return cast(F, wrapper)

    return decorator


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
