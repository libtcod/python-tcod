
import functools
import warnings

def deprecate(message, category=DeprecationWarning, stacklevel=0):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kargs):
            warnings.warn(message, category, stacklevel=stacklevel + 2)
            return func(*args, **kargs)
        return wrapper
    return decorator

def verify_order(order):
    order = order.upper()
    if order != 'C' and order != 'F':
        raise TypeError("order must be 'C' or 'F', not %r" % (order,))
    return order

def handle_order(shape, order):
    order = verify_order(order)
    if order == 'C':
        return shape
    else:
        return tuple(reversed(shape))
