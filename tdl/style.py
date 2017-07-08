"""
    Used internally to handle style changes without breaking backwards
    compatibility.
"""

import warnings as _warnings
import functools as _functools

def backport(func):
    """
        Backport a function name into an old style for compatibility.

        The docstring is updated to reflect that the new function returned is
        deprecated and that the other function is preferred.
        A DeprecationWarning is also raised for using this function.

        If the script is run with an optimization flag then the real function
        will be returned without being wrapped.
    """
    if not __debug__:
        return func

    @_functools.wraps(func)
    def deprecated_function(*args, **kargs):
        _warnings.warn('This function name is deprecated',
                       DeprecationWarning, 2)
        return func(*args, **kargs)
    deprecated_function.__doc__ = None
    return deprecated_function

__all__ = []
