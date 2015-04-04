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
    """
    @_functools.wraps(func)
    def deprecated_function(*args, **kargs):
        _warnings.warn('This finction name is deprecated',
                       DeprecationWarning, 2)
        return func(*args, **kargs)
    deprecated_function.__doc__ = """
    Deprecated version of the function L{%s}, call that instead of this.
    """ % (func.__name__)
    return deprecated_function
    
__all__ = []
