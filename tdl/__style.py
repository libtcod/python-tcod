
import warnings as _warnings
import functools as _functools

def backport(func):
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
