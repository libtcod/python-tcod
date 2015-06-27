
from .libtcod import _lib, _ffi, _str, _unpack_char_p

def parse(filename,random=None):
    _lib.TCOD_namegen_parse(filename,random or _ffi.NULL)

def generate(name):
    return _unpack_char_p(_lib.TCOD_namegen_generate(_str(name), False))

def generate_custom(name, rule):
    return _unpack_char_p(_lib.TCOD_namegen_generate(_str(name), rule, False))

def get_sets():
    sets = _lib.TCOD_namegen_get_sets()
    try:
        lst = []
        while not _lib.TCOD_list_is_empty(sets):
            lst.append(_unpack_char_p(_ffi.cast('char *', _lib.TCOD_list_pop(sets))))
    finally:
        _lib.TCOD_list_delete(sets)
    return lst

def destroy():
    _lib.TCOD_namegen_destroy()

__all__ = [name for name in list(globals()) if name[0] != '_']