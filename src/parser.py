
import ctypes as _ctypes
import threading as _threading

import tcod as _tcod
from .libtcod import _lib, _ffi, _str, _unpack_char_p, _unicode

_chr = chr
try:
    _chr = unichr # Python 2
except NameError:
    pass

def _unpack_union(type, union):
    '''
        unpack items from parser new_property (value_converter)
    '''
    if type == _lib.TCOD_TYPE_BOOL:
        return bool(union.b)
    elif type == _lib.TCOD_TYPE_CHAR:
        return _unicode(union.c)
    elif type == _lib.TCOD_TYPE_INT:
        return union.i
    elif type == _lib.TCOD_TYPE_FLOAT:
        return union.f
    elif (type == _lib.TCOD_TYPE_STRING or
         _lib.TCOD_TYPE_VALUELIST15 >= type >= _lib.TCOD_TYPE_VALUELIST00):
         return _unpack_char_p(union.s)
    elif type == _lib.TCOD_TYPE_COLOR:
        return _tcod.Color.from_cdata(union.col)
    elif type == _lib.TCOD_TYPE_DICE:
        return _tcod.Dice.from_cdata(union.dice)
    elif type & _lib.TCOD_TYPE_LIST:
        return _convert_TCODList(union.list, type & 0xFF)
    else:
        raise RuntimeError('Unknown libtcod type: %i' % type)

def _convert_TCODList(clist, type):
    res = list()
    for i in range(_lib.TCOD_list_size(clist)):
        if type == _lib.TCOD_TYPE_BOOL:
            elt = bool(_lib.TDL_list_get_bool(clist, i))
        elif type == _lib.TCOD_TYPE_CHAR:
            elt = _unicode(_lib.TDL_list_get_char(clist, i))
        elif type == _lib.TCOD_TYPE_INT:
            elt = _lib.TDL_list_get_int(clist, i)
        elif type == _lib.TCOD_TYPE_FLOAT:
            elt = _lib.TDL_list_get_float(clist, i)
        elif (type == _lib.TCOD_TYPE_STRING or
             _lib.TCOD_TYPE_VALUELIST15 >= type >= _lib.TCOD_TYPE_VALUELIST00):
            elt = _unpack_char_p(_lib.TDL_list_get_string(clist, i))
        elif type == _lib.TCOD_TYPE_COLOR:
            elt = _tcod.Color.from_cdata(_lib.TDL_list_get_color(clist, i))
        elif type == _lib.TCOD_TYPE_DICE:
            elt = _tcod.Dice.from_cdata(_lib.TDL_list_get_dice(clist, i))
        else:
            raise TypeError('Unknown libtcod type: %i' % type)
        res.append(elt)
    return res

def new():
    return _lib.TCOD_parser_new()

def new_struct(parser, name):
    return _lib.TCOD_parser_new_struct(parser, name)

# prevent multiple threads from messing with def_extern callbacks
_parser_callback_lock = _threading.Lock()

def run(parser, filename, listener=None):
    if not listener:
        _lib.TCOD_parser_run(parser, _str(filename), _ffi.NULL)
        return

    with _parser_callback_lock:
        clistener = _ffi.new('TCOD_parser_listener_t *')

        @_ffi.def_extern()
        def pycall_parser_new_struct(struct, name):
            return listener.end_struct(struct, _unpack_char_p(name))

        @_ffi.def_extern()
        def pycall_parser_new_flag(name):
            return listener.new_flag(_unpack_char_p(name))

        @_ffi.def_extern()
        def pycall_parser_new_property(propname, type, value):
            return listener.new_property(_unpack_char_p(propname), type,
                                         _unpack_union(type, value))

        @_ffi.def_extern()
        def pycall_parser_end_struct(struct, name):
            return listener.end_struct(struct, _unpack_char_p(name))

        @_ffi.def_extern()
        def pycall_parser_error(msg):
            listener.error(_unpack_char_p(msg))

        clistener.new_struct = _lib.pycall_parser_new_struct
        clistener.new_flag = _lib.pycall_parser_new_flag
        clistener.new_property = _lib.pycall_parser_new_property
        clistener.end_struct = _lib.pycall_parser_end_struct
        clistener.error = _lib.pycall_parser_error

        _lib.TCOD_parser_run(parser, _str(filename), clistener)

def delete(parser):
    _lib.TCOD_parser_delete(parser)

def get_bool_property(parser, name):
    return bool(_lib.TCOD_parser_get_bool_property(parser, _str(name)))

def get_int_property(parser, name):
    return _lib.TCOD_parser_get_int_property(parser, _str(name))

def get_char_property(parser, name):
    return _chr(_lib.TCOD_parser_get_char_property(parser, _str(name)))

def get_float_property(parser, name):
    return _lib.TCOD_parser_get_float_property(parser, _str(name))

def get_string_property(parser, name):
    return _unpack_char_p(_lib.TCOD_parser_get_string_property(parser, _str(name)))

def get_color_property(parser, name):
    return _tcod.Color.from_cdata(_lib.TCOD_parser_get_color_property(parser, _str(name)))

def get_dice_property(parser, name):
    d = _ffi.new('TCOD_dice_t *')
    _lib.TCOD_parser_get_dice_property_py(parser, _str(name), d)
    return _tcod.Dice.from_cdata(d)

def get_list_property(parser, name, type):
    clist = _lib.TCOD_parser_get_list_property(parser, _str(name), type)
    return _convert_TCODList(clist, type)

__all__ = [_name for _name in list(globals()) if _name[0] != '_']
