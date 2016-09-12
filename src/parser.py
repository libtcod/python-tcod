
import ctypes as _ctypes

import tcod as _tcod
from .libtcod import _lib, _ffi, _str, _unpack_char_p, _unicode

_chr = chr
try:
    _chr = unichr # Python 2
except NameError:
    pass

# property types
TYPE_NONE = 0
TYPE_BOOL = 1
TYPE_CHAR = 2
TYPE_INT = 3
TYPE_FLOAT = 4
TYPE_STRING = 5
TYPE_COLOR = 6
TYPE_DICE = 7
TYPE_VALUELIST00 = 8
TYPE_VALUELIST01 = 9
TYPE_VALUELIST02 = 10
TYPE_VALUELIST03 = 11
TYPE_VALUELIST04 = 12
TYPE_VALUELIST05 = 13
TYPE_VALUELIST06 = 14
TYPE_VALUELIST07 = 15
TYPE_VALUELIST08 = 16
TYPE_VALUELIST09 = 17
TYPE_VALUELIST10 = 18
TYPE_VALUELIST11 = 19
TYPE_VALUELIST12 = 20
TYPE_VALUELIST13 = 21
TYPE_VALUELIST14 = 22
TYPE_VALUELIST15 = 23
TYPE_LIST = 1024

def _unpack_union(typ, value):
    '''
        unpack items from parser new_property (value_converter)

        needs to be rewritten
    '''
    #union = _lib.TDL_get_union_from_void(value)
    if typ == TYPE_BOOL:
        return bool(union.b)
    elif typ == TYPE_CHAR:
        return _unicode(union.c)
    elif typ == TYPE_INT:
        return union.i
    elif typ == TYPE_FLOAT:
        return union.f#float(_ffi.cast('float', value))
    elif typ == TYPE_STRING or \
         TYPE_VALUELIST15 >= typ >= TYPE_VALUELIST00:
         return _unpack_char_p(union.s)
    elif typ == TYPE_COLOR:
        return _tcod.Color.from_cdata(_ffi.cast('TCOD_color_t *', union))
    elif typ == TYPE_DICE:
        return _tcod.Dice.from_cdata(_ffi.cast('TCOD_dice_t *', union))
    elif typ & TYPE_LIST:
        return _convert_TCODList(_ffi.cast('TCOD_list_t *', union), typ & 0xFF)
    else:
        raise RuntimeError('Unknown libtcod type: %i' % typ)

def _convert_TCODList(clist, typ):
    res = list()
    for i in range(_lib.TCOD_list_size(clist)):
        if typ == TYPE_BOOL:
            elt = bool(_lib.TDL_list_get_bool(clist, i))
        elif typ == TYPE_CHAR:
            elt = _unicode(_lib.TDL_list_get_char(clist, i))
        elif typ == TYPE_INT:
            elt = _lib.TDL_list_get_int(clist, i)
        elif typ == TYPE_FLOAT:
            elt = _lib.TDL_list_get_float(clist, i)
        elif typ == TYPE_STRING or TYPE_VALUELIST15 >= typ >= TYPE_VALUELIST00:
            elt = _unpack_char_p(_lib.TDL_list_get_string(clist, i))
        elif typ == TYPE_COLOR:
            elt = _tcod.Color.from_cdata(_lib.TDL_list_get_color(clist, i))
        elif typ == TYPE_DICE:
            elt = _tcod.Dice.from_cdata(_lib.TDL_list_get_dice(clist, i))
        else:
            raise TypeError('Unknown libtcod type: %i' % typ)
        res.append(elt)
    return res

def new():
    return _lib.TCOD_parser_new()

def new_struct(parser, name):
    return _lib.TCOD_parser_new_struct(parser, name)

def run(parser, filename, listener=None):
    if listener:
        return # STUB LISTENER

        # cast cffi parser to ctypes
        #ctypes_parser = parser#_ctypes.c_void_p.from_param(int(_ffi.cast('int', parser)))
        #clistener = _CParserListener()
        clistener = _ffi.new('TCOD_parser_listener_t *')
        #@_CFUNC_NEW_PROPERTY

        # code for handing unions, whenever cffi supports this
        @_ffi.callback('bool (*new_property)(const char *propname, TCOD_value_type_t type, TCOD_value_t value)')

        # cast TCOD_value_t to void* for the moment
        #@_ffi.callback('bool (*new_property)(const char *propname, TCOD_value_type_t type, void *value)')
        def value_converter(name, typ, value):
            name = _unpack_char_p(name)
            return listener.new_property(name, typ, _unpack_union(typ, value))

        #@_CFUNC_NEW_STRUCT
        @_ffi.callback('bool (*new_struct)(TCOD_parser_struct_t str,const char *name)')
        def new_struct(struct, c_char_p):
            struct = _ffi.cast('TCOD_parser_struct_t *', struct)
            return listener.new_struct(struct, _unpack_char_p(c_char_p))

        #@_CFUNC_NEW_STRUCT
        @_ffi.callback('bool (*end_struct)(TCOD_parser_struct_t str, const char *name)')
        def end_struct(struct, c_char_p):
            struct = _ffi.cast('TCOD_parser_struct_t *', struct)
            return listener.end_struct(struct, _unpack_char_p(c_char_p))

        #@_CFUNC_NEW_FLAG
        @_ffi.callback('void (*error)(const char *msg)')
        def error(msg):
            listener.error(_unpack_char_p(msg))

        # cast void* parameter to TCOD_value_t until cffi can handle unions
        clistener.new_property = _ffi.cast('bool (*new_property)(const char *propname, TCOD_value_type_t type, TCOD_value_t value)', value_converter)

        clistener.new_struct = new_struct
        clistener.end_struct = end_struct
        clistener.error = error

    else:
        clistener = _ffi.NULL
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

def get_list_property(parser, name, typ):
    clist = _lib.TCOD_parser_get_list_property(parser, _str(name), typ)
    return _convert_TCODList(clist, typ)

__all__ = [_name for _name in list(globals()) if _name[0] != '_']
