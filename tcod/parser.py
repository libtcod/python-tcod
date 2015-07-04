
import ctypes as _ctypes

import tcod as _tcod
from .libtcod import _lib, _ffi, _lib_ctypes, _str, _unpack_char_p, _unicode

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
            elt = _tcod.Color.from_tcod(_lib.TDL_list_get_color(clist, i))
        elif typ == TYPE_DICE:
            elt = _tcod.Dice.from_cdata(_lib.TDL_list_get_dice(clist, i))
        else:
            raise TypeError('No type for %i' % typ)
        res.append(elt)
    return res

def new():
    return _lib.TCOD_parser_new()

def new_struct(parser, name):
    return _lib.TCOD_parser_new_struct(parser, name)

class _CValue(_ctypes.Union):
    _fields_=[('c',_ctypes.c_uint8),
              ('i',_ctypes.c_int),
              ('f',_ctypes.c_float),
              ('s',_ctypes.c_char_p),
              # JBR03192012 See http://bugs.python.org/issue14354 for why these are not defined as their actual types
              ('col',_ctypes.c_uint8 * 3),
              ('dice',_ctypes.c_int * 4),
              ('custom',_ctypes.c_void_p),
              ]

_CFUNC_NEW_STRUCT = _ctypes.CFUNCTYPE(_ctypes.c_uint, _ctypes.c_void_p, 
                                      _ctypes.c_char_p)
_CFUNC_NEW_FLAG = _ctypes.CFUNCTYPE(_ctypes.c_uint, _ctypes.c_char_p)
_CFUNC_NEW_PROPERTY = _ctypes.CFUNCTYPE(_ctypes.c_uint, _ctypes.c_char_p,
                                        _ctypes.c_int, _CValue)
              
                   
class _CParserListener(_ctypes.Structure):
    _fields_=[('new_struct', _CFUNC_NEW_STRUCT),
              ('new_flag',_CFUNC_NEW_FLAG),
              ('new_property',_CFUNC_NEW_PROPERTY),
              ('end_struct',_CFUNC_NEW_STRUCT),
              ('error',_CFUNC_NEW_FLAG),
              ]
 
    
def run(parser, filename, listener=None):
    if listener:
        # cast cffi parser to ctypes
        ctypes_parser = _ctypes.c_void_p.from_param(int(_ffi.cast('int', parser)))
        clistener = _CParserListener()
        @_CFUNC_NEW_PROPERTY
        def value_converter(name, typ, value):
            name = _unicode(name)
            if typ == TYPE_BOOL:
                return listener.new_property(name, typ, value.c == 1)
            elif typ == TYPE_CHAR:
                return listener.new_property(name, typ, '%c' % (value.c & 0xFF))
            elif typ == TYPE_INT:
                return listener.new_property(name, typ, value.i)
            elif typ == TYPE_FLOAT:
                return listener.new_property(name, typ, value.f)
            elif typ == TYPE_STRING or \
                 TYPE_VALUELIST15 >= typ >= TYPE_VALUELIST00:
                 return listener.new_property(name, typ, value.s)
            elif typ == TYPE_COLOR:
                col = _tcod.Color(*value.col)
                return listener.new_property(name, typ, col)
            elif typ == TYPE_DICE:
                dice = _ffi.cast('TCOD_dice_t *', _ctypes.addressof(value.dice))
                return listener.new_property(name, typ,
                                             _tcod.Dice.from_cdata(dice))
            elif typ & TYPE_LIST:
                value = _ffi.cast('TCOD_list_t *', value.custom)
                return listener.new_property(name, typ,
                                _convert_TCODList(value, typ & 0xFF))
            return True
        
        @_CFUNC_NEW_STRUCT
        def new_struct(struct, c_char_p):
            struct = _ffi.cast('TCOD_parser_struct_t *', struct)
            return listener.new_struct(struct, _unicode(c_char_p))
        
        @_CFUNC_NEW_STRUCT
        def end_struct(struct, c_char_p):
            struct = _ffi.cast('TCOD_parser_struct_t *', struct)
            return listener.end_struct(struct, _unicode(c_char_p))
        
        @_CFUNC_NEW_FLAG
        def error(msg):
            listener.error(_unicode(msg))
        
        clistener.new_struct = new_struct
        clistener.end_struct = end_struct
        clistener.new_property = value_converter
        clistener.error = error
        
        _lib_ctypes.TCOD_parser_run(ctypes_parser,
                                    _ctypes.c_char_p(filename),
                                    _ctypes.byref(clistener))
    else:
        _lib.TCOD_parser_run(parser, _str(filename), _ffi.NULL)

def delete(parser):
    _lib.TCOD_parser_delete(parser)

def get_bool_property(parser, name):
    return bool(_lib.TCOD_parser_get_bool_property(parser, _str(name)))

def get_int_property(parser, name):
    return _lib.TCOD_parser_get_int_property(parser, _str(name))

def get_char_property(parser, name):
    return _unicode(_lib.TCOD_parser_get_char_property(parser, _str(name)))

def get_float_property(parser, name):
    return _lib.TCOD_parser_get_float_property(parser, _str(name))

def get_string_property(parser, name):
    return _unpack_char_p(_lib.TCOD_parser_get_string_property(parser, _str(name)))

def get_color_property(parser, name):
    return _tcod.Color.from_tcod(_lib.TCOD_parser_get_color_property(parser, _str(name)))

def get_dice_property(parser, name):
    d = _ffi.new('TCOD_dice_t *')
    _lib.TCOD_parser_get_dice_property_py(parser, _str(name), d)
    return _tcod.Dice.from_cdata(d)

def get_list_property(parser, name, typ):
    clist = _lib.TCOD_parser_get_list_property(parser, _str(name), typ)
    return _convert_TCODList(clist, typ)

__all__ = [_name for _name in list(globals()) if _name[0] != '_']
