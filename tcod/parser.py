
import tcod as _tcod
from .libtcod import _lib, _ffi, _str, _unpack_char_p

class _PythonParser(object):
    
    def __init__(self):
        pass
        
    def parse(self, string):
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

_type_dict = {TYPE_BOOL: 'bool',
              TYPE_CHAR: 'char',
              TYPE_INT: 'int',
              TYPE_FLOAT: 'float *',
              }

def _convert_TCODList(clist, typ):
    res = list()
    for i in range(_lib.TCOD_list_size(clist)):
        if typ == TYPE_BOOL:
            elt = _lib.TDL_list_get_bool(clist, i)
        elif typ == TYPE_CHAR:
            elt = _lib.TDL_list_get_char(clist, i)
        elif typ == TYPE_INT:
            elt = _lib.TDL_list_get_int(clist, i)
        elif typ == TYPE_FLOAT:
            elt = _lib.TDL_list_get_float(clist, i)
        elif typ == TYPE_STRING or TYPE_VALUELIST15 >= typ >= TYPE_VALUELIST00:
            elt = _unpack_char_p(_lib.TDL_list_get_string(clist, i))
        elif typ == TYPE_COLOR:
            elt = _tcod.Color.from_tcod(_lib.TDL_list_get_color(clist, i))
        elif typ == TYPE_DICE:
            elt = _lib.TDL_list_get_dice(clist, i)
        else:
            raise TypeError('No type for %i' % typ)
        res.append(elt)
    return res

def new():
    return _lib.TCOD_parser_new()

def new_struct(parser, name):
    return _lib.TCOD_parser_new_struct(parser, name)

def run(parser, filename, listener=None):
    if listener:
        clistener = _ffi.new('TCOD_parser_listener_t *')
        def value_converter(name, typ, value):
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
                col = cast(value.col, POINTER(Color)).contents
                return listener.new_property(name, typ, col)
            elif typ == TYPE_DICE:
                dice = cast(value.dice, POINTER(Dice)).contents
                return listener.new_property(name, typ, dice)
            elif typ & TYPE_LIST:
                return listener.new_property(name, typ,
                                        _convert_TCODList(value.custom, typ & 0xFF))
            return True
        _CFUNC_NEW_STRUCT = _ffi.callback('bool(TCOD_parser_struct_t, char*)')
        clistener.new_struct = _CFUNC_NEW_STRUCT(listener.new_struct)
        
        _CFUNC_NEW_FLAG = _ffi.callback('bool(char*)')
        clistener.new_flag = _CFUNC_NEW_FLAG(listener.new_flag)
        
        _CFUNC_NEW_PROPERTY = _ffi.callback(
                               'bool(char*, TCOD_value_type_t, TCOD_value_t)')

        clistener.new_property = _CFUNC_NEW_PROPERTY(value_converter)
        clistener.end_struct = _CFUNC_NEW_STRUCT(listener.end_struct)
        clistener.error = _CFUNC_NEW_FLAG(listener.error)
        _lib.TCOD_parser_run(parser, _str(filename), clistener)
    else:
        _lib.TCOD_parser_run(parser, _str(filename), _ffi.NULL)

def delete(parser):
    _lib.TCOD_parser_delete(parser)

def get_bool_property(parser, name):
    return _lib.TCOD_parser_get_bool_property(parser, _str(name))

def get_int_property(parser, name):
    return _lib.TCOD_parser_get_int_property(parser, _str(name))

def get_char_property(parser, name):
    return '%c' % _lib.TCOD_parser_get_char_property(parser, _str(name))

def get_float_property(parser, name):
    return _lib.TCOD_parser_get_float_property(parser, _str(name))

def get_string_property(parser, name):
    return _unpack_char_p(_lib.TCOD_parser_get_string_property(parser, _str(name)))

def get_color_property(parser, name):
    return _tcod.Color.from_tcod(_lib.TCOD_parser_get_color_property(parser, _str(name)))

def get_dice_property(parser, name):
    d = _tcod.Dice()
    _lib.TCOD_parser_get_dice_property_py(_str(parser), _str(name), d)
    return d

def get_list_property(parser, name, typ):
    clist = _lib.TCOD_parser_get_list_property(parser, _str(name), typ)
    return _convert_TCODList(clist, typ)

__all__ = [_name for _name in list(globals()) if _name[0] != '_']
