
from .libtcod import _lib, _unpack_char_p

def struct_add_flag(struct, name):
    _lib.TCOD_struct_add_flag(struct, name)

def struct_add_property(struct, name, typ, mandatory):
    _lib.TCOD_struct_add_property(struct, name, typ, mandatory)

def struct_add_value_list(struct, name, value_list, mandatory):
    CARRAY = c_char_p * (len(value_list) + 1)
    cvalue_list = CARRAY()
    for i in range(len(value_list)):
        cvalue_list[i] = cast(value_list[i], c_char_p)
    cvalue_list[len(value_list)] = 0
    _lib.TCOD_struct_add_value_list(struct, name, cvalue_list, mandatory)

def struct_add_list_property(struct, name, typ, mandatory):
    _lib.TCOD_struct_add_list_property(struct, name, typ, mandatory)

def struct_add_structure(struct, sub_struct):
    _lib.TCOD_struct_add_structure(struct, sub_struct)

def struct_get_name(struct):
    return _unpack_char_p(_lib.TCOD_struct_get_name(struct))

def struct_is_mandatory(struct, name):
    return _lib.TCOD_struct_is_mandatory(struct, name)

def struct_get_type(struct, name):
    return _lib.TCOD_struct_get_type(struct, name)
