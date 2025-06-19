"""Test old libtcodpy parser."""

from pathlib import Path
from typing import Any

import pytest

import tcod as libtcod


@pytest.mark.filterwarnings("ignore")
def test_parser() -> None:
    print("***** File Parser test *****")
    parser = libtcod.parser_new()
    struct = libtcod.parser_new_struct(parser, "myStruct")
    libtcod.struct_add_property(struct, "bool_field", libtcod.TYPE_BOOL, True)
    libtcod.struct_add_property(struct, "char_field", libtcod.TYPE_CHAR, True)
    libtcod.struct_add_property(struct, "int_field", libtcod.TYPE_INT, True)
    libtcod.struct_add_property(struct, "float_field", libtcod.TYPE_FLOAT, True)
    libtcod.struct_add_property(struct, "color_field", libtcod.TYPE_COLOR, True)
    libtcod.struct_add_property(struct, "dice_field", libtcod.TYPE_DICE, True)
    libtcod.struct_add_property(struct, "string_field", libtcod.TYPE_STRING, True)
    libtcod.struct_add_list_property(struct, "bool_list", libtcod.TYPE_BOOL, True)
    libtcod.struct_add_list_property(struct, "char_list", libtcod.TYPE_CHAR, True)
    libtcod.struct_add_list_property(struct, "integer_list", libtcod.TYPE_INT, True)
    libtcod.struct_add_list_property(struct, "float_list", libtcod.TYPE_FLOAT, True)
    libtcod.struct_add_list_property(struct, "string_list", libtcod.TYPE_STRING, True)
    libtcod.struct_add_list_property(struct, "color_list", libtcod.TYPE_COLOR, True)

    # default listener
    print("***** Default listener *****")
    libtcod.parser_run(parser, Path("libtcod/data/cfg/sample.cfg"))
    print("bool_field : ", libtcod.parser_get_bool_property(parser, "myStruct.bool_field"))
    print("char_field : ", libtcod.parser_get_char_property(parser, "myStruct.char_field"))
    print("int_field : ", libtcod.parser_get_int_property(parser, "myStruct.int_field"))
    print("float_field : ", libtcod.parser_get_float_property(parser, "myStruct.float_field"))
    print("color_field : ", libtcod.parser_get_color_property(parser, "myStruct.color_field"))
    print("dice_field : ", libtcod.parser_get_dice_property(parser, "myStruct.dice_field"))
    print("string_field : ", libtcod.parser_get_string_property(parser, "myStruct.string_field"))
    print("bool_list : ", libtcod.parser_get_list_property(parser, "myStruct.bool_list", libtcod.TYPE_BOOL))
    print("char_list : ", libtcod.parser_get_list_property(parser, "myStruct.char_list", libtcod.TYPE_CHAR))
    print("integer_list : ", libtcod.parser_get_list_property(parser, "myStruct.integer_list", libtcod.TYPE_INT))
    print("float_list : ", libtcod.parser_get_list_property(parser, "myStruct.float_list", libtcod.TYPE_FLOAT))
    print("string_list : ", libtcod.parser_get_list_property(parser, "myStruct.string_list", libtcod.TYPE_STRING))
    print("color_list : ", libtcod.parser_get_list_property(parser, "myStruct.color_list", libtcod.TYPE_COLOR))

    # custom listener
    print("***** Custom listener *****")

    class MyListener:
        def new_struct(self, struct: Any, name: str) -> bool:  # noqa: ANN401
            print("new structure type", libtcod.struct_get_name(struct), " named ", name)
            return True

        def new_flag(self, name: str) -> bool:
            print("new flag named ", name)
            return True

        def new_property(self, name: str, typ: int, value: Any) -> bool:  # noqa: ANN401
            type_names = ["NONE", "BOOL", "CHAR", "INT", "FLOAT", "STRING", "COLOR", "DICE"]
            type_name = type_names[typ & 0xFF]
            if typ & libtcod.TYPE_LIST:
                type_name = f"LIST<{type_name}>"
            print("new property named ", name, " type ", type_name, " value ", value)
            return True

        def end_struct(self, struct: Any, name: str) -> bool:  # noqa: ANN401
            print("end structure type", libtcod.struct_get_name(struct), " named ", name)
            return True

        def error(self, msg: str) -> bool:
            print("error : ", msg)
            return True

    libtcod.parser_run(parser, Path("libtcod/data/cfg/sample.cfg"), MyListener())


if __name__ == "__main__":
    test_parser()
