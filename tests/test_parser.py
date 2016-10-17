#!/usr/bin/env python

import os

import tcod as libtcod

def test_parser():
    print ('***** File Parser test *****')
    parser=libtcod.parser_new()
    struct=libtcod.parser_new_struct(parser, b'myStruct')
    libtcod.struct_add_property(struct, b'bool_field', libtcod.TYPE_BOOL, True)
    libtcod.struct_add_property(struct, b'char_field', libtcod.TYPE_CHAR, True)
    libtcod.struct_add_property(struct, b'int_field', libtcod.TYPE_INT, True)
    libtcod.struct_add_property(struct, b'float_field', libtcod.TYPE_FLOAT, True)
    libtcod.struct_add_property(struct, b'color_field', libtcod.TYPE_COLOR, True)
    libtcod.struct_add_property(struct, b'dice_field', libtcod.TYPE_DICE, True)
    libtcod.struct_add_property(struct, b'string_field', libtcod.TYPE_STRING,
                                True)
    libtcod.struct_add_list_property(struct, b'bool_list', libtcod.TYPE_BOOL,
                                True)
    libtcod.struct_add_list_property(struct, b'char_list', libtcod.TYPE_CHAR,
                                True)
    libtcod.struct_add_list_property(struct, b'integer_list', libtcod.TYPE_INT,
                                True)
    libtcod.struct_add_list_property(struct, b'float_list', libtcod.TYPE_FLOAT,
                                True)
    libtcod.struct_add_list_property(struct, b'string_list', libtcod.TYPE_STRING,
                                True)
    libtcod.struct_add_list_property(struct, b'color_list', libtcod.TYPE_COLOR,
                                True)
##    # dice lists doesn't work yet
##    libtcod.struct_add_list_property(struct, b'dice_list', libtcod.TYPE_DICE,
##                                True)

    # default listener
    print ('***** Default listener *****')
    libtcod.parser_run(parser, os.path.join('libtcod', 'data', 'cfg', 'sample.cfg'))
    print ('bool_field : ', \
          libtcod.parser_get_bool_property(parser, b'myStruct.bool_field'))
    print ('char_field : ', \
          libtcod.parser_get_char_property(parser, b'myStruct.char_field'))
    print ('int_field : ', \
          libtcod.parser_get_int_property(parser, b'myStruct.int_field'))
    print ('float_field : ', \
          libtcod.parser_get_float_property(parser, b'myStruct.float_field'))
    print ('color_field : ', \
          libtcod.parser_get_color_property(parser, b'myStruct.color_field'))
    print ('dice_field : ', \
          libtcod.parser_get_dice_property(parser, b'myStruct.dice_field'))
    print ('string_field : ', \
          libtcod.parser_get_string_property(parser, b'myStruct.string_field'))
    print ('bool_list : ', \
          libtcod.parser_get_list_property(parser, b'myStruct.bool_list',
                                                           libtcod.TYPE_BOOL))
    print ('char_list : ', \
          libtcod.parser_get_list_property(parser, b'myStruct.char_list',
                                                           libtcod.TYPE_CHAR))
    print ('integer_list : ', \
          libtcod.parser_get_list_property(parser, b'myStruct.integer_list',
                                                           libtcod.TYPE_INT))
    print ('float_list : ', \
          libtcod.parser_get_list_property(parser, b'myStruct.float_list',
                                                           libtcod.TYPE_FLOAT))
    print ('string_list : ', \
          libtcod.parser_get_list_property(parser, b'myStruct.string_list',
                                                           libtcod.TYPE_STRING))
    print ('color_list : ', \
          libtcod.parser_get_list_property(parser, b'myStruct.color_list',
                                                           libtcod.TYPE_COLOR))
##    print ('dice_list : ', \
##          libtcod.parser_get_list_property(parser, b'myStruct.dice_list',
##                                                           libtcod.TYPE_DICE))

    # custom listener
    print ('***** Custom listener *****')
    class MyListener:
        def new_struct(self, struct, name):
            print ('new structure type', libtcod.struct_get_name(struct), \
                  ' named ', name )
            return True
        def new_flag(self, name):
            print ('new flag named ', name)
            return True
        def new_property(self,name, typ, value):
            type_names = ['NONE', 'BOOL', 'CHAR', 'INT', 'FLOAT', 'STRING', \
                          'COLOR', 'DICE']
            type_name = type_names[typ & 0xff]
            if typ & libtcod.TYPE_LIST:
                type_name = 'LIST<%s>' % type_name
            print ('new property named ', name,' type ',type_name, \
                      ' value ', value)
            return True
        def end_struct(self, struct, name):
            print ('end structure type', libtcod.struct_get_name(struct), \
                  ' named ', name)
            return True
        def error(self,msg):
            print ('error : ', msg)
            return True
    libtcod.parser_run(parser, os.path.join('libtcod','data','cfg','sample.cfg'), MyListener())

if __name__ == '__main__':
    test_parser()
