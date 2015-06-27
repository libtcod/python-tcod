
from . import Bsp as _Bsp
from .libtcod import _lib, _ffi

BSP_CBK_FUNC = _ffi.callback('TCOD_bsp_callback_t') # bool(TCOD_bsp_t *, void *)

def new_with_size(x, y, w, h):
    return _Bsp(_lib.TCOD_bsp_new_with_size(x, y, w, h))

def split_once(node, horizontal, position):
    _lib.TCOD_bsp_split_once(node.p, horizontal, position)

def split_recursive(node, randomizer, nb, minHSize, minVSize, maxHRatio,
                        maxVRatio):
    _lib.TCOD_bsp_split_recursive(node.p, randomizer or _ffi.NULL, nb, minHSize, minVSize,
                                  maxHRatio, maxVRatio)

def resize(node, x, y, w, h):
    _lib.TCOD_bsp_resize(node.p, x, y, w, h)

def left(node):
    return _Bsp(_lib.TCOD_bsp_left(node.p))

def right(node):
    return _Bsp(_lib.TCOD_bsp_right(node.p))

def father(node):
    return _Bsp(_lib.TCOD_bsp_father(node.p))

def is_leaf(node):
    return _lib.TCOD_bsp_is_leaf(node.p)

def contains(node, cx, cy):
    return _lib.TCOD_bsp_contains(node.p, cx, cy)

def find_node(node, cx, cy):
    return _Bsp(_lib.TCOD_bsp_find_node(node.p, cx, cy))

def _bsp_traverse(node, callback, userData, func):
    # convert the c node into a python node
    #before passing it to the actual callback
    def node_converter(cnode, data):
        return callback(_Bsp(cnode), _ffi.from_handle(data))
    cbk_func = BSP_CBK_FUNC(node_converter)
    func(node.p, cbk_func, _ffi.new_handle(userData))

def traverse_pre_order(node, callback, userData=0):
    _bsp_traverse(node, callback, userData, _lib.TCOD_bsp_traverse_pre_order)

def traverse_in_order(node, callback, userData=0):
    _bsp_traverse(node, callback, userData, _lib.TCOD_bsp_traverse_in_order)

def traverse_post_order(node, callback, userData=0):
    _bsp_traverse(node, callback, userData, _lib.TCOD_bsp_traverse_post_order)

def traverse_level_order(node, callback, userData=0):
    _bsp_traverse(node, callback, userData, _lib.TCOD_bsp_traverse_level_order)

def traverse_inverted_level_order(node, callback, userData=0):
    _bsp_traverse(node, callback, userData,
                  _lib.TCOD_bsp_traverse_inverted_level_order)

def remove_sons(node):
    _lib.TCOD_bsp_remove_sons(node.p)

def delete(node):
    _lib.TCOD_bsp_delete(node.p)


__all__ = [name for name in list(globals()) if name[0] != '_']
