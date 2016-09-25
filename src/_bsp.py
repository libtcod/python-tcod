
from . import Bsp as _Bsp
from .libtcod import _lib, _ffi

@_ffi.def_extern()
def _pycall_bsp_callback(node, func_handle):
    '''static bool _pycall_bsp_callback(TCOD_bsp_t *node, void *userData);
    '''
    return _ffi.from_handle(func_handle)(_Bsp(node))

def bsp_new_with_size(x, y, w, h):
    return _Bsp(_lib.TCOD_bsp_new_with_size(x, y, w, h))

def bsp_split_once(node, horizontal, position):
    _lib.TCOD_bsp_split_once(node.p, horizontal, position)

def bsp_split_recursive(node, randomizer, nb, minHSize, minVSize, maxHRatio,
                        maxVRatio):
    _lib.TCOD_bsp_split_recursive(node.p, randomizer or _ffi.NULL, nb, minHSize, minVSize,
                                  maxHRatio, maxVRatio)

def bsp_resize(node, x, y, w, h):
    _lib.TCOD_bsp_resize(node.p, x, y, w, h)

def bsp_left(node):
    return _Bsp(_lib.TCOD_bsp_left(node.p))

def bsp_right(node):
    return _Bsp(_lib.TCOD_bsp_right(node.p))

def bsp_father(node):
    return _Bsp(_lib.TCOD_bsp_father(node.p))

def bsp_is_leaf(node):
    return _lib.TCOD_bsp_is_leaf(node.p)

def bsp_contains(node, cx, cy):
    return _lib.TCOD_bsp_contains(node.p, cx, cy)

def bsp_find_node(node, cx, cy):
    return _Bsp(_lib.TCOD_bsp_find_node(node.p, cx, cy))

def _bsp_traverse(node, func, callback):
    '''pack (callback, *args, **kargs) into a handle for use with the callback
    _pycall_bsp_callback
    '''
    python_data = _ffi.new_handle(callback)
    func(node.p, _lib._pycall_bsp_callback, python_data)

def bsp_traverse_pre_order(node, callback):
    _bsp_traverse(node, _lib.TCOD_bsp_traverse_pre_order, callback)

def bsp_traverse_in_order(node, callback):
    _bsp_traverse(node, _lib.TCOD_bsp_traverse_in_order, callback)

def bsp_traverse_post_order(node, callback):
    _bsp_traverse(node, _lib.TCOD_bsp_traverse_post_order, callback)

def bsp_traverse_level_order(node, callback):
    _bsp_traverse(node, _lib.TCOD_bsp_traverse_level_order, callback)

def bsp_traverse_inverted_level_order(node, callback):
    _bsp_traverse(node, _lib.TCOD_bsp_traverse_inverted_level_order, callback)

def bsp_remove_sons(node):
    _lib.TCOD_bsp_remove_sons(node.p)

def bsp_delete(node):
    _lib.TCOD_bsp_delete(node.p)


__all__ = [_name for _name in list(globals()) if _name[0] != '_']
