
import sys as _sys

import weakref as _weakref
import functools as _functools

from .libtcod import _lib, _ffi, _PropagateException

@_ffi.def_extern()
def _pycall_bsp_callback(node, handle):
    """static bool _pycall_bsp_callback(TCOD_bsp_t *node, void *userData);"""
    func, userData, propagate = _ffi.from_handle(handle)
    try:
        return func(BSP.from_cdata(node), userData)
    except BaseException:
        propagate(*_sys.exc_info())
        return None

class BSP(object):
    """

    .. versionchanged:: 2.0
       You can create BSP's with this class contructor instead of using
       `bsp_new_with_size`

       You can no longer set attributes: position, horizontal, or level.
       They had no effect when you changed them.

    """

    def __new__(cls, x, y, w, h):
        """

        .. versionchanged:: 2.0
        """
        self = object.__new__(cls)
        self.cdata = _ffi.gc(_lib.TCOD_bsp_new_with_size(x, y, w, h),
                             _lib.TCOD_bsp_delete)
        self._reference = None # to prevent garbage collection
        self._children = _weakref.WeakSet() # used by _invalidate_children
        return self

    @classmethod
    def from_cdata(cls, cdata, reference=None):
        """Create a BSP instance from a CData instance.

        This is an alternative constructor, normally for internal use.

        :param TCOD_bsp_t cdata: Pointer to a TCOD_bsp_t CData instance.
        :param BSP reference: Used internally to prevent the root BSP from
                              beomcing garbage collected.

        .. versionadded:: 2.0
        """
        self = object.__new__(cls)
        self.cdata = cdata
        self._reference = reference
        self._children = _weakref.WeakSet()
        if reference:
            reference._children.add(self)
        return self

    def _invalidate_children(self):
        """Invalidates BSP instances known to be based off of this one."""
        for child in self._children:
            child.cdata = _ffi.NULL
            child._reference = None
            child._invalidate_children()
        self._children.clear()

    def _assert_sanity(self):
        """Make sure nobody broke everything by using bsp_remove_sons"""
        assert self.cdata != _ffi.NULL, 'This BSP instance was deleted!'
        return True

    def __repr__(self):
        """Provide a useful readout when printed."""
        if not self.cdata:
            return '<%s NULL!>' % self.__class__.__name__

        status = 'leaf'
        if not self.is_leaf():
            status = ('split at position=%i,orientation=%r' %
                      (self.position, self.orientation()))

        return ('<%s(x=%i,y=%i,w=%i,h=%i)depth=%i,%s>' %
                (self.__class__.__name__,
                 self.x, self.y, self.w, self.h, self.depth(), status))

    def getx(self):
        assert self._assert_sanity()
        return self.cdata.x
    def setx(self, value):
        assert self._assert_sanity()
        self.cdata.x = value
    x = property(getx, setx)

    def gety(self):
        assert self._assert_sanity()
        return self.cdata.y
    def sety(self, value):
        assert self._assert_sanity()
        self.cdata.y = value
    y = property(gety, sety)

    def getw(self):
        assert self._assert_sanity()
        return self.cdata.w
    def setw(self, value):
        assert self._assert_sanity()
        self.cdata.w = value
    w = property(getw, setw)

    def geth(self):
        assert self._assert_sanity()
        return self.cdata.h
    def seth(self, value):
        assert self._assert_sanity()
        self.cdata.h = value
    h = property(geth, seth)

    def getpos(self):
        assert self._assert_sanity()
        return self.cdata.position
    def setpos(self, value):
        assert self._assert_sanity()
        self.cdata.position = value
    position = property(getpos)

    def gethor(self):
        assert self._assert_sanity()
        return self.cdata.horizontal
    def sethor(self,value):
        assert self._assert_sanity()
        self.cdata.horizontal = value
    horizontal = property(gethor)

    def getlev(self):
        assert self._assert_sanity()
        return self.cdata.level
    def setlev(self,value):
        assert self._assert_sanity()
        self.cdata.level = value
    level = property(getlev)

    def depth(self):
        """

        .. versionadded:: 2.0
        """
        assert self._assert_sanity()
        return self.cdata.level

    def orientation(self):
        """

        .. versionadded:: 2.0
        """
        assert self._assert_sanity()
        if self.is_leaf():
            return ''
        elif self.cdata.horizontal:
            return 'horizontal'
        else:
            return 'vertical'

    def split_once(self, orientation, position):
        """

        .. versionadded:: 2.0
        """
        # orientation = horz
        assert self._assert_sanity()
        _lib.TCOD_bsp_split_once(self.cdata, orientation, position)
        return self.children()

    def resize(self, x, y, w, h):
        """Resize this BSP to the provided rectangle.

        :param int x: new left coordinate
        :param int y: new top coordinate
        :param int w: new width
        :param int h: new height

        .. versionadded:: 2.0
        """
        assert self._assert_sanity()
        _lib.TCOD_bsp_resize(self.cdata, x, y, w, h)


    def left(self):
        """Return this BSP's 'left' child.

        Returns None if this BSP is a leaf node.

        :return: BSP's left/top child or None.

        .. versionadded:: 2.0
        """
        assert self._assert_sanity()
        if self.is_leaf():
            return None
        return BSP.from_cdata(_lib.TCOD_bsp_left(self.cdata), self)

    def right(self):
        """Return this BSP's 'right' child.

        Returns None if this BSP is a leaf node.

        :return: BSP's right/bottom child or None.

        .. versionadded:: 2.0
        """
        assert self._assert_sanity()
        if self.is_leaf():
            return None
        return BSP.from_cdata(_lib.TCOD_bsp_right(self.cdata), self)

    def parent(self):
        """Return this BSP's parent node.

        :return: Returns the parent node as a BSP instance.
                 Returns None if this BSP has no parent.

        .. versionadded:: 2.0
        """
        assert self._assert_sanity()
        if self.is_leaf():
            return None
        return BSP.from_cdata(_lib.TCOD_bsp_father(self.cdata), self)

    def children(self):
        """Return as a tuple, this instances immediate children, if any.

        An ideal usage of this function is:

        .. code-block:: python
            try:
                left, right = bsp.children()
            except ValueError:
                pass # this node is a leaf
            else:
                pass # work with children here

        :return: Returns a tuple of (left, right) BSP instances
                 Returns None if this BSP has no children.

        .. versionadded:: 2.0
        """
        assert self._assert_sanity()
        if self.is_leaf():
            return ()
        return (BSP.from_cdata(_lib.TCOD_bsp_left(self.cdata), self),
                BSP.from_cdata(_lib.TCOD_bsp_right(self.cdata), self))

    def walk(self):
        """Iterate over this BSP's hieracrhy.

        The iterator will include the instance which called it.
        It will traverse its own children and grandchildren, in no particular
        order.

        :return: Returns an iterator of BSP instances.

        .. versionadded:: 2.0
        """
        assert self._assert_sanity()
        yield self
        for child in self.children():
            for grandchild in child.walk():
                yield grandchild

    def is_leaf(self):
        """Returns True if this node is a leaf.  False when this node has children.

        .. versionadded:: 2.0
        """
        assert self._assert_sanity()
        return bool(_lib.TCOD_bsp_is_leaf(self.cdata))

    def contains(self, x, y):
        """Returns True if this node contains these coordinates.

        .. versionadded:: 2.0
        """
        assert self._assert_sanity()
        return bool(_lib.TCOD_bsp_contains(self.cdata, x, y))

    def find_node(self, x, y):
        assert self._assert_sanity()
        node = BSP.from_cdata(_lib.TCOD_bsp_find_node(self.cdata, x, y), self)
        if node.cdata == _ffi.NULL:
            node = None
        return node


def bsp_new_with_size(x, y, w, h):
    """
    .. deprecated:: 2.0
       Initialize by `BSP` directly.
    """
    return BSP(x, y, w, h)

def bsp_split_once(node, horizontal, position):
    """
    .. deprecated:: 2.0
       Use `BSP.split_once` instead.
    """
    return node.split_once()

def bsp_split_recursive(node, randomizer, nb, minHSize, minVSize, maxHRatio,
                        maxVRatio):
    _lib.TCOD_bsp_split_recursive(node.cdata, randomizer or _ffi.NULL, nb, minHSize, minVSize,
                                  maxHRatio, maxVRatio)

def bsp_resize(node, x, y, w, h):
    """
    .. deprecated:: 2.0
       Use `BSP.resize` instead.
    """
    node.resize(x, y, w, h)

def bsp_left(node):
    """
    .. deprecated:: 2.0
       Use `BSP.left` instead.
    """
    return node.left()

def bsp_right(node):
    """
    .. deprecated:: 2.0
       Use `BSP.right` instead.
    """
    return node.right()

def bsp_father(node):
    """
    .. deprecated:: 2.0
       Use `BSP.parent` instead.
    """
    return node.parent()

def bsp_is_leaf(node):
    """
    .. deprecated:: 2.0
       Use `BSP.is_leaf` instead.
    """
    return node.is_leaf()

def bsp_contains(node, cx, cy):
    """
    .. deprecated:: 2.0
       Use `BSP.contains` instead.
    """
    return node.contains(cx, cy)

def bsp_find_node(node, cx, cy):
    """
    .. deprecated:: 2.0
       Use `BSP.find_node` instead.
    """
    return node.find_node()

def _bsp_traverse(node, func, callback, userData):
    """pack callback into a handle for use with the callback
    _pycall_bsp_callback
    """
    with _PropagateException() as propagate:
        handle = _ffi.new_handle((callback, userData, propagate))
        func(node.cdata, _lib._pycall_bsp_callback, handle)

def bsp_traverse_pre_order(node, callback, userData=0):
    _bsp_traverse(node, _lib.TCOD_bsp_traverse_pre_order, callback, userData)

def bsp_traverse_in_order(node, callback, userData=0):
    _bsp_traverse(node, _lib.TCOD_bsp_traverse_in_order, callback, userData)

def bsp_traverse_post_order(node, callback, userData=0):
    _bsp_traverse(node, _lib.TCOD_bsp_traverse_post_order, callback, userData)

def bsp_traverse_level_order(node, callback, userData=0):
    """Traverse this nodes hierarchy with a callback.

    .. deprecated:: 2.0
       Use `BSP.walk` instead.
    """
    _bsp_traverse(node, _lib.TCOD_bsp_traverse_level_order, callback, userData)

def bsp_traverse_inverted_level_order(node, callback, userData=0):
    """Traverse this nodes hierarchy with a callback.

    .. deprecated:: 2.0
       Use `BSP.walk` instead.
    """
    _bsp_traverse(node, _lib.TCOD_bsp_traverse_inverted_level_order,
                  callback, userData)

def bsp_remove_sons(node):
    """Delete all children of a given node.  Not recommended.

    .. note::
       This function will add unnecessary complexity to your code.
       Don't use it.

    .. deprecated:: 2.0
       BSP deletion is automatic.
    """
    node._invalidate_children()
    _lib.TCOD_bsp_remove_sons(node.cdata)

def bsp_delete(node):
    """Exists for backward compatibility.  Does nothing.

    BSP's created by this library are automatically garbage collected once
    there are no references to the tree.
    This function exists for backwards compatibility.

    .. deprecated:: 2.0
       BSP deletion is automatic.
    """

__all__ = [_name for _name in list(globals()) if _name[0] != '_']
