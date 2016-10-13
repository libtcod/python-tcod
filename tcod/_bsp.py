
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
        return False

def _ensure_sanity(func):
    """Any BSP methods which use a cdata object in a TCOD call need to have
    a sanity check, otherwise it may end up passing a NULL pointer"""
    if __debug__:
        @_functools.wraps(func)
        def check_sanity(*args, **kargs):
            assert self.cdata != _ffi.NULL, 'This BSP instance was deleted!'
            return func(*args, **kargs)
    return func

class BSP(object):
    """

    .. attribute:: x
    .. attribute:: y
    .. attribute:: w
    .. attribute:: h

    :param int x: rectangle left coordinate
    :param int y: rectangle top coordinate
    :param int w: rectangle width
    :param int h: rectangle height

    .. versionchanged:: 2.0
       You can create BSP's with this class contructor instead of using
       :any:`bsp_new_with_size`.

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
        """Create a BSP instance from a "TCOD_bsp_t*" pointer.

        This is an alternative constructor, normally for internal use.

        :param TCOD_bsp_t* cdata: Must be a TCOD_bsp_t*
                                 :any:`CData <ffi-cdata>` instance.
        :param BSP reference: Used internally to prevent the root BSP
                              from becoming garbage collected.

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
            child._reference = None
            child._invalidate_children()
            child.cdata = _ffi.NULL
        self._children.clear()

    def __repr__(self):
        """Provide a useful readout when printed."""
        if not self.cdata:
            return '<%s NULL!>' % self.__class__.__name__

        status = 'leaf'
        if not self.is_leaf():
            status = ('split at dicision=%i,orientation=%r' %
                      (self.get_division(), self.get_orientation()))

        return ('<%s(x=%i,y=%i,w=%i,h=%i)depth=%i,%s>' %
                (self.__class__.__name__,
                 self.x, self.y, self.w, self.h, self.get_depth(), status))

    def __hash__(self):
        return hash(self.cdata)

    def __eq__(self, other):
        try:
            return self.cdata == other.cdata
        except AttributeError:
            return NotImplemented

    def __getattr__(self, attr):
        return getattr(self.__dict__['cdata'], attr)

    def __setattr__(self, attr, value):
        if attr != 'cdata' and hasattr(self.cdata, attr):
            setattr(self.cdata, attr, value)
            return
        object.__setattr__(self, attr, value)

    def get_depth(self):
        """Return the depth of this node.

        :rtype: int

        .. versionadded:: 2.0
        """
        return self.cdata.level

    def get_division(self):
        """Return the point where this node was divided into parts.

        :rtype: :any:`int` or :any:`None`

        .. versionadded:: 2.0
        """
        if self.is_leaf():
            return None
        return self.cdata.position

    def get_orientation(self):
        """

        :rtype: str

        .. versionadded:: 2.0
        """
        if self.is_leaf():
            return ''
        elif self.cdata.horizontal:
            return 'horizontal'
        else:
            return 'vertical'

    @_ensure_sanity
    def split_once(self, orientation, position):
        """

        :rtype: tuple

        .. versionadded:: 2.0
        """
        # orientation = horz
        if orientation[:1].lower() == 'h':
            _lib.TCOD_bsp_split_once(self.cdata, True, position)
        elif orientation[:1].lower() == 'v':
            _lib.TCOD_bsp_split_once(self.cdata, False, position)
        else:
            raise ValueError("orientation must be 'horizontal' or 'vertical'"
                             "\nNot %r" % orientation)
        return self.get_children()

    @_ensure_sanity
    def split_recursive(self, depth, min_width, min_height,
                        max_horz_ratio, max_vert_raito, random=None):
        """

        :rtype: iter

        .. versionadded:: 2.0
        """
        _lib.TCOD_bsp_split_recursive(self.cdata, random or _ffi.NULL,
                                      depth, min_width, min_height,
                                      max_horz_ratio, max_vert_raito)
        return self.walk()

    @_ensure_sanity
    def resize(self, x, y, w, h):
        """Resize this BSP to the provided rectangle.

        :param int x: rectangle left coordinate
        :param int y: rectangle top coordinate
        :param int w: rectangle width
        :param int h: rectangle height

        .. versionadded:: 2.0
        """
        _lib.TCOD_bsp_resize(self.cdata, x, y, w, h)

    @_ensure_sanity
    def get_left(self):
        """Return this BSP's 'left' child.

        Returns None if this BSP is a leaf node.

        :return: BSP's left/top child or None.
        :rtype: :any:`BSP` or :any:`None`

        .. versionadded:: 2.0
        """
        if self.is_leaf():
            return None
        return BSP.from_cdata(_lib.TCOD_bsp_left(self.cdata), self)

    @_ensure_sanity
    def get_right(self):
        """Return this BSP's 'right' child.

        Returns None if this BSP is a leaf node.

        :return: BSP's right/bottom child or None.
        :rtype: :any:`BSP` or :any:`None`

        .. versionadded:: 2.0
        """
        if self.is_leaf():
            return None
        return BSP.from_cdata(_lib.TCOD_bsp_right(self.cdata), self)

    @_ensure_sanity
    def get_parent(self):
        """Return this BSP's parent node.

        :return: Returns the parent node as a BSP instance.
                 Returns None if this BSP has no parent.
        :rtype: :any:`BSP` or :any:`None`

        .. versionadded:: 2.0
        """
        node = BSP.from_cdata(_lib.TCOD_bsp_father(self.cdata), self)
        if node.cdata == _ffi.NULL:
            return None
        return node

    @_ensure_sanity
    def get_children(self):
        """Return as a tuple, this instances immediate children, if any.

        :return: Returns a tuple of (left, right) BSP instances.
                 The returned tuple is empty if this BSP has no children.
        :rtype: tuple

        .. versionadded:: 2.0
        """
        if self.is_leaf():
            return ()
        return (BSP.from_cdata(_lib.TCOD_bsp_left(self.cdata), self),
                BSP.from_cdata(_lib.TCOD_bsp_right(self.cdata), self))

    @_ensure_sanity
    def walk(self):
        """Iterate over this BSP's hieracrhy.

        The iterator will include the instance which called it.
        It will traverse its own children and grandchildren, in no particular
        order.

        :return: Returns an iterator of BSP instances.
        :rtype: iter

        .. versionadded:: 2.0
        """
        for child in self.get_children():
            for grandchild in child.walk():
                yield grandchild
        yield self

    @_ensure_sanity
    def is_leaf(self):
        """Returns True if this node is a leaf.  False when this node has children.

        :rtype: bool

        .. versionadded:: 2.0
        """
        return bool(_lib.TCOD_bsp_is_leaf(self.cdata))

    @_ensure_sanity
    def contains(self, x, y):
        """Returns True if this node contains these coordinates.

        :rtype: bool

        .. versionadded:: 2.0
        """
        return bool(_lib.TCOD_bsp_contains(self.cdata, x, y))

    @_ensure_sanity
    def find_node(self, x, y):
        """Return the deepest node which contains these coordinates.

        :rtype: :any:`BSP` or :any:`None`

        .. versionadded:: 2.0
        """
        node = BSP.from_cdata(_lib.TCOD_bsp_find_node(self.cdata, x, y), self)
        if node.cdata == _ffi.NULL:
            node = None
        return node


def bsp_new_with_size(x, y, w, h):
    """Create a new :any:`BSP` instance with the given rectangle.

    :param int x: rectangle left coordinate
    :param int y: rectangle top coordinate
    :param int w: rectangle width
    :param int h: rectangle height
    :rtype: BSP

    .. deprecated:: 2.0
       Calling the :any:`BSP` class instead.
    """
    return BSP(x, y, w, h)

def bsp_split_once(node, horizontal, position):
    """
    .. deprecated:: 2.0
       Use :any:`BSP.split_once` instead.
    """
    node.split_once('h' if horizontal else 'v', position)

def bsp_split_recursive(node, randomizer, nb, minHSize, minVSize, maxHRatio,
                        maxVRatio):
    node.split_recursive(nb, minHSize, minVSize,
                         maxHRatio, maxVRatio, randomizer)

def bsp_resize(node, x, y, w, h):
    """
    .. deprecated:: 2.0
       Use :any:`BSP.resize` instead.
    """
    node.resize(x, y, w, h)

def bsp_left(node):
    """
    .. deprecated:: 2.0
       Use :any:`BSP.get_left` instead.
    """
    return node.get_left()

def bsp_right(node):
    """
    .. deprecated:: 2.0
       Use :any:`BSP.get_right` instead.
    """
    return node.get_right()

def bsp_father(node):
    """
    .. deprecated:: 2.0
       Use :any:`BSP.get_parent` instead.
    """
    return node.get_parent()

def bsp_is_leaf(node):
    """
    .. deprecated:: 2.0
       Use :any:`BSP.is_leaf` instead.
    """
    return node.is_leaf()

def bsp_contains(node, cx, cy):
    """
    .. deprecated:: 2.0
       Use :any:`BSP.contains` instead.
    """
    return node.contains(cx, cy)

def bsp_find_node(node, cx, cy):
    """
    .. deprecated:: 2.0
       Use :any:`BSP.find_node` instead.
    """
    return node.find_node(cx, cy)

def _bsp_traverse(node, func, callback, userData):
    """pack callback into a handle for use with the callback
    _pycall_bsp_callback
    """
    with _PropagateException() as propagate:
        handle = _ffi.new_handle((callback, userData, propagate))
        func(node.cdata, _lib._pycall_bsp_callback, handle)

def bsp_traverse_pre_order(node, callback, userData=0):
    """Traverse this nodes hierarchy with a callback.

    .. deprecated:: 2.0
       Use :any:`BSP.walk` instead.
    """
    _bsp_traverse(node, _lib.TCOD_bsp_traverse_pre_order, callback, userData)

def bsp_traverse_in_order(node, callback, userData=0):
    """Traverse this nodes hierarchy with a callback.

    .. deprecated:: 2.0
       Use :any:`BSP.walk` instead.
    """
    _bsp_traverse(node, _lib.TCOD_bsp_traverse_in_order, callback, userData)

def bsp_traverse_post_order(node, callback, userData=0):
    """Traverse this nodes hierarchy with a callback.

    .. deprecated:: 2.0
       Use :any:`BSP.walk` instead.
    """
    _bsp_traverse(node, _lib.TCOD_bsp_traverse_post_order, callback, userData)

def bsp_traverse_level_order(node, callback, userData=0):
    """Traverse this nodes hierarchy with a callback.

    .. deprecated:: 2.0
       Use :any:`BSP.walk` instead.
    """
    _bsp_traverse(node, _lib.TCOD_bsp_traverse_level_order, callback, userData)

def bsp_traverse_inverted_level_order(node, callback, userData=0):
    """Traverse this nodes hierarchy with a callback.

    .. deprecated:: 2.0
       Use :any:`BSP.walk` instead.
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
    pass

__all__ = [_name for _name in list(globals()) if _name[0] != '_']
