'''
The following example shows how to traverse the BSP tree using Python.  This
assumes `create_room` and `connect_rooms` will be replaced by custom code.

Example::

    import tcod.bsp

    def create_room(node):
        """Initialize the room at this current node."""
        print('Create a room for %s.' % node)

    def connect_rooms(node):
        """Connect two fully initialized rooms."""
        node1, node2 = node.children
        print('Connect the rooms:\\n%s\\n%s' % (node1, node2))

    def traverse(node):
        """Traverse a BSP tree dispatching nodes to the correct calls."""
        # For nodes without children, node.children is an empty tuple.
        for child in node.children:
            traverse(child)

        if node.children:
            connect_rooms(node)
        else:
            create_room(node)

    bsp = tcod.bsp.BSP(x=0, y=0, width=80, height=60)
    bsp.split_recursive(
        depth=5,
        min_width=3,
        min_height=3,
        max_horizontal_ratio=1.5,
        max_vertical_ratio=1.5,
        )
    traverse(bsp)
'''
from __future__ import absolute_import as _

from tcod.libtcod import lib, ffi


class BSP(object):
    """A binary space partitioning tree which can be used for simple dungeon
    generation.

    Attributes:
        x (int): Rectangle left coordinate.
        y (int): Rectangle top coordinate.
        width (int): Rectangle width.
        height (int): Rectangle height.
        level (int): This nodes depth.
        position (int): The integer of where the node was split.
        horizontal (bool): This nodes split orientation.
        parent (Optional[BSP]): This nodes parent or None
        children (Optional[Tuple[BSP, BSP]]):
            A tuple of (left, right) BSP instances, or
            None if this BSP has no children.

    Args:
        x (int): Rectangle left coordinate.
        y (int): Rectangle top coordinate.
        width (int): Rectangle width.
        height (int): Rectangle height.
    """

    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

        self.level = 0
        self.position = 0
        self.horizontal = False

        self.parent = None
        self.children = ()

    @property
    def w(self):
        return self.width
    @w.setter
    def w(self, value):
        self.width = value

    @property
    def h(self):
        return self.height
    @h.setter
    def h(self, value):
        self.height = value

    def _as_cdata(self):
        cdata = ffi.gc(lib.TCOD_bsp_new_with_size(self.x, self.y,
                                                  self.width, self.height),
                       lib.TCOD_bsp_delete)
        cdata.level = self.level
        return cdata

    def __str__(self):
        """Provide a useful readout when printed."""
        status = 'leaf'
        if self.children:
            status = ('split at position=%i,horizontal=%r' %
                      (self.position, self.horizontal))

        return ('<%s(x=%i,y=%i,width=%i,height=%i)level=%i,%s>' %
                (self.__class__.__name__,
                 self.x, self.y, self.width, self.height, self.level, status))

    def _unpack_bsp_tree(self, cdata):
        self.x = cdata.x
        self.y = cdata.y
        self.width = cdata.w
        self.height = cdata.h
        self.level = cdata.level
        self.position = cdata.position
        self.horizontal = bool(cdata.horizontal)
        if lib.TCOD_bsp_is_leaf(cdata):
            return
        self.children = (BSP(0, 0, 0, 0), BSP(0, 0, 0, 0))
        self.children[0].parent = self
        self.children[0]._unpack_bsp_tree(lib.TCOD_bsp_left(cdata))
        self.children[1].parent = self
        self.children[1]._unpack_bsp_tree(lib.TCOD_bsp_right(cdata))

    def split_once(self, horizontal, position):
        """Split this partition into 2 sub-partitions.

        Args:
            horizontal (bool):
            position (int):
        """
        cdata = self._as_cdata()
        lib.TCOD_bsp_split_once(cdata, horizontal, position)
        self._unpack_bsp_tree(cdata)

    def split_recursive(self, depth, min_width, min_height,
                        max_horizontal_ratio, max_vertical_ratio, seed=None):
        """Divide this partition recursively.

        Args:
            depth (int): The maximum depth to divide this object recursively.
            min_width (int): The minimum width of any individual partition.
            min_height (int): The minimum height of any individual partition.
            max_horizontal_ratio (float):
                Prevent creating a horizontal ratio more extreme than this.
            max_vertical_ratio (float):
                Prevent creating a vertical ratio more extreme than this.
            seed (Optional[tcod.random.Random]):
                The random number generator to use.
        """
        cdata = self._as_cdata()
        lib.TCOD_bsp_split_recursive(
            cdata,
            seed or ffi.NULL,
            depth,
            min_width, min_height,
            max_horizontal_ratio, max_vertical_ratio,
        )
        self._unpack_bsp_tree(cdata)

    def walk(self):
        """Iterate over this BSP's hieracrhy.

        The iterator will include the instance which called it.
        It will traverse its own children and grandchildren, in no particular
        order.

        Returns:
            Iterator[BSP]: An iterator of BSP nodes.

        .. deprecated:: 2.3
            See the module example for how to iterate over a BSP tree.
        """
        return self._iter_post_order()

    def _iter_pre_order(self):
        yield self
        for child in self.children:
            for grandchild in child._iter_pre_order():
                yield grandchild

    def _iter_in_order(self):
        if self.children:
            for grandchild in self.children[0]._iter_in_order():
                yield grandchild
            yield self
            for grandchild in self.children[1]._iter_in_order():
                yield grandchild
        else:
            yield self

    def _iter_post_order(self):
        for child in self.children:
            for grandchild in child._iter_post_order():
                yield grandchild
        yield self

    def _iter_level_order(self):
        return sorted(self._iter_pre_order(), key=lambda n:n.level)

    def _iter_inverted_level_order(self):
        return reversed(self._iter_level_order())

    def contains(self, x, y):
        """Returns True if this node contains these coordinates.

        Args:
            x (int): X position to check.
            y (int): Y position to check.

        Returns:
            bool: True if this node contains these coordinates.
                  Otherwise False.
        """
        return (self.x <= x < self.x + self.width and
                self.y <= y < self.y + self.height)

    def find_node(self, x, y):
        """Return the deepest node which contains these coordinates.

        Returns:
            Optional[BSP]: BSP object or None.
        """
        if not self.contains(x, y):
            return None
        for child in self.children:
            found = child.find_node(x, y)
            if found:
                return found
        return self
