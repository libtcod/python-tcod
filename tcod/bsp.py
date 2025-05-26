r"""Libtcod's Binary Space Partitioning.

The following example shows how to traverse the BSP tree using Python.  This
assumes `create_room` and `connect_rooms` will be replaced by custom code.

Example::

    import tcod.bsp

    bsp = tcod.bsp.BSP(x=0, y=0, width=80, height=60)
    bsp.split_recursive(
        depth=5,
        min_width=3,
        min_height=3,
        max_horizontal_ratio=1.5,
        max_vertical_ratio=1.5,
    )

    # In pre order, leaf nodes are visited before the nodes that connect them.
    for node in bsp.pre_order():
        if node.children:
            node1, node2 = node.children
            print('Connect the rooms:\n%s\n%s' % (node1, node2))
        else:
            print('Dig a room for %s.' % node)
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from typing_extensions import deprecated

from tcod.cffi import ffi, lib

if TYPE_CHECKING:
    import tcod.random


class BSP:
    """A binary space partitioning tree which can be used for simple dungeon generation.

    Attributes:
        x (int): Rectangle left coordinate.
        y (int): Rectangle top coordinate.
        width (int): Rectangle width.
        height (int): Rectangle height.
        level (int): This nodes depth.
        position (int): The integer of where the node was split.
        horizontal (bool): This nodes split orientation.
        parent (Optional[BSP]): This nodes parent or None
        children (Union[Tuple[()], Tuple[BSP, BSP]]):
            A tuple of (left, right) BSP instances, or
            an empty tuple if this BSP has no children.

    Args:
        x (int): Rectangle left coordinate.
        y (int): Rectangle top coordinate.
        width (int): Rectangle width.
        height (int): Rectangle height.
    """

    def __init__(self, x: int, y: int, width: int, height: int) -> None:
        """Initialize a root node of a BSP tree."""
        self.x = x
        self.y = y
        self.width = width
        self.height = height

        self.level = 0
        self.position = 0
        self.horizontal = False

        self.parent: BSP | None = None
        self.children: tuple[()] | tuple[BSP, BSP] = ()

    @property
    @deprecated("This attribute has been renamed to `width`.", category=FutureWarning)
    def w(self) -> int:  # noqa: D102
        return self.width

    @w.setter
    def w(self, value: int) -> None:
        self.width = value

    @property
    @deprecated("This attribute has been renamed to `height`.", category=FutureWarning)
    def h(self) -> int:  # noqa: D102
        return self.height

    @h.setter
    def h(self, value: int) -> None:
        self.height = value

    def _as_cdata(self) -> Any:  # noqa: ANN401
        cdata = ffi.gc(
            lib.TCOD_bsp_new_with_size(self.x, self.y, self.width, self.height),
            lib.TCOD_bsp_delete,
        )
        cdata.level = self.level
        return cdata

    def __repr__(self) -> str:
        """Provide a useful readout when printed."""
        status = "leaf"
        if self.children:
            status = f"split at position={self.position},horizontal={self.horizontal!r}"

        return f"<{self.__class__.__name__}(x={self.x},y={self.y},width={self.width},height={self.height}) level={self.level} {status}>"

    def _unpack_bsp_tree(self, cdata: Any) -> None:  # noqa: ANN401
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

    def split_once(self, horizontal: bool, position: int) -> None:
        """Split this partition into 2 sub-partitions.

        Args:
            horizontal (bool): If True then the sub-partition is split into an upper and bottom half.
            position (int): The position of where to put the divider relative to the current node.
        """
        cdata = self._as_cdata()
        lib.TCOD_bsp_split_once(cdata, horizontal, position)
        self._unpack_bsp_tree(cdata)

    def split_recursive(  # noqa: PLR0913
        self,
        depth: int,
        min_width: int,
        min_height: int,
        max_horizontal_ratio: float,
        max_vertical_ratio: float,
        seed: tcod.random.Random | None = None,
    ) -> None:
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
            min_width,
            min_height,
            max_horizontal_ratio,
            max_vertical_ratio,
        )
        self._unpack_bsp_tree(cdata)

    @deprecated("Use pre_order method instead of walk.")
    def walk(self) -> Iterator[BSP]:
        """Iterate over this BSP's hierarchy in pre order.

        .. deprecated:: 2.3
            Use :any:`pre_order` instead.
        """
        return self.post_order()

    def pre_order(self) -> Iterator[BSP]:
        """Iterate over this BSP's hierarchy in pre order.

        .. versionadded:: 8.3
        """
        yield self
        for child in self.children:
            yield from child.pre_order()

    def in_order(self) -> Iterator[BSP]:
        """Iterate over this BSP's hierarchy in order.

        .. versionadded:: 8.3
        """
        if self.children:
            yield from self.children[0].in_order()
            yield self
            yield from self.children[1].in_order()
        else:
            yield self

    def post_order(self) -> Iterator[BSP]:
        """Iterate over this BSP's hierarchy in post order.

        .. versionadded:: 8.3
        """
        for child in self.children:
            yield from child.post_order()
        yield self

    def level_order(self) -> Iterator[BSP]:
        """Iterate over this BSP's hierarchy in level order.

        .. versionadded:: 8.3
        """
        next = [self]
        while next:
            level = next
            next = []
            yield from level
            for node in level:
                next.extend(node.children)

    def inverted_level_order(self) -> Iterator[BSP]:
        """Iterate over this BSP's hierarchy in inverse level order.

        .. versionadded:: 8.3
        """
        levels: list[list[BSP]] = []
        next_: list[BSP] = [self]
        while next_:
            levels.append(next_)
            level = next_
            next_ = []
            for node in level:
                next_.extend(node.children)
        while levels:
            yield from levels.pop()

    def contains(self, x: int, y: int) -> bool:
        """Return True if this node contains these coordinates.

        Args:
            x (int): X position to check.
            y (int): Y position to check.

        Returns:
            bool: True if this node contains these coordinates.
                  Otherwise False.
        """
        return self.x <= x < self.x + self.width and self.y <= y < self.y + self.height

    def find_node(self, x: int, y: int) -> BSP | None:
        """Return the deepest node which contains these coordinates.

        Returns:
            BSP object or None.
        """
        if not self.contains(x, y):
            return None
        for child in self.children:
            found = child.find_node(x, y)
            if found:
                return found
        return self


__all__ = ["BSP"]
