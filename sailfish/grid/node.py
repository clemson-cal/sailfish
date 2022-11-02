"""
Provides a node type for a self-similar n-tree.
"""


class NodeList:
    """
    A restricted container to hold n-tree nodes.
    """

    def __init__(self, ratio, children):
        children = list(children)
        if len(children) != ratio:
            raise ValueError(f"need exactly {ratio} child nodes")
        for node in children:
            if not isinstance(node, Node):
                raise ValueError("must be of a subtype of Node")
        self._children = children

    def __getitem__(self, i):
        return self._children[i]

    def __setitem__(self, i, node):
        if not isinstance(node, Node):
            raise ValueError("must be of a subtype of Node")
        if node.ratio != len(self._children):
            raise ValueError("node has the wrong ratio")
        self._children[i] = node

    def __iter__(self):
        yield from self._children


class Node:
    """
    Represents a node in a self-similar n-tree.

    Nodes are allowed to have a value, a list of children, both, or neither. The
    branching ratio (number of children), is uniform across the tree. Tree nodes
    are mutable. Nodes do not have a parent pointer, so in principle a node
    could be the child of multiple parent nodes. Traversal is done pre-order and
    with memory-efficient (stackful) generator routines. This makes it efficient
    to do map and zip operations on the values of distinct trees with the same
    topology. Trees can be reconstructed from their sequence representation.
    """

    def __init__(self, value=None, children=None, items=None):
        """
        Construct a new node.

        Zero or one of the keyword arguments `value`, `children`, `items` may
        be specified. If zero arguments are given then the node has no
        children and `value` of `None`. If value is given, the node has no
        children and the given value. If `children` (a sequence containing
        exactly 4 Node instances) is given, the node has value `None` and the
        given child nodes. If `items` is given (a well-formed sequence of
        `(index, value)` pairs like that returned from `Node.items`), the node
        is reconstructed from that sequence.
        """
        if bool(value) + bool(children) + bool(items) > 1:
            raise ValueError("at most one of value, children, items may be specified")
        elif value is not None:
            self._value = value
            self._children = None
        elif children:
            self._value = None
            self._children = NodeList(self.ratio, children)
        elif items:
            root = self.__class__.from_items(items)
            self._value = root._value
            self._children = root._children
        else:
            self._value = None
            self._children = None

    @classmethod
    def _from_first_rest(cls, first, rest):
        index, value = first
        node = cls(value=value)
        try:
            iv = next(rest)
        except StopIteration:
            return node, None
        if len(iv[0]) > len(index):
            children = list()
            for _ in range(cls.ratio):
                child, iv = cls._from_first_rest(iv, rest)
                children.append(child)
            node.children = children
        return node, iv

    @classmethod
    def from_items(cls, items):
        """
        Return a node constructed from a well-formed `(index, value)` sequence.

        This makes it possible to reconstruct a node from a sequence of its
        items, for example:

        ``node == Node.from_items(node.items()) # True``
        """
        return cls._from_first_rest(next(items), items)[0]

    @property
    def children(self):
        """
        Return the children list, or an empty list if there are no children.
        """
        return self._children or list()

    @children.setter
    def children(self, children):
        """
        Assign a sequence of children to this node.
        """
        self._children = NodeList(self.ratio, children)

    @property
    def value(self):
        """
        Return the value of this node.
        """
        return self._value

    @value.setter
    def value(self, value):
        """
        Assign a value to this node.
        """
        self._value = value

    def nodes(self):
        """
        Pre-order traversal of the nodes at and below this one.
        """
        yield self
        for c in self.children:
            yield from c.nodes()

    def values(self):
        """
        Pre-order traversal of the values at and below this node.
        """
        yield from map(lambda node: node.value, self.nodes())

    def __eq__(self, other):
        for i1, i2 in zip(self.items(), other.items()):
            if i1 != i2:
                return False
        return True

    def __getitem__(self, index):
        """
        Child item access, index may be either int or tuple.
        """
        if type(index) == int:
            return self.children[index]
        else:
            return self.at(index)

    def __iter__(self):
        """
        Pre-order traversal of the values at and below this node.
        """
        return self.values()

    def __len__(self):
        """
        Return the number of values at or below this node.
        """
        return 1 + sum(len(c) for c in self.children)

    @property
    def depth(self):
        """
        Return the maximum depth of any node below this one.
        """
        if self._children is None:
            return 1
        else:
            return max(map(lambda n: 1 + n.depth, self.children))

    def indexes(self, parent=tuple()):
        """
        Return an iterator of index values reflecting this tree's topology.
        """
        yield parent
        for i, c in enumerate(self.children):
            yield from c.indexes((*parent, i))

    def items(self):
        """
        Return a zipped sequence of indexes and values.
        """
        return zip(self.indexes(), self.values())

    def at(self, index):
        """
        Return the node at the given index below this one.

        Raises `IndexError` if the node does not exist.
        """
        return self[index[0]].at(index[1:]) if index else self

    def require(self, index):
        """
        Create if necessary, and return the node at the given index.

        This function creates all intermediate nodes that are needed; the
        created nodes will have `value = None`.
        """
        if self._children is None:
            self.children = map(lambda _: self.__class__(), range(self.ratio))
        if len(index) == 1:
            return self[index[0]]
        else:
            return self[index[0]].require(index[1:])


class Node2(Node):
    ratio = classmethod(property(lambda _: 2))


class Node4(Node):
    ratio = classmethod(property(lambda _: 4))


class Node8(Node):
    ratio = classmethod(property(lambda _: 8))


class Index:
    """
    A canonical representation of a node index in an n-tree or rank-d grid.

    A topological index is an integer sequence identifying a node in an n-tree.

    A geometrical index of rank d is a pair: (level, d-tuple) where level is
    the depth of the node (length of the topological index) and the d-tuple is
    the logically Cartesian index (i, j, k) in d-dimensional space; i, j, k
    are each in the range [0, 2^level - 1].
    """

    @staticmethod
    def topo_to_bit(t: tuple, rank: int, d: int, l: int):
        return bool(t & 1 << d)

    @staticmethod
    def from_topological(t: tuple, rank: int):
        s = Index()
        s._data = tuple(tuple(bool(t & 1 << d) for t in t) for d in range(rank))
        return s

    @staticmethod
    def from_geometrical(g: tuple, level: int):
        s = Index()
        s._data = tuple(tuple(bool(g & 1 << l) for l in range(level)) for g in g)
        return s

    @staticmethod
    def top_to_geo(t: tuple, rank: int):
        return tuple(
            sum(bool(t & 1 << d) << l for l, t in enumerate(t)) for d in range(rank)
        )

    @staticmethod
    def geo_to_top(g: tuple, level: int):
        return tuple(
            sum(bool(g & 1 << l) << d for d, g in enumerate(g)) for l in range(level)
        )

    @property
    def level(self):
        return len(self._data[0])

    @property
    def rank(self):
        return len(self._data)

    @property
    def geometrical(self):
        return tuple(
            sum(self._data[d][l] << l for l in range(self.level))
            for d in range(self.rank)
        )

    @property
    def topological(self):
        return tuple(
            sum(self._data[d][l] << d for d in range(self.rank))
            for l in range(self.level)
        )


class MultigridArray:
    """
    A multi-level grid object, for block-structured (quad-tree, oct-tree) grids.

    It is organized as a tree where each node has one array (this array is
    called a patch). The patches have uniform shape. The leading 3 array axes
    are spatial indexes, and any remaining array axes represent data fields.
    Data may be point-like (zero-form), edge-like (1-form), face-like (2-form),
    or cell-like (volume form), and those fields may be intrinsic (e.g. electric
    field or mass density) or extrinsic (e.g. voltage or mass).

    There may be overlap (guard zones) between the arrays at adjacent nodes, and
    the overlap may go in different directions.

    Geometry object:

    - mapping from multi-level patch index (level, i, j, k) -> (x, y, z)
    - might also provide the "metric", i.e. the 1-forms, 2-forms, 3-forms

    Responsibilities:

    - store grid data at multiple levels
    - map geometry objects onto grid patches
    - more generally, map between MultigridArray instances with same topology
    - copying guard zones data between neighboring patches =>
    - doing prolongation / restriction of data
    - do flux corrections, or more generally data reconciliation on faces,
      edges, and points.
    """

    def __init__(
        self,
        blocks_shape: tuple,
        fields_shape: tuple,
        topology: Node = None,
    ):
        if len(blocks_shape) != 3:
            raise ValueError("blocks_shape must have length at least 3")
        self._blocks_shape = tuple(blocks_shape)
        self._fields_shape = tuple(fields_shape)
        self.topology = topology

    @property
    def blocks_shape(self):
        return self._blocks_shape

    @property
    def fields_shape(self):
        return self._fields_shape

    @property
    def rank(self):
        return sum(map(lambda n: n > 1, self.blocks_shape))

    @property
    def topology(self):
        return self._root_node

    @topology.setter
    def topology(self, new_topology: Node):
        if new_topology is not None:
            if new_topology.ratio != 1 << self.rank:
                raise ValueError("topology must be a node with ratio = 2^d")
        self._root_node = new_topology

    def patch_indexes(self):
        """
        Return an iterator over the geometrical indexes of patches.
        """
        return map(
            lambda i: Index.from_topological(i, self.rank).geometrical,
            self.topology.indexes(),
        )


if __name__ == "__main__":
    tree = Node4()
    tree.children = map(Node4, range(4))
    assert len(tree) == 5
    tree.children[0] = Node4(children=map(Node4, "WXYZ"))
    assert len(tree) == 9
    tree.children[0].children[2] = Node4(children=map(Node4, "abcd"))
    assert len(tree) == 13
    node = tree.require((1, 1, 1, 1, 1))
    assert tree == tree
    assert len(list(tree.items())) == len(tree)
    assert Node4(items=tree.items()) == tree
    assert tree.depth == 6

    for index in tree.indexes():
        index1 = Index.from_topological(index, 2)
        index2 = Index.from_geometrical(index1.geometrical, index1.level)
        assert index1.geometrical == index2.geometrical
        assert index1.topological == index2.topological
        assert index1.geometrical == Index.top_to_geo(index1.topological, 2)
        assert index1.topological == Index.geo_to_top(index1.geometrical, index1.level)
