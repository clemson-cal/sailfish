class NodeList:
    """
    A restricted container to hold n-tree nodes.
    """

    def __init__(self, children):
        children = list(children)
        if len(children) != 4:
            raise ValueError("need exactly 4 child nodes")
        for node in children:
            if type(node) != Node:
                raise ValueError("must be of type Node")
        self._children = children

    def __getitem__(self, i):
        return self._children[i]

    def __setitem__(self, i, node):
        if type(node) != Node:
            raise ValueError("must be of type Node")
        self._children[i] = node

    def __iter__(self):
        yield from self._children


class Node:
    """
    Represents a node in a self-similar n-tree.

    Nodes are allowed to have a value, a list of children, both, or neither.
    The branching ratio is the number of children, and it is uniform across
    the tree. Tree nodes are mutable. Nodes do not have a parent pointer, so
    in principle a node could be the child of multiple parent nodes. Traversal
    is done pre-order and with memory-efficient (stackful) generator routines.
    This makes it efficient to do map and zip operations on the values of
    distinct trees with the same topology. Trees can be reconstructed from
    their sequence representation.

    Right now, the branching ratio is hard-coded to 4, but this will change
    soon.
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
        elif value:
            self._value = value
            self._children = None
        elif children:
            self._value = None
            self._children = NodeList(children)
        elif items:
            root = Node.from_items(items)
            self._value = root._value
            self._children = root._children
        else:
            self._value = None
            self._children = None

    @classmethod
    def _from_first_rest(cls, first, rest):
        index, value = first
        node = Node(value=value)
        try:
            iv = next(rest)
        except StopIteration:
            return node, None
        if len(iv[0]) > len(index):
            children = list()
            for _ in range(4):
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
        self._children = NodeList(children)

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
            self.children = map(lambda _: Node(), range(4))
        if len(index) == 1:
            return self[index[0]]
        else:
            return self[index[0]].require(index[1:])


if __name__ == "__main__":
    tree = Node()
    tree.children = map(Node, range(4))
    assert len(tree) == 5
    tree.children[0] = Node(children=map(Node, "WXYZ"))
    assert len(tree) == 9
    tree.children[0].children[2] = Node(children=map(Node, "abcd"))
    assert len(tree) == 13
    node = tree.require((1, 1, 1, 1, 1))
    assert tree == tree
    assert len(list(tree.items())) == len(tree)
    assert Node(items=tree.items()) == tree
    assert tree.depth == 6
