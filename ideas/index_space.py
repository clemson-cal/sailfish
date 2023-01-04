from typing import Callable


def perm_of(a: str, b: str, inverse=False) -> tuple[int]:
    """
    Return a numeric permutation
    """
    if not inverse:
        return tuple(a.index(c) for c in b)
    else:
        return perm_of(b, a)


def three_space_axes(a, extra_axes=0):
    """
    Return a view of the given array that has three space axes
    """
    if len(a.shape) == 1 + extra_axes:
        return a[:, None, None]
    if len(a.shape) == 2 + extra_axes:
        return a[:, :, None]
    if len(a.shape) == 3 + extra_axes:
        return a[:, :, :]


class IndexSpace:
    """
    Encapsulates creation and indexing of arrays representing fields data

    This class helps simplify boilerplate code arising from two issues:

    1. arrays generally have guard zones on their space axes with length > 1
    2. arrays generally have memory layouts different from their logical shape
    """

    def __init__(
        self,
        shape: tuple[int, int, int],
        guard: int = 0,
        layout: str = "fields-last",
    ):
        """
        Initialize an index-space

        The shape here represents the internal region of an array
        """
        self.shape = shape
        self.guard = guard
        self.fields_last = {"fields-last": True, "fields-first": False}[layout]

    def shape_with_guard(self, fields: int = None, vectors: int = None) -> tuple[int]:
        """
        Return the logical array shape for given fields and vectors
        """
        s = tuple(n + (2 * self.guard if n > 1 else 0) for n in self.shape)
        v = (vectors,) if vectors is not None else tuple()
        f = (fields,) if fields is not None else tuple()
        return s + v + f

    def axes_permutation(
        self,
        fields: int = None,
        vectors: int = None,
        inverse=False,
    ) -> tuple[int]:
        """
        Return the permutation from logical to memory if `inverse=False`

        If `inverse=True`, then return the axis permutation going from memory
        to logical.
        """
        if fields and vectors:
            if self.fields_last:
                return perm_of("ijkdq", "dijkq", inverse)
            else:
                return perm_of("ijkdq", "dqijk", inverse)
        elif fields:
            if self.fields_last:
                return perm_of("ijkq", "ijkq", inverse)
            else:
                return perm_of("ijkq", "qijk", inverse)
        elif vectors:
            return perm_of("ijkd", "dijk", inverse)
        else:
            return perm_of("ijk", "ijk", inverse)

    def create(
        self,
        factory: Callable[[tuple[int, int, int]], "NDArray[float]"],
        fields: int = None,
        vectors: int = None,
        data: "NDArray[float]" = None,
    ) -> "NDArray[float]":
        """
        Create a new array using the given factory

        `fields` is a number of fields: the number of elements on the final
        logical array axis. If None, then the array has no fields axis.
        `vectors` is a number of vector components, useful for things like
        arrays of gradients or fluxes. If None, then the array has no vectors
        axis.

        If `data` is not `None` then it must be an array with same logical
        shape as the interior of the created array.
        """
        v = (vectors,) if vectors is not None else tuple()
        f = (fields,) if fields is not None else tuple()
        s = self.shape_with_guard(fields, vectors)
        p = self.axes_permutation(fields, vectors)
        q = self.axes_permutation(fields, vectors, inverse=True)
        arr = factory(tuple(s[a] for a in p)).transpose(q)

        if data is not None:
            data = three_space_axes(data, (fields and 1 or 0) + (vectors and 1 or 0))
            if data.shape == arr[self.interior].shape:
                arr[self.interior] = data
            elif data.shape == arr.shape:
                arr[...] = data
            else:
                raise ValueError(
                    f"given data must have the interior or total shape, got {data.shape}"
                )
        return arr

    def __getitem__(self, code: str) -> tuple[slice]:
        """
        Alias for `self.region(code)`
        """
        return self.region(code)

    def region(self, code: str) -> tuple[slice]:
        """
        Return an nd-slice referencing any of the 3x3x3 regions of an array

        `code` is a three-character string where each character is one of l,
        c, r, referring to the left guard region, interior (center), or right
        guard region of an array. The code is ignored for axes of length 1.
        """
        s = self.shape
        g = self.guard

        def ax_slice(c):
            if c is None:
                return slice(None, None)
            if c == "l":
                return slice(None, +g)
            if c == "c":
                return slice(+g, -g)
            if c == "r":
                return slice(-g, None)

        return tuple(ax_slice(c if s[n] > 1 else None) for n, c in enumerate(code))

    @property
    def interior(self) -> tuple[slice]:
        """
        An nd-slice referencing the array interior (convenience property)
        """
        return self.region("ccc")


if __name__ == "__main__":
    from numpy import zeros

    nprim = 4
    shape = (100, 100, 1)
    space = IndexSpace(shape=shape, guard=2, layout="fields-last")
    prim = space.create(zeros, fields=nprim, vectors=None)
    assert prim.shape == (104, 104, 1, 4)
    assert prim[space.interior].shape == (100, 100, 1, 4)
    assert prim[space["crc"]].shape == (100, 2, 1, 4)
