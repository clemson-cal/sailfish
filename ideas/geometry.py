from dataclasses import replace
from schema import schema
from numpy import array, linspace, meshgrid, sqrt, sin, cos, pi
from numpy.typing import NDArray


def partition(elements: int, num_parts: int):
    """
    Equitably divide the given number of elements into `num_parts` partitions.

    The sum of the partitions is `elements`. The number of partitions must be
    less than or equal to the number of elements.
    """
    n = elements // num_parts
    r = elements % num_parts

    for i in range(num_parts):
        yield n + (1 if i < r else 0)


def subdivide(interval: tuple[int, int], num_parts: int):
    """
    Divide an interval into non-overlapping contiguous sub-intervals.
    """
    a, b = interval

    for n in partition(b - a, num_parts):
        yield a, a + n
        a += n


@schema
class CoordinateBox:
    """
    Domain with uniformly spaced grid cells

    A coordinate box is a (hyper-)rectangular region in a coordinate chart. It
    is agnostic to the geometry of the chart, e.g. it could be a box in x, y,
    z space or r-theta-phi space.

    Fields
    ------

    extent_i:  extent of the coordinate box on the i-axis
    extent_j:  extent of the coordinate box on the j-axis
    extent_k:  extent of the coordinate box on the k-axis
    num_zones: number of zones on each axis
    """

    extent_i: tuple[float, float] = (0.0, 1.0)
    extent_j: tuple[float, float] = (0.0, 1.0)
    extent_k: tuple[float, float] = (0.0, 1.0)
    num_zones: tuple[int, int, int] = (128, 1, 1)

    @property
    def dimensionality(self):
        """
        number of fleshed-out spatial axes
        """
        return sum(n > 1 for n in self.num_zones)

    @property
    def grid_spacing(self):
        """
        spacing between zones on each axis
        """
        extent = (self.extent_i, self.extent_j, self.extent_k)
        return tuple((e[1] - e[0]) / n for e, n in zip(extent, self.num_zones))

    def extent(self, axis: int) -> tuple[float, float]:
        if axis == 0:
            return self.extent_i
        if axis == 1:
            return self.extent_j
        if axis == 2:
            return self.extent_k

    def _vertices(self, axis: int, drop_final=False) -> NDArray[float]:
        nc = self.num_zones[axis]
        x0, x1 = self.extent(axis)
        if drop_final:
            return linspace(x0, x1, nc + 1)[:-1]
        else:
            return linspace(x0, x1, nc + 1)

    def _centers(self, axis: int) -> NDArray[float]:
        xv = self._vertices(axis)
        return 0.5 * (xv[1:] + xv[:-1])

    def cell_centers(self, dim: int = None) -> NDArray[float]:
        """
        Return an array or tuple of arrays of the cell-center coordinates
        """
        dim = dim or self.dimensionality

        if dim == 1:
            return self._centers(axis=0)

        if dim == 2:
            xc = self._centers(axis=0)
            yc = self._centers(axis=1)
            return meshgrid(xc, yc, indexing="ij")

        if dim == 3:
            xc = self._centers(axis=0)
            yc = self._centers(axis=1)
            zc = self._centers(axis=2)
            return meshgrid(xc, yc, zc, indexing="ij")

    def cell_vertices(self, dim: int = None, drop_final=False) -> NDArray[float]:
        """
        Return an array or tuple of arrays of the coordinates of cell vertices

        If `drop_final` is `True` then the resulting arrays have the same
        shape as the result of `cell_centers` and the coordinates correspond
        to the lower-left corners of the zones.
        """
        dim = dim or self.dimensionality

        if dim == 1:
            return self._vertices(0, drop_final)

        if dim == 2:
            xv = self._vertices(0, drop_final)
            yv = self._vertices(1, drop_final)
            return meshgrid(xv, yv, indexing="ij")

        if dim == 3:
            xv = self._vertices(0, drop_final)
            yv = self._vertices(1, drop_final)
            zv = self._vertices(2, drop_final)
            return meshgrid(xv, yv, zv, indexing="ij")

    def decompose(self, num_parts: int):
        """
        Decompose a 1d coordinate box into a sequence of non-overlapping boxes

        The decomposition is done on the box's i-axis.
        """
        dx = self.grid_spacing[0]

        for i0, i1 in subdivide((0, self.num_zones[0]), num_parts):
            x0 = self.extent_i[0] + dx * i0
            x1 = self.extent_i[0] + dx * i1
            num_zones = (i1 - i0, *self.num_zones[1:])
            yield (i0, i1), replace(self, extent_i=(x0, x1), num_zones=num_zones)

    def extend(self, count: int):
        """
        Return a coordinate box extended on both sides of all non-unit axes
        """
        extent = [self.extent_i, self.extent_j, self.extent_k]
        num_zones = [1, 1, 1]

        for a in range(3):
            ni = self.num_zones[a]
            dx = self.grid_spacing[a]
            if ni > 1:
                x0 = extent[a][0] - count * dx
                x1 = extent[a][1] + count * dx
                num_zones[a] = ni + 2 * count
            else:
                x0 = extent[a][0]
                x1 = extent[a][1]
                num_zones[a] = ni
            extent[a] = (x0, x1)

        return replace(
            self,
            extent_i=extent[0],
            extent_j=extent[1],
            extent_k=extent[2],
            num_zones=tuple(num_zones),
        )

    def trim(self, count: int):
        """
        Return a coordinate box trimmed on both sides of all non-unit axes
        """
        return self.extend(-count)


class CartesianCoordinates:
    needs_geometrical_source_terms = False

    def face_areas(self, box: CoordinateBox) -> NDArray[float]:
        """
        Return an array of the face areas for the given box

        The shape of the returned array is (ni, nj, nk, dim) where dim is the
        box dimensionality.
        """
        x, y, z = box.cell_vertices(dim=3)
        dx = x[+1:, :-1, :-1] - x[:-1, :-1, :-1]
        dy = y[:-1, +1:, :-1] - y[:-1, :-1, :-1]
        dz = z[:-1, :-1, +1:] - z[:-1, :-1, :-1]

        tr = lambda arr: arr.transpose(1, 2, 3, 0)

        if box.dimensionality == 1:
            return tr(array([dy * dz]))
        if box.dimensionality == 2:
            return tr(array([dy * dz, dz * dx]))
        if box.dimensionality == 3:
            return tr(array([dy * dz, dz * dx, dx * dy]))

    def cell_volumes(self, box: CoordinateBox) -> NDArray[float]:
        """
        Return an array of the cell volume data for the given coordinate box

        The shape of the returned array is (ni, nj, nk).
        """
        x, y, z = box.cell_vertices(dim=3)
        dx = x[+1:, :-1, :-1] - x[:-1, :-1, :-1]
        dy = y[:-1, +1:, :-1] - y[:-1, :-1, :-1]
        dz = z[:-1, :-1, +1:] - z[:-1, :-1, :-1]
        return dx * dy * dz

    def cell_vertices(self, box: CoordinateBox) -> NDArray[float]:
        tr = lambda arr: arr.transpose(1, 2, 3, 0)
        x, y, z = box.cell_vertices(dim=3, drop_final=True)

        if box.dimensionality == 1:
            return tr(array([x]))
        if box.dimensionality == 2:
            return tr(array([x, y]))
        if box.dimensionality == 3:
            return tr(array([x, y, z]))


class SphericalPolarCoordinates:
    needs_geometrical_source_terms = True

    def _meridian(self, r0, r1, q0, q1):
        R0 = r0 * sin(q0)
        R1 = r1 * sin(q1)
        z0 = r0 * cos(q0)
        z1 = r1 * cos(q1)
        dR = R1 - R0
        dz = z1 - z0
        return pi * (R0 + R1) * sqrt(dR * dR + dz * dz)

    def face_areas(self, box: CoordinateBox) -> NDArray[float]:
        """
        Return an array of the face areas for the given box

        The shape of the returned array is (ni, nj, nk, dim) where dim is the
        box dimensionality.
        """
        tr = lambda arr: arr.transpose(1, 2, 3, 0)
        r, q, f = box.cell_vertices(dim=3)
        r0 = r[:-1, :-1, :-1]
        r1 = r[+1:, :-1, :-1]
        q0 = q[:-1, :-1, :-1]
        q1 = q[:-1, +1:, :-1]

        if box.dimensionality == 1:
            da_i = self._meridian(r0, r0, q0, q1)
            return tr(array([da_i]))
        if box.dimensionality == 2:
            da_i = self._meridian(r0, r0, q0, q1)
            da_j = self._meridian(r0, r1, q0, q0)
            return tr(array([da_i, da_j]))
        if box.dimensionality == 3:
            raise NotImplementedError

    def cell_volumes(self, box: CoordinateBox) -> NDArray[float]:
        """
        Return an array of the cell volume data for the given coordinate box

        The shape of the returned array is (ni, nj, nk).
        """
        if box.dimensionality == 3:
            raise NotImplementedError

        r, q, f = box.cell_vertices(dim=3)
        r0 = r[:-1, :-1, :-1]
        r1 = r[+1:, :-1, :-1]
        q0 = q[:-1, :-1, :-1]
        q1 = q[:-1, +1:, :-1]

        return -(r1**3 - r0**3) * (cos(q1) - cos(q0)) * 2.0 * pi / 3.0

    def cell_vertices(self, box: CoordinateBox) -> NDArray[float]:
        tr = lambda arr: arr.transpose(1, 2, 3, 0)
        r, q, f = box.cell_vertices(dim=3, drop_final=True)

        if box.dimensionality == 1:
            return tr(array([r]))
        if box.dimensionality == 2:
            return tr(array([r, q]))
        if box.dimensionality == 3:
            return tr(array([r, q, f]))
