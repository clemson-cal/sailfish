from dataclasses import replace
from schema import schema
from numpy import linspace, meshgrid


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

    extent_i: the extent of the coordinate box on the i-axis
    extent_j: the extent of the coordinate box on the j-axis
    extent_k: the extent of the coordinate box on the k-axis
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

    def cell_centers(self):
        """
        Return an array or tuple of arrays of the cell-center coordinates
        """
        if self.dimensionality == 1:
            ni = self.num_zones[0]
            x0, x1 = self.extent_i
            xv = linspace(x0, x1, ni + 1)
            xc = 0.5 * (xv[1:] + xv[:-1])
            return xc

        if self.dimensionality == 2:
            ni, nj = self.num_zones[0:2]
            x0, x1 = self.extent_i
            y0, y1 = self.extent_j
            xv = linspace(x0, x1, ni + 1)
            yv = linspace(y0, y1, nj + 1)
            xc = 0.5 * (xv[1:] + xv[:-1])
            yc = 0.5 * (yv[1:] + yv[:-1])
            return meshgrid(xc, yc, indexing="ij")

        if self.dimensionality == 3:
            ni, nj, nk = self.num_zones
            x0, x1 = self.extent_i
            y0, y1 = self.extent_j
            z0, z1 = self.extent_k
            xv = linspace(x0, x1, ni + 1)
            yv = linspace(y0, y1, nj + 1)
            zv = linspace(z0, z1, nk + 1)
            xc = 0.5 * (xv[1:] + xv[:-1])
            yc = 0.5 * (yv[1:] + yv[:-1])
            zc = 0.5 * (zv[1:] + zv[:-1])
            return meshgrid(xc, yc, zc, indexing="ij")

    def decompose(self, num_parts: int):
        """
        Decompose a 1d coordinate box into a sequence of non-overlapping boxes
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
