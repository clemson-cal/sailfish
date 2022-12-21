from schema import schema
from numpy import linspace, meshgrid


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
