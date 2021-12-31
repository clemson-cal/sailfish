"""
Contains classes for different mesh geometries and dimensionalities.
"""

from typing import NamedTuple
from math import log, log10, pi


class PlanarCartesianMesh(NamedTuple):
    """
    A 1D, planar cartesian mesh with equal grid spacing.
    """

    x0: float = 0.0
    x1: float = 1.0
    num_zones: int = 1000

    def __str__(self):
        return (
            f"<planar cartesian 1d: ({self.x0} -> {self.x1}), {self.num_zones} zones>"
        )

    @property
    def dx(self):
        return (self.x1 - self.x0) / self.num_zones

    def min_spacing(self, time=None):
        return self.dx

    @property
    def shape(self):
        return (self.num_zones,)

    @property
    def num_total_zones(self):
        return self.num_zones * self.num_zones

    def zone_center(self, t, i):
        x0, dx = self.x0, self.dx
        return x0 + (i + 0.5) * dx

    def zone_centers(self, t, i0=0, i1=None):
        return [self.zone_center(t, i) for i in range(i0, i1 or self.shape[0])]

    def faces(self, i0=0, i1=None):
        if i1 is None:
            i1 = self.shape[0]
        x0, dx = self.x0, self.dx
        return [x0 + i * dx for i in range(i0, i1 + 1)]


class LogSphericalMesh(NamedTuple):
    """
    A 1D or 2D mesh with logarithmic radial binning and homologous expansion.

    The comoving coordinates are time-independent; proper distances are
    affected by the scale factor derivative.
    """

    r0: float = 1.0
    r1: float = 10.0
    num_zones_per_decade: int = 1000
    scale_factor_derivative: float = None
    polar_grid: bool = False

    def __str__(self):
        if self.scale_factor_derivative is None:
            motion = f"no expansion"
        else:
            motion = f"homologous with a-dot = {self.scale_factor_derivative:0.2f}"
        return (
            f"<log spherical: ({self.r0} -> {self.r1}), {motion}, "
            f"{self.num_zones_per_decade} zones per decade, "
            f"shape {self.shape}>"
        )

    def min_spacing(self, time=None):
        """
        Return the smallest grid spacing.

        If the time is provided, the result is a proper distance and the scale
        factor and its derivative are taken into account. Otherwise if no time
        is provided the result is the minimum comoving grid spacing.
        """
        r0, r1 = self.faces(0, 1)
        return (r1 - r0) * self.scale_factor(time)

    def scale_factor(self, time):
        """
        Return the scale factor at a given time.

        If the scale factor is not changing, it's always equal to one.
        Otherwise, it's assumed that `a=0` at `t=0`.
        """
        if self.scale_factor_derivative is None:
            return 1.0
        else:
            return self.scale_factor_derivative * time

    @property
    def shape(self):
        if not self.polar_grid:
            return (self.num_radial_zones,)
        else:
            return (self.num_radial_zones, self.num_polar_zones)

    @property
    def num_total_zones(self):
        tot = 1
        for n in self.shape:
            tot *= n
        return tot

    def zone_center(self, t, i):
        """
        Return the proper radial coordinate of the ith zone center.
        """
        r0, k = self.r0, 1.0 / self.num_zones_per_decade
        return r0 * 10 ** ((i + 0.5) * k) * self.scale_factor(t)

    def zone_centers(self, t, i0=0, i1=None):
        return [self.zone_center(t, i) for i in range(i0, i1 or self.shape[0])]

    def faces(self, i0=0, i1=None):
        """
        Return radial face positions for zone indexes in the given range.

        The positions are given in comoving coordinates, i.e. are
        time-independent. The number of faces `i1 - i0 + 1` is one more than
        the number of zones `i1 - i0` in the index range.
        """
        if i1 is None:
            i1 = self.shape[0]
        r0, k = self.r0, 1.0 / self.num_zones_per_decade
        return [r0 * 10 ** (i * k) for i in range(i0, i1 + 1)]

    def cell_coordinates(self, t, i, j):
        """
        Return the 2D (r, theta) zone center proper coordinates at index (i, j).
        """
        r = self.zone_center(t, i)
        q = (j + 0.5) * pi / self.num_polar_zones
        return r, q

    @property
    def num_radial_zones(self):
        return int(log10(self.r1 / self.r0) * self.num_zones_per_decade)

    @property
    def num_polar_zones(self):
        if not self.polar_grid:
            raise ValueError("only defined for a 2D spherical polar mesh")
        return int(self.num_zones_per_decade * pi / log(10))

    @property
    def radial_vertices(self):
        return self.faces()

    @property
    def polar_vertices(self):
        return [j * pi / self.num_polar_zones for j in range(self.num_polar_zones + 1)]


class PlanarCartesian2DMesh(NamedTuple):
    """
    A 2D mesh with rectangular binning.

    The length of the domain is related to height by an aspect ratio. The
    minimum demarcations along an axis is also variable allowing for
    non-square meshing if need be.
    """

    x0: float = 0.0
    y0: float = 0.0
    x1: float = 1.0
    y1: float = 1.0
    ni: int = 1000
    nj: int = 1000

    def __str__(self):
        return f"<planar cartesian 2d: ({self.x0} -> {self.x1}) x ({self.y0} -> {self.y1}), shape {self.shape}>"

    @classmethod
    def centered_square(cls, domain_radius, resolution):
        x0 = -domain_radius
        y0 = -domain_radius
        x1 = +domain_radius
        y1 = +domain_radius
        ni = resolution
        nj = resolution
        dx = 2.0 * domain_radius / resolution
        dy = 2.0 * domain_radius / resolution
        return PlanarCartesian2DMesh(x0, y0, x1, y1, ni, nj)

    @classmethod
    def centered_rectangle(cls, height, resolution, aspect: int):
        if type(aspect) is not int:
            raise ValueError("centered_rectangle requires aspect to be int")
        x0 = -0.5 * height * aspect
        y0 = -0.5 * height
        x1 = +0.5 * height * aspect
        y1 = +0.5 * height
        ni = resolution * aspect
        nj = resolution
        return PlanarCartesian2DMesh(x0, y0, x1, y1, ni, nj)

    @property
    def dx(self):
        return (self.x1 - self.x0) / self.ni

    @property
    def dy(self):
        return (self.y1 - self.y0) / self.nj

    def shape(self):
        return self.ni, self.nj

    def num_total_zones(self):
        return self.ni * self.nj

    def cell_coordinates(self, i, j):
        x = self.x0 + (i + 0.5) * self.dx
        y = self.y0 + (j + 0.5) * self.dy
        return x, y

    def vertex_coordinates(self, i, j):
        x = self.x0 + i * self.dx
        y = self.y0 + j * self.dy
        return x, y

    def sub_mesh(self, di, dj):
        """
        Return a new mesh that is a subset of this one.

        The arguments di and dj are tuples, containing the lower and upper
        index range on this mesh.
        """
        x0, y0 = self.vertex_coordinates(di[0], dj[0])
        x1, y1 = self.vertex_coordinates(di[1], dj[1])
        ni = di[1] - di[0]
        nj = dj[1] - dj[0]
        return PlanarCartesian2DMesh(x0, y0, x1, y1, ni, nj)
