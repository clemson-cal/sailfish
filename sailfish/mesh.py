"""
Contains classes for different mesh geometries and dimensionalities.
"""

from typing import NamedTuple
from math import log10


class PlanarCartesianMesh(NamedTuple):
    """
    A 1D, planar cartesian mesh with equal grid spacing.
    """

    x0: float = 0.0
    x1: float = 1.0
    num_zones: int = 1000

    def __str__(self):
        return f"planar cartesian ({self.x0} -> {self.x1})"

    @property
    def dx(self):
        return (self.x1 - self.x0) / self.num_zones

    def min_spacing(self, time=None):
        return self.dx

    @property
    def shape(self):
        return (self.num_zones,)

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
    A 1D mesh with logarithmic radial binning and homologous expansion.

    The comoving coordinates are time-independent; proper distances are
    affected by the scale factor derivative.
    """

    r0: float = 1.0
    r1: float = 10.0
    num_zones_per_decade: int = 1000
    scale_factor_derivative: float = None

    def __str__(self):
        if self.scale_factor_derivative is None:
            motion = f"fixed mesh"
        else:
            motion = f"homologous with a-dot = {self.scale_factor_derivative:0.2f}"

        return (
            f"log spherical ({self.r0} -> {self.r1}) {motion}, "
            f"{self.num_zones_per_decade} zones per decade"
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
        return (int(log10(self.r1 / self.r0) * self.num_zones_per_decade),)

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
