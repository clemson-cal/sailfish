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

    @property
    def dx(self):
        return (self.x1 - self.x0) / self.num_zones

    def min_spacing(self, time=None):
        return self.dx

    @property
    def shape(self):
        return (self.num_zones,)

    def faces(self, i0, i1):
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

    def min_spacing(self, time=None):
        """
        Returns the smallest grid spacing.

        If the time is provided, the result is a proper distance and the scale
        factor and its derivative are taken into account. Otherwise if no time
        is provided the result is the minimum comoving grid spacing.
        """
        r0, r1 = self.faces(0, 0)
        return (r1 - r0) * self.scale_factor(time)

    def scale_factor(self, time):
        """
        Returns the scale factor at a given time.

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

    def faces(self, i0, i1):
        """
        Returns radial face positions for zone indexes in the given range.

        The positions are given in comoving coordinates, i.e. are
        time-independent. The number of faces `i1 - i0 + 1` is one more than
        the number of zones `i1 - i0` in the index range.
        """
        r0, k = self.r0, 1.0 / self.num_zones_per_decade
        return [r0 * 10 ** (i * k) for i in range(i0, i1 + 1)]
