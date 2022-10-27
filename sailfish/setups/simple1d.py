"""
Validation setups for various 1D solvers
"""

from math import pi, sin
from sailfish.setup_base import SetupBase, SetupError, param
from sailfish.mesh import PlanarCartesianMesh, LogSphericalMesh

__all__ = ["Advection", "Burgers", "DensityWave", "Shocktube", "Wind"]


class Advection(SetupBase):
    """
    Scalar advection, evolution of a smooth wave, using the DG solver.
    """

    def primitive(self, t, x, primitive):
        a = 0.1
        k = 2.0 * pi
        primitive[0] = 1.0 + a * sin(k * x)

    def mesh(self, num_zones):
        return PlanarCartesianMesh(0.0, 1.0, num_zones)

    @property
    def solver(self):
        return "scdg_1d"

    @property
    def physics(self):
        return dict(equation="advection")

    @property
    def boundary_condition(self):
        return "periodic"

    @property
    def default_end_time(self):
        return 1.0


class Burgers(SetupBase):
    """
    Burgers equation, evolution of a smooth wave using the DG solver.
    """

    def primitive(self, t, x, primitive):
        a = 0.1
        k = 2.0 * pi
        primitive[0] = 1.0 + a * sin(k * x)

    def mesh(self, num_zones):
        return PlanarCartesianMesh(0.0, 1.0, num_zones)

    @property
    def solver(self):
        return "scdg_1d"

    @property
    def physics(self):
        return dict(equation="burgers")

    @property
    def boundary_condition(self):
        return "periodic"

    @property
    def default_end_time(self):
        return 1.0


class Shocktube(SetupBase):
    """
    Discontinuous initial data, with uniform density and pressure to either
    side of the discontintuity at x=0.5.
    """

    def primitive(self, t, x, primitive):
        if x < 0.5:
            primitive[0] = 1.0
            primitive[2] = 1.0
        else:
            primitive[0] = 0.1
            primitive[2] = 0.125

    def mesh(self, num_zones):
        return PlanarCartesianMesh(0.0, 1.0, num_zones)

    @property
    def solver(self):
        return "srhd_1d"

    @property
    def boundary_condition(self):
        return "outflow"

    @property
    def default_end_time(self):
        return 0.25


class DensityWave(SetupBase):
    """
    A sinusoidal variation of the gas density, with possible uniform
    translation. The gas pressure is uniform.
    """

    wavenumber = param(1, "wavenumber of the sinusoid")
    amplitude = param(0.1, "amplitude of the density variation")
    velocity = param(0.0, "speed of the wave (gamma-beta if relativistic)")

    def primitive(self, t, x, primitive):
        k = self.wavenumber * 2.0 * pi
        a = self.amplitude
        u = self.velocity

        primitive[0] = 1.0 + a * sin(k * x)
        primitive[1] = u
        primitive[2] = 1.0

    def mesh(self, num_zones):
        return PlanarCartesianMesh(0.0, 1.0, num_zones)

    @property
    def solver(self):
        return "srhd_1d"

    @property
    def boundary_condition(self):
        return "periodic"

    @property
    def default_end_time(self):
        return 1.0

    def validate(self):
        if self.amplitude >= 1.0:
            raise SetupError("amplitude must be less than 1.0")


class Wind(SetupBase):
    """
    A cold, spherically symmetric relativistic wind.
    """

    velocity = param(1.0, "velocity of the wind (gamma-beta if relativistic)")

    def primitive(self, t, r, primitive):
        primitive[0] = 1.0 / r**2
        primitive[1] = self.velocity
        primitive[2] = 1e-4 * primitive[0] ** (4 / 3)

    def mesh(self, num_zones_per_decade):
        return LogSphericalMesh(1.0, 10.0, num_zones_per_decade)

    @property
    def solver(self):
        return "srhd_1d"

    @property
    def boundary_condition(self):
        return "inflow", "outflow"

    @property
    def default_end_time(self):
        return 1.0

    def validate(self):
        if self.velocity < 0.0:
            raise SetupError("velocity must be non-negative")
