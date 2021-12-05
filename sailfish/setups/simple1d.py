from math import pi, sin
from sailfish.setup import Setup, SetupError, param
from sailfish.mesh import PlanarCartesianMesh, LogSphericalMesh

__all__ = ["Shocktube", "DensityWave", "Wind"]


class Shocktube(Setup):
    """
    Discontinuous initial data, with uniform density and pressure to either
    side of the discontintuity at x=0.5.
    """

    def initial_primitive(self, x, primitive):
        if x < 0.5:
            primitive[0] = 1.0
            primitive[2] = 1.0
        else:
            primitive[0] = 0.1
            primitive[2] = 0.125

    def mesh(self, num_zones):
        return PlanarCartesianMesh(0.0, 1.0, num_zones)

    @property
    def boundary_condition(self):
        return "outflow"

    @property
    def default_end_time(self):
        return 0.25


class DensityWave(Setup):
    """
    A sinusoidal variation of the gas density, with possible uniform
    translation. The gas pressure is uniform.
    """

    wavenumber = param(1, "wavenumber of the sinusoid")
    amplitude = param(0.1, "amplitude of the density variation")
    velocity = param(0.0, "speed of the wave")

    def initial_primitive(self, x, primitive):
        k = self.wavenumber * 2 * pi
        a = self.amplitude
        v = self.velocity
        primitive[0] = 1.0 + a * sin(k * x)
        primitive[1] = v
        primitive[2] = 1.0

    def mesh(self, num_zones):
        return PlanarCartesianMesh(0.0, 1.0, num_zones)

    @property
    def boundary_condition(self):
        return "periodic"

    @property
    def default_end_time(self):
        return 1.0

    def validate(self):
        if self.amplitude >= 1.0:
            raise SetupError("amplitude must be less than 1.0")


class Wind(Setup):
    """
    A cold, spherically symmetric relativistic wind.
    """

    gamma_beta = param(1.0, "four-velocity of the wind")

    def initial_primitive(self, r, primitive):
        primitive[0] = 1.0 / r ** 2
        primitive[1] = self.gamma_beta
        primitive[2] = 1e-4

    def mesh(self, num_zones_per_decade):
        return LogSphericalMesh(1.0, 10.0, num_zones_per_decade)

    @property
    def boundary_condition(self):
        return "outflow"

    @property
    def default_end_time(self):
        return 1.0

    def validate(self):
        if self.gamma_beta < 0.0:
            raise SetupError("gamma_beta must be non-negative")
