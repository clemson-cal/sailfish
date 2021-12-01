from math import pi, sin
from sailfish.setup import Setup, SetupError, parameter


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

    @property
    def domain(self):
        return [0.0, 1.0]

    @property
    def boundary_condition(self):
        return "outflow"


class DensityWave(Setup):
    """
    A sinusoidal variation of the gas density, with possible uniform
    translation. The gas pressure is uniform.
    """

    wavenumber = parameter(1, "wavenumber of the sinusoid")
    amplitude = parameter(0.1, "amplitude of the density variation")
    velocity = parameter(0.0, "speed of the wave")

    def initial_primitive(self, x, primitive):
        k = self.wavenumber * 2 * pi
        a = self.amplitude
        v = self.velocity
        primitive[0] = 1.0 + a * sin(k * x)
        primitive[1] = v
        primitive[2] = 1.0

    @property
    def domain(self):
        return [0.0, 1.0]

    @property
    def boundary_condition(self):
        return "periodic"

    @property
    def end_time(self):
        return 1.0

    def validate(self):
        if self.amplitude >= 1.0:
            raise SetupError("amplitude must be less than 1.0")
