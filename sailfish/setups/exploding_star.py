"""
Setup for a thermal bomb set off inside an 18 solar mass progenitor.
"""

from math import exp
from sailfish.setup_base import SetupBase, param
from sailfish.mesh import LogSphericalMesh


class ExplodingStar(SetupBase):
    """
    A star exploding.

    The model is based on Duffell & MacFadyen (2015). Code was adapted from
    Marcus Dupont's setup in Simbi.
    """

    r_inner = param(0.005, "inner radius")
    r_outer = param(5.000, "outer radius")
    r_shell = param(0.020, "radius from which a fast shell is launched")

    def primitive(self, t, r, primitive):
        """
        Approximate density profile of a MESA star.

        Numerical values are from Table 1 of DM15.
        """

        k1 = 3.24  # first break slope
        k2 = 2.57  # second break slope
        n = 16.7  # atmosphere cutoff slope
        r1 = 0.0017  # first break radius
        r2 = 0.0125  # second break radius
        r3 = 0.65  # outer radius
        rho_c = 1.0  # central density
        rho_w = 1e-9 / 3e7  # wind density

        core = max(1 - r / r3, 0) ** n / (1 + (r / r1) ** k1 / (1 + (r / r2) ** k2))
        wind = (r / r3) ** -2.0

        primitive[0] = rho_c * core + rho_w * wind
        primitive[1] = 0.0
        primitive[2] = primitive[0] * 1e-6

        if r < self.r_shell:
            primitive[2] = primitive[0] * 100.0
        else:
            primitive[2] = primitive[0] * 1e-6

    def mesh(self, num_zones_per_decade):
        return LogSphericalMesh(
            r0=self.r_inner,
            r1=self.r_outer,
            num_zones_per_decade=num_zones_per_decade,
        )

    @property
    def solver(self):
        return "srhd_1d"

    @property
    def boundary_condition(self):
        return "reflect", "outflow"

    @property
    def default_end_time(self):
        return 1.0
