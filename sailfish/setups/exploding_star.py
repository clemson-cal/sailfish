"""
Setup for a thermal bomb set off inside an 18 solar mass progenitor.
"""

from sailfish.setup import Setup, param
from sailfish.mesh import LogSphericalMesh


class ExplodingStar(Setup):
    """
    A star exploding.

    The model is based on Duffell & MacFadyen (2015). Code was adapted from
    Marcus Dupont's setup in Simbi.
    """

    escale = param(1.0, "energy scale, normalized to 10^51 erg")
    ascale = param(0.1, "ambient medium density, normalized to A* = 1")
    r_inner = param(0.005, "inner radius")
    r_outer = param(10.0, "outer radius")

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

        if r < self.r_inner * 1.5:
            primitive[2] = primitive[0] * 10.0
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
        return "outflow"

    @property
    def default_end_time(self):
        return 1.0
