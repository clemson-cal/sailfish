"""
A cylindrical shock tube test for cbdgam_2d
"""

from sailfish.setup import Setup, param
from sailfish.mesh import PlanarCartesian2DMesh


class CylindricalExplosion(Setup):
    """
    A cylindrical explosion in 2D planar geometry; adiabatic hydro.

     This problem is useful for testing bare-bones setups with minimal physics.
     A circular region of high density and pressure is initiated at the center
     of a square domain. The gas has a gamma-law equation of state with adiabatic
     index 5.0 / 3.0

    """

    def primitive(self, t, coords, primitive):
        x, y = coords

        if x * x + y * y < 0.25**2:
            primitive[0] = 1.0
            primitive[3] = 1.0
        else:
            primitive[0] = 0.1
            primitive[3] = 0.1

        primitive[1] = 0.0
        primitive[2] = 0.0

    def mesh(self, resolution):
        return PlanarCartesian2DMesh.centered_square(1.0, resolution)

    @property
    def physics(self):
        return dict(alpha = 0.0,
                   mass_model1=0,
                   mass_model2=0)

    @property
    def solver(self):
        return "cbdgam_2d"

    @property
    def boundary_condition(self):
        return "outflow"

    @property
    def default_end_time(self):
        return 2.0
