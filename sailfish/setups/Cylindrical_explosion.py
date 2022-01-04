"""
Setup for a thermal bomb set off inside an 18 solar mass progenitor.
"""

from sailfish.setup import Setup, param
from sailfish.mesh import PlanarCartesian2DMesh


class CylindricalExplosion(Setup):
    """
    A cylindrical explosion in 2D planar geometry; isothermal hydro.

     This problem is useful for testing bare-bones setups with minimal physics.
     A circular region of high density and pressure is initiated at the center
     of a square domain. The gas has isothermal equation of state with global
     sound speed `cs=1`.
    
    """

    def primitive(self, t, coords, primitive):

        x, y = coords

        if (((x * x + y * y)**0.5) < 0.25): 
            primitive[0] = 1.0;
        else: 
            primitive[0] = 0.1;
        
        primitive[1] = 0.0;
        primitive[2] = 0.0;

    def mesh(self, resolution):
        return PlanarCartesian2DMesh.centered_square(1.0, resolution)

    @property
    def physics(self):
        return dict(sound_speed_squared = 1.0, viscosity_coefficient = 0.01)

    @property
    def solver(self):
        return "cbdiso_2d"

    @property
    def boundary_condition(self):
        return "outflow"

    @property
    def default_end_time(self):
        return 2.0
