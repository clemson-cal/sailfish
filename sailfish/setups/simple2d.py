"""
Validation setups for various 2D solvers
"""

from sailfish.mesh import LogSphericalMesh, PlanarCartesian2DMesh
from sailfish.physics.circumbinary import EquationOfState
from sailfish.setup import Setup, param

__all__ = ["UniformPolar", "CylindricalExplosion"]


class UniformPolar(Setup):
    """
    Tests the srhd_2d solver geometrical source terms.
    """

    def primitive(self, t, _, primitive):
        primitive[0] = 1.0
        primitive[1] = 0.0
        primitive[2] = 0.0
        primitive[3] = 1.0

    def mesh(self, num_zones_per_decade):
        return LogSphericalMesh(1.0, 50.0, num_zones_per_decade, polar_grid=True)

    @property
    def solver(self):
        return "srhd_2d"

    @property
    def boundary_condition(self):
        return "outflow"

    @property
    def default_end_time(self):
        return 1.0


class CylindricalExplosion(Setup):
    """
    A cylindrical explosion in 2D planar geometry; isothermal or gamma-law.

    This problem is useful for testing bare-bones setups with minimal physics.
    A circular region of high density and pressure is initiated at the center
    of a square domain. In isothermal mode, the sound speed is set to 1
    everywhere. In gamma-law mode, the adiabatic index is 5/3.

    Currently this setup can specify either the `cbdiso_2d` or `cbdgam_2d`
    solvers.
    """

    eos = param("isothermal", "EOS type: either isothermal or gamma-law")

    @property
    def is_isothermal(self):
        return self.eos == "isothermal"

    @property
    def is_gamma_law(self):
        return self.eos == "gamma-law"

    def primitive(self, t, coords, primitive):
        x, y = coords
        in_cylinder = (x * x + y * y) ** 0.5 < 0.25

        if self.is_isothermal:
            if in_cylinder:
                primitive[0] = 1.0
            else:
                primitive[0] = 0.1

        elif self.is_gamma_law:
            if in_cylinder:
                primitive[0] = 1.0
                primitive[3] = 1.0
            else:
                primitive[0] = 0.1
                primitive[3] = 0.125

    def mesh(self, resolution):
        return PlanarCartesian2DMesh.centered_square(1.0, resolution)

    @property
    def physics(self):
        if self.is_isothermal:
            return dict(eos_type=EquationOfState.GLOBALLY_ISOTHERMAL, sound_speed=1.0)
        elif self.is_gamma_law:
            return dict(eos_type=EquationOfState.GAMMA_LAW, gamma_law_index=5 / 3)

    @property
    def solver(self):
        if self.is_isothermal:
            return "cbdiso_2d"
        elif self.is_gamma_law:
            return "cbdgam_2d"

    @property
    def boundary_condition(self):
        return "outflow"

    @property
    def default_end_time(self):
        return 0.3

    def validate(self):
        if not self.is_isothermal and not self.is_gamma_law:
            raise ValueError(f"eos must be isothermal or gamma-law, got {self.eos}")
