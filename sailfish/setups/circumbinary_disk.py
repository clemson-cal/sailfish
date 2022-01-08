"""
2D disk setups for binary problems.
"""

from math import sqrt
from sailfish.mesh import LogSphericalMesh, PlanarCartesian2DMesh
from sailfish.physics.circumbinary import *
from sailfish.setup import Setup, param

class CircumbinaryDisk(Setup):
    """
    A circumbinary disk setup for binary problems; isothermal or gamma-law.

    This problem is the primary science setup for the cbdiso_2d and cbdgam_2d
    solvers. The initial data consists of nearly stationary axisymmetric solutions,
    which partly means that Keplerian velocity profiles are imposed with a
    softened radial coordinate.

    In isothermal mode, since only constant-nu viscosity is supported, the disk
    is constant density = 1.0. Note that the overall scale of the density is
    not physically meaningful for the isothermal hydro equations, since those
    equations are invariant under uniform rescalings of the density.

    In gamma-law mode, since only constant-alpha viscosity is supported, the
    disk is nominally `Shakura-Sunyaev (1973) <https://ui.adsabs.harvard.edu/abs/1973A%26A....24..337S/abstract>`. 
    However, we follow the purely Newtonian treatment given in
    `Goodman (2003) <https://ui.adsabs.harvard.edu/abs/2003MNRAS.339..937G>`,
    setting radiation pressure to zero. In particular, we impose the following
    power law profiles on the surface density :math:`\Sigma` and
    vertically-integrated pressure :math:`\mathcal{P}`:

    .. math::
        \Sigma \propto r^{-3/5}, \, \mathcal{P} \propto r^{-3/2}
    """

    eos = param("isothermal", "EOS type: either isothermal or gamma-law")
    initial_density = param(1.0, "Initial disk surface density at r=a")
    initial_pressure = param(1e-2, "Initial disk surface pressure at r=a")
    domain_radius = param(12.0, "Half side length of the square computational domain")

    @property
    def is_isothermal(self):
        return self.eos == "isothermal"

    @property
    def is_gamma_law(self):
        return self.eos == "gamma-law"

    def primitive(self, t, coords, primitive):
        x, y = coords
        r = sqrt(x * x + y * y)
        rs = sqrt(x * x + y * y + softening_length**2) #How to get softening_length?
        phi_hat_x = -y / max(r, 1e-12) #Will throw error because r is an array
        phi_hat_y = x / max(r, 1e-12) #Will throw error because r is an array

        if self.is_isothermal:
            primitive[0] = 1.0
            primitive[1] = phi_hat_x / sqrt(rs)
            primitive[2] = phi_hat_y / sqrt(rs)

        elif self.is_gamma_law:
            primitive[0] = self.initial_density * rs**(-3.0 / 5.0) # Eq. (A2) from Goodman (2003)
            primitive[1] = phi_hat_x / sqrt(rs)
            primitive[2] = phi_hat_y / sqrt(rs)
            primitive[3] = self.initial_pressure * rs**(-3.0 / 2.0) # Derived from Goodman (2003)

    def mesh(self, resolution):
        return PlanarCartesian2DMesh.centered_square(self.domain_radius, resolution)

    @property
    def physics(self):
        if self.is_isothermal:
            return dict(eos_type=EOS_TYPE_GLOBALLY_ISOTHERMAL, sound_speed=1.0)
        elif self.is_gamma_law:
            return dict(eos_type=EOS_TYPE_GAMMA_LAW, gamma_law_index=5 / 3)

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
        return 1000.0

    def validate(self):
        if not self.is_isothermal and not self.is_gamma_law:
            raise ValueError(f"eos must be isothermal or gamma-law, got {self.eos}")
