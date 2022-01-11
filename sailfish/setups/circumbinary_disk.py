"""
2D disk setups for binary problems.
"""

from math import sqrt, exp
from sailfish.mesh import LogSphericalMesh, PlanarCartesian2DMesh
from sailfish.physics.circumbinary import EquationOfState, PointMass, SinkModel
from sailfish.physics.kepler import OrbitalElements
from sailfish.setup import Setup, param


class CircumbinaryDisk(Setup):
    r"""
    A circumbinary disk setup for binary problems, isothermal or gamma-law.

    This problem is the primary science setup for the cbdiso_2d and cbdgam_2d
    solvers. The initial data consists of nearly stationary axisymmetric
    solutions, which partly means that Keplerian velocity profiles are imposed
    with a softened radial coordinate.

    In isothermal mode, since only constant-nu viscosity is supported, the
    disk is constant density = 1.0. Note that the overall scale of the density
    is not physically meaningful for the isothermal hydro equations, since
    those equations are invariant under uniform rescalings of the density.

    In gamma-law mode, since only constant-alpha viscosity is supported, the
    disk is nominally `Shakura & Sunyaev (1973)`_. However, we follow the
    purely Newtonian treatment given in `Goodman (2003)`_, setting radiation
    pressure to zero. In particular, we impose the following power law
    profiles on the surface density :math:`\Sigma` and vertically-integrated
    pressure :math:`\mathcal{P}`:

    .. math::
        \Sigma \propto r^{-3/5}, \, \mathcal{P} \propto r^{-3/2}

    .. _Shakura & Sunyaev (1973): https://ui.adsabs.harvard.edu/abs/1973A%26A....24..337S
    .. _Goodman (2003): https://ui.adsabs.harvard.edu/abs/2003MNRAS.339..937G
    """

    eos = param("isothermal", "EOS type: either isothermal or gamma-law")
    domain_radius = param(12.0, "half side length of the square computational domain")
    mach_number = param(10.0, "orbital Mach number (isothermal)", mutable=True)
    eccentricity = param(0.0, "orbital eccentricity of the binary", mutable=True)
    mass_ratio = param(1.0, "component mass ratio m2 / m1 <= 1", mutable=True)
    sink_rate = param(10.0, "component sink rate", mutable=True)
    sink_radius = param(0.05, "component sink radius", mutable=True)
    softening_length = param(0.05, "gravitational softening length", mutable=True)
    buffer_is_enabled = param(True, "whether the buffer zone is enabled")
    sink_model = param("torque_free", "sink [acceleration_free|force_free|torque_free]")

    initial_sigma = param(1.0, "initial disk surface density at r=a (gamma-law)")
    initial_pressure = param(1e-2, "initial disk surface pressure at r=a (gamma-law)")
    cooling_coefficient = param(0.0, "strength of the cooling term (gamma-law)")
    alpha = param(0.1, "alpha-viscosity parameter (gamma-law)")
    constant_softening = param(True, "whether to use constant softening (gamma-law)")
    gamma_law_index = param(5.0 / 3.0, "adiabatic index (gamma-law)")

    @property
    def is_isothermal(self):
        return self.eos == "isothermal"

    @property
    def is_gamma_law(self):
        return self.eos == "gamma-law"

    def primitive(self, t, coords, primitive):
        GM = 1.0
        x, y = coords
        r = sqrt(x * x + y * y)
        r_softened = sqrt(x * x + y * y + self.softening_length * self.softening_length)
        phi_hat_x = -y / max(r, 1e-12)
        phi_hat_y = +x / max(r, 1e-12)

        if self.is_isothermal:
            primitive[0] = 1.0
            primitive[1] = GM / sqrt(r_softened) * phi_hat_x
            primitive[2] = GM / sqrt(r_softened) * phi_hat_y

        elif self.is_gamma_law:
            # See eq. (A2) from Goodman (2003)
            primitive[0] = (
                self.initial_sigma
                * r_softened ** (-3.0 / 5.0)
                * (0.0001 + 0.9999 * exp(-((1.0 / r_softened) ** 30)))
            )
            primitive[1] = GM / sqrt(r_softened) * phi_hat_x
            primitive[2] = GM / sqrt(r_softened) * phi_hat_y
            primitive[3] = (
                self.initial_pressure
                * r_softened ** (-3.0 / 2.0)
                * (0.0001 + 0.9999 * exp(-((1.0 / r_softened) ** 30)))
            )

    def mesh(self, resolution):
        return PlanarCartesian2DMesh.centered_square(self.domain_radius, resolution)

    @property
    def default_resolution(self):
        return 1000

    @property
    def physics(self):
        if self.is_isothermal:
            return dict(
                eos_type=EquationOfState.LOCALLY_ISOTHERMAL,
                mach_number=self.mach_number,
                point_mass_function=self.point_masses,
            )

        elif self.is_gamma_law:
            return dict(
                eos_type=EquationOfState.GAMMA_LAW,
                gamma_law_index=self.gamma_law_index,
                point_mass_function=self.point_masses,
                buffer_is_enabled=self.buffer_is_enabled,
                cooling_coefficient=self.cooling_coefficient,
                constant_softening=self.constant_softening,
            )

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

    @property
    def orbital_elements(self):
        return OrbitalElements(
            semimajor_axis=1.0,
            total_mass=1.0,
            mass_ratio=self.mass_ratio,
            eccentricity=self.eccentricity,
        )

    def point_masses(self, time):
        m1, m2 = self.orbital_elements.orbital_state(time)

        return (
            PointMass(
                softening_length=self.softening_length,
                sink_model=SinkModel[self.sink_model.upper()],
                sink_rate=self.sink_rate,
                sink_radius=self.sink_radius,
                **m1._asdict(),
            ),
            PointMass(
                softening_length=self.softening_length,
                sink_model=SinkModel[self.sink_model.upper()],
                sink_rate=self.sink_rate,
                sink_radius=self.sink_radius,
                **m2._asdict(),
            ),
        )
