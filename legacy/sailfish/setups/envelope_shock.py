"""
Contains a setup for studying a relativistic type-II shockwave.
"""

from functools import lru_cache
from math import pi, exp, atan, log10
from sailfish.setup_base import SetupBase, SetupError, param
from sailfish.mesh import LogSphericalMesh

__all__ = ["EnvelopeShock"]

t_delay = 1.0
m1 = 1.0
psi = 0.25
m_cloud = 1e5 * m1
u_wind = 0.1
mdot_wind = m_cloud / t_delay


def shell_time_m(m):
    return m / mdot_wind


def shell_time_mprime(m):
    return 1.0 / mdot_wind


def shell_gamma_beta_m(m):
    return u_wind + (m / m1) ** (-psi)


def shell_gamma_beta_mprime(m):
    return -((m / m1) ** (-psi)) * psi / m


def shell_speed_m(m):
    u = shell_gamma_beta_m(m)
    return u / (1 + u**2) ** 0.5


def shell_speed_mprime(m):
    u = shell_gamma_beta_m(m)
    du_dm = shell_gamma_beta_mprime(m)
    dv_du = (1 + u**2) ** (-3 / 2)
    return dv_du * du_dm


def shell_radius_mt(m, t):
    v = shell_speed_m(m)
    t0 = shell_time_m(m)
    return v * (t - t0)


def shell_density_mt(m, t):
    t0 = shell_time_m(m)
    t0_prime = shell_time_mprime(m)
    u = shell_gamma_beta_m(m)
    u_prime = shell_gamma_beta_mprime(m)
    mdot_inverse = t0_prime - u_prime / u * (t - t0) / (1 + u**2)
    r = shell_radius_mt(m, t)
    return 1.0 / (4 * pi * r**2 * u * mdot_inverse)


def shell_mass_rt(r, t):
    def f(m):
        return r - shell_radius_mt(m, t)

    def g(m):
        v = shell_speed_m(m)
        t0 = shell_time_m(m)
        dv = shell_speed_mprime(m)
        dt0 = shell_time_mprime(m)
        return -(dv * (t - t0) - v * dt0)

    m = 1e-12
    n = 0

    while True:
        fm = f(m)
        gm = g(m)
        m -= fm / gm

        if abs(fm) < 1e-10:
            return m
        if n > 200:
            raise ValueError("too many iterations")

        n += 1


class EnvelopeShock(SetupBase):
    """
    A relativistic shell or jet launched into a homologous, relativistic envelope.
    """

    u_shell = param(30.0, "gamma-beta of the shell [must be zero if jet_energy > 0.0]")
    m_shell = param(1.0, "mass coordinate of the launched shell")
    w_shell = param(1.0, "width of the shell in dm/m")
    q_shell = param(0.1, "opening angle of the shell")
    t_start = param(1.0, "time when the simulation starts")
    r_inner = param(0.1, "inner radius (comoving if expand=True)")
    r_outer = param(1.0, "outer radius (comoving if expand=True)")
    expand = param(True, "whether to expand the mesh homologously")
    jet_energy = param(0.0, "jet energy, in units of cloud mass c^2 [0.0 for no jet]")
    jet_gamma_beta = param(10.0, "jet gamma-beta")
    jet_theta = param(0.1, "jet opening angle [radians]")
    jet_duration = param(1.0, "jet duration [s]")
    polar_extent = param(0.0, "polar domain extent over pi (equator is 0.5, 1D is 0.0)")

    def validate(self):
        if self.jet_energy > 0.0:
            if self.u_shell > 0.0:
                raise SetupError("we don't simulate a jet and a shell at the same time")
            if self.polar_extent == 0.0:
                raise SetupError("the jet boundary condition is only supported in 2d")

    @property
    def polar(self):
        return self.polar_extent > 0.0

    def primitive(self, t, coord, primitive):
        r = coord[0] if self.polar else coord
        m, d, u, p = self.envelope_state(r, t)

        if not self.polar:
            primitive[0] = d
            primitive[1] = u + self.shell_u_profile_mass(m)
            primitive[2] = p

            if m > self.m_shell and m < self.m_shell * (1.0 + self.w_shell):
                primitive[3] = 1.0
            else:
                primitive[3] = 0.0
        else:
            q = coord[1]
            primitive[0] = d
            primitive[1] = u + self.shell_u_profile_mass(
                m
            ) * self.shell_u_profile_polar(q)
            primitive[2] = 0.0
            primitive[3] = p

    @property
    def physics(self):
        """
        This function returns parameters for a jet boundary condition, if needed.

        The isotropic-equivalent jet power is M-dot (Gamma - 1). This should
        be compared to the mass shell ejected at t = t_delay, which is M_cloud
        = 1e5 m1 (m1 = 1); jet_energy (which is normalized to m_cloud) is
        M-dot (Gamma - 1) t_engine / M_cloud:

        jet_energy = 4 pi r^2 rho u (Gamma - 1) t_engine / M_cloud
        """

        if self.jet_energy == 0.0:
            return None

        jet_gamma_beta = self.jet_gamma_beta
        jet_gamma = (jet_gamma_beta * jet_gamma_beta + 1.0) ** 0.5
        jet_mdot = self.jet_energy / ((jet_gamma - 1.0) * self.jet_duration / m_cloud)

        return dict(
            jet_mdot=jet_mdot,
            jet_gamma_beta=jet_gamma_beta,
            jet_theta=self.jet_theta,
            jet_duration=self.jet_duration,
        )

    def mesh(self, num_zones_per_decade):
        return LogSphericalMesh(
            r0=self.r_inner,
            r1=self.r_outer,
            num_zones_per_decade=num_zones_per_decade,
            scale_factor_derivative=(1.0 / self.t_start) if self.expand else None,
            polar_grid=self.polar,
            polar_extent=self.polar_extent * pi,
        )

    @property
    def solver(self):
        if not self.polar:
            return "srhd_1d"
        else:
            return "srhd_2d"

    @property
    def start_time(self):
        return self.t_start

    @property
    def boundary_condition(self):
        if self.jet_energy > 0.0:
            return "jet", "outflow"
        else:
            return "outflow"

    @property
    def default_end_time(self):
        return 1.0

    @property
    def default_resolution(self):
        if self.polar:
            return 800
        else:
            return 20000

    @lru_cache(maxsize=None)
    def shell_u_profile_polar(self, q):
        return exp(-((q / self.q_shell) ** 2))

    @lru_cache(maxsize=None)
    def shell_u_profile_mass(self, m):
        if m < self.m_shell:
            return 0.0
        else:
            return self.u_shell * exp(-(m / self.m_shell - 1.0) / self.w_shell)

    @lru_cache(maxsize=None)
    def envelope_state(self, r, t):
        # These expressions are for the pure-envelope case, with no attached
        # wind:
        #
        # s = r / t
        # g = (1.0 - s * s) ** -0.5
        # u = s * g
        # m = m1 * u ** (-1.0 / psi)
        # d = m * g / (4 * pi * r ** 3 * psi)

        m = shell_mass_rt(r, t)
        d = shell_density_mt(m, t)
        u = shell_gamma_beta_m(m)
        p = 1e-6 * d
        return m, d, u, p


# ---------------------------------------------------------
# Code below can probably be removed
# ---------------------------------------------------------
#
# from typing import NamedTuple
# try:
#     from functools import cached_property
# except ImportError:
#     # revert to ordinary property on Python < 3.8
#     cached_property = property

# def r_shell(self) -> float:
#     u = self.m_shell ** -0.25
#     s = u / (1.0 + u * u) ** 0.5
#     return self.t_start * s

# @cached_property
# def ambient(self):
#     return RelativisticEnvelope(
#         envelope_m1=1.0,
#         envelope_fastest_beta=0.999,
#         envelope_slowest_beta=0.00,
#         envelope_psi=0.25,
#         wind_mdot=100.0,
#     )

# def gamma_shell(self) -> float:
#     return (1.0 + self.u_shell ** 2) ** 0.5

# def shell_energy(self) -> float:
#     return self.w_shell * self.m_shell * (self.gamma_shell() - 1.0)


# ZONE_ENVELOPE = 0
# ZONE_WIND = 1


# class RelativisticEnvelope(NamedTuple):
#     """
#     Describes a homologous expanding medium with power-law mass coordinate.
#     """

#     envelope_m1: float
#     """ Mass coordinate of the u=1 shell """

#     envelope_slowest_beta: float
#     """ Beta (v/c) of the slowest envelope shell """

#     envelope_fastest_beta: float
#     """ Beta (v/c) of the outer shell """

#     envelope_psi: float
#     """ Index psi in u(m) ~ m^-psi """

#     wind_mdot: float
#     """ The mass loss rate for the wind """

#     def zone(self, r: float, t: float) -> int:
#         v_min = self.envelope_slowest_beta
#         r_wind_envelop_interface = v_min * t

#         if r > r_wind_envelop_interface:
#             return ZONE_ENVELOPE
#         else:
#             return ZONE_WIND

#     def gamma_beta(self, r: float, t: float) -> float:
#         if self.zone(r, t) == ZONE_WIND:
#             return self.envelope_slowest_u()

#         if self.zone(r, t) == ZONE_ENVELOPE:
#             b = min(r / t, self.envelope_fastest_beta)
#             u = b / (1.0 - b * b) ** 0.5
#             return u

#     def mass_rate_per_steradian(self, r: float, t: float) -> float:
#         if self.zone(r, t) == ZONE_WIND:
#             return self.wind_mdot

#         if self.zone(r, t) == ZONE_ENVELOPE:
#             y = self.envelope_psi
#             s = min(r / t, self.envelope_fastest_beta)
#             f = s ** (-1.0 / y) * (1.0 - s * s) ** (0.5 / y - 1.0)
#             return self.envelope_m1 / (4.0 * pi * y * t) * f

#     def comoving_mass_density(self, r: float, t: float) -> float:
#         return self.mass_rate_per_steradian(r, t) / (self.gamma_beta(r, t) * r * r)
