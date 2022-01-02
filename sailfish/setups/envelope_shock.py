"""
Contains a setup for studying a relativistic type-II shockwave.
"""

from math import pi, exp, log10
from typing import NamedTuple
from sailfish.setup import Setup, SetupError, param
from sailfish.mesh import LogSphericalMesh

try:
    from functools import cached_property
except ImportError:
    # revert to ordinary property on Python < 3.8
    cached_property = property

__all__ = ["EnvelopeShock", "RelativisticEnvelope"]


ZONE_ENVELOPE = 0
ZONE_WIND = 1


class RelativisticEnvelope(NamedTuple):
    """
    Describes a homologous expanding medium with power-law mass coordinate.
    """

    envelope_m1: float
    """ Mass coordinate of the u=1 shell """

    envelope_slowest_beta: float
    """ Beta (v/c) of the slowest envelope shell """

    envelope_fastest_beta: float
    """ Beta (v/c) of the outer shell """

    envelope_psi: float
    """ Index psi in u(m) ~ m^-psi """

    wind_mdot: float
    """ The mass loss rate for the wind """

    def envelop_slowest_u(self) -> float:
        b = self.envelope_slowest_beta
        return b / (1.0 - b * b).sqrt()

    def zone(self, r: float, t: float) -> int:
        v_min = self.envelope_slowest_beta
        r_wind_envelop_interface = v_min * t

        if r > r_wind_envelop_interface:
            return ZONE_ENVELOPE
        else:
            return ZONE_WIND

    def gamma_beta(self, r: float, t: float) -> float:
        if self.zone(r, t) == ZONE_WIND:
            return self.envelope_slowest_u()

        if self.zone(r, t) == ZONE_ENVELOPE:
            b = min(r / t, self.envelope_fastest_beta)
            u = b / (1.0 - b * b) ** 0.5
            return u

    def mass_rate_per_steradian(self, r: float, t: float) -> float:
        if self.zone(r, t) == ZONE_WIND:
            return self.wind_mdot

        if self.zone(r, t) == ZONE_ENVELOPE:
            y = self.envelope_psi
            s = min(r / t, self.envelope_fastest_beta)
            f = s ** (-1.0 / y) * (1.0 - s * s) ** (0.5 / y - 1.0)
            return self.envelope_m1 / (4.0 * pi * y * t) * f

    def comoving_mass_density(self, r: float, t: float) -> float:
        return self.mass_rate_per_steradian(r, t) / (self.gamma_beta(r, t) * r * r)


class EnvelopeShock(Setup):
    """
    A relativistic shell launched into a homologous, relativistic envelope.
    """

    u_shell = param(30.0, "gamma-beta of the launched shell")
    m_shell = param(1.0, "mass coordinate of the launched shell")
    w_shell = param(1.0, "width of the shell in dm/m")
    q_shell = param(0.1, "opening angle of the shell")
    t_start = param(1.0, "time when the simulation starts")
    r_inner = param(0.1, "inner radius at start")
    r_outer = param(10.0, "outer radius at start")
    expand = param(True, "whether to expand the mesh homologously")
    polar_extent = param(0.5, "polar domain extent over pi (equator is 0.5, 1D is 0.0)")

    @property
    def polar(self):
        return self.polar_extent > 0.0

    def primitive(self, t, coord, primitive):
        ambient = self.ambient
        psi = ambient.envelope_psi
        m1 = ambient.envelope_m1

        r = coord[0] if self.polar else coord
        s = min(r / t, ambient.envelope_fastest_beta)
        u = s / (1.0 - s * s) ** 0.5
        m = m1 * u ** (-1.0 / psi)
        d = ambient.comoving_mass_density(r, t)
        p = 1e-4 * d

        def u_prof(m):
            if m < self.m_shell:
                return 0.0
            else:
                return exp(-(m / self.m_shell - 1.0) / self.w_shell)

        if not self.polar:
            primitive[0] = d
            primitive[1] = u + u_prof(m) * self.u_shell
            primitive[2] = p

            if m > self.m_shell and m < self.m_shell * (1.0 + self.w_shell):
                primitive[3] = 1.0
            else:
                primitive[3] = 0.0
        else:
            q_bar = coord[1] / self.q_shell
            primitive[0] = d
            primitive[1] = u + u_prof(m) * self.u_shell * exp(-(q_bar ** 2.0))
            primitive[2] = 0.0
            primitive[3] = p

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
        return "outflow"

    @property
    def default_end_time(self):
        return 1.0

    def r_shell(self) -> float:
        u = self.m_shell ** -0.25
        s = u / (1.0 + u * u) ** 0.5
        return self.t_start * s

    @cached_property
    def ambient(self):
        return RelativisticEnvelope(
            envelope_m1=1.0,
            envelope_fastest_beta=self.u_shell / (1.0 + self.u_shell ** 2) ** 0.5,
            envelope_slowest_beta=0.00,
            envelope_psi=0.25,
            wind_mdot=100.0,
        )

    def gamma_shell(self) -> float:
        return (1.0 + self.u_shell ** 2) ** 0.5

    def shell_energy(self) -> float:
        return self.w_shell * self.m_shell * (self.gamma_shell() - 1.0)
