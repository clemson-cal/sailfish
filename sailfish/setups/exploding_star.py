#! /usr/bin/env python

import numpy as np
import astropy.units as u
import astropy.constants as const
from sailfish.setup import Setup, param
from sailfish.mesh import LogSphericalMesh

# Stellar and Engine Parameters
c = const.c.cgs  # Speed of light converted to cm/s
m_0 = 2e33 * u.g  # solar radius
R_0 = 7e10 * u.cm  # Characteristic Length Scale
rho_ref = m_0 / ((4.0 / 3.0) * np.pi * R_0 ** 3)
rho_c = 3e7 * rho_ref  # Central Density
R_1 = 0.0017 * R_0  # First Break Radius
R_2 = 0.0125 * R_0  # Second Break Radius
R_3 = 0.65 * R_0  # Outer Radius
R_4 = 50 * R_0

k1 = 3.24  # First Break Slope
k2 = 2.57  # Second Break Slope
n = 16.7  # Atmosphere Cutoff Slope
rho_wind = 1e-9 * rho_ref  # Wind Density
rho_env = 1e-7 * rho_ref  # Envelope Density
alpha = 2.5

theta_0 = np.pi / 2  # Injection Angle
gamma_0 = 50  # Injected Lorentz Factor
eta_0 = 2.0  # Energy-to-Mass Ratio
r_0 = 0.1 * R_0  # Nozzle size
L_0 = (2.0e-3 * m_0 * c ** 3 / R_0).to(u.erg / u.s)  # Engine Power (One-Sided)
tau_0 = 4.3 * R_0 / c  # Engine Duration

E_0 = 1e51 * u.erg
mdot = 1e-5 * u.M_sun / u.yr  # Mass loss rate in Solar mass per year
vw = 1e3 * u.km / u.s  # Stellar wind speed in km/s
alpha = 2.5
epsilon = 2.0
m_env = 0.5 * 2e33 * u.g
n2 = 10.0
A_star = mdot.to(u.g / u.s) / (4 * np.pi * vw.to(u.cm / u.s))


def rho0(r, ascale):
    central_scale = rho_c / rho_ref
    env_denom = 1.0 + (r * R_0 / R_1) ** k1 / (1 + (r * R_0 / R_2) ** k2)
    he_core = max(1.0 - r * R_0 / R_3, 0.0) ** n / env_denom
    wind_density = ascale * A_star / R_3 ** 2
    return central_scale * he_core + wind_density / rho_ref * (r * R_0 / R_3) ** (-2.0)


class ExplodingStar(Setup):
    """
    A star exploding.
    """

    escale = param(1.0, "energy scale, normalized to 10^51 erg")
    ascale = param(0.1, "ambient medium density, normalized to A* = 1")
    r_inner = param(0.1, "inner radius at start")
    r_outer = param(10.0, "outer radius at start")
    # expand = param(True, "whether to expand the mesh homologously")

    def primitive(self, t, r, primitive):
        primitive[0] = rho0(r, self.ascale)
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
            # scale_factor_derivative=(1.0 / self.t_start) if self.expand else None,
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
