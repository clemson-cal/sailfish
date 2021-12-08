# from math import pi, sin
from sailfish.setup import Setup, SetupError, param
from sailfish.mesh import LogSphericalMesh

__all__ = ["EnvelopeShock"]


class EnvelopeShock(Setup):
    """
    A relativistic shell launched into a homologous, relativistic envelope.
    """

    u_shell = param(30.0, "gamma-beta of the launched shell")
    m_shell = param(1.0, "mass coordinate of the launched shell")
    w_shell = param(1.0, "width of the shell in dm/m")
    t_start = param(1.0, "time when the simulation starts")
    r_inner = param(0.1, "inner radius at start")
    r_outer = param(10.0, "outer radius at start")
    expand = param(True, "whether to expand the mesh homologously")

    def primitive(self, t, x, primitive):
        if x < 0.5:
            primitive[0] = 1.0
            primitive[2] = 1.0
        else:
            primitive[0] = 0.1
            primitive[2] = 0.125

    def mesh(self, num_zones_per_decade):
        return LogSphericalMesh(
            r0=1.0,
            r1=self.r_outer,
            num_zones_per_decade=num_zones_per_decade,
            scale_factor_derivative=1.0 / self.t_start if self.expand else None,
        )

    @property
    def solver(self):
        return "srhd_1d"

    @property
    def start_time(self):
        return self.t_start

    @property
    def boundary_condition(self):
        return "inflow", "outflow"

    @property
    def default_end_time(self):
        return 1.0
