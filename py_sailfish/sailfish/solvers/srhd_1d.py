from sailfish.library import Library
from sailfish.system import get_array_module


"""Adapter class to drive the srhd_1d C extension module.
"""


class Solver:
    def __init__(
        self, primitive, time=0.0, bc="inflow", coords="cartesian", mode="cpu"
    ):
        self.xp = get_array_module(mode)
        self.lib = Library(__file__, mode=mode)
        self.num_zones = primitive.shape[0]
        self.faces = self.xp.linspace(0.0, 1.0, self.num_zones + 1)
        self.boundary_condition = dict(inflow=0, zeroflux=1)[bc]
        self.coords = dict(cartesian=0, spherical=1)[coords]
        self.scale_factor_initial = 1.0
        self.scale_factor_derivative = 0.0
        self.time = self.time0 = time
        self.primitive1 = self.xp.array(primitive)
        self.conserved0 = self.primitive_to_conserved(self.primitive1)
        self.conserved1 = self.conserved0.copy()
        self.conserved2 = self.conserved0.copy()

    def primitive_to_conserved(self, primitive):
        conserved = self.xp.zeros_like(primitive)
        self.lib.srhd_1d_primitive_to_conserved(
            self.num_zones,
            self.faces,
            primitive,
            conserved,
            self.scale_factor(),
            self.coords,
        )
        return conserved

    def recompute_primitive(self):
        self.lib.srhd_1d_conserved_to_primitive(
            self.num_zones,
            self.faces,
            self.conserved1,
            self.primitive1,
            self.scale_factor(),
            self.coords,
        )

    def advance_rk(self, rk_param, dt):
        self.recompute_primitive()
        self.lib.srhd_1d_advance_rk(
            self.num_zones,
            self.faces,
            self.conserved0,
            self.primitive1,
            self.conserved1,
            self.conserved2,
            self.scale_factor_initial,
            self.scale_factor_derivative,
            self.time,
            rk_param,
            dt,
            self.coords,
            self.boundary_condition,
        )
        self.time = self.time0 * rk_param + (self.time0 + dt) * (1.0 - rk_param)
        self.conserved1, self.conserved2 = self.conserved2, self.conserved1

    def scale_factor(self):
        return self.scale_factor_initial + self.scale_factor_derivative * self.time

    def new_timestep(self):
        self.time0 = self.time
        self.conserved0[...] = self.conserved1[...]

    @property
    def primitive(self):
        self.recompute_primitive()
        return self.primitive1.copy()
