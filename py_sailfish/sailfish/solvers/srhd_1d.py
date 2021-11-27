from sailfish.library import Library
from sailfish.system import get_array_module
from sailfish.subdivide import subdivide

NUM_PATCHES = 1


class Patch:
    def __init__(
        self,
        idxrng,
        dx,
        primitive,
        time,
        lib,
        xp,
        bc="inflow",
        coords="cartesian",
    ):
        self.lib = lib
        self.xp = xp
        self.num_zones = primitive.shape[0]
        self.faces = self.xp.array([i * dx for i in range(idxrng[0], idxrng[1] + 1)])
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
        return self.primitive1


"""
Adapter class to drive the srhd_1d C extension module.
"""


class Solver:
    def __init__(self, primitive, time, mode="cpu", **kwargs):
        num_zones = primitive.shape[0]
        xp = get_array_module(mode)
        dx = 1.0 / num_zones
        ng = 2
        lib = Library(__file__, mode=mode, debug=True)

        self.patches = []
        self.num_zones = num_zones
        self.xp = xp

        for (a, b) in subdivide(self.num_zones, NUM_PATCHES):
            prim = xp.zeros([b - a + 2 * ng, 4])
            prim[ng:-ng] = primitive[a:b]
            self.patches.append(
                Patch((a - ng, b + ng), dx, prim, time, lib, xp, **kwargs)
            )

    def advance_rk(self, rk_param, dt):
        self.patches[0].primitive1[:+2] = self.patches[0].primitive1[2:4]
        self.patches[0].primitive1[-2:] = self.patches[0].primitive1[-4:-2]
        self.patches[0].conserved1[:+2] = self.patches[0].conserved1[2:4]
        self.patches[0].conserved1[-2:] = self.patches[0].conserved1[-4:-2]

        for patch in self.patches:
            patch.advance_rk(rk_param, dt)

    def new_timestep(self):
        for patch in self.patches:
            patch.new_timestep()

    @property
    def primitive(self):
        ng = 2
        primitive = self.xp.zeros([self.num_zones, 4])
        for (a, b), patch in zip(subdivide(self.num_zones, NUM_PATCHES), self.patches):
            primitive[a:b] = patch.primitive[ng:-ng]
        return primitive

    @property
    def time(self):
        return self.patches[0].time
