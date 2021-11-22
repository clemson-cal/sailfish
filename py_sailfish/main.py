import time
import numpy as np
from sailfish import Library


"""
Adapter class to drive the srhd_1d C extension module.
"""
class Solver:
    def __init__(self, primitive, time=0.0, bc='inflow', coords='cartesian', mode='cpu'):
        if mode == 'cpu':
            import numpy
            self.xp = numpy
        if mode == 'gpu':
            import cupy
            self.xp = cupy

        self.lib = Library('srhd_1d', mode=mode)
        self.num_zones = primitive.shape[0]
        self.faces = self.xp.linspace(0.0, 1.0, num_zones + 1)
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
        args = (
            self.num_zones,
            self.faces,
            primitive,
            conserved,
            self.scale_factor(),
            self.coords,
        )
        self.lib.invoke('srhd_1d_primitive_to_conserved', self.num_zones, args)
        return conserved

    def recompute_primitive(self):
        args = (
            self.num_zones,
            self.faces,
            self.conserved1,
            self.primitive1,
            self.scale_factor(),
            self.coords,
        )
        self.lib.invoke('srhd_1d_conserved_to_primitive', self.num_zones, args)

    def advance_rk(self, rk_param, dt):
        self.recompute_primitive()
        args = (
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
        self.lib.invoke('srhd_1d_advance_rk', self.num_zones, args)
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
        return np.array(self.primitive1)


if __name__ == "__main__":
    num_zones = 10000
    fold = 100
    cfl_number = 0.6
    dt = 1.0 / num_zones * cfl_number
    n = 0

    x = np.linspace(0.0, 1.0, num_zones)
    primitive = np.zeros([num_zones, 4])

    for i in range(num_zones):
        if x[i] < 0.5:
            primitive[i, 0] = 1.0
            primitive[i, 2] = 1.0
        else:
            primitive[i, 0] = 0.1
            primitive[i, 2] = 0.125

    solver = Solver(primitive, mode='cpu')

    while solver.time < 0.01:
        start = time.perf_counter()
        for _ in range(fold):
            solver.new_timestep()
            solver.advance_rk(0.0, dt)
            solver.advance_rk(0.5, dt)
            n += 1
        # cupy.cuda.Device().synchronize()
        stop = time.perf_counter()
        Mzps = num_zones / (stop - start) * 1e-6 * fold
        print(f"[{n:04d}] t={solver.time:0.3f} Mzps={Mzps:.3f}")

    # np.save('chkpt.0000.npy', solver.primitive)

    import matplotlib.pyplot as plt
    plt.plot(x, solver.primitive[:,0])
    plt.show()
