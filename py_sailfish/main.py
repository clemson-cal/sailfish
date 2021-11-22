import time
import glob
import numpy as np
from ctypes import c_double, c_int, POINTER, CDLL


lib_name = glob.glob('build/*/srhd_1d.*.so')[0]
print(f'load shared library {lib_name}')
lib = CDLL(lib_name)
lib.srhd_1d_primitive_to_conserved.restype = c_int
lib.srhd_1d_primitive_to_conserved.argtypes = [
    c_int,             # num_zones
    POINTER(c_double), # face_positions_ptr
    POINTER(c_double), # primitive_ptr
    POINTER(c_double), # conserved_ptr
    c_double,          # scale_factor
    c_int,             # coords
    c_int,             # mode
]
lib.srhd_1d_conserved_to_primitive.restype = c_int
lib.srhd_1d_conserved_to_primitive.argtypes = [
    c_int,             # num_zones
    POINTER(c_double), # face_positions_ptr
    POINTER(c_double), # conserved_ptr
    POINTER(c_double), # primitive_ptr
    c_double,          # scale_factor
    c_int,             # coords
    c_int,             # mode
]
lib.srhd_1d_advance_rk.restype = c_int
lib.srhd_1d_advance_rk.argtypes = [
    c_int,             # num_zones
    POINTER(c_double), # face_positions_ptr
    POINTER(c_double), # conserved_rk_ptr
    POINTER(c_double), # primitive_rd_ptr
    POINTER(c_double), # conserved_rd_ptr
    POINTER(c_double), # conserved_wr_ptr
    c_double,          # a0
    c_double,          # adot
    c_double,          # t
    c_double,          # a
    c_double,          # dt
    c_int,             # bc
    c_int,             # coords
    c_int,             # mode
]


"""
Return a pointer to the memory buffer of a numpy array
"""
def ptr(a):
    return a.ctypes.data_as(POINTER(c_double))


"""
Adapter class to drive the srhd_1d C extension module.
"""
class Solver:
    def __init__(self, primitive, time=0.0, bc='inflow', coords='cartesian', mode='cpu'):
        self.num_zones = primitive.shape[0]
        self.faces = np.linspace(0.0, 1.0, num_zones + 1)
        self.boundary_condition = dict(inflow=0, zeroflux=1)[bc]
        self.coords = dict(cartesian=0, spherical=1)[coords]
        self.mode = dict(cpu=0, omp=1, gpu=2)[mode]
        self.scale_factor_initial = 1.0
        self.scale_factor_derivative = 0.0
        self.time = time
        self.conserved0 = self.primitive_to_conserved(primitive)
        self.conserved1 = self.conserved0.copy()
        self.conserved2 = self.conserved0.copy()
        self.primitive1 = primitive.copy()

    def scale_factor(self):
        return self.scale_factor_initial + self.scale_factor_derivative * self.time

    def primitive_to_conserved(self, primitive):
        conserved = np.zeros_like(primitive)
        assert(lib.srhd_1d_primitive_to_conserved(
            self.num_zones,
            ptr(self.faces),
            ptr(primitive),
            ptr(conserved),
            self.scale_factor(),
            self.coords,
            self.mode,
        ) == 0)
        return conserved

    def _recompute_primitive(self):
        assert(lib.srhd_1d_conserved_to_primitive(
            self.num_zones,
            ptr(self.faces),
            ptr(self.conserved1),
            ptr(self.primitive1),
            self.scale_factor(),
            self.coords,
            self.mode,
        ) == 0)

    def new_timestep(self):
        self.time0 = self.time
        self.conserved0[...] = self.conserved1[...]

    def advance_rk(self, rk_param, dt):
        self._recompute_primitive()
        assert(lib.srhd_1d_advance_rk(
            self.num_zones,
            ptr(self.faces),
            ptr(self.conserved0),
            ptr(self.primitive1),
            ptr(self.conserved1),
            ptr(self.conserved2),
            self.scale_factor_initial,
            self.scale_factor_derivative,
            self.time,
            rk_param,
            dt,
            self.boundary_condition,
            self.coords,
            self.mode,
        ) == 0)
        self.time = self.time0 * rk_param + (self.time0 + dt) * (1.0 - rk_param)
        self.conserved1, self.conserved2 = self.conserved2, self.conserved1

    @property
    def primitive(self):
        self._recompute_primitive()
        return self.primitive1.copy()


if __name__ == "__main__":
    num_zones = 100000
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

    solver = Solver(primitive, mode='omp')

    while solver.time < 0.2:
        start = time.perf_counter_ns()
        for _ in range(fold):
            solver.new_timestep()
            solver.advance_rk(0.0, dt)
            solver.advance_rk(0.5, dt)
            n += 1
        stop = time.perf_counter_ns()
        Mzps = num_zones / (stop - start) * 1e3 * fold
        print(f"[{n:04d}] t={solver.time:0.3f} Mzps={Mzps:.3f}")

    import matplotlib.pyplot as plt
    plt.plot(x, solver.primitive[:,0])
    plt.show()
