from ctypes import c_double, c_int, POINTER, CDLL
import time
import glob
import numpy as np


block_size = 64


class Library:
    def __init__(self, mode='cpu'):
        with open('src/srhd_1d.c', 'r') as srcfile:
            code = srcfile.read()
        if mode == 'cpu':
            import cffi
            ffi = cffi.FFI()
            ffi.set_source(
                'srhd_1d',
                code,
                define_macros=[('EXEC_MODE', 0)],
                # extra_compile_args=['-Xpreprocessor', '-fopenmp'],
                # extra_link_args=['-lomp'],
            )
            target = 'lib/srhd_1d.so'
            target = ffi.compile(target=target)
            self.module = CDLL(target)
        if mode == 'gpu':
            import cupy
            module = cupy.RawModule(code=code, options=('-D EXEC_MODE=2',))
            module.compile()
            self.module = module
        self.mode = mode

    def invoke(self, symbol, num_zones, args):
        if self.mode == 'cpu':
            function = getattr(self.module, symbol)
            return function(*args)
        if self.mode == 'gpu':
            num_blocks = ((num_zones + block_size - 1) // block_size,)
            block_size = (block_size,)
            function = self.module.get_function(symbol)
            return function(num_blocks, block_size, args)


class Memory:
    def __init__(self, mode='cpu'):
        self.mode = mode

    def ptr(self, a):
        if self.mode == 'cpu':
            return a.ctypes.data_as(POINTER(c_double))
        if self.mode == 'gpu':
            return a.data.ptr

    def copy_to_host(self, a):
        if self.mode == 'cpu':
            return a.copy()
        if self.mode == 'gpu':
            import cupy
            return cupy.asnumpy(a)

    def on_device(self, a):
        if self.mode == 'cpu':
            return a
        if self.mode == 'gpu':
            import cupy
            return cupy.array(a)

    def zeros(self, *args):
        if self.mode == 'cpu':
            return np.zeros(*args)
        if self.mode == 'gpu':
            import cupy
            return cupy.zeros(*args)

    def zeros_like(self, *args):
        if self.mode == 'cpu':
            return np.zeros_like(*args)
        if self.mode == 'gpu':
            import cupy
            return cupy.zeros_like(*args)

    def linspace(self, *args):
        if self.mode == 'cpu':
            return np.linspace(*args)
        if self.mode == 'gpu':
            import cupy
            return cupy.linspace(*args)


"""
Adapter class to drive the srhd_1d C extension module.
"""
class Solver:
    def __init__(self, primitive, time=0.0, bc='inflow', coords='cartesian'):
        self.lib = Library(mode='cpu')
        self.mem = Memory(mode='cpu')
        self.num_zones = primitive.shape[0]
        self.faces = self.mem.linspace(0.0, 1.0, num_zones + 1)
        self.boundary_condition = dict(inflow=0, zeroflux=1)[bc]
        self.coords = dict(cartesian=0, spherical=1)[coords]
        self.scale_factor_initial = 1.0
        self.scale_factor_derivative = 0.0
        self.time = self.time0 = time
        self.primitive1 = self.mem.on_device(primitive)
        self.conserved0 = self.primitive_to_conserved(self.primitive1)
        self.conserved1 = self.conserved0.copy()
        self.conserved2 = self.conserved0.copy()

    def scale_factor(self):
        return self.scale_factor_initial + self.scale_factor_derivative * self.time

    def primitive_to_conserved(self, primitive):
        conserved = self.mem.zeros_like(primitive)
        args = (
            c_int(self.num_zones),
            self.mem.ptr(self.faces),
            self.mem.ptr(primitive),
            self.mem.ptr(conserved),
            c_double(self.scale_factor()),
            c_int(self.coords),
        )
        self.lib.invoke('srhd_1d_primitive_to_conserved', self.num_zones, args)
        return conserved

    def recompute_primitive(self):
        args = (
            c_int(self.num_zones),
            self.mem.ptr(self.faces),
            self.mem.ptr(self.conserved1),
            self.mem.ptr(self.primitive1),
            c_double(self.scale_factor()),
            c_int(self.coords),
        )
        self.lib.invoke('srhd_1d_conserved_to_primitive', self.num_zones, args)

    def new_timestep(self):
        self.time0 = self.time
        self.conserved0[...] = self.conserved1[...]

    def advance_rk(self, rk_param, dt):
        self.recompute_primitive()
        args = (
            c_int(self.num_zones),
            self.mem.ptr(self.faces),
            self.mem.ptr(self.conserved0),
            self.mem.ptr(self.primitive1),
            self.mem.ptr(self.conserved1),
            self.mem.ptr(self.conserved2),
            c_double(self.scale_factor_initial),
            c_double(self.scale_factor_derivative),
            c_double(self.time),
            c_double(rk_param),
            c_double(dt),
            c_int(self.coords),
            c_int(self.boundary_condition),
        )
        self.lib.invoke('srhd_1d_advance_rk', self.num_zones, args)
        self.time = self.time0 * rk_param + (self.time0 + dt) * (1.0 - rk_param)
        self.conserved1, self.conserved2 = self.conserved2, self.conserved1

    @property
    def primitive(self):
        self.recompute_primitive()
        return self.mem.copy_to_host(self.primitive1)


if __name__ == "__main__":
    num_zones = 1000000
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

    solver = Solver(primitive)

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
