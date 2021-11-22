from ctypes import c_double, c_int, cast, POINTER, CDLL
import time
import glob
import numpy as np


block_size = 64


class Library:
    def __init__(self, mode='cpu'):
        self.mode = mode
        self.load_lib()

    def load_lib(self):
        with open('src/srhd_1d.c', 'r') as srcfile:
            code = srcfile.read()
        if self.mode == 'cpu':
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
        if self.mode == 'gpu':
            import cupy
            module = cupy.RawModule(code=code, options=('-D EXEC_MODE=2',))
            module.compile()
            self.module = module

    def invoke(self, symbol, num_zones, args):
        converted_args = [self.convert(arg) for arg in args]
        if self.mode == 'cpu':
            kernel = getattr(self.module, symbol)
            return kernel(*converted_args)
        if self.mode == 'gpu':
            nb = ((num_zones + block_size - 1) // block_size,)
            bs = (block_size,)
            kernel = self.module.get_function(symbol)
            return kernel(nb, bs, converted_args)

    def convert(self, arg):
        if self.mode == 'cpu':
            if type(arg) == int:
                return c_int(arg)
            if type(arg) == float:
                return c_double(arg)
            if type(arg) == np.ndarray:
                assert(arg.dtype == float)
                return arg.ctypes.data_as(POINTER(c_double))
        if self.mode == 'gpu':
            import cupy
            if type(arg) == int:
                return arg
            if type(arg) == float:
                return arg
            if type(arg) == cupy.ndarray:
                assert(arg.dtype == float)
                return arg.data.ptr
        raise ValueError("kernel arguments must be int, float, or ndarray[float]")


class Memory:
    def __init__(self, mode='cpu'):
        self.mode = mode

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
    def __init__(self, primitive, time=0.0, bc='inflow', coords='cartesian', mode='cpu'):
        self.lib = Library(mode=mode)
        self.mem = Memory(mode=mode)
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

    def new_timestep(self):
        self.time0 = self.time
        self.conserved0[...] = self.conserved1[...]

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

    @property
    def primitive(self):
        self.recompute_primitive()
        return self.mem.copy_to_host(self.primitive1)


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
