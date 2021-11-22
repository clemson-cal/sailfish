from ctypes import c_double, c_int, POINTER, CDLL
import numpy as np


block_size = 64


"""
Holds either a CPU or GPU dynamically loaded module.
"""
class Library:
    def __init__(self, module, mode='cpu'):
        self.mode = mode
        self.load_lib(module)

    def load_lib(self, module):
        with open(f'src/{module}.c', 'r') as srcfile:
            code = srcfile.read()
        if self.mode == 'cpu':
            import cffi
            ffi = cffi.FFI()
            ffi.set_source(
                module,
                code,
                define_macros=[('EXEC_MODE', 0)],
                # extra_compile_args=['-Xpreprocessor', '-fopenmp'],
                # extra_link_args=['-lomp'],
            )
            target = f'lib/{module}.so'
            target = ffi.compile(target=target)
            self.module = CDLL(target)
        if self.mode == 'gpu':
            import cupy
            module = cupy.RawModule(code=code, options=('-D EXEC_MODE=2',))
            module.compile()
            self.module = module
            self.cupy

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
            if type(arg) == int:
                return arg
            if type(arg) == float:
                return arg
            if type(arg) == self.cupy.ndarray:
                assert(arg.dtype == float)
                return arg.data.ptr
        raise ValueError("kernel arguments must be int, float, or ndarray[float]")
