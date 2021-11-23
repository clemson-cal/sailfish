import logging
import os
from ctypes import c_double, c_int, POINTER, CDLL
from sailfish.system import build_config


logger = logging.getLogger(__name__)
block_size = 64


"""
Holds either a CPU or GPU dynamically loaded module.
"""
class Library:
    def __init__(self, module_file, mode='cpu'):
        self.mode = mode

        abs_path, _ = os.path.splitext(module_file)
        module = os.path.basename(abs_path)

        logger.info(f'load solver library {module} for {mode} execution')

        with open(f'{abs_path}.c', 'r') as srcfile:
            code = srcfile.read()

        if self.mode in ['cpu', 'omp']:
            import cffi
            import numpy
            ffi = cffi.FFI()
            ffi.set_source(
                module,
                code,
                define_macros=[('EXEC_MODE', dict(cpu=0, omp=1)[mode])],
                extra_compile_args=build_config['extra_compile_args'],
                extra_link_args=build_config['extra_link_args'],
            )
            target = f'lib/{module}.so'
            target = ffi.compile(target=target)
            self.module = CDLL(target)
            self.xp = numpy
        if self.mode == 'gpu':
            import cupy
            module = cupy.RawModule(code=code, options=('-D EXEC_MODE=2',))
            module.compile()
            self.module = module
            self.xp = cupy

    def invoke(self, symbol, num_zones, args):
        converted_args = [self.convert(arg) for arg in args]
        if self.mode in ['cpu', 'omp']:
            kernel = getattr(self.module, symbol)
            return kernel(*converted_args)
        elif self.mode == 'gpu':
            nb = ((num_zones + block_size - 1) // block_size,)
            bs = (block_size,)
            kernel = self.module.get_function(symbol)
            return kernel(nb, bs, converted_args)

    def convert(self, arg):
        if self.mode in ['cpu', 'omp']:
            if type(arg) == int:
                return c_int(arg)
            if type(arg) == float:
                return c_double(arg)
            if type(arg) == self.xp.ndarray:
                assert(arg.dtype == float)
                return arg.ctypes.data_as(POINTER(c_double))
        elif self.mode == 'gpu':
            if type(arg) == int:
                return arg
            if type(arg) == float:
                return arg
            if type(arg) == self.xp.ndarray:
                assert(arg.dtype == float)
                return arg.data.ptr
        raise ValueError("kernel arguments must be int, float, or ndarray[float]")
