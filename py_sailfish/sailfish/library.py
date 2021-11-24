import tempfile
import logging
import os
from ctypes import c_double, c_int, POINTER, CDLL
from sailfish.system import build_config


logger = logging.getLogger(__name__)
block_size = 64


"""
Builds and maintains (in memory) a CPU or GPU dynamically compiled module.

CPU modules are built with the cffi module. Build products including the .so
file itself are placed in a temporary directory, and removed as soon as the
module is loaded in memory. GPU modules are compiled with cupy.

The C source code must adhere to several conventions:

1. Be on the filesystem alongside `module_file`, with .c extension replacing
   the .py extension
2. Define a set of public API methods, or kernels
3. Kernel functions are implemented in one of three modes: cpu=0, omp=1, or
   gpu=2, specified to the C code as `EXEC_MODE`

The kernel functions are configured by preprocessor macros to:

- In CPU mode: wrap the function body in a serialized for-loop
- In OMP mode: wrap the function body in an OpenMP-parallelized for-loop
- In GPU mode: discover the kernel index and execute the function body once

The kernel functions and their argument signature are not specified here. It
is the responsibility of the user of the `Library` object to invoke the kernel
functions with appropriate arguments. Kernels must be void functions, and only
accept arguments in the form of int (or enum), double, or pointer-to-double.
"""


class Library:
    def __init__(self, module_file, mode="cpu"):
        self.mode = mode

        abs_path, _ = os.path.splitext(module_file)
        module = os.path.basename(abs_path)

        logger.info(f"load solver library {module} for {mode} execution")

        with open(f"{abs_path}.c", "r") as srcfile:
            code = srcfile.read()

        if self.mode in ["cpu", "omp"]:
            import cffi
            import numpy

            ffi = cffi.FFI()
            ffi.set_source(
                module,
                code,
                define_macros=[("EXEC_MODE", dict(cpu=0, omp=1)[mode])],
                extra_compile_args=build_config["extra_compile_args"],
                extra_link_args=build_config["extra_link_args"],
            )
            with tempfile.TemporaryDirectory() as tmpdir:
                target = ffi.compile(tmpdir=tmpdir)
                self.module = CDLL(target)
            self.xp = numpy
        if self.mode == "gpu":
            import cupy

            module = cupy.RawModule(code=code, options=("-D EXEC_MODE=2",))
            module.compile()
            self.module = module
            self.xp = cupy

    def invoke(self, symbol, num_zones, args):
        """
        Invoke a function in the library.

        The kernel name `symbol` must exist in the dynamic library. No
        argument checking is possible here; `args` must be a tuple containing
        the correct data types for the function signature, otherwise stack
        correction is likely. The argument `num_zones` must be supplied for
        GPU kernel launches, and is used to determine the total number of
        threads involved in the launch.
        """
        converted_args = [self.convert(arg) for arg in args]
        if self.mode in ["cpu", "omp"]:
            kernel = getattr(self.module, symbol)
            kernel(*converted_args)
        elif self.mode == "gpu":
            nb = ((num_zones + block_size - 1) // block_size,)
            bs = (block_size,)
            kernel = self.module.get_function(symbol)
            kernel(nb, bs, converted_args)

    def convert(self, arg):
        """
        Prepare an argument to be passed to the kernel function.
        """
        if self.mode in ["cpu", "omp"]:
            if type(arg) == int:
                return c_int(arg)
            if type(arg) == float:
                return c_double(arg)
            if type(arg) == self.xp.ndarray:
                assert arg.dtype == float
                return arg.ctypes.data_as(POINTER(c_double))
        elif self.mode == "gpu":
            if type(arg) == int:
                return arg
            if type(arg) == float:
                return arg
            if type(arg) == self.xp.ndarray:
                assert arg.dtype == float
                return arg.data.ptr
        raise ValueError("kernel arguments must be int, float, or ndarray[float]")
