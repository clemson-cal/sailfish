import tempfile
import logging
import os
from ctypes import c_double, c_int, POINTER, CDLL
from sailfish.system import build_config
from sailfish.parse_api import parse_api

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

Supported arguments to kernel functions are int, double, or pointer-to-double.
"""


class Library:
    def __init__(self, module_file, mode="cpu"):
        abs_path, _ = os.path.splitext(module_file)
        module = os.path.basename(abs_path)
        logger.info(f"load solver library {module} for {mode} execution")
        filename = f"{abs_path}.c"
        self.cpu_mode = mode != "gpu"
        self.api = parse_api(filename)

        for symbol in self.api:
            logger.info(f"- {symbol}")

        with open(filename, "r") as srcfile:
            code = srcfile.read()

        if self.cpu_mode:
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
        else:
            import cupy

            module = cupy.RawModule(code=code, options=("-D EXEC_MODE=2",))
            module.compile()
            self.module = module
            self.xp = cupy

    def __getattr__(self, symbol):
        expected_args = self.api[symbol]

        if self.cpu_mode:
            kernel = getattr(self.module, symbol)

            def invoke_cpu_kernel(*args):
                kernel(
                    *convert_args(self.cpu_mode, self.xp, symbol, expected_args, *args)
                )

            return invoke_cpu_kernel
        else:
            kernel = self.module.get_function(symbol)

            def invoke_gpu_kernel(*args):
                num_zones = args[0]
                nb = ((num_zones + block_size - 1) // block_size,)
                bs = (block_size,)
                kernel(
                    nb,
                    bs,
                    convert_args(self.cpu_mode, self.xp, symbol, expected_args, *args),
                )

            return invoke_cpu_kernel


def convert_args(cpu_mode, xp, symbol, expected_args, *args):
    if len(args) != len(expected_args):
        raise TypeError(
            f"{symbol} takes exactly {len(expected_args)} arguments ({len(args)}) given"
        )
    for n, (arg, (typename, argname, constraint)) in enumerate(
        zip(args, expected_args)
    ):
        if typename == "int":
            if type(arg) is not int:
                raise TypeError(
                    f"argument {n} to {symbol} has type {type(arg)}, expected int"
                )
            if cpu_mode:
                yield c_int(arg)
            else:
                yield arg
        elif typename == "double":
            if type(arg) is not float:
                raise TypeError(
                    f"argument {n} to {symbol} has type {type(arg)}, expected float64"
                )
            if cpu_mode:
                yield c_double(arg)
            else:
                assert type(arg) is float
                yield arg
        elif typename == "double*":
            if type(arg) is not xp.ndarray:
                raise TypeError(
                    f"argument {n} to {symbol} has type {type(arg)}, expected ndarray"
                )
            if arg.dtype != xp.float64:
                raise TypeError(
                    f"argument {n} to {symbol} has dtype {arg.dtype}, expected float64"
                )
            if cpu_mode:
                yield arg.ctypes.data_as(POINTER(c_double))
            else:
                yield arg
