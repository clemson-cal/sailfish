"""
Defines a `Library` utility class to encapulsate CPU/GPU compiled kernels.
"""

import logging
import hashlib
import time
from os import listdir
from os.path import join, splitext, basename, dirname
from ctypes import c_double, c_int, POINTER, CDLL
from kernel_lib.system import build_config, measure_time
from kernel_lib.parse_api import parse_api

logger = logging.getLogger(__name__)
THREAD_BLOCK_SIZE = 64

KERNEL_LIB_HEADER = r"""
#define EXEC_CPU 0
#define EXEC_OMP 1
#define EXEC_GPU 2

#if (EXEC_MODE != EXEC_GPU)
#include <math.h>
#include <stddef.h>
#define PRIVATE static
#define PUBLIC
#else
#define PRIVATE static __device__
#define PUBLIC extern "C" __global__
#endif

#if (EXEC_MODE == EXEC_CPU)
#define FOR_EACH_1D(NI) \
for (int i = 0; i < NI; ++i) \

#elif (EXEC_MODE == EXEC_OMP)
#define FOR_EACH_1D(NI) \
_Pragma("omp parallel for") \
for (int i = 0; i < NI; ++i) \

#elif (EXEC_MODE == EXEC_GPU)
#define FOR_EACH_1D(NI) \
int i = threadIdx.x + blockIdx.x * blockDim.x; \
if (i >= NI) return; \

#endif

#if (EXEC_MODE == EXEC_CPU)
#define FOR_EACH_2D(NI, NJ) \
for (int i = 0; i < NI; ++i) \
for (int j = 0; j < NJ; ++j) \

#elif (EXEC_MODE == EXEC_OMP)
#define FOR_EACH_2D(NI, NJ) \
_Pragma("omp parallel for") \
for (int i = 0; i < NI; ++i) \
for (int j = 0; j < NJ; ++j) \

#elif (EXEC_MODE == EXEC_GPU)
#define FOR_EACH_2D(NI, NJ) \
int i = threadIdx.x + blockIdx.x * blockDim.x; \
if (i >= NI) return; \

#endif
"""


class Library:
    """
    Builds and maintains (in memory) a CPU or GPU dynamically compiled module.

    CPU modules are built with the cffi module. Build products including the
    .so file itself are placed in a temporary directory, and removed as soon
    as the module is loaded in memory. GPU modules are compiled with cupy.

    The C source code must adhere to several conventions:

    1. Be on the filesystem alongside `module_file`, with .c extension
       replacing the .py extension
    2. Define a set of public API methods, or kernels
    3. Kernel functions are implemented in one of three modes: cpu=0, omp=1,
       or gpu=2, specified to the C code as `EXEC_MODE`

    The kernel functions are configured by preprocessor macros to:

    - In CPU mode: wrap the function body in a serialized for-loop
    - In OMP mode: wrap the function body in an OpenMP-parallelized for-loop
    - In GPU mode: discover the kernel index and execute the function body
      once

    Supported arguments to kernel functions are int, double, or
    pointer-to-double.
    """

    def __init__(self, code=None, mode="cpu", name="module", debug=True):
        code = f"{KERNEL_LIB_HEADER} {code}"
        logger.info(f"debug mode {'enabled' if debug else 'disabled'}")
        logger.info(f"prepare module {name} for {mode} execution")

        with measure_time() as prep_time:
            self.debug = debug
            self.cpu_mode = mode != "gpu"
            self.api = parse_api(code)

            if self.cpu_mode:
                self.load_cpu_module(code, name, mode=mode)
            else:
                self.load_gpu_module(code)

            logger.info(f"module preparation took {prep_time():0.3}s")

        for symbol in self.api:
            logger.info(f"+-- {symbol}")

    def load_cpu_module(self, code, name, mode="cpu"):
        import cffi
        import numpy

        define_macros = [("EXEC_MODE", dict(cpu=0, omp=1)[mode])]
        ffi = cffi.FFI()
        ffi.set_source(
            name,
            code,
            define_macros=define_macros,
            extra_compile_args=build_config["extra_compile_args"],
            extra_link_args=build_config["extra_link_args"],
        )

        # Build a hash for the compiled library based on code, define
        # macros, and build args.
        sha = hashlib.sha256()
        sha.update(code.encode("utf-8"))
        sha.update(str(define_macros).encode("utf-8"))
        sha.update(str(build_config).encode("utf-8"))
        cache_dir = join(dirname(__file__), "__pycache__", sha.hexdigest())

        try:
            so_name = join(
                cache_dir,
                next(filter(lambda f: f.endswith(".so"), listdir(cache_dir))),
            )
            self.module = CDLL(so_name)
            logger.info(f"load cached library")
        except (FileNotFoundError, StopIteration) as e:
            target = ffi.compile(cache_dir)
            self.module = CDLL(target)
            logger.info(f"recompile library")

        self.xp = numpy

    def load_gpu_module(self, code):
        import cupy

        module = cupy.RawModule(code=code, options=("-D EXEC_MODE=2",))
        module.compile()
        self.module = module
        self.xp = cupy

    def __getattr__(self, symbol):
        arg_format = self.api[symbol]

        if self.cpu_mode:
            kernel = getattr(self.module, symbol)
        else:
            kernel = self.module.get_function(symbol)

        def invoke_kernel(*args):
            if self.debug:
                validate_types(args, arg_format, symbol, self.xp)
                validate_constraints(args, arg_format, symbol)

            if self.cpu_mode:
                kernel(*to_ctypes(args, arg_format))
            else:
                num_zones = args[0]
                nb = ((num_zones + THREAD_BLOCK_SIZE - 1) // THREAD_BLOCK_SIZE,)
                bs = (THREAD_BLOCK_SIZE,)
                kernel(nb, bs, args)

        return invoke_kernel


def to_ctypes(args, arg_format):
    """
    Coerce a sequence of values to their appropriate ctype.

    The expected type is determined from the `arg_format` list.
    """
    for arg, (typename, _, _) in zip(args, arg_format):
        if typename == "int":
            yield c_int(arg)
        elif typename == "double":
            yield c_double(arg)
        elif typename == "double*":
            yield arg.ctypes.data_as(POINTER(c_double))


def type_error(sym, n, a, b):
    return TypeError(f"argument {n} to {sym} has type {type(a).__name__}, expected {b}")


def dtype_error(sym, n, a, b):
    return TypeError(f"argument {n} to {sym} has dtype {a.dtype}, expected {b}")


def arglen_error(sym, a, b):
    return TypeError(f"{sym} takes exactly {len(b)} arguments ({len(a)} given)")


def layout_error(sym, n):
    return TypeError(f"arg {n} to {sym} is not c-contiguous")


def validate_types(args, arg_format, symbol, xp):
    if len(args) != len(arg_format):
        raise arglen_error(symbol, args, arg_format)

    for n, (arg, (typename, argname, constraint)) in enumerate(zip(args, arg_format)):
        if typename == "int":
            if type(arg) is not int:
                raise type_error(symbol, n, arg, "int")
        elif typename == "double":
            if type(arg) is not float:
                raise type_error(symbol, n, arg, "float64")
        elif typename == "double*":
            if type(arg) is not xp.ndarray:
                raise type_error(symbol, n, arg, "ndarray")
            if arg.dtype != xp.float64:
                raise dtype_error(symbol, n, arg, "float64")
            if not arg.flags["C_CONTIGUOUS"]:
                raise layout_error(symbol, n)


def validate_constraints(args, arg_format, symbol):
    """
    Validate kernel argument constraints for a symbol.

    Constraints are optionally defined in C code and extracted in the
    `parse_api` module.
    """
    scope = dict(zip([a[1] for a in arg_format], args))
    for arg, (_, name, constraint) in zip(args, arg_format):
        if constraint:
            c = constraint.replace("$", name)
            if not eval(c, None, scope):
                raise ValueError(f"argument constraint for {symbol} not satisfied: {c}")
