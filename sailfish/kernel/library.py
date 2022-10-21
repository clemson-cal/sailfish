"""
Defines a `Library` utility class to encapulsate CPU/GPU compiled kernels.

CPU modules are built with the cffi module. Build products including the .so
file itself are placed in this module's __pycache__ directory, and stored for
reuse based on the SHA value of the source code and #define macros. GPU
modules are JIT-compiled with cupy. No caching is presently done for the GPU
modules.
"""

from platform import system
from ctypes import c_double, c_int, POINTER, CDLL
from hashlib import sha256
from logging import getLogger
from os import listdir
from os.path import join, dirname

from .parse_api import parse_api
from .system import build_config, measure_time

logger = getLogger(__name__)
THREAD_BLOCK_SIZE_1D = (64,)
THREAD_BLOCK_SIZE_2D = (8, 8)
THREAD_BLOCK_SIZE_3D = (4, 4, 4)

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

#define FOR_EACH_2D(NI, NJ) \
for (int i = 0; i < NI; ++i) \
for (int j = 0; j < NJ; ++j) \

#define FOR_EACH_3D(NI, NJ, NK) \
for (int i = 0; i < NI; ++i) \
for (int j = 0; j < NJ; ++j) \
for (int k = 0; k < NK; ++k) \

#elif (EXEC_MODE == EXEC_OMP)
#define FOR_EACH_1D(NI) \
_Pragma("omp parallel for") \
for (int i = 0; i < NI; ++i) \

#define FOR_EACH_2D(NI, NJ) \
_Pragma("omp parallel for") \
for (int i = 0; i < NI; ++i) \
for (int j = 0; j < NJ; ++j) \

#define FOR_EACH_3D(NI, NJ, NK) \
_Pragma("omp parallel for") \
for (int i = 0; i < NI; ++i) \
for (int j = 0; j < NJ; ++j) \
for (int k = 0; k < NK; ++k) \

#elif (EXEC_MODE == EXEC_GPU)
#define FOR_EACH_1D(NI) \
int i = threadIdx.x + blockIdx.x * blockDim.x; \
if (i >= NI) return; \

#define FOR_EACH_2D(NI, NJ) \
int i = threadIdx.x + blockIdx.x * blockDim.x; \
int j = threadIdx.y + blockIdx.y * blockDim.y; \
if (i >= NI || j >= NJ) return; \

#define FOR_EACH_3D(NI, NJ, NK) \
int i = threadIdx.x + blockIdx.x * blockDim.x; \
int j = threadIdx.y + blockIdx.y * blockDim.y; \
int k = threadIdx.z + blockIdx.z * blockDim.z; \
if (i >= NI || j >= NJ || k >= NK) return; \

#endif
"""


class KernelInvocation:
    """
    A kernel whose execution shape is specified and is ready to be invoked.
    """

    def __init__(self, kernel, shape):
        self.kernel = kernel
        self.shape = shape

    def __call__(self, *args):
        lib = self.kernel.lib
        rank = len(self.shape)
        name = self.kernel.symbol.name
        spec = self.kernel.symbol.args

        if lib.cpu_mode:
            kernel = getattr(lib.module, name)
        else:
            kernel = lib.module.get_function(name)

        args = list(self.shape) + list(args)

        if lib.debug:
            validate_types(args, tuple(spec), name, lib.xp)
            validate_constraints(args, tuple(spec), name)

        if lib.cpu_mode:
            kernel(*to_ctypes(args, spec))
        else:
            if rank == 1:
                (ti,) = bs = THREAD_BLOCK_SIZE_1D
                (ni,) = self.shape
                nb = ((ni + ti - 1) // ti,)
                kernel(nb, bs, args)

            elif rank == 2:
                ti, tj = bs = THREAD_BLOCK_SIZE_2D
                ni, nj = self.shape
                nb = ((ni + ti - 1) // ti, (nj + tj - 1) // tj)
                kernel(nb, bs, args)

            elif rank == 3:
                ti, tj, tk = bs = THREAD_BLOCK_SIZE_3D
                ni, nj, nk = self.shape
                nb = ((ni + ti - 1) // ti, (nj + tj - 1) // tj, (nk + tk - 1) // tk)
                kernel(nb, bs, args)


class Kernel:
    """
    An object that uses `__getitem__` syntax to return a
    :py:class:`KernelInvocation` instance.
    """

    def __init__(self, lib, symbol):
        self.lib = lib
        self.symbol = symbol

    def __getitem__(self, shape):
        if type(shape) == int:
            return self[(shape,)]
        if len(shape) != self.symbol.rank:
            raise ValueError(
                f"incompatible shape {shape} for kernel "
                f"{self.symbol.name} with rank {self.symbol.rank}"
            )
        else:
            return KernelInvocation(self, shape)


class Library:
    """
    Builds and maintains (in memory) a CPU or GPU dynamically compiled module.
    """

    def __init__(
        self, code=None, mode="cpu", name="module", debug=True, define_macros=dict()
    ):
        code = f"{KERNEL_LIB_HEADER} {code}"
        logger.info(f"debug mode {'enabled' if debug else 'disabled'}")
        logger.info(f"prepare {name} for {mode} execution")

        with measure_time(mode) as prep_time:
            self.debug = debug
            self.cpu_mode = mode != "gpu"
            self.api = parse_api(code)

            if self.cpu_mode:
                self.load_cpu_module(code, name, mode=mode, define_macros=define_macros)
            else:
                self.load_gpu_module(code, define_macros)

            logger.info(f"module preparation took {prep_time():0.3}s")

        for symbol in self.api:
            logger.info(f"+-- {symbol}")

    def load_cpu_module(self, code, name, mode="cpu", define_macros=dict()):
        import cffi
        import numpy

        if mode == "omp" and not build_config["enable_openmp"]:
            raise ValueError("need enable_openmp=True to compile with mode=omp")

        if system() == "Windows":
            # The ctypes is not finding library symbols for modules
            # JIT-compiled with the cffi module. We don't know why this is
            # happening, but there might be clues at the link below. Please
            # help us.
            #
            # https://cffi.readthedocs.io/en/latest/using.html#windows-calling-conventions
            raise ValueError("CPU execution mode not supported on windows")

        exec_mode = dict(cpu=0, omp=1)[mode]
        define_macros = list(define_macros.items()) + [("EXEC_MODE", exec_mode)]

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
        sha = sha256()
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
            target = ffi.compile(cache_dir, verbose=False)
            self.module = CDLL(target)
            logger.info(f"recompile library")

        self.xp = numpy

    def load_gpu_module(self, code, define_macros):
        import cupy

        options = tuple(f"-D {k}={v}" for k, v in define_macros.items()) + (
            "-D EXEC_MODE=2",
        )
        module = cupy.RawModule(code=code, options=options)
        module.compile()
        self.module = module
        self.xp = cupy

    def __getattr__(self, symbol):
        return Kernel(self, self.api[symbol])


def to_ctypes(args, spec):
    """
    Coerce a sequence of values to their appropriate ctype.

    The expected type is determined from the `spec` list.
    """
    for arg, (typename, _, _) in zip(args, spec):
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


def validate_types(args, spec, symbol, xp):
    if len(args) != len(spec):
        raise arglen_error(symbol, args, spec)

    for n, (arg, (typename, argname, constraint)) in enumerate(zip(args, spec)):
        if typename == "int":
            if type(arg) not in [int, xp.int32]:
                raise type_error(symbol, n, arg, "int")
        elif typename == "double":
            if type(arg) not in [float, xp.float64]:
                raise type_error(symbol, n, arg, "float64")
        elif typename == "double*":
            if type(arg) is not xp.ndarray:
                raise type_error(symbol, n, arg, "ndarray")
            if arg.dtype != xp.float64:
                raise dtype_error(symbol, n, arg, "float64")
            if not arg.flags["C_CONTIGUOUS"]:
                raise layout_error(symbol, n)


def validate_constraints(args, spec, symbol):
    """
    Validate kernel argument constraints for a symbol.

    Constraints are optionally defined in C code and extracted in the
    `parse_api` module.
    """
    scope = dict(zip([a[1] for a in spec], args))
    for arg, (_, name, constraint) in zip(args, spec):
        if constraint:
            c = constraint.replace("$", name)
            if not eval(c, None, scope):
                raise ValueError(f"argument constraint for {symbol} not satisfied: {c}")
