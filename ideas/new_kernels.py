"""
Enables interaction with raw inline C code.

Functions and class members that defer their implementation C code are called
kernels. These functions act on numpy arrays.

This module will be generalized to support GPU kernels.

Author: Jonathan Zrake (2022)
"""


# Python standard library imports
from ctypes import CDLL, POINTER, c_int, c_double
from functools import wraps
from hashlib import sha256
from os import listdir
from os.path import join, dirname
from textwrap import dedent


# Numpy imports
from numpy.typing import NDArray
from numpy import ndarray


KERNEL_VERBOSE_COMPILE = False
KERNEL_DISABLE_CACHE = True
KERNEL_DISABLE_CPU_MODE = False
KERNEL_DISABLE_GPU_MODE = False
KERNEL_DEFAULT_EXEC_MODE = "cpu"

PY_CTYPE_DICT = {
    int: c_int,
    float: c_double,
    NDArray[float]: POINTER(c_double),
}


KERNEL_DEFINE_MACROS_CPU = R"""
#define PRIVATE static
#define PUBLIC

#define FOR_EACH_1D(NI) \
for (int i = 0; i < NI; ++i) \

#define FOR_EACH_2D(NI, NJ) \
for (int i = 0; i < NI; ++i) \
for (int j = 0; j < NJ; ++j) \

#define FOR_EACH_3D(NI, NJ, NK) \
for (int i = 0; i < NI; ++i) \
for (int j = 0; j < NJ; ++j) \
for (int k = 0; k < NK; ++k) \
"""


THREAD_BLOCK_SIZE_1D = (64,)
THREAD_BLOCK_SIZE_2D = (8, 8)
THREAD_BLOCK_SIZE_3D = (4, 4, 4)


KERNEL_DEFINE_MACROS_GPU = R"""
#define PRIVATE static __device__
#define PUBLIC extern "C" __global__

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
"""


def configure_kernel_module(verbose=None,
                            disable_cache=None,
                            disable_cpu_mode=None,
                            disable_gpu_mode=None,
                            default_exec_mode=None):
    """
    Configure the module behavior.

    Calls to this function affect shared module state, so should be called
    from conspicuous / obvious locations of the user application.
    """
    global KERNEL_VERBOSE_COMPILE
    global KERNEL_DISABLE_CACHE
    global KERNEL_DISABLE_CPU_MODE
    global KERNEL_DISABLE_GPU_MODE
    global KERNEL_DEFAULT_EXEC_MODE
    
    if verbose:
        KERNEL_VERBOSE_COMPILE = True
    if disable_cache:
        KERNEL_DISABLE_CACHE = True
    if disable_cpu_mode:
        KERNEL_DISABLE_CPU_MODE = True
    if disable_gpu_mode:
        KERNEL_DISABLE_GPU_MODE = True
    if default_exec_mode is not None:
        if default_exec_mode not in ("cpu", "gpu"):
            raise ValueError("execution mode must be cpu or gpu")
        KERNEL_DEFAULT_EXEC_MODE = default_exec_mode
        

def argtypes(f):
    """
    Return a tuple of ctypes objects derived from a function's type hints.
    """
    return tuple(
        PY_CTYPE_DICT[t] for k, t in f.__annotations__.items() if k != "return"
    )


def restype(f):
    """
    Return a ctypes object for a function's type-hinted return type.
    """
    if "return" in f.__annotations__:
        return PY_CTYPE_DICT[f.__annotations__["return"]]
    else:
        return None


def to_ctypes(args, signature):
    """
    Return a generator that yields pointers from any ndarray arguments.
    """
    for arg, t in zip(args, signature):
        if isinstance(arg, ndarray):
            yield arg.ctypes.data_as(t)
        else:
            yield arg


class ErrorProxyFunction:
    def __init__(self, error):
        self._error = error

    def __call__(self, *args):
        raise self._error


class ErrorProxyModule:
    def __init__(self, error):
        self._error = error

    def __getitem__(self, key):
        return ErrorProxyFunction(self._error)

    def get_function(self, key):
        return ErrorProxyFunction(self._error)


def cpu_extension(code, name, define_macros=list()):
    """
    Either build or load a CPU extension module with the given code and name.

    The string of code is compiled as-is without modifications, except for the
    pre-pending the contents of the `KERNEL_DEFINE_MACROS` module string. The
    resulting build product is cached in the __pycache__ directory, under a
    subdirectory named by the hash (SHA-256) of the code string. If the module
    variable `KERNEL_DISABLE_CACHE` is `False`, then an attempt is made to load
    a module with the given hash, otherwise it is compiled even if a cached
    version was found.

    This method can fail with a `ValueError` if the compilation fails. The
    compiler's stderr should be written to the terminal to aid in identifying
    the compilation error.
    """

    if KERNEL_DISABLE_CPU_MODE:
        return ErrorProxyModule(RuntimeError("CPU mode is disabled"))

    # Add header macros with for-each loops, etc.
    code = KERNEL_DEFINE_MACROS_CPU + code
    verbose = KERNEL_VERBOSE_COMPILE

    # Create a hash for this snippet of code to facilitate caching the
    # build product.
    sha = sha256()
    sha.update(code.encode("utf-8"))
    sha.update(str(define_macros).encode("utf-8"))
    cache_dir = join(dirname(__file__), "__pycache__", sha.hexdigest())

    try:
        from cffi import FFI, VerificationError

        if not KERNEL_DISABLE_CACHE:
            # Attempt to load a cached build product based on the hash value
            # of the source code.
            target = join(
                cache_dir, next(f for f in listdir(cache_dir) if f.endswith(".so"))
            )
            module = CDLL(target)
            if verbose:
                print(f"loaded cached module: {target}")
            return module
    except (FileNotFoundError, StopIteration):
        pass

    except ImportError as e:
        # It should not be fatal if cffi is not available, since GPU kernels
        # could still possibly be used.
        return ErrorProxyModule(e)

    try:
        # If the build product is not already cached, then create it now,
        # and save it to the cache directory. Then load it as a shared
        # library (CDLL).
        ffi = FFI()
        ffi.set_source(name, code, define_macros=define_macros)
        target = ffi.compile(tmpdir=cache_dir or ".", verbose=verbose)
        module = CDLL(target)
        if verbose:
            print(f"compiled module {target}")
        return module

    except VerificationError as e:
        # This is hit when the C compiler fails.
        return ErrorProxyModule(ValueError(f"CPU compilation of {name} failed"))


def gpu_extension(code, name, define_macros=list()):
    if KERNEL_DISABLE_GPU_MODE:
        return ErrorProxyModule(RuntimeError("GPU mode is disabled"))

    try:
        from cupy import RawModule
        from cupy.cuda.compiler import CompileException

        code = KERNEL_DEFINE_MACROS_GPU + code
        options = tuple(f"-D {k}={v}" for k, v in define_macros)
        module = RawModule(code=code, options=options)
        module.compile()
        return module

    except ImportError as e:
        return ErrorProxyModule(e)

    except CompileException as e:
        return ErrorProxyModule(ValueError(f"GPU compilation {name} failed:\n {e}"))


def cpu_extension_function(module, stub, rank):
    """
    Return a function to replace the given stub with a call to a C function.

    The `stub.__name__` attribute is used to find the function in the module.

    The argtypes and restype of the function object are inferred from the stub
    type hints, so the stub must provide type hints for all of its arguments,
    and all of the type hints must map to ctypes objects through the module
    variable `PY_CTYPE_DICT`.

    The first `rank` arguments to the compiled C function must be integers
    representing the shape of the kernel index space. Subsequent arguments
    reflect the stub signature.
    """

    c_func = module[stub.__name__]
    c_func.argtypes = (c_int,) * rank + argtypes(stub)
    c_func.restype = restype(stub)

    @wraps(stub)
    def wrapper(*args):
        shape = stub(*args) or tuple()
        cargs = to_ctypes(shape + args, c_func.argtypes)

        if len(shape) != rank:
            raise ValueError(f"kernel stub must return a tuple of length rank={rank}")
        return c_func(*cargs)

    return wrapper


def gpu_extension_function(module, stub, rank):
    gpu_func = module.get_function(stub.__name__)

    if "return" in stub.__annotations__:
        return ErrorProxyFunction(ValueError(f"GPU kernel {stub.__name__} may not return a value"))

    @wraps(stub)
    def wrapper(*args):
        shape = stub(*args) or tuple()
        cargs = shape + args

        if len(shape) != rank:
            raise ValueError(f"kernel stub must return a tuple of length rank={rank}")

        if rank == 1:
            (ti,) = bs = THREAD_BLOCK_SIZE_1D
            (ni,) = shape
            nb = ((ni + ti - 1) // ti,)
            gpu_func(nb, bs, cargs)

        if rank == 2:
            ti, tj = bs = THREAD_BLOCK_SIZE_2D
            ni, nj = shape
            nb = ((ni + ti - 1) // ti, (nj + tj - 1) // tj)
            gpu_func(nb, bs, cargs)

        if rank == 3:
            ti, tj, tk = bs = THREAD_BLOCK_SIZE_3D
            ni, nj, nk = shape
            nb = ((ni + ti - 1) // ti, (nj + tj - 1) // tj, (nk + tk - 1) // tk)
            gpu_func(nb, bs, cargs)

    return wrapper


def universal_extension_function(cpu_module, gpu_module, stub, rank):
    cpu_func = cpu_extension_function(cpu_module, stub, rank) if cpu_module else None
    gpu_func = gpu_extension_function(gpu_module, stub, rank) if gpu_module else None

    @wraps(stub)
    def wrapper(*args, exec_mode=KERNEL_DEFAULT_EXEC_MODE):
        if exec_mode == "cpu":
            return cpu_func(*args)
        if exec_mode == "gpu":
            return gpu_func(*args)

    return wrapper


def kernel(code: str = None, rank: int = 0):
    """
    Returns a decorator that replaces a 'stub' function with a 'kernel'.

    The C code to be compiled is given either in the `code` variable, or is
    otherwise taken from the stub's doc string (which must then consist
    exclusively of valid C code).

    The stub function is a Python function that simply inspects its arguments
    and returns a shape for the kernel invocation. The kernel is a function
    compiled from C code that can be parallelized in some way, and is compiled
    to either CPU or GPU code.
    """

    def decorator(stub):
        cpu_module = cpu_extension(code or stub.__doc__, stub.__name__)
        gpu_module = gpu_extension(code or stub.__doc__, stub.__name__)
        return universal_extension_function(cpu_module, gpu_module, stub, rank)

    return decorator


def kernel_class(cls):
    """
    A decorator for classes that contains kernel stubs.

    The class instance is given a custom __init__ which compiles the attached
    C code at the time a kernel class is instantiated; the extension module is
    built (or loaded) lazily. This also creates an opportunity for the calling
    Python functions to manipulate the kernel class C code in custom ways
    for each kernel class instance.

    If the wrapped class has a property `define_macros` (list of key-value
    tuples), its value will be passed to the `cpu_extension` function.
    """

    cls_init = cls.__init__

    def __init__(self, **kwargs):
        cls_init(self, **kwargs)

        code = cls.__doc__
        kernels = list()

        for k in dir(cls):
            f = getattr(self, k)
            if hasattr(f, "__kernel_code"):
                code += f.__kernel_code or dedent(f.__doc__)
                kernels.append(k)

        define_macros = getattr(self, "define_macros", list())
        cpu_module = cpu_extension(code, cls.__name__, define_macros)
        gpu_module = gpu_extension(code, cls.__name__, define_macros)

        for kernel in kernels:
            stub = getattr(self, kernel)
            rank = stub.__kernel_rank
            func = universal_extension_function(cpu_module, gpu_module, stub, rank)
            setattr(self, kernel, func)

    cls.__init__ = __init__
    return cls


def kernel_method(rank: int = 0, code: str = None):
    """
    A decorator for class methods to be implemented as compiled C code.
    """

    def decorator(stub):
        stub.__kernel_rank = rank
        stub.__kernel_code = code
        return stub

    return decorator


if __name__ == "__main__":
    # ==============================================================================
    # Example usage of kernel functions and classes
    # ==============================================================================

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "--mode",
        dest="exec_mode",
        default="cpu",
        choices=["cpu", "gpu"],
        help="execution mode",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="verbose output from extension compile stages",
    )
    args = parser.parse_args()

    if args.exec_mode == "cpu":
        from numpy import array, linspace, zeros_like
    if args.exec_mode == "gpu":
        from cupy import array, linspace, zeros_like

    configure_kernel_module(verbose=args.verbose, default_exec_mode=args.exec_mode)

    # ==============================================================================
    # 1.
    #
    # Demonstrates a rank-0 kernel function. The C source for this example is
    # provided as an argument to the kernel decorator function. Note that native
    # functions that return a value cannot be GPU kernels, but they can be CPU
    # kernels.
    # ==============================================================================

    code = R"""
    PUBLIC double multiply(int a, double b)
    {
        return a * b;
    }
    """

    @kernel(code)
    def multiply(a: int, b: float) -> float:
        """
        Return the product of an integer a and a float b.
        """
        pass

    if args.exec_mode != "gpu":
        assert multiply(5, 25.0) == 125.0

    # ==============================================================================
    # 2.
    #
    # Demonstrates a rank-1 kernel function. The C source for this example is
    # written in the doc string.
    # ==============================================================================

    @kernel(rank=1)
    def rank_one_kernel(a: float, x: NDArray[float], y: NDArray[float]):
        R"""
        PUBLIC void rank_one_kernel(int ni, double a, double *x, double *y)
        {
            FOR_EACH_1D(ni)
            {
                y[i] = x[i] * a;
            }
        }
        """
        if x.shape != y.shape:
            raise ValueError("input and output arrays have different shapes")
        return x.shape

    a = linspace(0.0, 1.0, 5000)
    b = zeros_like(a)
    rank_one_kernel(0.25, a, b)
    assert (b == 0.25 * a).all()

    # ==============================================================================
    # 3.
    #
    # Demonstrates a kernel class, containing two kernel methods where the C
    # code is written in the stub doc strings. Code in the class doc string is
    # included at the top of the resulting source code.
    # ==============================================================================

    @kernel_class
    class Solver:
        R"""
        static const double gamma_law_index = 5.0 / 3.0;
        """

        @kernel_method(rank=1)
        def conserved_to_primitive(self, u: NDArray[float], p: NDArray[float]):
            R"""
            //
            // Compute the conversion of conserved variables to primitive ones.
            //
            PUBLIC void conserved_to_primitive(int ni, double *u, double *p)
            {
                FOR_EACH_1D(ni)
                {
                    double rho = u[5 * i + 0];
                    double px  = u[5 * i + 1];
                    double py  = u[5 * i + 2];
                    double pz  = u[5 * i + 3];
                    double nrg = u[5 * i + 4];
                    double p_squared = px * px + py * py + pz * pz;

                    p[5 * i + 0] = rho;
                    p[5 * i + 1] = px / rho;
                    p[5 * i + 2] = py / rho;
                    p[5 * i + 3] = py / rho;
                    p[5 * i + 4] = (nrg - 0.5 * p_squared / rho) * (gamma_law_index - 1.0);
                }
            }
            """
            if p.shape[-1] != 5:
                raise ValueError("p.shape[-1] must be 5")
            if p.shape != u.shape:
                raise ValueError("p and u must have the same shape")
            return (p.size // 5,)

        @kernel_method(rank=1)
        def primitive_to_conserved(self, p: NDArray[float], u: NDArray[float]):
            R"""
            //
            // Compute the conversion of primitive variables to conserved ones.
            //
            PUBLIC void primitive_to_conserved(int ni, double *p, double *u)
            {
                FOR_EACH_1D(ni)
                {
                    double rho = p[5 * i + 0];
                    double vx  = p[5 * i + 1];
                    double vy  = p[5 * i + 2];
                    double vz  = p[5 * i + 3];
                    double pre = p[5 * i + 4];
                    double v_squared = vx * vx + vy * vy + vz * vz;

                    u[5 * i + 0] = rho;
                    u[5 * i + 1] = vx * rho;
                    u[5 * i + 2] = vy * rho;
                    u[5 * i + 3] = vz * rho;
                    u[5 * i + 4] = 0.5 * rho * v_squared + pre / (gamma_law_index - 1.0);
                }
            }
            """
            if u.shape[-1] != 5:
                raise ValueError("u.shape[-1] must be 5")
            if u.shape != p.shape:
                raise ValueError("u and p must have the same shape")
            return (u.size // 5,)

    solver = Solver()
    p = array([[1.0, 0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0, 1.0]])
    u = zeros_like(p)
    q = zeros_like(p)

    solver.primitive_to_conserved(p, u)
    solver.conserved_to_primitive(u, q)

    assert (p == q).all()
