"""
Enables interaction with embedded C or CUDA code.

Functions and class members that defer their implementation C code are called
kernels. Kernels are meant to crunch numbers. They act on numpy or cupy
arrays.

Author: Jonathan Zrake (2022)
"""


# Python standard library imports
from ctypes import CDLL, POINTER, c_int, c_double, Structure
from functools import wraps
from hashlib import sha256
from os import listdir
from os.path import join, dirname
from textwrap import dedent
from time import perf_counter
from logging import getLogger

# Numpy imports
from numpy.typing import NDArray
from numpy import ndarray, array

logger = getLogger("sailfish")

KERNEL_VERBOSE_COMPILE = False  # passed to CFFI
KERNEL_DISABLE_CACHE = False
KERNEL_DISABLE_CPU_MODE = False
KERNEL_DISABLE_GPU_MODE = False
KERNEL_DEFAULT_EXEC_MODE = "cpu"

PY_CTYPE_DICT = {
    int: c_int,
    float: c_double,
    NDArray[float]: POINTER(c_double),
    NDArray[int]: POINTER(c_int),
}


KERNEL_DEFINE_MACROS_CPU = R"""
#define CPU_MODE
#define DEVICE static
#define KERNEL

#define FOR_RANGE_1D(I0, I1) \
for (int i = I0; i < I1; ++i) \

#define FOR_RANGE_2D(I0, I1, J0, J1) \
for (int i = I0; i < I1; ++i) \
for (int j = J0; j < J1; ++j) \

#define FOR_RANGE_3D(I0, I1, J0, J1, K0, K1) \
for (int i = I0; i < I1; ++i) \
for (int j = J0; j < J1; ++j) \
for (int k = K0; k < K1; ++k) \

#define FOR_EACH_1D(NI) FOR_RANGE_1D(0, NI)
#define FOR_EACH_2D(NI, NJ) FOR_RANGE_2D(0, NI, 0, NJ)
#define FOR_EACH_3D(NI, NJ, NK) FOR_RANGE_3D(0, NI, 0, NJ, 0, NK)
"""


KERNEL_DEFINE_MACROS_GPU = R"""
#define GPU_MODE
#define DEVICE static __device__
#define KERNEL extern "C" __global__

#define FOR_RANGE_1D(I0, I1) \
int i = threadIdx.x + blockIdx.x * blockDim.x; \
if (i < I0 || i >= I1) return; \

#define FOR_RANGE_2D(I0, I1, J0, J1) \
int i = threadIdx.x + blockIdx.x * blockDim.x; \
int j = threadIdx.y + blockIdx.y * blockDim.y; \
if (i < I0 || i >= I1 || j < J0 || j >= J1) return; \

#define FOR_RANGE_3D(I0, I1, J0, J1, K0, K1) \
int i = threadIdx.x + blockIdx.x * blockDim.x; \
int j = threadIdx.y + blockIdx.y * blockDim.y; \
int k = threadIdx.z + blockIdx.z * blockDim.z; \
if (i < I0 || i >= I1 || j < J0 || j >= J1 || K < K0 || K >= K1) return; \

#define FOR_EACH_1D(NI) FOR_RANGE_1D(0, NI)
#define FOR_EACH_2D(NI, NJ) FOR_RANGE_2D(0, NI, 0, NJ)
#define FOR_EACH_3D(NI, NJ, NK) FOR_RANGE_3D(0, NI, 0, NJ, 0, NK)
"""


THREAD_BLOCK_SIZE_1D = (64,)
THREAD_BLOCK_SIZE_2D = (8, 8)
THREAD_BLOCK_SIZE_3D = (4, 4, 4)


def perf_time_sequence(mode):
    """
    Generate a sequence of time differences between subsequent yield's.

    This is a useful utility function for crude profiling of iteration-based
    programs. It yields the number of seconds that have elapsed since the
    previous yield.

    WARNING: it does a device-wide synchronization each time it is called! So,
    do not call `next` on it in-line with dispatching work to multiple devices
    or streams.
    """

    def impl(mode):
        last = perf_counter()
        yield
        while True:
            if mode == "gpu":
                from cupy.cuda.runtime import deviceSynchronize

                deviceSynchronize()

            now = perf_counter()
            yield now - last
            last = now

    g = impl(mode)
    g.send(None)
    return g


def configure_kernel_module(
    verbose=None,
    disable_cache=None,
    disable_cpu_mode=None,
    disable_gpu_mode=None,
    default_exec_mode=None,
):
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

    logger.debug(f"KERNEL_VERBOSE_COMPILE={KERNEL_VERBOSE_COMPILE}")
    logger.debug(f"KERNEL_DISABLE_CACHE={KERNEL_DISABLE_CACHE}")
    logger.debug(f"KERNEL_DISABLE_CPU_MODE={KERNEL_DISABLE_CPU_MODE}")
    logger.debug(f"KERNEL_DISABLE_GPU_MODE={KERNEL_DISABLE_GPU_MODE}")
    logger.debug(f"KERNEL_DEFAULT_EXEC_MODE={KERNEL_DEFAULT_EXEC_MODE}")


def argtype(t):
    if issubclass(t, Structure):
        return t
    else:
        return PY_CTYPE_DICT[t]


def argtypes(f):
    """
    Return a tuple of ctypes objects derived from a function's type hints.
    """
    return tuple(argtype(t) for k, t in f.__annotations__.items() if k != "return")


def restype(f):
    """
    Return a ctypes object for a function's type-hinted return type.
    """
    if "return" in f.__annotations__:
        return PY_CTYPE_DICT[f.__annotations__["return"]]
    else:
        return None


def to_cpu_args(args, signature):
    """
    Return a generator that yields arguments suitable for args to a CPU kernel
    """
    for a, t in zip(args, signature):
        if isinstance(a, ndarray):
            yield a.ctypes.data_as(t)
        elif isinstance(a, Structure):
            yield a
        else:
            yield a


def to_gpu_args(args):
    """
    Return a generator that yields arguments suitable for args to a GPU kernel
    """
    for a in args:
        if isinstance(a, Structure):
            # Note: array(a) should work in principle here, but it generates a
            # warning due to a Python bug. See URL below:
            #
            # https://github.com/python/cpython/issues/54953
            yield array(memoryview(a).tobytes())
        else:
            yield a


class MissingFunction:
    """
    Represents a kernel function that failed to be created for some reason.

    Instances of this class are used when the failure should be silently
    forgiven unless calling code tries to invoke the function that could not
    be compiled. The main use case is where either cffi or cupy is not
    installed, so compilation of the respective CPU or GPU extension was not
    possible. This should not trigger an error unless code tries to invoke the
    kernel in the mode that was unavailable.
    """

    def __init__(self, error):
        self._error = error

    def __call__(self, *args):
        raise self._error


class MissingModule:
    """
    A CPU or GPU extension module that could not be created for some reason.
    """

    def __init__(self, error):
        self._error = error

    def __getitem__(self, key):
        return MissingFunction(self._error)

    def get_function(self, key):
        return MissingFunction(self._error)


def define_macros_string(define_macros):
    return ", ".join(f"{k.lower()}={v}" for k, v in define_macros)


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
        logger.debug(f"KERNEL_DISABLE_CPU_MODE=True; skip CPU extension")
        return MissingModule(RuntimeError("invoke skipped CPU extension"))

    # Add header macros with for-each loops, etc.
    code = KERNEL_DEFINE_MACROS_CPU + code
    verbose = KERNEL_VERBOSE_COMPILE

    # Create a hash for this snippet of code to facilitate caching the
    # build product.
    sha = sha256()
    sha.update(code.encode("utf-8"))
    sha.update(str(define_macros).encode("utf-8"))
    cache_dir = join(dirname(__file__), "__pycache__", sha.hexdigest())
    define_str = define_macros_string(define_macros)

    try:
        from cffi import FFI, VerificationError

        if KERNEL_DISABLE_CACHE:
            logger.debug(f"cache disabled")
        else:
            # Attempt to load a cached build product based on the hash value
            # of the source code.
            target = join(
                cache_dir, next(f for f in listdir(cache_dir) if f.endswith(".so"))
            )
            module = CDLL(target)
            logger.info(f"load cached module {name}({define_str})")
            # logger.debug(f"cached library filename {target}")
            return module

    except (FileNotFoundError, StopIteration):
        logger.debug(f"no cache found for module {name}")

    except ImportError as e:
        # It should not be fatal if cffi is not available, since GPU kernels
        # could still possibly be used.
        logger.debug(f"{e}; skip CPU extension")
        return MissingModule(e)

    try:
        # If the build product is not already cached, then create it now,
        # and save it to the cache directory. Then load it as a shared
        # library (CDLL).
        ffi = FFI()
        ffi.set_source(
            name,
            code,
            define_macros=define_macros,
            extra_compile_args=["-std=c99"],
        )
        target = ffi.compile(tmpdir=cache_dir or ".", verbose=verbose)
        module = CDLL(target)
        logger.info(f"compile CPU module {name}({define_str})")
        return module

    except VerificationError as e:
        logger.debug(f"{e}; skip CPU extension")
        return MissingModule(RuntimeError(f"invoke failed CPU extension"))


def gpu_extension(code, name, define_macros=list()):
    if KERNEL_DISABLE_GPU_MODE:
        logger.debug(f"KERNEL_DISABLE_GPU_MODE=True; skip GPU extension")
        return MissingModule(RuntimeError("invoke skipped GPU extension"))

    try:
        from cupy import RawModule
        from cupy.cuda.compiler import CompileException

        code = KERNEL_DEFINE_MACROS_GPU + code
        options = tuple(f"-D {k}={v}" for k, v in define_macros)
        define_str = define_macros_string(define_macros)
        module = RawModule(code=code, options=options)
        module.compile()
        logger.info(f"compile GPU module {name}[{define_str}]")
        return module

    except ImportError as e:
        logger.debug(f"{e}; skip GPU extension")
        return MissingModule(e)

    except CompileException as e:
        logger.warning(f"{e}; skip GPU extension")
        return MissingModule(RuntimeError(f"invoke failed GPU extension"))


def cpu_extension_function(module, stub):
    """
    Return a function to replace the given stub with a call to a C function.

    The `stub.__name__` attribute is used to find the function in the module.

    The argtypes and restype of the function object are inferred from the stub
    type hints, so the stub must provide type hints for all of its arguments,
    and all of the type hints must map to ctypes objects through the module
    variable `PY_CTYPE_DICT`.
    """

    c_func = module[stub.__name__]
    c_func.argtypes = argtypes(stub)
    c_func.restype = restype(stub)

    @wraps(stub)
    def wrapper(*args):
        shape, pyargs = stub(*args)
        cargs = to_cpu_args(pyargs, c_func.argtypes)
        return c_func(*cargs)

    wrapper.__cpu_func__ = c_func
    return wrapper


def gpu_extension_function(module, stub):
    gpu_func = module.get_function(stub.__name__)

    if "return" in stub.__annotations__:
        return MissingFunction(
            ValueError(f"GPU kernel {stub.__name__} may not return a value")
        )

    @wraps(stub)
    def wrapper(*args):
        shape, pyargs = stub(*args)

        if type(shape) is int:
            shape = (shape,)

        if len(shape) == 1:
            (ti,) = bs = THREAD_BLOCK_SIZE_1D
            (ni,) = shape
            nb = ((ni + ti - 1) // ti,)
        if len(shape) == 2:
            ti, tj = bs = THREAD_BLOCK_SIZE_2D
            ni, nj = shape
            nb = ((ni + ti - 1) // ti, (nj + tj - 1) // tj)
        if len(shape) == 3:
            ti, tj, tk = bs = THREAD_BLOCK_SIZE_3D
            ni, nj, nk = shape
            nb = ((ni + ti - 1) // ti, (nj + tj - 1) // tj, (nk + tk - 1) // tk)

        gpu_func(nb, bs, tuple(to_gpu_args(pyargs)))

    wrapper.__gpu_func__ = gpu_func
    return wrapper


def extension_function(cpu_module, gpu_module, stub):
    cpu_func = cpu_extension_function(cpu_module, stub)
    gpu_func = gpu_extension_function(gpu_module, stub)

    @wraps(stub)
    def wrapper(*args, exec_mode=None):
        if exec_mode is None:
            exec_mode = KERNEL_DEFAULT_EXEC_MODE
        if exec_mode == "cpu":
            return cpu_func(*args)
        if exec_mode == "gpu":
            return gpu_func(*args)

    wrapper.__cpu_func__ = getattr(cpu_func, "__cpu_func__", None)
    wrapper.__gpu_func__ = getattr(gpu_func, "__gpu_func__", None)

    return wrapper


def collate_source_code(device_funcs: list):
    """
    Return a string of source code built from a list of device funcs.

    Since device functions can name others as dependencies, this function
    is effectively a tree traversal, where unique functions are collected
    and then arranged in reverse-dependency order.
    """

    def recurse(funcs, collected):
        for func in funcs:
            if not func in collected:
                try:
                    recurse(func.__device_funcs, collected)
                except AttributeError:
                    pass
                collected.append(func)

    collected = list()
    recurse(device_funcs, collected)

    for item in collected:
        if not hasattr(item, "__device_func_marker"):
            raise ValueError(f"expect function marked with @device, got {item}")

    a = str().join(set(func.__static for func in collected))
    b = str().join(func.__code for func in collected)
    return a + b


def device_function(code: str = None, device_funcs=list(), static=str()):
    """
    Return a decorator that replaces a stub function with a 'device' function.

    Device functions cannot be called from Python not invoked as kernels. They
    are essentially a container for a string of source code, and a function
    name, that can be declared programmatically as a dependency of other
    device functions and kernels.

    The `static` argument is a string to be prepended to the function code,
    probably containing macros that might also apply to other related device
    functions. Duplicate static strings will be removed at the time of kernel
    code generation.
    """

    def decorator(stub):
        c = code or dedent(stub.__doc__)
        if c.count("DEVICE") != 1:
            raise ValueError("must include exactly one function marked 'DEVICE'")
        stub.__device_funcs = device_funcs
        stub.__static = static
        stub.__code = c
        stub.__device_func_marker = None

        @wraps(stub)
        def wrapper(*args, **kwargs):
            raise NotImplementedError(f"cannot call a device function {stub.__name__}")

        return wrapper

    return decorator


def device(*args, **kwargs):
    """
    Convenience for `device_function`, can be used with or without arguments.
    """
    if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
        return device_function()(args[0])
    else:
        return device_function(*args, **kwargs)


class KernelData:
    """
    Represents a maybe-compiled extension function (internal use)
    """

    def __init__(self, stub, code, device_funcs, define_macros):
        self._compiled = False
        self._stub = stub
        self._code = code
        self._device_funcs = device_funcs
        self._define_macros = define_macros

    def kernel_code(self):
        if self._code:
            return dedent(self._code)
        if self._stub.__doc__:
            return dedent(self._stub.__doc__)

    def device_funcs(self):
        return self._device_funcs

    def define_macros(self):
        return self._define_macros

    def code(self):
        a = collate_source_code(self._device_funcs)
        b = self.kernel_code()
        return a + b

    def require_compiled(self):
        if not self._compiled:
            code = self.code()
            name = self._stub.__name__
            cpu_module = cpu_extension(code, name, self._define_macros)
            gpu_module = gpu_extension(code, name, self._define_macros)
            self.inject_modules(cpu_module, gpu_module)

    def inject_modules(self, cpu_module, gpu_module):
        self._func = extension_function(cpu_module, gpu_module, self._stub)
        self._compiled = True

    def copy(self):
        return KernelData(
            self._stub, self._code, self._device_funcs, self._define_macros
        )


def kernel_function(code: str = None, device_funcs=list(), define_macros=list()):
    """
    Return a decorator that replaces a 'stub' function with a 'kernel'.

    The C code to be compiled is given either in the `code` variable, or is
    otherwise taken from the stub's doc string (which must then consist
    exclusively of valid C code). The stub function is a Python function that
    simply inspects its arguments and returns:

    1. A shape for the kernel launch (total threads per dim), and
    2. The arguments to be passed to the native function

    The kernel is a function compiled from C code that can be parallelized in
    some way, and is compiled to either CPU or GPU code. The wrapper function
    returned is a proxy to the respective CPU and GPU compiled extension
    functions. The execution mode is controlled by passing an extra keyword
    argument `exec_mode='cpu'|'gpu'` to the wrapper function. It defaults to
    the module-wide variable `KERNEL_DEFAULT_EXEC_MODE` which is in turn be
    set with `configure_kernel_module(default_exec_mode='gpu')`.
    """

    if type(define_macros) is dict:
        define_macros = list(define_macros.items())

    def decorator(stub):
        k = KernelData(stub, code, device_funcs, define_macros)

        @wraps(stub)
        def wrapper(*args, exec_mode=None):
            k.require_compiled()
            return k._func(*args, exec_mode=exec_mode)

        wrapper.__kernel_data__ = k
        return wrapper

    return decorator


def kernel(*args, **kwargs):
    """
    Convenience for `kernel_function`, can be used with or without arguments.
    """
    if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
        return kernel_function()(args[0])
    else:
        return kernel_function(*args, **kwargs)


def kernel_class(cls):
    """
    A decorator for classes that contains kernel stubs.

    The class instance is given a custom __init__ which collects source code
    from any member functions defined as kernels. Compilation takes place when
    the kernel class is instantiated.

    If the wrapped class has a property `define_macros` (list of key-value
    tuples or a dict), its value will be added to (and supersede) any define
    macros associated with the individual kernels. The kernel class is thus a
    convenient way to group bits of code that are parameterized around a
    common set of compile-time macros.
    """
    cls_init = cls.__init__

    def wrap_method(self, kernel_data):
        @wraps(kernel_data._stub)
        def wrapper(*args, exec_mode=None):
            return kernel_data._func(self, *args, exec_mode=exec_mode)

        wrapper.__cpu_func__ = kernel_data._func.__cpu_func__
        wrapper.__gpu_func__ = kernel_data._func.__gpu_func__
        return wrapper

    def __init__(self, *args, **kwargs):
        cls_init(self, *args, **kwargs)

        static = getattr(self, "static", str())
        kernel_code = str()
        kernel_data_dict = dict()
        device_funcs = list()
        define_macros = list()

        for k in dir(self):
            if hasattr(getattr(self, k), "__kernel_data__"):
                kernel_data = getattr(self, k).__kernel_data__
                kernel_code += kernel_data.kernel_code()
                device_funcs += kernel_data.device_funcs()
                define_macros += kernel_data.define_macros()
                kernel_data_dict[k] = kernel_data

        if d := getattr(self, "device_funcs", None):
            device_funcs += d

        if m := getattr(self, "define_macros", None):
            define_macros += m if type(m) is list else list(m.items())

        name = cls.__name__
        comment = f"// {name}({define_macros_string(define_macros)})"
        code = comment + static + collate_source_code(device_funcs) + kernel_code
        cpu_module = cpu_extension(code, name, define_macros)
        gpu_module = gpu_extension(code, name, define_macros)

        for key, kernel_data in kernel_data_dict.items():
            k = kernel_data.copy()
            k.inject_modules(cpu_module, gpu_module)
            setattr(self, key, wrap_method(self, k))

        self.__native_code__ = code

    cls.__init__ = __init__
    return cls


def kernel_metadata(kernel):
    metadata = dict()
    cpu_func = kernel.__cpu_func__
    gpu_func = kernel.__gpu_func__

    metadata["name"] = kernel.__name__

    try:
        metadata["num_regs"] = gpu_func.num_regs
    except AttributeError:
        pass

    return metadata


def main():
    # ==============================================================================
    # Example usage of kernel functions and classes
    # ==============================================================================

    from argparse import ArgumentParser
    from logging import basicConfig
    from sys import stdout

    try:
        from rich.logging import RichHandler

        logging_handler = RichHandler(omit_repeated_times=False)
    except ImportError:
        from logging import StreamHandler

        logging_handler = StreamHandler(stdout)

    parser = ArgumentParser()
    parser.add_argument(
        "--mode",
        dest="exec_mode",
        default="cpu",
        choices=["cpu", "gpu"],
        help="execution mode",
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error", "critical"],
        help="log messages at and above this severity level",
    )
    args = parser.parse_args()

    configure_kernel_module(default_exec_mode=args.exec_mode)
    basicConfig(
        level=args.log_level.upper(),
        format="%(message)s",
        handlers=[logging_handler],
    )

    if args.exec_mode == "cpu":
        from numpy import array, linspace, zeros_like
    if args.exec_mode == "gpu":
        from cupy import array, linspace, zeros_like

    # ==============================================================================
    # 1.
    #
    # Demonstrates a rank-0 kernel function. The C source for this example is
    # provided as an argument to the kernel decorator function. Note that native
    # functions that return a value cannot be GPU kernels, but they can be CPU
    # kernels.
    # ==============================================================================

    code = R"""
    KERNEL double multiply(int a, double b)
    {
        return a * b;
    }
    """

    @kernel(code)
    def multiply(a: int, b: float) -> float:
        """
        Return the product of an integer a and a float b.
        """
        return None, (a, b)

    if args.exec_mode != "gpu":
        assert multiply(5, 25.0) == 125.0

    # ==============================================================================
    # 2.
    #
    # Demonstrates a rank-1 kernel function. The C source for this example is
    # written in the doc string.
    # ==============================================================================

    @kernel
    def rank_one_kernel(a: float, x: NDArray[float], y: NDArray[float], ni: int = None):
        R"""
        KERNEL void rank_one_kernel(double a, double *x, double *y, int ni)
        {
            FOR_EACH_1D(ni)
            {
                y[i] = x[i] * a;
            }
        }
        """
        if x.shape != y.shape:
            raise ValueError("input and output arrays have different shapes")
        return x.shape, (a, x, y, x.shape[0])

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

    @device
    def dot3(a, b, c):
        R"""
        DEVICE double dot3(double a, double b, double c)
        {
            return a * a + b * b + c * c;
        }
        """
        pass

    @kernel_class
    class Solver:
        @property
        def define_macros(self):
            return dict(gamma_law_index=5.0 / 3.0)

        @kernel(device_funcs=[dot3])
        def conserved_to_primitive(
            self,
            u: NDArray[float],
            p: NDArray[float],
            ni: int = None,
        ):
            R"""
            //
            // Compute the conversion of conserved variables to primitive ones.
            //
            KERNEL void conserved_to_primitive(double *u, double *p, int ni)
            {
                FOR_EACH_1D(ni)
                {
                    double rho = u[5 * i + 0];
                    double px  = u[5 * i + 1];
                    double py  = u[5 * i + 2];
                    double pz  = u[5 * i + 3];
                    double nrg = u[5 * i + 4];
                    double p_squared = dot3(px, py, pz);

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
            return (p.size // 5,), (u, p, p.size // 5)

        @kernel(device_funcs=[dot3])
        def primitive_to_conserved(
            self,
            p: NDArray[float],
            u: NDArray[float],
            ni: int = None,
        ):
            R"""
            //
            // Compute the conversion of primitive variables to conserved ones.
            //
            KERNEL void primitive_to_conserved(double *p, double *u, int ni)
            {
                FOR_EACH_1D(ni)
                {
                    double rho = p[5 * i + 0];
                    double vx  = p[5 * i + 1];
                    double vy  = p[5 * i + 2];
                    double vz  = p[5 * i + 3];
                    double pre = p[5 * i + 4];
                    double v_squared = dot3(vx, vy, vz);

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
            return (u.size // 5,), (p, u, u.size // 5)

    solver = Solver()
    p = array([[1.0, 0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0, 1.0]])
    u = zeros_like(p)
    q = zeros_like(p)
    solver.primitive_to_conserved(p, u)
    solver.conserved_to_primitive(u, q)
    assert (p == q).all()

    # ==============================================================================
    # 4.
    #
    # Demonstrates how functions marked with the devive decorator are used for
    # programatic code generation.
    # ==============================================================================

    @device
    def device_func0(a: int):
        R"""
        DEVICE int device_func0(int a)
        {
            return a;
        }
        """
        pass

    @device(device_funcs=[device_func0])
    def device_func1(a: int):
        R"""
        DEVICE int device_func1(int a)
        {
            return device_func0(a);
        }
        """
        pass

    @device(device_funcs=[device_func0])
    def device_func2(a: int):
        R"""
        DEVICE int device_func2(int a)
        {
            return device_func0(a);
        }
        """
        pass

    collated = R"""
    DEVICE int device_func0(int a)
    {
        return a;
    }

    DEVICE int device_func1(int a)
    {
        return device_func0(a);
    }

    DEVICE int device_func2(int a)
    {
        return device_func0(a);
    }
    """
    assert collate_source_code([device_func1, device_func2]) == dedent(collated)

    # ==============================================================================
    # 5.
    #
    # Demonstrates how a kernel class can be used to parameterize a set of
    # kernel functions around instance-specific define macros.
    # ==============================================================================

    @kernel_class
    class ConfigurableModule:
        def __init__(self, value):
            self.value = value

        @property
        def define_macros(self):
            return dict(VALUE=self.value)

        @kernel
        def run(self) -> int:
            R"""
            KERNEL int run()
            {
                return VALUE;
            }
            """
            return None, tuple()

    if args.exec_mode != "gpu":
        m1 = ConfigurableModule(1)
        m2 = ConfigurableModule(2)
        assert m1.run() == 1
        assert m2.run() == 2

    # ==============================================================================
    # 6.
    #
    # Demonstrates how to pass data structure to a kernel function.
    # ==============================================================================

    class MyStruct(Structure):
        _fields_ = [
            ("x", c_int),
            ("y", c_int),
            ("a", c_double),
            ("b", c_double),
            ("c", c_int),
            ("d", c_double),
        ]

    @kernel
    def takes_struct_arg(arg: MyStruct):
        R"""
        struct MyStruct {
            int x;
            int y;
            double a;
            double b;
            int c;
            double d;
        };

        KERNEL void takes_struct_arg(struct MyStruct arg)
        {
            assert(arg.x + arg.y + arg.a + arg.b + arg.c + arg.d == 21.0);
        }
        """
        return (1,), (arg,)

    arg = MyStruct(x=1, y=2, a=3, b=4, c=5, d=6)
    takes_struct_arg(arg)


if __name__ == "__main__":
    main()
