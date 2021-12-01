import logging
import os
import hashlib
import tempfile
import time
from ctypes import c_double, c_int, POINTER, CDLL
from sailfish.system import build_config
from sailfish.parse_api import parse_api

logger = logging.getLogger(__name__)
THREAD_BLOCK_SIZE = 64


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

    def __init__(self, module_file, mode="cpu", debug=True):
        abs_path, _ = os.path.splitext(module_file)
        module = os.path.basename(abs_path)

        logger.info(f"debug mode {'enabled' if debug else 'disabled'}")
        logger.info(f"prepare module {module} for {mode} execution")

        start = time.perf_counter()

        filename = f"{abs_path}.c"
        self.debug = debug
        self.cpu_mode = mode != "gpu"
        self.api = parse_api(filename)

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
            libname = hashlib.sha256(code.encode("utf-8")).hexdigest()
            cache_dir = os.path.join(".sailfish", libname)

            # with tempfile.TemporaryDirectory() as tmpdir:
            try:
                so_name = os.path.join(
                    cache_dir,
                    next(filter(lambda f: f.endswith(".so"), os.listdir(cache_dir))),
                )
                self.module = CDLL(so_name)
                logger.info(f"load cached library")
            except (FileNotFoundError, StopIteration) as e:
                target = ffi.compile(cache_dir)
                self.module = CDLL(target)
                logger.info(f"recompile library")

            self.xp = numpy
        else:
            import cupy

            module = cupy.RawModule(code=code, options=("-D EXEC_MODE=2",))
            module.compile()
            self.module = module
            self.xp = cupy

        stop = time.perf_counter()
        logger.info(f"module preparation took {stop - start:0.3}s")

        for symbol in self.api:
            logger.info(f"+-- {symbol}")

    def __getattr__(self, symbol):
        arg_format = self.api[symbol]

        if self.cpu_mode:
            kernel = getattr(self.module, symbol)
        else:
            kernel = self.module.get_function(symbol)

        def invoke_kernel(*args):
            if self.debug:
                validate_constraints(args, arg_format, symbol)
                validate_types(args, arg_format, symbol, self.xp)

            if self.cpu_mode:
                kernel(*to_ctypes(args, arg_format))
            else:
                num_zones = args[0]
                nb = ((num_zones + THREAD_BLOCK_SIZE - 1) // THREAD_BLOCK_SIZE,)
                bs = (THREAD_BLOCK_SIZE,)
                kernel(nb, bs, args)

        return invoke_kernel


def to_ctypes(args, arg_format):
    for arg, (typename, _, _) in zip(args, arg_format):
        if typename == "int":
            yield c_int(arg)
        elif typename == "double":
            yield c_double(arg)
        elif typename == "double*":
            yield arg.ctypes.data_as(POINTER(c_double))


def validate_types(args, arg_format, symbol, xp):
    if len(args) != len(arg_format):
        raise TypeError(
            f"{symbol} takes exactly {len(arg_format)} arguments ({len(args)} given)"
        )

    for n, (arg, (typename, argname, constraint)) in enumerate(zip(args, arg_format)):
        if typename == "int":
            if type(arg) is not int:
                raise TypeError(
                    f"argument {n} to {symbol} has type {type(arg)}, expected int"
                )
        elif typename == "double":
            if type(arg) is not float:
                raise TypeError(
                    f"argument {n} to {symbol} has type {type(arg)}, expected float64"
                )
        elif typename == "double*":
            if type(arg) is not xp.ndarray:
                raise TypeError(
                    f"argument {n} to {symbol} has type {type(arg)}, expected ndarray"
                )
            if arg.dtype != xp.float64:
                raise TypeError(
                    f"argument {n} to {symbol} has dtype {arg.dtype}, expected float64"
                )


def validate_constraints(args, arg_format, symbol):
    scope = dict(zip([a[1] for a in arg_format], args))
    for arg, (_, name, constraint) in zip(args, arg_format):
        if constraint:
            c = constraint.replace("$", name)
            if not eval(c, None, scope):
                raise ValueError(f"argument constraint for {symbol} not satisfied: {c}")
