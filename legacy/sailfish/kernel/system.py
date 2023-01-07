"""
Functions for querying compute capabilities and configuring JIT builds.
"""

import contextlib
import logging
import multiprocessing
import platform
import time

try:
    from contextlib import nullcontext

except ImportError:
    from contextlib import AbstractContextManager

    class nullcontext(AbstractContextManager):
        """
        Scraped from contextlib source in Python >= 3.7 for backwards compatibility.
        """

        def __init__(self, enter_result=None):
            self.enter_result = enter_result

        def __enter__(self):
            return self.enter_result

        def __exit__(self, *excinfo):
            pass

        async def __aenter__(self):
            return self.enter_result

        async def __aexit__(self, *excinfo):
            pass


logger = logging.getLogger(__name__)


build_config = {
    "enable_openmp": True,
    "extra_compile_args": [],
    "extra_link_args": [],
}


def configure_build(
    enable_openmp=True,
    extra_compile_args=None,
    extra_link_args=None,
    execution_mode=None,
):
    """
    Initiate the `build_config` module-level variable.

    A default build configuration is inferred on MacOS or Linux. Defaults for
    Windows platforms and specific Linux flavors should be added soon. The
    keyword arguemnts may be Python objects, or strings to facilitate passing
    values right from a configparser instance.
    """

    if type(enable_openmp) is str:
        enable_openmp = {"True": True, "False": False}[enable_openmp]

    enable_openmp = enable_openmp and execution_mode == "omp"

    if type(extra_compile_args) is str:
        extra_compile_args = extra_compile_args.split()

    if type(extra_link_args) is str:
        extra_link_args = extra_link_args.split()

    if platform.system() == "Darwin":
        logger.info("configure JIT build for MacOS")
        sys_compile_args = ["-Xpreprocessor", "-fopenmp"]
        sys_link_args = ["-L/usr/local/lib", "-lomp"]
    elif platform.system() == "Linux":
        logger.info("configure JIT build for Linux")
        sys_compile_args = ["-fopenmp", "-std=c99"]
        sys_link_args = ["-fopenmp"]
    elif platform.system() == "Windows":
        logger.info("configure JIT build for Windows")
        sys_compile_args = []
        sys_link_args = []
    else:
        logger.info("configure JIT build for unknown system")
        sys_compile_args = []
        sys_link_args = []

    if enable_openmp:
        build_config["extra_compile_args"] = extra_compile_args or sys_compile_args
        build_config["extra_link_args"] = extra_link_args or sys_link_args

    build_config["enable_openmp"] = enable_openmp
    logger.info(f"OpenMP is {'enabled' if enable_openmp else 'disabled'}")


def get_array_module(mode):
    """
    Return either the numpy or cupy module, depending on the value of mode.

    If mode is "cpu" or "omp", then the `numpy` module is returned. Otherwise
    if mode is "gpu" then `cupy` is returned. The `cupy` documentation
    recommends assigning whichever module is returned to a variable called
    `xp`, and using that variable to access functions that are common to both,
    for example use :code:`xp.zeros(100)`. This pattern facilitates writing
    CPU-GPU agnostic code.
    """
    if mode in ["cpu", "omp"]:
        import numpy

        return numpy
    elif mode == "gpu":
        import cupy

        return cupy
    else:
        raise ValueError(f"unknown execution mode {mode}, must be [cpu|omp|gpu]")


def execution_context(mode, device_id=None):
    """
    Return a context manager appropriate for the given exuction mode.

    If `mode` is "gpu", then a specific device id may be provided to specify
    the GPU onto which kernel launches should be spawned.
    """
    if mode in ["cpu", "omp"]:
        return nullcontext()

    elif mode == "gpu":
        from cupy.cuda import Device

        return Device(device_id)


def num_devices(mode):
    if mode in ["cpu", "omp"]:
        return 1

    elif mode == "gpu":
        from cupy.cuda.runtime import getDeviceCount

        return getDeviceCount()


def log_system_info(mode):
    """
    Log relevant details of the system's compute capabilities.
    """
    if mode == "gpu":
        from cupy.cuda.runtime import getDeviceCount, getDeviceProperties

        num_devices = getDeviceCount()
        gpu_devices = ":".join(
            [getDeviceProperties(i)["name"].decode("utf-8") for i in range(num_devices)]
        )
        logger.info(f"gpu devices: {num_devices}x {gpu_devices}")
    logger.info(f"compute cores: {multiprocessing.cpu_count()}")


@contextlib.contextmanager
def measure_time(mode: str) -> float:
    """
    A context manager to measure the execution time of a piece of code.

    Example:

    .. code-block:: python

        with measure_time() as duration:
            expensive_function()
        print(f"execution took {duration()} seconds")
    """
    try:
        start = time.perf_counter()
        yield lambda: time.perf_counter() - start
    finally:
        if mode == "gpu":
            from cupy.cuda.runtime import deviceSynchronize

            deviceSynchronize()
