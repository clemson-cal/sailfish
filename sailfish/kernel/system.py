"""
Functions for querying compute capabilities and configuring JIT builds.
"""

import contextlib
import logging
import multiprocessing
import platform
import time

logger = logging.getLogger(__name__)


build_config = {
    "extra_compile_args": [],
    "extra_link_args": [],
}


def configure_build():
    """
    Initiate the `build_config` module-level variable.

    The build configuration is presently inferred from a crude check to see
    whether we're on Linux or MacOS. This procedure can be easily generalized
    to accommodate other or more specific platforms, check for hardware or
    software availability, or read a bulid configuration from a user or
    site-specified machine file.
    """
    if platform.system() == "Darwin":
        logger.info("configure JIT build for MacOS")
        build_config["extra_compile_args"] = ["-Xpreprocessor", "-fopenmp"]
        build_config["extra_link_args"] = ["-lomp"]
    elif platform.system() == "Linux":
        logger.info("configure JIT build for Linux")
        build_config["extra_compile_args"] = ["-fopenmp"]
        build_config["extra_link_args"] = ["-fopenmp"]
    else:
        logger.info("configure JIT build for unknown system")


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
def measure_time() -> float:
    """
    A context manager to measure the execution time of a piece of code.

    Example:

    .. code-block:: python

        with measure_time() as duration:
            expensive_function()
        print(f"execution took {duration()} seconds")
    """
    start = time.perf_counter()
    yield lambda: time.perf_counter() - start
