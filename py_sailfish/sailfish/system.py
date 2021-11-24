import logging
import multiprocessing
import platform

logger = logging.getLogger(__name__)


build_config = {
    "extra_compile_args": [],
    "extra_link_args": [],
}


def configure_build():
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
    if mode in ["cpu", "omp"]:
        import numpy

        return numpy
    elif mode == "gpu":
        import cupy

        return cupy
    else:
        raise ValueError(f"unknown execution mode {mode}, must be [cpu|omp|gpu]")


def log_system_info(mode):
    if mode == "gpu":
        from cupy.cuda.runtime import getDeviceCount, getDeviceProperties

        num_devices = getDeviceCount()
        gpu_devices = ":".join(
            [getDeviceProperties(i)["name"].decode("utf-8") for i in range(num_devices)]
        )
        logger.info(f"gpu devices: {num_devices}x {gpu_devices}")
    logger.info(f"compute cores: {multiprocessing.cpu_count()}")
