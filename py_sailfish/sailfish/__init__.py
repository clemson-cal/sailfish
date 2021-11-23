import logging
import platform

build_config = {
    'extra_compile_args': [],
    'extra_link_args': [],
}

if platform.system() == 'Darwin':
    logging.info('configure JIT build for MacOS')
    build_config['extra_compile_args'] = ['-Xpreprocessor', '-fopenmp']
    build_config['extra_link_args'] = ['-lomp']
elif platform.system() == 'Linux':
    logging.info('configure JIT build for Linux')
    build_config['extra_compile_args'] = ['-fopenmp']
    build_config['extra_link_args'] = ['-fopenmp']
else:
    logging.info('configure JIT build for unknown system')

def get_array_module(mode):
    if mode in ['cpu', 'omp']:
        import numpy
        return numpy
    elif mode == 'gpu':
        import cupy
        return cupy
    else:
        raise ValueError(f'unknown execution mode {mode}, must be [cpu|omp|gpu]')


from .library import Library
from . import solvers
