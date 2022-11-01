"""
A Python module to facilitate JIT-compiled CPU-GPU agnostic compute kernels.

Kernel libraries are collections of functions written in C code that can be
compiled for CPU execution using a normal C compiler via the CFFI module, or
for GPU execution using a CUDA or ROCm compiler via cupy.
"""
