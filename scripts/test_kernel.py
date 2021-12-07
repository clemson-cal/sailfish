#!/usr/bin/env python3

import sys

sys.path.insert(1, ".")

import numpy as np
from sailfish.kernel.library import Library

# Several implicit things:
#
# - If a C kernel has an n leading integer arguments, then the kernel launch
#   is n-dimensional.
#
# - Functions that are part of the API need to start with PUBLIC. Other
#   functions start with PRIVATE.
#
# - The signatures of PUBLIC functions need to go on separate lines.
#
# - For loops are replaced with a macro,
#
# - Function arguments can only be int, double, and double*.

code = """
PUBLIC void my_1d_kernel(
    int ni,
    double *data) // :: $.shape == (ni,)
{
    FOR_EACH_1D(ni)
    {
        data[i] = i;
    }
}

PUBLIC void my_2d_kernel(
    int ni,
    int nj,
    double *data) // :: $.shape == (ni, nj)
{
    FOR_EACH_2D(ni, nj)
    {
        data[i * nj + j] = i + j;
    }
}

PUBLIC void my_3d_kernel(
    int ni,
    int nj,
    int nk,
    double *data) // :: $.shape == (ni, nj, nk)
{
    FOR_EACH_3D(ni, nj, nk)
    {
        data[i * nj * nk + j * nk + k] = i + j + k;
    }
}
"""

library = Library(code, mode="cpu")
data_1d = np.zeros([10])
data_2d = np.zeros([10, 20])
data_3d = np.zeros([10, 20, 30])

library.my_1d_kernel[data_1d.shape](data_1d)
library.my_2d_kernel[data_2d.shape](data_2d)
library.my_3d_kernel[data_3d.shape](data_3d)

for (i,), x in np.ndenumerate(data_1d):
    assert i == x

for (i, j), x in np.ndenumerate(data_2d):
    assert i + j == x

for (i, j, k), x in np.ndenumerate(data_3d):
    assert i + j + k == x
