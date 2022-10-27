import sys
import logging

sys.path.insert(1, ".")

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


def main():
    import argparse
    import numpy as np
    from sailfish.kernel.library import Library
    from sailfish.kernel.system import configure_build

    configure_build(enable_openmp=True)
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="cpu", choices=["cpu", "omp", "gpu"])
    args = parser.parse_args()

    if args.mode == "gpu":
        import cupy as xp
    else:
        import numpy as xp

    library = Library(code, mode=args.mode)
    data_1d = xp.zeros([10])
    data_2d = xp.zeros([10, 20])
    data_3d = xp.zeros([10, 20, 30])

    library.my_1d_kernel[data_1d.shape](data_1d)
    library.my_2d_kernel[data_2d.shape](data_2d)
    library.my_3d_kernel[data_3d.shape](data_3d)

    if args.mode == "gpu":
        data_1d = data_1d.get()
        data_2d = data_2d.get()
        data_3d = data_3d.get()

    for (i,), x in np.ndenumerate(data_1d):
        assert i == x

    for (i, j), x in np.ndenumerate(data_2d):
        assert i + j == x

    for (i, j, k), x in np.ndenumerate(data_3d):
        assert i + j + k == x


if __name__ == "__main__":
    main()
