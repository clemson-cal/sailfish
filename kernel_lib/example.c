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
