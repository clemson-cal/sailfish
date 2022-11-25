"""
Exports functions for estimating gradients on rectilinear grids.
"""


from new_kernels import device
from numpy.typing import NDArray


@device()
def plm_minmod(yl: float, yc: float, yr: float, plm_theta: float):
    R"""
    DEVICE double plm_minmod(
        double yl,
        double yc,
        double yr,
        double plm_theta)
    {
        #define min2(a, b) ((a) < (b) ? (a) : (b))
        #define max2(a, b) ((a) > (b) ? (a) : (b))
        #define min3(a, b, c) min2(a, min2(b, c))
        #define max3(a, b, c) max2(a, max2(b, c))
        #define sign(x) copysign(1.0, x)
        #define minabs(a, b, c) min3(fabs(a), fabs(b), fabs(c))

        double a = (yc - yl) * plm_theta;
        double b = (yr - yl) * 0.5;
        double c = (yr - yc) * plm_theta;
        return 0.25 * fabs(sign(a) + sign(b)) * (sign(a) + sign(c)) * minabs(a, b, c);

        #undef min2
        #undef max2
        #undef min3
        #undef max3
        #undef sign
        #undef minabs
    }
    """


# plm_gradient_1d_code = R"""
# KERNEL void plm_gradient_1d(
#     int ni,
#     int nfields,
#     double *y,
#     double *g,
#     double plm_theta)
# {
#     FOR_RANGE_1D(1, ni - 1)
#     {
#         int ic = i;
#         int il = i - 1;
#         int ir = i + 1;

#         double *yc = &y[ic * nfields];
#         double *yl = &y[il * nfields];
#         double *yr = &y[ir * nfields];
#         double *gc = &g[ic * nfields];

#         for (int q = 0; q < nfields; ++q)
#         {
#             gc[q] = _minmod(yl[q], yc[q], yr[q], plm_theta);
#         }
#     }
# }
# """


# plm_gradient_2d_code = R"""
# KERNEL void plm_gradient_2d(
#     int ni,
#     int nj,
#     int nfields,
#     double *y,
#     double *gi,
#     double *gj,
#     double plm_theta)
# {
#     int sj = nfields;
#     int si = sj * nj;

#     FOR_RANGE_2D(1, ni - 1, 1, nj - 1)
#     {
#         int ic = i;
#         int il = i - 1;
#         int ir = i + 1;
#         int jc = j;
#         int jl = j - 1;
#         int jr = j + 1;

#         double *ycc = &y[ic * si + jc * sj];
#         double *ylc = &y[il * si + jc * sj];
#         double *yrc = &y[ir * si + jc * sj];
#         double *ycl = &y[ic * si + jl * sj];
#         double *ycr = &y[ic * si + jr * sj];

#         for (int q = 0; q < nfields; ++q)
#         {
#             gi[q] = _minmod(ylc[q], ycc[q], yrc[q], plm_theta);
#             gj[q] = _minmod(ycl[q], ycc[q], ycr[q], plm_theta);
#         }
#     }
# }
# """


# plm_gradient_3d_code = R"""
# KERNEL void plm_gradient_3d(
#     int ni,
#     int nj,
#     int nk,
#     int nfields,
#     double *y,
#     double *gi,
#     double *gj,
#     double *gk,
#     double plm_theta)
# {
#     int sk = nfields;
#     int sj = sk * nk;
#     int si = sj * nj;

#     FOR_RANGE_3D(1, ni - 1, 1, nj - 1, 1, nk - 1)
#     {
#         int ic = i;
#         int il = i - 1;
#         int ir = i + 1;
#         int jc = j;
#         int jl = j - 1;
#         int jr = j + 1;
#         int kc = k;
#         int kl = k - 1;
#         int kr = k + 1;

#         double *yccc = &y[ic * si + jc * sj + kc * sk];
#         double *ylcc = &y[il * si + jc * sj + kc * sk];
#         double *yrcc = &y[ir * si + jc * sj + kc * sk];
#         double *yclc = &y[ic * si + jl * sj + kc * sk];
#         double *ycrc = &y[ic * si + jr * sj + kc * sk];
#         double *yccl = &y[ic * si + jc * sj + kl * sk];
#         double *yccr = &y[ic * si + jc * sj + kr * sk];

#         for (int q = 0; q < nfields; ++q)
#         {
#             gi[q] = _minmod(ylcc[q], yccc[q], yrcc[q], plm_theta);
#             gj[q] = _minmod(yclc[q], yccc[q], ycrc[q], plm_theta);
#             gk[q] = _minmod(yccl[q], yccc[q], yccr[q], plm_theta);
#         }
#     }
# }
# """


# @kernel(common + plm_gradient_1d_code, rank=1, pre_argtypes=(int, int))
# def plm_gradient_1d(
#     y: NDArray[float],
#     g: NDArray[float],
#     plm_theta: float,
# ):
#     """
#     Estimate PLM gradients of data on an evenly spaced 1d rectilinear grid.

#     The first and last zone on each axis are not modified.
#     """
#     if len(y.shape) != 2:
#         raise ValueError("array must have rank 1 and one axis of fields")
#     if y.shape != g.shape:
#         raise ValueError("y and g must have the same shape")
#     if not 1.0 <= plm_theta <= 2.0:
#         raise ValueError("theta value must be between 1.0 and 2.0")
#     return y.shape


# @kernel(common + plm_gradient_2d_code, rank=2, pre_argtypes=(int, int, int))
# def plm_gradient_2d(
#     y: NDArray[float],
#     gi: NDArray[float],
#     gj: NDArray[float],
#     plm_theta: float,
# ):
#     """
#     Estimate PLM gradients of data on an evenly spaced 2d rectilinear grid.

#     The first and last zone on each axis are not modified.
#     """
#     if len(y.shape) != 3:
#         raise ValueError("array must have rank 2 and one axis of fields")
#     if not 1.0 <= plm_theta <= 2.0:
#         raise ValueError("theta value must be between 1.0 and 2.0")
#     return y.shape


# @kernel(common + plm_gradient_3d_code, rank=3, pre_argtypes=(int, int, int, int))
# def plm_gradient_3d(
#     y: NDArray[float],
#     gi: NDArray[float],
#     gj: NDArray[float],
#     gk: NDArray[float],
#     plm_theta: float,
# ):
#     """
#     Estimate PLM gradients of data on an evenly spaced 3d rectilinear grid.

#     The first and last zone on each axis are not modified.
#     """
#     if len(y.shape) != 4:
#         raise ValueError("array must have rank 3 and one axis of fields")
#     if not 1.0 <= plm_theta <= 2.0:
#         raise ValueError("theta value must be between 1.0 and 2.0")
#     return y.shape


# extrapolate_code = R"""
# KERNEL void extrapolate(
#     int ni,
#     int nfields,
#     double *y,
#     double *g,
#     double *ym,
#     double *yp)
# {
#     FOR_EACH_1D(ni)
#     {
#         for (int q = 0; q < nfields; ++q)
#         {
#             ym[i * nfields + q] = y[i * nfields + q] - 0.5 * g[i * nfields + q];
#             yp[i * nfields + q] = y[i * nfields + q] + 0.5 * g[i * nfields + q];
#         }
#     }
# }
# """


# @kernel(extrapolate_code, rank=1, pre_argtypes=(int, int))
# def extrapolate(
#     y: NDArray[float],
#     g: NDArray[float],
#     ym: NDArray[float],
#     yp: NDArray[float],
# ):
#     """
#     Evaluate ym = y - 0.5 * g and yp = y + 0.5 * g.

#     Context: the y array is probably zone-centered data and the g array is
#     probably the gradients (times the grid spacing) along a given axis.

#     The arrays y, g, ym, yp may be of rank 1, 2, or, 3 but they must have the
#     same shape. The final axis is the number of fields per zone.
#     """
#     if not all(y.shape == s for s in (g.shape, ym.shape, yp.shape)):
#         raise ValueError("arguments must have the same shape")
#     return (y.size // y.shape[-1], y.shape[-1])
