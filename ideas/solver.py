"""
A solver is a generator function and a state object
"""

from logging import getLogger
from typing import NamedTuple, Callable

from numpy import array, zeros, logical_not
from numpy.typing import NDArray

from kernels import kernel, kernel_class, device, kernel_metadata
from lib_euler import prim_to_cons, cons_to_prim, riemann_hlle
from models import Sailfish, Reconstruction

logger = getLogger("sailfish")


def numpy_or_cupy(exec_mode):
    if exec_mode == "gpu":
        import cupy

        return cupy

    if exec_mode == "cpu":
        import numpy

        return numpy


@device
def plm_minmod(yl: float, yc: float, yr: float, plm_theta: float):
    R"""
    #define min2(a, b) ((a) < (b) ? (a) : (b))
    #define min3(a, b, c) min2(a, min2(b, c))
    #define sign(x) copysign(1.0, x)
    #define minabs(a, b, c) min3(fabs(a), fabs(b), fabs(c))

    DEVICE double plm_minmod(
        double yl,
        double yc,
        double yr,
        double plm_theta)
    {
        double a = (yc - yl) * plm_theta;
        double b = (yr - yl) * 0.5;
        double c = (yr - yc) * plm_theta;
        return 0.25 * fabs(sign(a) + sign(b)) * (sign(a) + sign(c)) * minabs(a, b, c);
    }
    """


@kernel_class
class GradientEsimation:
    def __init__(self, nfields, transpose, reconstruction):
        plm_theta = reconstruction[1] if type(reconstruction) is tuple else 0.0
        self.nfields = nfields
        self.transpose = transpose
        self.plm_theta = plm_theta

    @property
    def define_macros(self):
        return dict(NFIELDS=self.nfields, TRANSPOSE=self.transpose)

    @property
    def device_funcs(self):
        return [plm_minmod]

    @kernel
    def plm_gradient(
        self,
        y: NDArray[float],
        g: NDArray[float],
        plm_theta: float = None,
        ni: int = None,
    ):
        R"""
        KERNEL void plm_gradient(double *y, double *g, double plm_theta, int ni)
        {
            #if TRANSPOSE == 0
            int sq = 1;
            int si = NFIELDS;
            #elif TRANSPOSE == 1
            int sq = ni;
            int si = 1;
            #endif

            FOR_RANGE_1D(1, ni - 1)
            {
                int ic = i;
                int il = i - 1;
                int ir = i + 1;

                for (int q = 0; q < NFIELDS; ++q)
                {
                    double yc = y[ic * si + q * sq];
                    double yl = y[il * si + q * sq];
                    double yr = y[ir * si + q * sq];
                    g[ic * si + q * sq] = plm_minmod(yl, yc, yr, plm_theta);
                }
            }
        }
        """
        plm = plm_theta if plm_theta is not None else self.plm_theta
        nq = self.nfields
        ii = -1 if self.transpose else 0
        iq = 0 if self.transpose else -1

        if y.shape[iq] != nq or y.shape != g.shape:
            raise ValueError("array has wrong number of fields")

        return y.shape[ii], (y, g, plm, y.shape[ii])


@kernel_class
class Fields:
    def __init__(self, dim, transpose):
        self.dim = dim
        self.transpose = transpose

    @property
    def define_macros(self):
        return dict(DIM=self.dim, TRANSPOSE=int(self.transpose))

    @property
    def device_funcs(self):
        return [prim_to_cons, cons_to_prim]

    @kernel
    def cons_to_prim_array(
        self,
        u: NDArray[float],
        p: NDArray[float],
        ni: int = None,
    ):
        R"""
        KERNEL void cons_to_prim_array(double *u, double *p, int ni)
        {
            #if TRANSPOSE == 0
            int sq = 1;
            int si = NCONS;
            #elif TRANSPOSE == 1
            int sq = ni;
            int si = 1;
            #endif

            FOR_EACH_1D(ni)
            {
                double u_reg[NCONS];
                double p_reg[NCONS];

                for (int q = 0; q < NCONS; ++q)
                {
                    u_reg[q] = u[i * si + q * sq];
                }
                cons_to_prim(u_reg, p_reg);

                for (int q = 0; q < NCONS; ++q)
                {
                    p[i * si + q * sq] = p_reg[q];
                }
            }
        }
        """
        nq = self.dim + 2
        iq = 0 if self.transpose else -1

        if u.shape[iq] != nq or u.shape != p.shape:
            raise ValueError("array has wrong number of fields")

        return u.size // nq, (u, p, u.size // nq)

    @kernel
    def prim_to_cons_array(
        self,
        p: NDArray[float],
        u: NDArray[float],
        ni: int = None,
    ):
        R"""
        KERNEL void prim_to_cons_array(double *p, double *u, int ni)
        {
            #if TRANSPOSE == 0
            int sq = 1;
            int si = NCONS;
            #elif TRANSPOSE == 1
            int sq = ni;
            int si = 1;
            #endif

            FOR_EACH_1D(ni)
            {
                double p_reg[NCONS];
                double u_reg[NCONS];

                for (int q = 0; q < NCONS; ++q)
                {
                    p_reg[q] = p[i * si + q * sq];
                }
                prim_to_cons(p_reg, u_reg);

                for (int q = 0; q < NCONS; ++q)
                {
                    u[i * si + q * sq] = u_reg[q];
                }
            }
        }
        """
        nq = self.dim + 2
        iq = 0 if self.transpose else -1

        if p.shape[iq] != nq or p.shape != u.shape:
            raise ValueError("array has wrong number of fields")

        return p.size // nq, (p, u, p.size // nq)


@kernel_class
class Scheme:
    """
    Godunov scheme using method-of-lines and many strategy modes
    """

    def __init__(
        self,
        dim: int,
        transpose: bool,
        reconstruction: Reconstruction,
        time_integration: str,
        cache_prim: bool,
        cache_grad: bool,
    ):
        if dim != 1:
            raise NotImplementedError

        device_funcs = list()
        define_macros = dict()
        define_macros["DIM"] = dim
        define_macros["TRANSPOSE"] = int(transpose)
        define_macros["CACHE_PRIM"] = int(cache_prim)
        define_macros["CACHE_GRAD"] = int(cache_grad)
        define_macros["USE_RK"] = int(time_integration != "fwd")

        if type(reconstruction) is str:
            mode, plm_theta = reconstruction, 0.0
            define_macros["USE_PLM"] = 0

        if type(reconstruction) is tuple:
            mode, plm_theta = reconstruction
            define_macros["USE_PLM"] = 1
            if not cache_grad:
                device_funcs.append(plm_minmod)

        if not cache_prim:
            device_funcs.append(cons_to_prim)

        device_funcs.append(riemann_hlle)
        device_funcs.append(self._godunov_fluxes)

        self._dim = dim
        self._plm_theta = plm_theta
        self._transpose = transpose
        self._define_macros = define_macros
        self._device_funcs = device_funcs

    @property
    def define_macros(self):
        return self._define_macros

    @property
    def device_funcs(self):
        return self._device_funcs

    def array_shape(self, a):
        if self._transpose:
            if self._dim == 1:
                return a.shape[1], 1, 1
            elif self._dim == 2:
                return a.shape[1], a.shape[2], 1
            elif self._dim == 3:
                return a.shape[1:4]
        else:
            if self._dim == 1:
                return a.shape[0], 1, 1
            elif self._dim == 2:
                return a.shape[0], a.shape[1], 1
            elif self._dim == 3:
                return a.shape[0:3]

    @device
    def _godunov_fluxes(self):
        R"""
        DEVICE void _godunov_fluxes(
            double *prd,
            double *grd,
            double *urd,
            double fh[NCONS],
            double plm_theta,
            int axis,
            int si,
            int sq)
        {
            double pp[NCONS];
            double pm[NCONS];

            // =====================================================
            #if USE_PLM == 0 && CACHE_PRIM == 0
            double ul[NCONS];
            double ur[NCONS];

            for (int q = 0; q < NCONS; ++q)
            {
                ul[q] = urd[-1 * si + q * sq];
                ur[q] = urd[+0 * si + q * sq];
            }
            cons_to_prim(ul, pm);
            cons_to_prim(ur, pp);

            // =====================================================
            #elif USE_PLM == 0 && CACHE_PRIM == 1
            for (int q = 0; q < NCONS; ++q)
            {
                pm[q] = prd[-1 * si + q * sq];
                pp[q] = prd[+0 * si + q * sq];
            }

            // =====================================================
            #elif USE_PLM == 1 && CACHE_PRIM == 0 && CACHE_GRAD == 0
            double u[4][NCONS];
            double p[4][NCONS];

            for (int q = 0; q < NCONS; ++q)
            {
                u[0][q] = urd[-2 * si + q * sq];
                u[1][q] = urd[-1 * si + q * sq];
                u[2][q] = urd[+0 * si + q * sq];
                u[3][q] = urd[+1 * si + q * sq];
            }
            cons_to_prim(u[0], p[0]);
            cons_to_prim(u[1], p[1]);
            cons_to_prim(u[2], p[2]);
            cons_to_prim(u[3], p[3]);

            for (int q = 0; q < NCONS; ++q)
            {
                double gl = plm_minmod(p[0][q], p[1][q], p[2][q], plm_theta);
                double gr = plm_minmod(p[1][q], p[2][q], p[3][q], plm_theta);
                pm[q] = p[1][q] + 0.5 * gl;
                pp[q] = p[2][q] - 0.5 * gr;
            }

            // =====================================================
            #elif USE_PLM == 1 && CACHE_PRIM == 0 && CACHE_GRAD == 1
            double ul[NCONS];
            double ur[NCONS];
            double pl[NCONS];
            double pr[NCONS];

            for (int q = 0; q < NCONS; ++q)
            {
                ul[q] = urd[-1 * si + q * sq];
                ur[q] = urd[+0 * si + q * sq];
            }
            cons_to_prim(ul, pl);
            cons_to_prim(ur, pr);

            for (int q = 0; q < NCONS; ++q)
            {
                double gl = grd[-1 * si + q * sq];
                double gr = grd[+0 * si + q * sq];
                pm[q] = pl[q] + 0.5 * gl;
                pp[q] = pr[q] - 0.5 * gr;
            }

            // =====================================================
            #elif USE_PLM == 1 && CACHE_PRIM == 1 && CACHE_GRAD == 0
            double p[4][NCONS];

            for (int q = 0; q < NCONS; ++q)
            {
                p[0][q] = prd[-2 * si + q * sq];
                p[1][q] = prd[-1 * si + q * sq];
                p[2][q] = prd[+0 * si + q * sq];
                p[3][q] = prd[+1 * si + q * sq];
            }

            for (int q = 0; q < NCONS; ++q)
            {
                double gl = plm_minmod(p[0][q], p[1][q], p[2][q], plm_theta);
                double gr = plm_minmod(p[1][q], p[2][q], p[3][q], plm_theta);
                pm[q] = p[1][q] + 0.5 * gl;
                pp[q] = p[2][q] - 0.5 * gr;
            }

            // =====================================================
            #elif USE_PLM == 1 && CACHE_PRIM == 1 && CACHE_GRAD == 1
            for (int q = 0; q < NCONS; ++q)
            {
                double pl = prd[-1 * si + q * sq];
                double pr = prd[+0 * si + q * sq];
                double gl = grd[-1 * si + q * sq];
                double gr = grd[+0 * si + q * sq];
                pm[q] = pl + 0.5 * gl;
                pp[q] = pr - 0.5 * gr;
            }
            #endif

            // =====================================================
            riemann_hlle(pm, pp, fh, axis);
        }
        """

    @kernel
    def godunov_fluxes(
        self,
        prd: NDArray[float],
        grd: NDArray[float],
        urd: NDArray[float],
        fwr: NDArray[float],
        plm_theta: float = None,
        ni: int = None,
        nj: int = None,
        nk: int = None,
    ):
        R"""
        KERNEL void godunov_fluxes(
            double *prd,
            double *grd,
            double *urd,
            double *fwr,
            double plm_theta,
            int ni,
            int nj,
            int nk)
        {
            int nq = NCONS;
            int nd = ni * nj * nk * nq;

            #if TRANSPOSE == 0
            #if DIM == 1
            int si = nq;
            int sq = 1;
            #elif DIM == 2
            int si = nq * nj;
            int sj = nq;
            int sq = 1;
            #elif DIM == 3
            int si = nq * nj * nk;
            int sj = nq * nj;
            int sk = nq;
            int sq = 1;
            #endif
            #elif TRANSPOSE == 1
            #if DIM == 1
            int sq = ni;
            int si = 1;
            #elif DIM == 2
            int sq = nj * ni;
            int si = nj;
            int sj = 1;
            #elif DIM == 3
            int sq = nk * nj * ni;
            int si = nk * nj;
            int sj = nk;
            int sk = 1;
            #endif
            #endif

            double fm[NCONS];

            #if DIM == 1
            FOR_RANGE_1D(2, ni - 1)
            #elif DIM == 2
            FOR_RANGE_2D(2, ni - 1, 2, nj - 1)
            #elif DIM == 3
            FOR_RANGE_3D(2, ni - 1, 2, nj - 2, 2, nk - 1)
            #endif
            {
                #if DIM == 1
                int nc = i * si;
                #elif DIM == 2
                int nc = i * si + j * sj;
                #elif DIM == 3
                int nc = i * si + j * sj + k * sk;
                #endif

                #if DIM >= 1
                _godunov_fluxes(prd + nc, grd + 0 * nd + nc, urd + nc, fm, plm_theta, 1, si, sq);
                for (int q = 0; q < NCONS; ++q)
                {
                    fwr[0 * nd + nc + q * sq] = fm[q];
                }
                #endif

                #if DIM >= 2
                double gm[NCONS];
                _godunov_fluxes(prd + nc, grd + 1 * nd + nc, urd + nc, fm, plm_theta, 2, sj, sq);
                for (int q = 0; q < NCONS; ++q)
                {
                    fwr[1 * nd + nc + q * sq] = fm[q];
                }
                #endif

                #if DIM >= 3
                double hm[NCONS];
                _godunov_fluxes(prd + nc, grd + 2 * nd + nc, urd + nc, fm, plm_theta, 3, sk, sq);
                for (int q = 0; q < NCONS; ++q)
                {
                    fwr[2 * nd + nc + q * sq] = fm[q];
                }
                #endif
            }
        }
        """
        plm = plm_theta if plm_theta is not None else self._plm_theta
        dim = self._dim
        s = self.array_shape(urd)
        return s[:dim], (prd, grd, urd, fwr, plm, *s)

    @kernel
    def update_cons(
        self,
        prd: NDArray[float],
        grd: NDArray[float],
        urk: NDArray[float],
        urd: NDArray[float],
        uwr: NDArray[float],
        dt: float,
        dx: float,
        rk: float = 0.0,
        plm_theta: float = None,
        ni: int = None,
        nj: int = None,
        nk: int = None,
    ):
        R"""
        KERNEL void update_cons(
            double *prd,
            double *grd,
            double *urk,
            double *urd,
            double *uwr,
            double dt,
            double dx,
            double rk,
            double plm_theta,
            int ni,
            int nj,
            int nk)
        {
            int nq = NCONS;
            int nd = ni * nj * nk * nq;

            #if TRANSPOSE == 0
            #if DIM == 1
            int si = nq;
            int sq = 1;
            #elif DIM == 2
            int si = nq * nj;
            int sj = nq;
            int sq = 1;
            #elif DIM == 3
            int si = nq * nj * nk;
            int sj = nq * nj;
            int sk = nq;
            int sq = 1;
            #endif
            #elif TRANSPOSE == 1
            #if DIM == 1
            int sq = ni;
            int si = 1;
            #elif DIM == 2
            int sq = nj * ni;
            int si = nj;
            int sj = 1;
            #elif DIM == 3
            int sq = nk * nj * ni;
            int si = nk * nj;
            int sj = nk;
            int sk = 1;
            #endif
            #endif

            #if DIM >= 1
            double fm[NCONS];
            double fp[NCONS];
            #endif
            #if DIM >= 2
            double gm[NCONS];
            double gp[NCONS];
            #endif
            #if DIM >= 3
            double hm[NCONS];
            double hp[NCONS];
            #endif

            #if DIM == 1
            FOR_RANGE_1D(2, ni - 2)
            #elif DIM == 2
            FOR_RANGE_2D(2, ni - 2, 2, nj - 2)
            #elif DIM == 3
            FOR_RANGE_3D(2, ni - 2, 2, nj - 2, 2, nk - 2)
            #endif
            {
                #if DIM == 1
                int nccc = (i + 0) * si;
                int nrcc = (i + 1) * si;
                #elif DIM == 2
                int nccc = (i + 0) * si + (j + 0) * sj;
                int nrcc = (i + 1) * si + (j + 0) * sj;
                int ncrc = (i + 0) * si + (j + 1) * sj;
                #elif DIM == 3
                int nccc = (i + 0) * si + (j + 0) * sj + (k + 0) * sk;
                int nrcc = (i + 1) * si + (j + 0) * sj + (k + 0) * sk;
                int ncrc = (i + 0) * si + (j + 1) * sj + (k + 0) * sk;
                int nccr = (i + 0) * si + (j + 0) * sj + (k + 1) * sk;
                #endif

                #if DIM >= 1
                _godunov_fluxes(prd + nccc, grd + 0 * nd + nccc, urd + nccc, fm, plm_theta, 1, si, sq);
                _godunov_fluxes(prd + nrcc, grd + 0 * nd + nrcc, urd + nrcc, fp, plm_theta, 1, si, sq);
                #endif
                #if DIM >= 2
                _godunov_fluxes(prd + nccc, grd + 1 * nd + nccc, urd + nccc, gm, plm_theta, 2, sj, sq);
                _godunov_fluxes(prd + ncrc, grd + 1 * nd + ncrc, urd + ncrc, gp, plm_theta, 2, sj, sq);
                #endif
                #if DIM >= 3
                _godunov_fluxes(prd + nccc, grd + 2 * nd + nccc, urd + nccc, hm, plm_theta, 3, sk, sq);
                _godunov_fluxes(prd + nccr, grd + 2 * nd + nccr, urd + nccr, hp, plm_theta, 3, sk, sq);
                #endif

                for (int q = 0; q < NCONS; ++q)
                {
                    int n = nccc + q * sq;
                    double du = 0.0;

                    #if DIM >= 1
                    du -= fp[q] - fm[q];
                    #endif
                    #if DIM >= 2
                    du -= gp[q] - gm[q];
                    #endif
                    #if DIM >= 3
                    du -= hp[q] - hm[q];
                    #endif

                    du *= dt / dx;
                    uwr[n] = urd[n] + du;

                    #if USE_RK == 1
                    if (rk != 0.0)
                    {
                        uwr[n] *= (1.0 - rk);
                        uwr[n] += rk * urk[n];
                    }
                    #endif
                }
            }
        }
        """
        plm = plm_theta if plm_theta is not None else self._plm_theta
        dim = self._dim
        s = self.array_shape(urd)
        return s[:dim], (prd, grd, urk, urd, uwr, dt, dx, rk, plm, *s)

    @kernel
    def update_cons_from_fluxes(
        self,
        urk: NDArray[float],
        u: NDArray[float],
        f: NDArray[float],
        dt: float,
        dx: float,
        rk: float,
        ni: int = None,
        nj: int = None,
        nk: int = None,
    ):
        R"""
        KERNEL void update_cons_from_fluxes(
            double *urk,
            double *u,
            double *f,
            double dt,
            double dx,
            double rk,
            int ni,
            int nj,
            int nk)
        {
            int nq = NCONS;
            int nd = ni * nj * nk * nq;

            #if TRANSPOSE == 0
            #if DIM == 1
            int si = nq;
            int sq = 1;
            #elif DIM == 2
            int si = nq * nj;
            int sj = nq;
            int sq = 1;
            #elif DIM == 3
            int si = nq * nj * nk;
            int sj = nq * nj;
            int sk = nq;
            int sq = 1;
            #endif
            #elif TRANSPOSE == 1
            #if DIM == 1
            int sq = ni;
            int si = 1;
            #elif DIM == 2
            int sq = nj * ni;
            int si = nj;
            int sj = 1;
            #elif DIM == 3
            int sq = nk * nj * ni;
            int si = nk * nj;
            int sj = nk;
            int sk = 1;
            #endif
            #endif

            #if DIM == 1
            FOR_RANGE_1D(2, ni - 2)
            #elif DIM == 2
            FOR_RANGE_2D(2, ni - 2, 2, nj - 2)
            #elif DIM == 3
            FOR_RANGE_3D(2, ni - 2, 2, nj - 2, 2, nk - 2)
            #endif
            {
                #if DIM == 1
                int nccc = (i + 0) * si;
                int nrcc = (i + 1) * si;
                #elif DIM == 2
                int nccc = (i + 0) * si + (j + 0) * sj;
                int nrcc = (i + 1) * si + (j + 0) * sj;
                int ncrc = (i + 0) * si + (j + 1) * sj;
                #elif DIM == 3
                int nccc = (i + 0) * si + (j + 0) * sj + (k + 0) * sk;
                int nrcc = (i + 1) * si + (j + 0) * sj + (k + 0) * sk;
                int ncrc = (i + 0) * si + (j + 1) * sj + (k + 0) * sk;
                int nccr = (i + 0) * si + (j + 0) * sj + (k + 1) * sk;
                #endif

                double *uc = &u[nccc];
                #if USE_RK == 1
                double *u0 = &urk[nccc];
                #endif

                for (int q = 0; q < NCONS; ++q)
                {
                    double u1 = uc[q];

                    #if DIM >= 1
                    double fm = f[0 * nd + nccc + q * sq];
                    double fp = f[0 * nd + nrcc + q * sq];
                    u1 -= (fp - fm) * dt / dx;
                    #endif
                    #if DIM >= 2
                    double gm = f[1 * nd + nccc + q * sq];
                    double gp = f[1 * nd + ncrc + q * sq];
                    u1 -= (gp - gm) * dt / dx;
                    #endif
                    #if DIM >= 3
                    double hm = f[2 * nd + nccc + q * sq];
                    double hp = f[2 * nd + nccr + q * sq];
                    u1 -= (hp - hm) * dt / dx;
                    #endif

                    #if USE_RK == 1
                    if (rk != 0.0)
                    {
                        u1 *= (1.0 - rk);
                        u1 += rk * u0[q];
                    }
                    #endif

                    uc[q] = u1;
                }
            }
        }
        """
        dim = self._dim
        s = self.array_shape(u)
        return s[:dim], (urk, u, f, dt, dx, rk, *s)


class State:
    def __init__(self, n, t, u, cons_to_prim, transpose):
        self._n = n
        self._t = t
        self._u = u
        self._cons_to_prim = cons_to_prim
        self._transpose = transpose

    @property
    def iteration(self):
        return self._n

    @property
    def time(self):
        return self._t

    @property
    def primitive(self):
        u = self._u
        p = u.copy()
        self._cons_to_prim(u, p)
        if self._transpose:
            p = p.T
        try:
            return p.get()
        except AttributeError:
            return p

    @property
    def total_zones(self):
        if self._transpose:
            return self._u.shape[1]
        else:
            return self._u.shape[0]


def linear_shocktube(x):
    """
    A linear shocktube setup
    """

    l = x < 0.5
    r = logical_not(l)
    p = zeros(x.shape + (3,))
    p[l, :] = [1.0, 0.0, 1.000]
    p[r, :] = [0.1, 0.0, 0.125]
    return p


def cell_centers_1d(ni):
    from numpy import linspace

    xv = linspace(0.0, 1.0, ni + 1)
    xc = 0.5 * (xv[1:] + xv[:-1])
    return xc


class solver_kernels(NamedTuple):
    plm_gradient: Callable
    update_cons: Callable
    update_cons_from_fluxes: Callable
    godunov_fluxes: Callable
    prim_to_cons: Callable
    cons_to_prim: Callable


def solver(
    kernels: solver_kernels,
    checkpoint: dict,
    hardware: str,
    resolution: int,
    transpose: bool,
    cache_flux: bool,
    cache_prim: bool,
    cache_grad: bool,
    reconstruction: Reconstruction,
    time_integration: str,
) -> State:
    """
    Solver for the 1d euler equations in many configurations
    """
    (
        plm_gradient,
        update_cons,
        update_cons_from_fluxes,
        godunov_fluxes,
        prim_to_cons,
        cons_to_prim,
    ) = kernels

    xp = numpy_or_cupy(hardware)
    nz = resolution
    dx = 1.0 / nz
    dt = dx * 1e-1
    x = cell_centers_1d(nz)

    if checkpoint:
        p = checkpoint["primitive"]
        t = checkpoint["time"]
        n = checkpoint["iteration"]
    else:
        p = linear_shocktube(x)
        t = 0.0
        n = 0

    p = xp.array(p)

    # =========================================================================
    # Whether the data layout is transposed, i.e. adjacent memory locations are
    # the same field but in adjacent zones.
    # =========================================================================
    if transpose:
        p = xp.ascontiguousarray(p.T)

    # =========================================================================
    # Time integration scheme: fwd and rk1 should produce the same result, but
    # rk1 can be used to test the expense of the data which is not required for
    # fwd.
    # =========================================================================
    if time_integration == "fwd":
        rks = []
    elif time_integration == "rk1":
        rks = [0.0]
    elif time_integration == "rk2":
        rks = [0.0, 0.5]
    elif time_integration == "rk3":
        rks = [0.0, 3.0 / 4.0, 1.0 / 3.0]

    # =========================================================================
    # A buffer for the array of cached Runge-Kutta conserved fields
    # =========================================================================
    if rks:
        u0 = xp.zeros_like(p)  # RK cons
    else:
        u0 = None

    # =========================================================================
    # Buffers for either read-and-write conserved arrays (if single-step update,
    # i.e. no cache-flux is used) or buffers for the conserved data and an array
    # of Godunov fluxes.
    # =========================================================================
    if cache_flux:
        fh = xp.zeros_like(p)
        u1 = xp.zeros_like(p)
        prim_to_cons(p, u1)
    else:
        p1 = p if cache_prim else None
        u1 = xp.zeros_like(p)
        u2 = xp.zeros_like(p)
        prim_to_cons(p, u1)
        prim_to_cons(p, u2)

    # =========================================================================
    # A buffer for the primitive fields they or gradients are cached
    # =========================================================================
    if cache_prim or cache_grad:
        p1 = p
    else:
        p1 = None

    # =========================================================================
    # A buffer for the primitive field gradients if gradients are being cached
    # =========================================================================
    if cache_grad:
        g1 = xp.zeros_like(p)  # gradients
    else:
        g1 = None

    del p  # p is no longer needed, will free memory if possible

    yield State(n, t, u1, cons_to_prim, transpose)

    # =========================================================================
    # Main loop: yield states until the caller stops calling next
    # =========================================================================
    while True:
        if rks:
            u0[...] = u1[...]

        for rk in rks or [0.0]:
            if cache_prim:
                cons_to_prim(u1, p1)
            if cache_grad:
                plm_gradient(p1, g1)
            if cache_flux:
                godunov_fluxes(p1, g1, u1, fh)
                update_cons_from_fluxes(u0, u1, fh, dt, dx, rk)
            else:
                update_cons(p1, g1, u0, u1, u2, dt, dx)
                u1, u2 = u2, u1

        t += dt
        n += 1
        yield State(n, t, u1, cons_to_prim, transpose)


def make_solver_kernels(
    transpose: bool,
    cache_prim: bool,
    cache_grad: bool,
    reconstruction: Reconstruction,
    time_integration: str,
):
    grad_est = GradientEsimation(3, transpose, reconstruction)
    fields = Fields(1, transpose)
    scheme = Scheme(
        1,
        transpose,
        reconstruction,
        time_integration,
        cache_prim,
        cache_grad,
    )
    return solver_kernels(
        grad_est.plm_gradient,
        scheme.update_cons,
        scheme.update_cons_from_fluxes,
        scheme.godunov_fluxes,
        fields.prim_to_cons_array,
        fields.cons_to_prim_array,
    )


def make_solver(config: Sailfish, checkpoint: dict = None):
    """
    Construct the 1d solver from a config instance
    """
    kernels = make_solver_kernels(
        config.strategy.transpose,
        config.strategy.cache_prim,
        config.strategy.cache_grad,
        config.scheme.reconstruction,
        config.scheme.time_integration,
    )
    for kernel in kernels:
        logger.info(f"using kernel {kernel_metadata(kernel)}")

    return solver(
        kernels,
        checkpoint,
        config.hardware,
        config.domain.num_zones[0],
        config.strategy.transpose,
        config.strategy.cache_flux,
        config.strategy.cache_prim,
        config.strategy.cache_grad,
        config.scheme.reconstruction,
        config.scheme.time_integration,
    )
