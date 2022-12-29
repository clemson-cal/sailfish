"""
A solver is a generator function and a state object
"""


from contextlib import nullcontext
from logging import getLogger
from math import prod
from multiprocessing.pool import ThreadPool
from typing import NamedTuple, Callable

from numpy import array, zeros, concatenate, argsort
from numpy.typing import NDArray

from kernels import kernel, kernel_class, device, kernel_metadata
from lib_euler import prim_to_cons, cons_to_prim, riemann_hlle, max_wavespeed
from config import Sailfish, Strategy, Reconstruction, BoundaryCondition
from geometry import CoordinateBox
from index_space import IndexSpace

logger = getLogger("sailfish")


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
    """
    Handles batch-generation of PLM gradient estimation
    """

    def __init__(self, config: Sailfish):
        self._dim = dim = config.domain.dimensionality
        self._nfields = dim + 2
        self._transpose = config.strategy.transpose
        self._plm_theta = (
            config.scheme.reconstruction[1]
            if type(config.scheme.reconstruction) is tuple
            else 0.0
        )

    @property
    def define_macros(self):
        return dict(
            NFIELDS=self._nfields,
            DIM=self._dim,
            TRANSPOSE=int(self._transpose),
        )

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
        nj: int = None,
        nk: int = None,
    ):
        R"""
        KERNEL void plm_gradient(double *y, double *g, double plm_theta, int ni, int nj, int nk)
        {
            int nq = NFIELDS;
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
            FOR_RANGE_1D(1, ni - 1)
            #elif DIM == 2
            FOR_RANGE_2D(1, ni - 1, 1, nj - 1)
            #elif DIM == 3
            FOR_RANGE_3D(1, ni - 1, 1, nj - 1, 1, nk - 1)
            #endif
            {
                #if DIM == 1
                int nccc = (i + 0) * si;
                int nlcc = (i - 1) * si;
                int nrcc = (i + 1) * si;
                #elif DIM == 2
                int nccc = (i + 0) * si + (j + 0) * sj;
                int nlcc = (i - 1) * si + (j + 0) * sj;
                int nrcc = (i + 1) * si + (j + 0) * sj;
                int nclc = (i + 0) * si + (j - 1) * sj;
                int ncrc = (i + 0) * si + (j + 1) * sj;
                #elif DIM == 3
                int nccc = (i + 0) * si + (j + 0) * sj + (k + 0) * sk;
                int nlcc = (i - 1) * si + (j + 0) * sj + (k + 0) * sk;
                int nrcc = (i + 1) * si + (j + 0) * sj + (k + 0) * sk;
                int nclc = (i + 0) * si + (j - 1) * sj + (k + 0) * sk;
                int ncrc = (i + 0) * si + (j + 1) * sj + (k + 0) * sk;
                int nccl = (i + 0) * si + (j + 0) * sj + (k - 1) * sk;
                int nccr = (i + 0) * si + (j + 0) * sj + (k + 1) * sk;
                #endif

                for (int q = 0; q < NFIELDS; ++q)
                {
                    #if DIM >= 1
                    {
                        double yc = y[nccc + q * sq];
                        double yl = y[nlcc + q * sq];
                        double yr = y[nrcc + q * sq];
                        g[0 * nd + nccc + q * sq] = plm_minmod(yl, yc, yr, plm_theta);
                    }
                    #endif
                    #if DIM >= 2
                    {
                        double yc = y[nccc + q * sq];
                        double yl = y[nclc + q * sq];
                        double yr = y[ncrc + q * sq];
                        g[1 * nd + nccc + q * sq] = plm_minmod(yl, yc, yr, plm_theta);
                    }
                    #endif
                    #if DIM >= 3
                    {
                        double yc = y[nccc + q * sq];
                        double yl = y[nccl + q * sq];
                        double yr = y[nccr + q * sq];
                        g[2 * nd + nccc + q * sq] = plm_minmod(yl, yc, yr, plm_theta);
                    }
                    #endif
                }
            }
        }
        """
        plm = plm_theta if plm_theta is not None else self._plm_theta
        dim = self._dim
        s = y.shape[:3]
        return s[:dim], (y, g, plm, *s)


@kernel_class
class Fields:
    """
    Handles conversion between primitive and conserved hydrodynamic fields

    This kernel class provides cons_to_prim_array and prim_to_cons_array
    functions which can operate either in fields-last or fields-first
    (struct-of-arrays, or transposed) data layout. These functions treat the
    input and output arrays as flattened, so they work for any domain
    dimensionality. The dim parameter is used to infer the number of fields.
    """

    def __init__(self, config: Sailfish):
        self.dim = config.domain.dimensionality
        self.transpose = config.strategy.transpose
        self.gamma_law_index = config.physics.equation_of_state.gamma_law_index

    @property
    def define_macros(self):
        return dict(
            DIM=self.dim,
            TRANSPOSE=int(self.transpose),
            GAMMA_LAW_INDEX=self.gamma_law_index,
        )

    @property
    def device_funcs(self):
        return [prim_to_cons, cons_to_prim, max_wavespeed]

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
        return p.size // nq, (p, u, p.size // nq)

    @kernel
    def max_wavespeeds_array(
        self,
        u: NDArray[float],
        a: NDArray[float],
        ni: int = None,
    ):
        R"""
        KERNEL void max_wavespeeds_array(double *u, double *a, int ni)
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
                    a[i] = max_wavespeed(p_reg);
                }
            }
        }
        """
        return a.size, (u, a, a.size)


@kernel_class
class Scheme:
    """
    Godunov scheme using method-of-lines and many strategy modes
    """

    def __init__(self, config: Sailfish):
        device_funcs = list()
        define_macros = dict()
        define_macros["DIM"] = config.domain.dimensionality
        define_macros["TRANSPOSE"] = int(config.strategy.transpose)
        define_macros[
            "GAMMA_LAW_INDEX"
        ] = config.physics.equation_of_state.gamma_law_index
        define_macros["CACHE_PRIM"] = int(config.strategy.cache_prim)
        define_macros["CACHE_GRAD"] = int(config.strategy.cache_grad)
        define_macros["USE_RK"] = int(config.scheme.time_integration != "fwd")

        r = config.scheme.reconstruction

        if type(r) is str:
            mode, plm_theta = r, 0.0
            define_macros["USE_PLM"] = 0

        if type(r) is tuple:
            mode, plm_theta = r
            define_macros["USE_PLM"] = 1
            if not config.strategy.cache_grad:
                device_funcs.append(plm_minmod)

        if not config.strategy.cache_prim:
            device_funcs.append(cons_to_prim)

        device_funcs.append(riemann_hlle)
        device_funcs.append(self._godunov_fluxes)

        self._dim = config.domain.dimensionality
        self._plm_theta = plm_theta
        self._transpose = config.strategy.transpose
        self._define_macros = define_macros
        self._device_funcs = device_funcs

    @property
    def define_macros(self):
        return self._define_macros

    @property
    def device_funcs(self):
        return self._device_funcs

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
                _godunov_fluxes(prd + nc, grd + 1 * nd + nc, urd + nc, fm, plm_theta, 2, sj, sq);
                for (int q = 0; q < NCONS; ++q)
                {
                    fwr[1 * nd + nc + q * sq] = fm[q];
                }
                #endif

                #if DIM >= 3
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
        s = urd.shape[:3]
        return s[:dim], (prd, grd, urd, fwr, plm, *s)

    @kernel
    def update_cons(
        self,
        prd: NDArray[float],
        grd: NDArray[float],
        urk: NDArray[float],
        urd: NDArray[float],
        uwr: NDArray[float],
        ubf: NDArray[float],
        rbf: NDArray[float],
        dt: float,
        dx: float,
        rk: float,
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
            double *ubf,
            double *rbf,
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
                int mccc = i * si;
                #elif DIM == 2
                int nccc = (i + 0) * si + (j + 0) * sj;
                int nrcc = (i + 1) * si + (j + 0) * sj;
                int ncrc = (i + 0) * si + (j + 1) * sj;
                int mccc = i * si + j * sj;
                #elif DIM == 3
                int nccc = (i + 0) * si + (j + 0) * sj + (k + 0) * sk;
                int nrcc = (i + 1) * si + (j + 0) * sj + (k + 0) * sk;
                int ncrc = (i + 0) * si + (j + 1) * sj + (k + 0) * sk;
                int nccr = (i + 0) * si + (j + 0) * sj + (k + 1) * sk;
                int mccc = i * si + j * sj + k * sk;
                #endif

                #if TRANSPOSE == 0
                mccc /= nq;
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

                    if (ubf && rbf)
                    {
                        du -= (urd[nccc + q * sq] - ubf[nccc + q * sq]) * rbf[mccc] * dt;
                    }
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
        s = urd.shape[:3]
        return s[:dim], (prd, grd, urk, urd, uwr, ubf, rbf, dt, dx, rk, plm, *s)

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
                    double u1 = uc[q * sq];

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
                        u1 += rk * u0[q * sq];
                    }
                    #endif

                    uc[q * sq] = u1;
                }
            }
        }
        """
        dim = self._dim
        s = u.shape[:3]
        return s[:dim], (urk, u, f, dt, dx, rk, *s)


def apply_bc(
    u: NDArray[float],
    location: str,
    patches: list[NDArray[float]],
    kind="outflow",
):
    if location == "lower_i":
        if kind == "outflow":
            u[:+2, :, :] = u[+2:+3, :, :]
        if kind == "periodic":
            u[:+2, :, :] = patches[-1][-4:-2, :, :]

    if location == "upper_i":
        if kind == "outflow":
            u[-2:, :, :] = u[-4:-3, :, :]
        if kind == "periodic":
            u[-2:, :, :] = patches[0][+2:+4, :, :]

    if location == "lower_j":
        if kind == "outflow":
            u[:, :+2, :] = u[:, +2:+3, :]
        if kind == "periodic":
            u[:, :+2, :] = patches[-1][:, -4:-2, :]

    if location == "upper_j":
        if kind == "outflow":
            u[:, -2:, :] = u[:, -4:-3, :]
        if kind == "periodic":
            u[:, -2:, :] = patches[0][:, +2:+4, :]

    if location == "lower_k":
        if kind == "outflow":
            u[:, :, :+2] = u[:, :, +2:+3]
        if kind == "periodic":
            u[:, :, :+2] = patches[-1][:, :, -4:-2]

    if location == "upper_k":
        if kind == "outflow":
            u[:, :, -2:] = u[:, :, -4:-3]
        if kind == "periodic":
            u[:, :, -2:] = patches[0][:, :, +2:+4]


def fill_guard_zones(arrays: list[NDArray[float]], boundary: BoundaryCondition):
    """
    Set guard zone data for a sequence of patches decomposed along the i-axis
    """
    for i, a in enumerate(arrays):
        # =========================================================================
        # Apply physical boundary conditions at the domain edges
        # =========================================================================
        if a.shape[0] > 1:
            if i == 0:
                apply_bc(a, "lower_i", arrays, boundary.lower_i)
            if i == len(arrays) - 1:
                apply_bc(a, "upper_i", arrays, boundary.upper_i)

        if a.shape[1] > 1:
            apply_bc(a, "lower_j", arrays, boundary.lower_j)
            apply_bc(a, "upper_j", arrays, boundary.upper_j)

        if a.shape[2] > 1:
            apply_bc(a, "lower_k", arrays, boundary.lower_k)
            apply_bc(a, "upper_k", arrays, boundary.upper_k)

        # =========================================================================
        # Copy data between neighboring strips
        # =========================================================================
        if i > 0:
            a[:+2, :, :] = arrays[i - 1][-4:-2, :, :]
        if i < len(arrays) - 1:
            a[-2:, :, :] = arrays[i + 1][+2:+4, :, :]


class FillGuardZones:
    """
    Message class; yielded by patch solvers te request boundary data
    """

    def __init__(self, array):
        self.array = array


class PatchState:
    """ """

    def __init__(self, n, t, u, box, to_user_prim, max_wavespeed):
        self._n = n
        self._t = t
        self._u = u
        self._box = box
        self._to_user_prim = to_user_prim
        self._max_wavespeed = max_wavespeed

    @property
    def box(self):
        return self._box

    @property
    def iteration(self):
        return self._n

    @property
    def time(self):
        return self._t

    @property
    def primitive(self):
        return self._to_user_prim(self._u)

    @property
    def total_zones(self):
        return prod(self._box.num_zones)

    @property
    def cell_centers(self):
        return self._box.cell_centers()

    def minimum_zone_size(self):
        return self._box.grid_spacing[0]  # assume square zones

    def maximum_wavespeed(self):
        return self._max_wavespeed(self._u)

    def timestep(self, cfl_number):
        return cfl_number * self.minimum_zone_size() / self.maximum_wavespeed()


class State:
    """ """

    def __init__(self, box: CoordinateBox, states: list[PatchState]):
        self._box = box
        self._states = states

    @property
    def box(self):
        return self._box

    @property
    def iteration(self):
        return self._states[0].iteration

    @property
    def time(self):
        return self._states[0].time

    @property
    def primitive(self):
        return concatenate([s.primitive for s in self._states])

    @property
    def total_zones(self):
        return sum(s.total_zones for s in self._states)

    @property
    def cell_centers(self):
        return concatenate([s.cell_centers for s in self._states])

    def minimum_zone_size(self):
        return float(min(s.minimum_zone_size() for s in self._states))

    def maximum_wavespeed(self):
        return float(max(s.maximum_wavespeed() for s in self._states))

    def timestep(self, cfl_number):
        return cfl_number * self.minimum_zone_size() / self.maximum_wavespeed()


class MockWorkerPool:
    """
    A no-overhead alternative to a thread pool

    This class should be used as the context manager rather than ThreadPool if
    num_threads=1.
    """

    def __enter__(self):
        return self

    def __exit__(self, *excinfo):
        pass

    map = map


def make_worker_pool(num_threads):
    """
    Return a thread pool if num_threads > 1 or otherwise a mock worker pool

    Use of a mock pool is for performance reasons; the overhead of job
    submission to a thread pool is not acceptable where concurrent kernel
    launches are needed, since getting kernels to overlap in time may require
    very low launch overhead. The overhead also exists for a thread pool with
    a single worker, so if one thread was requested then we revert to the mock
    pool.
    """
    if num_threads == 1:
        return MockWorkerPool()
    else:
        return ThreadPool(num_threads)


class SolverKernels(NamedTuple):
    """
    Collection of kernel functions used by the solver
    """

    plm_gradient: Callable
    update_cons: Callable
    update_cons_from_fluxes: Callable
    godunov_fluxes: Callable
    prim_to_cons: Callable
    cons_to_prim: Callable
    max_wavespeeds: Callable


def make_solver_kernels(config: Sailfish, native_code: bool = False):
    """
    Build and return kernels needed by solver, or just the native code
    """
    nfields = config.domain.dimensionality + 2
    grad_est = GradientEsimation(config)
    fields = Fields(config)
    scheme = Scheme(config)

    if native_code:
        return (
            grad_est.__native_code__,
            fields.__native_code__,
            scheme.__native_code__,
        )
    else:
        return SolverKernels(
            grad_est.plm_gradient,
            scheme.update_cons,
            scheme.update_cons_from_fluxes,
            scheme.godunov_fluxes,
            fields.prim_to_cons_array,
            fields.cons_to_prim_array,
            fields.max_wavespeeds_array,
        )


def make_stream(hardware: str, gpu_streams: str):
    """
    Return a maybe-concurrent execution context
    """
    if hardware == "cpu":
        return nullcontext()
    if hardware == "gpu":
        from cupy.cuda import Stream

        if gpu_streams == "per-thread":
            return Stream.ptds
        if gpu_streams == "per-patch":
            return Stream()


def patch_solver(
    primitive: NDArray[float],
    time: float,
    iteration: int,
    box: CoordinateBox,
    space: IndexSpace,
    kernels: SolverKernels,
    config: Sailfish,
) -> State:
    """
    A generator to drive time-integration of the physics state on a grid patch

    This generator yields either `State` objects or a message class:
    `FillGuardZones` or `CorrectFluxes`. It needs to be sent timestep sizes.
    Example usage:

    ```python
    solver = patch_solver(*args)
    timestep = None

    while True:
        event = solver.send(timestep)

        if type(event) is FillGuardZones:
            apply_bc(event.array)
        elif type(event) is CorrectFluxes:
            correct_fluxes(event.array)
        else:
            state = event
            timestep = state.timestep(cfl_number)
    ```
    """
    scheme = config.scheme
    strategy = config.strategy
    hardware = strategy.hardware
    transpose = strategy.transpose
    cache_flux = strategy.cache_flux
    cache_prim = strategy.cache_prim
    cache_grad = strategy.cache_grad
    time_integration = scheme.time_integration
    initial_prim = config.initial_data.primitive

    if hardware == "gpu":
        import cupy as xp
    if hardware == "cpu":
        import numpy as xp

    (
        plm_gradient,
        update_cons,
        update_cons_from_fluxes,
        godunov_fluxes,
        prim_to_cons,
        cons_to_prim,
        max_wavespeeds,
    ) = kernels

    dim = box.dimensionality
    nprim = primitive.shape[-1]
    ncons = nprim
    dx = box.grid_spacing[0]
    p = xp.array(primitive)
    t = time
    n = iteration
    a = space.create(xp.zeros)  # wavespeeds array
    interior_box = box.trim(2)

    del primitive
    yield FillGuardZones(p)

    def c2p_user(u):
        """
        Return primitives in standard layout host memory and with no guards
        """
        p = space.create(xp.zeros, fields=nprim)
        cons_to_prim(u, p)

        try:
            return p[space.interior].get()
        except AttributeError:
            return p[space.interior]

    def amax(u):
        """
        Return the maximum wavespeed on this grid patch (guard zones excluded)
        """
        max_wavespeeds(u, a)
        return a[space.interior].max()

    # =========================================================================
    # Time integration scheme: fwd and rk1 should produce the same result, but
    # rk1 can be used to test the expense of caching the conserved variables,
    # which is not required for fwd.
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
    # Array of cached Runge-Kutta conserved fields
    # =========================================================================
    if rks:
        u0 = space.create(xp.zeros, fields=ncons)  # RK cons
    else:
        u0 = None

    # =========================================================================
    # Arrays for either read-only and write-only conserved arrays if
    # single-step update (i.e. no cache-flux is used) or otherwise buffers for
    # the conserved data and an array of Godunov fluxes.
    # =========================================================================
    if cache_flux:
        fh = space.create(xp.zeros, fields=ncons, vectors=dim)
        u1 = space.create(xp.zeros, fields=ncons)
        prim_to_cons(p, u1)
    else:
        p1 = p if cache_prim else None
        u1 = space.create(xp.zeros, fields=ncons)
        u2 = space.create(xp.zeros, fields=ncons)
        prim_to_cons(p, u1)
        prim_to_cons(p, u2)

    # =========================================================================
    # Array for the primitive fields if primitives or gradients are cached
    # =========================================================================
    if cache_prim or cache_grad:
        p1 = p
    else:
        p1 = None

    del p  # p is no longer needed, will free memory if possible

    # =========================================================================
    # Array for the primitive field gradients if gradients are being cached
    # =========================================================================
    if cache_grad:
        g1 = space.create(xp.zeros, fields=ncons, vectors=dim)
    else:
        g1 = None

    # =========================================================================
    # Arrays for target conserved values (ubf) and the driving rate (rbf)
    # =========================================================================
    if (buf := config.buffer) is not None:
        if strategy.cache_flux:
            raise NotImplementedError("buffer zone not implemented in godunov_fluxes")
        if buf.ramp != 0.0:
            raise NotImplementedError("buffer ramp not implemented")
        coordinate, inequality, value = buf.where.split()
        if coordinate != "x":
            raise NotImplementedError("buffer only implemented for x-direction")
        if inequality != "<":
            raise NotImplementedError("buffer only implemented for <")
        x0 = float(value)
        pbf = space.create(xp.zeros, fields=nprim, data=initial_prim(box))
        ubf = space.create(xp.zeros, fields=ncons)
        rbf = space.create(xp.zeros, data=buf.rate * (box.cell_centers()[0] < x0))
        prim_to_cons(pbf, ubf)
        del pbf
    else:
        ubf = None
        rbf = None

    dt = yield PatchState(n, t, u1, interior_box, c2p_user, amax)

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
                update_cons(p1, g1, u0, u1, u2, ubf, rbf, dt, dx, rk)
                u1, u2 = u2, u1

            yield FillGuardZones(u1)

        t += dt
        n += 1
        dt = yield PatchState(n, t, u1, interior_box, c2p_user, amax)


def native_code(config: Sailfish):
    """
    Return a list of native code strings used by kernel classes
    """
    return make_solver_kernels(config, native_code=True)


def make_solver(config: Sailfish, checkpoint: dict = None):
    """
    Construct a solver (generator) from a config instance

    The solver is a generator which yields `State` objects and expects to be
    sent timestep sizes. Example:

    ```python
    solver = make_solver(config)
    timestep = None

    while True:
        state = solver.send(timestep)
        timestep = state.timestep(cfl_number)
    """
    for kernel in (kernels := make_solver_kernels(config)):
        logger.info(f"using kernel {kernel_metadata(kernel)}")

    boundary = config.boundary_condition
    strategy = config.strategy
    hardware = strategy.hardware
    num_patches = strategy.num_patches
    num_threads = strategy.num_threads
    gpu_streams = strategy.gpu_streams
    initial_prim = config.initial_data.primitive

    streams = list()
    solvers = list()

    for (i0, i1), box in config.domain.decompose(num_patches):
        space = IndexSpace(box.num_zones, guard=2, layout=strategy.data_layout)

        if checkpoint:
            t = checkpoint["time"]
            n = checkpoint["iteration"]
            p = checkpoint["primitive"][i0:i1]
        else:
            t = 0.0
            n = 0
            p = initial_prim(box)

        p = space.create(zeros, fields=p.shape[-1], data=p)
        b = box.extend(2)

        stream = make_stream(hardware, gpu_streams)
        solver = patch_solver(p, t, n, b, space, kernels, config)
        streams.append(stream)
        solvers.append(solver)

    timestep = None

    def next_with(arg):
        context, gen = arg

        with context:
            return gen.send(timestep)

    with make_worker_pool(num_threads) as pool:
        while True:
            events = list(pool.map(next_with, zip(streams, solvers)))

            if type(events[0]) is PatchState:
                timestep = yield State(config.domain, events)

            elif type(events[0]) is FillGuardZones:
                fill_guard_zones([e.array for e in events], boundary)
