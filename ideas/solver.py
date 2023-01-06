"""
Multi-dimensional 2nd order Godunov solver using the method of lines
"""

from contextlib import nullcontext
from logging import getLogger
from math import prod
from multiprocessing.pool import ThreadPool
from textwrap import dedent
from typing import NamedTuple, Callable

from numpy import array, zeros, concatenate, argsort
from numpy.typing import NDArray

from kernels import kernel, kernel_class, device, kernel_metadata
from config import Sailfish, Strategy, Reconstruction, BoundaryCondition
from geometry import CoordinateBox, CartesianCoordinates, SphericalPolarCoordinates
from index_space import IndexSpace


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
class GradientEstimation:
    """
    Handles batch-generation of PLM gradient estimation
    """

    def __init__(self, config: Sailfish):
        self._dim = config.domain.dimensionality
        self._nfields = len(config.initial_data.primitive_fields)
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
        if config.physics.metric == "newtonian":
            self.hydro_lib = __import__("lib_euler")
        if config.physics.metric == "minkowski":
            self.hydro_lib = __import__("lib_srhd")
        self.dim = config.domain.dimensionality
        self.nprim = len(config.initial_data.primitive_fields)
        self.transpose = config.strategy.transpose
        self.gamma_law_index = config.physics.equation_of_state.gamma_law_index

    @property
    def define_macros(self):
        return dict(
            DIM=self.dim,
            NPRIM=self.nprim,
            TRANSPOSE=int(self.transpose),
            GAMMA_LAW_INDEX=self.gamma_law_index,
        )

    @property
    def device_funcs(self):
        return [
            self.hydro_lib.prim_to_cons,
            self.hydro_lib.cons_to_prim,
            self.hydro_lib.max_wavespeed,
        ]

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
        nq = self.nprim
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
        nq = self.nprim
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
class SourceTerms:
    """
    Handles the calculation of geometric and driving source terms
    """

    def __init__(self, config: Sailfish):
        self._dim = config.domain.dimensionality
        self._nprim = len(config.initial_data.primitive_fields)
        self._transpose = config.strategy.transpose

    @property
    def define_macros(self):
        return dict(
            DIM=self._dim,
            NPRIM=self._nprim,
            TRANSPOSE=int(self._transpose),
        )

    @property
    def device_funcs(self):
        from lib_euler import source_terms_spherical_polar

        return [source_terms_spherical_polar]

    @kernel
    def geometric_source_terms(
        self,
        p: NDArray[float],
        x: NDArray[float],
        s: NDArray[float],
        ni: int = None,
        nj: int = None,
        nk: int = None,
    ):
        R"""
        KERNEL void geometric_source_terms(double *p, double *x, double *s, int ni, int nj, int nk)
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
            int sf = TRANSPOSE ? 1 : nq; // stride associated with field data

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
                double x0 = x[(0 * nd + nccc) / sf];
                double x1 = x[(0 * nd + nrcc) / sf];

                #elif DIM == 2
                #error("not implemented")
                int nccc = (i + 0) * si + (j + 0) * sj;
                int nrcc = (i + 1) * si + (j + 0) * sj;
                int ncrc = (i + 0) * si + (j + 1) * sj;

                #elif DIM == 3
                #error("not implemented")
                int nccc = (i + 0) * si + (j + 0) * sj + (k + 0) * sk;
                int nrcc = (i + 1) * si + (j + 0) * sj + (k + 0) * sk;
                int ncrc = (i + 0) * si + (j + 1) * sj + (k + 0) * sk;
                int nccr = (i + 0) * si + (j + 0) * sj + (k + 1) * sk;
                #endif

                double pc[NPRIM];
                double sc[NCONS];

                for (int q = 0; q < NPRIM; ++q)
                {
                    pc[q] = p[nccc + q * sq];
                }
                source_terms_spherical_polar(x0, x1, 0.5 * M_PI - 1e-6, 0.5 * M_PI + 1e-6, pc, sc); // TODO: polar extent

                for (int q = 0; q < NPRIM; ++q)
                {
                    s[nccc + q * sq] += sc[q];
                }
            }
        }
        """
        dim = self._dim
        shape = s.shape[:3]
        return shape[:dim], (p, x, s, *shape)


@kernel_class
class Scheme:
    """
    Godunov scheme using method-of-lines and many strategy modes
    """

    def __init__(self, config: Sailfish):
        if config.physics.metric == "newtonian":
            hydro_lib = __import__("lib_euler")
        if config.physics.metric == "minkowski":
            hydro_lib = __import__("lib_srhd")

        device_funcs = list()
        define_macros = dict()
        define_macros["DIM"] = config.domain.dimensionality
        define_macros["NPRIM"] = len(config.initial_data.primitive_fields)
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
            device_funcs.append(hydro_lib.cons_to_prim)

        device_funcs.append(hydro_lib.riemann_hlle)
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

    godunov_fluxes_code = R"""
    KERNEL void godunov_fluxes(
        double *prd,
        double *urd,
        double *grd,
        double *fwr,
        double *da,
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

        int sf = TRANSPOSE ? 1 : nq; // stride associated with field data
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

            double am;

            #if DIM >= 1
            am = da[(0 * nd + nc) / sf];

            _godunov_fluxes(prd + nc, grd + 0 * nd + nc, urd + nc, fm, plm_theta, 1, si, sq);
            for (int q = 0; q < NCONS; ++q)
            {
                fwr[0 * nd + nc + q * sq] = fm[q] * am;
            }
            #endif

            #if DIM >= 2
            am = da[(1 * nd + nc) / sf];

            _godunov_fluxes(prd + nc, grd + 1 * nd + nc, urd + nc, fm, plm_theta, 2, sj, sq);
            for (int q = 0; q < NCONS; ++q)
            {
                fwr[1 * nd + nc + q * sq] = fm[q] * am;
            }
            #endif

            #if DIM >= 3
            am = da[(2 * nd + nc) / sf];

            _godunov_fluxes(prd + nc, grd + 2 * nd + nc, urd + nc, fm, plm_theta, 3, sk, sq);
            for (int q = 0; q < NCONS; ++q)
            {
                fwr[2 * nd + nc + q * sq] = fm[q] * am;
            }
            #endif
        }
    }
    """

    @kernel(code=godunov_fluxes_code)
    def godunov_fluxes(
        self,
        prd: NDArray[float],
        urd: NDArray[float],
        grd: NDArray[float],
        fwr: NDArray[float],
        da: NDArray[float],
        plm_theta: float = None,
        ni: int = None,
        nj: int = None,
        nk: int = None,
    ):
        """
        Compute Godunov fluxes

        Parameters
        ----------

        prd : `ndarray[(ni, nj, nk, nprim), float64]`

            Read-only primitive variable array.

            May be `None` if `cache_prim` is `False`. Strides must be `(ni,
            nj, nk, nprim)` if `data_layout == "fields-last"` or `(nprim, ni,
            nj, nk)` if `data_layout == "fields-first"`.

            If given, must be valid in all zones including guard zones.

        urd : `ndarray[(ni, nj, nk, ncons), float64]`

            Read-only array of conserved variable densities at RK sub-step.

            May be `None` if `cache_prim is True`.

            If given, must be valid in all zones including guard zones.

            Strides must be `(ni, nj, nk, ncons)` if `data_layout ==
            "fields-last"` or `(ncons, ni, nj, nk)` if `data_layout ==
            "fields-first"`.

        grd : `ndarray[(ni, nj, nk, dim, ncons), float64]`

            Read-only array of primitive variable scaled gradients.

            May be `None` if `cache_grad` is `False` or `reconstruction ==
            "pcm"`.

            If given, may be invalid in one layer of guard zones on each
            non-trivial array axis.

            Gradient data must be scaled by the local grid spacing in the
            respective direction. The dimensions are the same as the
            respective primitive variable field (see `GradientEstimation`
            class).

            Strides must be `(dim, ni, nj, nk, nprim)` if `data_layout ==
            "fields-last"` or `(dim, nprim, ni, nj, nk)` if `data_layout ==
            "fields-first"`.

        fwr : `ndarray[(ni, nj, nk, dim, ncons), float64]`

            Write-only array of conserved variable charges, updated by `dt`.
            Same layout policy as `urd`.

            Will be invalid in two layers of guard zones on the left and one
            layer of guard zones at the right of each non-trivial array axis.

        da : `ndarray[(dim, ni, nj, nk), float64]`

            Read-only array of face areas.

            Data layout must be `(dim, ni, nj, nk)`.

            May be invalid in two layers of guard zones on the left and one
            layer of guard zones at the right of each non-trivial array axis.
        """
        plm = plm_theta if plm_theta is not None else self._plm_theta
        dim = self._dim
        s = urd.shape[:3]
        return s[:dim], (prd, urd, grd, fwr, da, plm, *s)

    """
    Native implementation of the update_cons kernel
    """
    update_cons_code = R"""
    KERNEL void update_cons(
        double *prd,
        double *grd,
        double *urk,
        double *urd,
        double *uwr,
        double *stm,
        double *da,
        double *dv,
        double dt,
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
        int sf = TRANSPOSE ? 1 : nq; // stride associated with field data

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
                double am;
                double ap;

                #if DIM >= 1
                am = da[(0 * nd + nccc) / sf];
                ap = da[(0 * nd + nrcc) / sf];
                du -= fp[q] * ap - fm[q] * am;
                #endif
                #if DIM >= 2
                am = da[(1 * nd + nccc) / sf];
                ap = da[(1 * nd + ncrc) / sf];
                du -= gp[q] * ap - gm[q] * am;
                #endif
                #if DIM >= 3
                am = da[(2 * nd + nccc) / sf];
                ap = da[(2 * nd + nccr) / sf];
                du -= hp[q] * ap - hm[q] * am;
                #endif

                if (stm)
                {
                    du += stm[n];
                }

                du *= dt / dv[n / sf];
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

    @kernel(code=update_cons_code)
    def update_cons(
        self,
        prd: NDArray[float],
        grd: NDArray[float],
        urk: NDArray[float],
        urd: NDArray[float],
        uwr: NDArray[float],
        stm: NDArray[float],
        da: NDArray[float],
        dv: NDArray[float],
        dt: float,
        rk: float,
        plm_theta: float = None,
        ni: int = None,
        nj: int = None,
        nk: int = None,
    ):
        """
        Update conserved quantities without pre-computed Godunov fluxes

        Parameters
        ----------

        prd : `ndarray[(ni, nj, nk, nprim), float64]`

            Read-only primitive variable array.

            May be `None` if `cache_prim` is `False`. Strides must be `(ni,
            nj, nk, nprim)` if `data_layout == "fields-last"` or `(nprim, ni,
            nj, nk)` if `data_layout == "fields-first"`.

            If given, must be valid in all zones including guard zones.

        grd : `ndarray[(ni, nj, nk, dim, ncons), float64]`

            Read-only array of primitive variable scaled gradients.

            May be `None` if `cache_grad` is `False` or `reconstruction ==
            "pcm"`.

            If given, may be invalid in one layer of guard zones on each
            non-trivial array axis.

            Gradient data must be scaled by the local grid spacing in the
            respective direction. The dimensions are the same as the
            respective primitive variable field (see `GradientEstimation`
            class).

            Strides must be `(dim, ni, nj, nk, nprim)` if `data_layout ==
            "fields-last"` or `(dim, nprim, ni, nj, nk)` if `data_layout ==
            "fields-first"`.

        urk : `ndarray[(ni, nj, nk, ncons), float64]`

            Read-only array of conserved variable densities, cached at the
            most recent integer time level. May be `None` if `time_integration
            == "fwd"`. Same layout policy as `urd`.

            May be invalid in two layers of guard zones on each non-trivial
            array axis.

        urd : `ndarray[(ni, nj, nk, ncons), float64]`

            Read-only array of conserved variable densities at the RK
            sub-step. Same layout policy as `urd`.

            Required. If `urd` is given, then it must be `urd = urd / dv`
            where `dv` is an array of local cell volumes. Same layout policy
            as `urd`.

            May be invalid in two layers of guard zones on each non-trivial
            array axis.

        uwr : `ndarray[(ni, nj, nk, ncons), float64]`

            Write-only array of conserved variable densities, updated by `dt`.
            Same layout policy as `urd`.

            Will be invalid in two layers of guard zones on each non-trivial
            array axis.

        stm : `ndarray[(ni, nj, nk, ncons), float64]`

            Read-only array of source terms.

            May be `None`. Source terms must be volume-integrated and per unit
            time (rate-of-charge). Same data layout as `urd`.

            May be invalid in two layers of guard zones on each non-trivial
            array axis.

        da : `ndarray[(ni, nj, nk, dim), float64]`

            Read-only array of face areas.

            Data layout must be `(dim, ni, nj, nk)`.

            May be invalid in two layers of guard zones on the left and one
            layer of guard zones at the right of each non-trivial array axis.

        dv : `ndarray[(ni, nj, nk), float64]`

            Read-only array of cell volumes.

            Data layout must be `(ni, nj, nk)`.

            May be invalid in two layers of guard zones on the left and the
            right of each non-trivial array axis.

        dt : `float64`

           Time step size.

        rk : `float64`

           Runge-Kutta parameter.

           Unused if `time_integration == "fwd"`. The formula used is: `uwr =
           urk * rk + (uwr + du) * (1 - rk)` where `du` is the time-difference
           of the conserved density.
        """
        plm = plm_theta if plm_theta is not None else self._plm_theta
        dim = self._dim
        s = urd.shape[:3]
        return s[:dim], (prd, grd, urk, urd, uwr, stm, da, dv, dt, rk, plm, *s)

    """
    Native implementation of the update_cons_from_fluxes_code kernel
    """
    update_cons_from_fluxes_code = R"""
    KERNEL void update_cons_from_fluxes(
        double *urk,
        double *q,
        double *f,
        double *stm,
        double *dv,
        double dt,
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
        int sf = TRANSPOSE ? 1 : nq; // stride associated with field data

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

            double *uc = &q[nccc];
            #if USE_RK == 1
            double *u0 = &urk[nccc];
            #endif

            for (int q = 0; q < NCONS; ++q)
            {
                double du = 0.0;

                #if DIM >= 1
                double fm = f[0 * nd + nccc + q * sq];
                double fp = f[0 * nd + nrcc + q * sq];
                du -= fp - fm;
                #endif
                #if DIM >= 2
                double gm = f[1 * nd + nccc + q * sq];
                double gp = f[1 * nd + ncrc + q * sq];
                du -= gp - gm;
                #endif
                #if DIM >= 3
                double hm = f[2 * nd + nccc + q * sq];
                double hp = f[2 * nd + nccr + q * sq];
                du -= hp - hm;
                #endif

                if (stm)
                {
                    du += stm[nccc + q * sq];
                }

                du *= dt / dv[nccc / sf];
                double u1 = uc[q * sq] + du;

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

    @kernel(code=update_cons_from_fluxes_code)
    def update_cons_from_fluxes(
        self,
        urk: NDArray[float],
        q: NDArray[float],
        f: NDArray[float],
        stm: NDArray[float],
        dv: NDArray[float],
        dt: float,
        rk: float,
        ni: int = None,
        nj: int = None,
        nk: int = None,
    ):
        """
        Update conserved quantities using pre-computed Godunov fluxes

        Parameters
        ----------

        urk : `ndarray[(ni, nj, nk, ncons), float64]`

            Read-only array of conserved variable charges, cached at the most
            recent integer time level. May be `None` if `time_integration ==
            "fwd"`.

            Strides must be `(ni, nj, nk, ncons)` if `data_layout ==
            "fields-last"` or `(ncons, ni, nj, nk)` if `data_layout ==
            "fields-first"`.

            May be invalid in two layers of guard zones on each non-trivial
            array axis.

        q : `ndarray[(ni, nj, nk, ncons), float64]`

            Read-write array of conserved variable charges at the RK sub-step.

            On input, must be valid in all zones including guard zones. On
            output, will be invalid in two layers of guard zones on each
            non-trivial array axis. Same layout policy as `urk`.

        f : `ndarray[(ni, nj, nk, dim, ncons), float64]`

            Read-only array of Godunov fluxes, multiplied by face areas.

            Strides must be `(dim, ni, nj, nk, ncons)` if `data_layout ==
            "fields-last"` or `(dim, ncons, ni, nj, nk)` if `data_layout ==
            "fields-first"`.

            May be invalid in two layers of guard zones on the left and one
            layer of guard zones at the right of each non-trivial array axis.

        stm : `ndarray[(ni, nj, nk, ncons), float64]`

            Read-only array of source terms.

            May be `None`. Source terms must be volume-integrated and per unit
            time (rate-of-charge). Same data layou ras `urd`.

            May be invalid in two layers of guard zones on each non-trivial
            array axis.

        dt : `float64`

           Time step size.

        rk : `float64`

           Runge-Kutta parameter.

           Unused if `time_integration == "fwd"`. The formula used is: `uwr =
           urk * rk + (uwr + du) * (1 - rk)` where `du` is the time-difference
           of the conserved charge.
        """
        dim = self._dim
        s = q.shape[:3]
        return s[:dim], (urk, q, f, stm, dv, dt, rk, *s)


def apply_bc(
    u: NDArray[float],
    location: str,
    patches: list[NDArray[float]],
    kind="outflow",
):
    if kind == "reflecing":
        raise NotImplementedError("reflecing BC")

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
    geometric_source_terms: Callable


def make_solver_kernels(config: Sailfish, native_code: bool = False):
    """
    Build and return kernels needed by solver, or just the native code
    """
    grad_est = GradientEstimation(config)
    fields = Fields(config)
    scheme = Scheme(config)
    source_terms = SourceTerms(config)

    if native_code:
        return (
            grad_est.__native_code__,
            fields.__native_code__,
            scheme.__native_code__,
            source_terms.__native_code__,
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
            source_terms.geometric_source_terms,
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
        geometric_source_terms,
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
    # Arrays for grid geometry (TODO: don't compute these if trivial grid geometry)
    # =========================================================================
    if config.coordinates == "cartesian":
        coords = CartesianCoordinates()
    if config.coordinates == "spherical-polar":
        coords = SphericalPolarCoordinates()

    dv = space.create(xp.zeros, fields=1, data=coords.cell_volumes(box))
    da = space.create(xp.zeros, vectors=dim, data=coords.face_areas(box))
    xv = space.create(xp.zeros, vectors=dim, data=coords.cell_vertices(box))

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
    # Arrays for source terms, including driving fields
    # =========================================================================
    if config.forcing is not None or coords.needs_geometrical_source_terms:
        stm = space.create(xp.zeros, fields=ncons)
    else:
        stm = None

    if (forcing := config.forcing) is not None:
        udr = space.create(xp.zeros, fields=ncons)
        pdr = space.create(xp.zeros, fields=nprim, data=initial_prim(box))
        rdr = space.create(xp.zeros, fields=1, data=forcing.rate_array(box))
        prim_to_cons(pdr, udr)
        del pdr

    dt = yield PatchState(n, t, u1, interior_box, c2p_user, amax)

    # =========================================================================
    # Main loop: yield states until the caller stops calling next
    # =========================================================================
    while True:
        if rks:
            u0[...] = u1[...]

        for rk in rks or [0.0]:
            if stm is not None:
                stm[...] = 0.0

            if cache_prim:
                cons_to_prim(u1, p1)

            if forcing is not None:
                stm += (udr - u1) * rdr * dv

            if coords.needs_geometrical_source_terms:
                if p1 is None:
                    raise ValueError(
                        "need --cache-prim if geometric source terms (for now)"
                    )
                geometric_source_terms(p1, xv, stm)

            if cache_grad:
                plm_gradient(p1, g1)
            if cache_flux:
                godunov_fluxes(p1, u1, g1, fh, da)
                update_cons_from_fluxes(u0, u1, fh, stm, dv, dt, rk)
            else:
                update_cons(p1, g1, u0, u1, u2, stm, da, dv, dt, rk)
                u1, u2 = u2, u1

            yield FillGuardZones(u1)

        t += dt
        n += 1
        dt = yield PatchState(n, t, u1, interior_box, c2p_user, amax)


def doc():
    """
    Return a dictionary of documented methods
    """
    return dict(
        godunov_fluxes=Scheme.godunov_fluxes.__doc__,
        update_cons=Scheme.update_cons.__doc__,
        update_cons_from_fluxes=Scheme.update_cons_from_fluxes.__doc__,
        patch_solver=dedent(patch_solver.__doc__),
        make_solver=dedent(make_solver.__doc__),
    )


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
    ```
    """
    logger = getLogger("sailfish")

    for kernel in (kernels := make_solver_kernels(config)):
        logger.info(f"using kernel {kernel_metadata(kernel)}")

    boundary = config.boundary_condition
    strategy = config.strategy
    hardware = strategy.hardware
    num_patches = strategy.num_patches
    num_threads = strategy.num_threads
    gpu_streams = strategy.gpu_streams
    initial_prim = config.initial_data.primitive
    nprim = len(config.initial_data.primitive_fields)

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
            # p = initial_prim(box)
        b = box.extend(2)
        p = initial_prim(b)
        p = space.create(zeros, fields=nprim, data=p)

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

            # elif type(events[0]) is FillGuardZones:
            #    fill_guard_zones([e.array for e in events], boundary)
