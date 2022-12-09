"""
A solver is a generator function and a state object
"""

from numpy import array, zeros, logical_not
from numpy.typing import NDArray
from new_kernels import kernel, kernel_class, device
from lib_euler import prim_to_cons, cons_to_prim, riemann_hlle
from app_config import Sailfish, Reconstruction


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
    def __init__(self, nfields, transpose):
        self.nfields = nfields
        self.transpose = transpose

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
        plm_theta: float,
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
        nq = self.nfields
        ii = -1 if self.transpose else 0
        iq = 0 if self.transpose else -1

        if y.shape[iq] != nq or y.shape != g.shape:
            raise ValueError("array has wrong number of fields")

        return y.shape[ii], (y, g, plm_theta, y.shape[ii])


@kernel_class
class PrimitiveToConserved:
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
class Solver:
    def __init__(self, reconstruction, time_integration, cache_prim, transpose):
        define_macros = dict()
        device_funcs = [
            cons_to_prim,
            riemann_hlle,
            self._godunov_fluxes,
        ]

        define_macros["TRANSPOSE"] = int(transpose)
        define_macros["CACHE_PRIM"] = int(cache_prim)
        define_macros["USE_RK"] = int(time_integration != "fwd")

        if type(reconstruction) is str:
            mode = reconstruction
            define_macros["USE_PLM"] = 0
            define_macros["FLUX_STENCIL_SIZE"] = 2
            self._plm_theta = 0.0  # unused
            assert mode == "pcm"

        if type(reconstruction) is tuple:
            mode, plm_theta = reconstruction
            self._plm_theta = plm_theta
            define_macros["USE_PLM"] = 1
            define_macros["FLUX_STENCIL_SIZE"] = 4
            device_funcs.insert(0, plm_minmod)
            assert mode == "plm"

        self._transpose = transpose
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
            double *prd, double *grd, double *urd, double fh[NCONS], double plm_theta, int ni, int i)
        {
            #if TRANSPOSE == 0
            int sq = 1;
            int si = NCONS;
            #elif TRANSPOSE == 1
            int sq = ni;
            int si = 1;
            #endif

            double pp[NCONS];
            double pm[NCONS];

            // =====================================================
            #if USE_PLM == 0 && CACHE_PRIM == 0

            double ul[NCONS];
            double ur[NCONS];

            for (int q = 0; q < NCONS; ++q)
            {
                ul[q] = urd[(i - 1) * si + q * sq];
                ur[q] = urd[(i + 0) * si + q * sq];
            }
            cons_to_prim(ul, pm);
            cons_to_prim(ur, pp);

            // =====================================================
            #elif USE_PLM == 0 && CACHE_PRIM == 1

            for (int q = 0; q < NCONS; ++q)
            {
                pm[q] = prd[(i - 1) * si + q * sq];
                pp[q] = prd[(i + 0) * si + q * sq];
            }
            (void) cons_to_prim; // unused

            // =====================================================
            #elif USE_PLM == 1 && CACHE_PRIM == 0 && CACHE_GRAD == 0
            double u[4][NCONS];
            double p[4][NCONS];

            for (int q = 0; q < NCONS; ++q)
            {
                u[0][q] = urd[(i - 2) * si + q * sq];
                u[1][q] = urd[(i + 1) * si + q * sq];
                u[2][q] = urd[(i + 0) * si + q * sq];
                u[3][q] = urd[(i + 1) * si + q * sq];
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
                ul[q] = urd[(i - 1) * si + q * sq];
                ur[q] = urd[(i + 0) * si + q * sq];
            }
            cons_to_prim(ul, pl);
            cons_to_prim(ur, pr);

            for (int q = 0; q < NCONS; ++q)
            {
                double gl = grd[(i - 1) * si + q * sq];
                double gr = grd[(i + 0) * si + q * sq];
                pm[q] = pl[q] + 0.5 * gl;
                pp[q] = pr[q] - 0.5 * gr;
            }

            // =====================================================
            #elif USE_PLM == 1 && CACHE_PRIM == 1 && CACHE_GRAD == 0
            double p[4][NCONS];

            for (int q = 0; q < NCONS; ++q)
            {
                p[0][q] = prd[(i - 2) * si + q * sq];
                p[1][q] = prd[(i - 1) * si + q * sq];
                p[2][q] = prd[(i + 0) * si + q * sq];
                p[3][q] = prd[(i + 1) * si + q * sq];
            }

            for (int q = 0; q < NCONS; ++q)
            {
                double gl = plm_minmod(p[0][q], p[1][q], p[2][q], plm_theta);
                double gr = plm_minmod(p[1][q], p[2][q], p[3][q], plm_theta);
                pm[q] = p[1][q] + 0.5 * gl;
                pp[q] = p[2][q] - 0.5 * gr;
            }
            (void) cons_to_prim; // unused

            // =====================================================
            #elif USE_PLM == 1 && CACHE_PRIM == 1 && CACHE_GRAD == 1
            double pp[NCONS];
            double pm[NCONS];

            for (int q = 0; q < NCONS; ++q)
            {
                double pl = prd[(i - 1) * si + q * sq];
                double pr = prd[(i + 0) * si + q * sq];
                double gl = grd[(i - 1) * si + q * sq];
                double gr = grd[(i + 0) * si + q * sq];
                pm[q] = pl + 0.5 * gl;
                pp[q] = pr - 0.5 * gr;
            }
            (void) cons_to_prim; // unused
            #endif

            // =====================================================
            riemann_hlle(pm, pp, fh, 1);
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
    ):
        R"""
        KERNEL void godunov_fluxes(double *prd, double *grd, double *urd, double *fwr, double plm_theta, int ni)
        {
            #if TRANSPOSE == 0
            int sq = 1;
            int si = NCONS;
            #elif TRANSPOSE == 1
            int sq = ni;
            int si = 1;
            #endif

            double fh[NCONS];

            FOR_RANGE_1D(2, ni - 1)
            {
                _godunov_fluxes(prd, grd, urd, fh, plm_theta, ni, i);

                for (int q = 0; q < NCONS; ++q)
                {
                    fwr[i * si + q * sq] = fh[q];
                }
            }
        }
        """
        plm = self._plm_theta if plm_theta is None else plm_theta
        ii = -1 if self._transpose else 0
        return urd.shape[ii], (prd, grd, urd, fwr, plm, urd.shape[ii])

    @kernel
    def update_cons(
        self,
        prd: NDArray[float],
        grd: NDArray[float],
        urd: NDArray[float],
        uwr: NDArray[float],
        plm_theta: float,
        dt: float,
        dx: float,
        ni: int = None,
    ):
        R"""
        KERNEL void update_cons(
            double *prd,
            double *grd,
            double *urd,
            double *uwr,
            double plm_theta,
            double dt,
            double dx,
            int ni)
        {
            #if TRANSPOSE == 0
            int sq = 1;
            int si = NCONS;
            #elif TRANSPOSE == 1
            int sq = ni;
            int si = 1;
            #endif

            double fm[NCONS];
            double fp[NCONS];

            FOR_RANGE_1D(2, ni - 2)
            {
                _godunov_fluxes(prd, grd, urd, fm, plm_theta, ni, i);
                _godunov_fluxes(prd, grd, urd, fp, plm_theta, ni, i + 1);

                for (int q = 0; q < NCONS; ++q)
                {
                    uwr[i * si + q * sq] = urd[i * si + q * sq] - (fp[q] - fm[q]) * dt / dx;
                }
            }
        }
        """
        plm = self._plm_theta if plm_theta is None else plm_theta
        ii = -1 if self._transpose else 0
        return urd.shape[ii], (prd, grd, urd, uwr, plm, dt, dx, urd.shape[ii])

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
    ):
        R"""
        KERNEL void update_cons_from_fluxes(
            double *urk,
            double *u,
            double *f,
            double dt,
            double dx,
            double rk,
            int ni)
        {
            #if TRANSPOSE == 0
            int sq = 1;
            int si = NCONS;
            #elif TRANSPOSE == 1
            int sq = ni;
            int si = 1;
            #endif

            FOR_RANGE_1D(2, ni - 2)
            {
                double *uc = &u[i * si];
                #if USE_RK == 1
                double *u0 = &urk[i * si];
                #endif

                for (int q = 0; q < NCONS; ++q)
                {
                    double fm = f[(i + 0) * si + q * sq];
                    double fp = f[(i + 1) * si + q * sq];
                    uc[q] -= (fp - fm) * dt / dx;
                    #if USE_RK == 1
                    if (rk != 0.0)
                    {
                        uc[q] *= (1.0 - rk);
                        uc[q] += rk * u0[q];
                    }
                    #endif
                }
            }
        }
        """
        ii = -1 if self._transpose else 0
        return u.shape[ii], (urk, u, f, dt, dx, rk, u.shape[ii])


def update_cons_from_fluxes_np(u, f, dt, dx, transpose, xp):
    if not transpose:
        u[2:-2, :] -= xp.diff(f[2:-1, :], axis=0) * (dt / dx)
    else:
        u[:, 2:-2] -= xp.diff(f[:, 2:-1], axis=1) * (dt / dx)


def average_rk(u0, u1, rk):
    if rk != 0.0:
        u1 *= 1.0 - rk
        u1 += u0 * rk


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


def solver(
    hardware: str,
    resolution: int,
    data_layout: str,
    cache_flux: bool,
    cache_prim: bool,
    cache_grad: bool,
    reconstruction: Reconstruction,
    time_integration: str,
) -> State:
    """
    Solver for the 1d euler equations in many configurations
    """

    xp = numpy_or_cupy(hardware)
    nz = resolution
    dx = 1.0 / nz
    dt = dx * 1e-1
    x = cell_centers_1d(nz)
    p = linear_shocktube(x)
    t = 0.0
    n = 0
    p = xp.array(p)
    transpose = data_layout == "fields-first"

    # =========================================================================
    # Solvers and solver functions
    # =========================================================================
    gradient_estimation = GradientEsimation(nfields=3, transpose=transpose)
    solver = Solver(reconstruction, time_integration, cache_prim, transpose)
    fields = PrimitiveToConserved(dim=1, transpose=transpose)
    plm_theta = reconstruction[1] if type(reconstruction) is tuple else 0.0
    plm_gradient = gradient_estimation.plm_gradient
    update_cons = solver.update_cons
    update_cons_from_fluxes_native = solver.update_cons_from_fluxes
    godunov_fluxes = solver.godunov_fluxes
    prim_to_cons = fields.prim_to_cons_array
    cons_to_prim = fields.cons_to_prim_array

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
    if not cache_flux:
        p1 = p if cache_prim else None
        u1 = xp.zeros_like(p)
        u2 = xp.zeros_like(p)
        prim_to_cons(p, u1)
        prim_to_cons(p, u2)
    else:
        fh = xp.zeros_like(p)
        u1 = xp.zeros_like(p)  # read-from cons
        prim_to_cons(p, u1)

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

    yield State(n, t, u1, cons_to_prim, transpose=transpose)

    # =========================================================================
    # Main loop: yield states until the caller stops calling next
    # =========================================================================
    while True:
        if not cache_flux:
            if not rks:
                if p1 is not None:
                    cons_to_prim(u1, p1)
                update_cons(p1, g1, u1, u2, plm_theta, dt, dx)
                u1, u2 = u2, u1
            else:
                u0[...] = u1[...]
                for rk in rks:
                    if p1 is not None:
                        cons_to_prim(u1, p1)
                    update_cons(p1, g1, u1, u2, plm_theta, dt, dx)
                    u1, u2 = u2, u1
                    average_rk(u0, u1, rk)
        else:
            if not rks:
                if p1 is not None:
                    cons_to_prim(u1, p1)
                if g1 is not None:
                    plm_gradient(p1, g1, plm_theta)
                godunov_fluxes(p1, g1, u1, fh)
                if True:
                    update_cons_from_fluxes_native(u0, u1, fh, dt, dx, 0.0)
                else:
                    update_cons_from_fluxes_np(u1, fh, dt, dx, transpose, xp)
            else:
                u0[...] = u1[...]
                for rk in rks:
                    if p1 is not None:
                        cons_to_prim(u1, p1)
                    if g1 is not None:
                        plm_gradient(p1, g1, plm_theta)
                    godunov_fluxes(p1, g1, u1, fh)
                    if True:
                        update_cons_from_fluxes_native(u0, u1, fh, dt, dx, rk)
                    else:
                        update_cons_from_fluxes_np(u1, fh, dt, dx, transpose, xp)
                        average_rk(u0, u1, rk)

        t += dt
        n += 1
        yield State(n, t, u1, cons_to_prim, transpose=transpose)


def make_solver(app: Sailfish):
    """
    Construct the 1d solver from an app instance
    """
    return solver(
        app.hardware,
        app.domain.num_zones[0],
        app.strategy.data_layout,
        app.strategy.cache_flux,
        app.strategy.cache_prim,
        app.strategy.cache_grad,
        app.scheme.reconstruction,
        app.scheme.time_integration,
    )
