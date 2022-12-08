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
    def __init__(self, reconstruction, cache_prim, transpose):
        define_macros = dict()
        device_funcs = [
            cons_to_prim,
            riemann_hlle,
            self._update_cons,
            self._godunov_fluxes,
        ]

        define_macros["TRANSPOSE"] = int(transpose)

        if cache_prim:
            define_macros["CACHE_PRIM"] = 1

        if type(reconstruction) is str:
            mode = reconstruction
            define_macros["FLUX_STENCIL_SIZE"] = 2
            self._plm_theta = 0.0  # unused
            assert mode == "pcm"

        if type(reconstruction) is tuple:
            mode, plm_theta = reconstruction
            self._plm_theta = plm_theta
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

    @property
    def static(self):
        return R"""
        #ifdef CPU_MODE
        #define blockIdx_x 0
        #define blockDim_x ni
        #define threadIdx_x i
        #else
        #define blockIdx_x (int)blockIdx.x
        #define blockDim_x (int)blockDim.x
        #define threadIdx_x (int)threadIdx.x
        #endif
        """

    @device
    def _update_cons(self):
        R"""
        DEVICE void _update_cons(
            double p[FLUX_STENCIL_SIZE + 1][NCONS],
            double u[NCONS],
            double plm_theta,
            double dt,
            double dx)
        {
            double fp[NCONS];
            double fm[NCONS];

            #if FLUX_STENCIL_SIZE == 2

            riemann_hlle(p[0], p[1], fm, 1);
            riemann_hlle(p[1], p[2], fp, 1);

            #elif FLUX_STENCIL_SIZE == 4

            double *pk = p[0];
            double *pl = p[1];
            double *pc = p[2];
            double *pr = p[3];
            double *ps = p[4];
            double pml[NCONS];
            double pmr[NCONS];
            double ppl[NCONS];
            double ppr[NCONS];

            for (int q = 0; q < NCONS; ++q)
            {
                double gl = plm_minmod(pk[q], pl[q], pc[q], plm_theta);
                double gc = plm_minmod(pl[q], pc[q], pr[q], plm_theta);
                double gr = plm_minmod(pc[q], pr[q], ps[q], plm_theta);

                pml[q] = pl[q] + 0.5 * gl;
                pmr[q] = pc[q] - 0.5 * gc;
                ppl[q] = pc[q] + 0.5 * gc;
                ppr[q] = pr[q] - 0.5 * gr;
            }

            riemann_hlle(pml, pmr, fm, 1);
            riemann_hlle(ppl, ppr, fp, 1);

            #endif

            for (int q = 0; q < NCONS; ++q)
            {
                u[q] -= (fp[q] - fm[q]) * dt / dx;
            }
        }
        """

    @kernel
    def update_cons(
        self,
        prd: NDArray[float],
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

            FOR_RANGE_1D(1, ni - 1)
            {
                double u[FLUX_STENCIL_SIZE + 1][NCONS];
                double p[FLUX_STENCIL_SIZE + 1][NCONS];
                int c = FLUX_STENCIL_SIZE / 2;

                #ifdef CACHE_PRIM

                for (int j = 0; j < FLUX_STENCIL_SIZE + 1; ++j)
                {
                    for (int q = 0; q < NCONS; ++q)
                    {
                        p[j][q] = prd[(i + j - c) * si + q * sq];
                    }
                }
                for (int q = 0; q < NCONS; ++q)
                {
                    u[c][q] = urd[i * si + q * sq];
                }

                #else

                for (int j = 0; j < FLUX_STENCIL_SIZE + 1; ++j)
                {
                    for (int q = 0; q < NCONS; ++q)
                    {
                        u[j][q] = urd[(i + j - c) * si + q * sq];
                    }
                    cons_to_prim(u[j], p[j]);
                }

                #endif

                _update_cons(p, u[c], plm_theta, dt, dx);

                for (int q = 0; q < NCONS; ++q)
                {
                    uwr[i * si + q * sq] = u[c][q];
                }
            }
        }
        """
        plm = self._plm_theta if plm_theta is None else plm_theta
        ii = -1 if self._transpose else 0
        return urd.shape[ii], (prd, urd, uwr, plm, dt, dx, urd.shape[ii])

    @device
    def _godunov_fluxes(self):
        R"""
        DEVICE void _godunov_fluxes(double p[FLUX_STENCIL_SIZE][NCONS], double fh[NCONS], double plm_theta)
        {
            double pm[NCONS];
            double pp[NCONS];

            #if FLUX_STENCIL_SIZE == 2

            double *pc = p[0];
            double *pr = p[1];

            for (int q = 0; q < NCONS; ++q)
            {
                pm[q] = pc[q];
                pp[q] = pr[q];
            }

            #elif FLUX_STENCIL_SIZE == 4

            double *pl = p[0];
            double *pc = p[1];
            double *pr = p[2];
            double *ps = p[3];

            for (int q = 0; q < NCONS; ++q)
            {
                pm[q] = pc[q] + 0.5 * plm_minmod(pl[q], pc[q], pr[q], plm_theta);
                pp[q] = pr[q] - 0.5 * plm_minmod(pc[q], pr[q], ps[q], plm_theta);
            }

            #endif

            riemann_hlle(pm, pp, fh, 1);
        }
        """

    @kernel
    def godunov_fluxes(
        self,
        urd: NDArray[float],
        fwr: NDArray[float],
        plm_theta: float = None,
        ni: int = None,
    ):
        R"""
        KERNEL void godunov_fluxes(double *urd, double *fwr, double plm_theta, int ni)
        {
            #if TRANSPOSE == 0
            int sq = 1;
            int si = NCONS;
            #elif TRANSPOSE == 1
            int sq = ni;
            int si = 1;
            #endif

            FOR_RANGE_1D(1, ni - 2)
            {
                double u[FLUX_STENCIL_SIZE][NCONS];
                double p[FLUX_STENCIL_SIZE][NCONS];
                double f[NCONS];

                for (int q = 0; q < NCONS; ++q)
                {
                    #if FLUX_STENCIL_SIZE == 2

                    u[0][q] = urd[(i + 0) * si + q * sq];
                    u[1][q] = urd[(i + 1) * si + q * sq];

                    #elif FLUX_STENCIL_SIZE == 4

                    u[0][q] = urd[(i - 1) * si + q * sq];
                    u[1][q] = urd[(i + 0) * si + q * sq];
                    u[2][q] = urd[(i + 1) * si + q * sq];
                    u[3][q] = urd[(i + 2) * si + q * sq];

                    #endif
                }
                for (int i = 0; i < FLUX_STENCIL_SIZE; ++i)
                {
                    cons_to_prim(u[i], p[i]);
                }
                _godunov_fluxes(p, f, plm_theta);

                for (int q = 0; q < NCONS; ++q)
                {
                    fwr[i * si + q * sq] = f[q];
                }
            }
        }
        """
        plm = self._plm_theta if plm_theta is None else plm_theta
        ii = -1 if self._transpose else 0
        return urd.shape[ii], (urd, fwr, plm, urd.shape[ii])


def update_cons_from_fluxes(u, f, dt, dx, transpose, xp):
    if not transpose:
        u[2:-2, :] -= xp.diff(f[1:-2, :], axis=0) * (dt / dx)
    else:
        u[:, 2:-2] -= xp.diff(f[:, 1:-2], axis=1) * (dt / dx)


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

    solver = Solver(reconstruction, cache_prim, transpose=transpose)
    p2c = PrimitiveToConserved(dim=1, transpose=transpose)

    prim_to_cons = p2c.prim_to_cons_array
    cons_to_prim = p2c.cons_to_prim_array
    update_cons = solver.update_cons
    godunov_fluxes = solver.godunov_fluxes

    if transpose:
        p = xp.ascontiguousarray(p.T)

    if not cache_flux:
        p1 = p if cache_prim else None
        u1 = xp.zeros_like(p)
        u2 = xp.zeros_like(p)
        prim_to_cons(p, u1)
        prim_to_cons(p, u2)
    else:
        fhat = xp.zeros_like(p)
        u1 = xp.zeros_like(p)
        prim_to_cons(p, u1)

    if time_integration == "fwd":
        rks = []
    elif time_integration == "rk1":
        rks = [0.0]
    elif time_integration == "rk2":
        rks = [0.0, 0.5]
    elif time_integration == "rk3":
        rks = [0.0, 3.0 / 4.0, 1.0 / 3.0]

    yield State(n, t, u1, cons_to_prim, transpose=transpose)

    while True:
        if not cache_flux:
            if not rks:
                if p1 is not None:
                    cons_to_prim(u1, p1)
                update_cons(p1, u1, u2, None, dt, dx)
                u1, u2 = u2, u1
            else:
                u0 = u1.copy()
                for rk in rks:
                    if p1 is not None:
                        cons_to_prim(u1, p1)
                    update_cons(p1, u1, u2, None, dt, dx)
                    u1, u2 = u2, u1
                    average_rk(u0, u1, rk)
        else:
            if not rks:
                godunov_fluxes(u1, fhat)
                update_cons_from_fluxes(u1, fhat, dt, dx, transpose, xp)
            else:
                u0 = u1.copy()
                for rk in rks:
                    godunov_fluxes(u1, fhat)
                    update_cons_from_fluxes(u1, fhat, dt, dx, transpose, xp)
                    average_rk(u0, u1, rk)

        t += dt
        n += 1
        yield State(n, t, u1, cons_to_prim, transpose=transpose)


def make_solver(app: Sailfish):
    return solver(
        app.hardware,
        app.domain.num_zones[0],
        data_layout=app.strategy.data_layout,
        cache_flux=app.strategy.cache_flux,
        cache_prim=app.strategy.cache_prim,
        reconstruction=app.scheme.reconstruction,
        time_integration=app.scheme.time_integration,
    )
