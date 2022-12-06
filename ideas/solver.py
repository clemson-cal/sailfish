"""
A solver is a generator function and a state object
"""

from numpy import array, zeros, logical_not
from numpy.typing import NDArray
from new_kernels import kernel, kernel_class, device
from lib_euler import prim_to_cons, cons_to_prim, riemann_hlle
from app_config import Sailfish


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
    def __init__(self, dim=1):
        self.dim = dim

    @property
    def define_macros(self):
        return dict(DIM=self.dim)

    @property
    def device_funcs(self):
        return [prim_to_cons, cons_to_prim]

    @kernel
    def cons_to_prim_fields_last(
        self,
        u: NDArray[float],
        p: NDArray[float],
        ni: int = None,
    ):
        R"""
        KERNEL void cons_to_prim_fields_last(double *u, double *p, int ni)
        {
            FOR_EACH_1D(ni)
            {
                cons_to_prim(&u[NCONS * i], &p[NCONS * i]);
            }
        }
        """
        nq = self.dim + 2

        if u.shape[-1] != nq or u.shape != p.shape:
            raise ValueError("array has wrong number of fields")

        return u.size // nq, (u, p, u.size // nq)

    @kernel
    def prim_to_cons_fields_last(
        self,
        p: NDArray[float],
        u: NDArray[float],
        ni: int = None,
    ):
        R"""
        KERNEL void prim_to_cons_fields_last(double *p, double *u, int ni)
        {
            FOR_EACH_1D(ni)
            {
                prim_to_cons(&p[NCONS * i], &u[NCONS * i]);
            }
        }
        """
        nq = self.dim + 2

        if p.shape[-1] != nq or p.shape != u.shape:
            raise ValueError("array has wrong number of fields")

        return p.size // nq, (p, u, p.size // nq)

    @kernel
    def cons_to_prim_fields_first(
        self,
        u: NDArray[float],
        p: NDArray[float],
        ni: int = None,
    ):
        R"""
        KERNEL void cons_to_prim_fields_first(double *u, double *p, int ni)
        {
            #if defined(CPU_MODE)
            #define blockIdx_x 0
            #define blockDim_x ni
            #define threadIdx_x i
            #elif defined(GPU_MODE)
            #define blockIdx_x (int)blockIdx.x
            #define blockDim_x (int)blockDim.x
            #define threadIdx_x (int)threadIdx.x
            #endif

            FOR_EACH_1D(ni)
            {
                double *u_blk[NCONS];
                double *p_blk[NCONS];
                double u_reg[NCONS];
                double p_reg[NCONS];

                for (int q = 0; q < NCONS; ++q)
                {
                    p_blk[q] = &p[q * ni + blockIdx_x * blockDim_x];
                    u_blk[q] = &u[q * ni + blockIdx_x * blockDim_x];
                    u_reg[q] = u_blk[q][threadIdx_x];
                }
                cons_to_prim(u_reg, p_reg);

                for (int q = 0; q < NCONS; ++q)
                {
                    p_blk[q][threadIdx_x] = p_reg[q];
                }
            }
        }
        """
        nq = self.dim + 2

        if u.shape[0] != nq or u.shape != p.shape:
            raise ValueError("array has wrong number of fields")

        return u.size // nq, (u, p, u.size // nq)

    @kernel
    def prim_to_cons_fields_first(
        self,
        p: NDArray[float],
        u: NDArray[float],
        ni: int = None,
    ):
        R"""
        KERNEL void prim_to_cons_fields_first(double *p, double *u, int ni)
        {
            #if defined(CPU_MODE)
            #define blockIdx_x 0
            #define blockDim_x ni
            #define threadIdx_x i
            #elif defined(GPU_MODE)
            #define blockIdx_x blockIdx.x
            #define blockDim_x blockDim.x
            #define threadIdx_x threadIdx.x
            #endif

            FOR_EACH_1D(ni)
            {
                double *u_blk[NCONS];
                double *p_blk[NCONS];
                double u_reg[NCONS];
                double p_reg[NCONS];

                for (int q = 0; q < NCONS; ++q)
                {
                    u_blk[q] = &u[q * ni + blockIdx_x * blockDim_x];
                    p_blk[q] = &p[q * ni + blockIdx_x * blockDim_x];
                    p_reg[q] = p_blk[q][threadIdx_x];
                }
                prim_to_cons(p_reg, u_reg);

                for (int q = 0; q < NCONS; ++q)
                {
                    u_blk[q][threadIdx_x] = u_reg[q];
                }
            }
        }
        """
        nq = self.dim + 2

        if p.shape[0] != nq or p.shape != u.shape:
            raise ValueError("array has wrong number of fields")

        return p.size // nq, (p, u, p.size // nq)


@kernel_class
class FluxPerZoneSolver:
    @property
    def define_macros(self):
        return dict(DIM=1)

    @property
    def device_funcs(self):
        return [cons_to_prim, riemann_hlle, self._update_cons]

    @device
    def _update_cons(self):
        R"""
        DEVICE void _update_cons(double u[3][NCONS], double dt, double dx)
        {
            double p[3][NCONS];
            double fp[NCONS];
            double fm[NCONS];

            cons_to_prim(u[0], p[0]);
            cons_to_prim(u[1], p[1]);
            cons_to_prim(u[2], p[2]);
            riemann_hlle(p[0], p[1], fm, 1);
            riemann_hlle(p[1], p[2], fp, 1);

            for (int q = 0; q < NCONS; ++q)
            {
                u[1][q] -= (fp[q] - fm[q]) * dt / dx;
            }
        }
        """

    @kernel
    def update_cons_fields_last(
        self,
        urd: NDArray[float],
        uwr: NDArray[float],
        dt: float,
        dx: float,
        ni: int = None,
    ):
        R"""
        KERNEL void update_cons_fields_last(double *urd, double *uwr, double dt, double dx, int ni)
        {
            FOR_RANGE_1D(1, ni - 1)
            {
                double u[3][NCONS];

                for (int q = 0; q < NCONS; ++q)
                {
                    u[0][q] = urd[(i - 1) * NCONS + q];
                    u[1][q] = urd[(i + 0) * NCONS + q];
                    u[2][q] = urd[(i + 1) * NCONS + q];
                }
                _update_cons(u, dt, dx);

                for (int q = 0; q < NCONS; ++q)
                {
                    uwr[i * NCONS + q] = u[1][q];
                }
            }
        }
        """
        return urd.shape[0], (urd, uwr, dt, dx, urd.shape[0])

    @kernel
    def update_cons_fields_first(
        self,
        urd: NDArray[float],
        uwr: NDArray[float],
        dt: float,
        dx: float,
        ni: int = None,
    ):
        R"""
        KERNEL void update_cons_fields_first(double *urd, double *uwr, double dt, double dx, int ni)
        {
            #ifdef CPU_MODE
            #define blockIdx_x 0
            #define blockDim_x ni
            #define threadIdx_x i
            #else
            #define blockIdx_x (int)blockIdx.x
            #define blockDim_x (int)blockDim.x
            #define threadIdx_x (int)threadIdx.x
            #endif

            FOR_RANGE_1D(1, ni - 1)
            {
                double u[3][NCONS];

                for (int q = 0; q < NCONS; ++q)
                {
                    double *u_blk = &urd[q * ni + blockIdx_x * blockDim_x];
                    u[0][q] = u_blk[threadIdx_x - 1];
                    u[1][q] = u_blk[threadIdx_x + 0];
                    u[2][q] = u_blk[threadIdx_x + 1];
                }
                _update_cons(u, dt, dx);

                for (int q = 0; q < NCONS; ++q)
                {
                    double *u_blk = &uwr[q * ni + blockIdx_x * blockDim_x];
                    u_blk[threadIdx_x] = u[1][q];
                }
            }
        }
        """
        return urd.shape[1], (urd, uwr, dt, dx, urd.shape[1])


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


def simulation(
    hardware: str,
    resolution: int,
    data_layout="fields-last",
    time_integration="fwd",
) -> State:
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

    xp = numpy_or_cupy(hardware)
    nz = resolution
    dx = 1.0 / nz
    dt = dx * 1e-1
    x = cell_centers_1d(nz)
    p = linear_shocktube(x)
    t = 0.0
    n = 0
    p = xp.array(p)

    solver = FluxPerZoneSolver()
    p2c = PrimitiveToConserved(dim=1)

    if data_layout == "fields-last":
        prim_to_cons = p2c.prim_to_cons_fields_last
        cons_to_prim = p2c.cons_to_prim_fields_last
        update_cons = solver.update_cons_fields_last
        transpose = False

    if data_layout == "fields-first":
        prim_to_cons = p2c.prim_to_cons_fields_first
        cons_to_prim = p2c.cons_to_prim_fields_first
        update_cons = solver.update_cons_fields_first
        transpose = True
        p = xp.ascontiguousarray(p.T)

    u1 = xp.zeros_like(p)
    u2 = xp.zeros_like(p)
    prim_to_cons(p, u1)
    prim_to_cons(p, u2)

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
        if not rks:
            update_cons(u1, u2, dt, dx)
            u1, u2 = u2, u1
        else:
            u0 = u1.copy()
            for rk in rks:
                update_cons(u1, u2, dt, dx)
                if rk == 0.0:
                    u1, u2 = u2, u1
                else:
                    u1 = u2 * (1.0 - rk) + u0 * rk
        t += dt
        n += 1
        yield State(n, t, u1, cons_to_prim, transpose=transpose)


def make_solver(app: Sailfish):
    return simulation(
        app.hardware,
        app.domain.num_zones[0],
        data_layout=app.strategy.data_layout,
        time_integration=app.scheme.time_integration,
    )
