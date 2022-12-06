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
class FluxPerZoneTransposedSolver:
    @property
    def define_macros(self):
        return dict(DIM=1)

    @property
    def device_funcs(self):
        return [cons_to_prim, riemann_hlle]

    @kernel
    def update_cons(
        self,
        urd: NDArray[float],
        uwr: NDArray[float],
        dt: float,
        dx: float,
        ni: int = None,
    ):
        R"""
        KERNEL void update_cons(double *urd, double *uwr, double dt, double dx, int ni)
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
                double *u_blk[NCONS];

                double p_reg[3][NCONS];
                double u_reg[3][NCONS];
                double fp[NCONS];
                double fm[NCONS];

                for (int q = 0; q < NCONS; ++q)
                {
                    u_blk[q] = &urd[q * ni + blockIdx_x * blockDim_x];
                }

                for (int q = 0; q < NCONS; ++q)
                {
                    u_reg[0][q] = u_blk[q][threadIdx_x - 1];
                    u_reg[1][q] = u_blk[q][threadIdx_x + 0];
                    u_reg[2][q] = u_blk[q][threadIdx_x + 1];
                }

                cons_to_prim(u_reg[0], p_reg[0]);
                cons_to_prim(u_reg[1], p_reg[1]);
                cons_to_prim(u_reg[2], p_reg[2]);
                riemann_hlle(p_reg[0], p_reg[1], fm, 1);
                riemann_hlle(p_reg[1], p_reg[2], fp, 1);

                for (int q = 0; q < NCONS; ++q)
                {
                    u_reg[1][q] -= (fp[q] - fm[q]) * dt / dx;
                }

                for (int q = 0; q < NCONS; ++q)
                {
                    u_blk[q] = &uwr[q * ni + blockIdx_x * blockDim_x];
                }

                for (int q = 0; q < NCONS; ++q)
                {
                    u_blk[q][threadIdx_x] = u_reg[1][q];
                }
            }
        }
        """
        return urd.shape[1], (urd, uwr, dt, dx, urd.shape[1])


class State:
    def __init__(self, n, t, u, p2c):
        self._n = n
        self._t = t
        self._u = u
        self._p2c = p2c

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
        self._p2c.cons_to_prim_fields_first(u, p)
        try:
            return p.T.get()
        except AttributeError:
            return p.T

    @property
    def total_zones(self):
        return self._u.shape[1]


def simulation(hardware: str, resolution: int) -> State:
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
    u = xp.zeros_like(p)
    t = 0.0
    n = 0
    p = xp.array(p)

    solver = FluxPerZoneTransposedSolver()
    p2c = PrimitiveToConserved(dim=1)
    p2c.prim_to_cons_fields_last(p, u)
    u1 = xp.ascontiguousarray(u.T)
    u2 = u1.copy()

    yield State(n, t, u1, p2c)

    while True:
        solver.update_cons(u1, u2, dt, dx)
        u1, u2 = u2, u1
        t += dt
        n += 1
        yield State(n, t, u1, p2c)


def make_solver(app: Sailfish):
    return simulation(app.hardware, app.domain.num_zones[0])
