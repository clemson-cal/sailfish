from contextlib import contextmanager
from time import perf_counter
from numpy.typing import NDArray
from new_kernels import kernel, kernel_class, kernel_method, configure_kernel_module


plm_gradient_code = R"""
#define min2(a, b) ((a) < (b) ? (a) : (b))
#define max2(a, b) ((a) > (b) ? (a) : (b))
#define min3(a, b, c) min2(a, min2(b, c))
#define max3(a, b, c) max2(a, max2(b, c))
#define sign(x) copysign(1.0, x)
#define minabs(a, b, c) min3(fabs(a), fabs(b), fabs(c))

PRIVATE double plm_gradient_scalar(double yl, double y0, double yr, double plm_theta)
{
    double a = (y0 - yl) * plm_theta;
    double b = (yr - yl) * 0.5;
    double c = (yr - y0) * plm_theta;
    return 0.25 * fabs(sign(a) + sign(b)) * (sign(a) + sign(c)) * minabs(a, b, c);
}

PUBLIC void plm_gradient(int ni, int nfields, double *y, double *g, double plm_theta)
{
    FOR_EACH_1D(ni)
    {
        int ii = i;
        int il = i ==      0 ?      0 : i - 1;
        int ir = i == ni - 1 ? ni - 1 : i + 1;

        double *yl = &y[il * nfields];
        double *yi = &y[ii * nfields];
        double *yr = &y[ir * nfields];
        double *gi = &g[ii * nfields];

        for (int q = 0; q < nfields; ++q)
        {
            gi[q] = plm_gradient_scalar(yl[q], yi[q], yr[q], plm_theta);
        }
    }
}
"""


@kernel(plm_gradient_code, rank=1, prepend_signature=(int, int))
def plm_gradient(
    y: NDArray[float],
    g: NDArray[float],
    plm_theta: float,
):
    if y.shape != g.shape:
        raise ValueError("y and g must have the same shape")
    if not 1.0 <= plm_theta <= 2.0:
        raise ValueError("theta value must be between 1.0 and 2.0")
    return y.shape


extrapolate_code = R"""
PUBLIC void extrapolate(int ni, int nfields, double *y, double *g, double *ym, double *yp)
{
    FOR_EACH_1D(ni)
    {
        for (int q = 0; q < nfields; ++q)
        {
            ym[i * nfields + q] = y[i * nfields + q] - 0.5 * g[i * nfields + q];
            yp[i * nfields + q] = y[i * nfields + q] + 0.5 * g[i * nfields + q];
        }
    }
}
"""


@kernel(extrapolate_code, rank=1, prepend_signature=(int, int))
def extrapolate(
    y: NDArray[float],
    g: NDArray[float],
    ym: NDArray[float],
    yp: NDArray[float],
):
    if not all(y.shape == s for s in (g.shape, ym.shape, yp.shape)):
        raise ValueError("arguments must have the same shape")
    return y.shape


solver_code = R"""
#define min2(a, b) ((a) < (b) ? (a) : (b))
#define max2(a, b) ((a) > (b) ? (a) : (b))
#define min3(a, b, c) min2(a, min2(b, c))
#define max3(a, b, c) max2(a, max2(b, c))

#ifndef DIM
#define DIM 1
#endif

#if DIM == 1
#define NCONS 3
#define RHO 0
#define VXX 1
#define PRE 2
#define DEN 0
#define PXX 1
#define NRG 2
#elif DIM == 2
#define NCONS 4
#define RHO 0
#define VXX 1
#define VYY 2
#define PRE 3
#define DEN 0
#define PXX 1
#define PYY 2
#define NRG 3
#elif DIM == 3
#define NCONS 5
#define RHO 0
#define VXX 1
#define VYY 2
#define VZZ 3
#define PRE 4
#define DEN 0
#define PXX 1
#define PYY 2
#define PZZ 3
#define NRG 4
#endif

#ifndef GAMMA_LAW_INDEX
#define GAMMA_LAW_INDEX (5.0 / 3.0)
#endif

static const double gamma_law_index = GAMMA_LAW_INDEX;

PRIVATE void _prim_to_cons(double *p, double *u)
{
    #if DIM == 1
    double rho = p[RHO];
    double vx  = p[VXX];
    double pre = p[PRE];
    double v_squared = vx * vx;
    u[DEN] = rho;
    u[PXX] = vx * rho;
    u[NRG] = 0.5 * rho * v_squared + pre / (gamma_law_index - 1.0);

    #elif DIM == 2
    double rho = p[RHO];
    double vx  = p[VXX];
    double vy  = p[VYY];
    double pre = p[PRE];
    double v_squared = vx * vx + vy * vy;
    u[DEN] = rho;
    u[PXX] = vx * rho;
    u[PYY] = vy * rho;
    u[NRG] = 0.5 * rho * v_squared + pre / (gamma_law_index - 1.0);

    #elif DIM == 3
    double rho = p[RHO];
    double vx  = p[VXX];
    double vy  = p[VYY];
    double vz  = p[VZZ];
    double pre = p[PRE];
    double v_squared = vx * vx + vy * vy + vz * vz;
    u[DEN] = rho;
    u[PXX] = vx * rho;
    u[PYY] = vy * rho;
    u[PZZ] = vz * rho;
    u[NRG] = 0.5 * rho * v_squared + pre / (gamma_law_index - 1.0);
    #endif
}

PRIVATE void _cons_to_prim(double *u, double *p)
{
    #if DIM == 1
    double rho = u[DEN];
    double px  = u[PXX];
    double nrg = u[NRG];
    double p_squared = px * px;
    p[RHO] = rho;
    p[VXX] = px / rho;
    p[PRE] = (nrg - 0.5 * p_squared / rho) * (gamma_law_index - 1.0);

    #elif DIM == 2
    double rho = u[DEN];
    double px  = u[PXX];
    double py  = u[PYY];
    double nrg = u[NRG];
    double p_squared = px * px + py * py;
    p[RHO] = rho;
    p[VXX] = px / rho;
    p[VYY] = py / rho;
    p[PRE] = (nrg - 0.5 * p_squared / rho) * (gamma_law_index - 1.0);

    #elif DIM == 3
    double rho = u[DEN];
    double px  = u[PXX];
    double py  = u[PYY];
    double pz  = u[PZZ];
    double nrg = u[NRG];
    double p_squared = px * px + py * py + pz * pz;
    p[RHO] = rho;
    p[VXX] = px / rho;
    p[VYY] = py / rho;
    p[VZZ] = py / rho;
    p[PRE] = (nrg - 0.5 * p_squared / rho) * (gamma_law_index - 1.0);
    #endif
}

PRIVATE void _prim_to_flux(double *p, double *u, double *f, int direction)
{
    double pre = p[PRE];
    double nrg = u[NRG];
    double vn = p[direction];

    #if DIM == 1
    f[DEN] = vn * u[DEN];
    f[PXX] = vn * u[PXX] + pre * (direction == 1);
    f[NRG] = vn * (nrg + pre);

    #elif DIM == 2
    f[DEN] = vn * u[DEN];
    f[PXX] = vn * u[PXX] + pre * (direction == 1);
    f[PYY] = vn * u[PYY] + pre * (direction == 2);
    f[NRG] = vn * (nrg + pre);

    #elif DIM == 3
    f[DEN] = vn * u[DEN];
    f[PXX] = vn * u[PXX] + pre * (direction == 1);
    f[PYY] = vn * u[PYY] + pre * (direction == 2);
    f[PZZ] = vn * u[PZZ] + pre * (direction == 3);
    f[NRG] = vn * (nrg + pre);
    #endif
}

PRIVATE double _sound_speed_squared(double *p)
{
    return p[PRE] / p[RHO] * gamma_law_index;
}

PRIVATE double _max_wavespeed(double *p)
{
    #if DIM == 1
    double cs = sqrt(_sound_speed_squared(p));
    double vx = p[VXX];
    double ax = max2(fabs(vx - cs), fabs(vx + cs));
    return ax;

    #elif DIM == 2
    double cs = sqrt(_sound_speed_squared(p));
    double vx = p[VXX];
    double vy = p[VYY];
    double ax = max2(fabs(vx - cs), fabs(vx + cs));
    double ay = max2(fabs(vy - cs), fabs(vy + cs));
    return max2(ax, ay);

    #elif DIM == 3
    double cs = sqrt(_sound_speed_squared(p));
    double vx = p[VXX];
    double vy = p[VYY];
    double vz = p[VZZ];
    double ax = max2(fabs(vx - cs), fabs(vx + cs));
    double ay = max2(fabs(vy - cs), fabs(vy + cs));
    double az = max2(fabs(vz - cs), fabs(vz + cs));
    return max3(ax, ay, az);
    #endif
}

PRIVATE void _outer_wavespeeds(
    double *p,
    double *wavespeeds,
    int direction)
{
    double cs = sqrt(_sound_speed_squared(p));
    double vn = p[direction];
    wavespeeds[0] = vn - cs;
    wavespeeds[1] = vn + cs;
}

PRIVATE void _hlle(double *pl, double *pr, double *flux, int direction)
{
    double ul[NCONS];
    double ur[NCONS];
    double fl[NCONS];
    double fr[NCONS];
    double al[2];
    double ar[2];

    _prim_to_cons(pl, ul);
    _prim_to_cons(pr, ur);
    _prim_to_flux(pl, ul, fl, direction);
    _prim_to_flux(pr, ur, fr, direction);
    _outer_wavespeeds(pl, al, direction);
    _outer_wavespeeds(pr, ar, direction);

    double am = min3(0.0, al[0], ar[0]);
    double ap = max3(0.0, al[1], ar[1]);

    for (int q = 0; q < NCONS; ++q)
    {
        flux[q] = (fl[q] * ap - fr[q] * am - (ul[q] - ur[q]) * ap * am) / (ap - am);
    }
}
"""


@kernel_class(solver_code)
class Solver:
    @kernel_method(rank=1)
    def cons_to_prim(self, u: NDArray[float], p: NDArray[float]):
        R"""
        PUBLIC void cons_to_prim(int ni, double *u, double *p)
        {
            FOR_EACH_1D(ni)
            {
                _cons_to_prim(&u[NCONS * i], &p[NCONS * i]);
            }
        }
        """
        return (u.size // self.ncons,)

    @kernel_method(rank=1)
    def prim_to_cons(self, p: NDArray[float], u: NDArray[float]):
        R"""
        PUBLIC void prim_to_cons(int ni, double *p, double *u)
        {
            FOR_EACH_1D(ni)
            {
                _prim_to_cons(&p[NCONS * i], &u[NCONS * i]);
            }
        }
        """
        return (p.size // self.ncons,)

    @kernel_method(rank=1)
    def prim_to_flux(self, p: NDArray[float], f: NDArray[float], direction: int):
        R"""
        PUBLIC void prim_to_flux(int ni, double *p, double *f, int direction)
        {
            double u[NCONS];

            FOR_EACH_1D(ni)
            {
                _prim_to_cons(&p[i * NCONS], u);
                _prim_to_flux(&p[i * NCONS], u, &f[i * NCONS], direction);
            }
        }
        """
        return (p.size // self.ncons,)

    @kernel_method(rank=1)
    def max_wavespeed(self, p: NDArray[float], a: NDArray[float]):
        R"""
        PUBLIC void max_wavespeed(int ni, double *p, double *a)
        {
            FOR_EACH_1D(ni)
            {
                a[i] = _max_wavespeed(&p[i * NCONS]);
            }
        }
        """
        return (p.size // self.ncons,)

    @kernel_method(rank=1)
    def godunov_flux(
        self,
        pl: NDArray[float],
        pr: NDArray[float],
        fhat: NDArray[float],
        direction: int,
    ):
        R"""
        PUBLIC void godunov_flux(int ni, double *pl, double *pr, double *fhat, int direction)
        {
            FOR_EACH_1D(ni)
            {
                _hlle(&pl[NCONS * i], &pr[NCONS * i], &fhat[NCONS * i], direction);
            }
        }
        """
        return (fhat.size // self.ncons,)

    def __init__(self, dim=1, gamma_law_index=5.0 / 3):
        self.ncons = dim + 2
        self.dim = dim
        self.gamma_law_index = gamma_law_index
        self.define_macros = [("GAMMA_LAW_INDEX", gamma_law_index), ("DIM", dim)]


@contextmanager
def measure_time() -> float:
    start = perf_counter()
    yield lambda: perf_counter() - start


def main():
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="verbose output from extension compile stages",
    )
    parser.add_argument(
        "--mode",
        dest="exec_mode",
        default="cpu",
        choices=["cpu", "gpu"],
        help="execution mode",
    )
    parser.add_argument(
        "--resolution",
        "-n",
        metavar="N",
        type=int,
        default=100000,
        help="grid resolution",
    )
    parser.add_argument(
        "--plm",
        action="store_true",
        help="use PLM reconstruction for second-order in space",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="show a plot after the run",
    )
    args = parser.parse_args()

    if args.exec_mode == "cpu":
        from numpy import array, linspace, zeros, zeros_like, diff
    if args.exec_mode == "gpu":
        from cupy import array, linspace, zeros, zeros_like, diff

    configure_kernel_module(verbose=False, default_exec_mode=args.exec_mode)
    solver = Solver(dim=1, gamma_law_index=5.0 / 3.0)

    num_zones = args.resolution
    dx = 1.0 / num_zones
    fold = 100
    dt = dx * 1e-1
    p = zeros((num_zones, solver.ncons))
    pp = zeros((num_zones, solver.ncons))
    pm = zeros((num_zones, solver.ncons))
    u = zeros_like(p)
    g = zeros_like(p)
    fhat = zeros((num_zones - 1, solver.ncons))

    p[: num_zones // 2, :] = array([1.0] + solver.dim * [0.0] + [1.0])
    p[num_zones // 2 :, :] = array([0.1] + solver.dim * [0.0] + [0.125])
    t = 0.0
    n = 0

    while t < 0.1:
        with measure_time() as fold_time:
            for _ in range(fold):
                if args.plm:
                    plm_gradient(p, g, 1.5)
                    extrapolate(p, g, pm, pp)
                    pl = pp[:-1]
                    pr = pm[+1:]
                else:
                    pl = p[:-1]
                    pr = p[+1:]

                solver.godunov_flux(pl, pr, fhat, 1)
                solver.prim_to_cons(p, u)
                u[1:-1] -= diff(fhat, axis=0) * (dt / dx)
                t += dt
                n += 1
                solver.cons_to_prim(u, p)

        kzps = num_zones / fold_time() * 1e-3 * fold
        print(f"[{n:04d}]: t={t:.4f} Mzps={kzps * 1e-3:.3f}")

    if args.plot:
        from matplotlib import pyplot as plt

        plt.plot(p[:, 0])
        plt.show()


main()
