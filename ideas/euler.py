"""
A stand-alone, first-order 1d euler code

This code demonstrates basic usage of the sailfish kernels written in C/CUDA.
Kernel C code is embedded in this file and, and driven by higher-level
numpy/cupy functions.

Author: Jonathan Zrake
"""

from contextlib import contextmanager
from time import perf_counter
from numpy.typing import NDArray
from new_kernels import kernel, kernel_class

static = R"""
#define min2(a, b) ((a) < (b) ? (a) : (b))
#define max2(a, b) ((a) > (b) ? (a) : (b))
#define min3(a, b, c) min2(a, min2(b, c))
#define max3(a, b, c) max2(a, max2(b, c))

#define NCONS 3
#define RHO 0
#define VXX 1
#define PRE 2
#define DEN 0
#define PXX 1
#define NRG 2

static const double gamma_law_index = 5.0 / 3.0;

DEVICE void _prim_to_cons(double *p, double *u)
{
    double rho = p[RHO];
    double vx  = p[VXX];
    double pre = p[PRE];
    double v_squared = vx * vx;
    u[DEN] = rho;
    u[PXX] = vx * rho;
    u[NRG] = 0.5 * rho * v_squared + pre / (gamma_law_index - 1.0);
}

DEVICE void _cons_to_prim(double *u, double *p)
{
    double rho = u[DEN];
    double px  = u[PXX];
    double nrg = u[NRG];
    double p_squared = px * px;
    p[RHO] = rho;
    p[VXX] = px / rho;
    p[PRE] = (nrg - 0.5 * p_squared / rho) * (gamma_law_index - 1.0);
}

DEVICE void _prim_to_flux(double *p, double *u, double *f, int direction)
{
    double pre = p[PRE];
    double nrg = u[NRG];
    double vn = p[direction];
    f[DEN] = vn * u[DEN];
    f[PXX] = vn * u[PXX] + pre * (direction == 1);
    f[NRG] = vn * (nrg + pre);
}

DEVICE double _sound_speed_squared(double *p)
{
    return p[PRE] / p[RHO] * gamma_law_index;
}

DEVICE double _max_wavespeed(double *p)
{
    double cs = sqrt(_sound_speed_squared(p));
    double vx = p[VXX];
    double ax = max2(fabs(vx - cs), fabs(vx + cs));
    return ax;
}

DEVICE void _outer_wavespeeds(
    double *p,
    double *wavespeeds,
    int direction)
{
    double cs = sqrt(_sound_speed_squared(p));
    double vn = p[direction];
    wavespeeds[0] = vn - cs;
    wavespeeds[1] = vn + cs;
}

DEVICE void _hlle(double *pl, double *pr, double *flux, int direction)
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


@kernel_class
class Hydro:

    static = static

    @kernel()
    def cons_to_prim(self, u: NDArray[float], p: NDArray[float], ni: int = None):
        R"""
        KERNEL void cons_to_prim(double *u, double *p, int ni)
        {
            FOR_EACH_1D(ni)
            {
                _cons_to_prim(&u[NCONS * i], &p[NCONS * i]);
            }
        }
        """
        return u.shape[0], (u, p, u.shape[0])

    @kernel()
    def prim_to_cons(self, p: NDArray[float], u: NDArray[float], ni: int = None):
        R"""
        KERNEL void prim_to_cons(double *p, double *u, int ni)
        {
            FOR_EACH_1D(ni)
            {
                _prim_to_cons(&p[NCONS * i], &u[NCONS * i]);
            }
        }
        """
        return p.shape[0], (p, u, p.shape[0])

    @kernel()
    def prim_to_flux(
        self, p: NDArray[float], f: NDArray[float], direction: int, ni: int = None
    ):
        R"""
        KERNEL void prim_to_flux(double *p, double *f, int direction, int ni)
        {
            double u[NCONS];

            FOR_EACH_1D(ni)
            {
                _prim_to_cons(&p[i * NCONS], u);
                _prim_to_flux(&p[i * NCONS], u, &f[i * NCONS], direction);
            }
        }
        """
        return p.shape[0], (p, f, direction, p.shape[0])

    @kernel()
    def max_wavespeed(self, p: NDArray[float], a: NDArray[float], ni: int = None):
        R"""
        KERNEL void max_wavespeed(double *p, double *a, int ni)
        {
            FOR_EACH_1D(ni)
            {
                a[i] = _max_wavespeed(&p[i * NCONS]);
            }
        }
        """
        return p.shape[0], (p, a, p.shape[0])

    @kernel()
    def godunov_flux(
        self,
        pl: NDArray[float],
        pr: NDArray[float],
        fhat: NDArray[float],
        direction: int,
        ni: int = None,
    ):
        R"""
        KERNEL void godunov_flux(double *pl, double *pr, double *fhat, int direction, int ni)
        {
            FOR_EACH_1D(ni)
            {
                _hlle(&pl[NCONS * i], &pr[NCONS * i], &fhat[NCONS * i], direction);
            }
        }
        """
        return fhat.shape[0], (pl, pr, fhat, direction, fhat.shape[0])


@contextmanager
def measure_time() -> float:
    start = perf_counter()
    yield lambda: perf_counter() - start


def main():
    from sys import stdout
    from argparse import ArgumentParser
    from loguru import logger
    from new_kernels import configure_kernel_module

    parser = ArgumentParser()
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
        "--plot",
        action="store_true",
        help="show a plot after the run",
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["trace", "debug", "info", "success", "warning", "error", "critical"],
        help="log messages at and above this severity level",
    )
    args = parser.parse_args()
    logger.remove()
    logger.add(stdout, level=args.log_level.upper())
    configure_kernel_module(default_exec_mode=args.exec_mode)

    args = parser.parse_args()

    if args.exec_mode == "cpu":
        from numpy import array, linspace, zeros, zeros_like, diff
    if args.exec_mode == "gpu":
        from cupy import array, linspace, zeros, zeros_like, diff

    hydro = Hydro()
    num_zones = args.resolution
    dx = 1.0 / num_zones
    fold = 100
    dt = dx * 1e-1
    p = zeros((num_zones, 3))
    pp = zeros((num_zones, 3))
    pm = zeros((num_zones, 3))
    u = zeros_like(p)
    g = zeros_like(p)
    fhat = zeros((num_zones - 1, 3))

    p[: num_zones // 2, :] = array([1.0, 0.0, 1.0])
    p[num_zones // 2 :, :] = array([0.1, 0.0, 0.125])
    t = 0.0
    n = 0

    while t < 0.1:
        with measure_time() as fold_time:
            for _ in range(fold):
                pl = p[:-1]
                pr = p[+1:]
                hydro.godunov_flux(pl, pr, fhat, 1)
                hydro.prim_to_cons(p, u)
                u[1:-1] -= diff(fhat, axis=0) * (dt / dx)
                t += dt
                n += 1
                hydro.cons_to_prim(u, p)

        kzps = num_zones / fold_time() * 1e-3 * fold
        print(f"[{n:04d}]: t={t:.4f} Mzps={kzps * 1e-3:.3f}")

    if args.plot:
        from matplotlib import pyplot as plt

        plt.plot(p[:, 0])
        plt.show()


if __name__ == "__main__":
    main()
