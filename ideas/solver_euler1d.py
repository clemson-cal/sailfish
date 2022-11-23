from loguru import logger
from numpy import linspace, meshgrid, zeros, logical_not
from numpy.typing import NDArray
from new_kernels import kernel, perf_time_sequence, configure_kernel_module
from lib_euler import prim_to_cons, cons_to_prim, riemann_hlle


@kernel(
    device_funcs=[cons_to_prim],
    define_macros=dict(DIM=1),
    code=R"""
    KERNEL void cons_to_prim_array(double *u, double *p, int ni)
    {
        FOR_EACH_1D(ni)
        {
            cons_to_prim(&u[NCONS * i], &p[NCONS * i]);
        }
    }
    """,
)
def cons_to_prim_array(u: NDArray[float], p: NDArray[float], ni: int = None):
    """
    Convert an array of conserved data to an array of primitive data
    """
    return u.size // 3, (u, p, u.size // 3)


@kernel(
    device_funcs=[prim_to_cons],
    define_macros=dict(DIM=1),
    code=R"""
    KERNEL void prim_to_cons_array(double *p, double *u, int ni)
    {
        FOR_EACH_1D(ni)
        {
            prim_to_cons(&p[NCONS * i], &u[NCONS * i]);
        }
    }
    """,
)
def prim_to_cons_array(p: NDArray[float], u: NDArray[float], ni: int = None):
    """
    Convert an array of primitive data to an array of conserved data
    """
    return p.size // 3, (p, u, p.size // 3)


@kernel(
    device_funcs=[
        prim_to_cons,
        cons_to_prim,
        riemann_hlle,
    ],
    define_macros=dict(DIM=1),
    code=R"""
    KERNEL void update_prim_rk1_pcm(double *p, double dt, double dx, int ni)
    {
        double uc[NCONS];
        double fhat_m[NCONS];
        double fhat_p[NCONS];

        FOR_RANGE_1D(1, ni - 1)
        {
            double *pc = &p[NCONS * i];
            double *pl = &p[NCONS * (i - 1)];
            double *pr = &p[NCONS * (i + 1)];

            prim_to_cons(pc, uc);
            riemann_hlle(pl, pc, fhat_m, 1);
            riemann_hlle(pc, pr, fhat_p, 1);

            for (int q = 0; q < NCONS; ++q)
            {
                uc[q] -= (fhat_p[q] - fhat_m[q]) * dt / dx;
            }
            cons_to_prim(uc, pc);
        }
    }
    """,
)
def update_prim_rk1_pcm(p: NDArray[float], dt: float, dx: float, ni: int = None):
    """
    A single-step first-order update using flux-per-zone.

    The first and final elements of the primitive array are not modified.
    """
    return p.shape[0], (p, dt, dx, p.shape[0])


@kernel(
    device_funcs=[riemann_hlle],
    define_macros=dict(DIM=1),
    code=R"""
    KERNEL void compute_godunov_fluxes_pcm(double *p, double *f, int ni)
    {
        FOR_RANGE_1D(0, ni - 1)
        {
            double *pc = &p[NCONS * i];
            double *pr = &p[NCONS * (i + 1)];
            double *fp = &f[NCONS * (i + 1)];
            riemann_hlle(pc, pr, fp, 1);
        }
    }
    """,
)
def compute_godunov_fluxes_pcm(p: NDArray[float], f: NDArray[float], ni: int = None):
    """
    Compute an array of Godunov fluxes using HLLE Riemann solver.

    The first and final elements of the flux array are not modified.
    """
    return p.shape[0], (p, f, p.shape[0])


class Scratch:
    def __init__(self, ni, xp):
        self.ni = ni
        self.u = xp.zeros([ni, 3])
        self.f = xp.zeros([ni, 3])


def update_prim(p, dt, dx, xp, scratch):
    """
    Drives a first-order, flux-per-face update of a primitive array
    """
    ni = p.shape[0]

    if scratch is None or scratch.ni != ni:
        scratch = Scratch(ni, xp)

    f = scratch.f
    u = scratch.u
    prim_to_cons_array(p, u)
    compute_godunov_fluxes_pcm(p, f)
    u[1:-1] -= xp.diff(f[1:], axis=0) * (dt / dx)
    cons_to_prim_array(u, p)

    return scratch


def cell_centers_1d(ni):
    from numpy import linspace

    xv = linspace(0.0, 1.0, ni)
    xc = 0.5 * (xv[1:] + xv[:-1])
    return xc


def linear_shocktube(x):
    """
    A linear shocktube setup
    """

    from numpy import array, zeros, logical_not

    l = x < 0.5
    r = logical_not(l)
    p = zeros(x.shape + (3,))
    p[l, :] = [1.0, 0.0, 1.000]
    p[r, :] = [0.1, 0.0, 0.125]
    return p


@logger.catch
def main():
    from argparse import ArgumentParser
    from reporting import (
        configure_logger,
        terminal,
        add_logging_arguments,
        iteration_msg,
    )

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
        default=None,
        help="grid resolution",
    )
    parser.add_argument(
        "--fold",
        "-f",
        type=int,
        default=50,
        help="number of iterations between messages",
    )
    parser.add_argument(
        "--patches-per-dim",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--plm",
        action="store_true",
        help="use PLM reconstruction for second-order in space",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--flux-correction",
        action="store_true",
        help="include the flux correction step for FMR",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="show a plot after the run",
    )
    add_logging_arguments(parser)
    args = parser.parse_args()

    configure_logger(logger, log_level=args.log_level)
    term = terminal(logger)

    configure_kernel_module(default_exec_mode=args.exec_mode)

    nz = args.resolution or 100000
    dx = 1.0 / nz
    dt = dx * 1e-1
    x = cell_centers_1d(nz)
    p = linear_shocktube(x)
    t = 0.0
    n = 0

    if args.exec_mode == "gpu":
        import cupy as xp

        to_host = lambda a: a.get()
    else:
        import numpy as xp

        to_host = lambda a: a

    p = xp.array(p)
    perf_timer = perf_time_sequence(mode=args.exec_mode)

    logger.info("start simulation")
    scratch = None

    while t < 0.1:
        scratch = update_prim(p, dt, dx, xp, scratch)
        t += dt
        n += 1

        if n % args.fold == 0:
            zps = nz / next(perf_timer) * args.fold
            term(iteration_msg(iter=n, time=t, zps=zps))

    p = to_host(p)
    from numpy import save

    save("prim.npy", p)

    if args.plot:
        from matplotlib import pyplot as plt

        plt.plot(p[:, 0])
        plt.show()


if __name__ == "__main__":
    main()
