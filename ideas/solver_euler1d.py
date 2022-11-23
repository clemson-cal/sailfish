from loguru import logger
from numpy import linspace, meshgrid, zeros, logical_not
from numpy.typing import NDArray
from new_kernels import kernel, perf_time_sequence, configure_kernel_module
from lib_euler import prim_to_cons, cons_to_prim, riemann
from configuration import configurable, all_schemas


@kernel(device_funcs=[cons_to_prim], define_macros=dict(DIM=1))
def cons_to_prim_array(u: NDArray[float], p: NDArray[float], ni: int = None):
    R"""
    //
    // Convert an array of conserved data to an array of primitive data
    //
    KERNEL void cons_to_prim_array(double *u, double *p, int ni)
    {
        FOR_EACH_1D(ni)
        {
            cons_to_prim(&u[NCONS * i], &p[NCONS * i]);
        }
    }
    """
    return u.size // 3, (u, p, u.size // 3)


@kernel(device_funcs=[prim_to_cons], define_macros=dict(DIM=1))
def prim_to_cons_array(p: NDArray[float], u: NDArray[float], ni: int = None):
    R"""
    //
    // Convert an array of primitive data to an array of conserved data
    //
    KERNEL void prim_to_cons_array(double *p, double *u, int ni)
    {
        FOR_EACH_1D(ni)
        {
            prim_to_cons(&p[NCONS * i], &u[NCONS * i]);
        }
    }
    """
    return p.size // 3, (p, u, p.size // 3)


@kernel()
def conservative_update(
    u: NDArray[float],
    f: NDArray[float],
    dt: float,
    dx: float,
    ni: int = None,
):
    R"""
    //
    // Conservative difference an array of fluxes to update an array of conserved
    // densities.
    //
    KERNEL void conservative_update(
        double *u,
        double *f,
        double dt,
        double dx,
        int ni)
    {
        FOR_RANGE_1D(1, ni - 1)
        {
            double *uc = &u[3 * i];
            double *fm = &f[3 * i];
            double *fp = &f[3 * (i + 1)];

            for (int q = 0; q < 3; ++q)
            {
                uc[q] -= (fp[q] - fm[q]) * dt / dx;
            }
        }
    }
    """
    return u.size // 3, (u, f, dt, dx, u.size // 3)


@kernel(device_funcs=[prim_to_cons, cons_to_prim, riemann], define_macros=dict(DIM=1))
def update_prim_rk1_pcm(p: NDArray[float], dt: float, dx: float, ni: int = None):
    R"""
    //
    // A single-step first-order update using flux-per-zone.
    //
    // The first and final elements of the primitive array are not modified.
    //
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
            riemann(pl, pc, fhat_m, 1);
            riemann(pc, pr, fhat_p, 1);

            for (int q = 0; q < NCONS; ++q)
            {
                uc[q] -= (fhat_p[q] - fhat_m[q]) * dt / dx;
            }
            cons_to_prim(uc, pc);
        }
    }
    """
    return p.shape[0], (p, dt, dx, p.shape[0])


@kernel(device_funcs=[riemann], define_macros=dict(DIM=1))
def compute_godunov_fluxes_pcm(p: NDArray[float], f: NDArray[float], ni: int = None):
    R"""
    //
    // Compute an array of Godunov fluxes using HLLE Riemann solver.
    //
    // The first and final elements of the flux array are not modified.
    //
    KERNEL void compute_godunov_fluxes_pcm(double *p, double *f, int ni)
    {
        FOR_RANGE_1D(0, ni - 1)
        {
            double *pc = &p[NCONS * i];
            double *pr = &p[NCONS * (i + 1)];
            double *fp = &f[NCONS * (i + 1)];
            riemann(pc, pr, fp, 1);
        }
    }
    """
    return p.shape[0], (p, f, p.shape[0])


def update_prim(
    p,
    dt,
    dx,
    strategy="flux_per_zone",
    xp=None,
):
    """
    Drives a first-order update of a primitive array
    """

    if strategy == "flux_per_face":
        f = xp.empty_like(p)
        u = xp.empty_like(p)

        prim_to_cons_array(p, u)
        compute_godunov_fluxes_pcm(p, f)
        conservative_update(u, f, dt, dx)
        cons_to_prim_array(u, p)
        return

    if strategy == "flux_per_zone":
        update_prim_rk1_pcm(p, dt, dx)
        return

    raise ValueError(f"unknown strategy {strategy}")


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


def numpy_or_cupy(mode):
    if mode == "gpu":
        import cupy

        return cupy, lambda a: a.get()

    if mode == "cpu":
        import numpy

        return numpy, lambda a: a


@logger.catch
@configurable
def driver(
    exec_mode: str = "cpu",
    resolution: int = 10000,
    strategy: str = "flux_per_zone",
    fold: int = 100,
    plot: bool = False,
):
    """
    Configuration
    -------------

    exec_mode:     execution mode [cpu|gpu]
    resolution:    number of grid zones
    strategy:      solver strategy [flux_per_zone|flux_per_face]
    fold:          number of iterations between iteration message
    plot:          whether to show a plot of the solution
    """
    from reporting import terminal, iteration_msg

    configure_kernel_module(default_exec_mode=exec_mode)
    term = terminal(logger)
    xp, to_host = numpy_or_cupy(exec_mode)

    nz = resolution or 100000
    dx = 1.0 / nz
    dt = dx * 1e-1
    x = cell_centers_1d(nz)
    p = linear_shocktube(x)
    t = 0.0
    n = 0

    p = xp.array(p)
    perf_timer = perf_time_sequence(mode=exec_mode)

    logger.info("start simulation")

    while t < 0.1:
        update_prim(p, dt, dx, strategy, xp)
        t += dt
        n += 1

        if n % fold == 0:
            zps = nz / next(perf_timer) * fold
            term(iteration_msg(iter=n, time=t, zps=zps))

    p = to_host(p)

    if plot:
        from matplotlib import pyplot as plt

        plt.plot(p[:, 0])
        plt.show()


if __name__ == "__main__":
    from argparse import ArgumentParser
    from reporting import add_logging_arguments, terminal, configure_logger

    parser = ArgumentParser()
    add_logging_arguments(parser)

    driver_group = parser.add_argument_group("driver")
    driver.schema.argument_parser(driver_group)
    args = parser.parse_args()

    config = {
        g.title: {a.dest: getattr(args, a.dest, None) for a in g._group_actions}
        for g in parser._action_groups
    }
    configure_logger(logger, log_level=args.log_level)

    driver.schema.print_schema(terminal(logger), config=vars(args), newline=True)
    driver(**config["driver"])
