from collections.abc import MutableMapping
from collections import ChainMap
from loguru import logger
from numpy import linspace, meshgrid, zeros, logical_not
from numpy.typing import NDArray
from new_kernels import kernel, perf_time_sequence, configure_kernel_module, device
from lib_euler import prim_to_cons, cons_to_prim, riemann
from configuration import configurable, all_schemas


@device()
def plm_minmod(yl: float, yc: float, yr: float, plm_theta: float):
    R"""
    DEVICE double plm_minmod(
        double yl,
        double yc,
        double yr,
        double plm_theta)
    {
        #define min2(a, b) ((a) < (b) ? (a) : (b))
        #define max2(a, b) ((a) > (b) ? (a) : (b))
        #define min3(a, b, c) min2(a, min2(b, c))
        #define max3(a, b, c) max2(a, max2(b, c))
        #define sign(x) copysign(1.0, x)
        #define minabs(a, b, c) min3(fabs(a), fabs(b), fabs(c))

        double a = (yc - yl) * plm_theta;
        double b = (yr - yl) * 0.5;
        double c = (yr - yc) * plm_theta;
        return 0.25 * fabs(sign(a) + sign(b)) * (sign(a) + sign(c)) * minabs(a, b, c);

        #undef min2
        #undef max2
        #undef min3
        #undef max3
        #undef sign
        #undef minabs
    }
    """


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
    // Does not modify the first-2 or final-2 zones in the conserved variable
    // array.
    //
    KERNEL void conservative_update(
        double *u,
        double *f,
        double dt,
        double dx,
        int ni)
    {
        FOR_RANGE_1D(2, ni - 2)
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


@kernel(device_funcs=[riemann, plm_minmod], define_macros=dict(DIM=1))
def compute_godunov_fluxes_plm(
    p: NDArray[float],
    f: NDArray[float],
    plm_theta: float,
    ni: int = None,
):
    R"""
    //
    // Compute an array of Godunov fluxes using HLLE Riemann solver and PLM
    // reconstruction.
    //
    // The first-2 and final-2 elements of the flux array are not modified.
    //
    KERNEL void compute_godunov_fluxes_plm(double *p, double *f, double plm_theta, int ni)
    {
        double pm[NCONS];
        double pp[NCONS];

        FOR_RANGE_1D(1, ni - 2)
        {
            double *pl = &p[NCONS * (i - 1)];
            double *pc = &p[NCONS * (i + 0)];
            double *pr = &p[NCONS * (i + 1)];
            double *ps = &p[NCONS * (i + 2)];
            double *fh = &f[NCONS * (i + 1)];

            for (int q = 0; q < NCONS; ++q)
            {
                pm[q] = pc[q] + 0.5 * plm_minmod(pl[q], pc[q], pr[q], plm_theta);
                pp[q] = pr[q] - 0.5 * plm_minmod(pc[q], pr[q], ps[q], plm_theta);
            }
            riemann(pm, pp, fh, 1);
        }
    }
    """
    return p.shape[0], (p, f, plm_theta, p.shape[0])


def update_prim(
    p,
    dt,
    dx,
    strategy="flux_per_zone",
    reconstruction="pcm",
    xp=None,
):
    """
    Drives a first-order update of a primitive array
    """

    if strategy == "flux_per_face":
        f = xp.empty_like(p)
        u = xp.empty_like(p)

        prim_to_cons_array(p, u)

        if reconstruction == "pcm":
            compute_godunov_fluxes_pcm(p, f)
        elif reconstruction == "plm":
            compute_godunov_fluxes_plm(p, f, 1.5)
        else:
            raise ValueError(f"reconstruction must be [pcm|plm], got {reconstruction}")

        conservative_update(u, f, dt, dx)
        cons_to_prim_array(u, p)
        return

    if strategy == "flux_per_zone":
        if reconstruction == "pcm":
            update_prim_rk1_pcm(p, dt, dx)
        elif reconstruction == "plm":
            raise NotImplementedError(f"{reconstruction}")
        else:
            raise ValueError(f"reconstruction must be [pcm|plm], got {reconstruction}")
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


@configurable
def driver(
    app_cfg,
    exec_mode: str = "cpu",
    resolution: int = 10000,
    strategy: str = "flux_per_zone",
    reconstruction: str = "pcm",
    fold: int = 100,
    plot: bool = False,
):
    """
    Configuration
    -------------

    exec_mode:      execution mode [cpu|gpu]
    resolution:     number of grid zones
    strategy:       solver strategy [flux_per_zone|flux_per_face]
    reconstruction: first or second-order reconstruction [pcm|plm]
    fold:           number of iterations between iteration message
    plot:           whether to show a plot of the solution
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
        update_prim(p, dt, dx, strategy, reconstruction, xp)
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


def flatten_dict(
    d: MutableMapping,
    parent_key: str = "",
    sep: str = ".",
) -> MutableMapping:
    """
    Create a flattened dictionary e from d, with e['a.b.c'] = d['a']['b']['c']
    """
    items = list()
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def dict_section(d: MutableMapping, section: str):
    """
    From a map with keys like section.b.c, return a dict with keys like b.c.
    """
    return {k[k.index(".") + 1 :]: v for k, v in d.items() if k.startswith(section)}


def load_config(config):
    """
    Attempt to load configuration data from a file: either JSON or YAML.
    """
    if config.endswith(".json"):
        from json import load

        with open(config, "r") as infile:
            return load(infile)

    elif config.endswith(".yaml"):
        from yaml import load, CLoader

        with open(config, "r") as infile:
            return load(infile, Loader=CLoader)

    else:
        raise ValueError(f"unknown configuration file {config}")


def short_help(args):
    args.parser.print_usage()


def run(args):
    app_cfg = ChainMap(
        {k: v for k, v in vars(args).items() if v is not None and "." in k}
    )
    app_cfg.maps.extend(flatten_dict(load_config(c)) for c in reversed(args.configs))
    driver_args = dict_section(app_cfg, "driver")
    driver.schema.validate(**driver_args)
    driver.schema.print_schema(
        args.term,
        config=driver_args,
        newline=True,
    )
    driver(app_cfg, **driver_args)


def show_config(args):
    if args.defaults:
        for schema in all_schemas():
            schema.print_schema(args.term)

    else:
        app_cfg = {s.component_name: s.defaults_dict() for s in all_schemas()}

        if args.format == "json":
            from json import dumps

            print(dumps(app_cfg, indent=4))

        if args.format == "yaml":
            try:
                from yaml import dump, CDumper

                print(dump(app_cfg, Dumper=CDumper))

            except ImportError as e:
                print(e)


@logger.catch
def main():
    from argparse import ArgumentParser
    from reporting import add_logging_arguments, terminal, configure_logger

    parser = ArgumentParser()
    parser.set_defaults(func=short_help)
    parser.set_defaults(term=terminal(logger))
    parser.set_defaults(parser=parser)
    parser.set_defaults(log_level="info")
    subparsers = parser.add_subparsers()

    show_config_parser = subparsers.add_parser(
        "show-config",
        help="show global configuration data",
    )
    show_config_parser.set_defaults(func=show_config)
    group = show_config_parser.add_mutually_exclusive_group()
    group.add_argument(
        "--format",
        type=str,
        default="json",
        choices=["json", "yaml"],
        help="output format for the configuration data",
    )
    group.add_argument(
        "--defaults",
        action="store_true",
        help="print defaults and help messages for configurable components",
    )
    run_parser = subparsers.add_parser(
        "run",
        help="run a simulation",
    )
    run_parser.set_defaults(func=run)
    run_parser.add_argument(
        "configs",
        nargs="*",
        help="sequence of presets, configuration files, or checkpoints",
    )
    driver.schema.argument_parser(run_parser, dest_prefix="driver")
    add_logging_arguments(run_parser)

    args = parser.parse_args()
    configure_logger(logger, log_level=args.log_level)

    args.func(args)


if __name__ == "__main__":
    main()
