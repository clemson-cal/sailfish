from collections.abc import MutableMapping
from collections import ChainMap
from loguru import logger
from numpy import linspace, meshgrid, zeros, logical_not
from numpy.typing import NDArray
from new_kernels import (
    kernel,
    kernel_class,
    perf_time_sequence,
    configure_kernel_module,
    device,
)
from lib_euler import prim_to_cons, cons_to_prim, riemann_hlle
from configuration import configurable, all_schemas


@device()
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


@kernel(device_funcs=[prim_to_cons], define_macros=dict(DIM=1))
def prim_to_cons_array(p: NDArray[float], u: NDArray[float], ni: int = None):
    R"""
    KERNEL void prim_to_cons_array(double *p, double *u, int ni)
    {
        FOR_RANGE_1D(1, ni - 1)
        {
            prim_to_cons(&p[NCONS * i], &u[NCONS * i]);
        }
    }
    """
    return p.size // 3, (p, u, p.size // 3)


@kernel_class
class FluxPerFaceSolver:
    """
    Solves 1d Euler equation with per-face fluxing at 1st or 2nd order.

    This solver supports 4 configurations -- with/without gradient estimation
    (via PLM) and with/without explicit Runge-Kutta support. Fluxing is done
    per-face, meaning that a Riemann problem is solved only once per face. This
    approach is best in compute-limited settings because it minimizes flops.
    """

    def __init__(self, runge_kutta=False, plm=False):
        self.runge_kutta = runge_kutta
        self.plm = plm

    @property
    def define_macros(self):
        return dict(DIM=1, RUNGE_KUTTA=int(self.runge_kutta), PLM=int(self.plm))

    @kernel(device_funcs=[riemann_hlle, plm_minmod], define_macros=dict(DIM=1))
    def compute_godunov_fluxes(
        self,
        p: NDArray[float],
        f: NDArray[float],
        plm_theta: float,
        ni: int = None,
    ):
        R"""
        KERNEL void compute_godunov_fluxes(double *p, double *f, double plm_theta, int ni)
        {
            FOR_RANGE_1D(1, ni - 2)
            {
                double pm[NCONS];
                double pp[NCONS];
                double *fh = &f[NCONS * (i + 1)];

                #if PLM == 0
                (void) plm_minmod; // unused function

                double *pc = &p[NCONS * (i + 0)];
                double *pr = &p[NCONS * (i + 1)];

                for (int q = 0; q < NCONS; ++q)
                {
                    pm[q] = pc[q];
                    pp[q] = pr[q];
                }
                #else

                double *pl = &p[NCONS * (i - 1)];
                double *pc = &p[NCONS * (i + 0)];
                double *pr = &p[NCONS * (i + 1)];
                double *ps = &p[NCONS * (i + 2)];

                for (int q = 0; q < NCONS; ++q)
                {
                    pm[q] = pc[q] + 0.5 * plm_minmod(pl[q], pc[q], pr[q], plm_theta);
                    pp[q] = pr[q] - 0.5 * plm_minmod(pc[q], pr[q], ps[q], plm_theta);
                }
                #endif

                riemann_hlle(pm, pp, fh, 1);
            }
        }
        """
        return p.shape[0], (p, f, plm_theta, p.shape[0])

    @kernel(device_funcs=[prim_to_cons, cons_to_prim])
    def update_prim(
        self,
        p: NDArray[float],
        f: NDArray[float],
        urk: NDArray[float],
        dt: float,
        dx: float,
        rk: float,
        ni: int = None,
    ):
        R"""
        KERNEL void update_prim(
            double *p,
            double *f,
            double *urk,
            double dt,
            double dx,
            double rk,
            int ni)
        {
            FOR_RANGE_1D(2, ni - 2)
            {
                double uc[3];
                double *pc = &p[3 * (i + 0)];
                double *fm = &f[3 * (i + 0)];
                double *fp = &f[3 * (i + 1)];

                prim_to_cons(pc, uc);

                for (int q = 0; q < 3; ++q)
                {
                    uc[q] -= (fp[q] - fm[q]) * dt / dx;
                    #if RUNGE_KUTTA == 1
                    uc[q] *= (1.0 - rk);
                    uc[q] += rk * urk[3 * i + q];
                    #endif
                }
                cons_to_prim(uc, pc);
            }
        }
        """
        return p.size // 3, (p, f, urk, dt, dx, rk, p.size // 3)


@kernel_class
class FluxPerZoneSolver:
    """
    Solves 1d Euler equation with per-zone fluxing at 1st or 2nd order.

    This solver supports 4 configurations -- with/without gradient estimation
    (via PLM) and with/without explicit Runge-Kutta support. Fluxing is done
    per-zone, meaning that a Riemann problem is solved with two-fold redundancy;
    both zones sharing a given face compute the Godunov flux through it. This
    approach is best in memory bandwidth-limited settings because it minimizes
    access to global memory.
    """

    def __init__(self, runge_kutta=False, plm=False):
        self.runge_kutta = runge_kutta
        self.plm = plm

    @property
    def define_macros(self):
        return dict(DIM=1, RUNGE_KUTTA=int(self.runge_kutta), PLM=int(self.plm))

    @kernel(device_funcs=[prim_to_cons, cons_to_prim, riemann_hlle, plm_minmod])
    def update_prim(
        self,
        prd: NDArray[float],  # read-from primitive
        pwr: NDArray[float],  # write-to-primitive
        urk: NDArray[float],  # u at time-level n
        dt: float,  # time step dt
        dx: float,  # grid spacing dx
        rk: float,  # RK parameter
        plm_theta: float,
        ni: int = None,
    ):
        R"""
        KERNEL void update_prim(
            double *prd,
            double *pwr,
            double *urk,
            double dt,
            double dx,
            double rk,
            double plm_theta,
            int ni)
        {
            FOR_RANGE_1D(2, ni - 2)
            {
                double uc[NCONS];
                double fm[NCONS];
                double fp[NCONS];
                double plp[NCONS];
                double pcm[NCONS];
                double pcp[NCONS];
                double prm[NCONS];

                #if PLM == 0
                (void) plm_minmod; // unused function

                double *pl = &prd[NCONS * (i - 1)];
                double *pc = &prd[NCONS * (i + 0)];
                double *pr = &prd[NCONS * (i + 1)];

                for (int q = 0; q < NCONS; ++q)
                {
                    plp[q] = pl[q];
                    pcm[q] = pc[q];
                    pcp[q] = pc[q];
                    prm[q] = pr[q];
                }
                #else

                double *pk = &prd[NCONS * (i - 2)];
                double *pl = &prd[NCONS * (i - 1)];
                double *pc = &prd[NCONS * (i + 0)];
                double *pr = &prd[NCONS * (i + 1)];
                double *ps = &prd[NCONS * (i + 2)];

                for (int q = 0; q < NCONS; ++q)
                {
                    double gl = plm_minmod(pk[q], pl[q], pc[q], plm_theta);
                    double gc = plm_minmod(pl[q], pc[q], pr[q], plm_theta);
                    double gr = plm_minmod(pc[q], pr[q], ps[q], plm_theta);

                    plp[q] = pl[q] + 0.5 * gl;
                    pcm[q] = pc[q] - 0.5 * gc;
                    pcp[q] = pc[q] + 0.5 * gc;
                    prm[q] = pr[q] - 0.5 * gr;
                }
                #endif

                riemann_hlle(plp, pcm, fm, 1);
                riemann_hlle(pcp, prm, fp, 1);
                prim_to_cons(pc, uc);

                for (int q = 0; q < NCONS; ++q)
                {
                    uc[q] -= (fp[q] - fm[q]) * dt / dx;
                    #if RUNGE_KUTTA == 1
                    uc[q] *= (1.0 - rk);
                    uc[q] += rk * urk[NCONS * i + q];
                    #endif
                }
                cons_to_prim(uc, &pwr[NCONS * i]);
            }
        }
        """
        return prd.shape[0], (prd, pwr, urk, dt, dx, rk, plm_theta, prd.shape[0])


class SolverClass:
    @classmethod
    def get(cls, time_integration, reconstruction):
        if reconstruction == "pcm":
            if time_integration == "fwd":
                return cls.fwd_pcm
            elif time_integration in ("rk1", "rk2", "rk3"):
                return cls.rkn_pcm
        elif reconstruction == "plm":
            if time_integration == "fwd":
                return cls.fwd_plm
            elif time_integration in ("rk1", "rk2", "rk3"):
                return cls.rkn_plm
        else:
            raise ValueError(f"no solver config {time_integration}/{reconstruction}")


class FPFSolvers(SolverClass):
    @classmethod
    def init(cls):
        cls.fwd_plm = FluxPerFaceSolver(plm=True, runge_kutta=False)
        cls.rkn_plm = FluxPerFaceSolver(plm=True, runge_kutta=True)
        cls.fwd_pcm = FluxPerFaceSolver(plm=False, runge_kutta=False)
        cls.rkn_pcm = FluxPerFaceSolver(plm=False, runge_kutta=True)


class FPZSolvers(SolverClass):
    @classmethod
    def init(cls):
        cls.fwd_plm = FluxPerZoneSolver(plm=True, runge_kutta=False)
        cls.rkn_plm = FluxPerZoneSolver(plm=True, runge_kutta=True)
        cls.fwd_pcm = FluxPerZoneSolver(plm=False, runge_kutta=False)
        cls.rkn_pcm = FluxPerZoneSolver(plm=False, runge_kutta=True)


def update_prim(
    p,
    dt,
    dx,
    fluxing="per_zone",
    reconstruction="pcm",
    plm_theta=2.0,
    time_integration="fwd",
    exec_mode="cpu",
):
    """
    Drives a first-order update of a primitive array
    """
    xp = numpy_or_cupy(exec_mode)

    if time_integration == "fwd":
        pass
    elif time_integration == "rk1":
        rks = [0.0]
    elif time_integration == "rk2":
        rks = [0.0, 0.5]
    elif time_integration == "rk3":
        rks = [0.0, 3.0 / 4.0, 1.0 / 3.0]
    else:
        raise ValueError(
            f"time_integration must be [fwd|r1k|rk2|rk3], got {time_integration}"
        )

    if fluxing == "per_face":
        update = FPFSolvers.get(time_integration, reconstruction).update_prim
        fluxes = FPFSolvers.get(time_integration, reconstruction).compute_godunov_fluxes

        f = xp.empty_like(p)

        if time_integration == "fwd":
            fluxes(p, f, plm_theta)
            update(p, f, None, dt, dx, 0.0)
        else:
            urk = xp.empty_like(p)
            prim_to_cons_array(p, urk)

            for rk in rks:
                fluxes(p, f, plm_theta)
                update(p, f, urk, dt, dx, rk)
        return p

    elif fluxing == "per_zone":
        update = FPZSolvers.get(time_integration, reconstruction).update_prim
        prd = p
        pwr = p.copy()

        if time_integration == "fwd":
            update(prd, pwr, None, dt, dx, 0.0, plm_theta)
            prd, pwr = pwr, prd

        else:
            urk = xp.empty_like(prd)
            prim_to_cons_array(prd, urk)

            for rk in rks:
                update(prd, pwr, urk, dt, dx, rk, plm_theta)
                prd, pwr = pwr, prd

        return prd
    else:
        raise ValueError(f"unknown fluxing {fluxing}")


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


def numpy_or_cupy(exec_mode):
    if exec_mode == "gpu":
        import cupy

        return cupy

    if exec_mode == "cpu":
        import numpy

        return numpy


class State:
    def __init__(self, n, t, p):
        self._n = n
        self._t = t
        self._p = p

    @property
    def iteration(self):
        return self._n

    @property
    def time(self):
        return self._t

    @property
    def primitive(self):
        try:
            return self._p.get()
        except AttributeError:
            return self._p

    @property
    def total_zones(self):
        return self._p.shape[0]


def simulation(exec_mode, resolution, fluxing, reconstruction, time_integration):
    from functools import partial

    xp = numpy_or_cupy(exec_mode)
    nz = resolution
    dx = 1.0 / nz
    dt = dx * 1e-1
    x = cell_centers_1d(nz)
    p = linear_shocktube(x)
    t = 0.0
    n = 0
    p = xp.array(p)
    iteration = 0

    advance = partial(
        update_prim,
        dx=dx,
        fluxing=fluxing,
        reconstruction=reconstruction,
        time_integration=time_integration,
        exec_mode=exec_mode,
    )
    yield State(n, t, p)

    while True:
        p = advance(p, dt)
        t += dt
        n += 1
        yield State(n, t, p)


@configurable
def driver(
    app_config,
    exec_mode: str = "cpu",
    resolution: int = 10000,
    tfinal: float = 0.1,
    fluxing: str = "per_zone",
    reconstruction: str = "pcm",
    plm_theta: float = 1.5,
    time_integration: str = "fwd",
    fold: int = 100,
    plot: bool = False,
):
    """
    Configuration
    -------------

    exec_mode:        execution mode [cpu|gpu]
    resolution:       number of grid zones
    tfinal:           time to end the simulation
    fluxing:          solver fluxing [per_zone|per_face]
    reconstruction:   first or second-order reconstruction [pcm|plm]
    plm_theta:        PLM parameter [1.0, 2.0]
    time_integration: Runge-Kutta order [fwd|rk1|rk2|rk3]
    fold:             number of iterations between iteration message
    plot:             whether to show a plot of the solution
    """
    from reporting import terminal, iteration_msg

    configure_kernel_module(default_exec_mode=exec_mode)
    FPZSolvers.init()
    FPFSolvers.init()

    term = terminal(logger)
    sim = simulation(
        exec_mode,
        resolution,
        fluxing,
        reconstruction,
        time_integration,
    )
    perf_timer = perf_time_sequence(mode=exec_mode)
    state = next(sim)
    logger.info(f"initialization tool {1e3 * next(perf_timer):.3f}ms")
    logger.info("start simulation")

    while state.time < tfinal:
        state = next(sim)

        # if tasks.checkpoint.is_due(state.time):
        #     write_checkpoint(state, timeseries, tasks)

        # if tasks.timeseries.is_due(state.time):
        #     for name, info in diagnostics.items():
        #         timeseries[name].append(state.diagnostic(info))

        if state.iteration % fold == 0:
            zps = state.total_zones / next(perf_timer) * fold
            term(iteration_msg(state.iteration, state.time, zps=zps))

    if plot:
        from matplotlib import pyplot as plt

        plt.plot(state.primitive[:, 0], "-o", mfc="none", label=fluxing)
        plt.show()


def flatten_dict(
    d: MutableMapping,
    parent_key: str = "",
    sep: str = ".",
) -> MutableMapping:
    """
    Create a flattened dictionary e from d, with e['a.b.c'] = d['a']['b']['c'].
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
    app_config = ChainMap(
        {k: v for k, v in vars(args).items() if v is not None and "." in k}
    )
    app_config.maps.extend(flatten_dict(load_config(c)) for c in reversed(args.configs))
    driver_args = dict_section(app_config, "driver")
    driver.schema.validate(**driver_args)
    driver.schema.print_schema(
        args.term,
        config=driver_args,
        newline=True,
    )
    driver(app_config, **driver_args)


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
    try:
        main()
    except KeyboardInterrupt:
        print()
        logger.success("ctrl-c interrupt")
