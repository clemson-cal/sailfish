"""
Sailfish main program
"""


from argparse import ArgumentParser, SUPPRESS
from collections import deque
from collections.abc import Mapping, Iterable
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
from inspect import getsource, isgeneratorfunction
from itertools import chain, product
from json import load as load_json, dump as dump_json, dumps as dumps_json
from logging import getLogger
from pickle import load as load_pickle, dump as dump_pickle
from sys import argv
from textwrap import dedent
from pydantic import ValidationError

from .config import (
    Sailfish,
    Strategy,
    Scheme,
    Driver,
    add_config_arguments,
)
from .models import get_model_data_classes
from .preset import preset, get_preset_functions
from .system import system_info


@dataclass
class iteration_report:
    """
    Light-weight status of an iteration or group of iterations
    """

    iteration: int
    time: float
    zps: float

    def __repr__(self):
        return (
            f"[{self.iteration:06d}] "
            f"t={self.time:0.5f} "
            f"Mzps={self.zps/1e6:0.3f} "
        )

    def __rich_console__(self, *args):
        yield (
            f"[cyan]{self.iteration:06d}[/cyan] "
            f"[green]t={self.time:0.5f}[/green] "
            f"[magenta]Mzps={self.zps/1e6:0.3f}[/magenta]"
        )


@dataclass
class run_summary:
    """
    Light-weight epilog of a run
    """

    config: Sailfish
    total_sec: float
    mean_zps: float


@dataclass
class event_state:
    number: int = 0
    last_time: float = None


def recurring_event(interval: float, number: int = 0, last_time: float = None):
    """
    A generator which yields True or False based on whether a task is due
    """
    is_due = None

    while True:
        time = yield is_due and event_state(number, last_time)

        if time == 0.0:
            is_due = True
            last_time = time

        elif time >= last_time + interval:
            is_due = True
            number += 1
            last_time = last_time + interval

        else:
            is_due = False


def timeseries(config: Sailfish, state, timestep, timeseries, event_states):
    number = event_states["timeseries"].number
    time = timeseries.setdefault("time", list())
    time.append(state.time)

    return f"timeseries {number}"


def checkpoint(
    config: Sailfish,
    state,
    timestep,
    timeseries,
    event_states,
    filename: str = None,
):
    if config.driver.checkpoint.format == "none":
        return "skip checkpoint (format=none)"

    if filename is None:
        number = event_states["checkpoint"].number
        filename = f"chkpt.{number:04d}.pk"

    with open(filename, "wb") as outf:
        dump_pickle(
            dict(
                config=asdict(config),
                primitive=state.primitive,
                time=state.time,
                iteration=state.iteration,
                timestep=timestep,
                timeseries=timeseries,
                event_states={k: asdict(v) for k, v in event_states.items()},
            ),
            outf,
        )

    return filename


def drive(setups: Iterable[tuple[Sailfish, dict]]):
    """
    Main simulation driver function

    This function drives a sequence of applications. The `setups` argument is
    a sequence of tuples: a `Sailfish` configuration instance, and a possible
    checkpoint file from which the run is to be restarted.

    This function does not write anything to the console, instead it yields
    items to be dealt with by a run monitor: either a terminal scrollback or
    dashboard, or maybe eventually a full UI.

    This function does handle checkpointing and the collection of time series
    data from the solver.
    """
    from .kernels import perf_time_sequence, configure_kernel_module
    from .solver import make_solver

    for config, chkpt in setups:
        yield config

        mode = config.strategy.hardware
        configure_kernel_module(default_exec_mode=mode)

        driver = config.driver
        fold = driver.report.cadence
        tfinal = driver.tfinal
        cfl_number = driver.cfl_number
        new_timestep_cadence = driver.new_timestep_cadence
        solver = make_solver(config, chkpt)
        timeseries_data = dict()
        zps_log = list()
        app_timer = perf_time_sequence(mode)
        fld_timer = perf_time_sequence(mode)

        # =====================================================================
        # Get initial state from solver, and start the main loop
        # =====================================================================
        state = next(solver)

        # =====================================================================
        # Set up timeseries and checkpoint task handlers
        # =====================================================================
        if chkpt:
            ts_state = chkpt["event_states"]["timeseries"]
            cp_state = chkpt["event_states"]["checkpoint"]
            timestep = chkpt["timestep"]
        else:
            ts_state = dict()
            cp_state = dict()
            timestep = state.timestep(cfl_number)

        events = dict(
            timeseries=recurring_event(driver.timeseries.cadence, **ts_state),
            checkpoint=recurring_event(driver.checkpoint.cadence, **cp_state),
        )
        event_states = dict(
            timeseries=event_state(**ts_state),
            checkpoint=event_state(**cp_state),
        )
        event_funcs = dict(
            timeseries=timeseries,
            checkpoint=checkpoint,
        )
        for event in events.values():
            event.send(None)

        while True:
            for name in events:
                if e := events[name].send(state.time):
                    event_states[name] = e
                    yield event_funcs[name](
                        config, state, timestep, timeseries_data, event_states
                    )

            if state.time >= tfinal:
                break

            state = solver.send(timestep)

            if state.iteration % fold == 0:
                sec = next(fld_timer)
                zps = state.total_zones / sec * fold
                if state.iteration > 0:
                    zps_log.append(zps)
                yield iteration_report(state.iteration, state.time, zps)

            if state.iteration % new_timestep_cadence == 0:
                timestep = state.timestep(cfl_number)

        yield checkpoint(
            config, state, timestep, timeseries_data, event_states, "chkpt.final.pk"
        )
        yield state
        yield run_summary(config, next(app_timer), sum(zps_log) / max(1, len(zps_log)))


def scrollback(print):
    """
    A run monitor that writes run status and iteration reports to the console

    The print function can be builtin, or something else such as rich.print.
    """
    while True:
        event = yield

        if type(event) is Sailfish:
            config = event
            print(config)

        elif type(event) is run_summary:
            pass
            # summary = event
            # print()
            # printing of run summary is currently disabled
            #
            # print(summary)
            # print()

        elif type(event) is iteration_report:
            report = event
            print(report)
        elif type(event) is str:
            print(event)


def reports_table(reports: Iterable[iteration_report]):
    """
    Generate a rich table from a sequence of iteration reports
    """
    from rich.table import Table
    from rich.panel import Panel

    table = Table(expand=True, show_edge=False)
    table.add_column("iteration", style="cyan", justify="right")
    table.add_column("time", style="green")
    table.add_column("zones per second (millions)", style="magenta", justify="left")

    for report in reports:
        table.add_row(
            f"{report.iteration}",
            f"{report.time:0.5f}",
            f"{report.zps/1e6:0.3f}",
        )

    table.add_section()

    if any(reports):
        avg_zps = sum(r.zps for r in reports) / max(1, len(reports))
        table.add_row(None, None, f"{avg_zps/1e6:0.3f}", style="italic")

    return Panel(table, style="dim", border_style="blue", padding=(2, 2))


def summaries_table(summaries: list[run_summary]):
    """
    Generate a rich table from a sequence of run summary reports
    """

    from rich.table import Table
    from rich.panel import Panel

    table = Table(expand=True, show_edge=False)
    table.add_column("run name", style="cyan", justify="left")
    table.add_column("<zones per second> (millions)", style="magenta", justify="left")

    for summary in summaries:
        table.add_row(f"{summary.config.name}", f"{summary.mean_zps/1e6:0.5f}")

    return Panel(table, style="dim", border_style="red", padding=(2, 2))


def checkpoints_table(checkpoints: list[str]):
    """
    Render a table of checkpoint filenames
    """
    from rich.columns import Columns
    from rich.panel import Panel

    return Panel(
        Columns(checkpoints),
        border_style="red",
        padding=(2, 2),
    )


def dashboard(console, screen=False):
    """
    A rich dashboard that displays key run information without scrollback

    NOTE: The dashboard can have a measurable effect on the code performance,
    so make sure not to use it for recording code perforamance.
    """
    from rich.live import Live
    from rich.pretty import Pretty
    from rich.progress import Progress
    from rich.layout import Layout
    from rich.panel import Panel

    reports = deque()
    summaries = list()
    progress = Progress()
    checkpoints = list()
    app_struct = str()
    job_num = 0

    config_view = Layout(str(), name="config", ratio=5)
    progress_view = Layout(
        Panel(
            progress,
            title="Job Progress",
            padding=(2, 2),
            border_style="green",
        ),
        ratio=3,
        name="progress",
    )
    reports_view = Layout(reports_table(reports), name="reports", ratio=3)
    summaries_view = Layout(summaries_table(reports), name="summaries", ratio=5)
    checkpoints_view = Layout(checkpoints_table(checkpoints), name="checkpoints")

    root = Layout(name="root")
    root.split_column(Layout(name="upper", ratio=2), Layout(name="lower", ratio=3))
    root["upper"].split_row(config_view, progress_view)
    root["lower"].split_row(Layout(name="lower-left", ratio=3), summaries_view)
    root["lower-left"].split_column(reports_view, checkpoints_view)

    with Live(
        root,
        console=console,
        auto_refresh=False,
        screen=screen,
    ) as live:
        while True:
            event = yield

            if type(event) is Sailfish:
                config = event
                job_num += 1
                duration = config.driver.tfinal - config.driver.tstart
                run_task = progress.add_task(f"job {job_num}", total=duration)
                config_view.update(
                    Panel(
                        Pretty(config),
                        title="Run Description",
                        padding=(2, 2),
                        border_style="bright_blue",
                    )
                )
                checkpoints.clear()

            elif type(event) is iteration_report:
                report = event
                reports.append(report)

                if len(reports) > 20:
                    reports.popleft()

                reports_view.update(reports_table(reports))
                evolved_time = report.time - config.driver.tstart
                progress.update(run_task, completed=evolved_time)

            elif type(event) is str and "chkpt" in event:
                checkpoints.append(event)
                checkpoints_view.update(checkpoints_table(checkpoints))

            elif type(event) is run_summary:
                summary = event
                reports = deque()
                summaries.append(summary)
                summaries_view.update(summaries_table(summaries))

            live.refresh()


def plot():
    from matplotlib import pyplot as plt

    while True:
        event = yield

        if hasattr(event, "primitive"):
            state = event

        if type(event) is run_summary:
            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            if state.box.dimensionality == 1:
                ax1.plot(
                    state.cell_centers,
                    state.primitive[:, 0],
                    "-o",
                    mfc="none",
                )
            if state.box.dimensionality == 2:
                x, y = state.cell_centers
                z = state.primitive[:, :, 0]
                cm = ax1.pcolormesh(x, y, z, vmin=None, vmax=None)
                ax1.set_aspect("equal")
                fig.colorbar(cm)
            plt.show()

    while True:
        yield


def no_plot():
    while True:
        yield


def deep_update(d: dict, u: dict) -> dict:
    """
    Update `d` and any nested dictionaries recursively with values from `u`
    """
    for k, v in u.items():
        if isinstance(v, Mapping):
            d[k] = deep_update(d.setdefault(k, dict()), v)
        elif isinstance(d, Mapping):
            d[k] = v
        elif d is None:
            d = u
    return d


def unflatten(d: dict) -> dict:
    """
    Create a nested dict from a flat one with keys like a.b.c
    """
    res = dict()
    for key, value in d.items():
        parts = key.split(".")
        d = res
        for part in parts[:-1]:
            if part not in d:
                d[part] = dict()
            d = d[part]
        d[parts[-1]] = value
    return res


def init_logging(level):
    from rich.console import Console
    from rich.logging import RichHandler

    console = Console()
    handler = RichHandler(omit_repeated_times=False, console=console)
    logger = getLogger("sailfish")
    logger.addHandler(handler)
    logger.setLevel(level.upper())

    return console


@preset
def scan_strategies():
    d = {
        "driver.checkpoint.format": ("none",),
        "strategy.data_layout": ("fields-last", "fields-first"),
        "strategy.cache_flux": (True, False),
        "strategy.cache_prim": (True, False),
        "strategy.cache_grad": (True, False),
    }
    for p in product(*d.values()):
        yield dict(name=str(p), **dict(zip(d.keys(), p)))


def sailfish(config, overrides, console):
    """
    Yield a sequence of sailfish app structs from a config

    The config may be the name of a preset, which may then be a dictionary
    or sequence of dictionaries, or it may be a path to a json file
    containing a dictionary or a list of dictionaries.
    """
    presets = get_preset_functions()
    chkpt = None

    if not config:
        cs = dict()
    elif "." not in config:
        # no extension; it might be a preset
        try:
            cs = presets[config]()
        except KeyError as e:
            console.print(f"No preset named {e}. Available presets:")
            console.print()
            console.print("\n".join(presets.keys()))
            console.print()
            console.print("for more information run 'sailfish doc presets'")
            return
    elif config.endswith(".json"):
        # it's a configuration file
        with open(config) as infile:
            cs = load_json(infile)
    elif config.endswith(".pk"):
        # it's a checkpoint file
        with open(config, "rb") as infile:
            chkpt = load_pickle(infile)
            cs = chkpt["config"]
    else:
        raise ValueError("config must be a preset name or a json file")

    if type(cs) is dict:
        cs = [cs]

    for c in cs:
        s = asdict(Sailfish())
        deep_update(s, unflatten(c))
        deep_update(s, overrides)
        try:
            config = Sailfish(**s)
            config.initialize()
        except ValidationError as e:
            console.print("[red]configuration error[red]:")
            print(e)
            return
        except ValueError as e:
            console.print("[red]configuration error[red]:", e)
            return
        yield config, chkpt


def run(args=None, console=None, parser=None):
    """
    Run a simulation or sequence of simulations
    """
    if parser:
        parser.add_argument(
            "_configs",
            metavar="configs",
            nargs="*",
            help="sequence of configuration files, checkpoints, or preset/setup names",
        )
        parser.add_argument(
            "--dash",
            dest="_dash",
            action="store_true",
            help="show a dashboard instead of a scrollback",
        )
        parser.add_argument(
            "--screen",
            dest="_screen",
            action="store_true",
            help="transient screen in dash mode (better look but disappears after the run)",
        )
        parser.add_argument(
            "--plot",
            dest="_plot",
            action="store_true",
            help="show a plot of the solution",
        )
        parser.add_argument(
            "--dump-summaries",
            dest="_dump_summaries",
            action="store_true",
            help="dump a JSON object with run summaries",
        )
        config = parser.add_argument_group("config")
        add_config_arguments(config)

    else:
        if not args._configs:
            console.print("Need a configuration file or a preset name, e.g.")
            console.print()
            console.print("> sailfish run sod")
            console.print("> sailfish run my_run.json")
            console.print()
            console.print("\n".join(get_preset_functions()))
            console.print()
            console.print("For more information:")
            console.print()
            console.print("> sailfish doc presets --more")
            return

        overrides = unflatten(
            {k: v for k, v in vars(args).items() if v is not None and k[0] != "_"}
        )
        runs = chain(*(sailfish(cfg, overrides, console) for cfg in args._configs))
        summaries = list()

        try:
            if args._dash:
                monitor = dashboard(console, args._screen)
            else:
                monitor = scrollback(console.print)
            if args._plot:
                plotter = plot()
            else:
                plotter = no_plot()

            next(monitor)
            next(plotter)

            for event in drive(runs):
                monitor.send(event)
                plotter.send(event)

                if type(event) is run_summary:
                    summaries.append(asdict(event))

        except Exception:
            monitor.close()
            plotter.close()
            console.print_exception(show_locals=False)

        finally:
            if args._dump_summaries:
                fname = datetime.now().strftime("%m-%d-%Y-%H%M.json")
                dump = dict(
                    cmdline=argv, summaries=summaries, system_info=system_info()
                )
                with open(fname, "w") as outf:
                    console.print(f"write summaries to {fname}")
                    dump_json(dump, outf, indent=4)


def doc(args=None, console=None, parser=None):
    """
    Display code documentation
    """
    choices = ("kernels", "config", "solver", "physics", "models", "presets")

    if parser:
        parser.add_argument("topic", nargs="?", choices=choices)
        parser.add_argument("subtopic", nargs="?")
        parser.add_argument("--more", action="store_true", help="show more detail")
    else:
        from rich.markdown import Markdown
        from rich.syntax import Syntax
        from rich.text import Text
        from rich.prompt import Prompt, DefaultType

        from .kernels import main as kernels_main
        from .solver import doc as solver_doc

        console.width = 80

        if (topic := args.topic) is None:
            topics = "# __Topics__"
            for choice in choices:
                topics += f"\n- {choice}"
            topics += "\n-----------"
            console.print(Markdown(topics))
            topic = Prompt.ask("select", choices=choices, show_choices=False)
            console.print("\n\n")
        if topic == "kernels":
            console.print(Syntax(getsource(kernels_main), lexer="python"))
        if topic == "config":
            console.print(next(Driver().rich_table(console, None)))
            console.print("\n\n")
            console.print(next(Strategy().rich_table(console, None)))
            console.print("\n\n")
            console.print(next(Scheme().rich_table(console, None)))
        if topic == "solver":
            try:
                doc = solver_doc()
                console.print(Markdown(doc[args.subtopic], inline_code_lexer="python"))
            except KeyError:
                console.print(f"Provide the name of a documented method, e.g.")
                console.print()
                console.print("> sailfish doc solver update_cons")
                console.print()
                console.print("sub-topics are:")
                console.print()
                console.print("\n".join(doc))
        if topic == "physics":
            console.print("Not yet documented")
        if topic == "models":
            for cls in get_model_data_classes():
                console.rule()
                console.print(Markdown(f"*{cls.__name__}*\n{dedent(cls.__doc__)}"))
        if topic == "presets":
            if args.more:
                for name, func in get_preset_functions().items():
                    if isgeneratorfunction(func):
                        pass
                    else:
                        console.rule()
                        console.print(f"{name}:")
                        if func.__doc__:
                            console.print(dedent(func.__doc__))
                        console.print(func())
                        console.print(
                            dedent(
                                Sailfish(
                                    **unflatten(func())
                                ).initial_data.__class__.__doc__
                            )
                        )
            else:
                console.print("\n".join(get_preset_functions()))


def dep(args=None, console=None, parser=None):
    """
    Display application dependencies
    """
    if parser:
        pass
    else:
        from importlib import import_module
        from rich.table import Table
        from rich import box

        def have(m):
            try:
                import_module(m)
                return f"[green]yes"
            except ImportError:
                return "[red]no"

        table = Table(box=box.MINIMAL)
        table.add_column("module")
        table.add_column("installed?")
        table.add_column("required for")
        table.add_column("pip3 install --user")
        table.add_row("[blue]pydantic", have("pydantic"), "everything", "pydantic")
        table.add_row("[blue]numpy", have("numpy"), "everything", "numpy")
        table.add_row("[blue]cupy", have("cupy"), "GPU acceleration", "cupy-cuda116")
        table.add_row("[blue]cffi", have("cffi"), "CPU native code", "cffi")
        table.add_row("[blue]rich", have("rich"), "formatted output", "rich")
        table.add_row(
            "[blue]matplotlib", have("matplotlib"), "plotting features", "matplotlib"
        )
        table.add_row(
            "[blue]cpuinfo", have("cpuinfo"), "detailed CPU info", "py-cpuinfo"
        )
        console.print(table)


def sys(args=None, console=None, parser=None):
    """
    Show platform information
    """
    if parser:
        pass
    else:
        console.print(system_info())


def code(args=None, console=None, parser=None):
    """
    Display generated native code in use by solver kernels
    """
    if parser:
        parser.add_argument(
            "_config",
            nargs="?",
            metavar="config",
            default="sod",
            help="configuration file, checkpoint, or preset/setup name",
        )
        parser.add_argument("--line-numbers", dest="_line_numbers", action="store_true")
        config = parser.add_argument_group("config")
        add_config_arguments(config)

    else:
        from rich.syntax import Syntax
        from rich.prompt import Prompt
        from .solver import native_code

        console.width = 100

        overrides = unflatten(
            {k: v for k, v in vars(args).items() if v is not None and k[0] != "_"}
        )
        config, chkpt = next(sailfish(args._config, overrides, console))

        for code in native_code(config):
            console.print(Syntax(code, "c", line_numbers=args._line_numbers))
            if Prompt.ask(str()) in ("q", "quit"):
                break


def todo(args=None, console=None, parser=None):
    """
    Display a list of development todo items
    """
    if parser:
        pass
    else:
        from textwrap import dedent
        from rich.markdown import Markdown

        text = """
        # Todo items

        - [x] maximum wavespeed / timestep calculation
        - [x] control over time-step recalculation cadence
        - [ ] code glossary
        - [ ] plotting and animation features
        - [x] graceful handling of validation errors
        - [/] boundary conditions:
            - [x] outflow
            - [x] inflow
            - [x] periodic
            - [ ] reflecting
        - [/] more advanced Riemann solvers
            - [ ] HLLC for Newtonian hydro
            - [ ] HLLC for relativistic hydro
            - [ ] exact for Newtonian hydro
            - [/] exact for relativistic hydro
        - [ ] models:
            - [x] 1d problems from Fu-Shu (2017)
            - [x] 1d density wave
            - [ ] 2d kelvin-helmholtz
            - [ ] 1d accretion disk
            - [ ] 2d accretion disk
            - [x] 1d isothermal vortex
            - [x] 2d isothermal vortex
            - [x] 1d relativistic shocktubes from RAM
        - [ ] crash detection
        - [ ] crash recovery (fallback strategies)
        - [ ] FMR
        - [x] relativistic solver
        - [ ] viscosity
        - [x] geometric source terms; cell geometry data
        - [x] spherical polar coordinate system
        - [x] cylindrical polar coordinates
        - [ ] expanding homologous mesh option
        - [ ] gravitational source terms
            - [ ] single-point-mass gravity
            - [ ] binary-point-mass gravity
        - [ ] point mass sink models
        - [ ] reductions needed for orbital evolution
        - [x] driving source terms (implements outer buffer)
        - [ ] isothermal EOS
        - [ ] angular-momentum conserving fix for 2d Cartesian geometry
        - [ ] angular-momentum in cylindrical and spherical coordinates
        - [ ] multi-GPU support
        - [ ] MPI parallel
        """
        console.print(Markdown(dedent(text)), width=100)


def argument_parser():
    """
    Create an argument parser instance for running from the command line
    """
    parser = ArgumentParser(
        prog="sailfish",
        usage=SUPPRESS,
        description="sailfish is a GPU-accelerated astrophysical gasdynamics code",
    )
    parser.set_defaults(_command=None)
    parser.add_argument(
        "--log-level",
        dest="_log_level",
        default="warning",
        choices=("debug", "info", "warning", "error", "critical"),
        help="log messages at and above this severity level",
    )
    parser.add_argument(
        "--user-module",
        "-u",
        dest="_user_module",
        metavar="U",
        default=None,
        type=str,
        help="a module containing user models and presets (e.g. -u my_setups)",
    )

    subparsers = parser.add_subparsers()
    _run = subparsers.add_parser("run", usage=SUPPRESS, help=run.__doc__)
    _sys = subparsers.add_parser("sys", usage=SUPPRESS, help=sys.__doc__)
    _doc = subparsers.add_parser("doc", usage=SUPPRESS, help=doc.__doc__)
    _dep = subparsers.add_parser("dep", usage=SUPPRESS, help=dep.__doc__)
    _code = subparsers.add_parser("code", usage=SUPPRESS, help=code.__doc__)
    _todo = subparsers.add_parser("todo", usage=SUPPRESS, help=todo.__doc__)

    _run.set_defaults(_command=run)
    _sys.set_defaults(_command=sys)
    _doc.set_defaults(_command=doc)
    _dep.set_defaults(_command=dep)
    _code.set_defaults(_command=code)
    _todo.set_defaults(_command=todo)

    run(parser=_run)
    sys(parser=_sys)
    doc(parser=_doc)
    dep(parser=_dep)
    code(parser=_code)
    todo(parser=_todo)

    return parser


def main():
    """
    Main sailfish entry point and command line interface
    """
    try:
        parser = argument_parser()
        args = parser.parse_args()
        console = init_logging(args._log_level)

        if m := args._user_module:
            __import__(m)

        if args._command:
            args._command(args, console)
        else:
            from rich.syntax import Syntax

            examples = [
                "> sailfish run sod --plot      # see a standard preset problem",
                "> sailfish doc models          # print information about model problems",
                "> sailfish dep                 # see python module dependencies",
                "> sailfish todo                # see status of development goals",
                "> sailfish code --line-numbers # print generated solver code",
            ]
            parser.print_help()
            console.print()
            console.print("Examples:")
            console.print()
            for example in examples:
                console.print(Syntax(example, lexer="bash"))

    except KeyboardInterrupt:
        print()
        print("ctrl-c interrupt")
