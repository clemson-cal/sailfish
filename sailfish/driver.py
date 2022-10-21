"""
Library functions and command-line access to the simulation driver.
"""

import os, pickle, pathlib
from typing import NamedTuple, Dict
from logging import getLogger
from sailfish.event import Recurrence, RecurringEvent, ParseRecurrenceError
from sailfish.setup import Setup, SetupError
from sailfish.solver import SolverBase
from sailfish.solvers import (
    SolverInitializationError,
    register_solver_extension,
    make_solver,
)

logger = getLogger(__name__)
user_build_config = dict()


class ConfigurationError(Exception):
    """An invalid runtime configuration"""


class ExtensionError(Exception):
    """An invalid extension was specified"""


def keyed_event(item):
    key, val = item.split("=")
    return key, Recurrence.from_str(val)


def keyed_value(item):
    try:
        key, val = item.split("=")
        return key, eval(val)

    except NameError:
        return key, val

    except SyntaxError:
        raise ConfigurationError(f"badly formed model parameter value {val} in {item}")

    except ValueError:
        raise ConfigurationError(f"badly formed model parameter {item}")


def first_not_none(*args):
    for arg in args:
        if arg is not None:
            return arg


def update_dict_where_none(new_dict, old_dict, frozen=[]):
    """
    Like `dict.update`, except `key=value` pairs in `old_dict` are only used
    to add / overwrite values in `new_dict` if they are `None` or missing.
    """
    for key in old_dict:
        old_val = old_dict.get(key)
        new_val = new_dict.get(key)

        if type(new_val) is dict and type(old_val) is dict:
            update_dict_where_none(new_val, old_val)

        elif old_val is not None:
            if new_val is None:
                new_dict[key] = old_val
            elif key in frozen and new_val != old_val:
                raise ConfigurationError(f"{key} cannot be changed")


def update_where_none(new, old, frozen=[]):
    """
    Same as `update_dict_where_none`, except operates on (immutable) named tuple
    instances and returns a new named tuple.
    """
    new_dict = new._asdict()
    old_dict = old._asdict()
    update_dict_where_none(new_dict, old_dict, frozen)
    return type(new)(**new_dict)


# The functions below were written to allow state to be written in terms of
# builtin Python objects (no sailfish application classes). That would be good
# practice because then pickle files can be opened on systems that don't have
# sailfish installed, so these functions should possibly be restored at some
# point. However in practice it's more convenient for post-processing to have
# immediate access to the sailfish objects after unpickling. It's also tedious
# to ensure that all sailfish objects have been removed in `asdict`, and are
# properly restored in `fromdict`.


# def asdict(t):
#     """
#     Convert named tuple instances to dictionaries.

#     This function operates recursively on the data members of a dictionary or
#     named tuple. Each object that is a named tuple is mapped to its dictionary
#     representation, with an additional `_type` key to indicate the named tuple
#     subclass. This mapping is applied to the simulation state before pickling,
#     so that `sailfish` module is not required to unpickle the checkpoint
#     files.
#     """
#     if type(t) is dict:
#         return {k: asdict(v) for k, v in t.items()}
#     if isinstance(t, tuple):
#         d = {k: asdict(v) for k, v in t._asdict().items()}
#         d["_type"] = ".".join([type(t).__module__, type(t).__name__])
#         return d
#     return t


# def fromdict(d):
#     """
#     Convert from dictionaries to named tuples.

#     This function performs the inverse of the `asdict` method, and is applied
#     to pickled simulation states.
#     """
#     import sailfish

#     if type(d) is dict:
#         if "_type" in d:
#             cls = eval(d["_type"])
#             del d["_type"]
#             return cls(**{k: fromdict(v) for k, v in d.items()})
#         else:
#             return {k: fromdict(v) for k, v in d.items()}
#     else:
#         return d


def write_checkpoint(number, outdir, state):
    """
    Write the simulation state to a file, as a pickle.
    """
    if type(number) is int:
        filename = f"chkpt.{number:04d}.pk"
    elif type(number) is str:
        filename = f"chkpt.{number}.pk"
    else:
        raise ValueError("number arg must be int or str")

    if outdir is not None:
        pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)
        filename = os.path.join(outdir, filename)

    state_checkpoint_dict = dict(
        iteration=state.iteration,
        time=state.solver.time,
        timestep_dt=state.timestep_dt,
        cfl_number=state.cfl_number,
        solution=state.solver.solution,
        primitive=state.solver.primitive,
        timeseries=state.timeseries,
        solver=state.setup.solver,
        solver_options=state.solver.options,
        event_states=state.event_states,
        driver=state.driver,
        model_parameters=state.setup.model_parameter_dict(),
        setup_name=state.setup.dash_case_class_name(),
        mesh=state.mesh,
        **state.setup.checkpoint_diagnostics(state.solver.time),
    )

    with open(filename, "wb") as chkpt:
        logger.info(f"write checkpoint {chkpt.name}")
        pickle.dump(state_checkpoint_dict, chkpt)


def load_checkpoint(chkpt_file):
    """
    Load the simulation state from a pickle file.
    """
    try:
        with open(chkpt_file, "rb") as file:
            return pickle.load(file)
    except FileNotFoundError:
        raise ConfigurationError(f"could not open checkpoint file {chkpt_file}")


def newest_chkpt_in_directory(directory_name):
    import re

    expr = re.compile("chkpt\.([0-9]+)\.pk")
    list_of_matches = list(
        filter(None, (expr.search(f) for f in os.listdir(directory_name)))
    )
    list_of_matches.sort(key=lambda l: int(l.groups()[0]))

    for match in reversed(list_of_matches):
        try:
            path = os.path.join(directory_name, match.group())
            load_checkpoint(path)  # exception if checkpoint is corrupted
            return path
        except:
            logger.warning(f"skipping corrupt checkpoint file {path}")

    raise ConfigurationError("the specified directory did not have usable checkpoints")


def append_timeseries(state):
    """
    Append to the driver state timeseries for post-processing.
    """

    reductions = state.solver.reductions()

    if reductions:
        state.timeseries.append(state.solver.reductions())
        logger.info(f"record timeseries event {len(state.timeseries)}")
    else:
        logger.warning(
            "timeseries event ignored because solver does not provide reductions"
        )


class DriverArgs(NamedTuple):
    """
    Contains data used by the driver.
    """

    setup_name: str = None
    chkpt_file: str = None
    model_parameters: dict = None
    solver_options: dict = None
    cfl_number: float = None
    end_time: float = None
    execution_mode: str = None
    fold: int = None
    resolution: int = None
    num_patches: int = None
    events: Dict[str, Recurrence] = dict()
    new_timestep_cadence: int = None
    verbose_output: str = ""

    def from_namespace(args):
        """
        Construct an instance from an argparse-type namespace object.
        """
        driver = DriverArgs(
            **{k: w for k, w in vars(args).items() if k in DriverArgs._fields}
        )
        parts = args.command.split(":")

        if args.restart_dir:
            setup_name = None
            chkpt_file = newest_chkpt_in_directory(parts[0])
        elif parts[0].endswith(".pk"):
            setup_name = None
            chkpt_file = parts[0]
        else:
            setup_name = parts[0]
            chkpt_file = None

        try:
            model_parameters = dict(keyed_value(a) for a in parts[1:])
        except IndexError:
            model_parameters = dict()

        model_parameters.update(args.model_parameters)
        return driver._replace(
            setup_name=setup_name,
            chkpt_file=chkpt_file,
            model_parameters=model_parameters,
        )


class DriverState(NamedTuple):
    """
    Contains the stateful variables in use by the :pyobj:`simulate` function.

    An instance of this class is yielded by :pyobj:`simulate` each time an
    event takes place.
    """

    iteration: int
    driver: DriverArgs
    mesh: object
    timeseries: list
    event_states: list
    solver: SolverBase
    setup: Setup
    cfl_number: float
    timestep_dt: float


def simulate(driver):
    """
    Main generator for running simulations.

    If invoked with a `DriverArgs` instance in `driver`, the other arguments
    are ignored. Otherwise, the driver is created from the setup name, model
    paramters, and keyword arguments.

    This function is a generator: it yields its state at a sequence of
    pause points, defined by the `events` dictionary.
    """
    from time import perf_counter
    from sailfish import __version__ as version
    from sailfish.kernel.system import configure_build, log_system_info, measure_time
    from sailfish.event import Recurrence
    from sailfish import solvers

    main_logger = getLogger("main_logger")
    main_logger.info(f"\nsailfish {version}\n")

    """
    Initialize and log state in the the system module. The build system
    influences JIT-compiled module code. Currently the build parameters are
    inferred from the platform (Linux or MacOS), but in the future these
    should also be extensible by a system-specific rc-style configuration
    file.
    """
    configure_build(**user_build_config)
    log_system_info(driver.execution_mode or "cpu")

    if driver.setup_name:
        """
        Generate an initial driver state from command line arguments, model
        parametrs, and a setup instance.
        """
        logger.info(f"start new simulation with setup {driver.setup_name}")
        setup = Setup.find_setup_class(driver.setup_name)(
            **driver.model_parameters or dict()
        )
        driver = driver._replace(
            resolution=driver.resolution or setup.default_resolution,
        )

        iteration = 0
        time = setup.start_time
        event_states = {name: RecurringEvent() for name in driver.events}
        solution = None
        timeseries = list()
        dt = None

    elif driver.chkpt_file:
        """
        Load driver state from a checkpoint file. The setup model parameters
        are updated with any items given on the command line after the setup
        name. All command line arguments are also restorted from the
        previous session, but are updated with the command line argument
        given for this session, except for "frozen" arguments.
        """
        logger.info(f"load checkpoint {driver.chkpt_file}")
        chkpt = load_checkpoint(driver.chkpt_file)
        setup_class = Setup.find_setup_class(chkpt["setup_name"])
        driver = update_where_none(driver, chkpt["driver"], frozen=["resolution"])

        update_dict_where_none(
            driver.model_parameters,
            chkpt["model_parameters"],
            frozen=list(setup_class.immutable_parameter_keys()),
        )

        update_dict_where_none(
            driver.solver_options,
            chkpt["solver_options"],
        )

        setup = setup_class(**driver.model_parameters)

        iteration = chkpt["iteration"]
        time = chkpt["time"]
        event_states = chkpt["event_states"]
        solution = chkpt["solution"]

        try:
            dt = chkpt["timestep_dt"]
        except KeyError:
            # Forgive missing timestep_dt in the checkpoint, this key was
            # added recently (JZ 4-25-22). Prior to this change, timestep_dt
            # was not stored in the checkpoint file, and a restarted
            # simulation could end up different from a continuous one, when:
            # (1) new_timestep_cadence > 1, and (2) a new dt was not computed
            # just before the checkpoint was written. The differences would be
            # due to a slightly different timestep used, after it's recomputed
            # following the restart, and they would be minor. Still, restarted
            # runs are supposed to be bitwise identical continuous ones. Older
            # checkpoints will still work, but they will not have this
            # guarantee.
            logger.warning(
                "timestep_dt not in checkpoint, will recompute it on first iteration"
            )
            dt = None

        try:
            timeseries = chkpt["timeseries"]
        except KeyError:
            logger.warning("older checkpoint version: no timeseries")

        for event in driver.events:
            if event not in event_states:
                event_states[event] = RecurringEvent()

    else:
        raise ConfigurationError("driver args must specify setup_name or chkpt_file")

    mode = driver.execution_mode or "cpu"
    fold = driver.fold or 10
    mesh = setup.mesh(driver.resolution)
    end_time = first_not_none(driver.end_time, setup.default_end_time, float("inf"))
    reference_time = setup.reference_time_scale
    new_timestep_cadence = driver.new_timestep_cadence or 1
    dt = None

    if "physics" in driver.verbose_output:
        logger.info(f"physics struct (setup -> solver) {setup.physics}")
    if (
        "options" in driver.verbose_output
        or "solver" in driver.verbose_output
        or "solver-options" in driver.verbose_output
    ):
        logger.info(f"options struct (cmdline -> solver) {driver.solver_options}")

    solver = make_solver(
        setup.solver,
        setup.physics,
        driver.solver_options,
        setup=setup,
        mesh=mesh,
        time=time,
        solution=solution,
        num_patches=driver.num_patches or 1,
        mode=mode,
    )

    if driver.cfl_number is not None and driver.cfl_number > solver.maximum_cfl:
        raise ConfigurationError(
            f"cfl number {driver.cfl_number} "
            f"is greater than {solver.maximum_cfl}, "
            f"max allowed by solver {setup.solver}"
        )

    cfl_number = driver.cfl_number or solver.recommended_cfl

    for name, event in driver.events.items():
        logger.info(f"recurrence for {name} event is {event}")

    logger.info(f"run until t={end_time}")
    logger.info(f"CFL number is {cfl_number}")
    logger.info(f"simulation time / user time is {reference_time:0.4f}")
    logger.info(f"recompute dt every {new_timestep_cadence} iterations")
    setup.print_model_parameters(newlines=True, logger=main_logger)

    def grab_state():
        """
        Collect items from the driver and solver state, as well as run
        details, sufficient for restarts and post processing.
        """
        return DriverState(
            iteration=iteration,
            driver=driver,
            mesh=mesh,
            timeseries=timeseries,
            event_states=event_states,
            solver=solver,
            setup=setup,
            cfl_number=cfl_number,
            timestep_dt=dt,
        )

    while True:
        siml_time = solver.time
        user_time = siml_time / reference_time

        """
        Run the main simulation loop. Iterations are grouped according the
        the fold parameter. Side effects including the iteration message are
        performed between fold boundaries.
        """

        for name, event in driver.events.items():
            state = event_states[name]
            if event_states[name].is_due(user_time, event):
                event_states[name] = state.next(user_time, event)
                yield name, state.number, grab_state()

        if end_time is not None and user_time >= end_time:
            break

        with measure_time(mode) as fold_time:
            for _ in range(fold):
                if dt is None or (iteration % new_timestep_cadence == 0):
                    dx = mesh.min_spacing(siml_time)
                    dt = dx / solver.maximum_wavespeed() * cfl_number
                solver.advance(dt)
                iteration += 1
                
        Mzps = mesh.num_total_zones / fold_time() * 1e-6 * fold
        main_logger.info(
            f"[{iteration:04d}] t={user_time:0.3f} dt={dt:.3e} Mzps={Mzps:.3f}"
        )

    yield "end", None, grab_state()


def run(setup_name, quiet=True, **kwargs):
    """
    Run a simulation with no side-effects, and return the final state.

    This function is intended for use by scripts that run a simulation and
    inspect the output in-memory, or otherwise handle archiving the final
    result themselves. Event monitoring is not supported. If `quiet=True`
    (default) then logging is suppressed.
    """
    import sailfish.setups

    if "events" in kwargs:
        raise ValueError("events are not supported")

    driver = DriverArgs(setup_name=setup_name, **kwargs)

    if not quiet:
        init_logging()

    load_user_config()

    return next(simulate(driver))[2]


def init_logging():
    """
    Convenience method to enable logging to standard output.

    This function is called from the `main` entry point (i.e. when sailfish is
    used as a command line tool). However when sailfish is used as a library,
    logging is not enabled by default (Python's `logging` module recommends
    that libraries should not install any event handlers on the root logger).
    This function enables a sensible logging configuration, so if the calling
    application or script is not particular about how logging should take
    place, but it doesn't want the driver to be silent, then invoking this
    function will do it for you. Note this function is also invoked by the
    `run` function if :code:`quiet=False` is passed to it.
    """
    from sys import stdout
    from logging import StreamHandler, Formatter, getLogger, INFO

    class RunFormatter(Formatter):
        def format(self, record):
            name = record.name.replace("sailfish.", "")

            if name == "main_logger":
                return f"{record.msg}"
            if record.levelno <= 20:
                return f"[{name}] {record.msg}"
            else:
                return f"[{name}:{record.levelname.lower()}] {record.msg}"

    handler = StreamHandler(stdout)
    handler.setFormatter(RunFormatter())

    root_logger = getLogger()
    root_logger.addHandler(handler)
    root_logger.setLevel(INFO)


def load_user_config():
    """
    Initialize user extensions: setups and solvers outside the main codebase.

    This function is called by the :pyobj:`main` entry point and the
    :pyobj:`run` API function to load custom setups provided by the user.
    Extensions are defined in the `extensions` section of the .sailfish
    file. The .sailfish file is loaded from the current working directory.
    """
    from configparser import ConfigParser, ParsingError
    from importlib import import_module

    try:
        config = ConfigParser()
        config.read(".sailfish")

        try:
            for setup_extension in config["extensions"]["setups"].split():
                import_module(setup_extension)
        except KeyError:
            pass

        try:
            for solver_extension in config["extensions"]["solvers"].split():
                register_solver_extension(solver_extension)
        except KeyError:
            pass

        try:
            for key, val in config["build"].items():
                user_build_config[key] = val
        except KeyError:
            pass

    except ModuleNotFoundError as e:
        raise ExtensionError(e)

    except ParsingError as e:
        raise ConfigurationError(e)


def main():
    """
    General-purpose command line interface.
    """
    import argparse
    import sailfish.setups

    class MakeDict(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            setattr(namespace, self.dest, dict(values))

    def add_dict_entry(key):
        class AddDictEntry(argparse.Action):
            def __call__(self, parser, namespace, values, option_string=None):
                getattr(namespace, self.dest)[key] = values

        return AddDictEntry

    parser = argparse.ArgumentParser(
        prog="sailfish",
        usage=argparse.SUPPRESS,
        description="sailfish is a gpu-accelerated astrophysical gasdynamics code",
    )
    parser.add_argument(
        "command",
        nargs="?",
        help="setup name or restart file (if directory, then load newest checkpoint)",
    )
    parser.add_argument(
        "--describe",
        action="store_true",
        help="print a description of the setup and exit",
    )
    parser.add_argument(
        "--resolution",
        "-n",
        metavar="N",
        type=int,
        help="grid resolution",
    )
    parser.add_argument(
        "--patches",
        metavar="N",
        type=int,
        dest="num_patches",
        help="number of patches for domain decomposition",
    )
    parser.add_argument(
        "--cfl",
        dest="cfl_number",
        metavar="C",
        type=float,
        help="CFL parameter",
    )
    parser.add_argument(
        "--fold",
        "-f",
        metavar="F",
        type=int,
        help="iterations between messages and side effects",
    )
    parser.add_argument(
        "--new-timestep-cadence",
        metavar="C",
        type=int,
        help="iterations between recomputing the timestep dt",
    )
    parser.add_argument(
        "--events",
        nargs="*",
        metavar="E=V",
        type=keyed_event,
        action=MakeDict,
        default=dict(),
        help="a sequence of events and recurrence rules to be emitted",
    )
    parser.add_argument(
        "--restart-dir",
        action="store_true",
        help="the command argument is a directory; restart from newest checkpoint therein",
    )
    parser.add_argument(
        "--final-chkpt",
        action="store_true",
        help="write chkpt.final.pk on exit",
    )
    parser.add_argument(
        "--checkpoint",
        "-c",
        metavar="C",
        type=Recurrence.from_str,
        action=add_dict_entry("checkpoint"),
        dest="events",
        help="checkpoint recurrence [<delta>|<log:mul>]",
    )
    parser.add_argument(
        "--timeseries",
        "-t",
        metavar="T",
        type=Recurrence.from_str,
        action=add_dict_entry("timeseries"),
        dest="events",
        help="timeseries recurrence [<delta>|<log:mul>]",
    )
    parser.add_argument(
        "--model",
        nargs="*",
        metavar="K=V",
        type=keyed_value,
        action=MakeDict,
        default=dict(),
        dest="model_parameters",
        help="key-value pairs given as models parameters to the setup",
    )
    parser.add_argument(
        "--solver",
        nargs="*",
        metavar="K=V",
        type=keyed_value,
        action=MakeDict,
        default=dict(),
        dest="solver_options",
        help="key-value pairs passed as options to the solver",
    )
    parser.add_argument(
        "--outdir",
        "-o",
        metavar="D",
        type=str,
        dest="output_directory",
        help="directory where checkpoints are written",
    )
    parser.add_argument(
        "--end-time",
        "-e",
        metavar="T",
        type=float,
        help="when to end the simulation",
    )
    parser.add_argument(
        "--event-handlers-file",
        metavar="F",
        type=str,
        help="path to a module defining a get_event_handlers function",
    )
    parser.add_argument(
        "--verbose-output",
        metavar="P",
        type=str,
        default="",
        help="detailed print solver structs [physics,options]",
    )
    exec_group = parser.add_mutually_exclusive_group()
    exec_group.add_argument(
        "--mode",
        dest="execution_mode",
        choices=["cpu", "omp", "gpu"],
        help="execution mode",
    )
    exec_group.add_argument(
        "--use-omp",
        "-p",
        dest="execution_mode",
        action="store_const",
        const="omp",
        help="multi-core with OpenMP",
    )
    exec_group.add_argument(
        "--use-gpu",
        "-g",
        dest="execution_mode",
        action="store_const",
        const="gpu",
        help="gpu acceleration",
    )

    try:
        init_logging()
        load_user_config()

        args = parser.parse_args()

        if args.describe and args.command is not None:
            setup_name = args.command.split(":")[0]
            Setup.find_setup_class(setup_name).describe_class()

        elif args.command is None:
            print("specify setup:")
            for setup in Setup.__subclasses__():
                print(f"    {setup.dash_case_class_name()}")

        else:
            driver = DriverArgs.from_namespace(args)
            outdir = (
                args.output_directory
                or (driver.chkpt_file and os.path.dirname(driver.chkpt_file))
                or "."
            )

            if args.event_handlers_file is not None:
                import importlib.util

                spec = importlib.util.spec_from_file_location(
                    "events_handler_module", args.event_handlers_file
                )
                events_handler_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(events_handler_module)
                events_dict = events_handler_module.get_event_handlers()
            else:
                events_dict = dict()

            for name, number, state in simulate(driver):
                if name == "timeseries":
                    append_timeseries(state)
                elif name == "checkpoint":
                    write_checkpoint(number, outdir, state)
                elif name == "end":
                    if args.final_chkpt:
                        write_checkpoint("final", outdir, state)
                elif name in events_dict:
                    events_dict[name](number, outdir, state, logger)
                else:
                    logger.warning(f"unrecognized event {name}")

    except ConfigurationError as e:
        print(f"bad configuration: {e}")

    except ExtensionError as e:
        print(f"bad extension: {e}")

    except SetupError as e:
        print(f"setup error: {e}")

    except ParseRecurrenceError as e:
        print(f"parse error: {e}")

    except SolverInitializationError as e:
        print(f"solver initialization error: {e}")

    except OSError as e:
        print(f"file system error: {e}")

    except ModuleNotFoundError as e:
        print(f"unsatisfied dependency: {e}")

    except KeyboardInterrupt:
        print("")
