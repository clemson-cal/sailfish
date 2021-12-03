import os, pickle, pathlib
from typing import NamedTuple
from sailfish.task import Recurrence, ParseRecurrenceError
from sailfish.setup import Setup, SetupError


class ConfigurationError(Exception):
    """An invalid runtime configuration"""


def update_dict_if_none(new_dict, old_dict, frozen=[]):
    """
    Like `dict.update`, except `key=value` pairs in `old_dict` are only used
    to add / overwrite values in `new_dict` if they are `None` or missing.
    """
    if type(new_dict) is dict and type(old_dict) is dict:
        for key in old_dict:
            old_val = old_dict.get(key)
            new_val = new_dict.get(key)
            if old_val is not None:
                if new_val is None:
                    new_dict[key] = old_val
                elif key in frozen and new_val != old_val:
                    raise ConfigurationError(f"{key} cannot be changed")


def first_not_none(*args):
    for arg in args:
        if arg is not None:
            return arg


def update_if_none(new, old, frozen=[]):
    """
    Same as `update_dict_if_none`, except operates on (immutable) named tuple
    instances and returns a new named tuple.
    """
    new_dict = new._asdict()
    old_dict = old._asdict()
    update_dict_if_none(new_dict, old_dict, frozen)
    return type(new)(**new_dict)


def write_checkpoint(number, outdir, logger, **kwargs):
    filename = f"chkpt.{number:04d}.pk"

    if outdir is not None:
        pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)
        filename = os.path.join(outdir, filename)

    with open(filename, "wb") as chkpt:
        if logger is not None:
            logger.info(f"write checkpoint {chkpt.name}")
        pickle.dump(kwargs, chkpt)


def load_checkpoint(chkpt_file):
    try:
        with open(chkpt_file, "rb") as file:
            return pickle.load(file)
    except FileNotFoundError:
        raise ConfigurationError(f"could not open checkpoint file {chkpt_file}")


def initial_condition(setup, num_zones, domain):
    import numpy as np

    xcells = np.linspace(domain[0], domain[1], num_zones)
    primitive = np.zeros([num_zones, 4])

    for x, p in zip(xcells, primitive):
        setup.initial_primitive(x, p)

    return primitive


class DriverArgs(NamedTuple):
    """
    Contains data used by the driver.
    """

    setup_name: str = None
    chkpt_file: str = None
    model_parameters: dict = None
    cfl_number: float = None
    checkpoint_recurrence: Recurrence = None
    end_time: float = None
    execution_mode: str = None
    fold: int = None
    output_directory: str = None
    resolution: int = None

    def from_namespace(args):
        """
        Construct an instance from an argparse-type namespace object.
        """

        def parse_item(item):
            try:
                key, val = item.split("=")
                return key, eval(val)
            except (NameError, ValueError):
                raise ConfigurationError(f"badly formed model parameter {item}")

        driver = DriverArgs(
            **{k: w for k, w in vars(args).items() if k in DriverArgs._fields}
        )
        parts = args.command.split(":")

        if not parts[0].endswith(".pk"):
            setup_name = parts[0]
            chkpt_file = None
        else:
            setup_name = None
            chkpt_file = parts[0]

        try:
            model_parameters = dict(parse_item(a) for a in parts[1:])
        except IndexError:
            model_parameters = dict()

        return driver._replace(
            setup_name=setup_name,
            chkpt_file=chkpt_file,
            model_parameters=model_parameters,
        )

    def updated_parameter_dict(self, old, setup_cls):
        """
        Return an updated parameter dict.

        The old parameter dict is expected to be read from a checkpoint file.
        The setup class is used to determine which of the parameter entries in
        the new and old parameter files need to match, or not be superseded.
        """
        new = self.model_parameters
        update_dict_if_none(new, old, frozen=list(setup_cls.immutable_parameter_keys))
        return new

    @property
    def fresh_setup(self):
        """
        Return true if this is not a restart.
        """
        return self.setup_name is not None


def run(setup_name=None, model_parameters=dict(), driver=None, **kwargs):
    """
    Entry point for running simulations.

    If this function is invoked with a `DriverArgs` instance in `driver`, the
    other arguments are ignored. Otherwise, the driver is created from the
    setup name, model paramters, and keyword arguments.
    """

    from time import perf_counter
    from logging import getLogger, basicConfig, INFO, StreamHandler, Formatter
    from sailfish import system
    from sailfish.setup import Setup
    from sailfish.solvers import srhd_1d
    from sailfish.task import RecurringTask

    # If driver is None, then other arguments are ignored
    if driver is None:
        kwargs.update(dict(setup_name=setup_name, model_parameters=model_parameters))
        driver = DriverArgs(**kwargs)

    """
    Initialize and log state in the the system module. The build system
    influences JIT-compiled module code. Currently the build parameters are
    inferred from the platform (Linux or MacOS), but in the future these
    should also be extensible by a system-specific rc-style configuration
    file.
    """

    system.configure_build()
    system.log_system_info(driver.execution_mode or "cpu")
    logger = getLogger(__name__)
    loop_logger = getLogger("loop_message")

    if driver.fresh_setup:
        """
        Generate an initial driver state from command line arguments, model
        parametrs, and a setup instance.
        """
        logger.info(f"generate initial data for setup {driver.setup_name}")
        setup = Setup.find_setup_class(driver.setup_name)(**driver.model_parameters)
        driver = driver._replace(
            resolution=driver.resolution or setup.default_resolution
        )

        iteration = 0
        time = 0.0
        checkpoint_task = RecurringTask(name="checkpoint")
        initial = initial_condition(setup, driver.resolution, setup.domain)
    else:
        """
        Load driver state from a checkpoint file. The setup model parameters
        are updated with any items given on the command line after the setup
        name. All command line arguments are also restorted from the
        previous session, but are updated with the command line argument
        given for this session, except for "frozen" arguments.
        """
        logger.info(f"load checkpoint {driver.chkpt_file}")
        chkpt = load_checkpoint(driver.chkpt_file)
        setup_cls = Setup.find_setup_class(chkpt["setup_name"])
        driver = update_if_none(driver, chkpt["driver_args"], frozen=["resolution"])
        params = driver.updated_parameter_dict(chkpt["parameters"], setup_cls)
        setup = setup_cls(**params)

        iteration = chkpt["iteration"]
        time = chkpt["time"]
        checkpoint_task = chkpt["tasks"]["checkpoint"]
        initial = chkpt["primitive"]

    chkpt_recurrence = driver.checkpoint_recurrence or Recurrence.from_str(
        setup.default_checkpoint_recurrence
    )
    mode = driver.execution_mode or "cpu"
    fold = driver.fold or 10
    cfl_number = driver.cfl_number or 0.6
    dx = (setup.domain[1] - setup.domain[0]) / driver.resolution
    dt = dx * cfl_number
    end_time = first_not_none(driver.end_time, setup.default_end_time, float("inf"))

    # Construct a solver instance. TODO: the solver should be obtained from
    # the setup instance.
    solver = srhd_1d.Solver(
        initial=initial,
        time=time,
        domain=setup.domain,
        num_patches=1,
        boundary_condition=setup.boundary_condition,
        mode=mode,
    )
    logger.info(f"checkpoint task recurrence is {chkpt_recurrence}")
    logger.info(f"run until t={end_time}")
    logger.info(f"CFL number is {cfl_number}")
    logger.info(f"timestep is {dt}")
    setup.print_model_parameters(newlines=True)

    def grab_state(tasks=dict()):
        """
        Collect items from the driver and solver state, as well as run
        details, sufficient for restarts and post processing.
        """
        return dict(
            iteration=iteration,
            time=solver.time,
            primitive=solver.primitive,
            tasks=tasks,
            driver_args=driver,
            parameters=setup.model_parameter_dict,
            setup_name=setup.dash_case_class_name,
            domain=setup.domain,
        )

    def checkpoint():
        """
        Write a checkpoint file.

        This function has no side effects on the driver state, but it returns
        an updated checkpoint task.
        """
        next_checkpoint_task = checkpoint_task.next(solver.time, chkpt_recurrence)
        write_checkpoint(
            checkpoint_task.number,
            driver.output_directory,
            logger,
            **grab_state(tasks=dict(checkpoint=next_checkpoint_task)),
        )
        return next_checkpoint_task

    while end_time is None or end_time > solver.time:
        """
        Run the main simulation loop. Iterations are grouped according the
        the fold parameter. Side effects including the iteration message are
        performed between fold boundaries.
        """
        if checkpoint_task.is_due(solver.time, chkpt_recurrence):
            checkpoint_task = checkpoint()

        with system.measure_time() as fold_time:
            for _ in range(fold):
                solver.new_timestep()
                solver.advance_rk(0.0, dt)
                solver.advance_rk(0.5, dt)
                iteration += 1

        Mzps = driver.resolution / fold_time() * 1e-6 * fold
        loop_logger.info(f"[{iteration:04d}] t={solver.time:0.3f} Mzps={Mzps:.3f}")

    if checkpoint_task.is_due(float("inf"), chkpt_recurrence):
        checkpoint()

    return grab_state()


def enable_logging():
    from logging import StreamHandler, Formatter, getLogger, INFO

    class RunFormatter(Formatter):
        def format(self, record):
            if record.name == "loop_message":
                return f"{record.msg}"
            else:
                return f"[{record.name.replace('sailfish.', '')}] {record.msg}"

    handler = StreamHandler()
    handler.setFormatter(RunFormatter())

    root_logger = getLogger()
    root_logger.addHandler(handler)
    root_logger.setLevel(INFO)


def main():
    import argparse

    enable_logging()

    parser = argparse.ArgumentParser(
        prog="sailfish",
        # usage="%(prog)s <command> [options]",
        description="gpu-accelerated astrophysical gasdynamics code",
    )
    parser.add_argument(
        "command",
        nargs="?",
        help="setup name or restart file",
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
        "--checkpoint",
        "-c",
        metavar="C",
        type=Recurrence.from_str,
        dest="checkpoint_recurrence",
        help="checkpoint recurrence [never|once|twice|<delta>|<log:mul>] ",
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
        metavar="E",
        type=float,
        help="when to end the simulation",
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
        args = parser.parse_args()

        if args.describe and args.command is not None:
            setup_name = args.command.split(":")[0]
            Setup.find_setup_class(setup_name).describe_class()

        elif args.command is None:
            print("specify setup:")
            for setup in Setup.__subclasses__():
                print(f"    {setup.dash_case_class_name}")

        else:
            run(driver=DriverArgs.from_namespace(args))

    except ConfigurationError as e:
        print(f"bad configuration: {e}")

    except SetupError as e:
        print(f"setup error: {e}")

    except ParseRecurrenceError as e:
        print(f"parse error: {e}")

    except OSError as e:
        print(f"file system error: {e}")

    except ModuleNotFoundError as e:
        print(f"unsatisfied dependency: {e}")

    except KeyboardInterrupt:
        print("")
