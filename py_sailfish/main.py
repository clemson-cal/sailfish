#!/usr/bin/env python3


class ConfigurationError(Exception):
    """An invalid runtime configuration"""


class RecurringTask:
    def __init__(self, name):
        self.name = name
        self.interval = None
        self.last_time = None
        self.number = 0

    @classmethod
    def from_dict(cls, state):
        task = RecurringTask(state["name"])
        task.interval = state["interval"]
        task.last_time = state["last_time"]
        task.number = state["number"]
        return task

    def next_time(self, time):
        if self.last_time is None:
            return time
        else:
            return self.last_time + self.interval

    def is_due(self, time):
        return time >= self.next_time(time)

    def next(self, time):
        self.last_time = self.next_time(time)
        self.number += 1


def first_rest(a):
    """
    Return the first element of a list or tuple, followed by the rest as a
    possibly empty list.
    """
    if len(a) > 1:
        return a[0], a[1:]
    else:
        return a[0], []


def first_not_none(*args):
    for arg in args:
        if arg is not None:
            return arg


def update_if_none(new_dict, old_dict, frozen=[]):
    """
    Like `dict.update`, except key-value pairs in `old_dict` are only used to
    add / overwrite values in `new_dict` if they are `None` or missing.
    """
    for key in old_dict:
        old_val = old_dict.get(key)
        new_val = new_dict.get(key)
        if old_val is not None:
            if new_val is None:
                new_dict[key] = old_val
            elif key in frozen and new_val != old_val:
                raise ConfigurationError(f"{key} cannot be changed")


def parse_parameters(item_list):
    parameters = dict()
    for item in item_list:
        try:
            key, val = item.split("=")
            parameters[key] = eval(val)
        except NameError:
            raise ConfigurationError(f"badly formed model parameter {item}")
    return parameters


def write_checkpoint(number, logger=None, **kwargs):
    import pickle

    with open(f"chkpt.{number:04d}.pk", "wb") as chkpt:
        if logger is not None:
            logger.info(f"write checkpoint {chkpt.name}")
        pickle.dump(kwargs, chkpt)


def load_checkpoint(chkpt_name):
    import pickle

    try:
        with open(chkpt_name, "rb") as file:
            return pickle.load(file)
    except FileNotFoundError:
        raise ConfigurationError(f"could not open checkpoint file {chkpt_name}")


def initial_condition(setup, num_zones, domain):
    import numpy as np

    xcells = np.linspace(domain[0], domain[1], num_zones)
    primitive = np.zeros([num_zones, 4])

    for x, p in zip(xcells, primitive):
        setup.initial_primitive(x, p)

    return primitive


def main(**kwargs):
    from time import perf_counter
    from logging import getLogger
    from sailfish.solvers import srhd_1d
    from sailfish.setup import Setup
    from sailfish import system

    # Initialize and log state in the the system module. The build system
    # influences JIT-compiled module code. Currently the build parameters are
    # inferred from the platform (Linux or MacOS), but in the future these
    # should also be extensible by a system-specific rc-style configuration
    # file.
    system.configure_build()
    system.log_system_info(kwargs["mode"] or "cpu")

    logger = getLogger("driver")
    setup_or_checkpoint, parameter_list = first_rest(kwargs["command"].split(":"))

    if setup_or_checkpoint.endswith(".pk"):
        # Load driver state from a checkpoint file. The setup model parameters
        # are updated with any items given on the command line after the setup
        # name. All command line arguments are also restorted from the
        # previous session, but are updated with the `kwargs` variable (command
        # line argument given for this session), except for "frozen"
        # arguments.
        chkpt_name = setup_or_checkpoint
        logger.info(f"load checkpoint {chkpt_name}")

        chkpt = load_checkpoint(chkpt_name)
        setup_name = chkpt["setup_name"]
        parameters = parse_parameters(parameter_list)

        SetupCls = Setup.find_setup_class(setup_name)
        frozen_params = list(SetupCls.immutable_parameter_keys())

        update_if_none(parameters, chkpt["parameters"], frozen=frozen_params)
        update_if_none(kwargs, chkpt["driver_args"], frozen=["resolution"])

        setup = SetupCls(**parameters)
        iteration = chkpt["iteration"]
        time = chkpt["time"]
        checkpoint_task = RecurringTask.from_dict(chkpt["tasks"]["checkpoint"])
        initial = chkpt["primitive"]
        del parameters

    else:
        """
        Generate an initial driver state from command line arguments, model
        parametrs, and a setup instance.
        """
        if kwargs["resolution"] is None:
            kwargs["resolution"] = 10000

        logger.info(f"generate initial data for setup {setup_or_checkpoint}")

        setup_name = setup_or_checkpoint
        parameters = parse_parameters(parameter_list)
        setup = Setup.find_setup_class(setup_name)(**parameters)
        iteration = 0
        time = 0.0
        checkpoint_task = RecurringTask("checkpoint")
        initial = initial_condition(setup, kwargs["resolution"], setup.domain)
        del parameters

    num_zones = kwargs["resolution"]
    mode = kwargs["mode"] or "cpu"
    fold = kwargs["fold"] or 10
    cfl_number = kwargs["cfl_number"] or 0.6
    dx = (setup.domain[1] - setup.domain[0]) / num_zones
    dt = dx * cfl_number
    checkpoint_task.interval = kwargs["checkpoint"] or 0.1
    end_time = first_not_none(kwargs["end_time"], setup.end_time, float("inf"))

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
    logger.info(f"run until t={end_time}")
    logger.info(f"CFL number is {cfl_number}")
    logger.info(f"timestep is {dt}")

    setup.print_model_parameters()

    def checkpoint():
        checkpoint_task.next(solver.time)
        write_checkpoint(
            checkpoint_task.number - 1,
            logger=logger,
            time=solver.time,
            iteration=iteration,
            primitive=solver.primitive,
            driver_args=kwargs,
            tasks=dict(checkpoint=vars(checkpoint_task)),
            parameters=setup.model_parameter_dict,
            setup_name=setup_name,
            domain=setup.domain,
        )

    while end_time is None or end_time > solver.time:
        # Run the main simulation loop. Iterations are grouped according the
        # the fold parameter. Side effects including the iteration message are
        # performed between fold boundaries.

        if checkpoint_task.is_due(solver.time):
            checkpoint()

        start = perf_counter()
        for _ in range(fold):
            solver.new_timestep()
            solver.advance_rk(0.0, dt)
            solver.advance_rk(0.5, dt)
            iteration += 1
        stop = perf_counter()
        Mzps = num_zones / (stop - start) * 1e-6 * fold

        print(f"[{iteration:04d}] t={solver.time:0.3f} Mzps={Mzps:.3f}")

    checkpoint()


if __name__ == "__main__":
    import argparse
    import logging
    from sailfish.setup import Setup, SetupError

    logging.basicConfig(level=logging.INFO, format="-> %(name)s: %(message)s")

    parser = argparse.ArgumentParser(
        prog="sailfish",
        usage="%(prog)s <command> [options]",
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
        type=float,
        help="how often to write a checkpoint file",
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
        help="execution mode",
        choices=["cpu", "omp", "gpu"],
    )
    exec_group.add_argument(
        "--use-omp",
        "-p",
        dest="mode",
        action="store_const",
        const="omp",
        help="multi-core with OpenMP",
    )
    exec_group.add_argument(
        "--use-gpu",
        "-g",
        dest="mode",
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
            main(**vars(args))

    except ConfigurationError as e:
        print(f"bad configuration: {e}")

    except ModuleNotFoundError as e:
        print(f"unsatisfied dependency: {e}")

    except SetupError as e:
        print(f"error: {e}")

    except KeyboardInterrupt:
        print("")
