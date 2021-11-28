#!/usr/bin/env python3


class ConfigurationError(Exception):
    """An invalid runtime configuration"""


from abc import ABC, abstractmethod


class Setup(ABC):
    @abstractmethod
    def initial_condition(self, x):
        pass

    @property
    @abstractmethod
    def domain(self):
        pass

    @property
    @abstractmethod
    def boundary_condition(self):
        pass


class Shocktube(Setup):
    def __init__(self, pressure=1.0):
        pass

    def initial_condition(self, x):
        pass

    @property
    def domain(self):
        return [0.0, 1.0]

    @property
    def boundary_condition(self):
        pass


class DensityWave(Setup):
    def __init__(self, pressure=1.0):
        pass

    def initial_condition(self, x):
        pass

    @property
    def domain(self):
        return [0.0, 1.0]

    @property
    def boundary_condition(self):
        pass


class RecurringTask:
    def __init__(self, name_or_dict):
        if type(name_or_dict) is dict:
            self.name = name_or_dict["name"]
            self.interval = name_or_dict["interval"]
            self.last_time = name_or_dict["last_time"]
            self.number = name_or_dict["number"]
        else:
            self.name = name_or_dict
            self.interval = None
            self.last_time = None
            self.number = 0

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
    if len(a) > 1:
        return a[0], a[1:]
    else:
        return a[0], []


def update_namespace(new_args, old_args, frozen=[]):
    for key in vars(new_args):
        new_val = getattr(new_args, key)
        old_val = getattr(old_args, key)
        if old_val is not None:
            if new_val is None:
                setattr(new_args, key, old_val)
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


def initial_condition(num_zones, left_pressure=1.0):
    import math
    import numpy as np

    xcells = np.linspace(0.0, 1.0, num_zones)
    primitive = np.zeros([num_zones, 4])

    for x, p in zip(xcells, primitive):
        # p[0] = 1.0 + np.sin(x * 2 * math.pi) * 0.5
        # p[1] = 0.1
        # p[2] = 1.0
        if x < 0.5:
            p[0] = 1.0
            p[2] = 1.0
        else:
            p[0] = 0.1
            p[2] = 0.125
    return primitive


def main(args):
    import numpy as np
    import pickle
    import logging

    from time import perf_counter
    from sailfish.solvers import srhd_1d
    from sailfish import system

    logger = logging.getLogger("driver")
    setup_or_checkpoint, parameter_list = first_rest(args.command.split(":"))

    if setup_or_checkpoint.endswith(".pk"):
        logger.info("load checkpoint")

        with open(setup_or_checkpoint, "rb") as file:
            chkpt = pickle.load(file)
        update_namespace(args, chkpt["args"], frozen=["resolution"])

        parameters = chkpt["parameters"]
        parameters.update(parse_parameters(parameter_list))
        iteration = chkpt["iteration"]
        time = chkpt["time"]
        checkpoint_task = RecurringTask(chkpt["tasks"]["checkpoint"])
        initial = chkpt["primitive"]

    else:
        logger.info("generate initial data")

        if args.resolution is None:
            args.resolution = 10000

        parameters = parse_parameters(parameter_list)
        iteration = 0
        time = 0.0
        checkpoint_task = RecurringTask("checkpoint")

        try:
            initial = initial_condition(args.resolution, **parameters)
        except TypeError as e:
            raise ConfigurationError(e)

    num_zones = args.resolution
    mode = args.mode or "cpu"
    fold = args.fold or 10
    cfl_number = 0.6
    dt = 1.0 / num_zones * cfl_number
    checkpoint_task.interval = args.checkpoint or 0.1

    system.log_system_info(mode)
    system.configure_build()

    solver = srhd_1d.Solver(
        initial,
        time,
        num_patches=4,
        boundary_condition="outflow",
        mode=mode,
    )
    logger.info("start simulation")

    end_time = args.end_time if args.end_time is not None else 0.4

    def checkpoint():
        write_checkpoint(
            checkpoint_task.number,
            logger=logger,
            time=solver.time,
            iteration=iteration,
            primitive=solver.primitive,
            args=args,
            tasks=dict(checkpoint=vars(checkpoint_task)),
            parameters=parameters,
        )

    while end_time is None or end_time > solver.time:
        if checkpoint_task.is_due(solver.time):
            checkpoint()
            checkpoint_task.next(solver.time)
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


def setups():
    for subclass in Setup.__subclasses__():
        yield "".join(
            ["-" + c.lower() if c.isupper() else c for c in subclass.__name__]
        ).lstrip("-")


if __name__ == "__main__":
    import argparse
    import logging
    import textwrap

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
        nargs="*",
        metavar="",
        # action="store_true",
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
        if args.describe:
            for setup in [args.command] + args.describe:
                if setup is not None:
                    print(f"printing setup description here for {setup}")
        elif args.command is None:
            print("specify setup:")
            for setup in setups():
                print(f"    {setup}")
        else:
            main(args)

    except ConfigurationError as e:
        print(f"bad configuration: {e}")

    except ModuleNotFoundError as e:
        print(f"unsatisfied dependency: {e}")

    except KeyboardInterrupt:
        print("")
