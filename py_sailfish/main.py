import logging

logging.basicConfig(level=logging.INFO, format="-> %(name)s: %(message)s")


class RecurringTask:
    def __init__(self, name, interval):
        self.name = name
        self.interval = interval
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
        return a[0], None


def write_checkpoint(number, logger=None, **kwargs):
    import pickle

    with open(f"chkpt.{number:04d}.pk", "wb") as chkpt:
        if logger is not None:
            logger.info(f"write checkpoint {chkpt.name}")
        pickle.dump(kwargs, chkpt)


def initial_condition(num_zones):
    import numpy as np

    xcells = np.linspace(0.0, 1.0, num_zones)
    primitive = np.zeros([num_zones, 4])

    for x, p in zip(xcells, primitive):
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
    from time import perf_counter
    from sailfish.solvers import srhd_1d
    from sailfish import system

    logger = logging.getLogger("driver")
    mode = args.mode or "cpu"
    system.log_system_info(mode)
    system.configure_build()

    num_zones = args.resolution or 10000
    fold = args.fold or 10
    cfl_number = 0.6
    dt = 1.0 / num_zones * cfl_number
    checkpoint_task = RecurringTask("checkpoint", args.checkpoint or 0.1)

    setup_or_checkpoint, model_parameter_str = first_rest(args.command.split(":"))

    if setup_or_checkpoint.endswith(".pk"):
        logger.info("load checkpoint")
        with open(setup_or_checkpoint, "rb") as file:
            chkpt = pickle.load(file)
        initial = chkpt["primitive"]
        iteration = chkpt["iteration"]
        time = chkpt["time"]
    else:
        logger.info("generate initial data")
        initial = initial_condition(num_zones)
        iteration = 0
        time = 0.0

    solver = srhd_1d.Solver(initial, time, mode=mode)
    logger.info("start simulation")

    end_time = args.end_time or 0.4

    def checkpoint():
        write_checkpoint(
            checkpoint_task.number,
            logger=logger,
            time=solver.time,
            iteration=iteration,
            primitive=solver.primitive,
            args=args,
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


if __name__ == "__main__":
    import argparse

    try:
        parser = argparse.ArgumentParser()
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
        parser.add_argument(
            "command",
            help="setup name or restart file",
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
        main(parser.parse_args())

    except ModuleNotFoundError as e:
        print(f"unsatisfied dependency: {e}")

    except KeyboardInterrupt:
        print("")
