import logging
import argparse
import platform
import time

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


def write_checkpoint(number, logger=None, **kwargs):
    import pickle

    with open(f"chkpt.{number:04d}.pk", "wb") as chkpt:
        if logger is not None:
            logger.info(f"write checkpoint {chkpt.name}")
        pickle.dump(kwargs, chkpt)


def initial_condition(xcells):
    import numpy as np

    primitive = np.zeros([xcells.size, 4])
    for x, p in zip(xcells, primitive):
        if x < 0.5:
            p[0] = 1.0
            p[2] = 1.0
        else:
            p[0] = 0.1
            p[2] = 0.125
    return primitive


def main(args):
    from sailfish.solvers import srhd_1d
    from sailfish import system
    import numpy as np

    logger = logging.getLogger("driver")
    mode = args.mode
    system.log_system_info(mode)
    system.configure_build()

    num_zones = args.resolution
    fold = args.fold
    cfl_number = 0.6
    dt = 1.0 / num_zones * cfl_number
    n = 0
    checkpoint_task = RecurringTask("checkpoint", args.checkpoint)

    logger.info("generate initial data")
    xcells = np.linspace(0.0, 1.0, num_zones)
    solver = srhd_1d.Solver(initial_condition(xcells), mode=mode)
    logger.info("start simulation")

    def checkpoint():
        write_checkpoint(
            checkpoint_task.number,
            logger=logger,
            time=solver.time,
            iteration=n,
            primitive=solver.primitive,
        )

    while args.end_time is None or args.end_time > solver.time:
        if checkpoint_task.is_due(solver.time):
            checkpoint()
            checkpoint_task.next(solver.time)
        start = time.perf_counter()
        for _ in range(fold):
            solver.new_timestep()
            solver.advance_rk(0.0, dt)
            solver.advance_rk(0.5, dt)
            n += 1
        stop = time.perf_counter()
        Mzps = num_zones / (stop - start) * 1e-6 * fold

        print(f"[{n:04d}] t={solver.time:0.3f} Mzps={Mzps:.3f}")

    checkpoint()


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        exec_group = parser.add_mutually_exclusive_group()
        exec_group.add_argument(
            "--mode",
            default="cpu",
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
            "--resolution",
            "-n",
            metavar="N",
            default=10000,
            type=int,
            help="grid resolution",
        )
        parser.add_argument(
            "--fold",
            "-f",
            metavar="F",
            default=10,
            type=int,
            help="iterations between messages and side effects",
        )
        parser.add_argument(
            "--checkpoint",
            "-c",
            metavar="C",
            default=1.0,
            type=float,
            help="how often to write a checkpoint file",
        )
        parser.add_argument(
            "--end-time",
            "-e",
            metavar="E",
            default=None,
            type=float,
            help="when to end the simulation",
        )
        args = parser.parse_args()
        main(args)

    except KeyboardInterrupt:
        print("")
        pass

# except Exception as e:
#     print(e)
