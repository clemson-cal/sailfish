import logging
import argparse
import platform
import time

logging.basicConfig(level=logging.INFO, format="-> %(name)-22s %(message)s")


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

    xcells = np.linspace(0.0, 1.0, num_zones)
    solver = srhd_1d.Solver(initial_condition(xcells), mode=mode)

    logger.info("start simulation")

    while solver.time < 0.01:
        start = time.perf_counter()
        for _ in range(fold):
            solver.new_timestep()
            solver.advance_rk(0.0, dt)
            solver.advance_rk(0.5, dt)
            n += 1
        stop = time.perf_counter()
        Mzps = num_zones / (stop - start) * 1e-6 * fold
        print(f"[{n:04d}] t={solver.time:0.3f} Mzps={Mzps:.3f}")

    # np.save('chkpt.0000.npy', solver.primitive)
    # import matplotlib.pyplot as plt
    # plt.plot(xcells, solver.primitive[:,0])
    # plt.show()


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        exec_group = parser.add_mutually_exclusive_group()
        exec_group.add_argument(
            "--use-cpu",
            "-c",
            dest="mode",
            action="store_const",
            const="cpu",
            help="single-core (default)",
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
        exec_group.add_argument(
            "--mode",
            default="cpu",
            help="execution mode",
            choices=["cpu", "omp", "gpu"],
        )
        parser.add_argument(
            "--resolution",
            "-n",
            metavar="N",
            default=10000,
            type=int,
            help="Grid resolution",
        )
        parser.add_argument(
            "--fold",
            "-f",
            metavar="F",
            default=10,
            type=int,
            help="iterations between messages and side effects",
        )
        args = parser.parse_args()
        main(args)

    except KeyboardInterrupt:
        print("")
        pass
