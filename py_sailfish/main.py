import logging
import argparse
import platform
import time
import numpy as np
logging.basicConfig(level=logging.INFO, format='[sailfish] %(message)s')


def initial_condition(xcells):
    primitive = np.zeros([xcells.size, 4])
    for x, p in zip(xcells, primitive):
        if x < 0.5:
            p[0] = 1.0
            p[2] = 1.0
        else:
            p[0] = 0.1
            p[2] = 0.125
    return primitive


def main():
    from sailfish.solvers import srhd_1d

    parser = argparse.ArgumentParser()
    exec_group = parser.add_mutually_exclusive_group()
    exec_group.add_argument('--use-cpu', action='store_true', default=False, help='single-core (default)')
    exec_group.add_argument('--use-omp', '-p', action='store_true', default=False, help='multi-core with OpenMP')
    exec_group.add_argument('--use-gpu', '-g', action='store_true', default=False, help='gpu acceleration')
    parser.add_argument('--resolution', '-n', metavar='N', default=10000, type=int, help='Grid resolution')
    parser.add_argument('--fold', '-f', metavar='F', default=10, type=int, help='iterations between messages and side effects')

    args = parser.parse_args()
    mode = ['cpu', 'omp', 'gpu'][[args.use_cpu, args.use_omp, args.use_gpu, True].index(True) % 3]
    
    num_zones = args.resolution
    fold = args.fold
    cfl_number = 0.6
    dt = 1.0 / num_zones * cfl_number
    n = 0

    xcells = np.linspace(0.0, 1.0, num_zones)
    solver = srhd_1d.Solver(initial_condition(xcells), mode=mode)

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
        main()
    except KeyboardInterrupt:
        print('')
        pass
