from contextlib import contextmanager
from time import perf_counter
from numpy.typing import NDArray


@contextmanager
def measure_time() -> float:
    start = perf_counter()
    yield lambda: perf_counter() - start


def main():
    from argparse import ArgumentParser
    from new_kernels import configure_kernel_module

    parser = ArgumentParser()
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="verbose output from extension compile stages",
    )
    parser.add_argument(
        "--mode",
        dest="exec_mode",
        default="cpu",
        choices=["cpu", "gpu"],
        help="execution mode",
    )
    parser.add_argument(
        "--resolution",
        "-n",
        metavar="N",
        type=int,
        default=None,
        help="grid resolution",
    )
    parser.add_argument(
        "--plm",
        action="store_true",
        help="use PLM reconstruction for second-order in space",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="show a plot after the run",
    )
    args = parser.parse_args()
    configure_kernel_module(verbose=args.verbose, default_exec_mode=args.exec_mode)

    if args.exec_mode == "cpu":
        from numpy import (
            array,
            linspace,
            zeros,
            zeros_like,
            diff,
            meshgrid,
            logical_not,
        )
    if args.exec_mode == "gpu":
        from cupy import (
            array,
            linspace,
            zeros,
            zeros_like,
            diff,
            meshgrid,
            logical_not,
        )

    if args.dim == 1:
        from hydro_euler import EulerEquations
        from gradient_estimation import plm_gradient_1d, extrapolate

        hydro = EulerEquations(dim=1, gamma_law_index=5.0 / 3.0)
        num_zones = args.resolution or 100000
        dx = 1.0 / num_zones
        fold = 100
        dt = dx * 1e-1
        p = zeros((num_zones, hydro.ncons))
        u = zeros_like(p)
        g = zeros_like(p)
        pp = zeros_like(p)
        pm = zeros_like(p)
        fhat = zeros((num_zones - 1, hydro.ncons))

        p[: num_zones // 2, :] = array([1.0] + hydro.dim * [0.0] + [1.0])
        p[num_zones // 2 :, :] = array([0.1] + hydro.dim * [0.0] + [0.125])
        t = 0.0
        n = 0

        while t < 0.1:
            with measure_time() as fold_time:
                for _ in range(fold):
                    if args.plm:
                        plm_gradient_1d(p, g, 1.5)
                        extrapolate(p, g, pm, pp)
                        pl = pp[:-1]
                        pr = pm[+1:]
                    else:
                        pl = p[:-1]
                        pr = p[+1:]

                    hydro.riemann_hlle(pl, pr, fhat, 1)
                    hydro.prim_to_cons(p, u)
                    u[1:-1] -= diff(fhat, axis=0) * (dt / dx)
                    t += dt
                    n += 1
                    hydro.cons_to_prim(u, p)

            kzps = num_zones / fold_time() * 1e-3 * fold
            print(f"[{n:04d}]: t={t:.4f} Mzps={kzps * 1e-3:.3f}")

        if args.plot:
            from matplotlib import pyplot as plt

            plt.plot(p[:, 0])
            plt.show()

    elif args.dim == 2:
        from hydro_euler import EulerEquations
        from gradient_estimation import plm_gradient_2d, extrapolate

        hydro = EulerEquations(dim=2, gamma_law_index=5.0 / 3.0)
        num_zones = args.resolution or 100
        dx = 1.0 / num_zones
        fold = 10
        dt = dx * 1e-1
        p = zeros((num_zones, num_zones, hydro.ncons))
        u = zeros_like(p)
        gx = zeros_like(p)
        gy = zeros_like(p)
        pp = zeros_like(p)
        pm = zeros_like(p)
        fhat = zeros((num_zones - 1, num_zones, hydro.ncons))
        ghat = zeros((num_zones, num_zones - 1, hydro.ncons))

        xv = linspace(-0.5, 0.5, num_zones + 1)
        yv = linspace(-0.5, 0.5, num_zones + 1)
        xc = 0.5 * (xv[1:] + xv[:-1])
        yc = 0.5 * (xv[1:] + xv[:-1])
        X, Y = meshgrid(xc, yc)

        disk = (X**2 + Y**2) ** 0.5 < 0.1
        fisk = logical_not(disk)
        p[disk, 0] = 1.000
        p[fisk, 0] = 0.100
        p[disk, 3] = 1.000
        p[fisk, 3] = 0.125

        t = 0.0
        n = 0

        while t < 0.1:
            with measure_time() as fold_time:
                for _ in range(fold):
                    plm_gradient_2d(p, gx, gy, 1.5)

                    extrapolate(p, gx, pm, pp)
                    pl = pp[:-1, :].copy()
                    pr = pm[+1:, :].copy()
                    hydro.riemann_hlle(pl, pr, fhat, 1)

                    extrapolate(p, gy, pm, pp)
                    pl = pp[:, :-1].copy()
                    pr = pm[:, +1:].copy()
                    hydro.riemann_hlle(pl, pr, ghat, 2)

                    hydro.prim_to_cons(p, u)
                    u[1:-1, :] -= diff(fhat, axis=0) * (dt / dx)
                    u[:, 1:-1] -= diff(ghat, axis=1) * (dt / dx)
                    hydro.cons_to_prim(u, p)

                    t += dt
                    n += 1

            kzps = num_zones * num_zones / fold_time() * 1e-3 * fold
            print(f"[{n:04d}]: t={t:.4f} Mzps={kzps * 1e-3:.3f}")

        if args.plot:
            from matplotlib import pyplot as plt

            plt.imshow(p[:, :, 0])
            plt.colorbar()
            plt.show()


main()
