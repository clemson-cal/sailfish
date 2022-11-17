def update_prim_1d(p, hydro, dt, dx, xp, plm=False):
    """
    One-dimensional update function.
    """
    from gradient_estimation import plm_gradient_1d, extrapolate

    ni, nfields = p.shape
    u = xp.empty_like(p)
    g = xp.empty_like(p)
    pp = xp.empty_like(p)
    pm = xp.empty_like(p)
    fhat = xp.empty((ni - 1, nfields))
    g[...] = 0.0

    if plm:
        plm_gradient_1d(p, g, 1.5)
        extrapolate(p, g, pm, pp)
        pl = pp[:-1]
        pr = pm[+1:]
    else:
        pl = p[:-1]
        pr = p[+1:]

    hydro.riemann_hlle(pl, pr, fhat, 1)
    hydro.prim_to_cons(p, u)
    u[1:-1] -= xp.diff(fhat, axis=0) * (dt / dx)
    hydro.cons_to_prim(u, p)


def update_prim_2d(p, hydro, dt, dx, xp):
    from gradient_estimation import plm_gradient_2d, extrapolate

    ni, nj, nfields = p.shape
    u = xp.empty_like(p)
    gx = xp.empty_like(p)
    gy = xp.empty_like(p)
    pp = xp.empty_like(p)
    pm = xp.empty_like(p)
    pli = xp.empty_like(p[:-1, :])
    pri = xp.empty_like(p[+1:, :])
    plj = xp.empty_like(p[:, :-1:])
    prj = xp.empty_like(p[:, +1:])
    fhat = xp.empty((ni - 1, nj, nfields))
    ghat = xp.empty((ni, nj - 1, nfields))

    gx[...] = 0.0
    gy[...] = 0.0

    plm_gradient_2d(p, gx, gy, 1.5)

    extrapolate(p, gx, pm, pp)
    pli[...] = pp[:-1, :]
    pri[...] = pm[+1:, :]
    hydro.riemann_hlle(pli, pri, fhat, 1)

    extrapolate(p, gy, pm, pp)
    plj[...] = pp[:, :-1]
    prj[...] = pm[:, +1:]
    hydro.riemann_hlle(plj, prj, ghat, 2)

    hydro.prim_to_cons(p, u)
    u[1:-1, :] -= xp.diff(fhat, axis=0) * (dt / dx)
    u[:, 1:-1] -= xp.diff(ghat, axis=1) * (dt / dx)
    hydro.cons_to_prim(u, p)


def cell_centers_1d(ni):
    from numpy import linspace

    xv = linspace(0.0, 1.0, ni)
    xc = 0.5 * (xv[1:] + xv[:-1])
    return xc


def cell_centers_2d(i, j, ni_patches, nj_patches, ni, nj):
    from numpy import linspace, meshgrid

    dx = 1.0 / ni_patches
    dy = 1.0 / nj_patches
    ddx = dx / ni
    ddy = dy / nj
    x0 = -0.5 + (i + 0) * dx
    x1 = -0.5 + (i + 1) * dx
    y0 = -0.5 + (j + 0) * dy
    y1 = -0.5 + (j + 1) * dy
    xv = linspace(x0 - 2 * ddx, x1 + 2 * ddy, ni + 5)
    yv = linspace(y0 - 2 * ddx, y1 + 2 * ddy, nj + 5)
    xc = 0.5 * (xv[1:] + xv[:-1])
    yc = 0.5 * (yv[1:] + yv[:-1])
    return meshgrid(xc, yc, indexing="ij")


def initial_patches(ni_patches, nj_patches):
    for i in range(ni_patches):
        for j in range(nj_patches):
            yield i, j


def copy_guard_zones(grid):
    for i, j in grid:
        cc = grid.get((i, j))
        lc = grid.get((i - 1, j), None)
        rc = grid.get((i + 1, j), None)
        cl = grid.get((i, j - 1), None)
        cr = grid.get((i, j + 1), None)

        if lc is not None:
            cc[:+2, 2:-2] = lc[-4:-2, 2:-2]
        if rc is not None:
            cc[-2:, 2:-2] = rc[+2:+4, 2:-2]
        if cl is not None:
            cc[2:-2, :+2] = cl[2:-2, -4:-2]
        if cr is not None:
            cc[2:-2, -2:] = cr[2:-2, +2:+4]


def cylindrical_shocktube(x, y, radius: float = 0.1, pressure: float = 1.0):
    """
    A cylindrical shocktube setup

    ----------
    radius ........ radius of the high-pressure region
    pressure ...... gas pressure inside the cylinder
    """

    from numpy import zeros, logical_not

    disk = (x**2 + y**2) ** 0.5 < radius
    fisk = logical_not(disk)
    p = zeros(disk.shape + (4,))

    p[disk, 0] = 1.000
    p[fisk, 0] = 0.100
    p[disk, 3] = pressure
    p[fisk, 3] = 0.125
    return p


def linear_shocktube(x):
    """
    A linear shocktube setup
    """

    from numpy import array, zeros, logical_not

    l = x < 0.5
    r = logical_not(l)
    p = zeros(x.shape + (3,))
    p[l, :] = [1.0, 0.0, 1.000]
    p[r, :] = [0.1, 0.0, 0.125]
    return p


def main():
    from sys import stdout
    from argparse import ArgumentParser
    from loguru import logger
    from new_kernels import configure_kernel_module, perf_time_sequence
    from hydro_euler import EulerEquations

    parser = ArgumentParser()
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
        "--fold",
        "-f",
        type=int,
        default=50,
        help="number of iterations between messages",
    )
    parser.add_argument(
        "--patches-per-dim",
        type=int,
        default=1,
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

    parser.add_argument(
        "--log-level",
        default="info",
        choices=["trace", "debug", "info", "success", "warning", "error", "critical"],
        help="log messages at and above this severity level",
    )
    args = parser.parse_args()

    log_format = (
        "<green>{time:MM-DD-YY HH:mm:ss.SS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    loop_str = "<blue><b>{iter:04d}</b></blue> <black>time</black>:{time:.4f} <black>Mzps</black>:{Mzps:.3f}"
    loop_msg = logger.opt(ansi=True).info

    logger.remove()
    logger.add(
        stdout,
        level=args.log_level.upper(),
        format=log_format,
        filter=lambda r: r["level"].name != "INFO",
    )
    logger.add(
        stdout,
        format="{message}",
        filter=lambda r: r["level"].name == "INFO",
    )
    configure_kernel_module(default_exec_mode=args.exec_mode)

    # -------------------------------------------------------------------------
    # A 1d evolution scheme that uses a single patch, and a
    # generator-coroutine to cache cache scratch arrays.
    # -------------------------------------------------------------------------
    if args.dim == 1:
        if args.patches_per_dim != 1:
            raise ValueError("only 1 patch is supported in 1d")

        hydro = EulerEquations(dim=1, gamma_law_index=5.0 / 3.0)
        nz = args.resolution or 100000
        dx = 1.0 / nz
        dt = dx * 1e-1
        x = cell_centers_1d(nz)
        p = linear_shocktube(x)
        t = 0.0
        n = 0

        if args.exec_mode == "gpu":
            import cupy as xp

            to_host = lambda a: a.get()
        else:
            import numpy as xp

            to_host = lambda a: a

        p = xp.array(p)
        perf_timer = perf_time_sequence(mode=args.exec_mode)

        while t < 0.1:
            update_prim_1d(p, hydro, dt, dx, xp, plm=args.plm)
            t += dt
            n += 1

            if n % args.fold == 0:
                Mzps = nz / next(perf_timer) * args.fold * 1e-6
                loop_msg(loop_str.format(iter=n, time=t, Mzps=Mzps))

        p = to_host(p)
        from numpy import save

        save("prim.npy", p)

        if args.plot:
            from matplotlib import pyplot as plt

            plt.plot(p[:, 0])
            plt.show()

    # -------------------------------------------------------------------------
    # A 2d evolution scheme that uses a grid of uniformly refined patches.
    # -------------------------------------------------------------------------
    elif args.dim == 2:
        np = args.patches_per_dim
        nz = (args.resolution or 100) // np
        dx = 1.0 / (np * nz)
        dt = dx * 1e-1
        t = 0.0
        n = 0

        hydro = EulerEquations(dim=2, gamma_law_index=5.0 / 3.0)
        patches = set(initial_patches(np, np))
        cell_arrays = {ij: cell_centers_2d(*ij, np, np, nz, nz) for ij in patches}
        prim_arrays = {ij: cylindrical_shocktube(*xy) for ij, xy in cell_arrays.items()}

        if args.exec_mode == "gpu":
            import cupy as xp

            stream_cls = xp.cuda.Stream
            to_host = lambda a: a.get()
        else:
            import numpy as xp
            import contextlib

            stream_cls = contextlib.nullcontext
            to_host = lambda a: a

        streams = list()

        for ij in prim_arrays:
            stream = stream_cls()
            prim_arrays[ij] = xp.array(prim_arrays[ij])
            streams.append(stream)

        perf_timer = perf_time_sequence(mode=args.exec_mode)

        while t < 0.1:
            copy_guard_zones(prim_arrays)
            for stream, prim in zip(streams, prim_arrays.values()):
                with stream:
                    update_prim_2d(prim, hydro, dt, dx, xp)

            t += dt
            n += 1

            if n % args.fold == 0:
                Mzps = nz**2 * len(patches) / next(perf_timer) * args.fold * 1e-6
                loop_msg(loop_str.format(iter=n, time=t, Mzps=Mzps))

        if args.plot:
            from matplotlib import pyplot as plt

            vmin = max(p[2:-2, 2:-2, 0].min() for p in prim_arrays.values())
            vmax = max(p[2:-2, 2:-2, 0].max() for p in prim_arrays.values())

            for i, j in patches:
                z = prim_arrays[(i, j)][..., 0]
                x, y = cell_arrays[(i, j)]

                plt.pcolormesh(
                    x[2:-2, 2:-2],
                    y[2:-2, 2:-2],
                    z[2:-2, 2:-2],
                    vmin=vmin,
                    vmax=vmax,
                )
            plt.colorbar()
            plt.axis("equal")
            plt.show()


if __name__ == "__main__":
    main()
