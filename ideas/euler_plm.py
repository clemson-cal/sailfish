from sys import stdout
from argparse import ArgumentParser
from loguru import logger
from numpy import linspace, meshgrid, zeros, logical_not
from reporting import configure_logger, terminal, add_logging_arguments, iteration_msg
from kernels import configure_kernel_module, perf_time_sequence
from hydro_euler import EulerEquations
from gradient_estimation import plm_gradient_1d, plm_gradient_2d, extrapolate
import fmr_grid


def update_prim_1d(p, hydro, dt, dx, xp, plm=False):
    """
    One-dimensional update function.
    """
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


def update_prim_2d(p, hydro, dt: float, spacing: tuple, xp):
    """
    Two-dimensional update function.
    """
    ni, nj, nfields = p.shape
    dx, dy = spacing
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
    u[:, 1:-1] -= xp.diff(ghat, axis=1) * (dt / dy)
    hydro.cons_to_prim(u, p)


def godunov_fluxes_2d(p, fhat, ghat, hydro, spacing: tuple, xp):
    gx = xp.empty_like(p)
    gy = xp.empty_like(p)
    pp = xp.empty_like(p)
    pm = xp.empty_like(p)
    pli = xp.empty_like(p[:-1, :])
    pri = xp.empty_like(p[+1:, :])
    plj = xp.empty_like(p[:, :-1:])
    prj = xp.empty_like(p[:, +1:])
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


def transmit_flux(p, fhat, ghat, hydro, dt: float, spacing: tuple, xp):
    dx, dy = spacing
    u = xp.empty_like(p)
    hydro.prim_to_cons(p, u)
    u[1:-1, :] -= xp.diff(fhat, axis=0) * (dt / dx)
    u[:, 1:-1] -= xp.diff(ghat, axis=1) * (dt / dy)
    hydro.cons_to_prim(u, p)


def cell_centers_1d(ni):
    from numpy import linspace

    xv = linspace(0.0, 1.0, ni)
    xc = 0.5 * (xv[1:] + xv[:-1])
    return xc


def patch_spacing(index, nz, np):
    level, (i, j) = index
    dx = 1.0 / np / nz / (1 << level)
    dy = 1.0 / np / nz / (1 << level)
    return dx, dy


def patch_extent(index, nz, np):
    level, (i, j) = index
    dx = 1.0 / np / (1 << level)
    dy = 1.0 / np / (1 << level)
    x0 = -0.5 + (i + 0) * dx
    x1 = -0.5 + (i + 1) * dx
    y0 = -0.5 + (j + 0) * dy
    y1 = -0.5 + (j + 1) * dy
    return (x0, x1), (y0, y1)


def cell_centers_2d(index, nz, np):

    (x0, x1), (y0, y1) = patch_extent(index, nz, np)
    ni = nz
    nj = nz
    ddx = (x1 - x0) / ni
    ddy = (y1 - y0) / nj
    xv = linspace(x0 - 2 * ddx, x1 + 2 * ddy, ni + 5)
    yv = linspace(y0 - 2 * ddx, y1 + 2 * ddy, nj + 5)
    xc = 0.5 * (xv[1:] + xv[:-1])
    yc = 0.5 * (yv[1:] + yv[:-1])
    return meshgrid(xc, yc, indexing="ij")


def grid_of_patches(ni_patches, nj_patches):
    for i in range(ni_patches):
        for j in range(nj_patches):
            yield i, j


def copy_guard_zones(grid):
    for level, (i, j) in grid:
        cc = grid.get((level, (i, j)))
        lc = grid.get((level, (i - 1, j)), None)
        rc = grid.get((level, (i + 1, j)), None)
        cl = grid.get((level, (i, j - 1)), None)
        cr = grid.get((level, (i, j + 1)), None)

        if lc is not None:
            cc[:+2, 2:-2] = lc[-4:-2, 2:-2]
        if rc is not None:
            cc[-2:, 2:-2] = rc[+2:+4, 2:-2]
        if cl is not None:
            cc[2:-2, :+2] = cl[2:-2, -4:-2]
        if cr is not None:
            cc[2:-2, -2:] = cr[2:-2, +2:+4]


def copy_guard_zones_fmr(grid):
    for index in grid:
        fmr_grid.fill_guard_cl(index, grid)
        fmr_grid.fill_guard_cr(index, grid)
        fmr_grid.fill_guard_lc(index, grid)
        fmr_grid.fill_guard_rc(index, grid)


def correct_flux_fmr(fhat, ghat):
    for index in fhat:
        fmr_grid.correct_flux_cl(index, ghat)
        fmr_grid.correct_flux_cr(index, ghat)
    for index in ghat:
        fmr_grid.correct_flux_lc(index, fhat)
        fmr_grid.correct_flux_rc(index, fhat)


def cylindrical_shocktube(x, y, radius: float = 0.1, pressure: float = 1.0):
    """
    A cylindrical shocktube setup

    ----------
    radius ........ radius of the high-pressure region
    pressure ...... gas pressure inside the cylinder
    """
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


@logger.catch
def main():
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
        "--flux-correction",
        action="store_true",
        help="include the flux correction step for FMR",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="show a plot after the run",
    )
    add_logging_arguments(parser)
    args = parser.parse_args()

    configure_logger(logger, log_level=args.log_level)
    term = terminal(logger)

    configure_kernel_module(default_exec_mode=args.exec_mode)

    # -------------------------------------------------------------------------
    # A 1d evolution scheme that uses a single patch, and a
    # generator-coroutine to cache cache scratch arrays.
    # -------------------------------------------------------------------------
    if args.dim == 1:
        plm_gradient_1d.compile()
        extrapolate.compile()

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

        logger.info("start simulation")

        while t < 0.1:
            update_prim_1d(p, hydro, dt, dx, xp, plm=args.plm)
            t += dt
            n += 1

            if n % args.fold == 0:
                zps = nz / next(perf_timer) * args.fold
                term(iteration_msg(iter=n, time=t, zps=zps))

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
        plm_gradient_2d.compile()
        extrapolate.compile()

        # np = args.patches_per_dim
        # nz = (args.resolution or 100) // np
        np = 2  # number of patches at the root level
        nz = args.resolution  # zones per patch (per dim)
        t = 0.0
        n = 0

        patches = set()
        patches.add((0, (0, 0)))
        patches.add((0, (1, 0)))
        patches.add((0, (0, 1)))
        patches.add((1, (2, 2)))
        patches.add((1, (2, 3)))
        patches.add((1, (3, 2)))
        patches.add((1, (3, 3)))

        patch_spacings = {k: patch_spacing(k, nz, np) for k in patches}
        hydro = EulerEquations(dim=2, gamma_law_index=5.0 / 3.0)

        ng = 2
        cell_arrays = {k: cell_centers_2d(k, nz, np) for k in patches}
        prim_arrays = {k: cylindrical_shocktube(*xy) for k, xy in cell_arrays.items()}
        fhat_arrays = {
            k: zeros((nz + 2 * ng - 1, nz + 2 * ng, hydro.ncons)) for k in patches
        }
        ghat_arrays = {
            k: zeros((nz + 2 * ng, nz + 2 * ng - 1, hydro.ncons)) for k in patches
        }

        if args.exec_mode == "gpu":
            import cupy as xp

            stream_cls = xp.cuda.Stream
            to_host = lambda a: a.get()
        else:
            import numpy as xp
            import contextlib

            stream_cls = contextlib.nullcontext
            to_host = lambda a: a

        streams = dict()

        for k in prim_arrays:
            stream = stream_cls()
            prim_arrays[k] = xp.array(prim_arrays[k])
            streams[k] = stream

        smallest_spacing = min(min(dx, dy) for dx, dy in patch_spacings.values())
        dt = smallest_spacing * 0.1

        logger.info("start simulation")
        perf_timer = perf_time_sequence(mode=args.exec_mode)

        while t < 0.1:
            copy_guard_zones_fmr(prim_arrays)

            if args.flux_correction:
                for k in patches:
                    stream = streams[k]
                    spacing = patch_spacings[k]
                    prim = prim_arrays[k]
                    fhat = fhat_arrays[k]
                    ghat = ghat_arrays[k]
                    with stream:
                        godunov_fluxes_2d(prim, fhat, ghat, hydro, spacing, xp)

                correct_flux_fmr(fhat_arrays, ghat_arrays)

                for k in patches:
                    stream = streams[k]
                    spacing = patch_spacings[k]
                    prim = prim_arrays[k]
                    fhat = fhat_arrays[k]
                    ghat = ghat_arrays[k]
                    with stream:
                        transmit_flux(prim, fhat, ghat, hydro, dt, spacing, xp)
            else:
                for k in patches:
                    stream = streams[k]
                    spacing = patch_spacings[k]
                    prim = prim_arrays[k]
                    with stream:
                        update_prim_2d(prim, hydro, dt, spacing, xp)

            t += dt
            n += 1

            if n % args.fold == 0:
                zps = nz**2 * len(patches) / next(perf_timer) * args.fold
                term(iteration_msg(iter=n, time=t, zps=zps))

        from pickle import dump

        host_prim = {index: to_host(p) for index, p in prim_arrays.items()}
        with open("chkpt.pk", "wb") as outf:
            dump((host_prim, cell_arrays), outf)

        if args.plot:
            from matplotlib import pyplot as plt

            vmin = max(p[2:-2, 2:-2, 0].min() for p in prim_arrays.values())
            vmax = max(p[2:-2, 2:-2, 0].max() for p in prim_arrays.values())

            for k in patches:
                z = prim_arrays[k][..., 0]
                x, y = cell_arrays[k]

                plt.pcolormesh(
                    x[2:-2, 2:-2],
                    y[2:-2, 2:-2],
                    z[2:-2, 2:-2],
                    vmin=vmin,
                    vmax=vmax,
                )
                # (x0, x1), (y0, y1) = patch_extent(k, nz, np)
                # plt.plot([x0, x1], [y0, y0], c="w", lw=0.5)
                # plt.plot([x0, x1], [y1, y1], c="w", lw=0.5)
                # plt.plot([x0, x0], [y0, y1], c="w", lw=0.5)
                # plt.plot([x1, x1], [y0, y1], c="w", lw=0.5)

            plt.colorbar()
            plt.axis("equal")
            plt.show()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
        logger.success("ctrl-c interrupt")
