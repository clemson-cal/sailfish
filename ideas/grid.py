from numpy import zeros, linspace, meshgrid, exp


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


def cell_center_coordinates(i, j, ni_patches, nj_patches, ni, nj):
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


def initial_data(x, y):
    z = exp(-(x**2 + y**2) / 0.01)
    z[:+2, :] = 0.0
    z[-2:, :] = 0.0
    z[:, :+2] = 0.0
    z[:, -2:] = 0.0
    return z


if __name__ == "__main__":
    patches = set(initial_patches(8, 8))
    coordinate = {ij: cell_center_coordinates(*ij, 8, 8, 10, 10) for ij in patches}
    primitives = {ij: initial_data(*xy) for ij, xy in coordinate.items()}

    copy_guard_zones(primitives)

    import matplotlib.pyplot as plt

    for i, j in patches:
        if i % 2 == 0 or j % 2 == 0:
            continue

        z = primitives[(i, j)]
        x, y = coordinate[(i, j)]

        plt.pcolormesh(x, y, z, vmin=0, vmax=1)

    plt.axis("equal")
    plt.show()
