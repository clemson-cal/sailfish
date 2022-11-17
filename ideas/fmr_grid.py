from numpy import linspace, meshgrid, exp, zeros
from matplotlib import pyplot as plt


def cell_centers_2d(level, ij, ni, nj):
    i, j = ij
    dx = 1.0 / (1 << level)
    dy = 1.0 / (1 << level)
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


def downsample(a):
    ni, nj = a.shape
    res = zeros((ni // 2, nj // 2))
    res += a[0::2, 0::2]
    res += a[1::2, 0::2]
    res += a[0::2, 1::2]
    res += a[1::2, 1::2]
    res *= 0.25
    return res


def upsample(a):
    ni, nj = a.shape
    res = zeros((ni * 2, nj * 2))
    res[0::2, 0::2] = a
    res[1::2, 0::2] = a
    res[0::2, 1::2] = a
    res[1::2, 1::2] = a
    return res


def fill_guard_rc(index, grid):
    level, (i, j) = index
    ng = 2
    cc = grid[index]
    ni = cc.shape[0] - 2 * ng
    nj = cc.shape[1] - 2 * ng

    try:
        rc = grid[(level, (i + 1, j))]
        cc[-ng:, ng:-ng] = rc[ng : 2 * ng, ng:-ng]
        return
    except KeyError as e:
        print(f"no neighbor {e} for block {index}")

    try:
        rc = grid[(level - 1, (i // 2 + 1, j // 2))]
        if j % 2 == 0:
            cc[-ng:, ng:-ng] = upsample(rc[ng : ng + 1, +ng : ng + nj // 2])
        else:
            cc[-ng:, ng:-ng] = upsample(rc[ng : ng + 1, ng + nj // 2 : -ng])
        return
    except KeyError as e:
        print(f"no neighbor {e} for block {index}")

    try:
        rc0 = grid[(level + 1, (2 * (i + 1), 2 * j + 0))]
        rc1 = grid[(level + 1, (2 * (i + 1), 2 * j + 1))]
        cc[-ng:, ng : +ng + nj // 2] = downsample(rc0[ng : 3 * ng, ng:-ng])
        cc[-ng:, ng + nj // 2 : -ng] = downsample(rc1[ng : 3 * ng, ng:-ng])
        return
    except KeyError as e:
        print(f"no neighbor {e} for block {index}")


def fill_guard_lc(index, grid):
    level, (i, j) = index
    ng = 2
    cc = grid[index]
    ni = cc.shape[0] - 2 * ng
    nj = cc.shape[1] - 2 * ng

    try:
        lc = grid[(level, (i - 1, j))]
        cc[ng:, ng:-ng] = lc[-2 * ng : -ng, ng:-ng]
        return
    except KeyError as e:
        print(f"no neighbor {e} for block {index}")

    try:
        lc = grid[(level - 1, (i // 2 - 1, j // 2))]
        if j % 2 == 0:
            cc[:ng, ng:-ng] = upsample(lc[ni : ni + 1, +ng : ng + nj // 2])
        else:
            cc[:ng, ng:-ng] = upsample(lc[ni : ni + 1, ng + nj // 2 : -ng])
        return
    except KeyError as e:
        print(f"no neighbor {e} for block {index}")

    try:
        lc0 = grid[(level + 1, (2 * (i - 1), 2 * j + 0))]
        lc1 = grid[(level + 1, (2 * (i - 1), 2 * j + 1))]
        cc[:ng, ng : +ng + nj // 2] = downsample(lc0[-3 * ng : -ng, ng:-ng])
        cc[:ng, ng + nj // 2 : -ng] = downsample(lc1[-3 * ng : -ng, ng:-ng])
        return
    except KeyError as e:
        print(f"no neighbor {e} for block {index}")


def fill_guard_cr(index, grid):
    level, (i, j) = index
    ng = 2
    cc = grid[index]
    ni = cc.shape[0] - 2 * ng
    nj = cc.shape[1] - 2 * ng

    try:
        cr = grid[(level, (i, j + 1))]
        cc[ng:-ng, -ng:] = cr[ng:-ng, ng : 2 * ng]
        return
    except KeyError as e:
        print(f"no neighbor {e} for block {index}")

    try:
        cr = grid[(level - 1, (i // 2, j // 2 + 1))]
        if i % 2 == 0:
            cc[ng:-ng, -ng:] = upsample(cr[+ng : ng + ni // 2, ng : ng + 1])
        else:
            cc[ng:-ng, -ng:] = upsample(cr[ng + ni // 2 : -ng, ng : ng + 1])
        return
    except KeyError as e:
        print(f"no neighbor {e} for block {index}")

    try:
        cr0 = grid[(level + 1, (2 * i + 0, 2 * (j + 1)))]
        cr1 = grid[(level + 1, (2 * i + 1, 2 * (j + 1)))]
        cc[ng : +ng + ni // 2, -ng:] = downsample(cr0[ng:-ng, ng : 3 * ng])
        cc[ng + ni // 2 : -ng, -ng:] = downsample(cr1[ng:-ng, ng : 3 * ng])
        return
    except KeyError as e:
        print(f"no neighbor {e} for block {index}")


def fill_guard_cl(index, grid):
    level, (i, j) = index
    ng = 2
    cc = grid[index]
    ni = cc.shape[0] - 2 * ng
    nj = cc.shape[1] - 2 * ng

    try:
        cl = grid[(level, (i, j - 1))]
        cc[ng:-ng, ng:] = cl[ng:-ng, -2 * ng : -ng]
        return
    except KeyError as e:
        print(f"no neighbor {e} for block {index}")

    try:
        cl = grid[(level - 1, (i // 2, j // 2 - 1))]
        if j % 2 == 0:
            cc[ng:-ng, :ng] = upsample(cl[+ng : ng + ni // 2, ni : ni + 1])
        else:
            cc[ng:-ng, :ng] = upsample(cl[ng + ni // 2 : -ng, ni : ni + 1])
        return
    except KeyError as e:
        print(f"no neighbor {e} for block {index}")

    try:
        cl0 = grid[(level + 1, (2 * i + 0, 2 * (j - 1)))]
        cl1 = grid[(level + 1, (2 * i + 1, 2 * (j - 1)))]
        cc[ng : +ng + nj // 2, :ng] = downsample(cl0[ng:-ng, -3 * ng : -ng])
        cc[ng + nj // 2 : -ng, :ng] = downsample(cl1[ng:-ng, -3 * ng : -ng])
        return
    except KeyError as e:
        print(f"no neighbor {e} for block {index}")


nz = 16
grid = dict()


grid[(1, (0, 1))] = cell_centers_2d(1, (0, 1), nz, nz)
grid[(2, (2, 3))] = cell_centers_2d(2, (2, 3), nz, nz)
grid[(2, (0, 1))] = cell_centers_2d(2, (0, 1), nz, nz)
z_arrays = {key: (exp(-(x**2 + y**2) / 0.05)) for key, (x, y) in grid.items()}

for z in z_arrays.values():
    z[:+2, :] = 0.0
    z[-2:, :] = 0.0
    z[:, :+2] = 0.0
    z[:, -2:] = 0.0


fill_guard_lc((2, (2, 3)), z_arrays)
fill_guard_cr((2, (0, 1)), z_arrays)


vmin = max(z[2:-2, 2:-2].min() for z in z_arrays.values())
vmax = max(z[2:-2, 2:-2].max() for z in z_arrays.values())

for ind in grid:

    z = z_arrays[ind]
    x, y = grid[ind]

    plt.pcolormesh(
        x,  # [2:-2, 2:-2],
        y,  # [2:-2, 2:-2],
        z,  # [2:-2, 2:-2],
        vmin=vmin,
        vmax=vmax,
    )

plt.colorbar()
plt.axis("equal")
plt.xlim(-0.5, 0.5)
plt.ylim(-0.5, 0.5)
plt.show()
