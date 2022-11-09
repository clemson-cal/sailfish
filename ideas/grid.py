from numpy import zeros, linspace, meshgrid, exp


grid = dict()
dx = 0.125
dy = 0.125
ddx = dx / 10
ddy = dy / 10

for i in range(8):
    for j in range(8):
        x0 = -0.5 + (i + 0) * dx
        x1 = -0.5 + (i + 1) * dx
        y0 = -0.5 + (j + 0) * dy
        y1 = -0.5 + (j + 1) * dy
        xv = linspace(x0 - 2 * ddx, x1 + 2 * ddy, 15)
        yv = linspace(y0 - 2 * ddx, y1 + 2 * ddy, 15)
        xc = 0.5 * (xv[1:] + xv[:-1])
        yc = 0.5 * (yv[1:] + yv[:-1])
        x, y = meshgrid(xc, yc, indexing="ij")
        z = zeros([14, 14])
        z[...] = exp(-(x**2 + y**2) / 0.01)
        grid[(i, j)] = (x, y, z)

for i, j in grid:
    cc = grid.get((i, j))
    lc = grid.get((i - 1, j), None)
    rc = grid.get((i + 1, j), None)
    cl = grid.get((i, j - 1), None)
    cr = grid.get((i, j + 1), None)

    if lc is not None:
        cc[2][:+2, :] = lc[2][-4:-2, :]
    if rc is not None:
        cc[2][-2:, :] = rc[2][+2:+4, :]
    if cl is not None:
        cc[2][:, :+2] = cl[2][:, -4:-2]
    if cr is not None:
        cc[2][:, -2:] = cr[2][:, +2:+4]


import matplotlib.pyplot as plt

for i, j in grid:
    if i % 2 == 0 or j % 2 == 0:
        continue

    g = grid[(i, j)]

    plt.pcolormesh(
        g[0],
        g[1],
        g[2],
        vmin=0,
        vmax=1,
    )

plt.axis("equal")
plt.show()
