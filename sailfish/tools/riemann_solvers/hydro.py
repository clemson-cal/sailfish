#!/usr/bin/env python3
from time import perf_counter
from matplotlib import pyplot as plt
import math
import numpy as np
from euler_solvers import *

# Toro Table 10.1
# Test 1
# prim0l = [1.0, 0.75, 1.0]
# prim0r = [0.125, 0.0, 0.1]
# Test 2
prim0l = [1.0, -2.0, 0.4]
prim0r = [1.0, 2.0, 0.4]
# Test 3
# prim0l = [1.0, 0.0, 1000.0]
# prim0r = [1.0, 0.0, 0.01]
# Test 4
# prim0l = [5.99924, 19.5975, 460.894]
# prim0r = [5.99942, -6.19633, 46.0950]
# Test 5
# prim0l = [1.0, -19.5975, 1000.0]
# prim0r = [1.0, -19.5975, 0.01]

sl, sr = wavespeeds_simple(prim0l, prim0r)
smax = max(sl, sr)

tmax = 0.15


def init_toro1(x):

    if x < 0.5:
        rho, v, p = prim0l
    else:
        rho, v, p = prim0r
    return prim2cons([rho, v, p])


def main():
    nx = 100
    xmin = 0.0
    xmax = 1.0
    dx = (xmax - xmin) / nx
    dt = 0.5 * dx / smax
    uhlle = np.zeros([nx, 3])
    uhllc = np.zeros([nx, 3])
    uexact = np.zeros([nx, 3])
    x = [xmin + (i + 0.5) * dx for i in range(nx)]
    for i in range(nx):
        uhlle[i] = init_toro1(x[i])
        uhllc[i] = init_toro1(x[i])
        uexact[i] = init_toro1(x[i])

    uhlle0 = uhlle.copy()
    uhllc0 = uhllc.copy()
    uexact0 = uexact.copy()

    t = 0.0
    while t < tmax:
        t_start = perf_counter()
        for i in range(nx):
            il = i - 1
            ir = i + 1
            if i == 0:
                il = 0
            if i == nx - 1:
                ir = nx - 1

            prim = cons2prim(uhlle[i])
            priml = cons2prim(uhlle[il])
            primr = cons2prim(uhlle[ir])
            for n in range(3):
                uhlle0[i, n] = uhlle[i, n] + dt / dx * (
                    hlle(priml, prim)[n] - hlle(prim, primr)[n]
                )

            prim = cons2prim(uhllc[i])
            priml = cons2prim(uhllc[il])
            primr = cons2prim(uhllc[ir])
            for n in range(3):
                uhllc0[i, n] = uhllc[i, n] + dt / dx * (
                    hllc(priml, prim)[n] - hllc(prim, primr)[n]
                )

            prim = cons2prim(uexact[i])
            priml = cons2prim(uexact[il])
            primr = cons2prim(uexact[ir])
            for n in range(3):
                uexact0[i, n] = uexact[i, n] + (
                    dt / dx * (exact(priml, prim)[n] - exact(prim, primr)[n])
                )

        uhlle = uhlle0.copy()
        uhllc = uhllc0.copy()
        uexact = uexact0.copy()

        t += dt
        t_end = perf_counter()
        kzps = nx / (t_end - t_start) / 1e3
        print(f"t = {t:0.4f} kzps = {kzps:0.3f}")

    phlle = np.array([cons2prim(ui) for ui in uhlle])
    phllc = np.array([cons2prim(ui) for ui in uhllc])
    pexact = np.array([cons2prim(ui) for ui in uexact])

    f = plt.figure(1)

    ax = f.add_subplot(311)
    plt.plot(x, phlle[:, 0])
    plt.plot(x, phllc[:, 0])
    plt.plot(x, pexact[:, 0])
    plt.xlabel(r"$x$")
    plt.ylabel(r"$\rho$")

    ax = f.add_subplot(312)
    plt.plot(x, phlle[:, 1])
    plt.plot(x, phllc[:, 1])
    plt.plot(x, pexact[:, 1])
    plt.xlabel(r"$x$")
    plt.ylabel("v")

    ax = f.add_subplot(313)
    plt.plot(x, phlle[:, 2], label="hlle")
    plt.plot(x, phllc[:, 2], label="hllc")
    plt.plot(x, pexact[:, 2], label="exact")
    plt.xlabel(r"$x$")
    plt.ylabel("P")

    plt.legend()
    plt.show()


main()
