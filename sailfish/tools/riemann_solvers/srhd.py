#!/usr/bin/env python3
from time import perf_counter
from matplotlib import pyplot as plt
import math
import numpy as np
from srhd_solvers import *

# Mignone & Bodo problem 1
prim0l = [1.0, 0.9, 1.0]
prim0r = [1.0, 0.0, 10.0]

pguess = 0.5 * (prim0l[2] + prim0r[2])

tmax = 0.4


def init_mb1(x):

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
    sl, sr = wavespeeds_simple(prim0l, prim0r)
    dt = 0.9 * dx / max(abs(sl), abs(sr))
    uhlle = np.zeros([nx, 3])
    uhllc = np.zeros([nx, 3])
    x = [xmin + (i + 0.5) * dx for i in range(nx)]
    for i in range(nx):
        uhlle[i] = init_mb1(x[i])
        uhllc[i] = init_mb1(x[i])

    uhlle0 = uhlle.copy()
    uhllc0 = uhllc.copy()

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

            prim = cons2prim(uhlle[i], pguess)
            priml = cons2prim(uhlle[il], pguess)
            primr = cons2prim(uhlle[ir], pguess)
            for n in range(3):
                uhlle0[i, n] = uhlle[i, n] + dt / dx * (
                    f_hlle(priml, prim)[n] - f_hlle(prim, primr)[n]
                )

            prim = cons2prim(uhllc[i], pguess)
            priml = cons2prim(uhllc[il], pguess)
            primr = cons2prim(uhllc[ir], pguess)
            for n in range(3):
                uhllc0[i, n] = uhllc[i, n] + dt / dx * (
                    f_hllc(priml, prim)[n] - f_hllc(prim, primr)[n]
                )

        uhlle = uhlle0.copy()
        uhllc = uhllc0.copy()

        t += dt
        t_end = perf_counter()
        kzps = nx / (t_end - t_start) / 1e3
        print(f"t = {t:0.4f} kzps = {kzps:0.3f}")

    phlle = np.array([cons2prim(ui, pguess) for ui in uhlle])
    phllc = np.array([cons2prim(ui, pguess) for ui in uhllc])

    f = plt.figure(1)

    ax = f.add_subplot(311)
    plt.plot(x, phlle[:, 0], "-o", mfc="none", label="hlle")
    plt.plot(x, phllc[:, 0], "-o", mfc="none", label="hllc")
    plt.xlabel(r"$x$")
    plt.ylabel(r"$\rho$")

    ax = f.add_subplot(312)
    plt.plot(x, phlle[:, 1], "-o", mfc="none", label="hlle")
    plt.plot(x, phllc[:, 1], "-o", mfc="none", label="hllc")
    plt.xlabel(r"$x$")
    plt.ylabel("v")

    ax = f.add_subplot(313)
    plt.plot(x, phlle[:, 2], "-o", mfc="none", label="hlle")
    plt.plot(x, phllc[:, 2], "-o", mfc="none", label="hllc")
    plt.xlabel(r"$x$")
    plt.ylabel("P")

    plt.legend()
    plt.show()


main()
