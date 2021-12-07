#!/usr/bin/env python3

import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(1, ".")
from sailfish.driver import run

state = run("wind", end_time=6e-2, num_patches=4, fold=10, quiet=False, resolution=500)

mesh = state["mesh"]
faces = np.array(mesh.faces(0, mesh.shape[0]))
r = 0.5 * (faces[1:] + faces[:-1])
d = state["primitive"][:, 0]
u = state["primitive"][:, 1]
p = state["primitive"][:, 2]

fig, ax1 = plt.subplots()

ax1.plot(r, u, label=r"$\gamma \beta$")
ax1.plot(r, p, label=r"$p$")
ax1.plot(r, d, label=r"$\rho$")
# ax1.plot(r, 1 / r ** 2, label=r"$r^{-2}$", lw=1, ls="--", c="k")
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xlabel(r"radius")
ax1.set_ylabel(r"density")
ax1.legend()
plt.show()
