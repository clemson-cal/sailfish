"""
Demonstrates how to run a spherically symmetric relativistic wind setup.

This script invokes the sailfish driver.run function directly, rather than using
the sailfish command line entry point. The script assumes that it lives in a
sailfish sub-directory (e.g. sailfish/scripts). If you try to run a similar
script from a different directory, you will need to add the sailfish project
directory to your PYTHONPATH.
"""

from sys import path
from pathlib import Path

path.append(str(Path(__file__).parent.parent))

from numpy import array
from matplotlib import pyplot as plt
from sailfish.driver import run

state = run("wind", end_time=6e-2, num_patches=4, fold=10, quiet=False, resolution=500)

mesh = state.mesh
faces = array(mesh.faces(0, mesh.shape[0]))
r = 0.5 * (faces[1:] + faces[:-1])
d = state.solver.primitive[:, 0]
u = state.solver.primitive[:, 1]
p = state.solver.primitive[:, 2]

fig, ax1 = plt.subplots()
ax1.plot(r, u, label=r"$\gamma \beta$")
ax1.plot(r, p, label=r"$p$")
ax1.plot(r, d, label=r"$\rho$")
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xlabel(r"radius")
ax1.set_ylabel(r"density")
ax1.legend()
plt.show()
