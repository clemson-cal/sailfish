"""
Demonstrates how to run a shocktube problem.

This script invokes the sailfish driver.run function directly, rather than using
the sailfish command line entry point. The script assumes that it lives in a
sailfish sub-directory (e.g. sailfish/scripts). If you try to run a similar
script from a different directory, you will need to add the sailfish project
directory to your PYTHONPATH.
"""

from sys import path
from pathlib import Path

path.append(str(Path(__file__).parent.parent))

from matplotlib import pyplot as plt
from sailfish.driver import run

state = run("shocktube", end_time=0.2, resolution=1000)
rho = state.solver.primitive[:, 0]
plt.plot(rho)
plt.show()
