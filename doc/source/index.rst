Sailfish
~~~~~~~~

.. toctree::
   :hidden:

   setups
   workflow
   kernels
   api

Sailfish is a GPU-accelerated astrophysical gasdynamics code.

The main repository is hosted here: `github.com/clemson-cal <https://github.com/clemson-cal/sailfish>`_.

Quick-start
~~~~~~~~~~~

Basic use from the command line is like this: ``bin/sailfish shocktube
--end-time=0.2 --resolution=1000``. This command will run a built-in setup
called `shocktube` to a simulation time of 0.2 seconds, on a 1d grid with
1000 zones. The command line tool will print messages to the terminal, and
write a file called `chkpt.final.pk` to the disk in the current working
directory when the simulation finishes. This file is a Python pickle,
containing the simulation state and hydrodynamics solution data, and you can
load it in Python to plot the data.

Sailfish can also be used as a Python module from a custom script, as in the
code below:

.. code-block:: python

    from sailfish.driver import run
    from matplotlib import pyplot as plt
 
    state = run("shocktube", end_time=0.2, resolution=1000)
    rho = state.solver.primitive[:, 0]
    plt.plot(rho)
    plt.show()

The `run` function takes a sequence of arguments to control the driver and
problem setup, and then returns the simulation final state. However no
side-effects are performed by `run` and the terminal output is suppressed (you
can re-enable it with the keyword argument ``quiet=False``).

API documentation
~~~~~~~~~~~~~~~~~

.. autosummary::
   :recursive:

   sailfish.driver
   sailfish.event
   sailfish.kernel
   sailfish.mesh
   sailfish.physics
   sailfish.quad_tree
   sailfish.setup_base
   sailfish.setups
   sailfish.solver_base
   sailfish.solvers
   sailfish.subdivide
