Problem setups
==============

Simulations are configured by an object called a `setup`. The setup specifies
initial conditions, a mesh, boundary conditions, and a solver class, among
several other optional things. Setups can be specific test problems used for
code validation, or they can encompass whole classes of research problems,
with many internal parameters. The degrees of freedom within a setup are
called `model parameters`, and these have default values which define a
fiducial model, but they can also be overridden from command line input
or other sources of configuration.

Below is a minimal setup, which defines a 1D relativistic hydrodynamics
problem consisting of a periodic density wave:

.. code-block:: python

    from math import sin, pi
    from sailfish.setup import Setup
    from sailfish.mesh import PlanarCartesianMesh

    class DensityWave(Setup):
        def primitive(self, t, x, p):
            p[0] = 1.0 + 0.1 * sin(2.0 * pi * x) # gas density
            p[1] = 0.0 # velocity
            p[2] = 1.0 # gas pressure

        def mesh(self, num_zones):
            return PlanarCartesianMesh(x0=0.0, x1=1.0, num_zones=num_zones)

        @property
        def solver(self):
            return "srhd_1d"

        @property
        def boundary_condition(self):
            return "periodic"
