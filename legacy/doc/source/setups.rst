Problem setups
==============

.. py:currentmodule:: sailfish

Simulations are configured by an object called a `setup`. The setup specifies
initial conditions, a mesh, boundary conditions, and a solver class, among
several other optional things. Setups can be specific test problems used for
code validation, or they can encompass whole classes of research problems,
with many internal parameters. The degrees of freedom within a setup are
called `model parameters`, and these have default values which define a
fiducial model, but they can also be overridden from command line input
or other sources of configuration.

Minimal example
^^^^^^^^^^^^^^^

Here is a minimal setup, which defines a 1D relativistic hydrodynamics problem
consisting of a traveling density wave:

.. code-block:: python

    from math import sin, pi
    from sailfish.setup_base import SetupBase
    from sailfish.mesh import PlanarCartesianMesh

    class DensityWave(SetupBase):
        """
        A sinusoidal variation of density propagating to the right.
        """
        def primitive(self, t, x, p):
            p[0] = 1.0 + 0.1 * sin(2.0 * pi * x) # gas density
            p[1] = 0.5 # velocity
            p[2] = 1.0 # gas pressure

        def mesh(self, resolution):
            return PlanarCartesianMesh(0.0, 1.0, resolution)

        @property
        def solver(self):
            return "srhd_1d"

        @property
        def boundary_condition(self):
            return "periodic"

This setup uses the `srhd_1d` solver (relativistic 1D hydrodynamics) with
cartesian coordinates, on a domain from 0.0 to 1.0. The boundary condition is
periodic, and the hydrodynamic primitive data so the density has a sinusoidal
variation, and the velocity is uniform.

Required methods
^^^^^^^^^^^^^^^^

Custom setup classes must inherit the :obj:`SetupBase <setup_base.SetupBase>`
base class. The setup base class provides a lot of functionality in the form
of methods that have trivial default implementations, but can be overridden if
needed. The four methods implemented in the example above are required for the
setup to be instantiated (they are `abstract` methods in the base class).

The :obj:`primitive <setup_base.SetupBase.primitive>` method takes as
arguments the time, a coordinate position, and a slice of a primitive variable
array representing a single zone. The result is written to this slice rather
than returned from the function for efficiency reasons. Generally, the
primitive method is only used at the start of a new simulation, so the time
will typically be `t=0.0`. However, in the case of an inflow boundary
condition, the solver will also call the setup's primitive method in the guard
zones of the mesh each time boundary conditions need to be applied. In other
words, the primitive method always serves as an initial condition, but it can
also provide a time-dependent Dirichlet boundary condition. The number and
meaning of primitive variables depends on the system being solved (the system
is in turn a property of the solver class, which is identified by the
:obj:`solver <setup_base.SetupBase.solver>` property).

The object returned by the :obj:`mesh <sailfish.setup_base.SetupBase.mesh>`
method describes the physical extent of the simulation domain, as well as how
it is discretized and what kind of coordinates are used (i.e. cartesian or
spherical). The `mesh` method accepts a `resolution` argument, the value of
which comes from the command line option :code:`--resolution | -n`, or the
`resolution` member of the :obj:`DriverArgs <driver.DriverArgs>` class. It
should be interpreted sensibly in the context of your setup to create a mesh
(for example no logical issues would arise if you were to ignore this argument
altogether, but don't do that). For example if your setup uses a
:obj:`LogSphericalMesh <mesh.LogSphericalMesh>`, it makes sense to interpret
the `resolution` as the number of zones per decade in radius. If it's a 2D
problem in spherical coordinates, it could mean the number of polar zones, or
in a 3D problem it could be the number of zones on each side of the domain.

The :obj:`boundary_condition <setup_base.SetupBase.boundary_condition>` method
returns a string to identify the type of boundary condition to be applied. The
mesh and boundary condition objects must be compatible with (supported by) the
solver, otherwise the solver will throw an exception when it's constructed.


Choosing a setup
^^^^^^^^^^^^^^^^

The first argument to the sailfish command line tool is the name of a setup
class, converted to dash-case, i.e. you could run the setup above in a default
configuration by running :code:`sailfish density-wave`. The command line tool
will print the setup documentation to the terminal if invoked as
:code:`sailfish density-wave --describe`. It's good practice to give your
setup class an accurate doc string.

Setup sub-classes are automatically discovered when sailfish is imported or
run from the command line. However to be found, the module containing your
class must be imported before :obj:`driver.simulate` is called. If you add a
new source file to the :obj:`setups` module, you must import that module in
the :code:`setups/__init__.py` file for the setup class(es) in it to be
discovered.


Model parameters
^^^^^^^^^^^^^^^^

A setup can have internal degrees of freedom to be configured at runtime,
which are referred to as `model parameters`. To add a model parameter to a
setup, just define it as a class variable using the :obj:`param
<setup_base.Parameter>` constructor:

.. code-block:: python

    from sailfish.setup_base import SetupBase, param

    class DensityWave(SetupBase):
        wavenumber = param(1, "integer wavenumber of the sinusoid")
        velocity = param(0.0, "speed of the wave")

        # ...

The two positional arguments to :obj:`param <setup_base.Parameter>` are a
default value (from which the parameter type is inferred), and a help message.
An optional keyword argument :code:`mutable=True` can be supplied to indicate
that a parameter can be changed in a restarted run from its initial value. For
the model parameters that only influence the initial condition, it doesn't
make sense to make them mutable.

Model parameters are passed to the setup class from the command line as
key-value pairs like this:
:code:`sailfish density-wave --model amplitude=0.5 wavenumber=2`.


Optional methods
^^^^^^^^^^^^^^^^
