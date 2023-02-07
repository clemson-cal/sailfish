# Code configuration

## Configuration sources

Run configuration refers to selecting things like initial conditions, the
physics to be modeled, the type of hardware (CPU or GPU), the accuracy of
the numerical integration scheme, and more. Sailfish uses a single data
structure, the `Sailfish` class, to specify a configuration. Each time you run
the code, a `Sailfish` class instance is created by gathering configuration
data from a sequence of inputs. Here is the procedure used by the code:

1. A default configuration is created. This default should not change as the code evolves.
2. The configuration is updated with key-value pairs from a _sequence_ of the
   following kinds of input:
	1. A JSON-formatted configuration file (ends in .json)
	2. A checkpoint file (ends in .pk)
	3. The name of a registered _preset_ (has no extension)
3. The configuration is finally updated with key-value pairs taken from command
   line flags

Configuration items coming later in the procedure take precedence. This
enables the creation of base configurations, which can be "tweaked" with input
from JSON files or from the command line.

A most-basic example is to start the code from a single JSON file. Try saving
the following code in a file called `my-run.json`,

```json
{
    "initial_data.model": "sod",
    "domain.num_zones": [200, 1, 1],
    "driver.tfinal": 0.1
}
```

and then run sailfish with that file supplied as input:

```bash
sailfish run my-run.json
```

This configuration is very close to the default one, so not many configuration
items need to be overridden. This is also a very easy and common test case, and
as such it is a Sailfish built-in _preset_. A preset is a function that is
built-in to the code and which emits these configuration items. This
particular configuration can be found in a function called `sod` in the
`sailfish.models` module, which looks like this:

```python
@preset
def sod():
    return {
        "initial_data.model": "sod",
        "domain.num_zones": [200, 1, 1],
        "driver.tfinal": 0.1,
    }
```

The decorator `@preset` tells the application that this function should be
recognized as a preset. Running `sailfish doc presets` prints a list of the
available presets. To invoke this preset, you would type on the command line
`sailfish run sod`.

Now suppose that you want to reuse _most_ of a particular preset, or
configuration file, but you also want to override certain pieces of it. You
have two options: (a) you can _chain_ the configuration files or (b) use
command line flags. Option (a) is more general, since there is not a command
line flag to control all the configuration items. For example if you wanted to
run the Sod problem but with higher resolution, you could put the following
text in a second JSON file `res.json`

```json
{
    "domain.num_zones": [20000, 1, 1],
}
```

and then invoke the code like this: `sailfish run sod res.json`. The
`domain.num_zones` configuration item from the JSON file takes precedence, so
you would get all the configuration items from the `sod` preset, except with
20,000 zones instead of 200. Since the grid resolution is a commonly modified
parameter, so there also is a command line flag `--resolution/-n` to control
it directly, which is generally more convenient:


```bash
sailfish run sod -n 20000
```

## The `Sailfish` class

You can read the full description of the code configuration in the
`sailfish.config` module. That module defines the `Sailfish` class, which
looks like this:

```python
class Sailfish:
    """
    Top-level application configuration struct

    Fields
    ------

    name:               a name given to the run
    driver:             runs the simulation main loop, handles IO
    physics:            the physics equations to be solved
    initial_data:       initial condition, used if not a restart
    boundary_condition: boundary condition to apply at domain edges
    domain:             the physical domain of the problem
    strategy:           the solution strategy; does not affect the numerical solution
    scheme:             algorithmic choices which can affect the solution accuracy
    forcing:            driving of the hydrodynamic fields to implement a soft BC
    """

    name: str = None
    driver: Driver = Driver()
    physics: Physics = Physics()
    initial_data: dict = None
    boundary_condition: BoundaryCondition = BoundaryCondition()
    domain: CoordinateBox = CoordinateBox()
    coordinates: Coordinates = "cartesian"
    strategy: Strategy = Strategy()
    scheme: Scheme = Scheme()
    forcing: Forcing = None
```

Several of the class attributes refer to configuration sub-classes, for
example `Driver`, `Physics`, and `Strategy`. You can also find these classes
in the `config` module. You can figure out everything about the possible code
configurations by reading the source code of that module.

Configuration data you supply through JSON files, presets, checkpoints, or
command line flags is first packed into a Python dictionary, and then
converted to a `Sailfish` struct and validated. The mapping of a Python
dictionary to the `Sailfish` class is pretty self-explanatory. It is allowable
to use either dot-syntax in the dictionary keys, or nested dictionaries, i.e.
the following two JSON configuration files are equivalent:

_Using nested dictionaries_
```json
{
    "initial_data": {
    	"model": "sod"
    }
}
```

_Using dot-syntax_
```json
{
    "initial_data.model": "sod"
}
```

 

## Model data classes


The `Sailfish.initial_data` attribute contains configuration items that will
be used to construct a class that provides initial conditions for the
hydrodynamic fields. When a new simulation is started, the solver will
construct initial data by calling a member function on the initial data class
with the signature `primitive(self, box: CoordinateBox) -> ndarray`. The
classes that provide initial data classes are also called "model data"
classes, and they are marked with the `@modeldata` decorator. This decorator
registers model data classes and allows configuration files to refer to those
classes by name (note that the name is converted from `TitleCase` to
`dash-case`).

The `sailfish.models` module contains many examples of model data classes. A
simple example is one that defines the initial conditions for a sinusoidal
density wave in a one-dimensional hydrodynamics problem:

```python
@modeldata
class DensityWave:
    """
    Sinusoidal density wave translating rigidly
    """

    amplitude: float = 0.2

    @property
    def primitive_fields(self):
        return "density", "x-velocity", "pressure"

    def primitive(self, box: CoordinateBox):
        x = box.cell_centers()
        p = zeros(x.shape + (3,))
        p[..., 0] = 1.0 + self.amplitude * sin(2 * pi * x)
        p[..., 1] = 1.0
        p[..., 2] = 1.0
        return p
```

This model data class defines one _model parameter_ called `amplitude`, which
can be controlled through the configuration data, for example:

```json
{
    "initial_data": {
        "model": "density-wave",
        "amplitude": 0.5
    }
}
```

Also note there is a (property) function called `primitive_fields`. This
property must be implemented by the model data class to return the names of
the primitive variable fields. This provides some self-documentation, but it
is also used by the solver to infer the _number_ of hydrodynamic fields that
will be evolved, which could be 3, 4, or 5 depending on the number of vector
components of the velocity field need to be evolved. Note that the ordering of
these fields is imposed by the solver, and is e.g. `[density, vx, pressure]`
if one vector component of the velocity field is evolved, or e.g. `[density,
vx, vy, vz, pressure]` if three vector components are evolved. Also note that
while the solver reads the _number_ of fields that are returned, it does not
actually read the names of the fields. The field names are there only for
self-documentation purposes. Especially with the velocity components, it could
make sense to name them e.g. `"density, x-velocity, pressure"` if your setup
is intended for a one-dimensional planar Cartesian geometry, or e.g.
`"density, radial-velocity", "polar-velocity, pressure"` if your setup is
intended for a two-dimensional axisymmetric spherical-polar coordinate system.
