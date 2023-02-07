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

Configuration items coming later in the procedure take precedence. This enables the creation of base configurations, which can be "tweaked" with input from JSON files or from the command line. A most-basic example is to start the code from a single JSON file. Save the following code in a file called `my-run.json`

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

This configuration is very close to the default one, which is why not many key-value pairs need to be provided. It is also a very easy and common test case, and as such it is a Sailfish built-in _preset_. A preset is a function that is built-in to the code and which emits these configuration items. This particular configuration can be found in a function called `sod` in the `sailfish.models` module, which looks like this:

```python
@preset
def sod():
    return {
        "initial_data.model": "sod",
        "domain.num_zones": [200, 1, 1],
        "driver.tfinal": 0.1,
    }
```

The decorator `@preset` tells the application that this function should be recognized as a preset. Running `sailfish doc presets` prints a list of the available presets. To invoke this preset, you would type on the command line `sailfish run sod`.

Now suppose that you want to reuse _most_ of a particular preset, or configuration file, but you want to override certain pieces of it. You have two options: (a) you can _chain_ the configuration files or (b) use command line flags. Option (a) is more general, since there is not a command line flag to control every possible configuration item. For example if you wanted to run the Sod problem but with higher resolution, you could put the following text in a second JSON file `res.json`

```json
{
    "domain.num_zones": [20000, 1, 1],
}
```

and then invoke the code like this: `sailfish run sod res.json`. The `domain.num_zones` configuration item from the JSON file takes precedence, so you would get all the configuration items from the `sod` preset, except with 20,000 zones instead of 200. Since the grid resolution is a commonly modified parameter, so there also is a command line flag `--resolution/-n` to control it directly, which is generally more convenient:

```bash
sailfish run sod -n 20000
```

## The Sailfish Class

You can read the full description of the code configuration in the `sailfish.config` module. In that module you can find the `Sailfish` class, which looks like this:

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

Several of these items refer to configuration sub-classes, for example `Driver`, `Physics`, and `Strategy`. These are classes you can also find in the `config` module. You can figure out everything about the possible code configurations by reading the source code of that module.

Configuration data you supply through JSON files, presets, checkpoints, or command line flags, is first packed into a Python dictionary, and then converted to a `Sailfish` struct and validated. The mapping of a Python dictionary to the `Sailfish` class is pretty self-explanatory. Be aware that it is allowable to use either dot-syntax in the dictionary keys, or nested dictionaries. The following two JSON configuration files are equivalent.

##### Using nested dictionaries:
```json
{
    "initial_data": {
    	"model": "sod"
    }
}
```

##### Using dot-syntax:
```json
{
    "initial_data.model": "sod"
}
```



## Model data classes

Soon...
