# Opening checkpoint files

## Using Pickle to open a checkpoint

Sailfish checkpoint files use Python's `pickle` module to store the state of
the simulation. Checkpoint files can be used for science post-processing of the
simulation data, or for the purpose of restarting a simulation. Let's start
with loading a checkpoint:

```python
from pickle import load

with open("chkpt.0000.pk", "rb") as infile:
    chkpt = load(infile)
```

This piece of code uses the Python builtin `open` function to open the
checkpoint `"chkpt.0000.pk"` (in the current directory) for reading, in binary
mode. Note that this "zero" checkpoint will generally be written by the code
when it starts up. It contains the initial condition. The `with` statement
creates a thing called a context manager. It ensures that the file object is
closed at the end of the `with` statement.

The `chkpt` object returned by the `load` function is a Python dictionary. To
see its contents, you might start by adding `print(chkpt.keys())` to the code
above. You should see a list with the following keys:

- `"config"`
- `"time"`
- `"primitive"`
- `"iteration"`
- `"timestep"`
- `"timeseries"`
- `"event_states"`

For now we need to pay attention to the first three. `chkpt["config"]` is a
dictionary of the run configuration. You can inspect it to discover all the
details of how Sailfish was configured when it produced that checkpoint.
`chkpt["time"]` is the simulation time when the checkpoint was written. Since
this checkpoint contains the initial condition, the time will be zero.
`chkpt["primitive"]` is a `numpy` array containing the hydrodynamics data. It
is an array, where the leading axes correspond to the spatial axes in the
computational grid, and the final axis contains the different hydrodynamics
fields. Which fields are there depends on which equations are being solved. In
a 1d setting, those fields will be `("density", "velocity", "pressure")` in
that order.

## Basic 1d plotting    

Let's suppose you'd like to plot the density and pressure profiles in a 1d
simulation. Since density is the first field and pressure is the last (and
Python indexes start from zero), you could obtain and plot that data like this:

```python
from matplotlib import pyplot as plt

rho = chkpt["primitive"][:,0]
pre = chkpt["primitive"][:,1]
plt.plot(rho)
plt.plot(pre)
plt.show()
```

Note that when `plt.plot` receives a single array, it treats it as data for the
y-axis, and plots it against an array of indexes on the horizontal axis.
Actually this is not too bad. But generally it will make more sense to plot
your data against the physical coordinates of the grid cell locations. This
means you need to know (a) the physical extent of the domain, and (b) the
number of grid zones. This information is here:

```python
domain = chkpt["config"]["domain"]
ni = domain["num_zones"][0]
x0 = domain["extent_i"][0]
x1 = domain["extent_i"][1]
```

From here, you can use `numpy.linspace` to generate coordinates for the zones, and then plot your data against physical coordinates:

```python
from numpy import linspace

dx = (x1 - x0) / ni
x = linspace(x0 + 0.5 * dx, x1 - 0.5 * dx, ni)
plt.plot(x, rho, label=r"$\rho$")
plt.plot(x, pre, label=r"$p$")
plt.legend()
plt.show()
```

Now you should see that the horizontal axis shows the physical coordinates
rather than array indexes. Note that in the call to `linspace`, the left-end
point is one-half of a grid-cell to the right of `x0`, and the right-end point
is one-half of a grid-cell to the left of `x1`. This is because we want to
associate the hydrodynamic fields with the location of the centers of the grid
zones. The code above also uses `matplotlib`'s TeX-rendering feature to make a
nice legend for the data shown in the figure.

## Using Sailfish internals

The same functions and classes that Sailfish uses internally to do the
simulation can also be used in post-processing. Let's start with the `Sailfish`
configuration class itself. That class is located in the `sailfish.config`
module. To use it, the `sailfish` module needs to be in your `PYTHONPATH`
environment variable. If you have put your analysis/plotting script in the
Sailfish project directory, or you are at an interactive Python terminal in the
project directory, then Python will find the `sailfish` module without you
having to do anything else. This code will reconstruct a `Sailfish`
configuration class instance:

```python
from sailfish.config import Sailfish

config = Sailfish(**chkpt["config"])
```

Whereas `chkpt["config"]` is a dictionary, `config` is now an instance of the `Sailfish` class, and its attributes have useful member functions attached. For example, to obtain the cell coordinates, you can exploit the fact that `config.domain` is an instance of `sailfish.geometry.CoordinateBox`, and that class has a member function called `cell_centers`:

```python
x = config.domain.cell_centers()
```

You could also take advantage of the fact that `config.initial_data` is the
initial data function; it can recreate the primitive variable arrays as they
would have been at `t=0`! This might be useful, let's say, if you were tweaking
a steady-state initial condition, and you wanted to validate the initial
condition by comparing the solution at `t=0` to the evolved solution at say
`t=1` (if you have a good steady state, it should not evolve away from the
initial condition). The `initial_data` instance has a member function with
signature `primitive(box: CoordinateBox) -> ndarray`. Here is a complete piece
of code to plot the difference in the density field in the first (non-zero)
checkpoint to the density field in the initial condition:

```python
from pickle import load
from matplotlib import pyplot as plt
from sailfish.config import Sailfish

with open("chkpt.0000.pk", "rb") as infile:
    chkpt = load(infile)

config = Sailfish(**chkpt["config"])
domain = config.domain
x = domain.cell_centers()
rho1 = chkpt["primitive"][:,0]
rho0 = config.initial_data(domain)[:,0]
plt.plot(x, rho1 - rho0)
plt.show()
```
