from configmodel import configmodel
from typing import Literal, Union


@configmodel
class Report:
    cadence: int = 100


@configmodel
class Checkpoint:
    cadence: float = 1.0
    format: str = "pickle"


@configmodel
class Diagnostic:
    name: str


@configmodel
class Timeseries:
    cadence: float = 1.0
    diagnostics: list[Diagnostic] = None


@configmodel
class Driver:
    """
    Fields
    ------

    timeseries: high-cadence recording of science products
    """

    tstart: float = 0.0
    tfinal: float = 1.0
    report: Report = Report()
    checkpoint: Checkpoint = Checkpoint()
    timeseries: Timeseries = Timeseries()
    cfl_number: Union[float, Literal["auto"]] = "auto"


@configmodel
class Euler:
    gamma_law_index: float = 5.0 / 3.0


@configmodel
class UniformMachNumber:
    """
    Isothermal EOS with a uniform nominal Mach number
    """

    mach_number: float = 10.0


@configmodel
class UniformSoundSpeed:
    """
    Isothermal EOS with a uniform sound speed
    """

    sound_speed: float = 1.0


@configmodel
class Isothermal:
    """
    Isothermal gasdynamics equations

    Isothermal gas has a temperature that is prescribed externally, to be
    either globally uniform or some explicit function of space and time. The
    fluid total energy density is not evolved.
    """

    equation_of_state: Union[UniformSoundSpeed, UniformMachNumber] = UniformSoundSpeed(
        1.0
    )


@configmodel
class CoordinateBox:
    """
    Domain with uniformly spaced grid cells

    A coordinate box is a (hyper-)rectangular region in whatever coordinates
    are used. Examples include cartesian coordinates, and spherical polar
    coordinates with logarithmic radial grid spacing.

    Fields
    ------

    extent_i: the extent_i
    extent_j: the extent_j
    extent_k: the extent_k
    """

    extent_i: tuple[float, float] = (0.0, 1.0)
    extent_j: tuple[float, float] = (0.0, 1.0)
    extent_k: tuple[float, float] = (0.0, 1.0)
    num_zones: tuple[int, int, int] = (128, 1, 1)

    @property
    def dimensionality(self):
        """
        number of fleshed-out spatial axes
        """
        return sum(n > 1 for n in self.num_zones)


Reconstruction = Union[Literal["pcm"], tuple[Literal["plm"], float]]
TimeIntegration = Literal["fwd", "rk1", "rk2", "rk3"]


@configmodel
class Scheme:
    """
    Algorithm parameters for the solution scheme

    A scheme refers to the parts of a solver that affect the numerical
    solution up to the truncation order.

    Fields
    ------

    reconstruction:    spatial reconstruction method
    time_integration:  time-integration scheme (fwd|rk1|rk2|rk3)
    """

    reconstruction: Reconstruction = "pcm"
    time_integration: TimeIntegration = "rk2"


@configmodel
class Strategy:
    """
    Algorithm parameters only affecting performance

    With everything else held fixed, differences in the solver strategy should
    only affect how fast the code runs, and not the numerical solution it
    generates.

    Fields
    ------

    data_layout: fields are contiguous (fields-last) or disjoint (fields-first)
    cache_flux: one Riemann problem per face, difference the resulting array
    cache_prim: pre-compute primitive quantities, vs. re-compute over stencil
    """

    data_layout: Literal["fields-first", "fields-last"] = "fields-last"
    cache_flux: bool = False
    cache_prim: bool = False


@configmodel
class Sailfish:
    """
    Fields
    ------

    driver:    runs the simulation main loop, handles IO
    physics:   the physics equations to be solved
    domain:    the physical domain of the problem
    strategy:  the solution strategy
    hardware:  compute device [cpu|gpu]
    """

    driver: Driver = Driver()
    physics: Union[Euler, Isothermal] = Euler()
    domain: CoordinateBox = CoordinateBox()
    strategy: Strategy = Strategy()
    scheme: Scheme = Scheme()
    hardware: Literal["cpu", "gpu"] = "cpu"
    show_config: Literal["pretty", "table", "json", "dict"] = "pretty"
    plot: Literal["live", "end"] = None


if __name__ == "__main__":
    from rich.console import Console
    from rich.live import Live

    console = Console()

    app = Sailfish(physics=Isothermal())

    for field in app.__dataclass_fields__:
        console.print()
        console.print(getattr(app, field))
