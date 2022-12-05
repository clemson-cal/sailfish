from configmodel import configmodel
from typing import Literal


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
    diagnostics: list[Diagnostic] = (
        Diagnostic("thing1"),
        Diagnostic("thing2"),
        Diagnostic("thing3"),
    )


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
    cfl_number: float | Literal["auto"] = "auto"


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

    equation_of_state: UniformSoundSpeed | UniformMachNumber = UniformSoundSpeed(1.0)


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


@configmodel
class Scheme:
    """
    Algorithm parameters for the solution scheme

    A scheme refers to the parts of a solver that affect the numerical
    solution up to the truncation order.

    Fields
    ------

    space: spatial reconstruction method
    time:  time-integration scheme (fwd|rk1|rk2|rk3)
    """

    Reconstruction = Literal["pcm"] | tuple[Literal["plm"], float]
    TimeIntegration = Literal["fwd"] | Literal["rk1"] | Literal["rk2"] | Literal["rk3"]

    space: Reconstruction = "pcm"
    time: TimeIntegration = "rk2"


@configmodel
class Strategy:
    """
    Algorithm parameters only affecting performance

    With everything else held fixed, differences in the solver strategy should
    only affect how fast the code runs, and not the numerical solution it
    generates.
    """

    fluxing: Literal["per-zone"] | Literal["per-face"] = "per-zone"


@configmodel
class Application:
    """
    Fields
    ------

    driver:   runs the simulation main loop, handles IO
    physics:  the physics equations to be solved
    domain:   the physical domain of the problem
    strategy: the solution strategy
    """

    driver: Driver = Driver()
    physics: Euler | Isothermal = Euler()
    domain: CoordinateBox = CoordinateBox()
    strategy: Strategy = Strategy()


if __name__ == "__main__":
    from rich.console import Console
    from rich.live import Live

    console = Console()

    app = Application(physics=Isothermal())

    for field in app.__dataclass_fields__:
        console.print()
        console.print(getattr(app, field))
