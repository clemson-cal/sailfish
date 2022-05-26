"""
The `Setup` class dictates physics and solver choices to the driver.
"""

from abc import ABC, abstractmethod
from textwrap import dedent
from typing import NamedTuple, Any


class SetupError(Exception):
    """Something went wrong during setup configuration"""


class Parameter(NamedTuple):
    """
    A model parameter: enables runtime configuring of a setup.

    `Parameter` instances are assigned as class variables to `Setup`
    sub-classes. The `mutable` flag should be set to to `True` if the
    parameter is logically adjustable either during the run, or can be
    superseded when a run is restarted.
    """

    default: Any
    about: str
    mutable: bool = False


param = Parameter


class Setup(ABC):
    """
    Abstract base class to describe a simulation model setup.

    Subclasses are used by the driver to define initial and boundary
    conditions, select a hydrodynamics solver and parameters, and describe
    physics conditions such as gravity and thermodynamics. Basic setups only
    need to implement a subset of the possible methods; most of the methods
    below have stub default implementations.
    """

    def __init__(self, **kwargs):
        for key, val, about in type(self).default_model_parameters():
            if key in kwargs:
                if type(kwargs[key]) != type(val):
                    raise SetupError(
                        f"parameter '{key}' has type {type(val).__name__} "
                        f"(got {type(kwargs[key]).__name__})"
                    )
                setattr(self, key, kwargs[key])
            else:
                setattr(self, key, val)

        for key, val in kwargs.items():
            if not hasattr(self, key):
                raise SetupError(
                    f"'{self.dash_case_class_name()}' has no parameter '{key}'"
                )

        self.validate()

    @classmethod
    def default_model_parameters(cls):
        """
        Return an iterator over the default model parameters for this class.
        """
        for key, val in vars(cls).items():
            if type(val) == Parameter:
                yield key, val.default, val.about
            elif hasattr(cls, "__annotations__"):
                print(cls.__annotations__)

    @classmethod
    def immutable_parameter_keys(cls):
        """
        Return an iterator over the immutable model parameter keys.
        """
        for key, val in vars(cls).items():
            if type(val) == Parameter and not val.mutable:
                yield key

    @classmethod
    def describe_class(cls):
        """
        Print formatted text describing the setup.

        The output will include a normlized version of the class doc string,
        and the model parameter names, default values, and about message.
        """
        print(f"setup: {cls.dash_case_class_name()}")
        print(dedent(cls.__doc__))
        cls().print_model_parameters()

    @classmethod
    def dash_case_class_name(cls):
        """
        Return a `dash-case` name of this setup class.

        Dash case is used in configuration, including the setup name passed to
        the driver, and written in checkpoint files.
        """
        return "".join(
            ["-" + c.lower() if c.isupper() else c for c in cls.__name__]
        ).lstrip("-")

    @classmethod
    def find_setup_class(cls, name):
        """
        Finds a setup class with the given name.

        The class name is expected in dash-case format. If no setup is found,
        a `SetupError` exception is raised.
        """
        match = lambda s: s.dash_case_class_name() == name
        try:
            return next(filter(match, cls.__subclasses__()))
        except StopIteration:
            raise SetupError(f"no setup named {name}")

    @classmethod
    def has_model_parameters(cls):
        """
        Determine if this class has any model parameters.
        """
        for _ in cls.default_model_parameters():
            return True
        return False

    def print_model_parameters(self, logger=None, newlines=False):
        """
        Print parameter names, values, and about messages to `stdout`.
        """

        def _p(m):
            if logger is not None:
                logger.info(m)
            else:
                print(m)

        if newlines:
            _p("")
        if self.has_model_parameters():
            _p("model parameters:\n")
            for name, val, about in self.model_parameters():
                _p(f"{name:.<20s} {str(val):<12} {about}")
        else:
            _p("setup has no model parameters")
        if newlines:
            _p("")

    def model_parameters(self):
        """
        Return an iterator over the model parameters chosen for this setup.
        """
        for key, val, about in self.default_model_parameters():
            yield key, getattr(self, key), about

    def model_parameter_dict(self):
        """
        Return a dictionary of the model parameters.
        """
        return {key: val for key, val, _ in self.model_parameters()}

    @abstractmethod
    def primitive(self, time, coordinate, primitive):
        """
        Set initial or boundary data at a point.

        This method must be overridden to set the primitive hydrodynamic
        variables at a single point (given by `coordinate`) and time. The
        meaning of the coordinate is influenced by the type of mesh being
        used. Data is written to the output variable `primitive` for
        efficiency, since otherwise a tiny allocation is made for each zone in
        the simulation grid.

        The time coordinate enables the use of a time-dependent boundary
        condition.
        """
        pass

    @abstractmethod
    def mesh(self, resolution: int):
        """
        Return a mesh instance describing the problem domain.

        This method must be overridden, and the domain type must be supported
        by the solver. A resolution parameter is passed here from the driver.
        """
        pass

    @property
    @abstractmethod
    def solver(self):
        """
        Return the name of the setup class.
        """
        pass

    @property
    def physics(self):
        """
        Return physics parameters used by the solver: EOS, gravity, etc.

        Physics parameters should be distinct from the solver options, such as
        PLM theta value, RK integration type, or DG order.
        """
        return dict()

    @property
    @abstractmethod
    def boundary_condition(self):
        """
        Return a boundary condition mode.

        This method must be overridden, and the mode must be supported by the
        solver. 1D solvers should accept either a single return value
        specifying the BC mode at both the domain edges, or a pair of modes,
        one for each edge.
        """
        pass

    @property
    def start_time(self):
        """
        The start time for the simulation. This is 0.0 by default.
        """
        return 0.0

    @property
    def reference_time_scale(self):
        """
        The time scale used to convert raw simulation time to user time.

        Typically the reference time can be 1.0, meaning that raw simulation
        time and user time are the same. However it sometimes makes sense for
        the user time to be something else, like the orbital period of 2 pi
        for example. The `start_time` and `default_time_time` properties are
        in uner time, not raw simulation time. Similarly, messages written to
        stdout, and event recurrences are both in user time, i.e. if the
        reference time is 2 pi and `--checkpoint=1.0`, you'll get one
        checkpoint written per orbital period. The "time" field of the
        checkpoint file contains the raw simulation time, not the user time.
        """
        return 1.0

    @property
    def default_end_time(self):
        """
        Provide a default end-time to the simulation driven.

        This value will be superceded if a non-None end time was provided to
        the driver. If neither the driver nor the setup has an end time, then
        the simulation runs until it's killed.
        """
        return None

    @property
    def default_resolution(self):
        """
        Provide a default grid resolution.
        """
        return 10000

    @property
    def diagnostics(self):
        """
        A list of diagnostics to be dispatched to the solver.
        """
        return list()

    def validate(self):
        """
        Confirm that the model parameters are physically valid.

        This method should be overridden to indicate a failure by throwing a
        `SetupError` exception.
        """
        pass

    def checkpoint_diagnostics(self, time):
        """
        Return a dict of post-processing data to include in checkpoint files.

        An example use case is to record the positions of point masses (with
        prescribed trajectory) in a gravitating hydrodynmics problem.
        """
        return dict()
