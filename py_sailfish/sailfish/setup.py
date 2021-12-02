from textwrap import dedent
from abc import ABC, abstractmethod


class SetupError(Exception):
    """Something went wrong during setup configuration"""


class Parameter:
    def __init__(self, default, about, mutable=False):
        self.default = default
        self.about = about
        self.mutable = mutable


def parameter(default, about, mutable=False):
    """
    Return a `Parameter` instance.

    `Parameter` instances should be set as a class variable on sub-classes of
    `Setup`. Set the `mutable` flag to `True` if that parameter should be
    adjustable in restarted runs.
    """
    return Parameter(default, about, mutable=mutable)


class Setup(ABC):
    """
    An abstract base class to describe a simulation model setup.

    Subclasses are used by the driver to define initial and boundary
    conditions, select a hydrodynamics solver and parameters, and describe
    physics conditions such as gravity and thermodynamics. Basic setups only
    need to implement a subset of the possible methods; most of the methods
    below have stub default implementations.
    """

    def __init__(self, **kwargs):
        for key, val, about in type(self).default_model_parameters:
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
                raise SetupError(f"setup has no parameter '{key}'")

        self.validate()

    @classmethod
    @property
    def default_model_parameters(cls):
        """
        Return an iterator over the default model parameters for this class.
        """
        for key, val in vars(cls).items():
            if type(val) == Parameter:
                yield key, val.default, val.about

    @classmethod
    @property
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
        print(f"setup: {cls.dash_case_class_name}")
        print(dedent(cls.__doc__))
        cls().print_model_parameters()

    @classmethod
    @property
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
        match = lambda s: s.dash_case_class_name == name
        try:
            return next(filter(match, cls.__subclasses__()))
        except StopIteration:
            raise SetupError(f"no setup named {name}")

    @classmethod
    @property
    def has_model_parameters(cls):
        """
        Determine if this class has any model parameters.
        """
        for _ in cls.default_model_parameters:
            return True
        return False

    def print_model_parameters(self, newlines=False):
        """
        Print parameter names, values, and about messages to `stdout`.
        """
        if self.has_model_parameters:
            if newlines:
                print()
                print("model parameters:\n")
                for name, default, about in self.model_parameters:
                    print(f"{name:.<16s} {default:<5} {about}")
            if newlines:
                print()

    @property
    def model_parameters(self):
        """
        Return an iterator over the model parameters chosen for this setup.
        """
        for key, val, about in self.default_model_parameters:
            yield key, getattr(self, key), about

    @property
    def model_parameter_dict(self):
        """
        Return a dictionary of the model parameters.
        """
        return {key: val for key, val, _ in self.model_parameters}

    @abstractmethod
    def initial_primitive(self, coordinate, primitive):
        """
        Set hydrodynamic data at a point.

        This method must be overridden to set the initial primitive
        hydrodynamic variables at a single point (given by `coordinate`). The
        meaning of the coordinate is influenced by the type of mesh being
        used. Data is written to the output variable `primitive` for
        efficiency, since otherwise a tiny allocation is made for each zone in
        the simulation grid.
        """
        pass

    @property
    @abstractmethod
    def domain(self):
        pass

    @property
    @abstractmethod
    def boundary_condition(self):
        """
        Return a boundary condition mode.

        This mode must be supported by the solver.
        """
        pass

    @property
    def end_time(self):
        """
        Provide a default end-time to the simulation driven.

        This value will be superceded if a non-None end time was provided to
        the driver. If neither the driver nor the setup has an end time, then
        the simulation runs until it's killed.
        """
        return None

    def validate(self):
        """
        Confirm that the model parameters are physically valid.

        This method should be overridden to indicate a failure by throwing a
        `SetupError` exception.
        """
        pass
