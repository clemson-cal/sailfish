from abc import ABC, abstractmethod


class SetupError(Exception):
    """Something went wrong during setup configuration"""


class Parameter:
    def __init__(self, default, about):
        self.default = default
        self.about = about


def parameter(default, about):
    return Parameter(default, about)


class Setup(ABC):
    def __init__(self, **kwargs):
        for key, val, about in self.model_parameters():
            if key in kwargs:
                if type(kwargs[key]) != type(val.default):
                    raise SetupError(
                        f"parameter '{key}' has type {type(val.default).__name__} "
                        f"(got {type(kwargs[key]).__name__})"
                    )
                setattr(self, key, kwargs[key])
            else:
                setattr(self, key, val.default)

        for key, val in kwargs.items():
            if not hasattr(self, key):
                raise SetupError(f"setup has no parameter '{key}'")

    @classmethod
    def model_parameters(cls):
        for key, val in vars(cls).items():
            if type(val) == Parameter:
                yield key, val.default, val.about

    @classmethod
    def dash_case_class_name(cls):
        n = cls.__name__
        return "".join(["-" + c.lower() if c.isupper() else c for c in n]).lstrip("-")

    @classmethod
    def find_setup(cls, name):
        match = lambda s: s.dash_case_class_name() == name
        try:
            return next(filter(match, cls.__subclasses__()))
        except StopIteration:
            raise SetupError(f"no setup named {name}")

    @abstractmethod
    def initial_primitive(self, coordinate, primitive):
        pass

    @property
    @abstractmethod
    def domain(self):
        pass

    @property
    @abstractmethod
    def boundary_condition(self):
        pass
