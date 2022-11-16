"""
Enables detection and validation of configuration schemas from functions
"""


from functools import wraps
from inspect import signature, _empty
from textwrap import dedent
from sys import stdout

SCHEMAS = list()


class ValidationError(Exception):
    """Runtime configuration does not match schema for the component"""


class SchemaError(Exception):
    """A schema was not properly described"""


class Schema:
    """
    Default values, type-hints and about messages extracted from a function

    Default values and type hints are extracted from the function signature.
    About messages are extracte from the doc string, see examples below for
    the expected format.
    """

    def __init__(self, func):
        lines = iter(func.__doc__.splitlines())
        parameters = signature(func).parameters
        data = dict()
        name = func.__name__

        try:
            while True:
                line = next(lines)

                if "Configuration" in line:
                    if next(lines).strip() != "-------------":
                        raise SchemaError("expect a line of '-' below 'Configuration'")
                    if next(lines).strip():
                        raise SchemaError("expect a blank line below 'Configuration'")
                    break

            while True:
                if not (line := next(lines)).strip():
                    break

                key, about = (x.strip() for x in line.split(":"))
                type_hint = parameters[key].annotation
                default = parameters[key].default

                if default is _empty:
                    raise SchemaError(f"missing default value for {name}.{key}")
                if type_hint is _empty:
                    raise SchemaError(f"missing type hint on {name}.{key}")

                data[key] = (type_hint, default, about)

        except (StopIteration):
            pass

        if not data:
            raise ValueError(f"no configuration data were found for {name}")

        self.data = data
        self.func = func

    @property
    def component_name(self):
        return self.func.__name__

    def validate(self, **kwargs):
        parameters = signature(self.func).parameters
        for key, val in kwargs.items():
            type_hint = parameters[key].annotation
            if not isinstance(val, type_hint):
                raise ValidationError(
                    f"parameter {key} must be {type_hint.__name__} (got {type(val).__name__})"
                )

    def print_schema(self, file=stdout):
        key_col = max(max(len(name) for name in self.data) + 3, 22)
        def_col = max(len(str(d)) for (_, d, _) in self.data.values()) + 1

        print(f"\n\n{self.component_name}\n", file=stdout)

        for key, (_, default, about) in self.data.items():
            print(f"{key :.<{key_col}} {default :<{def_col}} {about}", file=stdout)


def configurable(func):
    """
    Attaches a `Schema` instance to a function that provides a schema
    """
    schema = Schema(func)
    func.schema = schema
    SCHEMAS.append(schema)

    return func


@configurable
def planar_shocktube(x, which: str = "sod1", centerline: float = 0.5):
    """
    A planar shocktube model for 1d adiabatic Euler equations

    Configuration
    -------------

    which:       Type of shocktube setup [sod1, sod2]
    centerline:  Position of the discontinuity
    """
    pass


@configurable
def cylindrical_shocktube(
    x,
    y,
    radius: float = 0.1,
    pressure_inside: float = 1.0,
    gamma_law_index: float = 1.66,
):
    """
    A cylindrical shocktube initial condition

    Configuration
    -------------

    radius:           Radius of the high-pressure region
    pressure_inside:  Pressure of the high-pressure region
    gamma_law_index:  The gamma-law index of the gas
    """
    pass


planar_shocktube.schema.validate(which="sod1")


def print_configurations():
    for schema in SCHEMAS:
        schema.print_schema()
