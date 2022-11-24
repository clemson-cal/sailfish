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
    Meta-data for a user-configurable application component

    The schema stores default values, type-hints and about messages for a set
    of configuration items.

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
                try:
                    type_hint = parameters[key].annotation
                    default = parameters[key].default
                except KeyError:
                    raise SchemaError(f"{name}.{key} missing from function signature")

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
        """
        The name of the configurable component this schema represents
        """
        return self.func.__name__

    def validate(self, **kwargs):
        """
        Raise an exception if any keyword args incompatible with the schema
        """
        parameters = signature(self.func).parameters
        for key, val in kwargs.items():
            type_hint = parameters[key].annotation
            if not isinstance(val, type_hint):
                raise ValidationError(
                    f"parameter {key} must be {type_hint.__name__} (got {type(val).__name__})"
                )

    def print_schema(self, log, color=True, newline=False, config=dict()):
        """
        Print a formatted table of configuration items to a function

        If `color` is `True` then the output is formatted in color, so the given
        `log` function should interpret the color directives appropriately.
        Otherwise, or if `log` is builtin `print`, the color formatting is
        omitted.
        """
        key_col = max(max(len(name) for name in self.data) + 3, 22)
        def_col = max(len(str(d)) for (_, d, _) in self.data.values()) + 1

        if color and log is not print:
            log(f"\n<cyan><u>{self.component_name}</u></cyan>\n")
        else:
            log(f"\n{self.component_name}\n")

        for key, (_, value, about) in self.data.items():
            try:
                value = config[key]
            except KeyError:
                pass

            if color and log is not print:
                msg = (
                    f"<blue>{key :.<{key_col}}</blue> "
                    f"<yellow>{str(value) :<{def_col}}</yellow> "
                    f"<green>{about}</green>"
                )
            else:
                msg = f"{key :.<{key_col}} {str(value) :<{def_col}} {about}"
            log(msg)

        if newline:
            log("\n")

    def defaults_dict(self):
        return {key: value for key, (_, value, _) in self.data.items()}

    def argument_parser(self, parser=None, dest_prefix=None):
        if parser is None:
            from argparse import ArgumentParser

            parser = ArgumentParser()

        for key, (type_hint, default, about) in self.data.items():
            if type_hint is bool and default is False:
                parser.add_argument(
                    "--" + key.replace("_", "-"),
                    action="store_true",
                    help=about,
                )
            else:
                parser.add_argument(
                    "--" + key.replace("_", "-"),
                    type=type_hint,
                    default=None,
                    dest=f"{dest_prefix}.{key}" if dest_prefix else key,
                    help=about,
                    metavar="",
                )

        return parser


def configurable(func):
    """
    Attaches a schema to a function that provides sufficient meta-data.

    Meta-data is supplied through the names, type-hints, and default values of
    the function arguments, and also a special "Configuration" section inside
    the function doc string. The configuration section contains about messages
    for the function arguments that are published as a configuration.
    Functions that are decorated with `configurable` are registered in a
    module variable, enabling the top-level application to inspect of a
    collection of configurable components across all of the application
    sub-modules.

    The `Schema` instance, which will be attached to the decorated function,
    enables pretty-printing of the component configuration, and validation of
    dictionary-like configuration objects.
    """
    schema = Schema(func)
    func.schema = schema
    SCHEMAS.append(schema)

    return func


def all_schemas():
    for schema in SCHEMAS:
        yield schema


def main():
    """
    Examples of how to create configurable application components from
    functions.
    """
    from loguru import logger

    term = lambda msg: logger.opt(ansi=True).log("TERM", msg)
    logger.level("TERM", 0)
    logger.remove()
    logger.add(stdout, level="TERM", format="{message}")

    @configurable
    def planar_shocktube(
        x,
        which: str = "sod1",
        centerline: float = 0.5,
    ):
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
    cylindrical_shocktube.schema.validate(radius=3.0)

    for schema in SCHEMAS:
        schema.print_schema(term)


if __name__ == "__main__":
    main()
