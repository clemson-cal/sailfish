"""
Configurable models for the purose of app configuration

This module exports a `configmodel` decorator which builds on a Python
dataclass with validation via `pydantic`, and pretty printing via the `rich`
module.

Short and long descriptions of the model, and field decscriptions are read
from the class doc string, which must have the following format:

```python
@configmodel
class Bird:
    \"""
    A model to represent a bird

    This model represents a bird in terms of its color, sound, and how good it
    is at finding cached food items. For exaple, crows are apparently quite
    good at this, they always find the food they hid the previous season. It's
    quite impressive. This model should not be used to represent squirrels.

    Fields
    ------

    name:     the name of the bird
    color:    the color of the bird
    sound:    what kind of sound the bird makes
    memory:   how good the bird is at remembering things
    \"""

    name: str
    color: str = "black"
    sound: str = "squawk"
    memory: int = 99
```

Instances of the `Bird` class will be type-validated on construction:

```python
joe = Bird(name="Joe") # ok
jan = Bird(name=[42]) # ValidationError
```

They can also be printed as a table with descriptions using a `Console`
instance.

```python
console.print(joe) # looks nice!
```
"""


def parse_docstring(cls):
    """
    Parse a configmodel docstring.
    """
    from textwrap import dedent

    cls_short_descr = str()
    cls_long_descr = list()
    field_descriptions = dict()
    lines = iter(dedent(cls.__doc__).splitlines())

    for line in lines:
        if "Fields" in line:
            if next(lines).strip() != "------":
                raise ValueError("expect a line of '-' below 'Fields'")
            if next(lines).strip():
                raise ValueError("expect a blank line below 'Fields'")
            break
        elif not cls_short_descr:
            cls_short_descr = line
        elif line:
            cls_long_descr.append(line)

    for line in lines:
        if line:
            key, description = (x.strip() for x in line.split(":"))
            field_descriptions[key] = description

    return cls_short_descr, " ".join(cls_long_descr), field_descriptions


def configmodel_rich_table(d, console, options):
    """
    Returns a rich-renderable table generated from a configmodel.
    """
    from rich.table import Table
    from rich.pretty import Pretty
    from rich.style import Style

    def rep(key):
        value = getattr(d, key)
        if hasattr(value, "__configmodel__"):
            return Pretty(value)
        else:
            return str(value)

    fields = d.__dataclass_fields__
    short_descr = d.__configmodel__["short_descr"]
    long_descr = d.__configmodel__["long_descr"]
    prop_descriptions = d.__configmodel__["prop_descriptions"]

    if prop_descriptions:
        long_descr = " ".join([long_descr, "*Derived property."])

    table = Table(
        title=f"{d.__class__.__name__}: {short_descr.lower()}"
        if short_descr
        else d.__class__.__name__,
        caption=long_descr,
        caption_justify="left",
        title_justify="left",
        show_edge=True,
        show_lines=False,
        show_header=False,
        min_width=80,
    )
    table.add_column("property", style="cyan")
    table.add_column("value", style="green")
    table.add_column("description", style="magenta")

    for key, field in fields.items():
        value = getattr(d, key)
        descr = field.metadata.get("description", None)
        table.add_row(key, rep(key), descr)

    for key, descr in prop_descriptions.items():
        value = getattr(d, key)
        table.add_row(key, rep(key), descr + "*", style=Style(dim=True))

    yield table


def configmodel(cls):
    from pydantic.dataclasses import dataclass

    if cls.__doc__ is None:
        cls.__doc__ = " "
    dataclass(cls)

    short_descr, long_descr, field_descriptions = parse_docstring(cls)
    prop_descriptions = {
        k: v.__doc__.strip() for k, v in vars(cls).items() if type(v) is property
    }
    fields = cls.__dataclass_fields__

    for key, description in field_descriptions.items():
        fields[key].metadata = dict(description=description)

    cls.__rich_console__ = configmodel_rich_table
    cls.__configmodel__ = dict(
        short_descr=short_descr,
        long_descr=long_descr,
        prop_descriptions=prop_descriptions,
    )
    return cls


def main():
    """
    Examples of how to create configurable models.
    """

    from rich.pretty import Pretty
    from rich.console import Console
    from rich.markdown import Markdown

    @configmodel
    class Physics:
        """
        Fields
        ------

        cooling_rate:  the cooling rate
        optical_depth: the optical depth
        """

        cooling_rate: float = 1.0
        optical_depth: float = 2.0

    @configmodel
    class CylindricalShocktube:
        """
        A circular explosion setup

        This model supports euler2d and iso2d systems. It sets up a circular
        region of high density and pressure where you can watch the explosion
        expand outward.

        Fields
        ------

        radius:           radius of the high-pressure region
        pressure_inside:  pressure of the high-pressure region
        gamma_law_index:  the gamma-law index of the gas
        physics:          more physics details
        """

        radius: float = 0.1
        pressure_inside: float = 1.0
        gamma_law_index: float = 1.66666666
        physics: Physics = Physics(1.0, 0.01)

    console = Console(width=80)
    model = CylindricalShocktube(
        radius=12,
        physics=(1.0, 0.01),
    )

    console.print(Markdown(__doc__))
    console.width = None
    print()
    console.print(model)
    print()
    console.print(Pretty(Physics(), expand_all=True, indent_guides=True))


if __name__ == "__main__":
    main()
