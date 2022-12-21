"""
Defines test problems and generic hydrodynamic setups

Note: There is a caveat in the way the application does hierarchical
configuration: dictionaries are recursively updated, and this means that
dictionaries that map to schemas that are discriminated unions will have their
keys joined if superseded, and this can introduce extra parameters that fail
validation. The workaround for this is to ensure that the default model in a
discriminated union does not define any parameters other than the
discriminating field.
"""


from typing import Literal, Union
from schema import schema
from geometry import CoordinateBox
from numpy import logical_not, zeros, sqrt, sin, cos, pi


@schema
class Shocktube:
    """
    A linear shocktube setup
    """

    model: Literal["shocktube"] = "shocktube"

    def primitive(self, box: CoordinateBox):
        if box.dimensionality == 1:
            x = box.cell_centers()
            l = x < 0.5
            r = logical_not(l)
            p = zeros(x.shape + (3,))
            p[l] = [1.0, 0.0, 1.000]
            p[r] = [0.1, 0.0, 0.125]

        if box.dimensionality == 2:
            x, y = box.cell_centers()
            angle = 0.0
            a = cos(0.5 * angle * pi)
            b = sin(0.5 * angle * pi)
            l = a * x + b * y < 0.5
            r = logical_not(l)
            p = zeros(x.shape + (4,))
            p[l] = [1.0, 0.0, 0.0, 1.000]
            p[r] = [0.1, 0.0, 0.0, 0.125]

        return p


@schema
class CylindricalExplosion:
    """
    A cylindrical explosion setup
    """

    model: Literal["cylindrical-explosion"] = "cylindrical-explosion"

    def primitive(self, box: CoordinateBox):
        if box.dimensionality != 2:
            raise NotImplementedError("setup only works in 2d")

        x, y = box.cell_centers()
        l = sqrt(x**2 + y**2) < 0.1
        r = logical_not(l)
        p = zeros(x.shape + (4,))
        p[l] = [1.0, 0.0, 0.0, 1.000]
        p[r] = [0.1, 0.0, 0.0, 0.125]

        return p


@schema
class CylinderInWind:
    """
    A round cylinder immersed in a dilute wind
    """

    model: Literal["cylinder-in-wind"] = "cylinder-in-wind"

    def primitive(self, box: CoordinateBox):
        if box.dimensionality != 2:
            raise NotImplementedError("setup only works in 2d")

        x, y = box.cell_centers()
        l = sqrt(x**2 + y**2) < 0.1
        r = logical_not(l)
        p = zeros(x.shape + (4,))

        p[l] = [1e2, 0.0, 0.0, 1.0]
        p[r] = [1.0, 1.0, 0.0, 1.0]

        return p


InitialData = Union[Shocktube, CylindricalExplosion, CylinderInWind]
